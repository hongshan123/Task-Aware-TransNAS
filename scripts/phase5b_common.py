"""Phase 5B prompt, intervention, and output-format utilities."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch

from phase5_common import (
    ICL_SHOTS,
    MAX_ICL_PROMPT_TOKENS,
    answer_tokens_for_custom,
    cap_answer_tokens,
    parse_yes_no,
    prompt_messages,
    rouge_l_f1,
    token_f1,
)
from phase5_common import _draw_support, _encode_prompt, _icl_user_content
from run_formal_baseline import encode_training_record, safe_prompt


FORMAT_HINT = (
    "Return only the requested answer text. Do not add an explanation, preamble, "
    "label, quotation marks, or confidence statement."
)


@dataclass
class FourShotPrompt:
    task_id: str
    query_id: str
    input_ids: list[int]
    target_ids: list[int]
    support_ids: list[str]
    shots: int
    prompt_tokens: int
    query_input_tokens_kept: int
    demonstration_input_tokens_kept: list[int]


def with_format_hint(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "definition": f"{str(record['definition']).rstrip()}\n\nOutput format: {FORMAT_HINT}",
    }


def prepare_zero_shot_ids(
    tokenizer: Any,
    record: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[list[int], list[int], dict[str, Any]]:
    input_ids, labels, diagnostics = encode_training_record(
        tokenizer,
        record,
        int(model_config["max_sequence_length"]),
        int(model_config["max_target_tokens"]),
    )
    prompt_len = next(index for index, label in enumerate(labels) if label != -100)
    return input_ids[:prompt_len], input_ids[prompt_len:], diagnostics


def _decode_suffix(tokenizer: Any, tokens: list[int], keep: int) -> str:
    return tokenizer.decode(tokens[-keep:], skip_special_tokens=True) if keep > 0 else ""


def make_exact_four_shot_prompt(
    tokenizer: Any,
    task_id: str,
    query: dict[str, Any],
    train_rows: list[dict[str, Any]],
    query_index: int,
    max_sequence_length: int,
    max_target_tokens: int,
) -> FourShotPrompt:
    """Build a correct-label ICL prompt and preserve all four demonstrations."""
    support = _draw_support(task_id, train_rows, query_index)
    if len(support) != ICL_SHOTS:
        raise ValueError(f"Need exactly {ICL_SHOTS} train demonstrations for {task_id}")
    outputs = [str(row["targets"][0]).strip() for row in support]
    if len(set(str(row["instance_id"]) for row in support)) != ICL_SHOTS:
        raise ValueError(f"Repeated support rows for {task_id}/{query['instance_id']}")

    demo_tokens = [
        tokenizer.encode(str(row["input"]), add_special_tokens=False) for row in support
    ]
    query_tokens = tokenizer.encode(str(query["input"]), add_special_tokens=False)
    demo_keeps = [len(value) for value in demo_tokens]
    query_keep = len(query_tokens)
    prompt_budget = min(
        MAX_ICL_PROMPT_TOKENS,
        int(max_sequence_length) - int(max_target_tokens),
    )

    def encode_current() -> list[int]:
        demos = []
        for row, tokens, keep, output in zip(support, demo_tokens, demo_keeps, outputs):
            demos.append(({**row, "input": _decode_suffix(tokenizer, tokens, keep)}, output))
        shortened_query = {**query, "input": _decode_suffix(tokenizer, query_tokens, query_keep)}
        content = _icl_user_content(shortened_query, demos)
        return _encode_prompt(tokenizer, prompt_messages(str(query["definition"]), content))

    prompt_ids = encode_current()
    # Keep the task definition and all four demonstration answers intact. Trim
    # only the tails of the query and demonstration inputs until the exact chat
    # template fits the same prompt-plus-target budget used by zero-shot scoring.
    while len(prompt_ids) > prompt_budget:
        candidates = [
            (keep, "demo", index)
            for index, keep in enumerate(demo_keeps)
            if keep > 8
        ]
        if query_keep > 8:
            candidates.append((query_keep, "query", -1))
        if not candidates:
            raise ValueError(
                f"Cannot fit four demonstrations within {prompt_budget} tokens for "
                f"{task_id}/{query['instance_id']}"
            )
        _, kind, index = max(candidates)
        if kind == "query":
            query_keep = max(8, int(query_keep * 0.75))
        else:
            demo_keeps[index] = max(8, int(demo_keeps[index] * 0.75))
        prompt_ids = encode_current()

    target_ids = answer_tokens_for_custom(
        tokenizer,
        str(query["definition"]),
        _icl_user_content(
            {**query, "input": _decode_suffix(tokenizer, query_tokens, query_keep)},
            [
                (
                    {**row, "input": _decode_suffix(tokenizer, tokens, keep)},
                    output,
                )
                for row, tokens, keep, output in zip(support, demo_tokens, demo_keeps, outputs)
            ],
        ),
        str(query["targets"][0]),
    )
    target_ids = cap_answer_tokens(target_ids, tokenizer, int(max_target_tokens))
    if len(prompt_ids) + len(target_ids) > int(max_sequence_length):
        raise AssertionError("Four-shot prompt plus capped target exceeds the configured sequence length")
    return FourShotPrompt(
        task_id=task_id,
        query_id=str(query["instance_id"]),
        input_ids=prompt_ids,
        target_ids=target_ids,
        support_ids=[str(row["instance_id"]) for row in support],
        shots=ICL_SHOTS,
        prompt_tokens=len(prompt_ids),
        query_input_tokens_kept=query_keep,
        demonstration_input_tokens_kept=demo_keeps,
    )


class Phase5BResidualIntervention:
    """Add an FV at one Qwen3 attention o_proj output during scoring/generation."""

    def __init__(self, model: Any):
        self.model = model
        self.layer: int | None = None
        self.vector: torch.Tensor | None = None
        self.scale = 0.0
        self.mode: str | None = None
        self.start_position: int | None = None
        self.events: list[dict[str, Any]] = []
        self.handles = []
        for layer_id, block in enumerate(model.model.layers):
            self.handles.append(block.self_attn.o_proj.register_forward_hook(self._hook(layer_id)))

    def _hook(self, layer_id: int):
        def hook(_module, _args, output):
            if self.vector is None or self.layer != layer_id or self.mode is None:
                return output
            if not torch.is_tensor(output) or output.ndim != 3:
                raise TypeError(f"Unexpected Qwen3 decoder block output at layer {layer_id}")
            if self.mode == "last":
                start = output.shape[1] - 1
            elif self.mode == "suffix":
                if self.start_position is None:
                    raise RuntimeError("Suffix intervention is missing its start position")
                start = self.start_position
            else:
                raise ValueError(f"Unknown intervention mode: {self.mode}")
            if start < 0 or start >= output.shape[1]:
                raise IndexError(
                    f"Intervention position {start} is outside sequence length {output.shape[1]}"
                )

            vector = self.vector.to(device=output.device, dtype=torch.float32) * float(self.scale)
            patched = output.clone()
            before = output[:, start:, :].float()
            after = (before + vector.view(1, 1, -1)).to(dtype=output.dtype)
            patched[:, start:, :] = after
            applied = patched[:, start:, :].float() - before
            self.events.append(
                {
                    "layer": layer_id,
                    "hook_point": "selected_decoder_layer.self_attn.o_proj.output.additive_v1",
                    "mode": self.mode,
                    "sequence_length": int(output.shape[1]),
                    "patched_token_count": int(output.shape[1] - start),
                    "hidden_size": int(output.shape[2]),
                    "output_dtype": str(output.dtype),
                    "requested_delta_rms": float(vector.square().mean().sqrt().item()),
                    "applied_delta_rms": float(applied.square().mean().sqrt().item()),
                    "max_applied_delta": float(applied.abs().max().item()),
                    "pre_patch_hidden_rms": float(before.square().mean().sqrt().item()),
                }
            )
            return patched

        return hook

    def activate(
        self,
        layer: int,
        vector: torch.Tensor,
        scale: float,
        mode: str,
        start_position: int | None = None,
    ) -> None:
        self.layer = int(layer)
        self.vector = vector
        self.scale = float(scale)
        self.mode = str(mode)
        self.start_position = None if start_position is None else int(start_position)
        self.events = []

    def deactivate(self) -> None:
        self.layer = None
        self.vector = None
        self.scale = 0.0
        self.mode = None
        self.start_position = None
        self.events = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.deactivate()


def generated_label_metrics(predictions: list[str]) -> dict[str, Any]:
    """Separate exact binary outputs, parsed label concentration, and invalid format."""
    total = len(predictions)
    if not total:
        return {
            "exact_yes_output_rate": None,
            "exact_no_output_rate": None,
            "exact_yes_no_output_rate": None,
            "most_common_label_concentration_among_parseable": None,
            "parseable_yes_no_label_rate": None,
            "invalid_answer_format_rate": None,
            "parseable_yes_count": 0,
            "parseable_no_count": 0,
        }
    exact = [
        unicodedata.normalize("NFKC", value).strip().casefold()
        for value in predictions
    ]
    exact_yes = sum(value == "yes" for value in exact)
    exact_no = sum(value == "no" for value in exact)
    parsed = [parse_yes_no(value) for value in predictions]
    parsed_yes = sum(value == "Yes" for value in parsed)
    parsed_no = sum(value == "No" for value in parsed)
    parseable_count = parsed_yes + parsed_no
    return {
        "exact_yes_output_rate": exact_yes / total,
        "exact_no_output_rate": exact_no / total,
        "exact_yes_no_output_rate": (exact_yes + exact_no) / total,
        "most_common_label_concentration_among_parseable": (
            max(parsed_yes, parsed_no) / parseable_count if parseable_count else None
        ),
        "parseable_yes_no_label_rate": parseable_count / total,
        "invalid_answer_format_rate": (total - parseable_count) / total,
        "parseable_yes_count": parsed_yes,
        "parseable_no_count": parsed_no,
    }


def generation_scores(prediction: str, references: list[str]) -> dict[str, float]:
    references = references or [""]
    return {
        "rouge_l_f1": max(rouge_l_f1(prediction, reference) for reference in references),
        "token_f1": max(token_f1(prediction, reference) for reference in references),
    }


def reference_token_score(log_probabilities: torch.Tensor) -> dict[str, float | int]:
    values = log_probabilities.detach().float()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("Reference token log probabilities must be nonempty and finite")
    return {
        "nll_sum": float(-values.sum().item()),
        "token_count": int(values.numel()),
        "mean_token_nll": float(-values.mean().item()),
        "first_reference_token_probability": float(values[0].exp().item()),
    }
