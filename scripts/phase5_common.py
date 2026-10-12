"""Shared causal-analysis utilities for Phase 5 (cross-platform, local-only inference)."""

from __future__ import annotations

import gc
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from architecture_modules import apply_architecture  # noqa: E402
from phase4_common import TASK_IDS, load_project_config, prepare_context, resolved_model_path  # noqa: E402
from run_formal_baseline import (  # noqa: E402
    answer_tokens_for,
    encode_training_record,
    load_model,
    load_tokenizer,
    norm_text,
    parse_yes_no,
    rouge_l_f1,
    safe_prompt,
    score_rows,
    token_ids,
    token_f1,
)

ALLOWED_MISSING_TIED_KEYS = {"lm_head.weight"}

TASK_GENERATION = {
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
}
TASK_CLASSIFICATION = {
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
}
TASK_1384 = "task1384_deal_or_no_dialog_classification"
TASK_1501 = "task1501_dstc3_answer_generation"
TASK_565 = "task565_circa_answer_generation"
TASK_1603 = "task1603_smcalflow_sentence_generation"
SYSTEM_TEXT = "You are an instruction-following assistant. Follow the task definition and return only the requested answer."
PHASE5_ROOT = storage_path("results") / "phase5"
FIGURE_ROOT = storage_path("figures") / "phase5"
REPORT_ROOT = storage_path("reports")
MAX_ICL_PROMPT_TOKENS = 2048
ICL_SHOTS = 4
SPLITS = {"build": (0, 10), "tune": (10, 25), "evaluation": (25, 50)}
STAGE_TASKS = [TASK_565, TASK_1603, TASK_1501, TASK_1384]
PATCH_BATCH_SIZE = 8
TOP_K = 10
SEED = 314159
FV_VERSION = "v2_bf16_dequantized_fp32"


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_phase5_context():
    config, model_path, tokenizer, device, train, dev = prepare_context()
    return config, model_path, tokenizer, device, train, dev


def stage_checkpoint_specs() -> list[dict[str, Any]]:
    """The homogeneous Phase 3 stream retained a checkpoint after each task."""
    specs = [{"checkpoint_id": "pretrained", "architecture": "A0", "path": None, "stage": 0, "task": None}]
    names = [
        "stage_1_task565_circa_answer_generation",
        "stage_2_task1603_smcalflow_sentence_generation",
        "stage_3_task1501_dstc3_answer_generation",
        "stage_4_task1384_deal_or_no_dialog_classification",
    ]
    for stage_no, (task_id, folder) in enumerate(zip(STAGE_TASKS, names), start=1):
        specs.append(
            {
                "checkpoint_id": f"homogeneous_stage_{stage_no}",
                "architecture": "A0",
                "path": storage_path("checkpoints") / "baseline_full" / "homogeneous" / folder,
                "stage": stage_no,
                "task": task_id,
            }
        )
    return specs


def phase4b_checkpoint(architecture: str = "A0", sequence: str = "S1", seed: int = 10) -> Path:
    return storage_path("results") / "phase4b" / "checkpoints" / architecture / sequence / f"seed_{seed}"


def load_checkpoint(model_path: Path, checkpoint_path: Path | None, architecture: str = "A0"):
    model = load_model(model_path, local_only=True, training=False)
    if architecture != "A0":
        apply_architecture(model, architecture)
    if checkpoint_path is not None:
        weights_path = checkpoint_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint weights: {weights_path}")
        state = load_file(str(weights_path), device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        del state
        invalid_missing = missing - ALLOWED_MISSING_TIED_KEYS
        if invalid_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint reload mismatch for {checkpoint_path}: "
                f"missing={sorted(invalid_missing)}, unexpected={unexpected[:12]}"
            )
    model.eval()
    model.config.use_cache = True
    return model


def cleanup_model(model: Any | None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prompt_messages(definition: str, user_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": f"Task definition:\n{definition}\n\n{user_content}"},
    ]


def _encode_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return token_ids(encoded)


def _tail(text: str, tokenizer: Any, token_budget: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= token_budget:
        return text
    return tokenizer.decode(ids[-token_budget:], skip_special_tokens=True)


def _icl_user_content(query: dict[str, Any], demonstrations: list[tuple[dict[str, Any], str]]) -> str:
    lines = ["Examples:"]
    for index, (demo, output) in enumerate(demonstrations, start=1):
        lines.extend(
            [
                f"Example {index} input:",
                demo["input"],
                f"Example {index} answer:",
                output,
            ]
        )
    lines.extend(["Query input:", query["input"], "Answer:"])
    return "\n".join(lines)


@dataclass
class ICLPrompt:
    task_id: str
    query_id: str
    correct_ids: list[int]
    counterfactual_ids: list[int]
    target_ids: list[int]
    support_ids: list[str]
    shots: int
    correct_prompt_tokens: int
    counterfactual_prompt_tokens: int
    truncated_demo_count: int
    query_tail_tokens_kept: int


def _draw_support(task_id: str, train_rows: list[dict[str, Any]], query_index: int) -> list[dict[str, Any]]:
    rng = random.Random(SEED + query_index * 1009 + int(task_id[4:7]))
    rows = train_rows
    for _ in range(200):
        chosen = rng.sample(rows, min(ICL_SHOTS, len(rows)))
        labels = [str(row["targets"][0]).strip() for row in chosen]
        counts = Counter(labels)
        if len(set(labels)) >= 2 and max(counts.values()) <= len(labels) // 2:
            return chosen
    return rng.sample(rows, min(ICL_SHOTS, len(rows)))


def make_icl_prompt(
    tokenizer: Any,
    task_id: str,
    query: dict[str, Any],
    support_rows: list[dict[str, Any]],
    query_index: int,
    max_prompt_tokens: int = MAX_ICL_PROMPT_TOKENS,
) -> ICLPrompt:
    """Build a deterministic correct and truly label-shuffled counterfactual ICL pair."""
    support = list(support_rows)
    correct_outputs = [str(row["targets"][0]).strip() for row in support]
    if len(support) < 2:
        raise ValueError(f"Need at least 2 support examples for {task_id}")
    rng = random.Random(SEED + 707 * query_index + int(task_id[4:7]))
    shuffled = correct_outputs[:]
    for _ in range(200):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(shuffled, correct_outputs)):
            break
    else:
        shuffled = correct_outputs[1:] + correct_outputs[:1]
        if any(a == b for a, b in zip(shuffled, correct_outputs)):
            raise ValueError(f"Could not construct label-shuffled examples for {task_id}/{query['instance_id']}")

    query_text = str(query["input"])
    work = [(row, output) for row, output in zip(support, correct_outputs)]
    shuffled_pairs = [(row, output) for row, output in zip(support, shuffled)]
    query_keep = len(tokenizer.encode(query_text, add_special_tokens=False))
    final_query_text = query_text
    truncated_demo_count = 0

    def pair_ids(query_input: str, demos: list[tuple[dict[str, Any], str]]) -> tuple[list[int], list[int]]:
        correct_text = _icl_user_content({**query, "input": query_input}, demos)
        cf_text = _icl_user_content({**query, "input": query_input}, shuffled_pairs[: len(demos)])
        correct_messages = prompt_messages(str(query["definition"]), correct_text)
        cf_messages = prompt_messages(str(query["definition"]), cf_text)
        return _encode_prompt(tokenizer, correct_messages), _encode_prompt(tokenizer, cf_messages)

    correct_ids, counterfactual_ids = pair_ids(query_text, work)
    # Long dialogue contexts are shortened from the beginning, matching the
    # project's suffix-preserving prompt policy. Task definitions remain intact.
    while max(len(correct_ids), len(counterfactual_ids)) > max_prompt_tokens and len(work) > 2:
        work.pop()
        truncated_demo_count += 1
        correct_ids, counterfactual_ids = pair_ids(query_text, work)
    if max(len(correct_ids), len(counterfactual_ids)) > max_prompt_tokens:
        for keep in (192, 128, 96, 64, 32):
            trimmed = _tail(query_text, tokenizer, keep)
            query_keep = min(query_keep, keep)
            correct_ids, counterfactual_ids = pair_ids(trimmed, work)
            if max(len(correct_ids), len(counterfactual_ids)) <= max_prompt_tokens:
                final_query_text = trimmed
                break
    while max(len(correct_ids), len(counterfactual_ids)) > max_prompt_tokens and len(work) > 1:
        work.pop()
        truncated_demo_count += 1
        correct_ids, counterfactual_ids = pair_ids(query_text, work)
    if max(len(correct_ids), len(counterfactual_ids)) > max_prompt_tokens:
        raise ValueError(
            f"ICL prompt exceeds {max_prompt_tokens} tokens after safe shortening: "
            f"{task_id}/{query['instance_id']}={len(correct_ids)}"
        )
    final_query = {**query, "input": final_query_text}
    correct_target_ids = answer_tokens_for_custom(
        tokenizer,
        query["definition"],
        _icl_user_content(final_query, work),
        query["targets"][0],
    )
    target_ids = cap_answer_tokens(correct_target_ids, tokenizer, 64)
    return ICLPrompt(
        task_id=task_id,
        query_id=str(query["instance_id"]),
        correct_ids=correct_ids,
        counterfactual_ids=counterfactual_ids,
        target_ids=target_ids,
        support_ids=[str(row["instance_id"]) for row, _ in work],
        shots=len(work),
        correct_prompt_tokens=len(correct_ids),
        counterfactual_prompt_tokens=len(counterfactual_ids),
        truncated_demo_count=truncated_demo_count,
        query_tail_tokens_kept=query_keep,
    )


def answer_tokens_for_custom(tokenizer: Any, definition: str, user_content: str, target: str) -> list[int]:
    messages = prompt_messages(definition, user_content)
    prefix = _encode_prompt(tokenizer, messages)
    complete = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": str(target)}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    complete = token_ids(complete)
    if complete[: len(prefix)] != prefix:
        raise ValueError("Custom ICL chat prompt/answer prefix mismatch")
    return complete[len(prefix) :]


def cap_answer_tokens(ids: list[int], tokenizer: Any, cap: int) -> list[int]:
    if not ids:
        raise ValueError("Expected nonempty reference completion tokens")
    if len(ids) >= cap:
        return ids[: cap - 1] + [int(tokenizer.eos_token_id)]
    return ids


def make_icl_sets(tokenizer: Any, train: dict[str, list[dict[str, Any]]], dev: dict[str, list[dict[str, Any]]]):
    packs: dict[str, list[ICLPrompt]] = {}
    for task_id in TASK_IDS:
        start, end = SPLITS["build"]
        packs[task_id] = []
        for index, query in enumerate(dev[task_id][start:end], start=start):
            support = _draw_support(task_id, train[task_id], index)
            packs[task_id].append(make_icl_prompt(tokenizer, task_id, query, support, index))
    return packs


def token_probabilities(model: Any, input_ids: torch.Tensor, target_start: int, target_ids: list[int]):
    """Return sequence log probabilities while projecting only target positions."""
    mask = torch.ones_like(input_ids)
    body = model.model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
    hidden = body.last_hidden_state
    positions = torch.arange(target_start - 1, target_start - 1 + len(target_ids), device=input_ids.device)
    selected_hidden = hidden.index_select(1, positions)
    logits = model.lm_head(selected_hidden).float()
    log_probs = F.log_softmax(logits, dim=-1)
    labels = torch.tensor(target_ids, dtype=torch.long, device=input_ids.device).view(1, -1, 1).expand(input_ids.shape[0], -1, -1)
    selected = log_probs.gather(-1, labels).squeeze(-1)
    return selected, logits[:, 0, :].detach()


def make_complete_ids(prompt_ids: list[int], target_ids: list[int], device: torch.device):
    full = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=device)
    return full


def model_head_geometry(model: Any) -> dict[str, int]:
    config = model.config
    geometry = {
        "layers": int(config.num_hidden_layers),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        "hidden_size": int(config.hidden_size),
        "o_proj_input_width": int(model.model.layers[0].self_attn.o_proj.in_features),
    }
    if geometry["query_heads"] * geometry["head_dim"] != geometry["o_proj_input_width"]:
        raise RuntimeError(f"GQA geometry mismatch at o_proj: {geometry}")
    return geometry


class HeadActivationRecorder:
    """Record each Q-head's exact output-projection contribution at the last token."""

    def __init__(self, model: Any):
        self.model = model
        self.inputs: dict[int, torch.Tensor] = {}
        self.current: torch.Tensor | None = None
        self.max_reconstruction_error = 0.0
        self.max_reconstruction_relative_rms = 0.0
        self.handles = []
        self.geometry = model_head_geometry(model)
        for layer_id, layer in enumerate(model.model.layers):
            module = layer.self_attn.o_proj
            self.handles.append(module.register_forward_pre_hook(self._pre(layer_id)))
            self.handles.append(module.register_forward_hook(self._post(layer_id)))

    def _pre(self, layer_id: int):
        def hook(_module, args):
            self.inputs[layer_id] = args[0]
        return hook

    def _post(self, layer_id: int):
        def hook(module, _args, output):
            source = self.inputs.pop(layer_id)
            h, d = self.geometry["query_heads"], self.geometry["head_dim"]
            # Qwen3's autocast o_proj multiplies BF16-quantized inputs and
            # weights, accumulating the products in FP32 before its final
            # output cast. Reconstruct that operation per head in FP32.
            x = source[:, -1, :].to(torch.bfloat16).float().reshape(source.shape[0], h, d)
            weight = (
                module.weight.detach().to(torch.bfloat16).float()
                .reshape(self.geometry["hidden_size"], h, d)
                .permute(1, 2, 0)
            )
            with torch.autocast(device_type="cuda", enabled=False):
                contributions = torch.einsum("bhd,hdo->bho", x, weight)
            if module.bias is not None:
                # Bias belongs to o_proj as a whole; it is not assigned to an individual head.
                expected = output[:, -1].float() - module.bias.detach().float()
            else:
                expected = output[:, -1].float()
            reconstructed = contributions.sum(dim=1)
            residual_error = reconstructed - expected
            reconstruction_error = residual_error.abs().max().item()
            self.max_reconstruction_error = max(self.max_reconstruction_error, float(reconstruction_error))
            error_rms = residual_error.square().mean().sqrt()
            output_rms = expected.square().mean().sqrt().clamp_min(1e-12)
            relative_rms = float((error_rms / output_rms).item())
            self.max_reconstruction_relative_rms = max(self.max_reconstruction_relative_rms, relative_rms)
            if self.current is None:
                self.current = torch.empty(
                    self.geometry["layers"], h, self.geometry["hidden_size"],
                    dtype=torch.float32, device=source.device,
                )
            self.current[layer_id] = contributions[0]
            return output
        return hook

    def record(self, prompt_ids: list[int], device: torch.device) -> torch.Tensor:
        self.current = None
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = self.model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
        if self.current is None:
            raise RuntimeError("Head activation hooks did not capture any layers")
        result = self.current[...]
        del self.current
        self.current = None
        return result.detach().cpu()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class HeadActivationPatcher:
    """Patch one query-head residual contribution per batch row at the query's last token."""

    def __init__(self, model: Any, means: torch.Tensor):
        self.model = model
        self.means = means.to(device=next(model.parameters()).device, dtype=torch.float32)
        self.geometry = model_head_geometry(model)
        self.inputs: dict[int, torch.Tensor] = {}
        self.candidates: list[tuple[int, int]] = []
        self.position = 0
        self.handles = []
        for layer_id, layer in enumerate(model.model.layers):
            module = layer.self_attn.o_proj
            self.handles.append(module.register_forward_pre_hook(self._pre(layer_id)))
            self.handles.append(module.register_forward_hook(self._post(layer_id)))

    def _pre(self, layer_id: int):
        def hook(_module, args):
            self.inputs[layer_id] = args[0]
        return hook

    def _post(self, layer_id: int):
        def hook(module, _args, output):
            active = [(row, head) for row, (layer, head) in enumerate(self.candidates) if layer == layer_id]
            source = self.inputs.pop(layer_id)
            if not active:
                return output
            result = output.clone()
            d = self.geometry["head_dim"]
            hidden = self.geometry["hidden_size"]
            weight = module.weight.detach().float()
            for row, head in active:
                begin, end = head * d, (head + 1) * d
                quantized_input = source[row, self.position, begin:end].to(torch.bfloat16).float()
                quantized_weight = weight[:, begin:end].to(torch.bfloat16).float()
                with torch.autocast(device_type="cuda", enabled=False):
                    current = F.linear(quantized_input, quantized_weight)
                desired = self.means[layer_id, head]
                result[row, self.position] = result[row, self.position].float() + desired - current
            return result.to(dtype=output.dtype)
        return hook

    def run_candidates(
        self,
        prompt_ids: list[int],
        target_ids: list[int],
        candidates: list[tuple[int, int]],
        batch_size: int = PATCH_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        device = next(self.model.parameters()).device
        prompt_len = len(prompt_ids)
        full = make_complete_ids(prompt_ids, target_ids, device)
        per_candidate: list[dict[str, Any]] = []
        self.position = prompt_len - 1
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start : start + batch_size]
            self.candidates = chunk
            x = full.expand(len(chunk), -1)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logp, _ = token_probabilities(self.model, x, prompt_len, target_ids)
            logp_cpu = logp.float().cpu()
            for offset, (layer_id, head_id) in enumerate(chunk):
                per_candidate.append(
                    {
                        "layer": int(layer_id),
                        "query_head": int(head_id),
                        "mean_sequence_logp": float(logp_cpu[offset].mean().item()),
                        "first_token_logp": float(logp_cpu[offset, 0].item()),
                    }
                )
            del logp, logp_cpu, x
        self.candidates = []
        return per_candidate

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ResidualVectorIntervention:
    """Add/subtract an FV after one Qwen decoder block at a chosen sequence position."""

    def __init__(self, model: Any):
        self.model = model
        self.layer: int | None = None
        self.vector: torch.Tensor | None = None
        self.scale = 0.0
        self.position: int | None = None
        self.handles = []
        for layer_id, block in enumerate(model.model.layers):
            self.handles.append(block.register_forward_hook(self._hook(layer_id)))

    def _hook(self, layer_id: int):
        def hook(_module, _args, output):
            if self.vector is None or self.layer != layer_id or self.position is None:
                return output
            if not torch.is_tensor(output) or self.position >= output.shape[1]:
                return output
            patched = output.clone()
            vector = self.vector.to(device=output.device, dtype=torch.float32) * float(self.scale)
            patched[:, self.position, :] = (patched[:, self.position, :].float() + vector).to(output.dtype)
            return patched
        return hook

    def activate(self, layer: int, vector: torch.Tensor | None, scale: float, position: int) -> None:
        self.layer = int(layer)
        self.vector = None if vector is None else vector
        self.scale = float(scale)
        self.position = int(position)

    def deactivate(self):
        self.vector = None
        self.layer = None
        self.position = None
        self.scale = 0.0

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.deactivate()


def reference_sequence(model: Any, prompt_ids: list[int], target_ids: list[int], device: torch.device):
    x = make_complete_ids(prompt_ids, target_ids, device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        selected_logp, first_logits = token_probabilities(model, x, len(prompt_ids), target_ids)
    values = selected_logp[0].float().cpu()
    return {
        "mean_logp": float(values.mean().item()),
        "sum_logp": float(values.sum().item()),
        "nll_sum": float((-values).sum().item()),
        "token_count": int(values.numel()),
        "first_token_logp": float(values[0].item()),
        "first_token_probability": float(values[0].exp().item()),
        "first_token_prediction_id": int(first_logits[0].argmax().item()),
        "first_token_logits": first_logits[0].detach().cpu(),
    }


def zero_shot_teacher_forced(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    model_config: dict[str, Any],
    device: torch.device,
    intervention: ResidualVectorIntervention | None = None,
    vector_spec: tuple[int, torch.Tensor | None, float] | None = None,
):
    input_ids, labels, diag = encode_training_record(
        tokenizer,
        record,
        int(model_config["max_sequence_length"]),
        int(model_config["max_target_tokens"]),
    )
    prompt_len = next(i for i, label in enumerate(labels) if label != -100)
    prompt_len = int(prompt_len)
    target_ids = [int(token) for token in input_ids[prompt_len:]]
    full = torch.tensor([input_ids], dtype=torch.long, device=device)
    if intervention is not None:
        if vector_spec is None:
            intervention.deactivate()
        else:
            layer, vector, scale = vector_spec
            intervention.activate(layer, vector, scale, prompt_len - 1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        selected_logp, first_logits = token_probabilities(model, full, prompt_len, target_ids)
    selected_logp = selected_logp[0].float().cpu()
    first_logits = first_logits[0].float().cpu()
    return {
        "mean_logp": float(selected_logp.mean().item()),
        "sum_logp": float(selected_logp.sum().item()),
        "nll_sum": float((-selected_logp).sum().item()),
        "token_count": int(selected_logp.numel()),
        "first_token_logp": float(selected_logp[0].item()),
        "first_token_probability": float(selected_logp[0].exp().item()),
        "first_token_prediction_id": int(first_logits.argmax().item()),
        "first_token_logits": first_logits,
        "target_token_ids": target_ids,
        "prompt_token_ids": input_ids[:prompt_len],
        "prompt_diagnostics": diag,
    }


def zero_shot_generate(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    model_config: dict[str, Any],
    eval_config: dict[str, Any],
    device: torch.device,
    intervention: ResidualVectorIntervention | None = None,
    vector_spec: tuple[int, torch.Tensor | None, float] | None = None,
):
    prompt_ids, _shortened, diagnostics = safe_prompt(
        tokenizer, record, int(model_config["max_sequence_length"])
    )
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if intervention is not None:
        if vector_spec is None:
            intervention.deactivate()
        else:
            layer, vector, scale = vector_spec
            intervention.activate(layer, vector, scale, len(prompt_ids) - 1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=int(eval_config["generation_max_new_tokens"]),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    completion = generated[0, len(prompt_ids) :]
    text = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return {
        "prediction": text,
        "generated_token_count": int(completion.numel()),
        "eos_terminated": bool(completion.numel() and int(completion[-1]) == int(tokenizer.eos_token_id)),
        "prompt_diagnostics": diagnostics,
        "prompt_token_ids": prompt_ids,
    }


def score_prediction(task_id: str, record: dict[str, Any], generated: dict[str, Any], task_type: dict[str, Any]):
    prediction = generated["prediction"]
    row = {
        "task_id": task_id,
        "instance_id": record["instance_id"],
        "prediction": prediction,
        "references": list(record.get("targets", [])),
        "generated_token_count": int(generated["generated_token_count"]),
        "eos_terminated": bool(generated["eos_terminated"]),
        "prompt_diagnostics": generated["prompt_diagnostics"],
    }
    if task_type["type"] == "classification":
        parsed = parse_yes_no(prediction)
        gold = parse_yes_no(str(record["targets"][0]))
        row["parsed_prediction"] = parsed
        row["gold_label"] = gold
        row["correct"] = parsed == gold if parsed is not None and gold is not None else False
        row["parse_ok"] = parsed is not None
    else:
        refs = record.get("targets", [""]) or [""]
        row["rouge_l_f1"] = max(rouge_l_f1(prediction, ref) for ref in refs)
        row["token_f1"] = max(token_f1(prediction, ref) for ref in refs)
        row["exact_match"] = float(any(norm_text(prediction) == norm_text(ref) for ref in refs))
    return row


def aggregate_task(task_id: str, rows: list[dict[str, Any]], task_type: dict[str, Any], nll_rows: list[dict[str, Any]]):
    metrics = score_rows(task_id, rows, task_type)
    token_count = sum(int(row["token_count"]) for row in nll_rows)
    nll_sum = sum(float(row["nll_sum"]) for row in nll_rows)
    metrics.update(
        {
            "teacher_forced_token_nll": nll_sum / max(1, token_count),
            "teacher_forced_nll_sum": nll_sum,
            "teacher_forced_target_tokens": token_count,
            "mean_per_example_nll": sum(float(row["nll_sum"]) / max(1, int(row["token_count"])) for row in nll_rows)
            / max(1, len(nll_rows)),
            "mean_first_reference_token_probability": sum(float(row["first_token_probability"]) for row in nll_rows)
            / max(1, len(nll_rows)),
            "mean_generated_token_count": sum(int(row["generated_token_count"]) for row in rows) / max(1, len(rows)),
            "evaluation_count": len(rows),
        }
    )
    total = max(1, len(rows))
    parsed = [parse_yes_no(row.get("prediction", "")) for row in rows]
    yes = sum(label == "Yes" for label in parsed)
    no = sum(label == "No" for label in parsed)
    metrics.update(
        {
            "yes_output_share": yes / total,
            "no_output_share": no / total,
            "yes_no_collapse_rate": max(yes, no) / total,
        }
    )
    if task_type["type"] == "classification":
        metrics["unparsed_output_share"] = (total - yes - no) / total
    else:
        metrics["mean_rouge_l_f1"] = sum(float(row.get("rouge_l_f1", 0.0)) for row in rows) / max(1, len(rows))
        metrics["mean_token_f1"] = sum(float(row.get("token_f1", 0.0)) for row in rows) / max(1, len(rows))
    return metrics


def split_records(dev: dict[str, list[dict[str, Any]]], split_name: str):
    start, end = SPLITS[split_name]
    return {task_id: rows[start:end] for task_id, rows in dev.items()}


def norm_match_random(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_vector = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return random_vector * (reference.float().norm() / random_vector.norm().clamp_min(1e-12))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None

