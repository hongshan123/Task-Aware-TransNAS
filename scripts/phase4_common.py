"""Shared local-only training and evaluation helpers for Phase 4A."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from run_formal_baseline import (
    PROJECT_ROOT,
    answer_tokens_for,
    encode_training_record,
    evaluate_task,
    load_manifest_records,
    load_model,
    load_project_config,
    load_tokenizer,
    resolved_model_path,
    safe_prompt,
    safe_task_seed,
)


TASK_IDS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
GENERATION_TASKS = TASK_IDS[:2]


def prepare_context() -> tuple[dict[str, Any], Path, Any, torch.device, dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_project_config(PROJECT_ROOT / "configs" / "continual_baseline_full.yaml")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4A requires CUDA; no CUDA device is available")
    device = torch.device("cuda:0")
    model_path = resolved_model_path(config)
    tokenizer = load_tokenizer(model_path, local_only=True)
    train = load_manifest_records(config, "train", TASK_IDS)
    dev = load_manifest_records(config, "dev", TASK_IDS)
    return config, model_path, tokenizer, device, train, dev


def teacher_forced_nll(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Token-weighted completion NLL with the formal SFT prompt/target encoder."""
    was_training = model.training
    model.eval()
    model.config.use_cache = False
    total_nll = 0.0
    total_tokens = 0
    per_example: list[float] = []
    with torch.inference_mode():
        for record in records:
            ids, labels, _ = encode_training_record(
                tokenizer,
                record,
                int(model_cfg["max_sequence_length"]),
                int(model_cfg["max_target_tokens"]),
            )
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
            shifted_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
            shifted_labels = y[:, 1:].reshape(-1)
            token_mask = shifted_labels.ne(-100)
            row_tokens = int(token_mask.sum().item())
            row_loss_sum = float(
                F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100, reduction="sum").item()
            )
            if row_tokens:
                per_example.append(row_loss_sum / row_tokens)
                total_nll += row_loss_sum
                total_tokens += row_tokens
    if was_training:
        model.train()
    model.config.use_cache = True
    return {
        "token_nll": total_nll / max(1, total_tokens),
        "supervised_tokens": total_tokens,
        "examples": len(per_example),
        "mean_per_example_nll": sum(per_example) / max(1, len(per_example)),
        "per_example_nll": per_example,
    }


def first_token_probabilities(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    device: torch.device,
    top_k: int = 8,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    yes_ids = set(tokenizer.encode("Yes", add_special_tokens=False)) | set(tokenizer.encode(" Yes", add_special_tokens=False))
    no_ids = set(tokenizer.encode("No", add_special_tokens=False)) | set(tokenizer.encode(" No", add_special_tokens=False))
    yes_ids.discard(tokenizer.unk_token_id)
    no_ids.discard(tokenizer.unk_token_id)
    first_token_probabilities: list[float] = []
    yes_probabilities: list[float] = []
    no_probabilities: list[float] = []
    top1_token_counts: dict[int, int] = {}
    topk_token_counts: dict[int, int] = {}
    top_token_probability_sums: dict[int, float] = {}
    target_token_probability: list[float] = []
    with torch.inference_mode():
        for record in records:
            prompt, shortened_input, _ = safe_prompt(tokenizer, record, int(model_cfg["max_sequence_length"]))
            x = torch.tensor([prompt], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=True).logits[0, -1].float()
            probs = logits.softmax(dim=-1)
            values, indices = torch.topk(probs, min(top_k, probs.numel()))
            top1_id = int(indices[0].item())
            top1_token_counts[top1_id] = top1_token_counts.get(top1_id, 0) + 1
            for token_id, probability in zip(indices.tolist(), values.tolist()):
                topk_token_counts[token_id] = topk_token_counts.get(token_id, 0) + 1
                top_token_probability_sums[token_id] = top_token_probability_sums.get(token_id, 0.0) + probability
            first_token_probabilities.append(float(values[0].item()))
            yes_probabilities.append(float(probs[list(yes_ids)].sum().item()) if yes_ids else 0.0)
            no_probabilities.append(float(probs[list(no_ids)].sum().item()) if no_ids else 0.0)
            answer_ids = answer_tokens_for(tokenizer, record, shortened_input, record["targets"][0])
            if answer_ids:
                target_token_probability.append(float(probs[answer_ids[0]].item()))
    if was_training:
        model.train()
    ranked = sorted(top1_token_counts, key=lambda token_id: (-top1_token_counts[token_id], token_id))[:top_k]
    top_distribution = []
    for token_id in ranked:
        token = tokenizer.convert_ids_to_tokens(token_id)
        top_distribution.append(
            {
                "token_id": token_id,
                "token": token,
                "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
                "top1_frequency": top1_token_counts[token_id] / max(1, len(records)),
                "topk_frequency": topk_token_counts[token_id] / max(1, len(records)),
                "mean_probability_when_topk": top_token_probability_sums[token_id] / max(1, topk_token_counts[token_id]),
            }
        )
    return {
        "mean_top1_probability": sum(first_token_probabilities) / max(1, len(first_token_probabilities)),
        "mean_reference_first_token_probability": sum(target_token_probability) / max(1, len(target_token_probability)),
        "mean_yes_token_probability": sum(yes_probabilities) / max(1, len(yes_probabilities)),
        "mean_no_token_probability": sum(no_probabilities) / max(1, len(no_probabilities)),
        "yes_token_ids": sorted(yes_ids),
        "no_token_ids": sorted(no_ids),
        "top1_token_distribution": top_distribution,
    }


def train_task_recorded(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    training_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    learning_rate: float,
    steps: int,
    seed: int,
    on_checkpoint: Callable[[int, dict[str, Any]], None] | None = None,
    checkpoint_steps: set[int] | None = None,
    on_optimizer_step: Callable[[int, Any], None] | None = None,
) -> dict[str, Any]:
    """Formal-baseline training loop with token-budget and sample-order auditing."""
    torch.manual_seed(seed)
    randomizer = random.Random(seed)
    model.train()
    model.config.use_cache = False
    max_len = int(model_cfg["max_sequence_length"])
    max_target = int(model_cfg["max_target_tokens"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=float(training_cfg["weight_decay"]),
        foreach=False,
    )
    order: list[int] = []
    cursor = 0
    consumed_instance_ids: list[str] = []
    losses: list[float] = []
    tokens_per_step: list[int] = []
    truncation_count = 0
    peak_allocated = 0
    peak_reserved = 0
    started = time.perf_counter()

    def next_row() -> dict[str, Any]:
        nonlocal order, cursor
        if cursor >= len(order):
            order = list(range(len(records)))
            randomizer.shuffle(order)
            cursor = 0
        row = records[order[cursor]]
        cursor += 1
        return row

    for step in range(1, steps + 1):
        rows = [next_row() for _ in range(grad_accum)]
        consumed_instance_ids.extend(str(row["instance_id"]) for row in rows)
        retry = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            local_tokens = 0
            local_truncations = 0
            try:
                for row in rows:
                    ids, labels, diag = encode_training_record(tokenizer, row, max_len, max_target)
                    local_truncations += int(diag["input_tail_truncated"])
                    local_tokens += sum(label != -100 for label in labels)
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    y = torch.tensor([labels], dtype=torch.long, device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
                    (loss / grad_accum).backward()
                    step_loss += float(loss.detach().float().item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["max_grad_norm"]))
                optimizer.step()
                losses.append(step_loss / grad_accum)
                tokens_per_step.append(local_tokens)
                truncation_count += local_truncations
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                break
            except BaseException as error:
                optimizer.zero_grad(set_to_none=True)
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    isinstance(error, RuntimeError) and "out of memory" in str(error).casefold()
                )
                if not is_oom:
                    raise
                torch.cuda.empty_cache()
                old_len = max_len
                max_len = max(192, max_len - 64)
                retry += 1
                if max_len == old_len or retry > 3:
                    raise RuntimeError("CUDA OOM continued after reducing the sequence length") from error
                print(f"OOM recovery: retry step {step} with max_sequence_length={max_len}", flush=True)
        if on_optimizer_step:
            on_optimizer_step(step, model)
        if on_checkpoint and checkpoint_steps and step in checkpoint_steps:
            elapsed = {
                "optimizer_steps": step,
                "mean_loss": sum(losses) / len(losses),
                "interval_mean_loss": sum(losses[max(0, step - 25) :]) / len(losses[max(0, step - 25) :]),
                "last_loss": losses[-1],
                "supervised_target_tokens": sum(tokens_per_step),
                "consumed_examples": step * grad_accum,
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
            }
            on_checkpoint(step, elapsed)
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"train {step}/{steps}: loss={losses[-1]:.4f}, max_seq={max_len}", flush=True)

    model.config.use_cache = True
    model.eval()
    return {
        "optimizer_steps": steps,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "interval_mean_loss": [
            sum(losses[i : i + 25]) / len(losses[i : i + 25]) for i in range(0, len(losses), 25)
        ],
        "losses": losses,
        "supervised_target_tokens": sum(tokens_per_step),
        "supervised_target_tokens_per_step": tokens_per_step,
        "consumed_examples": len(consumed_instance_ids),
        "sample_order_sha256": hashlib.sha256(
            json.dumps(consumed_instance_ids, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "effective_max_sequence_length": max_len,
        "training_prompts_with_input_tail_truncation": truncation_count,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "duration_seconds": time.perf_counter() - started,
    }


def evaluate_dev_task(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    task_id: str,
    task_type: dict[str, Any],
    model_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    device: torch.device,
    output_path: Path,
    max_examples: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_eval = dict(eval_cfg)
    if max_examples is not None:
        local_eval["max_examples_per_task"] = max_examples
    return evaluate_task(model, tokenizer, records, task_id, task_type, model_cfg, local_eval, device, output_path)


def human_vram() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "free_gib_at_end": free / (1024**3),
        "total_gib": total / (1024**3),
        "max_allocated_gib": torch.cuda.max_memory_allocated(0) / (1024**3),
        "max_reserved_gib": torch.cuda.max_memory_reserved(0) / (1024**3),
    }


def write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
