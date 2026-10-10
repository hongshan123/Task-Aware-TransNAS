"""Tune and run full-parameter continual SFT using local CITB data and a local model."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path("D:/Task-Aware-TransNAS")
os.environ.setdefault("HF_HOME", str(STORAGE_ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(STORAGE_ROOT / "cache" / "huggingface" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(STORAGE_ROOT / "cache" / "huggingface" / "datasets"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def token_ids(value: Any) -> list[int]:
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def norm_text(value: str) -> str:
    value = str(value)
    value = value.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", value).strip().strip(" \t\r\n.,!?;:'\"`()[]{}")


def messages_for(record: dict[str, Any], input_text: str | None = None) -> list[dict[str, str]]:
    system = "You are an instruction-following assistant. Follow the task definition and return only the requested answer."
    content = "Task definition:\n" + record["definition"] + "\n\nInput:\n" + (record["input"] if input_text is None else input_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def render_prompt(tokenizer: Any, record: dict[str, Any], input_text: str | None = None) -> list[int]:
    return token_ids(
        tokenizer.apply_chat_template(
            messages_for(record, input_text), tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )


def safe_prompt(
    tokenizer: Any, record: dict[str, Any], budget: int
) -> tuple[list[int], str, dict[str, Any]]:
    """Fit a prompt by truncating only the input suffix while preserving all instructions."""
    full_prompt = render_prompt(tokenizer, record)
    if len(full_prompt) <= budget:
        decoded = norm_text(tokenizer.decode(full_prompt, skip_special_tokens=True))
        if norm_text(record["definition"]) not in decoded:
            raise AssertionError(f"Task definition was lost in the rendered prompt for {record['instance_id']}")
        return full_prompt, record["input"], {
            "original_prompt_tokens": len(full_prompt),
            "retained_prompt_tokens": len(full_prompt),
            "input_tail_truncated": False,
            "retained_input_tokens": len(tokenizer.encode(record["input"], add_special_tokens=False)),
        }
    empty_input_prompt = render_prompt(tokenizer, record, "")
    if len(empty_input_prompt) > budget:
        raise ValueError(
            f"Complete task definition exceeds prompt budget for {record['instance_id']}: "
            f"{len(empty_input_prompt)}>{budget}"
        )
    input_tokens = tokenizer.encode(record["input"], add_special_tokens=False)
    low, high = 0, len(input_tokens)
    best_ids, best_text, best_keep = empty_input_prompt, "", 0
    while low <= high:
        keep = (low + high) // 2
        text = tokenizer.decode(input_tokens[-keep:], skip_special_tokens=True) if keep else ""
        candidate = render_prompt(tokenizer, record, text)
        if len(candidate) <= budget:
            best_ids, best_text, best_keep = candidate, text, keep
            low = keep + 1
        else:
            high = keep - 1
    # Retokenization at the prompt boundary can make lengths slightly non-monotonic.
    while len(best_ids) > budget and best_keep > 0:
        best_keep -= 1
        best_text = tokenizer.decode(input_tokens[-best_keep:], skip_special_tokens=True) if best_keep else ""
        best_ids = render_prompt(tokenizer, record, best_text)
    for keep in range(best_keep + 1, min(len(input_tokens), best_keep + 8) + 1):
        text = tokenizer.decode(input_tokens[-keep:], skip_special_tokens=True)
        candidate = render_prompt(tokenizer, record, text)
        if len(candidate) <= budget:
            best_ids, best_text, best_keep = candidate, text, keep
        else:
            break
    if len(best_ids) > budget:
        raise AssertionError("Prompt budget enforcement failed")
    decoded = norm_text(tokenizer.decode(best_ids, skip_special_tokens=True))
    if norm_text(record["definition"]) not in decoded:
        raise AssertionError(f"Task definition was lost during input truncation for {record['instance_id']}")
    return best_ids, best_text, {
        "original_prompt_tokens": len(full_prompt),
        "retained_prompt_tokens": len(best_ids),
        "input_tail_truncated": best_keep < len(input_tokens),
        "retained_input_tokens": best_keep,
    }


def answer_tokens_for(tokenizer: Any, record: dict[str, Any], input_text: str, target: str) -> list[int]:
    messages = messages_for(record, input_text)
    prompt = token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    complete = token_ids(
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError(f"Chat-template prompt/answer prefix mismatch for {record['instance_id']}")
    return complete[len(prompt) :]


def encode_training_record(
    tokenizer: Any, record: dict[str, Any], max_sequence_length: int, max_target_tokens: int
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not record["targets"] or not record["targets"][0].strip():
        raise ValueError(f"No nonempty target for {record['instance_id']}")
    target = record["targets"][0]
    original_answer = answer_tokens_for(tokenizer, record, record["input"], target)
    if not original_answer:
        raise ValueError(f"No assistant answer tokens for {record['instance_id']}")
    target_was_capped = len(original_answer) >= max_target_tokens
    answer = (
        original_answer[: max_target_tokens - 1] + [tokenizer.eos_token_id]
        if target_was_capped
        else original_answer
    )
    prompt_budget = max_sequence_length - len(answer)
    if prompt_budget < 16:
        raise ValueError(f"Target leaves fewer than 16 prompt tokens for {record['instance_id']}")
    prompt, shortened_input, prompt_info = safe_prompt(tokenizer, record, prompt_budget)
    answer = answer_tokens_for(tokenizer, record, shortened_input, target)
    if len(answer) >= max_target_tokens:
        answer = answer[: max_target_tokens - 1] + [tokenizer.eos_token_id]
        target_was_capped = True
    if len(prompt) + len(answer) > max_sequence_length:
        prompt_budget = max_sequence_length - len(answer)
        prompt, shortened_input, prompt_info = safe_prompt(tokenizer, record, prompt_budget)
        answer = answer_tokens_for(tokenizer, record, shortened_input, target)
        if len(answer) >= max_target_tokens:
            answer = answer[: max_target_tokens - 1] + [tokenizer.eos_token_id]
            target_was_capped = True
    if len(prompt) + len(answer) > max_sequence_length:
        raise AssertionError(f"Sequence budget enforcement failed for {record['instance_id']}")
    labels = [-100] * len(prompt) + answer
    diag = {
        **prompt_info,
        "target_tokens_before_cap": len(original_answer),
        "target_tokens_used": len(answer),
        "target_was_capped": target_was_capped,
        "task_definition_preserved": True,
    }
    return prompt + answer, labels, diag


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’][\w]+)?", text.casefold(), flags=re.UNICODE)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = word_tokens(prediction), word_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for pred_token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            current.append(previous[index - 1] + 1 if pred_token == ref_token else max(previous[index], current[index - 1]))
        previous = current
    return 2 * previous[-1] / (len(pred) + len(ref))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(word_tokens(prediction)), Counter(word_tokens(reference))
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((pred & ref).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def parse_yes_no(prediction: str) -> str | None:
    text = prediction.strip()
    match = re.match(
        r"^\s*(?:(?:the\s+)?(?:answer|label|class)\s*(?:is|=|:|-)?\s*)?['\"`(]*\b(yes|no)\b(?=\s|$|[.,;:!?])",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).title() if match else None


def macro_f1(golds: list[str], predictions: list[str | None], labels: list[str]) -> float:
    per_label = []
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(golds, predictions))
        fp = sum(gold != label and pred == label for gold, pred in zip(golds, predictions))
        fn = sum(gold == label and pred != label for gold, pred in zip(golds, predictions))
        denominator = 2 * tp + fp + fn
        per_label.append(2 * tp / denominator if denominator else 0.0)
    return sum(per_label) / len(per_label)


def score_rows(task_id: str, rows: list[dict[str, Any]], task_type: dict[str, Any]) -> dict[str, Any]:
    if task_type["type"] == "classification":
        labels = task_type["labels"]
        label_map = {norm_text(label): label for label in labels}
        golds: list[str] = []
        predictions: list[str | None] = []
        correct = 0
        for row in rows:
            gold = [label_map[norm_text(ref)] for ref in row["references"] if norm_text(ref) in label_map]
            if not gold:
                raise ValueError(f"Reference is not a configured class label: {task_id}/{row['instance_id']}")
            prediction = parse_yes_no(row["prediction"])
            golds.append(gold[0])
            predictions.append(prediction)
            correct += int(prediction in gold)
        accuracy = correct / len(rows)
        return {
            "task_type": "classification",
            "primary_metric": "accuracy",
            "primary_score": accuracy,
            "accuracy": accuracy,
            "macro_f1": macro_f1(golds, predictions, labels),
            "parse_success_rate": sum(pred is not None for pred in predictions) / len(predictions),
            "class_labels": labels,
            "evaluated": len(rows),
        }
    per_row = []
    for row in rows:
        refs = row["references"] or [""]
        per_row.append(
            {
                "rouge_l_f1": max(rouge_l_f1(row["prediction"], ref) for ref in refs),
                "token_f1": max(token_f1(row["prediction"], ref) for ref in refs),
                "exact_match": float(any(norm_text(row["prediction"]) == norm_text(ref) for ref in refs)),
            }
        )
    rouge = sum(item["rouge_l_f1"] for item in per_row) / len(per_row)
    return {
        "task_type": "generation",
        "primary_metric": "rouge_l_f1",
        "primary_score": rouge,
        "rouge_l_f1": rouge,
        "token_f1": sum(item["token_f1"] for item in per_row) / len(per_row),
        "exact_match": sum(item["exact_match"] for item in per_row) / len(per_row),
        "evaluated": len(rows),
    }


def safe_task_seed(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def train_task(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    training_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    learning_rate: float,
    steps: int,
    seed: int,
) -> dict[str, Any]:
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

    def next_row() -> dict[str, Any]:
        nonlocal order, cursor
        if cursor >= len(order):
            order = list(range(len(records)))
            randomizer.shuffle(order)
            cursor = 0
        row = records[order[cursor]]
        cursor += 1
        return row

    losses: list[float] = []
    retries: list[dict[str, Any]] = []
    truncation_count = 0
    max_cuda_allocated = 0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        rows = [next_row() for _ in range(grad_accum)]
        retry = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            try:
                local_truncations = 0
                for row in rows:
                    ids, labels, diag = encode_training_record(tokenizer, row, max_len, max_target)
                    local_truncations += int(diag["input_tail_truncated"])
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    y = torch.tensor([labels], dtype=torch.long, device=device)
                    attention = torch.ones_like(x)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(input_ids=x, attention_mask=attention, labels=y, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
                    (loss / grad_accum).backward()
                    step_loss += float(loss.detach().float().item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["max_grad_norm"]))
                optimizer.step()
                truncation_count += local_truncations
                losses.append(step_loss / grad_accum)
                max_cuda_allocated = max(max_cuda_allocated, torch.cuda.max_memory_allocated(device))
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
                    raise RuntimeError("CUDA OOM continued after lowering the sequence length") from error
                retries.append({"step": step, "from_max_sequence_length": old_len, "to_max_sequence_length": max_len})
                print(f"OOM recovery: retry step {step} at max_sequence_length={max_len}", flush=True)
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
        "losses": losses,
        "effective_max_sequence_length": max_len,
        "training_prompts_with_input_tail_truncation": truncation_count,
        "oom_retries": retries,
        "peak_cuda_allocated_bytes": max_cuda_allocated,
        "duration_seconds": time.perf_counter() - started,
    }


def evaluate_task(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    task_id: str,
    task_type: dict[str, Any],
    model_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    prompt_diagnostics = []
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    for index, record in enumerate(records[: int(eval_cfg["max_examples_per_task"])], start=1):
        prompt, _, diag = safe_prompt(tokenizer, record, int(model_cfg["max_sequence_length"]))
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        attention = torch.ones_like(x)
        max_new_tokens = int(eval_cfg["generation_max_new_tokens"])
        oom_retry = 0
        while True:
            try:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        input_ids=x,
                        attention_mask=attention,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                prediction = tokenizer.decode(generated[0, x.shape[1] :], skip_special_tokens=True).strip()
                break
            except BaseException as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    isinstance(error, RuntimeError) and "out of memory" in str(error).casefold()
                )
                if not is_oom:
                    raise
                torch.cuda.empty_cache()
                oom_retry += 1
                if max_new_tokens > 32:
                    max_new_tokens //= 2
                elif len(prompt) > 192:
                    prompt, _, diag = safe_prompt(tokenizer, record, max(192, len(prompt) - 64))
                    x = torch.tensor([prompt], dtype=torch.long, device=device)
                    attention = torch.ones_like(x)
                else:
                    raise RuntimeError(f"Evaluation OOM at minimum token budget: {task_id}/{record['instance_id']}") from error
                if oom_retry > 4:
                    raise RuntimeError(f"Too many evaluation OOM recoveries for {record['instance_id']}") from error
            finally:
                if oom_retry:
                    torch.cuda.empty_cache()
        row = {
            "task_id": task_id,
            "instance_id": record["instance_id"],
            "prediction": prediction,
            "references": record["targets"],
            "scores": {},
            "prompt_diagnostics": diag,
            "oom_retries": oom_retry,
            "eos_terminated": bool(generated[0, -1].item() == tokenizer.eos_token_id),
            "generated_token_count": int(generated.shape[1] - x.shape[1]),
            "generation_max_new_tokens_used": max_new_tokens,
        }
        if task_type["type"] == "classification":
            parsed = parse_yes_no(prediction)
            row["scores"] = {"parsed_prediction": parsed, "parse_ok": parsed is not None}
        else:
            refs = record["targets"] or [""]
            row["scores"] = {
                "rouge_l_f1": max(rouge_l_f1(prediction, ref) for ref in refs),
                "token_f1": max(token_f1(prediction, ref) for ref in refs),
                "exact_match": float(any(norm_text(prediction) == norm_text(ref) for ref in refs)),
            }
        rows.append(row)
        prompt_diagnostics.append(diag)
        if index == 1 or index % 10 == 0 or index == min(len(records), int(eval_cfg["max_examples_per_task"])):
            print(f"eval {task_id}: {index}/{min(len(records), int(eval_cfg['max_examples_per_task']))}", flush=True)
    if was_training:
        model.train()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metric = score_rows(task_id, rows, task_type)
    metric["input_tail_truncated_count"] = sum(item["input_tail_truncated"] for item in prompt_diagnostics)
    metric["definition_preservation_rate"] = 1.0
    metric["mean_prompt_tokens"] = sum(item["retained_prompt_tokens"] for item in prompt_diagnostics) / max(1, len(prompt_diagnostics))
    metric["eos_termination_rate"] = sum(row["eos_terminated"] for row in rows) / max(1, len(rows))
    metric["oom_retry_count"] = sum(row["oom_retries"] for row in rows)
    metric["prediction_file"] = str(output_path)
    return metric, rows


def normalized_gain(score: float, baseline: float) -> float:
    return (score - baseline) / max(1.0 - baseline, 1e-8)


def score_summary(
    stage_scores: dict[str, dict[str, dict[str, Any]]], task_ids: list[str], stage_names: list[str], sequence: list[str]
) -> dict[str, Any]:
    base = {task: stage_scores[stage_names[0]][task]["primary_score"] for task in task_ids}
    normalized = {
        stage: {task: normalized_gain(stage_scores[stage][task]["primary_score"], base[task]) for task in task_ids}
        for stage in stage_names
    }
    result = {}
    for index, stage in enumerate(stage_names):
        seen = sequence[:index]
        gains = normalized[stage]
        if index == 0:
            forgetting = 0.0
        else:
            prior = sequence[: index - 1]
            drops = [max(normalized[earlier][task] for earlier in stage_names[1:index]) - gains[task] for task in prior]
            forgetting = sum(drops) / len(drops) if drops else 0.0
        result[stage] = {
            "normalized_average_accuracy_all_tasks": sum(gains.values()) / len(task_ids),
            "aaa_normalized_seen_tasks": sum(gains[task] for task in seen) / len(seen) if seen else 0.0,
            "average_forgetting_normalized": forgetting,
            "normalized_gain_by_task": gains,
            "raw_classification_accuracy_mean": mean_metric(stage_scores[stage], task_ids, "classification", "accuracy"),
            "raw_generation_rouge_l_mean": mean_metric(stage_scores[stage], task_ids, "generation", "rouge_l_f1"),
        }
    return result


def mean_metric(stage: dict[str, dict[str, Any]], task_ids: list[str], kind: str, metric: str) -> float | None:
    values = [stage[task][metric] for task in task_ids if stage[task]["task_type"] == kind]
    return sum(values) / len(values) if values else None


def load_project_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolved_model_path(config: dict[str, Any]) -> Path:
    model_value = os.environ.get(config["model"]["env_var"])
    if not model_value:
        raise EnvironmentError(f"Set {config['model']['env_var']} in the project .env file")
    path = Path(model_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {path}")
    return path


def load_tokenizer(model_path: Path, local_only: bool) -> Any:
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=local_only, trust_remote_code=False)


def load_model(model_path: Path, local_only: bool, training: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=local_only, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(torch.device("cuda:0"))
    for parameter in model.parameters():
        parameter.requires_grad_(training)
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
    else:
        model.config.use_cache = True
        model.eval()
    return model


def load_manifest_records(config: dict[str, Any], split: str, task_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    manifest_path = Path(config["dataset"]["processed_dir"]) / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("leakage_check", {}).get("passed"):
        raise RuntimeError("Prepared CITB split manifest does not pass its leakage check")
    info_by_task = {row["task_id"]: row for row in manifest["tasks"]}
    if set(task_ids) != set(info_by_task):
        raise ValueError("Prepared split and configured task set differ")
    output = {}
    for task_id in task_ids:
        info = info_by_task[task_id]
        path = Path(info["split_files"][split])
        output[task_id] = read_jsonl(path)
        if len(output[task_id]) != info["counts"][split]:
            raise ValueError(f"Split row count mismatch for {task_id}/{split}")
    return output


def validate_data(config: dict[str, Any], tokenizer: Any, task_ids: list[str]) -> dict[str, Any]:
    training = load_manifest_records(config, "train", task_ids)
    development = load_manifest_records(config, "dev", task_ids)
    model_cfg = config["model"]
    reports = {}
    for task_id in task_ids:
        task_cfg = config["dataset"]["task_types"][task_id]
        seen_classes: Counter[str] = Counter()
        aggregate = {"train": Counter(), "dev": Counter()}
        for split_name, records in (("train", training[task_id]), ("dev", development[task_id])):
            for row in records:
                ids, labels, diag = encode_training_record(
                    tokenizer, row, int(model_cfg["max_sequence_length"]), int(model_cfg["max_target_tokens"])
                )
                if len(ids) > int(model_cfg["max_sequence_length"]) or not any(label != -100 for label in labels):
                    raise AssertionError(f"Invalid SFT encoding for {task_id}/{row['instance_id']}")
                aggregate[split_name]["examples"] += 1
                aggregate[split_name]["input_tail_truncated"] += int(diag["input_tail_truncated"])
                aggregate[split_name]["target_capped"] += int(diag["target_was_capped"])
                aggregate[split_name]["max_total_tokens"] = max(aggregate[split_name]["max_total_tokens"], len(ids))
                if task_cfg["type"] == "classification":
                    label_map = {norm_text(label): label for label in task_cfg["labels"]}
                    values = [label_map[norm_text(value)] for value in row["targets"] if norm_text(value) in label_map]
                    if not values:
                        raise ValueError(f"No configured class label for {task_id}/{row['instance_id']}")
                    if split_name == "train":
                        seen_classes[values[0]] += 1
            if not aggregate[split_name]["examples"]:
                raise AssertionError(f"Empty {split_name} for {task_id}")
        if task_cfg["type"] == "classification" and set(seen_classes) != set(task_cfg["labels"]):
            raise ValueError(f"Training split misses configured class labels for {task_id}: {seen_classes}")
        reports[task_id] = {
            "type": task_cfg["type"],
            "class_counts_train": dict(seen_classes),
            "train": dict(aggregate["train"]),
            "dev": dict(aggregate["dev"]),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_read": False,
        "all_task_definitions_preserved": True,
        "model_max_sequence_length": model_cfg["max_sequence_length"],
        "tasks": reports,
    }


def run_evaluation_matrix(
    model: Any,
    tokenizer: Any,
    records_by_task: dict[str, list[dict[str, Any]]],
    task_ids: list[str],
    config: dict[str, Any],
    device: torch.device,
    root: Path,
) -> dict[str, dict[str, Any]]:
    stage = {}
    for task_id in task_ids:
        metric, _ = evaluate_task(
            model,
            tokenizer,
            records_by_task[task_id],
            task_id,
            config["dataset"]["task_types"][task_id],
            config["model"],
            config["evaluation"],
            device,
            root / f"{task_id}.jsonl",
        )
        stage[task_id] = metric
        print(f"{task_id}: {metric['primary_metric']}={metric['primary_score']:.4f}", flush=True)
    return stage


def tune(config_path: Path, config: dict[str, Any], output_root: Path) -> None:
    if config["training"].get("steps_per_task") is not None or config["training"].get("learning_rate") is not None:
        raise ValueError("Tuning expects unlocked training.steps_per_task and learning_rate in the config")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = resolved_model_path(config)
    task_ids = list(config["dataset"]["tasks"])
    train_records = load_manifest_records(config, "train", task_ids)
    dev_records = load_manifest_records(config, "dev", task_ids)
    tune_sequence = config["sequences"][config["tuning"]["sequence"]]
    if set(tune_sequence) != set(task_ids):
        raise ValueError("Tuning sequence must contain each selected task exactly once")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    tokenizer = load_tokenizer(model_path, bool(config["model"]["local_files_only"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tuning preflight: validating only train/dev; the test split is not opened.", flush=True)
    validation = validate_data(config, tokenizer, task_ids)
    write_json(output_root / "tuning" / "preflight.json", validation)
    baseline_model = load_model(model_path, bool(config["model"]["local_files_only"]), training=False)
    baseline = run_evaluation_matrix(baseline_model, tokenizer, dev_records, task_ids, config, device, output_root / "tuning" / "predictions" / "base_dev")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    grid = [(float(lr), int(steps)) for lr in config["tuning"]["learning_rates"] for steps in config["tuning"]["steps_per_task"]]
    baseline_flat = {task: baseline[task]["primary_score"] for task in task_ids}
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "local_files_only": True,
        "data_manifest": str(Path(config["dataset"]["processed_dir"]) / "data_manifest.json"),
        "source_commit": json.loads((Path(config["dataset"]["processed_dir"]) / "data_manifest.json").read_text(encoding="utf-8"))["source"]["commit"],
        "selection_split": "dev",
        "test_split_opened": False,
        "tuning_sequence": tune_sequence,
        "baseline_dev_scores": baseline,
        "candidates": [],
    }
    for candidate_index, (learning_rate, steps) in enumerate(grid, start=1):
        candidate_id = f"lr_{learning_rate:.0e}_steps_{steps}"
        print(f"\n=== Dev tuning candidate {candidate_index}/{len(grid)}: {candidate_id} ===", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        model = load_model(model_path, bool(config["model"]["local_files_only"]), training=True)
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if total_params != trainable_params:
            raise RuntimeError("Formal tuning must update all model parameters")
        candidate_train = {}
        for task_id in tune_sequence:
            task_seed = safe_task_seed(int(config["seed"]), task_id)
            candidate_train[task_id] = train_task(
                model, tokenizer, train_records[task_id], config["training"], config["model"], device,
                learning_rate, steps, task_seed,
            )
        scores = run_evaluation_matrix(
            model, tokenizer, dev_records, task_ids, config, device,
            output_root / "tuning" / "predictions" / candidate_id / "dev",
        )
        gains = {task: normalized_gain(scores[task]["primary_score"], baseline_flat[task]) for task in task_ids}
        candidate = {
            "candidate_id": candidate_id,
            "learning_rate": learning_rate,
            "steps_per_task": steps,
            "dev_scores": scores,
            "normalized_gain_by_task": gains,
            "normalized_average_accuracy_all_tasks": sum(gains.values()) / len(gains),
            "mean_training_loss_by_task": {task: candidate_train[task]["mean_loss"] for task in tune_sequence},
            "training_metrics_by_task": candidate_train,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
        result["candidates"].append(candidate)
        write_json(output_root / "tuning" / "tuning_results_partial.json", result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    best = max(
        result["candidates"],
        key=lambda row: (
            row["normalized_average_accuracy_all_tasks"],
            -row["steps_per_task"],
            -row["learning_rate"],
        ),
    )
    result["selected"] = {
        "candidate_id": best["candidate_id"],
        "learning_rate": best["learning_rate"],
        "steps_per_task": best["steps_per_task"],
        "selection_metric": "normalized_average_accuracy_all_tasks",
        "selection_score": best["normalized_average_accuracy_all_tasks"],
        "dev_scores": best["dev_scores"],
        "normalized_gain_by_task": best["normalized_gain_by_task"],
        "selection_rule": "Highest mean baseline-headroom-normalized primary score over all four dev tasks after the homogeneous tuning stream; ties prefer fewer optimizer steps then lower learning rate.",
    }
    result["test_access_before_selection"] = False
    write_json(output_root / "tuning" / "tuning_results.json", result)
    config["training"]["learning_rate"] = best["learning_rate"]
    config["training"]["steps_per_task"] = best["steps_per_task"]
    config["tuning"]["selected"] = result["selected"]
    config["tuning"]["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Selected from dev: lr={best['learning_rate']}, steps/task={best['steps_per_task']}, score={best['normalized_average_accuracy_all_tasks']:.5f}", flush=True)
    print(f"Frozen config: {config_path}", flush=True)


def save_checkpoint(model: Any, tokenizer: Any, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)


def metrics_to_csv(metrics: dict[str, Any], task_ids: list[str], out_path: Path) -> None:
    columns = ["stage", *[f"{task}__primary_score" for task in task_ids]]
    for task in task_ids:
        kind = metrics["task_types"][task]["type"]
        columns.extend([f"{task}__accuracy", f"{task}__macro_f1"] if kind == "classification" else [f"{task}__rouge_l_f1", f"{task}__token_f1", f"{task}__exact_match"])
    columns.extend(["normalized_average_accuracy_all_tasks", "aaa_normalized_seen_tasks", "average_forgetting_normalized", "raw_classification_accuracy_mean", "raw_generation_rouge_l_mean"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stage in metrics["stage_order"]:
            row: dict[str, Any] = {"stage": stage}
            for task in task_ids:
                metric = metrics["stage_scores"][stage][task]
                row[f"{task}__primary_score"] = metric["primary_score"]
                for key in ("accuracy", "macro_f1", "rouge_l_f1", "token_f1", "exact_match"):
                    col = f"{task}__{key}"
                    if col in columns:
                        row[col] = metric.get(key, "")
            row.update(metrics["cl_summary"][stage])
            row.pop("normalized_gain_by_task", None)
            writer.writerow(row)


def run_sequence(
    config: dict[str, Any], sequence_name: str, output_root: Path, run_seed: int | None = None
) -> Path:
    if config["training"].get("learning_rate") is None or config["training"].get("steps_per_task") is None:
        raise ValueError("Training hyperparameters must be frozen by dev tuning before test evaluation")
    if "selected" not in config.get("tuning", {}):
        raise ValueError("Missing frozen dev tuning decision in the full baseline config")
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = resolved_model_path(config)
    run_seed = int(config["seed"] if run_seed is None else run_seed)
    allowed_seeds = [int(value) for value in config.get("training_seeds", [config["seed"]])]
    if run_seed not in allowed_seeds:
        raise ValueError(f"Training seed {run_seed} is not in the frozen seed plan: {allowed_seeds}")
    task_ids = list(config["dataset"]["tasks"])
    sequence = list(config["sequences"][sequence_name])
    if len(sequence) != len(task_ids) or set(sequence) != set(task_ids):
        raise ValueError(f"Sequence {sequence_name} must contain each task once")
    train_records = load_manifest_records(config, "train", task_ids)
    test_records = load_manifest_records(config, "test", task_ids)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = load_tokenizer(model_path, bool(config["model"]["local_files_only"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if run_seed == int(config["seed"]):
        run_root = output_root / sequence_name
        checkpoint_root = Path(config["storage"]["checkpoints"]) / sequence_name
    else:
        run_root = output_root / sequence_name / f"seed_{run_seed}"
        checkpoint_root = Path(config["storage"]["checkpoints"]) / sequence_name / f"seed_{run_seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, bool(config["model"]["local_files_only"]), training=True)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_params != total_params:
        raise RuntimeError(f"Expected full-parameter updates; {trainable_params}/{total_params}")
    started = datetime.now(timezone.utc)
    print(f"Formal sequence {sequence_name}: {sequence}", flush=True)
    baseline = run_evaluation_matrix(
        model, tokenizer, test_records, task_ids, config, device, run_root / "predictions" / "baseline"
    )
    stage_names = ["baseline"]
    stage_scores: dict[str, dict[str, Any]] = {"baseline": baseline}
    training_metrics = {}
    write_json(run_root / "metrics_partial.json", {"sequence": sequence_name, "stage_scores": stage_scores})
    for stage_index, task_id in enumerate(sequence, start=1):
        print(f"\n=== Formal {sequence_name}: stage {stage_index}/{len(sequence)} {task_id} ===", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        training_metrics[task_id] = train_task(
            model,
            tokenizer,
            train_records[task_id],
            config["training"],
            config["model"],
            device,
            float(config["training"]["learning_rate"]),
            int(config["training"]["steps_per_task"]),
            safe_task_seed(run_seed, task_id),
        )
        training_metrics[task_id]["peak_cuda_allocated_bytes"] = max(
            training_metrics[task_id]["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device)
        )
        checkpoint_path = checkpoint_root / f"stage_{stage_index}_{task_id}"
        save_checkpoint(model, tokenizer, checkpoint_path)
        training_metrics[task_id]["checkpoint"] = str(checkpoint_path)
        stage_name = f"after_{stage_index}_{task_id}"
        stage_names.append(stage_name)
        stage_scores[stage_name] = run_evaluation_matrix(
            model, tokenizer, test_records, task_ids, config, device, run_root / "predictions" / stage_name
        )
        partial_summary = score_summary(stage_scores, task_ids, stage_names, sequence)
        partial = {
            "sequence": sequence_name,
            "training_seed": run_seed,
            "task_order": sequence,
            "stage_order": stage_names,
            "stage_scores": stage_scores,
            "continual_learning_summary": partial_summary,
            "training_metrics": training_metrics,
        }
        write_json(run_root / "metrics_partial.json", partial)
        print(f"Saved {stage_name} checkpoint and full test matrix.", flush=True)
    summary = score_summary(stage_scores, task_ids, stage_names, sequence)
    finished = datetime.now(timezone.utc)
    manifest = json.loads((Path(config["dataset"]["processed_dir"]) / "data_manifest.json").read_text(encoding="utf-8"))
    metrics = {
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "sequence_name": sequence_name,
        "training_seed": run_seed,
        "task_order": sequence,
        "task_ids": task_ids,
        "stage_order": stage_names,
        "stage_scores": stage_scores,
        "cl_summary": summary,
        "task_types": config["dataset"]["task_types"],
        "training_metrics": training_metrics,
        "model_path": str(model_path),
        "local_files_only": True,
        "device": torch.cuda.get_device_name(device),
        "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "peak_cuda_allocated_bytes_training": max(row["peak_cuda_allocated_bytes"] for row in training_metrics.values()),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "source_commit": manifest["source"]["commit"],
        "data_manifest": str(Path(config["dataset"]["processed_dir"]) / "data_manifest.json"),
        "config": config,
        "test_used_for_selection_or_tuning": False,
        "metric_definitions": {
            "classification": "Anchored yes/no answer parser; Accuracy and fixed-label Macro-F1.",
            "generation": "ROUGE-L F1 primary; Token-F1 and normalized Exact Match auxiliary.",
            "normalized_gain": "(stage primary score - base primary score) / max(1 - base primary score, 1e-8).",
        },
    }
    write_json(run_root / "metrics.json", metrics)
    metrics_to_csv(metrics, task_ids, run_root / "accuracy_matrix.csv")
    report_lines = [
        f"# Formal continual SFT: {sequence_name}",
        "",
        f"- Tasks: {' -> '.join(sequence)}",
        f"- Training seed: {run_seed}",
        f"- Learning rate: {config['training']['learning_rate']}",
        f"- Optimizer steps per task: {config['training']['steps_per_task']}",
        f"- Full parameters updated: {trainable_params:,}/{total_params:,}",
        f"- Wall time: {metrics['duration_seconds']:.1f} s",
        f"- Peak allocated VRAM: {metrics['peak_cuda_allocated_bytes_training'] / (1024**3):.2f} GiB",
        "- Test split was opened only after the dev tuning decision was frozen.",
        "",
        "## Task-by-stage metrics",
        "",
        "| Stage | Task | Primary metric | Score | Accuracy | Macro-F1 | ROUGE-L | Token-F1 | Exact match |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stage_names:
        for task_id in task_ids:
            metric = stage_scores[stage][task_id]
            report_lines.append(
                f"| {stage} | {task_id} | {metric['primary_metric']} | {metric['primary_score']:.4f} | "
                f"{metric.get('accuracy', '')} | {metric.get('macro_f1', '')} | {metric.get('rouge_l_f1', '')} | "
                f"{metric.get('token_f1', '')} | {metric.get('exact_match', '')} |"
            )
    report_lines.extend(["", "## Continual learning summary", "", "| Stage | Normalized average all tasks | AAA on seen tasks | Average forgetting |", "|---|---:|---:|---:|"])
    for stage in stage_names:
        summary_row = summary[stage]
        report_lines.append(
            f"| {stage} | {summary_row['normalized_average_accuracy_all_tasks']:.4f} | "
            f"{summary_row['aaa_normalized_seen_tasks']:.4f} | {summary_row['average_forgetting_normalized']:.4f} |"
        )
    report_lines.extend(["", "Raw classification and generation metrics are reported separately. See metrics.json for all prediction files, truncation/EOS details, training losses, and checkpoints.", ""])
    (run_root / "run_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Formal sequence complete: {run_root}", flush=True)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validate", "tune", "run"], required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "continual_baseline_full.yaml")
    parser.add_argument("--sequence", choices=["homogeneous", "heterogeneous", "mixed"])
    parser.add_argument("--run-seed", type=int, default=None, help="Training randomness seed; data splits remain fixed")
    parser.add_argument("--reset-tuning", action="store_true", help="Unlock saved hyperparameters and rerun the dev-only search grid")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_project_config(args.config)
    if args.phase == "validate":
        task_ids = list(config["dataset"]["tasks"])
        model_path = resolved_model_path(config)
        tokenizer = load_tokenizer(model_path, bool(config["model"]["local_files_only"]))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        report = validate_data(config, tokenizer, task_ids)
        output = Path(config["storage"]["results"]) / "preflight.json"
        write_json(output, report)
        print(f"Preflight complete (train/dev only; test not read): {output}")
        for task, info in report["tasks"].items():
            print(f"{task}: train={info['train']} dev={info['dev']} labels={info['class_counts_train']}")
        return
    if args.phase == "tune":
        if args.reset_tuning:
            config["training"]["learning_rate"] = None
            config["training"]["steps_per_task"] = None
            config["tuning"].pop("selected", None)
            config["tuning"].pop("frozen_utc", None)
        tune(args.config, config, Path(config["storage"]["results"]))
        return
    if not args.sequence:
        raise ValueError("--sequence is required for --phase run")
    run_sequence(config, args.sequence, Path(config["storage"]["results"]), args.run_seed)


if __name__ == "__main__":
    main()
