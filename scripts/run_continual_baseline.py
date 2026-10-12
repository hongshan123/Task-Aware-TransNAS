"""Full-parameter, sequential instruction tuning on the local Qwen model and CITB tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .runtime_paths import configure_cache_environment, resolve_config_paths, resolve_runtime_path, storage_root
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import configure_cache_environment, resolve_config_paths, resolve_runtime_path, storage_root

load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_STORAGE_ROOT = storage_root(PROJECT_ROOT)
configure_cache_environment(PROJECT_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalise_answer(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,!?;:'\"`“”‘’")


def messages_for(record: dict[str, Any]) -> list[dict[str, str]]:
    system = "You are an instruction-following assistant. Follow the task definition and return only the requested answer."
    user = "Task definition:\n" + record["definition"] + "\n\nInput:\n" + record["input"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def apply_template(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False, **kwargs
    )
    if isinstance(tokens, dict) or hasattr(tokens, "keys"):
        tokens = tokens["input_ids"]
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def encode_training_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prompt_messages = messages_for(record)
    prompt = apply_template(tokenizer, prompt_messages, add_generation_prompt=True)
    target = record["targets"][0] if record["targets"] else ""
    if not target:
        raise ValueError(f"Empty target for {record['instance_id']}")
    complete = apply_template(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": target}],
        add_generation_prompt=False,
    )
    if complete[: len(prompt)] != prompt:
        raise ValueError(f"Chat-template prompt is not a prefix of the full sample: {record['instance_id']}")
    answer_tokens = complete[len(prompt) :]
    if not answer_tokens:
        raise ValueError(f"No assistant tokens for {record['instance_id']}")
    if len(answer_tokens) >= max_length:
        answer_tokens = answer_tokens[: max_length - 1] + [tokenizer.eos_token_id]
        prompt = []
    else:
        prompt = prompt[-(max_length - len(answer_tokens)) :]
    input_ids = prompt + answer_tokens
    labels = [-100] * len(prompt) + answer_tokens.copy()
    return {"input_ids": input_ids, "labels": labels}


def make_batch(tokenizer: Any, records: list[dict[str, Any]], max_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = [encode_training_record(tokenizer, record, max_length) for record in records]
    width = max(len(row["input_ids"]) for row in encoded)
    pad = tokenizer.pad_token_id
    input_ids, attention_mask, labels = [], [], []
    for row in encoded:
        n_pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad] * n_pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * n_pad)
        labels.append(row["labels"] + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).casefold()
    )


def train_task(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    training_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    steps = int(training_cfg["steps_per_task"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    max_length = int(model_cfg["max_sequence_length"])
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        foreach=False,
    )
    rng = random.Random(seed)
    order: list[int] = []
    cursor = 0

    def next_record() -> dict[str, Any]:
        nonlocal order, cursor
        if cursor >= len(order):
            order = list(range(len(records)))
            rng.shuffle(order)
            cursor = 0
        record = records[order[cursor]]
        cursor += 1
        return record

    losses: list[float] = []
    retries: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        step_records = [next_record() for _ in range(grad_accum)]
        retry = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                loss_sum = 0.0
                for record in step_records:
                    batch = make_batch(tokenizer, [record], max_length, device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
                    (loss / grad_accum).backward()
                    loss_sum += float(loss.detach().float().item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss_sum / grad_accum)
                break
            except BaseException as error:
                optimizer.zero_grad(set_to_none=True)
                if not is_cuda_oom(error):
                    raise
                torch.cuda.empty_cache()
                retry += 1
                next_length = {512: 384, 384: 256, 256: 192, 192: 128}.get(max_length, max_length // 2)
                if next_length >= max_length or next_length < 64 or retry > 4:
                    raise RuntimeError("CUDA OOM persisted after automatic sequence-length reductions") from error
                retries.append({"step": step, "from_max_length": max_length, "to_max_length": next_length, "reason": str(error)[:400]})
                max_length = next_length
                print(f"OOM recovery: retry step {step} with max_sequence_length={max_length}", flush=True)
        if step == 1 or step % 5 == 0 or step == steps:
            print(f"train step {step}/{steps}: loss={losses[-1]:.4f}, max_length={max_length}", flush=True)
    model.config.use_cache = True
    return {
        "optimizer_steps": steps,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": float(training_cfg["learning_rate"]),
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "effective_max_sequence_length": max_length,
        "oom_retries": retries,
        "duration_seconds": time.perf_counter() - start,
    }


def eval_task(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    device: torch.device,
    prediction_file: Path,
) -> dict[str, Any]:
    limit = min(int(eval_cfg["max_examples_per_task"]), len(records))
    eval_records = records[:limit]
    max_length = int(model_cfg["max_sequence_length"])
    max_new = int(model_cfg["max_new_tokens"])
    predictions: list[dict[str, Any]] = []
    oom_retries: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    for record in eval_records:
        prompt_messages = messages_for(record)
        prompt = apply_template(tokenizer, prompt_messages, add_generation_prompt=True)
        prompt = prompt[-max_length:]
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        attention = torch.ones_like(input_ids)
        retry = 0
        while True:
            try:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention,
                        do_sample=False,
                        max_new_tokens=max_new,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                answer = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
                break
            except BaseException as error:
                if not is_cuda_oom(error):
                    raise
                torch.cuda.empty_cache()
                retry += 1
                old = {"max_input_length": max_length, "max_new_tokens": max_new}
                if max_length > 192:
                    max_length = max(192, max_length - 128)
                elif max_new > 16:
                    max_new = max(16, max_new // 2)
                else:
                    raise RuntimeError("CUDA OOM persisted after evaluation limits were reduced") from error
                prompt = apply_template(tokenizer, prompt_messages, add_generation_prompt=True)[-max_length:]
                input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
                attention = torch.ones_like(input_ids)
                oom_retries.append({"instance_id": record["instance_id"], **old, "new_max_input_length": max_length, "new_max_new_tokens": max_new})
                print(f"Evaluation OOM recovery for {record['instance_id']}: input={max_length}, new_tokens={max_new}", flush=True)
                if retry > 5:
                    raise RuntimeError("Too many evaluation OOM retries") from error
        expected = record["targets"]
        normalized_prediction = normalise_answer(answer)
        correct = any(normalized_prediction == normalise_answer(target) for target in expected)
        predictions.append(
            {
                "task_id": record["task_id"],
                "instance_id": record["instance_id"],
                "prediction": answer,
                "references": expected,
                "correct": correct,
            }
        )
        if len(predictions) % 10 == 0 or len(predictions) == limit:
            print(f"eval {record['task_id']}: {len(predictions)}/{limit}", flush=True)
    model.config.use_cache = True
    if was_training:
        model.train()
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with prediction_file.open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "task_id": eval_records[0]["task_id"] if eval_records else None,
        "correct": sum(row["correct"] for row in predictions),
        "evaluated": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / len(predictions) if predictions else None,
        "prediction_file": str(prediction_file),
        "oom_retries": oom_retries,
    }


def save_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def write_summary_artifacts(run: dict[str, Any], result_root: Path, task_infos: list[dict[str, Any]]) -> None:
    task_ids = run["task_ids"]
    matrix = run["accuracy_matrix"]
    csv_path = result_root / "accuracy_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage", *task_ids, "aaa", "current_task_accuracy", "average_forgetting"]
    stage_to_task: dict[str, str] = {}
    after_stages = [stage for stage in matrix if stage.startswith("after_")]
    for index, stage in enumerate(after_stages):
        stage_to_task[stage] = task_ids[index]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, scores in matrix.items():
            row: dict[str, Any] = {"stage": stage, **scores}
            task_id = stage_to_task.get(stage)
            if task_id:
                task_metrics = run["task_training"][task_id]
                row["aaa"] = task_metrics.get("aaa", task_metrics.get("average_accuracy_learned_tasks"))
                row["current_task_accuracy"] = task_metrics.get("current_task_accuracy", scores.get(task_id))
                row["average_forgetting"] = task_metrics.get("average_forgetting")
            writer.writerow(row)

    source_commit = run.get("source_commit", "unknown")
    task_order = run.get("task_order", [])
    lines = [
        "# Baseline pilot run report",
        "",
        f"- Run mode: {run['mode']}",
        f"- Status: completed at {run.get('finished_utc', 'unknown')}",
        f"- Total wall time: {run.get('duration_seconds', 0.0):.1f} seconds",
        f"- Model: {run['model_path']} (local_files_only=True)",
        f"- GPU: {run['device']} ({run['total_vram_bytes'] / (1024 ** 3):.1f} GiB)",
        f"- Parameters trainable: {run['trainable_parameters']:,} / {run['total_parameters']:,} ({run['trainable_fraction']:.1%})",
        f"- Seed: {run.get('seed', 'unknown')}",
        f"- Source: CITB commit {source_commit}",
        f"- Total test predictions: {run.get('total_eval_predictions', 0)}",
        f"- Peak CUDA allocated during training: {run.get('peak_cuda_allocated_bytes_training', run.get('peak_cuda_allocated_bytes', 0)) / (1024 ** 3):.2f} GiB",
        f"- OOM retries: {run.get('oom_retry_count', 0)}",
        "",
        "## Task order and counts",
        "",
        "| Step | Task | Category | Train / dev / test | Train seconds | Current task accuracy | AAA | Forgetting |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, info in enumerate(task_order, start=1):
        task_id = info["task_id"]
        train_metrics = run["task_training"].get(task_id, {})
        counts = info.get("counts", {})
        lines.append(
            f"| {index} | {task_id} | {', '.join(info.get('category', []))} | "
            f"{counts.get('train', 0)} / {counts.get('dev', 0)} / {counts.get('test', 0)} | "
            f"{train_metrics.get('duration_seconds', 0.0):.1f} | "
            f"{train_metrics.get('current_task_accuracy', '')} | "
            f"{train_metrics.get('aaa', '')} | {train_metrics.get('average_forgetting', '')} |"
        )
    lines.extend(
        [
            "",
            "## Accuracy matrix",
            "",
            "Values are normalized exact-match accuracy. A blank means the task had not yet been learned at that stage.",
            "",
            "| Stage | " + " | ".join(task_ids) + " | AAA | Forgetting |",
            "|---|" + "---:|" * (len(task_ids) + 2),
        ]
    )
    for stage, scores in matrix.items():
        task_id = stage_to_task.get(stage)
        task_metrics = run["task_training"].get(task_id, {}) if task_id else {}
        values = [str(scores.get(name, "")) for name in task_ids]
        aaa = task_metrics.get("aaa", task_metrics.get("average_accuracy_learned_tasks", ""))
        forgetting = task_metrics.get("average_forgetting", "")
        lines.append(f"| {stage} | " + " | ".join(values) + f" | {aaa} | {forgetting} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This pilot is limited to 20 optimizer updates per task and is for pipeline validation, not a full benchmark. Exact match is strict for open-ended generation. See metrics.json for losses, predictions, checkpoints, full configuration, and timings.",
            "",
        ]
    )
    (result_root / "run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot"], required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "continual_baseline.yaml")
    parser.add_argument("--steps-per-task", type=int, default=None)
    args = parser.parse_args()
    run_started_at = datetime.now(timezone.utc)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    config_path = resolve_runtime_path(args.config, project_root=PROJECT_ROOT)
    config = resolve_config_paths(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), project_root=PROJECT_ROOT
    )
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = dict(config["training"])
    eval_cfg = config["evaluation"]
    if args.steps_per_task is not None:
        training_cfg["steps_per_task"] = args.steps_per_task
    if training_cfg["steps_per_task"] <= 0:
        raise ValueError("steps-per-task must be positive")
    if int(training_cfg["micro_batch_size"]) != 1:
        raise ValueError("This memory-bounded runner currently requires micro_batch_size=1")

    model_value = os.environ.get(model_cfg["env_var"])
    if not model_value:
        raise EnvironmentError(f"Set {model_cfg['env_var']} in the project .env file")
    model_path = resolve_runtime_path(model_value, project_root=PROJECT_ROOT)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")
    data_root = resolve_runtime_path(dataset_cfg["processed_dir"], project_root=PROJECT_ROOT)
    run_name = "baseline_pilot" if args.mode == "pilot" else "smoke"
    result_root = resolve_runtime_path(config["storage"]["results"], project_root=PROJECT_ROOT) / run_name
    checkpoint_root = resolve_runtime_path(config["storage"]["checkpoints"], project_root=PROJECT_ROOT) / run_name
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["leakage_check"]["passed"] is not True:
        raise RuntimeError("Dataset leakage check is not marked as passed")

    task_infos = manifest["tasks"]
    if args.mode == "smoke":
        task_infos = task_infos[:1]
    train_by_task = {
        item["task_id"]: read_jsonl(resolve_runtime_path(item["split_files"]["train"], project_root=PROJECT_ROOT))
        for item in task_infos
    }
    test_by_task = {
        item["task_id"]: read_jsonl(resolve_runtime_path(item["split_files"]["test"], project_root=PROJECT_ROOT))
        for item in task_infos
    }
    if any(not train_by_task[item["task_id"]] or not test_by_task[item["task_id"]] for item in task_infos):
        raise RuntimeError("At least one configured task has an empty train or test split")

    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_params != total_params:
        raise RuntimeError(f"Expected full-parameter tuning; trainable={trainable_params}, total={total_params}")
    if training_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    run: dict[str, Any] = {
        "started_utc": run_started_at.isoformat(),
        "mode": args.mode,
        "seed": int(config["seed"]),
        "model_path": str(model_path),
        "local_files_only": True,
        "device": torch.cuda.get_device_name(device),
        "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_fraction": trainable_params / total_params,
        "data_manifest": str(manifest_path),
        "training_config": training_cfg,
        "model_config": model_cfg,
        "evaluation_config": eval_cfg,
        "optimizer_policy": "fresh AdamW optimizer per task; model weights continue sequentially",
        "task_ids": [item["task_id"] for item in task_infos],
        "source_commit": manifest["source"]["commit"],
        "task_order": [
            {
                "task_id": item["task_id"],
                "stream_position_1based": item["stream_position_1based"],
                "category": item["category"],
                "counts": item["counts"],
            }
            for item in task_infos
        ],
        "accuracy_matrix": {},
        "task_training": {},
        "evaluations": {},
    }
    print(f"Loaded local model: {model_path}", flush=True)
    print(f"Parameters trainable: {trainable_params:,}/{total_params:,}", flush=True)
    print(f"GPU: {run['device']} ({run['total_vram_bytes'] / (1024**3):.1f} GiB)", flush=True)

    baseline: dict[str, float] = {}
    for item in task_infos:
        task_id = item["task_id"]
        path = result_root / "predictions" / "baseline" / f"{task_id}.jsonl"
        result = eval_task(model, tokenizer, test_by_task[task_id], eval_cfg, model_cfg, device, path)
        baseline[task_id] = result["accuracy"]
        run["evaluations"].setdefault("baseline", {})[task_id] = result
    run["accuracy_matrix"]["baseline"] = baseline
    save_checkpoint(model, tokenizer, checkpoint_root / "initial_base_model")

    for index, item in enumerate(task_infos, start=1):
        task_id = item["task_id"]
        print(f"\n=== Sequential task {index}/{len(task_infos)}: {task_id} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        train_metrics = train_task(
            model,
            tokenizer,
            train_by_task[task_id],
            training_cfg,
            model_cfg,
            device,
            int(config["seed"]) + index,
        )
        train_metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        train_metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        run["task_training"][task_id] = train_metrics
        checkpoint_path = checkpoint_root / f"stage_{index}_{task_id}"
        save_checkpoint(model, tokenizer, checkpoint_path)
        row: dict[str, float] = {}
        for learned in task_infos[:index]:
            learned_id = learned["task_id"]
            prediction_path = result_root / "predictions" / f"after_{index}_{task_id}" / f"{learned_id}.jsonl"
            result = eval_task(model, tokenizer, test_by_task[learned_id], eval_cfg, model_cfg, device, prediction_path)
            row[learned_id] = result["accuracy"]
            run["evaluations"].setdefault(f"after_{index}_{task_id}", {})[learned_id] = result
        run["accuracy_matrix"][f"after_{index}_{task_id}"] = row
        run["task_training"][task_id]["checkpoint"] = str(checkpoint_path)
        learned_accuracies = list(row.values())
        run["task_training"][task_id]["average_accuracy_learned_tasks"] = sum(learned_accuracies) / len(learned_accuracies)
        run["task_training"][task_id]["aaa"] = run["task_training"][task_id]["average_accuracy_learned_tasks"]
        run["task_training"][task_id]["current_task_accuracy"] = row[task_id]
        if index > 1:
            previous_ids = [prior["task_id"] for prior in task_infos[: index - 1]]
            forget = []
            for prior_id in previous_ids:
                prior_scores = [
                    run["accuracy_matrix"][key][prior_id]
                    for key in run["accuracy_matrix"]
                    if key.startswith("after_") and prior_id in run["accuracy_matrix"][key]
                ]
                forget.append(max(prior_scores) - row[prior_id])
            run["task_training"][task_id]["average_forgetting"] = sum(forget) / len(forget)
        else:
            run["task_training"][task_id]["average_forgetting"] = 0.0
        write_json(result_root / "metrics.json", run)

    run["finished_utc"] = datetime.now(timezone.utc).isoformat()
    started_at = datetime.fromisoformat(run["started_utc"])
    finished_at = datetime.fromisoformat(run["finished_utc"])
    run["duration_seconds"] = (finished_at - started_at).total_seconds()
    run["total_eval_predictions"] = sum(
        metric["evaluated"] for stage in run["evaluations"].values() for metric in stage.values()
    )
    run["oom_retry_count"] = sum(
        len(metrics["oom_retries"]) for metrics in run["task_training"].values()
    ) + sum(
        len(metric["oom_retries"]) for stage in run["evaluations"].values() for metric in stage.values()
    )
    run["peak_cuda_allocated_bytes"] = max(
        [metrics["peak_cuda_allocated_bytes"] for metrics in run["task_training"].values()]
        or [torch.cuda.max_memory_allocated()]
    )
    run["peak_cuda_allocated_bytes_training"] = run["peak_cuda_allocated_bytes"]
    run["peak_cuda_reserved_bytes_training"] = max(
        [metrics.get("peak_cuda_reserved_bytes", 0) for metrics in run["task_training"].values()]
        or [torch.cuda.max_memory_reserved()]
    )
    run["free_vram_bytes_at_finish"] = torch.cuda.mem_get_info()[0]
    write_json(result_root / "metrics.json", run)
    write_summary_artifacts(run, result_root, task_infos)
    print(f"\nRun complete: {result_root / 'metrics.json'}", flush=True)
    print(f"Peak CUDA allocated: {run['peak_cuda_allocated_bytes'] / (1024**3):.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
