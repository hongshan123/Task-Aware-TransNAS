"""Phase 4A dev-only analysis of generation collapse and training sufficiency."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from phase4_common import (
    GENERATION_TASKS,
    TASK_IDS,
    evaluate_dev_task,
    first_token_probabilities,
    human_vram,
    prepare_context,
    safe_task_seed,
    teacher_forced_nll,
    train_task_recorded,
    write_json,
)
from run_formal_baseline import load_model, norm_text, parse_yes_no, save_checkpoint


def resolve_mechanism_stages(config: dict[str, Any], base_model_path: Path) -> list[tuple[str, Path]]:
    root = Path(config["storage"]["checkpoints"]) / "homogeneous"
    rows = [
        ("baseline", base_model_path),
        ("after_2_generation_tasks", root / "stage_2_task1603_smcalflow_sentence_generation"),
        ("after_first_classification_task", root / "stage_3_task1501_dstc3_answer_generation"),
        ("after_second_classification_task", root / "stage_4_task1384_deal_or_no_dialog_classification"),
    ]
    for name, path in rows[1:]:
        if not path.is_dir():
            raise FileNotFoundError(f"Required Phase 3 checkpoint is missing: {name}: {path}")
    return rows


def summarize_generation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [str(row.get("prediction", "")) for row in rows]
    normalized = [norm_text(prediction) for prediction in predictions]
    token_lengths = [int(row.get("generated_token_count", 0)) for row in rows]
    parsed = [parse_yes_no(prediction) for prediction in predictions]
    return {
        "mean_generated_tokens": statistics.mean(token_lengths) if token_lengths else 0.0,
        "median_generated_tokens": statistics.median(token_lengths) if token_lengths else 0.0,
        "zero_length_share": sum(length == 0 for length in token_lengths) / max(1, len(token_lengths)),
        "exact_yes_share": sum(text == "yes" for text in normalized) / max(1, len(normalized)),
        "exact_no_share": sum(text == "no" for text in normalized) / max(1, len(normalized)),
        "exact_yes_or_no_share": sum(text in {"yes", "no"} for text in normalized) / max(1, len(normalized)),
        "parsed_yes_no_share": sum(value is not None for value in parsed) / max(1, len(parsed)),
        "eos_termination_rate": sum(bool(row.get("eos_terminated")) for row in rows) / max(1, len(rows)),
    }


def run_mechanism() -> dict[str, Any]:
    config, base_path, tokenizer, device, _, dev = prepare_context()
    output_root = Path(config["storage"]["root"]) / "results" / "architecture_pilot" / "generation_collapse"
    stages = resolve_mechanism_stages(config, base_path)
    stage_results: dict[str, Any] = {}
    for stage_name, checkpoint_path in stages:
        print(f"\n=== Mechanism stage: {stage_name} ===", flush=True)
        model = load_model(checkpoint_path, local_only=True, training=False)
        torch.cuda.reset_peak_memory_stats(device)
        stage_tasks: dict[str, Any] = {}
        for task_id in GENERATION_TASKS:
            print(f"Assessing {task_id} on dev only", flush=True)
            records = dev[task_id]
            nll = teacher_forced_nll(model, tokenizer, records, config["model"], device)
            first_tokens = first_token_probabilities(model, tokenizer, records, config["model"], device)
            prediction_path = output_root / stage_name / "dev" / f"{task_id}.jsonl"
            metric, rows = evaluate_dev_task(
                model,
                tokenizer,
                records,
                task_id,
                config["dataset"]["task_types"][task_id],
                config["model"],
                config["evaluation"],
                device,
                prediction_path,
            )
            stage_tasks[task_id] = {
                "teacher_forced": nll,
                "first_token_probabilities": first_tokens,
                "generation_metrics": metric,
                "generation_behavior": summarize_generation_rows(rows),
                "prediction_file": str(prediction_path),
            }
        stage_results[stage_name] = {
            "checkpoint": str(checkpoint_path),
            "tasks": stage_tasks,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        write_json(
            output_root / "mechanism_metrics_partial.json",
            {"split": "dev", "local_files_only": True, "stages": stage_results},
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    final = {
        "study": "generation collapse mechanisms",
        "split": "dev only; no test records loaded by this script",
        "local_files_only": True,
        "task_ids": GENERATION_TASKS,
        "stages": stage_results,
        "result_root": str(output_root),
    }
    write_json(output_root / "mechanism_metrics.json", final)
    return final


def run_sufficiency() -> dict[str, Any]:
    config, base_path, tokenizer, device, train, dev = prepare_context()
    output_root = Path(config["storage"]["root"]) / "results" / "architecture_pilot" / "training_sufficiency"
    schedule = (50, 125, 250)
    results: dict[str, Any] = {}
    for task_id in GENERATION_TASKS:
        print(f"\n=== Training sufficiency: {task_id} ===", flush=True)
        task_root = output_root / task_id
        model = load_model(base_path, local_only=True, training=True)
        torch.cuda.reset_peak_memory_stats(device)
        checkpoints: dict[str, Any] = {}

        def evaluate_checkpoint(step: int, training_snapshot: dict[str, Any]) -> None:
            print(f"\nDev evaluation at step {step}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            nll = teacher_forced_nll(model, tokenizer, dev[task_id], config["model"], device)
            metric, rows = evaluate_dev_task(
                model,
                tokenizer,
                dev[task_id],
                task_id,
                config["dataset"]["task_types"][task_id],
                config["model"],
                config["evaluation"],
                device,
                task_root / "predictions" / f"step_{step}" / "dev.jsonl",
            )
            snapshot = {
                "optimizer_step": step,
                "training": training_snapshot,
                "validation_token_nll": nll,
                "validation_generation_metrics": metric,
                "validation_generation_behavior": summarize_generation_rows(rows),
                "evaluation_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "evaluation_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "prediction_file": metric["prediction_file"],
            }
            checkpoints[str(step)] = snapshot
            write_json(
                task_root / "sufficiency_partial.json",
                {"task_id": task_id, "schedule": schedule, "checkpoints": checkpoints},
            )
            model.train()

        seed = safe_task_seed(int(config["seed"]), task_id)
        training = train_task_recorded(
            model,
            tokenizer,
            train[task_id],
            config["training"],
            config["model"],
            device,
            float(config["training"]["learning_rate"]),
            max(schedule),
            seed,
            on_checkpoint=evaluate_checkpoint,
            checkpoint_steps=set(schedule),
        )
        checkpoint_path = task_root / "checkpoint_step_250"
        save_checkpoint(model, tokenizer, checkpoint_path)
        task_result = {
            "task_id": task_id,
            "training_seed": seed,
            "training": training,
            "checkpoints": checkpoints,
            "checkpoint_step_250": str(checkpoint_path),
            "training_peak_cuda_allocated_bytes": training["peak_cuda_allocated_bytes"],
            "training_peak_cuda_reserved_bytes": training["peak_cuda_reserved_bytes"],
        }
        results[task_id] = task_result
        write_json(
            output_root / "sufficiency_metrics_partial.json",
            {"schedule": schedule, "tasks": results, "local_files_only": True, "split": "dev"},
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    final = {
        "study": "training sufficiency",
        "schedule_optimizer_steps": list(schedule),
        "optimizer_state_persisted_within_each_task_run": True,
        "initial_model": str(base_path),
        "local_files_only": True,
        "selection_split": "dev only",
        "test_split_loaded": False,
        "tasks": results,
        "result_root": str(output_root),
    }
    write_json(output_root / "sufficiency_metrics.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["mechanism", "sufficiency", "all"])
    args = parser.parse_args()
    started = time.perf_counter()
    if args.phase in {"mechanism", "all"}:
        run_mechanism()
    if args.phase in {"sufficiency", "all"}:
        run_sufficiency()
    print(f"Phase 4A diagnostics complete in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
