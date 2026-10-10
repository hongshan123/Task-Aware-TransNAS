"""Run all 24 task orders with fixed-seed development-only evaluation."""

from __future__ import annotations

import csv
import gc
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_formal_baseline import (  # noqa: E402
    evaluate_task,
    load_manifest_records,
    load_model,
    load_tokenizer,
    read_jsonl,
    resolved_model_path,
    safe_task_seed,
    save_checkpoint,
    score_rows,
    train_task,
    write_json,
)
from task_aware_transnas.cl.metrics import continual_learning_metrics  # noqa: E402


SEED = 10
CSV_COLUMNS = [
    "order_id",
    "task_order",
    "task_type_order",
    "type_switch_count",
    "final_task_id",
    "final_task_type",
    "mean_new_task_learning_normalized_gain",
    "final_task_learning_normalized_gain",
    "final_mean_normalized_gain_all_tasks",
    "final_cross_stage_aaa_normalized",
    "final_nonnegative_forgetting_normalized",
    "final_backward_transfer_normalized",
    "training_seed",
    "learning_rate",
    "steps_per_task",
    "max_sequence_length",
    "max_target_tokens",
    "max_new_tokens",
    "final_checkpoint_path",
    "final_task565_circa_answer_generation__raw_rouge_l_f1",
    "final_task1603_smcalflow_sentence_generation__raw_rouge_l_f1",
    "final_task1501_dstc3_answer_generation__raw_accuracy",
    "final_task1384_deal_or_no_dialog_classification__raw_accuracy",
    "final_task565_circa_answer_generation__normalized_gain",
    "final_task1603_smcalflow_sentence_generation__normalized_gain",
    "final_task1501_dstc3_answer_generation__normalized_gain",
    "final_task1384_deal_or_no_dialog_classification__normalized_gain",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    write_json(temp_path, value)
    temp_path.replace(path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def count_type_switches(order: tuple[str, ...], type_by_task: dict[str, str]) -> int:
    return sum(type_by_task[left] != type_by_task[right] for left, right in zip(order, order[1:]))


def make_csv_row(
    order_id: str,
    result: dict[str, Any],
    order: list[str],
    type_by_task: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    final_stage = result["stage_order"][-1]
    audit = result["cl_metric_audit"]
    final_metrics = audit["stage_metrics"][final_stage]
    acquisition_gains = [
        audit["normalized_gain_matrix"][result["stage_order"][index + 1]][task]
        for index, task in enumerate(order)
    ]
    final_task_gain = audit["normalized_gain_matrix"][final_stage][order[-1]]
    row = {
        "order_id": order_id,
        "task_order": " -> ".join(order),
        "task_type_order": " -> ".join(type_by_task[task] for task in order),
        "type_switch_count": count_type_switches(tuple(order), type_by_task),
        "final_task_id": order[-1],
        "final_task_type": type_by_task[order[-1]],
        "mean_new_task_learning_normalized_gain": sum(acquisition_gains) / len(acquisition_gains),
        "final_task_learning_normalized_gain": final_task_gain,
        "final_mean_normalized_gain_all_tasks": final_metrics.get(
            "mean_normalized_gain_all_tasks", final_metrics["normalized_average_accuracy_all_tasks"]
        ),
        "final_cross_stage_aaa_normalized": final_metrics["cross_stage_aaa_normalized"],
        "final_nonnegative_forgetting_normalized": final_metrics["average_forgetting_nonnegative_normalized"],
        "final_backward_transfer_normalized": final_metrics["backward_transfer_normalized"],
        "training_seed": SEED,
        "learning_rate": config["training"]["learning_rate"],
        "steps_per_task": config["training"]["steps_per_task"],
        "max_sequence_length": config["model"]["max_sequence_length"],
        "max_target_tokens": config["model"]["max_target_tokens"],
        "max_new_tokens": config["evaluation"]["generation_max_new_tokens"],
        "final_checkpoint_path": result.get("review_checkpoint_path") or "",
    }
    for task_id, metric in result["cl_metric_audit"]["primary_metric_by_task"].items():
        safe_metric = "rouge_l_f1" if metric == "rouge_l_f1" else "accuracy"
        row[f"final_{task_id}__raw_{safe_metric}"] = audit["raw_task_primary_score_matrix"][final_stage][task_id]
        row[f"final_{task_id}__normalized_gain"] = audit["normalized_gain_matrix"][final_stage][task_id]
    return row


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config_path = PROJECT_ROOT / "configs" / "continual_baseline_full.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["model"].get("local_files_only") is not True:
        raise RuntimeError("Order sensitivity requires local_files_only=True")
    if "selected" not in config.get("tuning", {}):
        raise RuntimeError("The development-tuned hyperparameters must be frozen first")

    task_ids = list(config["dataset"]["tasks"])
    task_types = config["dataset"]["task_types"]
    type_by_task = {task_id: task_types[task_id]["type"] for task_id in task_ids}
    metric_by_task = {task_id: task_types[task_id]["primary_metric"] for task_id in task_ids}
    permutations = list(itertools.permutations(task_ids))
    if len(permutations) != 24 or len(set(permutations)) != 24:
        raise RuntimeError("Expected all 24 unique permutations of four tasks")
    order_rows = [
        {"order_id": f"O{index:02d}", "order": order, "switches": count_type_switches(order, type_by_task)}
        for index, order in enumerate(permutations, start=1)
    ]
    switch_counts: dict[int, int] = {}
    for row in order_rows:
        switch_counts[row["switches"]] = switch_counts.get(row["switches"], 0) + 1
    if switch_counts != {1: 8, 2: 8, 3: 8}:
        raise RuntimeError(f"Unexpected type-switch group sizes: {switch_counts}")

    model_path = resolved_model_path(config)
    result_root = Path(config["storage"]["results"]).parent / "order_sensitivity"
    checkpoint_root = Path(config["storage"]["checkpoints"]).parent / "order_sensitivity" / "review"
    result_root.mkdir(parents=True, exist_ok=True)

    # This baseline comes from the already generated local base-model dev evaluation;
    # every permutation shares it, so the baseline forward pass need not be repeated.
    baseline_prediction_root = Path(config["storage"]["results"]) / "generation_diagnostics" / "predictions" / "baseline" / "dev"
    baseline = {}
    for task_id in task_ids:
        baseline_rows = read_jsonl(baseline_prediction_root / f"{task_id}.jsonl")
        baseline[task_id] = score_rows(task_id, baseline_rows, task_types[task_id])
    baseline_scores = {task_id: float(baseline[task_id]["primary_score"]) for task_id in task_ids}

    train_records = load_manifest_records(config, "train", task_ids)
    dev_records = load_manifest_records(config, "dev", task_ids)
    train_ids = {task_id: [row["instance_id"] for row in train_records[task_id]] for task_id in task_ids}
    dev_ids = {task_id: [row["instance_id"] for row in dev_records[task_id]] for task_id in task_ids}
    expected_train_count = int(config["dataset"]["train_per_task"])
    expected_dev_count = int(config["dataset"]["dev_per_task"])
    for task_id in task_ids:
        if len(train_ids[task_id]) != expected_train_count or len(set(train_ids[task_id])) != expected_train_count:
            raise RuntimeError(f"Train sample IDs are missing or duplicated for {task_id}")
        if len(dev_ids[task_id]) != expected_dev_count or len(set(dev_ids[task_id])) != expected_dev_count:
            raise RuntimeError(f"Dev sample IDs are missing or duplicated for {task_id}")
        baseline_rows = read_jsonl(baseline_prediction_root / f"{task_id}.jsonl")
        if {row["instance_id"] for row in baseline_rows} != set(dev_ids[task_id]) or len(baseline_rows) != expected_dev_count:
            raise RuntimeError(f"Shared baseline prediction IDs do not match dev for {task_id}")

    # Persist enough metadata to check the fixed-seed, shared-ID design without opening test data.
    experiment_config = {
        "experiment": "all_24_task_orders_development_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": SEED,
        "task_ids": task_ids,
        "task_types": type_by_task,
        "primary_metric_by_task": metric_by_task,
        "permutations": [
            {"order_id": row["order_id"], "task_order": list(row["order"]), "type_switch_count": row["switches"]}
            for row in order_rows
        ],
        "type_switch_group_counts": switch_counts,
        "sample_counts": {"train_per_task": expected_train_count, "dev_per_task": expected_dev_count},
        "train_instance_ids_by_task": train_ids,
        "dev_instance_ids_by_task": dev_ids,
        "model_path": str(model_path),
        "local_files_only": True,
        "hyperparameters": {
            "learning_rate": config["training"]["learning_rate"],
            "steps_per_task": config["training"]["steps_per_task"],
            "micro_batch_size": config["training"]["micro_batch_size"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "weight_decay": config["training"]["weight_decay"],
            "max_grad_norm": config["training"]["max_grad_norm"],
            "gradient_checkpointing": config["training"]["gradient_checkpointing"],
            "max_sequence_length": config["model"]["max_sequence_length"],
            "max_target_tokens": config["model"]["max_target_tokens"],
            "generation_max_new_tokens": config["evaluation"]["generation_max_new_tokens"],
            "prompt_budget": config["model"]["max_sequence_length"],
        },
        "per_task_training_seed_rule": "safe_task_seed(10, task_id); every task receives the same sample order and torch seed across all permutations.",
        "evaluation_split": "dev only",
        "test_data_loaded": False,
        "checkpoint_policy": "Only final checkpoints from one 1-switch and one 3-switch order are retained.",
        "shared_baseline_dev_scores": baseline_scores,
    }
    write_json_atomic(result_root / "experiment_config.json", experiment_config)

    # Keep two final review checkpoints: one low-switch and one high-switch sequence.
    low_order_id = next(row["order_id"] for row in order_rows if row["switches"] == 1)
    high_order_id = next(row["order_id"] for row in order_rows if row["switches"] == 3)
    review_order_ids = {low_order_id, high_order_id}

    partial_path = result_root / "order_metrics_partial.json"
    existing: dict[str, Any] = read_json(partial_path) if partial_path.exists() else {
        "experiment_config": experiment_config,
        "completed_orders": {},
        "active_order": None,
    }
    completed: dict[str, Any] = existing.get("completed_orders", {})
    csv_path = result_root / "all_24_orders.csv"
    order_lookup = {row["order_id"]: row for row in order_rows}
    initial_rows = [
        make_csv_row(order_id, result, list(order_lookup[order_id]["order"]), type_by_task, config)
        for order_id, result in completed.items()
    ]
    initial_rows.sort(key=lambda row: row["order_id"])
    if initial_rows:
        write_csv_atomic(csv_path, initial_rows)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing a CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    tokenizer = load_tokenizer(model_path, local_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for order_info in order_rows:
        order_id = order_info["order_id"]
        order = list(order_info["order"])
        if order_id in completed:
            print(f"Skipping completed {order_id}: {order}", flush=True)
            continue
        print(
            f"\n=== {order_id}/O24 order={order}; types={[type_by_task[task] for task in order]}; "
            f"switches={order_info['switches']} ===",
            flush=True,
        )
        started = time.perf_counter()
        model = load_model(model_path, local_only=True, training=True)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if total_parameters != trainable_parameters:
            raise RuntimeError(f"Expected full-parameter tuning, got {trainable_parameters}/{total_parameters}")

        stage_order = ["baseline"]
        score_matrix = {"baseline": dict(baseline_scores)}
        stage_metrics = {"baseline": baseline}
        training_metrics: dict[str, Any] = {}
        acquisition_stages: dict[str, str] = {}
        order_result: dict[str, Any] = {
            "order_id": order_id,
            "task_order": order,
            "task_type_order": [type_by_task[task] for task in order],
            "type_switch_count": order_info["switches"],
            "training_seed": SEED,
            "seed_by_task": {task: safe_task_seed(SEED, task) for task in order},
            "stage_order": stage_order,
            "stage_scores": stage_metrics,
            "raw_primary_score_matrix": score_matrix,
            "training_metrics_by_task": training_metrics,
            "acquisition_stage_by_task": acquisition_stages,
            "device": torch.cuda.get_device_name(device),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "review_checkpoint_path": None,
        }
        existing["active_order"] = order_result
        write_json_atomic(partial_path, existing)

        for stage_index, task_id in enumerate(order, start=1):
            print(f"\nTraining {order_id} task {stage_index}/4: {task_id}", flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            train_metric = train_task(
                model,
                tokenizer,
                train_records[task_id],
                config["training"],
                config["model"],
                device,
                float(config["training"]["learning_rate"]),
                int(config["training"]["steps_per_task"]),
                safe_task_seed(SEED, task_id),
            )
            train_metric["peak_cuda_allocated_bytes"] = max(
                train_metric["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device)
            )
            for parameter in model.parameters():
                parameter.grad = None
            training_metrics[task_id] = train_metric

            stage_name = f"after_{stage_index}_{task_id}"
            stage_order.append(stage_name)
            acquisition_stages[task_id] = stage_name
            stage_scores_for_all_tasks: dict[str, Any] = {}
            prediction_root = result_root / "predictions" / order_id / "dev" / stage_name
            for eval_task_id in task_ids:
                prediction_path = prediction_root / f"{eval_task_id}.jsonl"
                metric, rows = evaluate_task(
                    model,
                    tokenizer,
                    dev_records[eval_task_id],
                    eval_task_id,
                    task_types[eval_task_id],
                    config["model"],
                    config["evaluation"],
                    device,
                    prediction_path,
                )
                observed_ids = [row["instance_id"] for row in rows]
                if len(observed_ids) != expected_dev_count or len(set(observed_ids)) != expected_dev_count or set(observed_ids) != set(dev_ids[eval_task_id]):
                    raise RuntimeError(f"Dev output ID mismatch in {order_id}/{stage_name}/{eval_task_id}")
                stage_scores_for_all_tasks[eval_task_id] = metric
                print(
                    f"dev {order_id}/{stage_index}/{eval_task_id}: {metric['primary_metric']}={metric['primary_score']:.4f}",
                    flush=True,
                )
            stage_metrics[stage_name] = stage_scores_for_all_tasks
            score_matrix[stage_name] = {
                eval_task_id: float(stage_scores_for_all_tasks[eval_task_id]["primary_score"])
                for eval_task_id in task_ids
            }
            order_result["stage_order"] = list(stage_order)
            order_result["cl_metric_audit_partial"] = continual_learning_metrics(
                score_matrix, order, metric_by_task
            ) if len(score_matrix) == len(order) + 1 else None
            existing["active_order"] = order_result
            existing["active_stage"] = stage_name
            write_json_atomic(partial_path, existing)
            del order_result["cl_metric_audit_partial"]
            torch.cuda.empty_cache()

            if stage_index == len(order) and order_id in review_order_ids:
                checkpoint_path = checkpoint_root / f"{order_id}_{order_info['switches']}_switches_final"
                if not checkpoint_path.exists():
                    save_checkpoint(model, tokenizer, checkpoint_path)
                else:
                    print(f"Review checkpoint already exists; retaining it: {checkpoint_path}", flush=True)
                order_result["review_checkpoint_path"] = str(checkpoint_path)
                print(f"Saved review checkpoint: {checkpoint_path}", flush=True)

        order_result["duration_seconds"] = time.perf_counter() - started
        order_result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        order_result["cl_metric_audit"] = continual_learning_metrics(score_matrix, order, metric_by_task)
        order_result["mean_new_task_learning_normalized_gain"] = sum(
            order_result["cl_metric_audit"]["normalized_gain_matrix"][acquisition_stages[task]][task]
            for task in order
        ) / len(order)
        order_result["final_task_learning_normalized_gain"] = order_result["cl_metric_audit"]["normalized_gain_matrix"][stage_order[-1]][order[-1]]
        completed[order_id] = order_result
        existing["completed_orders"] = completed
        existing["active_order"] = None
        existing.pop("active_stage", None)
        write_json_atomic(partial_path, existing)

        completed_rows = []
        for completed_id, result in completed.items():
            completed_order = list(order_lookup[completed_id]["order"])
            completed_rows.append(make_csv_row(completed_id, result, completed_order, type_by_task, config))
        completed_rows.sort(key=lambda row: row["order_id"])
        write_csv_atomic(csv_path, completed_rows)
        print(
            f"Completed {order_id}: new-task gain={order_result['mean_new_task_learning_normalized_gain']:.4f}; "
            f"final forgetting={order_result['cl_metric_audit']['stage_metrics'][stage_order[-1]]['average_forgetting_nonnegative_normalized']:.4f}; "
            f"finished orders={len(completed)}/24",
            flush=True,
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if len(completed) != 24:
        raise RuntimeError(f"Expected 24 completed orders, found {len(completed)}")
    final_results = {
        "experiment_config": experiment_config,
        "completed_order_count": len(completed),
        "stage_metric_definition": "All primary task metrics are evaluated on development only; normalized gain, nonnegative forgetting, BWT, and cross-stage AAA use src/task_aware_transnas/cl/metrics.py.",
        "completed_orders": completed,
    }
    write_json_atomic(result_root / "order_metrics.json", final_results)
    final_rows = []
    for order_info in order_rows:
        final_rows.append(
            make_csv_row(order_info["order_id"], completed[order_info["order_id"]], list(order_info["order"]), type_by_task, config)
        )
    write_csv_atomic(csv_path, final_rows)
    print(f"All 24 orders complete: {csv_path}", flush=True)
    print(f"Detailed order metrics: {result_root / 'order_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
