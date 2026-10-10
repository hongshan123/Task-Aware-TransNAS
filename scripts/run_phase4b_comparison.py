"""Paired A0/A1/A1-control continual-learning study; dev split only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from architecture_modules import apply_architecture, parameter_counts
from phase4_common import (
    TASK_IDS,
    evaluate_dev_task,
    prepare_context,
    safe_task_seed,
    teacher_forced_nll,
    train_task_recorded,
    write_json,
)
from run_formal_baseline import load_model, norm_text, parse_yes_no, safe_prompt, save_checkpoint
from task_aware_transnas.cl.metrics import continual_learning_metrics


ARCHITECTURES = ("A0", "A1", "A1-control")
SEQUENCES = {
    "S1": TASK_IDS,
    "S2": [TASK_IDS[0], TASK_IDS[2], TASK_IDS[1], TASK_IDS[3]],
}
SEEDS = (10, 20, 30)
TASK_CALIBRATED_STEPS = {
    TASK_IDS[0]: 50,
    TASK_IDS[1]: 250,
    TASK_IDS[2]: 50,
    TASK_IDS[3]: 50,
}
EQUAL_TOKEN_BUDGET_REFERENCE = 2900
ALLOWED_MISSING_TIED_KEYS = {"lm_head.weight"}


def digest_instance_ids(records: list[dict[str, Any]]) -> str:
    values = [str(record["instance_id"]) for record in records]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [str(row.get("prediction", "")) for row in rows]
    normalized = [norm_text(output) for output in outputs]
    lengths = [int(row.get("generated_token_count", 0)) for row in rows]
    parsed = [parse_yes_no(output) for output in outputs]
    return {
        "examples": len(rows),
        "mean_generated_token_ids_including_eos": statistics.mean(lengths) if lengths else 0.0,
        "median_generated_token_ids_including_eos": statistics.median(lengths) if lengths else 0.0,
        "exact_yes_share": sum(output == "yes" for output in normalized) / max(1, len(outputs)),
        "exact_no_share": sum(output == "no" for output in normalized) / max(1, len(outputs)),
        "yes_no_exact_share": sum(output in {"yes", "no"} for output in normalized) / max(1, len(outputs)),
        "yes_no_parse_share": sum(value is not None for value in parsed) / max(1, len(parsed)),
        "eos_termination_rate": sum(bool(row.get("eos_terminated")) for row in rows) / max(1, len(rows)),
    }


def last_position_logits(model: Any, tokenizer: Any, record: dict[str, Any], config: dict[str, Any], device: torch.device) -> torch.Tensor:
    prompt, _, _ = safe_prompt(tokenizer, record, int(config["model"]["max_sequence_length"]))
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
    return logits.cpu()


def monitor_gate_step(step: int, model: Any, architecture: str, sequence_name: str, run_seed: int, task_id: str) -> dict[str, Any]:
    adapters = [layer.self_attn.ta_head_interaction for layer in model.model.layers]
    gates = torch.stack([adapter.residual_scale.detach().float() for adapter in adapters]).cpu().tolist()
    gradients = torch.stack(
        [
            adapter.residual_scale.grad.detach().float().abs() if adapter.residual_scale.grad is not None else torch.zeros(())
            for adapter in adapters
        ]
    ).cpu().tolist()
    branch_rms = torch.stack(
        [adapter.last_branch_rms.detach().float() if adapter.last_branch_rms is not None else torch.zeros(()) for adapter in adapters]
    ).cpu().tolist()
    return {
        "architecture": architecture,
        "sequence": sequence_name,
        "run_seed": run_seed,
        "task_id": task_id,
        "optimizer_step_in_task": step,
        "gate_value_after_update": gates,
        "gate_gradient_l2_after_global_clip": gradients,
        "added_branch_rms_from_last_microbatch": branch_rms,
    }


def gate_trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"recorded_steps": 0}
    layer_count = len(rows[0]["gate_value_after_update"])
    output: dict[str, Any] = {"recorded_steps": len(rows), "layers": []}
    for layer in range(layer_count):
        gates = [row["gate_value_after_update"][layer] for row in rows]
        grads = [row["gate_gradient_l2_after_global_clip"][layer] for row in rows]
        rms = [row["added_branch_rms_from_last_microbatch"][layer] for row in rows]
        output["layers"].append(
            {
                "layer": layer,
                "first_gate_value": gates[0],
                "last_gate_value": gates[-1],
                "mean_abs_gate_value": statistics.mean(abs(value) for value in gates),
                "mean_gate_gradient_l2": statistics.mean(grads),
                "max_gate_gradient_l2": max(grads),
                "mean_added_branch_rms": statistics.mean(rms),
                "max_added_branch_rms": max(rms),
                "nonzero_branch_steps": sum(value > 0 for value in rms),
            }
        )
    return output


def evaluation_matrix(
    model: Any,
    architecture: str,
    sequence_name: str,
    stage_name: str,
    config: dict[str, Any],
    tokenizer: Any,
    dev: dict[str, list[dict[str, Any]]],
    device: torch.device,
    run_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage_tasks: dict[str, Any] = {}
    primary_scores: dict[str, float] = {}
    started = time.perf_counter()
    for task_id in TASK_IDS:
        rows_for_task = dev[task_id]
        nll = teacher_forced_nll(model, tokenizer, rows_for_task, config["model"], device)
        prediction_path = run_root / "predictions" / stage_name / f"{task_id}.jsonl"
        metrics, predictions = evaluate_dev_task(
            model,
            tokenizer,
            rows_for_task,
            task_id,
            config["dataset"]["task_types"][task_id],
            config["model"],
            config["evaluation"],
            device,
            prediction_path,
        )
        metric_name = config["dataset"]["task_types"][task_id]["primary_metric"]
        primary_scores[task_id] = float(metrics[metric_name])
        stage_tasks[task_id] = {
            "metrics": metrics,
            "teacher_forced_token_nll": nll,
            "generation_behavior": summarize_predictions(predictions),
            "prediction_file": str(prediction_path),
        }
    torch.cuda.synchronize(device)
    stage_tasks["_evaluation_summary"] = {
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "dev_examples_per_task": len(dev[TASK_IDS[0]]),
    }
    return stage_tasks, primary_scores


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reload_checkpoint_audit(
    base_path: Path,
    checkpoint_path: Path,
    architecture: str,
    tokenizer: Any,
    record: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    reference_logits: torch.Tensor,
) -> dict[str, Any]:
    reloaded = load_model(base_path, local_only=True, training=False)
    apply_architecture(reloaded, architecture)
    state_path = checkpoint_path / "model.safetensors"
    state = load_file(str(state_path), device="cpu")
    incompatible = reloaded.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    del state
    if set(missing) - ALLOWED_MISSING_TIED_KEYS or unexpected:
        raise RuntimeError(f"Checkpoint reload key mismatch: missing={missing}, unexpected={unexpected}")
    restored_logits = last_position_logits(reloaded, tokenizer, record, config, device)
    max_difference = float((restored_logits - reference_logits).abs().max().item())
    result = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": state_path.stat().st_size,
        "missing_tied_keys": missing,
        "unexpected_keys": unexpected,
        "max_absolute_logit_difference": max_difference,
        "passes_reload_check": max_difference <= 1e-6,
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(config: dict[str, Any], model_path: Path, output_root: Path) -> dict[str, Any]:
    disk_root = Path(config["storage"]["root"])
    free_bytes = shutil.disk_usage(disk_root).free
    source_bytes = (model_path / "model.safetensors").stat().st_size
    audit_path = output_root / "audit" / "architecture_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Run scripts/audit_phase4b_architectures.py before disk preflight")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    parameter_count_by_architecture = {
        row["architecture"]: int(row["parameter_counts"]["total_parameters"])
        for row in audit["functional_audits"]
    }
    estimated_checkpoint_bytes_by_architecture = {
        architecture: int(parameter_count_by_architecture[architecture] * 4 + 2 * 1024**2)
        for architecture in ARCHITECTURES
    }
    checkpoint_count = len(ARCHITECTURES) * len(SEQUENCES) * len(SEEDS)
    runs_per_architecture = len(SEQUENCES) * len(SEEDS)
    checkpoint_bytes = sum(estimated_checkpoint_bytes_by_architecture.values()) * runs_per_architecture
    prediction_and_metrics_bytes = 1024**3
    temporary_save_bytes = 3 * 1024**3
    safety_margin_bytes = 8 * 1024**3
    required_bytes = checkpoint_bytes + prediction_and_metrics_bytes + temporary_save_bytes + safety_margin_bytes
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free disk space: estimated need {required_bytes / 1024**3:.1f} GiB, "
            f"available {free_bytes / 1024**3:.1f} GiB. No training has started."
        )
    rough_train_seconds = 18 * sum(TASK_CALIBRATED_STEPS.values()) * 1.0
    result = {
        "run_count": checkpoint_count,
        "architectures": list(ARCHITECTURES),
        "sequences": SEQUENCES,
        "seeds": list(SEEDS),
        "budget_protocol": "task_calibrated",
        "steps_by_task": TASK_CALIBRATED_STEPS,
        "checkpoint_policy": "Keep only the final checkpoint for each of the 18 completed runs.",
        "source_safetensors_bytes": source_bytes,
        "checkpoint_parameter_counts": parameter_count_by_architecture,
        "estimated_checkpoint_bytes_each_by_architecture": estimated_checkpoint_bytes_by_architecture,
        "estimated_checkpoint_total_gib": checkpoint_bytes / 1024**3,
        "reserved_for_predictions_metrics_gib": prediction_and_metrics_bytes / 1024**3,
        "temporary_save_margin_gib": temporary_save_bytes / 1024**3,
        "additional_safety_margin_gib": safety_margin_bytes / 1024**3,
        "estimated_required_free_gib": required_bytes / 1024**3,
        "available_free_gib_before_start": free_bytes / 1024**3,
        "rough_training_time_hours_lower_bound": rough_train_seconds / 3600,
        "output_root": str(output_root),
        "preflight_passed": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "formal_preflight.json", result)
    print(
        f"Disk preflight passed: need about {required_bytes / 1024**3:.1f} GiB; "
        f"free {free_bytes / 1024**3:.1f} GiB. Final checkpoints only.",
        flush=True,
    )
    return result


def run_one(
    architecture: str,
    sequence_name: str,
    seed: int,
    config: dict[str, Any],
    model_path: Path,
    tokenizer: Any,
    train: dict[str, list[dict[str, Any]]],
    dev: dict[str, list[dict[str, Any]]],
    device: torch.device,
    output_root: Path,
    model_sha256: str,
) -> dict[str, Any]:
    order = SEQUENCES[sequence_name]
    run_root = output_root / "runs" / architecture / sequence_name / f"seed_{seed}"
    final_metrics_path = run_root / "metrics.json"
    checkpoint_path = output_root / "checkpoints" / architecture / sequence_name / f"seed_{seed}"
    if final_metrics_path.exists() and (checkpoint_path / "model.safetensors").exists():
        print(f"Already complete: {architecture}/{sequence_name}/seed_{seed}", flush=True)
        return json.loads(final_metrics_path.read_text(encoding="utf-8"))

    print(f"\n=== RUN {architecture} | {sequence_name} | seed={seed} ===", flush=True)
    run_started = time.perf_counter()
    model = load_model(model_path, local_only=True, training=True)
    torch.manual_seed(seed)
    apply_architecture(model, architecture)
    counts = parameter_counts(model)
    if counts["trainable_parameters"] != counts["total_parameters"]:
        raise RuntimeError(f"Not all parameters are trainable: {counts}")
    reference_record = dev[TASK_IDS[0]][0]
    stage_order = ["baseline"]
    scores_by_stage: dict[str, dict[str, float]] = {}
    task_metrics: dict[str, Any] = {}
    training_metrics: dict[str, Any] = {}
    evaluation_metrics: dict[str, Any] = {}
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0

    torch.cuda.reset_peak_memory_stats(device)
    baseline_tasks, baseline_scores = evaluation_matrix(
        model, architecture, sequence_name, "baseline", config, tokenizer, dev, device, run_root
    )
    scores_by_stage["baseline"] = baseline_scores
    task_metrics["baseline"] = baseline_tasks
    evaluation_metrics["baseline"] = baseline_tasks.pop("_evaluation_summary")
    peak_allocated_bytes = max(peak_allocated_bytes, evaluation_metrics["baseline"]["peak_cuda_allocated_bytes"])
    peak_reserved_bytes = max(peak_reserved_bytes, evaluation_metrics["baseline"]["peak_cuda_reserved_bytes"])
    write_json(
        run_root / "metrics_partial.json",
        {
            "architecture": architecture,
            "sequence_name": sequence_name,
            "task_order": order,
            "seed": seed,
            "stage_order": stage_order,
            "stage_scores_by_primary_metric": scores_by_stage,
            "task_metrics": task_metrics,
            "training_metrics": training_metrics,
            "source_model_sha256": model_sha256,
        },
    )

    for stage_index, task_id in enumerate(order, start=1):
        steps = TASK_CALIBRATED_STEPS[task_id]
        task_seed = safe_task_seed(seed, task_id)
        torch.cuda.reset_peak_memory_stats(device)
        gate_rows: list[dict[str, Any]] = []
        callback = None
        if architecture in {"A1", "A1-control"}:
            callback = lambda step, current_model: gate_rows.append(
                monitor_gate_step(step, current_model, architecture, sequence_name, seed, task_id)
            )
        train_row = train_task_recorded(
            model,
            tokenizer,
            train[task_id],
            config["training"],
            config["model"],
            device,
            float(config["training"]["learning_rate"]),
            steps,
            task_seed,
            on_optimizer_step=callback,
        )
        train_row["task_id"] = task_id
        train_row["training_seed"] = task_seed
        train_row["max_cuda_allocated_after_training_bytes"] = max(
            train_row["peak_cuda_allocated_bytes"], torch.cuda.max_memory_allocated(device)
        )
        train_row["max_cuda_reserved_after_training_bytes"] = max(
            train_row["peak_cuda_reserved_bytes"], torch.cuda.max_memory_reserved(device)
        )
        if gate_rows:
            trace_path = run_root / "gate_training" / f"{task_id}.jsonl"
            write_jsonl(trace_path, gate_rows)
            train_row["gate_trace_file"] = str(trace_path)
            train_row["gate_statistics"] = gate_trace_summary(gate_rows)
        training_metrics[task_id] = train_row
        peak_allocated_bytes = max(peak_allocated_bytes, train_row["max_cuda_allocated_after_training_bytes"])
        peak_reserved_bytes = max(peak_reserved_bytes, train_row["max_cuda_reserved_after_training_bytes"])

        stage_name = f"after_{stage_index}_{task_id}"
        stage_order.append(stage_name)
        torch.cuda.reset_peak_memory_stats(device)
        stage_tasks, stage_scores = evaluation_matrix(
            model, architecture, sequence_name, stage_name, config, tokenizer, dev, device, run_root
        )
        scores_by_stage[stage_name] = stage_scores
        task_metrics[stage_name] = stage_tasks
        evaluation_metrics[stage_name] = stage_tasks.pop("_evaluation_summary")
        peak_allocated_bytes = max(peak_allocated_bytes, evaluation_metrics[stage_name]["peak_cuda_allocated_bytes"])
        peak_reserved_bytes = max(peak_reserved_bytes, evaluation_metrics[stage_name]["peak_cuda_reserved_bytes"])
        partial = {
            "architecture": architecture,
            "sequence_name": sequence_name,
            "task_order": order,
            "seed": seed,
            "stage_order": stage_order,
            "stage_scores_by_primary_metric": scores_by_stage,
            "task_metrics": task_metrics,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics,
            "source_model_sha256": model_sha256,
        }
        write_json(run_root / "metrics_partial.json", partial)
        print(
            f"Stage {stage_index}/4 {task_id}: steps={steps}, supervised tokens="
            f"{train_row['supervised_target_tokens']}, train={train_row['duration_seconds']:.1f}s, "
            f"eval={evaluation_metrics[stage_name]['duration_seconds']:.1f}s",
            flush=True,
        )

    metric_by_task = {
        task_id: str(config["dataset"]["task_types"][task_id]["primary_metric"]) for task_id in TASK_IDS
    }
    cl = continual_learning_metrics(scores_by_stage, order, metric_by_task)
    acquisition: dict[str, Any] = {}
    for index, task_id in enumerate(order, start=1):
        stage_name = stage_order[index]
        base_score = scores_by_stage["baseline"][task_id]
        learned_score = scores_by_stage[stage_name][task_id]
        acquisition[stage_name] = {
            "new_task": task_id,
            "primary_metric": metric_by_task[task_id],
            "baseline_score": base_score,
            "score_after_acquisition": learned_score,
            "raw_acquisition_gain": learned_score - base_score,
            "normalized_acquisition_gain": cl["normalized_gain_matrix"][stage_name][task_id],
        }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    reference_logits = last_position_logits(model, tokenizer, reference_record, config, device)
    save_checkpoint(model, tokenizer, checkpoint_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reload_audit = reload_checkpoint_audit(
        model_path,
        checkpoint_path,
        architecture,
        tokenizer,
        reference_record,
        config,
        device,
        reference_logits,
    )
    if not reload_audit["passes_reload_check"]:
        raise RuntimeError(f"Checkpoint reload changed logits: {reload_audit}")

    torch.cuda.synchronize(device)
    final = {
        "architecture": architecture,
        "sequence_name": sequence_name,
        "task_order": order,
        "seed": seed,
        "training_protocol": {
            "name": "task_calibrated",
            "steps_by_task": TASK_CALIBRATED_STEPS,
            "microbatch_size": 1,
            "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
            "learning_rate": float(config["training"]["learning_rate"]),
            "weight_decay": float(config["training"]["weight_decay"]),
            "max_sequence_length": int(config["model"]["max_sequence_length"]),
            "max_target_tokens": int(config["model"]["max_target_tokens"]),
            "optimizer_reinitialized_at_task_boundary": True,
            "optimizer": "AdamW(foreach=False)",
        },
        "parameter_counts": counts,
        "source_model": str(model_path),
        "source_model_sha256": model_sha256,
        "local_files_only": True,
        "test_split_loaded": False,
        "split": "dev",
        "train_instance_id_sha256_by_task": {task: digest_instance_ids(train[task]) for task in TASK_IDS},
        "dev_instance_id_sha256_by_task": {task: digest_instance_ids(dev[task]) for task in TASK_IDS},
        "stage_order": stage_order,
        "stage_scores_by_primary_metric": scores_by_stage,
        "task_metrics": task_metrics,
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "continual_learning_metrics": cl,
        "new_task_acquisition": acquisition,
        "checkpoint_reload_audit": reload_audit,
        "checkpoint_path": str(checkpoint_path),
        "peak_cuda_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_reserved_bytes,
        "total_duration_seconds": time.perf_counter() - run_started,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    write_json(final_metrics_path, final)
    print(
        f"Saved {architecture}/{sequence_name}/seed_{seed}; peak allocated "
        f"{peak_allocated_bytes / 1024**3:.2f} GiB; checkpoint reload exact.",
        flush=True,
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config, model_path, tokenizer, device, train, dev = prepare_context()
    output_root = Path(config["storage"]["root"]) / "results" / "phase4b"
    preflight = run_preflight(config, model_path, output_root)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return

    audit_path = output_root / "audit" / "architecture_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError("Run scripts/audit_phase4b_architectures.py before formal training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("base_weights_same_across_A0_A1_A1_control") or not all(
        item.get("passes_all_output_checks") for item in audit["functional_audits"]
    ):
        raise RuntimeError("The architecture compatibility audit did not pass; no experiment started")
    model_sha256 = str(audit["source_model_safetensors_sha256"])
    all_results: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    manifest_path = output_root / "formal_run_manifest_partial.json"
    for sequence_name in SEQUENCES:
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                row = run_one(
                    architecture,
                    sequence_name,
                    seed,
                    config,
                    model_path,
                    tokenizer,
                    train,
                    dev,
                    device,
                    output_root,
                    model_sha256,
                )
                all_results.append(
                    {
                        "architecture": architecture,
                        "sequence_name": sequence_name,
                        "seed": seed,
                        "metrics_file": str(
                            output_root / "runs" / architecture / sequence_name / f"seed_{seed}" / "metrics.json"
                        ),
                    }
                )
                write_json(
                    manifest_path,
                    {
                        "completed_runs": all_results,
                        "planned_run_count": 18,
                        "elapsed_seconds": time.perf_counter() - total_started,
                        "test_split_loaded": False,
                    },
                )
    final_manifest = {
        "completed_runs": all_results,
        "planned_run_count": 18,
        "elapsed_seconds": time.perf_counter() - total_started,
        "test_split_loaded": False,
        "completed": len(all_results) == 18,
    }
    write_json(output_root / "formal_run_manifest.json", final_manifest)
    print(f"All paired runs complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
