"""Run the test-free Phase 7A smoke and single-seed continual-learning pilot."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv(PROJECT_ROOT / ".env", override=False)

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase4_common import evaluate_dev_task, safe_task_seed, teacher_forced_nll, train_task_recorded, write_json
from phase7a_engine import (
    ACTIVATION_RESIDUAL_TOLERANCE,
    ConstraintAuditor,
    _make_mask_specs,
    audit_rows_to_jsonl,
    capture_input_covariances,
    compute_functional_importance,
    prepare_parsp_constraints,
    write_importance_artifacts,
)
from phase7a_parsp import load_phase7a_task_split, official_repo_metadata
from run_formal_baseline import load_project_config, resolved_model_path


PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase7a.yaml"
PHASE_RESULTS = PROJECT_ROOT / "results" / "phase7a"
TASK_TYPES = {
    "task565_circa_answer_generation": "generation",
    "task1603_smcalflow_sentence_generation": "generation",
    "task1501_dstc3_answer_generation": "classification",
    "task1384_deal_or_no_dialog_classification": "classification",
}


def load_phase_context() -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    phase_cfg = yaml.safe_load(PHASE_CONFIG.read_text(encoding="utf-8"))
    baseline_cfg = load_project_config(PROJECT_ROOT / "configs" / "continual_baseline_full.yaml")
    model_path = resolved_model_path(baseline_cfg)
    manifest_path = Path(baseline_cfg["dataset"]["processed_dir"]) / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared CITB manifest is unavailable: {manifest_path}")
    return phase_cfg, baseline_cfg, model_path, manifest_path


def load_tokenizer(model_path: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_path: Path, device: torch.device, training: bool) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
    else:
        model.config.use_cache = True
        model.eval()
    return model


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_model(model: Any, path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Phase 7A checkpoint: {path}")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)


def _audit_has_violation(rows: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    mask_violations = [
        row for row in rows if row.get("mask_out_update_ratio") is not None and row["mask_out_update_ratio"] > 1e-8
    ]
    activation_violations = [
        row
        for row in rows
        if int(row.get("anchor_tokens", 0)) > 0
        and float(row.get("normalized_history_activation_residual", 0.0)) > ACTIVATION_RESIDUAL_TOLERANCE
    ]
    evidence = {
        "mask_violation_rows": len(mask_violations),
        "max_mask_out_update_ratio": max(
            (float(row["mask_out_update_ratio"]) for row in mask_violations), default=0.0
        ),
        "activation_violation_rows": len(activation_violations),
        "max_normalized_history_activation_residual": max(
            (float(row["normalized_history_activation_residual"]) for row in activation_violations), default=0.0
        ),
        "thresholds": {
            "mask_out_update_ratio": 1e-8,
            "normalized_history_activation_residual": ACTIVATION_RESIDUAL_TOLERANCE,
        },
    }
    return bool(mask_violations or activation_violations), evidence


def _condition_training(
    sequence_name: str,
    condition: str,
    device: torch.device,
    phase_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
    model_path: Path,
    manifest_path: Path,
    steps_per_task: int,
    localization_count: int,
    covariance_count: int,
    task_limit: int | None = None,
    run_dir_override: Path | None = None,
) -> dict[str, Any]:
    seed = int(phase_cfg["seed"])
    sequence = list(phase_cfg["sequences"][sequence_name])
    if task_limit is not None:
        sequence = sequence[:task_limit]
    run_dir = run_dir_override or (PHASE_RESULTS / sequence_name / condition)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing Phase 7A run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device, training=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    task_training: dict[str, Any] = {}
    opened_paths: list[str] = []
    seen_train: dict[str, list[dict[str, Any]]] = {}
    all_audit_rows: list[dict[str, Any]] = []
    start_utc = datetime.now(timezone.utc)

    for stage_index, task_id in enumerate(sequence, start=1):
        print(f"[{sequence_name}/{condition}] task {stage_index}/{len(sequence)}: {task_id}", flush=True)
        train_records = load_phase7a_task_split(manifest_path, task_id, "train", opened_paths)
        seen_train[task_id] = train_records
        importance = None
        auditor = None
        hook_handles: list[Any] = []
        projection_metadata: dict[str, Any] = {}
        history_records: dict[str, list[dict[str, Any]]] = {}
        if condition != "C0":
            importance = compute_functional_importance(
                model,
                tokenizer,
                train_records,
                baseline_cfg["model"],
                device,
                seed=seed,
                sample_count=localization_count,
                layer_ratio=float(phase_cfg["layer_ratio"]),
                neuron_ratio=float(phase_cfg["neuron_ratio"]),
            )
            write_importance_artifacts(run_dir / "importance" / task_id, task_id, importance)
            target_names, _ = _make_mask_specs(model, importance["selected"], apply_mask=False)
            history_records = {
                previous_task: seen_train[previous_task]
                for previous_task in sequence[: stage_index - 1]
            }
            history_covariances = {}
            if history_records:
                history_covariances = capture_input_covariances(
                    model,
                    tokenizer,
                    history_records,
                    target_names,
                    baseline_cfg["model"],
                    device,
                    seed=seed,
                    sample_count_per_task=covariance_count,
                )
            auditor, hook_handles, projection_metadata = prepare_parsp_constraints(
                model,
                condition,
                importance,
                history_covariances,
                run_dir / "projectors" / f"stage_{stage_index}",
                sequence_name,
                task_id,
                seed,
                auditor_strict=(condition == "C4"),
            )
            if history_records:
                calibration_ids = next(iter(history_covariances.values())).get("sample_ids_by_task", {})
                calibration_record = {
                    "task_ids": list(calibration_ids),
                    "sample_count_per_task": covariance_count,
                    "sample_id_sha256": hashlib.sha256(
                        json.dumps(
                            {task: sorted(ids) for task, ids in calibration_ids.items()},
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "sample_ids_by_task": calibration_ids,
                    "used_for": "forward-only covariance calibration; no loss, backward pass, or optimizer update",
                }
            else:
                calibration_record = {"task_ids": [], "sample_count_per_task": 0, "used_for": "identity projector"}
            write_json(run_dir / "importance" / task_id / "calibration_manifest.json", calibration_record)

        torch.cuda.reset_peak_memory_stats(device)
        train_metrics = train_task_recorded(
            model,
            tokenizer,
            train_records,
            baseline_cfg["training"],
            baseline_cfg["model"],
            device,
            float(baseline_cfg["training"]["learning_rate"]),
            steps_per_task,
            safe_task_seed(seed, task_id),
            before_optimizer_step=auditor.before_step if auditor is not None else None,
            after_optimizer_step=auditor.after_step if auditor is not None else None,
            on_optimizer_attempt_start=auditor.reset_attempt if auditor is not None else None,
        )
        train_metrics["peak_cuda_allocated_bytes"] = max(
            int(train_metrics["peak_cuda_allocated_bytes"]), int(torch.cuda.max_memory_allocated(device))
        )
        train_metrics["peak_cuda_reserved_bytes"] = max(
            int(train_metrics["peak_cuda_reserved_bytes"]), int(torch.cuda.max_memory_reserved(device))
        )
        train_metrics["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        train_metrics["trainable_parameter_fraction"] = train_metrics["trainable_parameters"] / total_parameters
        train_metrics["condition"] = condition
        train_metrics["task_id"] = task_id
        train_metrics["importance_sample_ids"] = importance["sample_ids"] if importance else []
        train_metrics["calibration_task_ids"] = list(history_records)
        train_metrics["projector"] = projection_metadata
        task_training[task_id] = train_metrics
        if auditor is not None:
            audit_path = run_dir / "constraint_audit" / f"{task_id}.jsonl"
            audit_rows_to_jsonl(audit_path, auditor.step_rows)
            all_audit_rows.extend(auditor.step_rows)
            for handle in hook_handles:
                handle.remove()
        checkpoint_path = checkpoint_dir / f"stage_{stage_index}_{task_id}"
        _save_model(model, checkpoint_path)
        task_training[task_id]["checkpoint"] = str(checkpoint_path.resolve())
        task_training[task_id]["checkpoint_size_bytes"] = sum(path.stat().st_size for path in checkpoint_path.iterdir() if path.is_file())
        write_json(
            run_dir / "metrics_training_partial.json",
            {
                "sequence": sequence_name,
                "condition": condition,
                "seed": seed,
                "task_order": sequence,
                "task_training": task_training,
                "opened_split_paths": opened_paths,
                "evaluation_splits_loaded": [],
            },
        )
        del auditor, hook_handles, importance
        gc.collect()
        torch.cuda.empty_cache()

    finished_utc = datetime.now(timezone.utc)
    violation, violation_evidence = _audit_has_violation(all_audit_rows)
    training_summary = {
        "started_utc": start_utc.isoformat(),
        "finished_utc": finished_utc.isoformat(),
        "training_duration_seconds": (finished_utc - start_utc).total_seconds(),
        "sequence": sequence_name,
        "condition": condition,
        "seed": seed,
        "task_order": sequence,
        "steps_per_task": steps_per_task,
        "learning_rate": float(baseline_cfg["training"]["learning_rate"]),
        "gradient_accumulation_steps": int(baseline_cfg["training"]["gradient_accumulation_steps"]),
        "weight_decay": float(baseline_cfg["training"]["weight_decay"]),
        "optimizer": "torch.optim.AdamW (fresh optimizer per task; same as Phase 4 baseline)",
        "model_path": str(model_path.resolve()),
        "model_parameters": total_parameters,
        "device": torch.cuda.get_device_name(device),
        "gpu_index": device.index,
        "task_training": task_training,
        "constraint_violation_detected": violation,
        "constraint_violation_evidence": violation_evidence,
        "opened_split_paths": opened_paths,
        "evaluation_splits_loaded": [],
        "official_parsp": official_repo_metadata(),
    }
    write_json(run_dir / "metrics_training.json", training_summary)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return training_summary


def _primary_score(metric: dict[str, Any]) -> float:
    return float(metric["primary_score"])


def _load_named_parameter_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _update_density(previous: dict[str, torch.Tensor], model: Any) -> dict[str, Any]:
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        current = parameter.detach().cpu()
        before = previous[name]
        changed += int(torch.count_nonzero(current != before).item())
        total += current.numel()
    return {"changed_parameter_values": changed, "total_parameter_values": total, "effective_updated_parameter_ratio": changed / max(1, total)}


def evaluate_sequence(
    sequence_name: str,
    conditions: list[str],
    device: torch.device,
    phase_cfg: dict[str, Any],
    baseline_cfg: dict[str, Any],
    model_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    sequence = list(phase_cfg["sequences"][sequence_name])
    if len(sequence) != 4:
        raise ValueError("Full Phase 7A evaluation requires the four-task CITB sequence")
    access_log: list[str] = []
    dev_records = {
        task_id: load_phase7a_task_split(manifest_path, task_id, "dev", access_log)
        for task_id in sequence
    }
    tokenizer = load_tokenizer(model_path)
    model_cfg = baseline_cfg["model"]
    eval_cfg = baseline_cfg["evaluation"]
    task_types = baseline_cfg["dataset"]["task_types"]
    results: dict[str, Any] = {"sequence": sequence_name, "seed": int(phase_cfg["seed"]), "baseline": {}, "conditions": {}}
    print(f"[{sequence_name}] dev-only evaluation: base model", flush=True)
    base_model = load_model(model_path, device, training=False)
    previous_state = _load_named_parameter_state(base_model)
    for task_id in sequence:
        metric, _ = evaluate_dev_task(
            base_model,
            tokenizer,
            dev_records[task_id],
            task_id,
            task_types[task_id],
            model_cfg,
            eval_cfg,
            device,
            PHASE_RESULTS / sequence_name / "baseline" / "predictions" / f"{task_id}.jsonl",
        )
        metric["teacher_forced_nll"] = teacher_forced_nll(
            base_model, tokenizer, dev_records[task_id], model_cfg, device
        )
        if task_types[task_id]["type"] == "classification":
            metric["yes_no_error_rate"] = 1.0 - float(metric["parse_success_rate"])
        results["baseline"][task_id] = metric
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    for condition in conditions:
        run_dir = PHASE_RESULTS / sequence_name / condition
        training_summary = json.loads((run_dir / "metrics_training.json").read_text(encoding="utf-8"))
        stage_scores: dict[str, dict[str, Any]] = {"baseline": results["baseline"]}
        stage_names = ["baseline"]
        condition_stages: dict[str, Any] = {}
        base_state_model = load_model(model_path, device, training=False)
        previous_state = _load_named_parameter_state(base_state_model)
        del base_state_model
        gc.collect()
        # Only the CPU base weights are kept for update counts.
        torch.cuda.empty_cache()
        for stage_index, task_id in enumerate(sequence, start=1):
            checkpoint = Path(training_summary["task_training"][task_id]["checkpoint"])
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False, dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()
            model.config.use_cache = True
            update_density = _update_density(previous_state, model)
            previous_state = _load_named_parameter_state(model)
            stage_name = f"after_{stage_index}_{task_id}"
            current_metrics = {}
            for learned_task in sequence[:stage_index]:
                print(f"[{sequence_name}/{condition}] dev {stage_name}: {learned_task}", flush=True)
                metric, _ = evaluate_dev_task(
                    model,
                    tokenizer,
                    dev_records[learned_task],
                    learned_task,
                    task_types[learned_task],
                    model_cfg,
                    eval_cfg,
                    device,
                    run_dir / "predictions" / stage_name / f"{learned_task}.jsonl",
                )
                metric["teacher_forced_nll"] = teacher_forced_nll(
                    model, tokenizer, dev_records[learned_task], model_cfg, device
                )
                if task_types[learned_task]["type"] == "classification":
                    metric["yes_no_error_rate"] = 1.0 - float(metric["parse_success_rate"])
                current_metrics[learned_task] = metric
            stage_scores[stage_name] = current_metrics
            stage_names.append(stage_name)
            previous_scores = []
            for old_index, old_task in enumerate(sequence[: stage_index - 1], start=1):
                observed_before = [
                    _primary_score(stage_scores[prior_stage][old_task])
                    for prior_stage in stage_names[1:]
                    if old_task in stage_scores[prior_stage]
                ]
                previous_scores.append(max(observed_before) - _primary_score(current_metrics[old_task]))
            bwt_values = [
                _primary_score(current_metrics[old_task])
                - _primary_score(stage_scores[f"after_{old_index}_{old_task}"][old_task])
                for old_index, old_task in enumerate(sequence[: stage_index - 1], start=1)
            ]
            current_score = _primary_score(current_metrics[task_id])
            base_score = _primary_score(results["baseline"][task_id])
            condition_stages[stage_name] = {
                "task_metrics": current_metrics,
                "aaa": sum(_primary_score(current_metrics[item]) for item in current_metrics) / len(current_metrics),
                "average_forgetting": sum(previous_scores) / len(previous_scores) if previous_scores else 0.0,
                "bwt": sum(bwt_values) / len(bwt_values) if bwt_values else 0.0,
                "acquisition": current_score - base_score,
                "normalized_acquisition": (current_score - base_score) / max(1.0 - base_score, 1e-8),
                "update_density": update_density,
            }
            training_summary["task_training"][task_id].update(update_density)
            condition_stages[stage_name]["training"] = training_summary["task_training"][task_id]
            del model
            gc.collect()
            torch.cuda.empty_cache()
        condition_result = {
            "training": training_summary,
            "stage_metrics": stage_scores,
            "continual_learning_summary": condition_stages,
        }
        results["conditions"][condition] = condition_result
        write_json(run_dir / "metrics.json", condition_result)
        with (run_dir / "task_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["stage", "task_id", "task_type", "primary_metric", "primary_score", "rouge_l_f1", "token_f1", "accuracy", "macro_f1", "yes_no_error_rate", "teacher_forced_nll"],
            )
            writer.writeheader()
            for stage_name, metrics in stage_scores.items():
                for task_id, metric in metrics.items():
                    writer.writerow(
                        {
                            "stage": stage_name,
                            "task_id": task_id,
                            "task_type": task_types[task_id]["type"],
                            "primary_metric": metric.get("primary_metric"),
                            "primary_score": metric.get("primary_score"),
                            "rouge_l_f1": metric.get("rouge_l_f1"),
                            "token_f1": metric.get("token_f1"),
                            "accuracy": metric.get("accuracy"),
                            "macro_f1": metric.get("macro_f1"),
                            "yes_no_error_rate": metric.get("yes_no_error_rate"),
                            "teacher_forced_nll": metric.get("teacher_forced_nll", {}).get("token_nll"),
                        }
                    )

    if any("/test" in value.casefold() or "\\test" in value.casefold() for value in access_log):
        raise RuntimeError("Phase 7A opened a test split path")
    results["opened_split_paths"] = access_log
    results["evaluation_splits_loaded"] = ["dev"]
    results["test_split_files_opened"] = []
    results["official_parsp"] = official_repo_metadata()
    write_json(PHASE_RESULTS / sequence_name / "evaluation_summary.json", results)
    return results


def _load_completed_condition_summary(run_dir: Path) -> dict[str, Any] | None:
    """Load a complete condition so a failed later condition can resume safely."""
    summary_path = run_dir / "metrics_training.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_sequence(
    sequence_name: str,
    device: torch.device,
    smoke: bool = False,
    smoke_steps: int = 5,
    smoke_task_count: int = 1,
) -> None:
    phase_cfg, baseline_cfg, model_path, manifest_path = load_phase_context()
    sequence = list(phase_cfg["sequences"][sequence_name])
    seed_everything(int(phase_cfg["seed"]), device)
    if smoke:
        if not 1 <= smoke_task_count <= 4:
            raise ValueError("Smoke task count must be between 1 and 4")
        smoke_root = PHASE_RESULTS / ("smoke" if smoke_task_count == 1 else f"smoke_{smoke_task_count}task")
        summary = _condition_training(
            sequence_name,
            "C1",
            device,
            phase_cfg,
            baseline_cfg,
            model_path,
            manifest_path,
            steps_per_task=smoke_steps,
            localization_count=4,
            covariance_count=4,
            task_limit=smoke_task_count,
            run_dir_override=smoke_root / "C1",
        )
        write_json(smoke_root / "smoke_summary.json", summary)
        last_task_id = summary["task_order"][-1]
        checkpoint = Path(summary["task_training"][last_task_id]["checkpoint"])
        restored = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        if sum(parameter.numel() for parameter in restored.parameters()) != summary["model_parameters"]:
            raise RuntimeError("Smoke checkpoint parameter count changed after reload")
        del restored
        gc.collect()
        torch.cuda.empty_cache()
        return

    conditions = ["C0", "C1", "C2", "C3"]
    summaries = {}
    for condition in conditions:
        run_dir = PHASE_RESULTS / sequence_name / condition
        completed = _load_completed_condition_summary(run_dir)
        if completed is not None:
            summaries[condition] = completed
            print(f"[{sequence_name}/{condition}] reusing completed training summary", flush=True)
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Incomplete run exists at {run_dir}; preserve or rename it before resuming this condition"
            )
        summaries[condition] = _condition_training(
            sequence_name,
            condition,
            device,
            phase_cfg,
            baseline_cfg,
            model_path,
            manifest_path,
            steps_per_task=int(phase_cfg["steps_per_task"]),
            localization_count=int(phase_cfg["localization_samples"]),
            covariance_count=int(phase_cfg["calibration_samples_per_task"]),
        )
    if summaries["C1"]["constraint_violation_detected"]:
        c4_run_dir = PHASE_RESULTS / sequence_name / "C4"
        completed_c4 = _load_completed_condition_summary(c4_run_dir)
        if completed_c4 is not None:
            summaries["C4"] = completed_c4
            print(f"[{sequence_name}/C4] reusing completed training summary", flush=True)
        else:
            if c4_run_dir.exists() and any(c4_run_dir.iterdir()):
                raise FileExistsError(
                    f"Incomplete run exists at {c4_run_dir}; preserve or rename it before resuming C4"
                )
            print(f"[{sequence_name}] C1 audit proves a constraint violation; running strict C4.", flush=True)
            summaries["C4"] = _condition_training(
                sequence_name,
                "C4",
                device,
                phase_cfg,
                baseline_cfg,
                model_path,
                manifest_path,
                steps_per_task=int(phase_cfg["steps_per_task"]),
                localization_count=int(phase_cfg["localization_samples"]),
                covariance_count=int(phase_cfg["calibration_samples_per_task"]),
            )
    else:
        print(f"[{sequence_name}] C1 audit found no thresholded violation; C4 is omitted.", flush=True)

    result = evaluate_sequence(sequence_name, list(summaries), device, phase_cfg, baseline_cfg, model_path, manifest_path)
    write_json(
        PHASE_RESULTS / sequence_name / "pilot_manifest.json",
        {
            "sequence": sequence_name,
            "seed": int(phase_cfg["seed"]),
            "conditions": list(summaries),
            "c4_inclusion_reason": summaries["C1"]["constraint_violation_evidence"],
            "evaluation_splits_loaded": result["evaluation_splits_loaded"],
            "test_split_files_opened": [],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot"], required=True)
    parser.add_argument("--sequence", choices=["S1", "S2"], default="S1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--smoke-task-count", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "smoke" and not 1 <= args.smoke_steps <= 10:
        raise ValueError("Phase 7A smoke accepts 1–10 optimizer steps")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Phase 7A does not run CPU fallbacks")
    device = torch.device(args.device)
    if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
        raise ValueError(f"Invalid requested CUDA device: {args.device}")
    run_sequence(
        args.sequence,
        device,
        smoke=args.mode == "smoke",
        smoke_steps=args.smoke_steps,
        smoke_task_count=args.smoke_task_count,
    )


if __name__ == "__main__":
    main()
