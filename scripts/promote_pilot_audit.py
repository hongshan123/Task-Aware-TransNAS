"""Make task-aware audit scores the canonical evaluation fields in the pilot artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .runtime_paths import storage_root
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import storage_root


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    audit_script = PROJECT_ROOT / "scripts" / "audit_pilot.py"
    spec = importlib.util.spec_from_file_location("audit_pilot", audit_script)
    audit_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(audit_module)

    data_root = storage_root(PROJECT_ROOT)
    pilot_root = data_root / "results" / "baseline_pilot"
    audit_root = data_root / "results" / "pilot_audit"
    run = read_json(pilot_root / "metrics.json")
    audit = read_json(audit_root / "metrics.json")
    if "legacy_pilot_scores" not in run:
        run["legacy_pilot_scores"] = {
            "description": "Original pilot normalized exact match; task611 is invalid because it compared whole option text to a letter, and generation EM is not a sufficient metric.",
            "accuracy_matrix": run.get("accuracy_matrix", {}),
            "evaluations": run.get("evaluations", {}),
            "task_aaa_and_forgetting": {
                task: {
                    key: run["task_training"].get(task, {}).get(key)
                    for key in ("average_accuracy_learned_tasks", "average_forgetting", "current_task_accuracy")
                }
                for task in run.get("task_ids", [])
            },
        }

    stages = audit["stage_ids"]
    tasks = audit["task_ids"]
    run["evaluation_protocol_version"] = "pilot_audit_v2_task_appropriate_metrics"
    run["audit_metrics_path"] = str(audit_root / "metrics.json")
    run["audit_predictions_path"] = str(audit_root / "all_predictions.jsonl")
    run["audit_sample_path"] = str(audit_root / "audit_samples.jsonl")
    run["task_primary_metrics"] = {
        task: audit["task_metrics"]["baseline"][task]["primary_metric"] for task in tasks
    }
    run["task_performance_matrix"] = audit["task_metrics"]
    run["cl_summary_normalized"] = audit["cl_summary"]
    run["accuracy_matrix"] = {
        stage: {task: audit["task_metrics"][stage][task]["primary_score"] for task in tasks}
        for stage in stages
    }
    run["evaluations"] = {
        stage: {
            task: {
                **audit["task_metrics"][stage][task],
                "prediction_file": str(audit_root / "all_predictions.jsonl"),
            }
            for task in tasks
        }
        for stage in stages
    }
    for index, task in enumerate(tasks, start=1):
        stage = stages[index]
        task_summary = audit["cl_summary"][stage]
        metric = audit["task_metrics"][stage][task]
        train_metrics = run["task_training"][task]
        train_metrics["primary_metric"] = metric["primary_metric"]
        if metric["task_type"] == "classification":
            train_metrics["current_task_accuracy"] = metric["accuracy"]
        else:
            train_metrics.pop("current_task_accuracy", None)
        train_metrics["current_task_performance"] = metric["primary_score"]
        train_metrics["current_task_primary_score"] = metric["primary_score"]
        train_metrics["current_task_metric_details"] = metric
        train_metrics["aaa"] = task_summary["aaa_normalized_seen_tasks"]
        train_metrics["average_accuracy_learned_tasks"] = task_summary["aaa_normalized_seen_tasks"]
        train_metrics["average_accuracy_all_tasks_normalized"] = task_summary["normalized_average_accuracy_all_tasks"]
        train_metrics["average_forgetting"] = task_summary["average_forgetting_normalized"]
        train_metrics["average_forgetting_normalized"] = task_summary["average_forgetting_normalized"]
        train_metrics["forgetting_by_prior_task_normalized"] = task_summary["forgetting_by_prior_task_normalized"]
    run["generation_diagnostics"] = audit["generation_diagnostics"]
    run["input_truncation_diagnostics_by_split"] = audit["input_truncation_diagnostics_by_split"]
    run["corrected_metric_definitions"] = audit["metric_definitions"]
    write_json(pilot_root / "metrics.json", run)

    audit_module.write_csv_matrix(
        pilot_root / "accuracy_matrix.csv",
        stages,
        tasks,
        audit["task_metrics"],
        audit["cl_summary"],
    )
    tracked_report = PROJECT_ROOT / "reports" / "pilot_evaluation_audit.md"
    report_text = tracked_report.read_text(encoding="utf-8")
    training_seconds = sum(run["task_training"][task]["duration_seconds"] for task in tasks)
    header = [
        "# Pilot run report — corrected evaluation",
        "",
        f"Training: full-parameter Qwen3-0.6B SFT; {run['total_parameters']:,} parameters; seed {run['seed']}; "
        f"20 optimizer steps/task; task training time {training_seconds:.1f}s; "
        f"peak training allocation {run['peak_cuda_allocated_bytes_training'] / (1024 ** 3):.2f} GiB.",
        "",
        "Evaluation values below supersede the original Exact Match-only task611/task1590 report. Original raw metrics remain in metrics.json under legacy_pilot_scores.",
        "",
    ]
    (pilot_root / "run_report.md").write_text("\n".join(header) + report_text, encoding="utf-8")
    print(f"Promoted corrected pilot metrics to {pilot_root}")


if __name__ == "__main__":
    main()
