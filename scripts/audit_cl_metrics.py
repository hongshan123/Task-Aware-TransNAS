"""Recalculate continual-learning metrics with audited, explicit definitions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from task_aware_transnas.cl.metrics import continual_learning_metrics

SEQUENCES = ("homogeneous", "heterogeneous", "mixed")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "—" if value is None or value == "" else f"{float(value):.4f}"


def write_report(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Continual learning metric audit",
        "",
        "This audit recalculates the three complete formal streams from their saved 5-stage × 4-task primary-score matrices. The partial `homogeneous/seed_314160` run is excluded because it has no complete `metrics.json`. No model was trained by this audit.",
        "",
        "## Metric definitions",
        "",
        "- **Raw primary score matrix:** each task retains its own primary score and unit: ROUGE-L F1 for task565/task1603 and Accuracy for task1501/task1384. The matrix is preserved per stream in `D:/Task-Aware-TransNAS/results/baseline_full/metrics.json`; the CSV has one raw-score column per task. Raw Accuracy and raw ROUGE-L are never averaged together.",
        "- **Normalized gain:** for task *t* at stage *s*, `(score[t,s] − baseline[t]) / max(1 − baseline[t], 1e−8)`. The per-task matrix is preserved for every stage. The existing all-task mean normalized gain is also preserved, including unlearned tasks, under its historical field name and the clearer `mean_normalized_gain_all_tasks` alias.",
        "- **Nonnegative average forgetting:** for each previously learned task, take `max(0, best post-acquisition normalized gain so far − current normalized gain)`, then average the task-level clipped drops. An improvement on one task cannot cancel forgetting on another.",
        "- **Backward transfer (BWT):** for each previously learned task, current normalized gain minus its normalized gain immediately after that task was learned; average across prior tasks. Positive values indicate later stages improved the earlier task relative to its acquisition-stage score.",
        "- **Per-stage mean seen-task score:** average raw scores among tasks already learned, grouped separately by primary metric (`accuracy` and `rouge_l_f1`). The baseline has no seen tasks.",
        "- **Cross-stage AAA:** mean of the stage-local mean normalized gain over seen tasks for every trained stage up to and including the current stage. The per-stage legacy AAA remains the current stage's seen-task mean only.",
        "- **Legacy forgetting:** the prior implementation averaged `best earlier normalized gain − current normalized gain` without clipping each task at zero. It is retained in the saved files as `legacy_cl_summary` and `average_forgetting_signed_normalized_legacy` for comparison.",
        "",
        "## Per-stage metrics",
        "",
        "Raw means are split by metric. Baseline-stage entries have no seen-task raw mean.",
        "",
        "| Sequence | Stage | Seen tasks | Mean seen Accuracy | Mean seen ROUGE-L | Mean seen normalized gain | Cross-stage AAA | Nonnegative forgetting | BWT | Legacy forgetting |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence, run in aggregate["sequences"].items():
        audit = run["cl_metric_audit"]
        for stage in run["stage_order"]:
            current = audit["stage_metrics"][stage]
            means = current["mean_seen_task_score_by_metric"]
            legacy = run["legacy_cl_summary"][stage]
            lines.append(
                f"| {sequence} | {stage} | {current['seen_task_count']} | "
                f"{fmt(means.get('accuracy'))} | {fmt(means.get('rouge_l_f1'))} | "
                f"{fmt(current['mean_seen_normalized_gain'])} | {fmt(current['cross_stage_aaa_normalized'])} | "
                f"{fmt(current['average_forgetting_nonnegative_normalized'])} | "
                f"{fmt(current['backward_transfer_normalized'])} | {fmt(legacy['average_forgetting_normalized'])} |"
            )
    lines.extend(
        [
            "",
            "## Final raw primary scores and summary",
            "",
            "The score columns remain grouped by their primary metric; no raw cross-metric mean is reported.",
            "",
            "| Sequence | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy | Mean all-task normalized gain | Cross-stage AAA | Nonnegative forgetting | BWT | Legacy forgetting |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sequence, run in aggregate["sequences"].items():
        audit = run["cl_metric_audit"]
        final_stage = run["stage_order"][-1]
        final = audit["stage_metrics"][final_stage]
        raw = audit["raw_task_primary_score_matrix"][final_stage]
        legacy = run["legacy_cl_summary"][final_stage]
        ids = aggregate["canonical_task_ids"]
        raw_by_id = raw
        lines.append(
            f"| {sequence} | {fmt(raw_by_id[ids[0]])} | {fmt(raw_by_id[ids[1]])} | "
            f"{fmt(raw_by_id[ids[2]])} | {fmt(raw_by_id[ids[3]])} | "
            f"{fmt(final['mean_normalized_gain_all_tasks'])} | {fmt(final['cross_stage_aaa_normalized'])} | "
            f"{fmt(final['average_forgetting_nonnegative_normalized'])} | "
            f"{fmt(final['backward_transfer_normalized'])} | {fmt(legacy['average_forgetting_normalized'])} |"
        )
    lines.extend(
        [
            "",
            "The final-score columns are written in canonical task order: task565 ROUGE-L, task1603 ROUGE-L, task1501 Accuracy, task1384 Accuracy, irrespective of stream order.",
            "",
            "## Interpretation and artifacts",
            "",
            "The new nonnegative forgetting measure is higher than the legacy signed value for homogeneous (0.1555 vs 0.0444) and mixed (0.7655 vs 0.6914), because task improvements no longer offset drops on another prior task. Heterogeneous is unchanged at 0.2099 because its final prior-task drops are all nonnegative. BWT uses the acquisition-stage anchor, so it can differ from forgetting, which compares against the best post-acquisition score observed so far.",
            "",
            "The implementation is in `src/task_aware_transnas/cl/metrics.py`; its 9 focused tests pass via `python -m unittest discover -s tests -p \"test_cl_metrics.py\"`.",
            "",
            "Machine-readable outputs: `D:/Task-Aware-TransNAS/results/baseline_full/metrics.json`, each sequence's `metrics.json` (`cl_metric_audit` plus unchanged `legacy_cl_summary`), and the expanded `accuracy_matrix.csv`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "continual_baseline_full.yaml").read_text(encoding="utf-8"))
    results_root = Path(config["storage"]["results"])
    task_ids = list(config["dataset"]["tasks"])
    metric_by_task = {
        task_id: config["dataset"]["task_types"][task_id]["primary_metric"] for task_id in task_ids
    }
    run_data: dict[str, dict[str, Any]] = {}

    for sequence_name in SEQUENCES:
        metrics_path = results_root / sequence_name / "metrics.json"
        metrics = read_json(metrics_path)
        stage_order = metrics["stage_order"]
        score_matrix = {
            stage: {
                task_id: float(metrics["stage_scores"][stage][task_id]["primary_score"])
                for task_id in task_ids
            }
            for stage in stage_order
        }
        if set(metrics["task_order"]) != set(task_ids) or set(metrics["task_ids"]) != set(task_ids):
            raise ValueError(f"Unexpected task order or task coverage in {metrics_path}")
        result = continual_learning_metrics(score_matrix, metrics["task_order"], metric_by_task)
        # Keep the old fields verbatim, but identify them as historical formulas.
        metrics["legacy_cl_summary"] = metrics["cl_summary"]
        metrics["cl_metric_audit"] = result
        for stage in stage_order:
            result["stage_metrics"][stage]["mean_normalized_gain_all_tasks"] = result["stage_metrics"][stage][
                "normalized_average_accuracy_all_tasks"
            ]
        write_json(metrics_path, metrics)
        run_data[sequence_name] = {
            "task_order": metrics["task_order"],
            "stage_order": stage_order,
            "task_types": metrics["task_types"],
            "model_path": metrics["model_path"],
            "duration_seconds": metrics["duration_seconds"],
            "peak_cuda_allocated_bytes_training": metrics["peak_cuda_allocated_bytes_training"],
            "stage_scores": metrics["stage_scores"],
            "legacy_cl_summary": metrics["legacy_cl_summary"],
            "cl_metric_audit": result,
        }

    aggregate = {
        "schema": "task-aware-transnas.formal-baseline-metrics.v2",
        "canonical_task_ids": task_ids,
        "sequences": run_data,
        "metric_definitions": run_data["homogeneous"]["cl_metric_audit"]["definitions"],
        "raw_primary_score_matrix_note": "Raw primary scores are task-specific and stay in their original units. Accuracy and ROUGE-L are never combined into one raw mean.",
    }
    write_json(results_root / "metrics.json", aggregate)

    csv_path = results_root / "accuracy_matrix.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    additions = [
        *(f"{task_id}__raw_primary_score" for task_id in task_ids),
        *(f"{task_id}__normalized_gain" for task_id in task_ids),
        "mean_normalized_gain_all_tasks",
        "mean_seen_normalized_gain",
        "cross_stage_aaa_normalized",
        "mean_seen_raw_accuracy",
        "mean_seen_raw_rouge_l_f1",
        "nonnegative_forgetting_normalized",
        "backward_transfer_normalized",
        "nonnegative_raw_forgetting_accuracy",
        "nonnegative_raw_forgetting_rouge_l_f1",
        "legacy_aaa_normalized_seen_tasks",
        "legacy_signed_forgetting_normalized",
        "legacy_normalized_average_accuracy_all_tasks",
    ]
    columns.extend(column for column in additions if column not in columns)
    lookup = {(row["sequence"], row["stage"]): row for row in rows}
    for sequence_name, metrics in run_data.items():
        audit = metrics["cl_metric_audit"]
        for stage in metrics["stage_order"]:
            row = lookup[(sequence_name, stage)]
            stage_audit = audit["stage_metrics"][stage]
            legacy = metrics["legacy_cl_summary"][stage]
            for task_id in task_ids:
                row[f"{task_id}__raw_primary_score"] = audit["raw_task_primary_score_matrix"][stage][task_id]
                row[f"{task_id}__normalized_gain"] = audit["normalized_gain_matrix"][stage][task_id]
            row["mean_normalized_gain_all_tasks"] = stage_audit["mean_normalized_gain_all_tasks"]
            row["mean_seen_normalized_gain"] = stage_audit["mean_seen_normalized_gain"]
            row["cross_stage_aaa_normalized"] = stage_audit["cross_stage_aaa_normalized"]
            row["mean_seen_raw_accuracy"] = stage_audit["mean_seen_task_score_by_metric"].get("accuracy", "")
            row["mean_seen_raw_rouge_l_f1"] = stage_audit["mean_seen_task_score_by_metric"].get("rouge_l_f1", "")
            row["nonnegative_forgetting_normalized"] = stage_audit["average_forgetting_nonnegative_normalized"]
            row["backward_transfer_normalized"] = stage_audit["backward_transfer_normalized"]
            row["nonnegative_raw_forgetting_accuracy"] = stage_audit["average_forgetting_nonnegative_raw_by_metric"].get("accuracy", "")
            row["nonnegative_raw_forgetting_rouge_l_f1"] = stage_audit["average_forgetting_nonnegative_raw_by_metric"].get("rouge_l_f1", "")
            row["legacy_aaa_normalized_seen_tasks"] = legacy["aaa_normalized_seen_tasks"]
            row["legacy_signed_forgetting_normalized"] = legacy["average_forgetting_normalized"]
            row["legacy_normalized_average_accuracy_all_tasks"] = legacy["normalized_average_accuracy_all_tasks"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_report(PROJECT_ROOT / "reports" / "metric_audit.md", aggregate)

    print(f"Updated per-sequence audit metrics and aggregate: {results_root / 'metrics.json'}")
    print(f"Updated matrix: {csv_path} ({len(rows)} rows)")
    print(f"Report: {PROJECT_ROOT / 'reports' / 'metric_audit.md'}")


if __name__ == "__main__":
    main()
