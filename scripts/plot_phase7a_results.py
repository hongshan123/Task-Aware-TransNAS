"""Write a compact Phase 7A stage summary and AAA trajectory figure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "phase7a"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "phase7a"
CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
SEQUENCES = ("S1", "S2")


def collect_stage_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence in SEQUENCES:
        summary_path = RESULT_ROOT / sequence / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for condition in CONDITIONS:
            condition_data = summary["conditions"][condition]
            training = condition_data["training"]
            task_runs = list(training["task_training"].values())
            task_order = training["task_order"]
            for stage_index, (stage_key, metrics) in enumerate(
                condition_data["continual_learning_summary"].items(), start=1
            ):
                row: dict[str, object] = {
                    "sequence": sequence,
                    "condition": condition,
                    "stage_index": stage_index,
                    "stage": stage_key,
                    "learned_task": task_order[stage_index - 1],
                    "cumulative_optimizer_steps": stage_index * training["steps_per_task"],
                    "aaa": metrics["aaa"],
                    "average_forgetting": metrics["average_forgetting"],
                    "bwt": metrics["bwt"],
                    "acquisition": metrics["acquisition"],
                    "normalized_acquisition": metrics["normalized_acquisition"],
                    "stage_effective_updated_parameter_ratio": metrics["update_density"][
                        "effective_updated_parameter_ratio"
                    ],
                    "condition_wall_seconds": training["training_duration_seconds"],
                    "peak_cuda_allocated_gb": max(
                        run["peak_cuda_allocated_bytes"] for run in task_runs
                    )
                    / 1e9,
                }
                for task_id in task_order:
                    task_metrics = metrics["task_metrics"].get(task_id)
                    row[f"score_{task_id}"] = (
                        task_metrics["primary_score"] if task_metrics else ""
                    )
                    row[f"metric_{task_id}"] = (
                        task_metrics["primary_metric"] if task_metrics else ""
                    )
                    if task_metrics and task_metrics.get("teacher_forced_nll"):
                        row[f"nll_{task_id}"] = task_metrics["teacher_forced_nll"]["token_nll"]
                    else:
                        row[f"nll_{task_id}"] = ""
                rows.append(row)
    return rows


def write_stage_csv(rows: list[dict[str, object]]) -> Path:
    output = RESULT_ROOT / "pilot_stage_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_final_task_metrics() -> Path:
    rows: list[dict[str, object]] = []
    for sequence in SEQUENCES:
        summary_path = RESULT_ROOT / sequence / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        conditions = [("Baseline", summary["baseline"])]
        for condition in CONDITIONS:
            condition_data = summary["conditions"][condition]
            final_key = list(condition_data["continual_learning_summary"])[-1]
            conditions.append(
                (condition, condition_data["continual_learning_summary"][final_key]["task_metrics"])
            )
        for condition, task_metrics in conditions:
            for task_id, metrics in task_metrics.items():
                nll = metrics.get("teacher_forced_nll", {})
                rows.append(
                    {
                        "sequence": sequence,
                        "condition": condition,
                        "task_id": task_id,
                        "task_type": metrics["task_type"],
                        "primary_metric": metrics["primary_metric"],
                        "primary_score": metrics["primary_score"],
                        "rouge_l_f1": metrics.get("rouge_l_f1", ""),
                        "token_f1": metrics.get("token_f1", ""),
                        "accuracy": metrics.get("accuracy", ""),
                        "macro_f1": metrics.get("macro_f1", ""),
                        "yes_no_error_rate": metrics.get("yes_no_error_rate", ""),
                        "teacher_forced_token_nll": nll.get("token_nll", ""),
                        "evaluated_examples": metrics.get("evaluated", ""),
                    }
                )
    output = RESULT_ROOT / "pilot_final_task_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_figure(rows: list[dict[str, object]]) -> Path:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    colors = {"C0": "#666666", "C1": "#0072B2", "C2": "#E69F00", "C3": "#009E73", "C4": "#CC79A7"}
    for axis, sequence in zip(axes, SEQUENCES):
        for condition in CONDITIONS:
            points = sorted(
                (
                    row
                    for row in rows
                    if row["sequence"] == sequence and row["condition"] == condition
                ),
                key=lambda row: int(row["stage_index"]),
            )
            axis.plot(
                [int(row["stage_index"]) for row in points],
                [float(row["aaa"]) for row in points],
                marker="o",
                linewidth=2,
                markersize=5,
                label=condition,
                color=colors[condition],
            )
        axis.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
        axis.set_title(sequence)
        axis.set_xlabel("Completed tasks")
        axis.set_xticks((1, 2, 3, 4))
        axis.set_xlim(0.8, 4.2)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Average accuracy (AAA)")
    axes[1].legend(title="Condition", loc="best", frameon=False)
    figure.suptitle("Phase 7A dev performance across task streams")
    output = FIGURE_ROOT / "pilot_aaa_trajectory.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    rows = collect_stage_rows()
    print(f"Wrote {write_stage_csv(rows)} ({len(rows)} stage rows)")
    print(f"Wrote {write_final_task_metrics()}")
    print(f"Wrote {write_figure(rows)}")


if __name__ == "__main__":
    main()
