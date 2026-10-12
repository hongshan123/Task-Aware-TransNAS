"""Describe type-switch associations while controlling for final task identity."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
try:
    from .runtime_paths import storage_root
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import storage_root

load_dotenv(PROJECT_ROOT / ".env", override=False)
STORAGE_ROOT = storage_root(PROJECT_ROOT)
RESULTS_ROOT = storage_path("results") / "order_sensitivity"
REPORT_PATH = storage_path("reports") / "order_effect_analysis.md"
RESULT_REPORT_PATH = RESULTS_ROOT / "order_effect_analysis.md"


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def fixed_final_task_slope(rows: list[dict[str, str]], outcome: str) -> tuple[float, dict[str, float]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["final_task_id"]].append(row)
    numerator = 0.0
    denominator = 0.0
    task_slopes: dict[str, float] = {}
    for task_id, group in groups.items():
        xs = [number(row, "type_switch_count") for row in group]
        ys = [number(row, outcome) for row in group]
        x_bar, y_bar = mean(xs), mean(ys)
        numerator += sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        denominator += sum((x - x_bar) ** 2 for x in xs)
        task_denominator = sum((x - x_bar) ** 2 for x in xs)
        task_slopes[task_id] = (
            sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / task_denominator
            if task_denominator
            else float("nan")
        )
    return (numerator / denominator if denominator else float("nan")), task_slopes


def fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    csv_path = RESULTS_ROOT / "all_24_orders.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24 or len({row["order_id"] for row in rows}) != 24:
        raise RuntimeError(f"Expected 24 unique orders in {csv_path}, found {len(rows)}")
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_final: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[int(row["type_switch_count"])].append(row)
        by_final[row["final_task_id"]].append(row)
    observed_group_sizes = {key: len(value) for key, value in sorted(groups.items())}
    if observed_group_sizes != {1: 8, 2: 8, 3: 8}:
        raise RuntimeError(f"Switch-count groups must be 8 each, got {observed_group_sizes}")
    if any(len(group) != 6 for group in by_final.values()) or len(by_final) != 4:
        raise RuntimeError("Each final task ID must occur in exactly six orders")

    forgetting_field = "final_nonnegative_forgetting_normalized"
    learning_field = "mean_new_task_learning_normalized_gain"
    forgetting_slope, forgetting_task_slopes = fixed_final_task_slope(rows, forgetting_field)
    learning_slope, learning_task_slopes = fixed_final_task_slope(rows, learning_field)

    lines = [
        "# Type-switch effects across all 24 task orders",
        "",
        "## Design",
        "",
        "The experiment covers every permutation of the four fixed tasks. All orders use training seed 10, the same 500 train IDs and 50 dev IDs per task, the same local Qwen checkpoint, learning rate, optimizer steps, sequence/target budgets, and generation budget. Each task uses `safe_task_seed(10, task_id)`, so its row order and Torch random seed are held constant across permutations. Evaluation used dev only; the runner did not load test records. The full settings and ID lists are in `results/order_sensitivity/experiment_config.json`; stage scores and loss curves are in `results/order_sensitivity/order_metrics.json`.",
        "",
        "Type-switch count is the number of adjacent task-type changes in an order. The three groups contain 8 orders each (1, 2, or 3 switches). Each task ID is the final task in 6 orders.",
        "",
        "`mean_new_task_learning_normalized_gain` averages each task's normalized gain at the stage where that task is first learned. `final_nonnegative_forgetting_normalized` averages per-task clipped drops from the best post-acquisition score to the final stage. Task-specific metrics are normalized against each task's own base score; raw Accuracy and ROUGE-L remain separate in the CSV.",
        "",
        "## Descriptive means by switch count",
        "",
        "Values are mean (sample SD), with eight orders per group.",
        "",
        "| Type switches | Orders | Mean new-task learning gain | Mean final nonnegative forgetting | Mean final BWT |",
        "|---:|---:|---:|---:|---:|",
    ]
    for switch_count in (1, 2, 3):
        group = groups[switch_count]
        gain_values = [number(row, learning_field) for row in group]
        forgetting_values = [number(row, forgetting_field) for row in group]
        bwt_values = [number(row, "final_backward_transfer_normalized") for row in group]
        lines.append(
            f"| {switch_count} | {len(group)} | {mean(gain_values):.4f} ({sd(gain_values):.4f}) | "
            f"{mean(forgetting_values):.4f} ({sd(forgetting_values):.4f}) | "
            f"{mean(bwt_values):.4f} ({sd(bwt_values):.4f}) |"
        )

    lines.extend(
        [
            "",
            "## Controlling for final task identity",
            "",
            "The adjusted slope is an ordinary least-squares slope for `outcome ~ type_switch_count + final_task_id fixed effects`. It is calculated by demeaning switch count and outcome within each final-task ID before fitting one common slope. Since task ID determines task type, this controls both final identity and type. The coefficient is the descriptive change in normalized outcome per additional type switch among orders with the same final task.",
            "",
            f"- Final nonnegative forgetting slope: **{fmt(forgetting_slope)}** per additional type switch.",
            f"- Mean new-task learning gain slope: **{fmt(learning_slope)}** per additional type switch.",
            "",
            "| Final task ID | Type | Orders | Mean switches | Mean new-task gain | Mean forgetting | Within-task gain slope | Within-task forgetting slope |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    type_by_task = {row["final_task_id"]: row["final_task_type"] for row in rows}
    for task_id, group in sorted(by_final.items()):
        lines.append(
            f"| {task_id} | {type_by_task[task_id]} | {len(group)} | "
            f"{mean([number(row, 'type_switch_count') for row in group]):.2f} | "
            f"{mean([number(row, learning_field) for row in group]):.4f} | "
            f"{mean([number(row, forgetting_field) for row in group]):.4f} | "
            f"{fmt(learning_task_slopes[task_id])} | {fmt(forgetting_task_slopes[task_id])} |"
        )
    gain_signs = {1 if value > 0 else -1 if value < 0 else 0 for value in learning_task_slopes.values() if not math.isnan(value)}
    forgetting_signs = {1 if value > 0 else -1 if value < 0 else 0 for value in forgetting_task_slopes.values() if not math.isnan(value)}
    if gain_signs == {-1}:
        acquisition_pattern = "Mean acquisition gain has a negative within-final-task slope for all four final task IDs."
    elif gain_signs == {1}:
        acquisition_pattern = "Mean acquisition gain has a positive within-final-task slope for all four final task IDs."
    else:
        acquisition_pattern = "Mean acquisition gain has mixed within-final-task slope signs across final task IDs."
    if forgetting_signs == {-1}:
        forgetting_pattern = "Final forgetting also has a negative within-final-task slope for all four final task IDs."
    elif forgetting_signs == {1}:
        forgetting_pattern = "Final forgetting has a positive within-final-task slope for all four final task IDs."
    else:
        positive_forgetting_ids = sorted(task_id for task_id, value in forgetting_task_slopes.items() if value > 0)
        negative_forgetting_ids = sorted(task_id for task_id, value in forgetting_task_slopes.items() if value < 0)
        forgetting_pattern = (
            "Final forgetting has mixed within-final-task slopes: positive for "
            + ", ".join(positive_forgetting_ids)
            + " and negative for "
            + ", ".join(negative_forgetting_ids)
            + "."
        )
    relation_stability = (
        f"{acquisition_pattern} The adjusted acquisition slope is {fmt(learning_slope)} per additional type switch. "
        f"{forgetting_pattern} The adjusted forgetting slope is {fmt(forgetting_slope)}. "
        "The switch-group means decrease as switch count rises for both outcomes, but their standard deviations are large. "
        "This is a descriptive pattern in one seed, not evidence of a stable general effect."
    )
    lines.extend(
        [
            "",
            "## Per-order results",
            "",
            "| Order | Type sequence | Switches | Final task | Mean new-task gain | Final forgetting |",
            "|---|---|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['order_id']} | {row['task_type_order']} | {row['type_switch_count']} | "
            f"{row['final_task_id']} | {number(row, learning_field):.4f} | {number(row, forgetting_field):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion and limits",
            "",
            relation_stability,
            "",
            "The controlled slopes and switch-group means describe these exact four tasks, one training seed, and one dev split. They are not significance estimates and do not establish a causal effect of type switching. The experiment contains no test evaluation and makes no architecture or NAS claims.",
            "",
            f"Artifacts: `{RESULTS_ROOT / 'all_24_orders.csv'}`, `{RESULTS_ROOT / 'order_metrics.json'}`, `{RESULTS_ROOT / 'experiment_config.json'}`, and two final review checkpoints under `{STORAGE_ROOT / 'checkpoints' / 'order_sensitivity' / 'review'}`.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    RESULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Analysis report: {REPORT_PATH}")
    print(f"Results copy: {RESULT_REPORT_PATH}")
    print(f"Fixed-final-task slopes: forgetting={forgetting_slope:.6f}, acquisition_gain={learning_slope:.6f}")
    print(f"Type-switch group sizes: {observed_group_sizes}")


if __name__ == "__main__":
    main()
