"""Continual-learning metrics with explicit treatment of mixed task metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def normalized_gain(score: float, baseline_score: float) -> float:
    """Baseline-headroom-normalized score change for a [0, 1] task metric."""
    if not math.isfinite(score) or not math.isfinite(baseline_score):
        raise ValueError("Scores must be finite")
    if not 0.0 <= score <= 1.0 or not 0.0 <= baseline_score <= 1.0:
        raise ValueError("Scores must be in [0, 1]")
    return (score - baseline_score) / max(1.0 - baseline_score, 1e-8)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_means(values: Mapping[str, float], task_ids: Sequence[str], metric_by_task: Mapping[str, str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for task_id in task_ids:
        groups[metric_by_task[task_id]].append(values[task_id])
    return {metric: sum(scores) / len(scores) for metric, scores in sorted(groups.items())}


def continual_learning_metrics(
    scores_by_stage: Mapping[str, Mapping[str, float]],
    task_order: Sequence[str],
    metric_by_task: Mapping[str, str],
    baseline_stage: str = "baseline",
) -> dict[str, Any]:
    """Calculate stage matrices and CL summaries without averaging raw metric units.

    ``normalized_gain`` is (score - baseline) / max(1 - baseline, 1e-8).
    The legacy signed forgetting value is retained for comparisons. The new
    nonnegative forgetting clips each prior task's drop separately before
    averaging, so an improvement on another task cannot cancel that drop.
    BWT compares each earlier task with its score immediately after it was
    learned. Raw seen-task means remain grouped by the task's primary metric.
    Cross-stage AAA is the average of stage-local mean normalized gains over
    trained stages up to and including the current stage.
    """
    tasks = list(task_order)
    stages = list(scores_by_stage)
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("task_order must contain unique task IDs")
    if not stages or stages[0] != baseline_stage:
        raise ValueError(f"The first stage must be {baseline_stage!r}")
    if len(stages) != len(tasks) + 1:
        raise ValueError("Expected one baseline stage plus one stage per task")
    if set(metric_by_task) != set(tasks):
        raise ValueError("metric_by_task must contain exactly the ordered task IDs")

    raw_matrix: dict[str, dict[str, float]] = {}
    for stage in stages:
        row = scores_by_stage[stage]
        if set(row) != set(tasks):
            raise ValueError(f"Stage {stage!r} must contain a score for every task")
        clean: dict[str, float] = {}
        for task in tasks:
            score = float(row[task])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Score for {stage}/{task} must be finite and in [0, 1]")
            clean[task] = score
        raw_matrix[stage] = clean

    baseline = raw_matrix[baseline_stage]
    gain_matrix = {
        stage: {task: normalized_gain(raw_matrix[stage][task], baseline[task]) for task in tasks}
        for stage in stages
    }
    stage_metrics: dict[str, dict[str, Any]] = {}
    seen_gain_history: list[float] = []
    raw_metric_stage_history: dict[str, list[float]] = defaultdict(list)

    for stage_index, stage in enumerate(stages):
        seen = tasks[:stage_index]
        prior_tasks = tasks[: max(0, stage_index - 1)]
        current_gains = gain_matrix[stage]
        seen_gain = _mean([current_gains[task] for task in seen])
        if stage_index:
            seen_gain_history.append(float(seen_gain))

        positive_forgetting: dict[str, float] = {}
        signed_forgetting: dict[str, float] = {}
        positive_raw_forgetting: dict[str, float] = {}
        bwt_by_task: dict[str, float] = {}
        for task in prior_tasks:
            acquisition_stage_index = tasks.index(task) + 1
            previous_stages = stages[acquisition_stage_index:stage_index]
            best_prior_gain = max(gain_matrix[previous][task] for previous in previous_stages)
            signed_drop = best_prior_gain - current_gains[task]
            signed_forgetting[task] = signed_drop
            positive_forgetting[task] = max(0.0, signed_drop)
            best_prior_raw = max(raw_matrix[previous][task] for previous in previous_stages)
            positive_raw_forgetting[task] = max(0.0, best_prior_raw - raw_matrix[stage][task])
            acquisition_stage = stages[acquisition_stage_index]
            bwt_by_task[task] = current_gains[task] - gain_matrix[acquisition_stage][task]

        raw_forgetting_groups: dict[str, list[float]] = defaultdict(list)
        for task, value in positive_raw_forgetting.items():
            raw_forgetting_groups[metric_by_task[task]].append(value)
        raw_forgetting_by_metric = {
            metric: sum(values) / len(values) for metric, values in sorted(raw_forgetting_groups.items())
        }

        stage_metric_means = _group_means(raw_matrix[stage], seen, metric_by_task)
        for metric, value in stage_metric_means.items():
            raw_metric_stage_history[metric].append(value)

        stage_metrics[stage] = {
            "raw_task_primary_scores": dict(raw_matrix[stage]),
            "normalized_gain_by_task": dict(current_gains),
            "normalized_average_accuracy_all_tasks": _mean(list(current_gains.values())) or 0.0,
            "mean_normalized_gain_all_tasks": _mean(list(current_gains.values())) or 0.0,
            "legacy_aaa_normalized_seen_tasks": float(seen_gain) if seen_gain is not None else 0.0,
            "mean_seen_task_score_by_metric": stage_metric_means,
            "mean_seen_normalized_gain": float(seen_gain) if seen_gain is not None else 0.0,
            "average_forgetting_nonnegative_normalized": _mean(list(positive_forgetting.values())) or 0.0,
            "nonnegative_forgetting_by_prior_task_normalized": positive_forgetting,
            "average_forgetting_signed_normalized_legacy": _mean(list(signed_forgetting.values())) or 0.0,
            "signed_forgetting_by_prior_task_normalized_legacy": signed_forgetting,
            "average_forgetting_nonnegative_raw_by_metric": raw_forgetting_by_metric,
            "backward_transfer_by_task_normalized": bwt_by_task,
            "backward_transfer_normalized": _mean(list(bwt_by_task.values())) or 0.0,
            "tasks_seen": seen,
            "seen_task_count": len(seen),
            "stage_index_after_training": stage_index,
        }
        stage_metrics[stage]["cross_stage_aaa_normalized"] = _mean(seen_gain_history) or 0.0

    cross_stage_by_metric = {
        metric: sum(values) / len(values) for metric, values in sorted(raw_metric_stage_history.items())
    }
    return {
        "definitions": {
            "normalized_gain": "(task score - that task's baseline score) / max(1 - that task's baseline score, 1e-8)",
            "average_forgetting_nonnegative_normalized": "For each previously learned task, max(0, best earlier post-acquisition normalized gain - current normalized gain), then average. Clipping happens per task before averaging.",
            "average_forgetting_nonnegative_raw_by_metric": "The analogous nonnegative drop in each task's raw score, averaged separately by primary metric (e.g. Accuracy or ROUGE-L).",
            "backward_transfer_normalized": "For each prior task, current normalized gain minus its normalized gain immediately after acquisition, averaged over prior tasks.",
            "mean_seen_task_score_by_metric": "Raw primary scores averaged only among seen tasks sharing the same metric; Accuracy and ROUGE-L remain separate.",
            "cross_stage_aaa_normalized": "Mean of stage-local mean normalized gains on seen tasks across post-training stages through the current stage.",
            "average_forgetting_signed_normalized_legacy": "Previous implementation: mean(best earlier normalized gain - current gain) without per-task clipping; retained to quantify legacy cancellation.",
        },
        "task_order": tasks,
        "primary_metric_by_task": dict(metric_by_task),
        "stage_order": stages,
        "raw_task_primary_score_matrix": raw_matrix,
        "normalized_gain_matrix": gain_matrix,
        "stage_metrics": stage_metrics,
        "cross_stage_aaa_normalized_final": stage_metrics[stages[-1]]["cross_stage_aaa_normalized"],
        "cross_stage_seen_task_score_by_metric": cross_stage_by_metric,
    }
