"""Summarize and audit the completed Phase 4B paired architecture runs."""

from __future__ import annotations

import json
import math
import statistics
import sys
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
    from .runtime_paths import resolve_runtime_path, storage_root
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import resolve_runtime_path, storage_root

load_dotenv(PROJECT_ROOT / ".env", override=False)
STORAGE_ROOT = storage_root(PROJECT_ROOT)
TASK_IDS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
ARCHITECTURES = ["A0", "A1", "A1-control"]
SEQUENCES = ["S1", "S2"]
SEEDS = [10, 20, 30]
T95_DF2 = 4.302652729911275


def load_runs(results_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for sequence in SEQUENCES:
        for seed in SEEDS:
            for architecture in ARCHITECTURES:
                path = results_root / "runs" / architecture / sequence / f"seed_{seed}" / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing completed run metrics: {path}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("architecture") != architecture or row.get("sequence_name") != sequence or int(row.get("seed", -1)) != seed:
                    raise ValueError(f"Run identity mismatch in {path}")
                if row.get("split") != "dev" or row.get("test_split_loaded") is not False:
                    raise ValueError(f"Unexpected split access in {path}")
                if not row.get("checkpoint_reload_audit", {}).get("passes_reload_check"):
                    raise ValueError(f"Checkpoint reload audit failed in {path}")
                checkpoint = Path(row["checkpoint_path"]) / "model.safetensors"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing final checkpoint: {checkpoint}")
                runs.append(row)
    if len(runs) != 18:
        raise ValueError(f"Expected 18 runs, found {len(runs)}")
    return runs


def final_stage(run: dict[str, Any]) -> str:
    return run["stage_order"][-1]


def number_metrics(run: dict[str, Any]) -> dict[str, float]:
    stage = final_stage(run)
    stage_scores = run["stage_scores_by_primary_metric"][stage]
    task_rows = run["task_metrics"][stage]
    output: dict[str, float] = {}
    generation_scores: list[float] = []
    classification_scores: list[float] = []
    generation_nll: list[float] = []
    classification_nll: list[float] = []
    generation_lengths: list[float] = []
    classifier_yesno: list[float] = []
    for task_id in TASK_IDS:
        item = task_rows[task_id]
        score = float(stage_scores[task_id])
        nll = float(item["teacher_forced_token_nll"]["token_nll"])
        output[f"final_score::{task_id}"] = score
        output[f"final_token_nll::{task_id}"] = nll
        behavior = item["generation_behavior"]
        output[f"exact_yes_share::{task_id}"] = float(behavior["exact_yes_share"])
        output[f"exact_no_share::{task_id}"] = float(behavior["exact_no_share"])
        output[f"yes_no_exact_share::{task_id}"] = float(behavior["yes_no_exact_share"])
        output[f"mean_generation_tokens::{task_id}"] = float(behavior["mean_generated_token_ids_including_eos"])
        if task_id in TASK_IDS[:2]:
            generation_scores.append(score)
            generation_nll.append(nll)
            generation_lengths.append(float(behavior["mean_generated_token_ids_including_eos"]))
        else:
            classification_scores.append(score)
            classification_nll.append(nll)
            classifier_yesno.append(float(behavior["yes_no_exact_share"]))
    output["final_generation_rouge_l_mean"] = statistics.mean(generation_scores)
    output["final_classification_accuracy_mean"] = statistics.mean(classification_scores)
    output["final_generation_token_nll_mean"] = statistics.mean(generation_nll)
    output["final_classification_token_nll_mean"] = statistics.mean(classification_nll)
    output["final_generation_tokens_mean"] = statistics.mean(generation_lengths)
    output["classification_yes_no_exact_share_mean"] = statistics.mean(classifier_yesno)

    acquisition = list(run["new_task_acquisition"].values())
    output["acquisition_normalized_mean"] = statistics.mean(
        float(item["normalized_acquisition_gain"]) for item in acquisition
    )
    output["acquisition_raw_mean"] = statistics.mean(float(item["raw_acquisition_gain"]) for item in acquisition)
    for task_id in TASK_IDS:
        row = next(item for item in acquisition if item["new_task"] == task_id)
        output[f"acquisition_normalized::{task_id}"] = float(row["normalized_acquisition_gain"])
        output[f"acquisition_raw::{task_id}"] = float(row["raw_acquisition_gain"])

    cl = run["continual_learning_metrics"]
    final_cl = cl["stage_metrics"][stage]
    output["cross_stage_aaa_normalized_final"] = float(final_cl["cross_stage_aaa_normalized"])
    output["average_forgetting_nonnegative_normalized"] = float(final_cl["average_forgetting_nonnegative_normalized"])
    output["backward_transfer_normalized"] = float(final_cl["backward_transfer_normalized"])
    output["total_duration_seconds"] = float(run["total_duration_seconds"])
    output["peak_cuda_allocated_gib"] = float(run["peak_cuda_allocated_bytes"]) / 1024**3
    output["peak_cuda_reserved_gib"] = float(run["peak_cuda_reserved_bytes"]) / 1024**3
    output["checkpoint_gib"] = float(run["checkpoint_reload_audit"]["checkpoint_bytes"]) / 1024**3
    return output


def gate_summary(run: dict[str, Any]) -> dict[str, float]:
    if run["architecture"] == "A0":
        return {}
    steps = 0
    layer_step_denominator = 0
    active_branch_steps = 0
    mean_abs_gate: list[float] = []
    mean_gate_grad: list[float] = []
    mean_branch: list[float] = []
    peak_abs_gate = 0.0
    for task_id in TASK_IDS:
        trace = run["training_metrics"][task_id]["gate_statistics"]
        task_steps = int(trace["recorded_steps"])
        steps += task_steps
        for layer in trace["layers"]:
            layer_step_denominator += task_steps
            active_branch_steps += int(layer["nonzero_branch_steps"])
            mean_abs_gate.append(float(layer["mean_abs_gate_value"]))
            mean_gate_grad.append(float(layer["mean_gate_gradient_l2"]))
            mean_branch.append(float(layer["mean_added_branch_rms"]))
            peak_abs_gate = max(peak_abs_gate, abs(float(layer["last_gate_value"])))
    return {
        "recorded_optimizer_steps": float(steps),
        "branch_active_layer_steps_share": active_branch_steps / max(1, layer_step_denominator),
        "mean_layer_task_abs_gate": statistics.mean(mean_abs_gate),
        "max_final_abs_gate_across_layers": peak_abs_gate,
        "mean_layer_task_gate_gradient_l2": statistics.mean(mean_gate_grad),
        "mean_layer_task_branch_rms": statistics.mean(mean_branch),
    }


def mean_sd(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": math.nan, "sd": math.nan}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def interval(values: list[float]) -> dict[str, float | int]:
    stats = mean_sd(values)
    n = int(stats["n"])
    mean = float(stats["mean"])
    sd = float(stats["sd"])
    tcrit = T95_DF2 if n == 3 else (T95_DF5 if n == 6 else 0.0)
    margin = tcrit * sd / math.sqrt(n) if n > 1 else math.nan
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - margin, "ci95_high": mean + margin}


def paired_effects(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    index = {(r["sequence_name"], int(r["seed"]), r["architecture"]): r for r in rows}
    result: dict[str, Any] = {}
    for candidate, reference in [("A1", "A0"), ("A1-control", "A0"), ("A1", "A1-control")]:
        contrast = f"{candidate} - {reference}"
        contrast_result: dict[str, Any] = {}
        for metric in metric_names:
            per_order: dict[str, list[float]] = {}
            by_seed: dict[int, list[float]] = defaultdict(list)
            for sequence in SEQUENCES:
                deltas = []
                for seed in SEEDS:
                    delta = float(index[(sequence, seed, candidate)]["numeric_metrics"][metric]) - float(
                        index[(sequence, seed, reference)]["numeric_metrics"][metric]
                    )
                    deltas.append(delta)
                    by_seed[seed].append(delta)
                per_order[sequence] = deltas
            seed_averaged = [statistics.mean(by_seed[seed]) for seed in SEEDS]
            overall = interval(seed_averaged)
            overall["unit"] = "3 independent seeds; each seed averages its S1 and S2 paired deltas"
            contrast_result[metric] = {
                "overall_seed_averaged": overall,
                "by_order": {seq: interval(values) for seq, values in per_order.items()},
                "order_interaction_s2_minus_s1": interval(
                    [per_order["S2"][i] - per_order["S1"][i] for i in range(len(SEEDS))]
                ),
            }
        result[contrast] = contrast_result
    return result


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def effect_cell(stats: dict[str, Any], digits: int = 3) -> str:
    return (
        f"{fmt(stats['mean'], digits)} (SD {fmt(stats['sd'], digits)}; "
        f"95% CI {fmt(stats['ci95_low'], digits)} to {fmt(stats['ci95_high'], digits)})"
    )


def make_report(summary: dict[str, Any], report_path: Path, results_root: Path) -> None:
    runs = summary["runs"]
    overall = summary["overall_architecture_summaries"]
    effects = summary["paired_effects"]
    lines = [
        "# Phase 4B Controlled Attention Comparison",
        "",
        "## Scope and completion",
        "",
        f"All {len(runs)} planned runs completed: three architectures x two orders x three seeds. Every run used the same local Qwen3-0.6B source weights, dev-only evaluation, and a final checkpoint reload audit. The test split was not loaded.",
        "",
        "All task-by-stage metrics, predictions, teacher-forced NLL, generation behavior, gate traces, timing, VRAM, parameter counts, and reload records are retained under the configured storage root at `results/phase4b/`.",
        "",
        "## Main outcomes",
        "",
        "Values are means +/- sample SD across six runs (three seeds in each sequence). Accuracy and ROUGE-L stay in separate columns.",
        "",
        "| Architecture | Final generation ROUGE-L | Final classification accuracy | Mean normalized acquisition gain | Final cross-stage AAA | Nonnegative forgetting | Backward transfer | Peak allocated VRAM (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary_metrics = [
        "final_generation_rouge_l_mean", "final_classification_accuracy_mean",
        "acquisition_normalized_mean", "cross_stage_aaa_normalized_final",
        "average_forgetting_nonnegative_normalized", "backward_transfer_normalized",
        "peak_cuda_allocated_gib",
    ]
    for architecture in ARCHITECTURES:
        cells = [f"{fmt(overall[architecture][m]['mean'])} +/- {fmt(overall[architecture][m]['sd'])}" for m in summary_metrics]
        lines.append(f"| {architecture} | " + " | ".join(cells) + " |")

    lines.extend([
        "", "## Paired architecture effects", "",
        "Effects are candidate minus reference. Overall intervals first average the S1 and S2 paired deltas within each seed, then summarize across three seeds (nominal paired t 95% interval, df=2). Order-specific intervals use three paired seeds per order. These are descriptive intervals for a small exploratory study.", "",
    ])
    comparison_metrics = [
        "acquisition_normalized_mean", "final_generation_rouge_l_mean",
        "final_classification_accuracy_mean", "cross_stage_aaa_normalized_final",
        "average_forgetting_nonnegative_normalized", "backward_transfer_normalized",
        "final_generation_token_nll_mean", "final_classification_token_nll_mean",
    ]
    for contrast in ["A1 - A0", "A1-control - A0", "A1 - A1-control"]:
        lines.extend([
            f"### {contrast}", "",
            "| Metric | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |",
            "|---|---:|---:|---:|---:|",
        ])
        for metric in comparison_metrics:
            item = effects[contrast][metric]
            s1, s2 = item["by_order"]["S1"], item["by_order"]["S2"]
            interaction = item["order_interaction_s2_minus_s1"]
            lines.append(
                f"| `{metric}` | {effect_cell(item['overall_seed_averaged'])} | {fmt(s1['mean'])} | {fmt(s2['mean'])} | "
                f"{fmt(interaction['mean'])} ({fmt(interaction['ci95_low'])} to {fmt(interaction['ci95_high'])}) |"
            )
        lines.append("")

    lines.extend([
        "## Per-task final metrics and acquisition", "",
        "Raw acquisition is reported in the task's own primary-metric units. Normalized gain divides by the task's baseline headroom.", "",
        "| Architecture | Task | Final primary score (mean +/- SD) | Normalized acquisition gain | Raw acquisition | Token NLL | Exact Yes share | Exact No share | Mean generated tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for architecture in ARCHITECTURES:
        subset = [r["numeric_metrics"] for r in runs if r["architecture"] == architecture]
        for task_id in TASK_IDS:
            def task_stats(prefix: str) -> dict[str, Any]:
                return mean_sd([float(r[f"{prefix}::{task_id}"]) for r in subset])
            score = task_stats("final_score")
            gain = task_stats("acquisition_normalized")
            raw_gain = task_stats("acquisition_raw")
            nll = task_stats("final_token_nll")
            yes = task_stats("exact_yes_share")
            no = task_stats("exact_no_share")
            length = task_stats("mean_generation_tokens")
            lines.append(
                f"| {architecture} | `{task_id}` | {fmt(score['mean'])} +/- {fmt(score['sd'])} | "
                f"{fmt(gain['mean'])} +/- {fmt(gain['sd'])} | {fmt(raw_gain['mean'])} +/- {fmt(raw_gain['sd'])} | "
                f"{fmt(nll['mean'])} +/- {fmt(nll['sd'])} | {fmt(yes['mean'])} +/- {fmt(yes['sd'])} | "
                f"{fmt(no['mean'])} +/- {fmt(no['sd'])} | {fmt(length['mean'])} +/- {fmt(length['sd'])} |"
            )

    outcomes = ["final_score", "acquisition_normalized", "acquisition_raw"]
    for contrast in ["A1 - A0", "A1-control - A0", "A1 - A1-control"]:
        lines.extend([
            "", f"### Task-specific paired effects: {contrast}", "",
            "| Task | Outcome | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for task_id in TASK_IDS:
            for outcome in outcomes:
                metric = f"{outcome}::{task_id}"
                item = effects[contrast][metric]
                s1, s2 = item["by_order"]["S1"], item["by_order"]["S2"]
                interaction = item["order_interaction_s2_minus_s1"]
                lines.append(
                    f"| `{task_id}` | `{outcome}` | {effect_cell(item['overall_seed_averaged'])} | {fmt(s1['mean'])} | {fmt(s2['mean'])} | "
                    f"{fmt(interaction['mean'])} ({fmt(interaction['ci95_low'])} to {fmt(interaction['ci95_high'])}) |"
                )

    lines.extend([
        "", "## Gate and branch activity", "",
        "| Architecture | Runs | Optimizer updates traced | Branch-active layer-step share | Mean absolute gate | Mean clipped gate-gradient L2 | Mean branch RMS | Maximum final absolute gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for architecture in ["A1", "A1-control"]:
        gate_rows = [r["gate_summary"] for r in runs if r["architecture"] == architecture]
        def gate_stats(key: str) -> dict[str, Any]:
            return mean_sd([float(g[key]) for g in gate_rows])
        total_steps = sum(int(g["recorded_optimizer_steps"]) for g in gate_rows)
        share = gate_stats("branch_active_layer_steps_share")
        gate = gate_stats("mean_layer_task_abs_gate")
        grad = gate_stats("mean_layer_task_gate_gradient_l2")
        branch = gate_stats("mean_layer_task_branch_rms")
        peak = max(float(g["max_final_abs_gate_across_layers"]) for g in gate_rows)
        lines.append(
            f"| {architecture} | 6 | {total_steps} | {fmt(share['mean'] * 100, 2)}% | {fmt(gate['mean'], 7)} | "
            f"{fmt(grad['mean'], 7)} | {fmt(branch['mean'], 7)} | {fmt(peak, 7)} |"
        )

    acq = effects["A1 - A0"]["acquisition_normalized_mean"]["overall_seed_averaged"]
    acq_control = effects["A1 - A1-control"]["acquisition_normalized_mean"]["overall_seed_averaged"]
    forgetting = effects["A1 - A0"]["average_forgetting_nonnegative_normalized"]["overall_seed_averaged"]
    forgetting_control = effects["A1 - A1-control"]["average_forgetting_nonnegative_normalized"]["overall_seed_averaged"]
    order = effects["A1 - A0"]["cross_stage_aaa_normalized_final"]["order_interaction_s2_minus_s1"]
    a1_gate = [r["gate_summary"] for r in runs if r["architecture"] == "A1"]
    control_gate = [r["gate_summary"] for r in runs if r["architecture"] == "A1-control"]
    lines.extend([
        "", "## Answers to the four study questions", "",
        f"1. **New-task acquisition.** A1-A0 mean normalized acquisition delta: {fmt(acq['mean'])} (95% CI {fmt(acq['ci95_low'])} to {fmt(acq['ci95_high'])}); A1-A1-control: {fmt(acq_control['mean'])} ({fmt(acq_control['ci95_low'])} to {fmt(acq_control['ci95_high'])}). Task-specific acquisition deltas are listed above.",
        f"2. **Retention.** A1-A0 change in final nonnegative forgetting: {fmt(forgetting['mean'])} (95% CI {fmt(forgetting['ci95_low'])} to {fmt(forgetting['ci95_high'])}); A1-A1-control: {fmt(forgetting_control['mean'])} ({fmt(forgetting_control['ci95_low'])} to {fmt(forgetting_control['ci95_high'])}). Lower forgetting is better; backward transfer and final task scores are also shown.",
        f"3. **Order dependence.** S2-S1 interaction in the A1-A0 final cross-stage AAA effect: {fmt(order['mean'])} (95% CI {fmt(order['ci95_low'])} to {fmt(order['ci95_high'])}). See the per-order effects and task-specific order interactions above.",
        f"4. **Gate participation.** Branch-active layer-step share: {fmt(statistics.mean(x['branch_active_layer_steps_share'] for x in a1_gate) * 100, 2)}% for A1 and {fmt(statistics.mean(x['branch_active_layer_steps_share'] for x in control_gate) * 100, 2)}% for A1-control across 400 updates/run and 28 layers. Nonzero branch RMS indicates path activity; it does not establish that gate changes caused metric differences.",
        "", "## Limits", "",
        "This is a small exploratory comparison with three independent seeds. The t intervals describe uncertainty in these runs and cannot establish broad generalization. The task-stream manifest defines a 50-row dev partition per task; every row was evaluated at every stage. No test data were used. The equal-token reference was specified but not run, and A2-lite was not formally trained.",
        "", "## Reproducibility artifacts", "",
        f"- Aggregated machine-readable summary: `{results_root / 'comparison_summary.json'}`.",
        f"- Per-run stage metrics, predictions, NLL, gate traces, and checkpoint audits: `{results_root / 'runs'}`.",
        f"- Final full-parameter checkpoints: `{results_root / 'checkpoints'}`.",
        "- Frozen protocol: `reports/phase4b_training_protocol.md`.", "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results_root = (
        resolve_runtime_path(sys.argv[1], project_root=PROJECT_ROOT)
        if len(sys.argv) > 1
        else storage_path("results") / "phase4b"
    )
    report_path = storage_path("reports") / "phase4b_comparison.md"
    manifest = json.loads((results_root / "formal_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("completed") is not True or int(manifest.get("planned_run_count", -1)) != 18 or len(manifest.get("completed_runs", [])) != 18:
        raise ValueError("Formal run manifest does not confirm all 18 planned runs")
    if manifest.get("test_split_loaded") is not False:
        raise ValueError("Formal run manifest indicates test split access")
    run_rows = load_runs(results_root)
    audit = json.loads((results_root / "audit" / "architecture_audit.json").read_text(encoding="utf-8"))
    if not audit.get("base_weights_same_across_A0_A1_A1_control"):
        raise ValueError("Architecture audit did not confirm shared base weights")

    # Verify matched IDs and schedules within every architecture comparison pair.
    pair_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        pair_groups[(row["sequence_name"], int(row["seed"]))].append(row)
    for key, group in pair_groups.items():
        if len(group) != 3:
            raise ValueError(f"Expected three architecture runs for pair {key}, found {len(group)}")
        baseline = next(row for row in group if row["architecture"] == "A0")
        by_architecture = {row["architecture"]: row for row in group}
        if set(by_architecture) != set(ARCHITECTURES):
            raise ValueError(f"Architecture set mismatch in paired run {key}: {set(by_architecture)}")
        for row in group:
            for field in ["task_order", "source_model_sha256", "train_instance_id_sha256_by_task", "dev_instance_id_sha256_by_task"]:
                if row[field] != baseline[field]:
                    raise ValueError(f"Paired run mismatch for {field} in {key}")
            if row["training_protocol"] != baseline["training_protocol"]:
                raise ValueError(f"Training protocol mismatch in {key}")
            if row["stage_order"] != baseline["stage_order"]:
                raise ValueError(f"Evaluation stage order mismatch in {key}")
            for task_id in TASK_IDS:
                reference_task = baseline["training_metrics"][task_id]
                paired_task = row["training_metrics"][task_id]
                for field in ["sample_order_sha256", "supervised_target_tokens", "supervised_target_tokens_per_step"]:
                    if paired_task[field] != reference_task[field]:
                        raise ValueError(f"Paired training mismatch for {field} in {key}/{task_id}")
                expected_steps = int(row["training_protocol"]["steps_by_task"][task_id])
                if int(paired_task["optimizer_steps"]) != expected_steps:
                    raise ValueError(f"Incorrect optimizer step count in {key}/{task_id}")
                if row["architecture"] != "A0":
                    gate_stats = paired_task["gate_statistics"]
                    if int(gate_stats["recorded_steps"]) != expected_steps or len(gate_stats["layers"]) != 28:
                        raise ValueError(f"Incomplete gate trace summary in {key}/{task_id}")
                    trace_path = Path(paired_task["gate_trace_file"])
                    with trace_path.open(encoding="utf-8") as handle:
                        trace = [json.loads(line) for line in handle if line.strip()]
                    if len(trace) != expected_steps:
                        raise ValueError(f"Gate trace row count mismatch in {key}/{task_id}")
                    for step_row in trace:
                        if len(step_row["gate_value_after_update"]) != 28 or len(step_row["gate_gradient_l2_after_global_clip"]) != 28 or len(step_row["added_branch_rms_from_last_microbatch"]) != 28:
                            raise ValueError(f"Gate trace layer count mismatch in {key}/{task_id}")
            for stage in row["stage_order"]:
                for task_id in TASK_IDS:
                    evaluated = int(row["task_metrics"][stage][task_id]["metrics"]["evaluated"])
                    if evaluated != 50:
                        raise ValueError(f"Expected complete 50-row dev evaluation, got {evaluated} in {key}/{stage}/{task_id}")

    metric_names = [
        "acquisition_normalized_mean", "final_generation_rouge_l_mean",
        "final_classification_accuracy_mean", "final_generation_token_nll_mean",
        "final_classification_token_nll_mean", "final_generation_tokens_mean",
        "classification_yes_no_exact_share_mean", "cross_stage_aaa_normalized_final",
        "average_forgetting_nonnegative_normalized", "backward_transfer_normalized",
        "total_duration_seconds", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib",
    ]
    for task_id in TASK_IDS:
        metric_names.extend([
            f"final_score::{task_id}", f"final_token_nll::{task_id}",
            f"acquisition_normalized::{task_id}", f"acquisition_raw::{task_id}",
            f"exact_yes_share::{task_id}", f"exact_no_share::{task_id}",
            f"yes_no_exact_share::{task_id}", f"mean_generation_tokens::{task_id}",
        ])
    run_summaries: list[dict[str, Any]] = []
    for row in run_rows:
        run_summaries.append({
            "architecture": row["architecture"], "sequence_name": row["sequence_name"], "seed": row["seed"],
            "task_order": row["task_order"], "metric_file": str(results_root / "runs" / row["architecture"] / row["sequence_name"] / f"seed_{row['seed']}" / "metrics.json"),
            "numeric_metrics": number_metrics(row), "gate_summary": gate_summary(row),
        })
    overall: dict[str, Any] = {}
    by_order: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        subset = [r for r in run_summaries if r["architecture"] == architecture]
        overall[architecture] = {metric: mean_sd([float(r["numeric_metrics"][metric]) for r in subset]) for metric in metric_names}
        by_order[architecture] = {
            sequence: {metric: mean_sd([float(r["numeric_metrics"][metric]) for r in subset if r["sequence_name"] == sequence]) for metric in metric_names}
            for sequence in SEQUENCES
        }
    effects = paired_effects(run_summaries, metric_names)
    output = {
        "schema": "task-aware-transnas.phase4b-comparison.v1",
        "planned_runs": 18,
        "completed_runs": len(run_summaries),
        "split": "dev",
        "test_split_loaded": False,
        "architectures": ARCHITECTURES,
        "sequences": SEQUENCES,
        "seeds": SEEDS,
        "metric_names": metric_names,
        "runs": run_summaries,
        "overall_architecture_summaries": overall,
        "architecture_summaries_by_order": by_order,
        "paired_effects": effects,
    }
    (results_root / "comparison_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    make_report(output, report_path, results_root)
    print(f"Validated {len(run_summaries)} paired Phase 4B runs.")
    print(f"Wrote {results_root / 'comparison_summary.json'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
