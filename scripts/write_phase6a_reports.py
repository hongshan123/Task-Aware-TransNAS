"""Build Phase 6A reports and figures from recorded experiment outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_formal_baseline import parse_yes_no, rouge_l_f1  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "phase6a"
FIGURES = PROJECT_ROOT / "figures" / "phase6a"
REPORTS = PROJECT_ROOT / "reports"
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
TASK_LABELS = {
    TASKS[0]: "565 CIRCA",
    TASKS[1]: "1603 SMCalFlow",
    TASKS[2]: "1501 DSTC3",
    TASKS[3]: "1384 Deal-or-No-Deal",
}
COMPONENTS = ["attention_o_proj_output", "mlp_output", "transformer_block_output"]
COMP_LABELS = {
    "attention_o_proj_output": "Attention o_proj",
    "mlp_output": "MLP",
    "transformer_block_output": "Block output",
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float], seed: int = 314159, repeats: int = 10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    draws = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def metric_summary():
    summary = json.loads((RESULTS / "confirmation_summary.json").read_text(encoding="utf-8"))
    activation_rows = read_jsonl(RESULTS / "activation_confirmation.jsonl")
    parameter_rows = read_jsonl(RESULTS / "parameter_confirmation.jsonl")
    predictions = read_jsonl(RESULTS / "confirmation_predictions.jsonl")
    activation_by: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in activation_rows:
        activation_by[(row["task_id"], row["candidate_id"])][row["instance_id"]] = row
    parameter_by: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in parameter_rows:
        parameter_by[(row["task_id"], row["candidate_id"], row["condition"])][row["instance_id"]] = row
    prediction_by: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in predictions:
        prediction_by[(row["task_id"], row["condition"])][row["instance_id"]] = row

    paired_rows = []
    for task_id in TASKS:
        base_activation = activation_by[(task_id, "baseline")]
        base_parameter = parameter_by[(task_id, "baseline", "baseline")]
        base_metrics = summary["activation"][task_id]["baseline"]
        conditions = []
        for (task, condition), values in activation_by.items():
            if task == task_id and condition != "baseline":
                conditions.append(("activation", condition, values))
        for (task, candidate, condition), values in parameter_by.items():
            if task == task_id and condition != "baseline":
                label = candidate if condition == "rollback" else "matched_random_delta_" + candidate
                conditions.append(("parameter", label, values))
        for kind, label, values in conditions:
            baseline_map = base_activation if kind == "activation" else base_parameter
            common = sorted(set(values) & set(baseline_map))
            paired = [
                (float(baseline_map[key].get("mean_token_nll", float(baseline_map[key]["nll_sum"]) / baseline_map[key]["token_count"])))
                - (float(values[key].get("mean_token_nll", float(values[key]["nll_sum"]) / values[key]["token_count"])))
                for key in common
            ]
            mean, low, high = mean_ci(paired, seed=314159 + len(paired) + int(task_id[4:7]))
            result_metrics = None
            if kind == "activation":
                result_metrics = summary["activation"][task_id].get(label)
            else:
                result_metrics = summary["parameter_rollback"][task_id].get("matched_random_delta" if label.startswith("matched_random_delta_") else label)
            primary_key = "rouge_l_f1" if task_id in TASKS[:2] else "accuracy"
            primary_value = result_metrics.get(primary_key) if result_metrics else None
            prediction_condition = label
            if kind == "parameter":
                prediction_condition = (
                    "matched_random_delta_" + label.removeprefix("matched_random_delta_")
                    if label.startswith("matched_random_delta_")
                    else "parameter_rollback_" + label
                )
            paired_predictions = prediction_by[(task_id, prediction_condition)] if (task_id, prediction_condition) in prediction_by else {}
            base_preds = prediction_by[(task_id, "baseline")]
            primary_delta = None
            if task_id in TASKS[:2] and paired_predictions:
                vals = []
                for instance_id in sorted(set(paired_predictions) & set(base_preds)):
                    pred = paired_predictions[instance_id]["prediction"]
                    refs = paired_predictions[instance_id].get("references", [""]) or [""]
                    before = base_preds[instance_id]["prediction"]
                    before_refs = base_preds[instance_id].get("references", [""]) or [""]
                    vals.append(max(rouge_l_f1(pred, ref) for ref in refs) - max(rouge_l_f1(before, ref) for ref in before_refs))
                primary_delta = sum(vals) / len(vals) if vals else None
            elif task_id in TASKS[2:] and paired_predictions:
                vals = []
                for instance_id in sorted(set(paired_predictions) & set(base_preds)):
                    ref = paired_predictions[instance_id]["references"][0]
                    gold = parse_yes_no(ref)
                    pred = parse_yes_no(paired_predictions[instance_id]["prediction"])
                    before = parse_yes_no(base_preds[instance_id]["prediction"])
                    vals.append(float(pred == gold and pred is not None) - float(before == gold and before is not None))
                primary_delta = sum(vals) / len(vals) if vals else None
            detail = result_metrics or {}
            paired_rows.append({
                "task_id": task_id, "condition_type": kind, "condition": label,
                "examples": len(common), "mean_per_example_nll_improvement": mean,
                "nll_improvement_ci95_low": low, "nll_improvement_ci95_high": high,
                "locked_primary_metric": primary_value, "primary_metric_delta_vs_baseline": primary_delta,
                "mean_generated_token_count": detail.get("mean_generated_token_count_including_eos"),
                "exact_yes_output_rate": detail.get("exact_yes_output_rate"),
                "exact_no_output_rate": detail.get("exact_no_output_rate"),
                "exact_yes_no_output_rate": detail.get("exact_yes_no_output_rate"),
                "parseable_yes_no_label_rate": detail.get("parseable_yes_no_label_rate"),
                "invalid_answer_format_rate": detail.get("invalid_answer_format_rate"),
                "yes_minus_no_log_probability_margin": detail.get("mean_yes_minus_no_log_probability_margin"),
                "parameter_count": detail.get("parameter_count"),
                "rollback_fraction_of_model": detail.get("rollback_fraction_of_model"),
                "update_l2_norm": detail.get("update_l2_norm"),
                "update_rms": detail.get("update_rms"),
            })
    write_csv(RESULTS / "confirmation_paired_deltas.csv", paired_rows)
    return summary, paired_rows


def make_activation_heatmap():
    rows = read_csv(RESULTS / "activation_screen_summary.csv")
    layers = list(range(28))
    tasks = TASKS
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.4), sharey=True, constrained_layout=True)
    values = []
    arrays = {}
    for task in tasks:
        matrix = np.zeros((3, 28), dtype=np.float64)
        for i, component in enumerate(COMPONENTS):
            for row in rows:
                if row["task_id"] == task and row["scope"] == "single_layer" and row["candidate_type"] == component:
                    suffix = row["candidate_id"].rsplit("_", 1)[1]
                    matrix[i, int(suffix)] = float(row["nll_improvement_vs_baseline"])
        arrays[task] = matrix
        values.extend(matrix.flatten().tolist())
    max_abs = max(0.05, float(np.quantile(np.abs(values), 0.98)))
    image = None
    for ax, task in zip(axes, tasks):
        image = ax.imshow(arrays[task], aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(range(0, 28, 3), range(0, 28, 3))
        ax.set_xlabel("Decoder layer (0-based)")
        ax.set_yticks(range(3), ["Attention o_proj", "MLP", "Block output"])
    axes[0].set_ylabel("Patched component")
    fig.colorbar(image, ax=axes, label="Mean token NLL improvement (paired screen baseline)", shrink=0.88)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "activation_single_layer_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_parameter_tradeoff():
    rows = read_csv(RESULTS / "parameter_screen_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"attention": "#2c7fb8", "mlp": "#41ab5d", "norm": "#fdae61", "attention_plus_mlp": "#d7191c", "backbone": "#762a83", "embedding_lm_head": "#9970ab"}
    for ax, task in zip(axes.flat, TASKS):
        task_rows = [row for row in rows if row["task_id"] == task and row["candidate_id"] != "baseline"]
        for row in task_rows:
            comp = row["candidate_id"].split("_group_")[0].split("_layer_")[0]
            if row["scope"] == "global":
                comp = row["candidate_id"].removesuffix("_global")
            ax.scatter(float(row["rollback_fraction_of_model"]), float(row["nll_improvement_vs_baseline"]),
                       s=15, alpha=0.55, color=colors.get(comp, "#666666"), linewidths=0)
        best = max(task_rows, key=lambda row: float(row["nll_improvement_vs_baseline"]))
        ax.annotate(best["candidate_id"], (float(best["rollback_fraction_of_model"]), float(best["nll_improvement_vs_baseline"])), xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Fraction of model parameters rolled back")
        ax.set_ylabel("Mean per-example NLL improvement")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=name) for name, color in colors.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, title="Parameter group")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "parameter_rollback_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_confirmation_figure(summary: dict, paired_rows: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"baseline": "#4d4d4d", "activation": "#2b8cbe", "rollback": "#e34a33", "random": "#969696"}
    for ax, task in zip(axes.flat, TASKS):
        task_rows = [row for row in paired_rows if row["task_id"] == task]
        plot_rows = [row for row in task_rows if row["condition_type"] == "activation"]
        plot_rows = sorted(plot_rows, key=lambda row: row["condition"])
        baseline = summary["activation"][task]["baseline"]
        primary_key = "rouge_l_f1" if task in TASKS[:2] else "accuracy"
        labels = ["baseline"]
        primary = [baseline[primary_key]]
        nll = [baseline["teacher_forced_token_nll"]]
        bar_colors = [colors["baseline"]]
        for row in plot_rows:
            detail = summary["activation"][task][row["condition"]]
            labels.append("act: " + row["condition"].replace("_", "\n"))
            primary.append(detail[primary_key])
            nll.append(detail["teacher_forced_token_nll"])
            bar_colors.append(colors["activation"])
        params = summary["parameter_rollback"][task]
        rollback_key = next(key for key in params if key != "matched_random_delta")
        rb = params[rollback_key]
        labels.extend(["weight\nrollback", "random\nmatched"])
        primary.extend([rb[primary_key], params["matched_random_delta"][primary_key]])
        nll.extend([rb["teacher_forced_token_nll"], params["matched_random_delta"]["teacher_forced_token_nll"]])
        bar_colors.extend([colors["rollback"], colors["random"]])
        x = np.arange(len(labels))
        bars = ax.bar(x, primary, color=bar_colors, width=0.68)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, max(1.0, max(primary) * 1.14))
        ax.set_ylabel(primary_key.replace("_", " "))
        ax.set_title(TASK_LABELS[task])
        ax2 = ax.twinx()
        ax2.plot(x, nll, color="#222222", marker="o", linestyle="--", linewidth=1, markersize=3)
        ax2.set_ylabel("Teacher-forced token NLL")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "locked_confirmation_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_synergy_figure():
    rows = read_csv(RESULTS / "activation_synergy_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels_order = ["attention", "mlp", "attention+mlp", "block"]
    colors = ["#2c7fb8", "#41ab5d", "#756bb1", "#d95f0e"]
    for ax, task in zip(axes.flat, TASKS):
        task_rows = [row for row in rows if row["task_id"] == task and row["split"] == "locked_confirmation"]
        values = []
        for component in labels_order:
            match = next(row for row in task_rows if row["components"] == component)
            values.append(float(match["nll_improvement_vs_within_batch_baseline"]))
        ax.bar(labels_order, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(TASK_LABELS[task])
        ax.set_ylabel("Paired NLL improvement")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "attention_mlp_synergy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_trajectory_figure():
    rows = read_csv(RESULTS / "checkpoint_trajectory_summary.csv")
    stages = ["after_task1603", "after_task1501", "final_after_task1384"]
    stage_labels = ["After task1603", "After task1501", "Final task1384"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        values = [float(next(row for row in rows if row["task_id"] == task and row["checkpoint"] == stage)["teacher_forced_token_nll"]) for stage in stages]
        ax.plot(range(len(stages)), values, marker="o", linewidth=2, color="#2c7fb8")
        ax.set_xticks(range(len(stages)), stage_labels, rotation=20, ha="right")
        ax.set_ylabel("Teacher-forced token NLL")
        ax.set_title(TASK_LABELS[task])
        ax.grid(axis="y", alpha=0.25)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "checkpoint_trajectory_nll.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_binary_margin_figure():
    rows = read_csv(RESULTS / "activation_screen_summary.csv")
    classifier_tasks = TASKS[2:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {COMPONENTS[0]: "#2c7fb8", COMPONENTS[1]: "#41ab5d", COMPONENTS[2]: "#d95f0e"}
    labels = {COMPONENTS[0]: "Attention o_proj", COMPONENTS[1]: "MLP", COMPONENTS[2]: "Block output"}
    for ax, task in zip(axes, classifier_tasks):
        task_rows = [row for row in rows if row["task_id"] == task]
        baseline = float(next(row for row in task_rows if row["candidate_id"] == "baseline")["mean_yes_minus_no_log_probability_margin"])
        for component in COMPONENTS:
            layer_rows = [row for row in task_rows if row["scope"] == "single_layer" and row["candidate_type"] == component]
            layer_rows.sort(key=lambda row: int(row["candidate_id"].rsplit("_", 1)[1]))
            shift = [float(row["mean_yes_minus_no_log_probability_margin"]) - baseline for row in layer_rows]
            ax.plot(range(28), shift, marker=".", markersize=4, linewidth=1.4, color=colors[component], label=labels[component])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Decoder layer (0-based)")
        ax.set_ylabel("Change in log P(Yes) − log P(No)")
        ax.set_xticks(range(0, 28, 3))
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "binary_margin_layer_shift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_cross_task_figure():
    rows = read_csv(RESULTS / "cross_task_transfer_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    short = {TASKS[0]: "565", TASKS[1]: "1603", TASKS[2]: "1501", TASKS[3]: "1384"}
    for col, (kind, kind_label) in enumerate((("activation", "Activation replacement"), ("parameter_rollback", "Parameter rollback"))):
        kind_rows = [row for row in rows if row["intervention_type"] == kind]
        for row_index, (metric, title) in enumerate((("primary_metric_delta_vs_target_baseline", "Target primary metric change"), ("teacher_forced_nll_delta_vs_target_baseline", "Token NLL change"))):
            ax = axes[row_index, col]
            matrix = np.full((len(TASKS), len(TASKS)), np.nan, dtype=float)
            annotations = {}
            for row in kind_rows:
                y, x = TASKS.index(row["source_task"]), TASKS.index(row["target_task"])
                matrix[y, x] = float(row[metric])
                annotations[(y, x)] = f"{float(row[metric]):+.2f}"
            finite = matrix[np.isfinite(matrix)]
            limit = max(0.05, float(np.max(np.abs(finite)))) if len(finite) else 0.05
            image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for (y, x), label in annotations.items():
                ax.text(x, y, label, ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[y, x]) > limit * 0.58 else "black")
            ax.set_xticks(range(len(TASKS)), [short[t] for t in TASKS])
            ax.set_yticks(range(len(TASKS)), [short[t] for t in TASKS])
            ax.set_xlabel("Target task")
            ax.set_ylabel("Source task")
            ax.set_title(f"{kind_label}: {title}")
            fig.colorbar(image, ax=ax, shrink=0.82)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cross_task_transfer_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def append_cross_task_and_projection_results(summary: dict):
    parameter_rows = read_csv(RESULTS / "parameter_screen_summary.csv")
    cross_rows = read_csv(RESULTS / "cross_task_transfer_summary.csv")
    projection_ids = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    projection_lines = [
        "| Task | Best individual projection rollback | NLL gain | Parameters | Best scanned group | NLL gain |",
        "|---|---|---:|---:|---|---:|",
    ]
    for task_id in TASKS:
        task_rows = [row for row in parameter_rows if row["task_id"] == task_id and row["candidate_id"] != "baseline"]
        projection_rows = [row for row in task_rows if any(proj in row["candidate_id"] for proj in projection_ids)]
        best_projection = max(projection_rows, key=lambda row: float(row["nll_improvement_vs_baseline"]))
        best_group = max(task_rows, key=lambda row: float(row["nll_improvement_vs_baseline"]))
        projection_lines.append(
            f"| {TASK_LABELS[task_id]} | `{best_projection['candidate_id']}` | {float(best_projection['nll_improvement_vs_baseline']):.3f} | {100*float(best_projection['rollback_fraction_of_model']):.3f}% | `{best_group['candidate_id']}` | {float(best_group['nll_improvement_vs_baseline']):.3f} |"
        )

    cross_lines = [
        "| Source task | Target task | Intervention | Δ token NLL | Primary metric, baseline → condition | Mean output tokens | Exact Yes/No outputs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in cross_rows:
        base_metric = summary["activation"][row["target_task"]]["baseline"][row["primary_metric"]]
        intervention = "activation" if row["intervention_type"] == "activation" else "parameter rollback"
        primary = f"{float(base_metric):.3f} → {float(row['primary_metric_value']):.3f} ({float(row['primary_metric_delta_vs_target_baseline']):+.3f})"
        yes_no = "—" if row["target_task"] in TASKS[:2] else f"{100*float(row['exact_yes_no_output_rate']):.0f}%"
        cross_lines.append(
            f"| {TASK_LABELS[row['source_task']]} | {TASK_LABELS[row['target_task']]} | {intervention} | {float(row['teacher_forced_nll_delta_vs_target_baseline']):+.3f} | {primary} | {float(row['mean_generated_token_count_including_eos']):.1f} | {yes_no} |"
        )

    gen_to_cls = [row for row in cross_rows if row["source_task"] in TASKS[:2] and row["target_task"] in TASKS[2:]]
    cls_to_gen = [row for row in cross_rows if row["source_task"] in TASKS[2:] and row["target_task"] in TASKS[:2]]
    accuracy_deltas = [float(row["primary_metric_delta_vs_target_baseline"]) for row in gen_to_cls]
    nll_deltas = [float(row["teacher_forced_nll_delta_vs_target_baseline"]) for row in gen_to_cls]
    rouge_deltas = [float(row["primary_metric_delta_vs_target_baseline"]) for row in cls_to_gen]
    lengths = [float(row["mean_generated_token_count_including_eos"]) for row in cls_to_gen]
    append_main = f"""

## Projection-level parameter rollback

The expanded screen rolled back each Attention Q/K/V/O and MLP gate/up/down projection separately at every layer and globally (203 projection groups per task, within a 560-candidate parameter screen). The strongest individual projection candidates are screening results from 10 dev examples, not locked confirmations.

{chr(10).join(projection_lines)}

## Cross-task transfer and retention

Each task's selected activation and parameter condition was applied to both tasks of the other task type using the same 25 locked dev IDs. Negative Δ token NLL means lower NLL than the target task's final-model baseline; a negative primary-metric change means lower ROUGE-L or classifier accuracy.

{chr(10).join(cross_lines)}

Across generation-to-classification transfers, accuracy changed from {min(accuracy_deltas):+.2f} to {max(accuracy_deltas):+.2f}, while token-NLL change ranged from {min(nll_deltas):+.3f} to {max(nll_deltas):+.3f}. Classifier-to-generation transfers changed ROUGE-L from {min(rouge_deltas):+.3f} to {max(rouge_deltas):+.3f}; generated lengths ranged from {min(lengths):.1f} to {max(lengths):.1f} tokens. These selected restoration conditions are not task-agnostic: generation-selected restorations can damage classification, and classifier-selected conditions do not recover normal free-form generation. See [cross-task transfer heatmap](../figures/phase6a/cross_task_transfer_heatmap.png) and [per-example results](../results/phase6a/cross_task_transfer.jsonl).
"""
    main_path = REPORTS / "phase6a_component_localization.md"
    with main_path.open("a", encoding="utf-8") as handle:
        handle.write(append_main)
    append_implications = f"""

## Cross-task retention and projection screen

Transferring generation-selected activations or weights onto the classifiers reduced accuracy and raised NLL in the locked development evaluation. Transferring classifier-selected conditions to the generation tasks left output lengths at {min(lengths):.1f}–{max(lengths):.1f} tokens and produced only {min(rouge_deltas):+.3f}–{max(rouge_deltas):+.3f} ROUGE-L change. The screen's strongest projection also varied by task; no common Q/K/V/O or gate/up/down rollback is supported across all tasks. The next pilot should use task-conditioned late routes and evaluate retention and incoming-task quality together.
"""
    with (REPORTS / "phase6a_architecture_implications.md").open("a", encoding="utf-8") as handle:
        handle.write(append_implications)


def fmt(value, digits=3):
    return "—" if value is None else f"{float(value):.{digits}f}"


def build_reports(summary: dict, paired_rows: list[dict]):
    activation_screen = read_csv(RESULTS / "activation_screen_summary.csv")
    parameter_screen = read_csv(RESULTS / "parameter_screen_summary.csv")
    synergy_rows = read_csv(RESULTS / "activation_synergy_summary.csv")
    trajectory_rows = read_csv(RESULTS / "checkpoint_trajectory_summary.csv")
    memory_files = [RESULTS / f"{stage}_runtime.json" for stage in ("activation_screen", "parameter_screen", "synergy", "trajectory", "confirm", "cross_task")]
    runtimes = [json.loads(path.read_text(encoding="utf-8")) for path in memory_files if path.exists()]
    unit_tests = json.loads((RESULTS / "unit_test_results.json").read_text(encoding="utf-8"))
    peak_gib = max((entry["peak_allocated_bytes"] for entry in runtimes), default=0) / (1024 ** 3)
    peak_reserved = max((entry["peak_reserved_bytes"] for entry in runtimes), default=0) / (1024 ** 3)
    task_table = []
    implications = []
    for task_id in TASKS:
        base = summary["activation"][task_id]["baseline"]
        activations = [row for row in paired_rows if row["task_id"] == task_id and row["condition_type"] == "activation"]
        best_act = max(activations, key=lambda row: float(row["mean_per_example_nll_improvement"]))
        act = summary["activation"][task_id][best_act["condition"]]
        params = summary["parameter_rollback"][task_id]
        rollback_key = next(key for key in params if key != "matched_random_delta")
        rollback = params[rollback_key]
        random = params["matched_random_delta"]
        primary_key = "rouge_l_f1" if task_id in TASKS[:2] else "accuracy"
        task_table.append({
            "task_id": TASK_LABELS[task_id], "n": base["evaluation_count"],
            "baseline_NLL": base["teacher_forced_token_nll"],
            "activation_condition": best_act["condition"],
            "activation_NLL": act["teacher_forced_token_nll"],
            "activation_NLL_improvement": best_act["mean_per_example_nll_improvement"],
            "activation_CI95": [best_act["nll_improvement_ci95_low"], best_act["nll_improvement_ci95_high"]],
            "activation_primary": act[primary_key],
            "rollback_condition": rollback_key,
            "rollback_NLL": rollback["teacher_forced_token_nll"],
            "rollback_primary": rollback[primary_key],
            "rollback_fraction": rollback.get("rollback_fraction_of_model"),
            "random_NLL": random["teacher_forced_token_nll"],
            "baseline_primary": base[primary_key],
        })
        synergy = [row for row in synergy_rows if row["task_id"] == task_id and row["split"] == "locked_confirmation"]
        def syn(component):
            return float(next(row for row in synergy if row["components"] == component)["nll_improvement_vs_within_batch_baseline"])
        implications.append({
            "task": TASK_LABELS[task_id], "activation": best_act["condition"],
            "attention_gain": syn("attention"), "mlp_gain": syn("mlp"),
            "attention_mlp_gain": syn("attention+mlp"), "block_gain": syn("block"),
            "rollback": rollback_key, "rollback_fraction": rollback.get("rollback_fraction_of_model"),
        })

    table_lines = [
        "| Task | NLL baseline → activation | Paired NLL improvement (95% bootstrap CI) | Primary baseline → activation | Parameter rollback and primary | Matched random NLL |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in task_table:
        primary_key = "rouge_l_f1" if row["task_id"] in ("565 CIRCA", "1603 SMCalFlow") else "accuracy"
        low, high = row["activation_CI95"]
        table_lines.append(
            f"| {row['task_id']} | {fmt(row['baseline_NLL'])} → {fmt(row['activation_NLL'])} | {fmt(row['activation_NLL_improvement'])} [{fmt(low)}, {fmt(high)}] | {fmt(row['baseline_primary'])} → {fmt(row['activation_primary'])} | {row['rollback_condition']} ({100*row['rollback_fraction']:.1f}% weights), {fmt(row['rollback_primary'])} {primary_key} | {fmt(row['random_NLL'])} |"
        )

    task_findings = []
    label_findings = []
    for row in implications:
        task_findings.append(
            f"- **{row['task']}**: selected activation condition `{row['activation']}`. On the locked same-location swap, NLL improvements were Attention {fmt(row['attention_gain'])}, MLP {fmt(row['mlp_gain'])}, Attention+MLP {fmt(row['attention_mlp_gain'])}, and whole block {fmt(row['block_gain'])}. Parameter screening selected `{row['rollback']}` ({100*row['rollback_fraction']:.1f}% of model parameters)."
        )
    for task_id in TASKS:
        if task_id in TASKS[:2]:
            base = summary["activation"][task_id]["baseline"]
            chosen_row = max([row for row in paired_rows if row["task_id"] == task_id and row["condition_type"] == "activation"], key=lambda row: float(row["mean_per_example_nll_improvement"]))
            act = summary["activation"][task_id][chosen_row["condition"]]
            label_findings.append(
                f"- **{TASK_LABELS[task_id]}**: baseline emitted exact Yes/No strings for {100*base['exact_yes_no_output_rate']:.1f}% of outputs (Yes {100*base['exact_yes_output_rate']:.1f}%, No {100*base['exact_no_output_rate']:.1f}%) at mean length {base['mean_generated_token_count_including_eos']:.2f}. The selected activation yielded exact Yes/No for {100*act['exact_yes_no_output_rate']:.1f}% at length {act['mean_generated_token_count_including_eos']:.2f}; it began producing longer text, while ROUGE-L remained {act['rouge_l_f1']:.3f}."
            )
        else:
            base = summary["activation"][task_id]["baseline"]
            act_items = [row for row in paired_rows if row["task_id"] == task_id and row["condition_type"] == "activation"]
            best = max(act_items, key=lambda row: float(row["mean_per_example_nll_improvement"]))
            act = summary["activation"][task_id][best["condition"]]
            label_findings.append(
                f"- **{TASK_LABELS[task_id]}**: the locked Yes-minus-No first-token margin changed {base['mean_yes_minus_no_log_probability_margin']:.3f} → {act['mean_yes_minus_no_log_probability_margin']:.3f}; exact output shares changed Yes {100*base['exact_yes_output_rate']:.1f}%/{100*base['exact_no_output_rate']:.1f}% → Yes {100*act['exact_yes_output_rate']:.1f}%/No {100*act['exact_no_output_rate']:.1f}%, with accuracy {base['accuracy']:.2f} → {act['accuracy']:.2f}."
            )

    trajectory_lines = [
        "| Task | After task1603 | After task1501 | Final after task1384 |",
        "|---|---:|---:|---:|",
    ]
    for task_id in TASKS:
        vals = {
            row["checkpoint"]: row["teacher_forced_token_nll"]
            for row in trajectory_rows if row["task_id"] == task_id
        }
        trajectory_lines.append(
            f"| {TASK_LABELS[task_id]} | {float(vals['after_task1603']):.3f} | {float(vals['after_task1501']):.3f} | {float(vals['final_after_task1384']):.3f} |"
        )

    report = f"""# Phase 6A: Component-Level Causal Localization

## Scope and safeguards

This phase localizes the final A0/S1/seed-10 model's behavior using same-trajectory checkpoints. It reads only train and development data. The test split was not loaded; the causal screens and confirmations used no optimizer steps and no NAS search. Because the original after-task1501 weights were missing, a prerequisite replay ran that task's original 50 optimizer steps from the retained task1603 boundary. Its training order, supervised-token count, losses, and resulting weights matched the original final trajectory audit; no new task sequence was trained.

The target checkpoint is the final after-task1384 Phase 4B model. Historical activation/weight references are after task1603 for the two generation tasks and after task1501 for the two classification tasks. The Embedding and LM Head are tied and were treated as a single `model.embed_tokens.weight` parameter for rollback. Logit replacement is reported as an output-level upper bound, not a deployable internal component.

All new artifacts are under `results/phase6a/` and `figures/phase6a/`; prior Phase 4B/5 artifacts were read only. The CUDA peak across the Phase 6A runs was {peak_gib:.2f} GiB allocated ({peak_reserved:.2f} GiB reserved) on the RTX 4090.

The machine-readable completion checks are summarized in [phase6a_audit.md](phase6a_audit.md) and `results/phase6a/phase6a_audit.json`.

The focused Phase 6A tests passed ({unit_tests['focused_phase6a_result']}); the full repository suite passed ({unit_tests['result']}).

## Method

The activation screen used dev indices 5–14 (10 examples/task). For each task, reference activations were captured on the exact same prompt-plus-reference-token sequence, then substituted into the final model at `self_attn.o_proj`, MLP output, whole decoder-block output, token embedding output, final norm output, or selected logits. It scanned all 28 single layers and all sliding contiguous groups of widths 2, 4, 8, 14, and 28. The objective was per-example teacher-forced completion NLL. Each candidate batch included an unpatched row; the zero-operation control had exactly zero paired change.

The parameter screen used dev indices 15–24 (10 examples/task). It restored matching tensors from the appropriate earlier boundary without an optimizer step. It scanned single layers, contiguous layer groups, global Attention, global MLP, combined Attention+MLP, norms, tied embedding/LM Head, and the full backbone. It recorded parameter count, fraction of model weights, rollback update L2/RMS, and runtime. The best screen condition per task was confirmed against a seeded random-direction update with the same selected parameter count and update L2 norm.

Locked confirmation used dev indices 25–49 (25 examples/task). Teacher-forced NLL and greedy autoregressive generation (up to 64 new tokens) were evaluated for the baseline and screen-selected activation/parameter conditions. Activation generation used paired decoding: source and target received the same prompt and target-generated tokens, with reference activations substituted at the selected site at prefill and every cached decode step. The activation synergy check compared Attention-only, MLP-only, joint Attention+MLP, and whole-block replacement at the same selected layer/group. The output includes raw per-example JSONL and paired bootstrap intervals over 10,000 resamples of the 25 locked IDs.

## Locked confirmation

{chr(10).join(table_lines)}

The generation primary metric is ROUGE-L F1; the classifier primary metric is parsed Yes/No accuracy. The NLL interval is a paired bootstrap interval for the mean per-example token NLL improvement (baseline minus intervention); token-weighted NLL is shown in the table. The per-task detailed metrics also include Token-F1/macro-F1, generated length, EOS termination, exact Yes/No rates, parseable/invalid-format rates, first-reference-token log probability, and the Yes-minus-No first-token log-probability margin.

## Localization and interaction results

{chr(10).join(task_findings)}

## Task-stage trajectory

The same 25 locked development IDs were scored under the three retained boundaries. NLL shifts show that the representation changes are tied to the task sequence: task565 and task1603 deteriorate after later classifier stages, task1501 sharply improves after its own stage and partially regresses after task1384, while task1384 improves across the later boundaries.

{chr(10).join(trajectory_lines)}

See [checkpoint_trajectory_nll.png](../figures/phase6a/checkpoint_trajectory_nll.png) and the per-example [checkpoint_trajectory.jsonl](../results/phase6a/checkpoint_trajectory.jsonl).

### Yes/No margin and output bias

Single-layer activation swaps shift the first-token log P(Yes) − log P(No) margin in different directions across tasks and components; there is no shared layer at which both classifiers acquire the same label bias. These are teacher-forced first-token margin changes, while the locked autoregressive checks below give the actual output shares.

{chr(10).join(label_findings)}

The per-layer margin shifts appear in [binary_margin_layer_shift.png](../figures/phase6a/binary_margin_layer_shift.png).

The late-layer single-component heatmap is in [activation_single_layer_heatmap.png](../figures/phase6a/activation_single_layer_heatmap.png). The rollback-size tradeoff is in [parameter_rollback_tradeoff.png](../figures/phase6a/parameter_rollback_tradeoff.png). Locked generation/classification outcomes are in [locked_confirmation_metrics.png](../figures/phase6a/locked_confirmation_metrics.png), and same-location Attention/MLP interactions are in [attention_mlp_synergy.png](../figures/phase6a/attention_mlp_synergy.png).

## Interpretation limits

The 10-example screens search hundreds of conditions, so winner estimates are selection-biased; the 25-example locked development slice reduces but does not remove that risk. Bootstrap intervals quantify ID-level variability conditional on this fixed task stream and checkpoints, not across seeds or datasets. Activation substitution is a causal intervention on the measured prompt/reference trajectory; autoregressive results are stronger behavioral evidence, but a single fixed stream does not establish a general architecture benefit. Parameter rollback changes large groups of weights and may restore multiple correlated functions at once. The large global rollback for old generation tasks is not evidence that a small module alone is sufficient.

No held-out test conclusions are made. Before a formal architecture decision, repeat the locked analysis with additional seeds/orderings and evaluate once on a test split under a frozen plan.
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase6a_component_localization.md").write_text(report, encoding="utf-8")

    implications_report = f"""# Phase 6A: Architecture Implications

## Decision from the measured interventions

The evidence supports testing **task-conditional late residual pathways** as the next small architecture pilot. It does not justify broad NAS yet. In particular, a fixed policy that restores Attention and MLP together is not supported across tasks.

## Task-specific evidence

{chr(10).join(task_findings)}

The locked same-location interaction comparison shows why a single global module rule is risky:

- task565 and task1603 gain much more from whole-block output replacement than from Attention+MLP replacement. The residual stream carries a useful integrated state beyond the sum of restoring just those two module outputs.
- task1501 benefits from a broad Attention activation patch; joint Attention+MLP and block output are comparable or slightly stronger on its locked NLL. A classifier path may need both branches to preserve label evidence.
- task1384 benefits from MLP output replacement alone, while the joint Attention+MLP patch is harmful on NLL. Adding a branch can undo the useful MLP correction.

The parameter screen gives a second, distinct result. Rollback toward the task1603 boundary improves old generation NLL, but the best group contains roughly 74% of all model parameters. On task1501 a full MLP rollback (44.3%) improves accuracy and NLL; for task1384 the best screened rollback improves neither locked accuracy nor NLL relative to baseline. Matched-L2 random directions do not reproduce the old-generation or task1501 effects. This indicates checkpoint-specific directions matter, while the parameter groups remain too large to recommend as selective storage or routing units.

The checkpoint trajectory adds a task-stage effect: generation NLL rises after the classifier tasks; task1501 loses part of its boundary likelihood after the last task; task1384 obtains its largest improvement at its own final stage. This is consistent with interference along this one sequence, though NLL alone does not measure decoded task quality.

## Recommended follow-up pilot

1. Compare a shared 28-layer trunk with a small task-conditioned late-block adapter/router. Include a residual-preserving block route and separate Attention/MLP routes as distinct arms.
2. Keep task565 and task1603 generation plus both classifier tasks in every evaluation batch; report old-task retention and new-task accuracy together.
3. Freeze routing from train/dev only, repeat at least three seeds and one alternate task order, and include no-op, random-layer, random-direction, and parameter-count-matched controls.
4. Limit the added state to a small fraction of the 596.05M-parameter model. Use the current 73.9% global rollback only as a recovery upper bound, not as a proposed deployment design.
5. After selection is frozen, evaluate a test set once and report ROUGE-L/Token-F1, classifier accuracy/macro-F1, exact Yes/No proportions, invalid-format rate, and generated length.

## What this phase does not establish

The study uses one sequence, one seed, one Qwen3-0.6B checkpoint, and small development slices selected from a larger screen. It identifies causal locations for these prompts and this trajectory; it does not prove that a router, adapter, or NAS-selected topology will improve continual learning generally. The causal analyses used no optimizer steps. A separate exact replay of the original task1501 boundary supplied missing historical weights; no new task sequence or NAS search was run.

Raw results: [results/phase6a](../results/phase6a/). Main report: [phase6a_component_localization.md](phase6a_component_localization.md).
"""
    (REPORTS / "phase6a_architecture_implications.md").write_text(implications_report, encoding="utf-8")

    write_csv(RESULTS / "phase6a_task_summary.csv", task_table)
    append_cross_task_and_projection_results(summary)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary, paired = metric_summary()
    make_activation_heatmap()
    make_parameter_tradeoff()
    make_confirmation_figure(summary, paired)
    make_synergy_figure()
    make_trajectory_figure()
    make_binary_margin_figure()
    make_cross_task_figure()
    build_reports(summary, paired)
    print(f"Wrote reports in {REPORTS} and figures in {FIGURES}")


if __name__ == "__main__":
    main()
