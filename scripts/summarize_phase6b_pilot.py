"""Consolidate the completed seed-10 Phase 6B pilot without loading data or models."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
# Editable vector text: svg.fonttype='none'; pdf.fonttype=42.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment
from phase6b_modules import select_random_post_layers


ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
OUT = storage_path("results") / "phase6b"
FIGURES = storage_path("figures") / "phase6b"
REPORTS = storage_path("reports")
SEQUENCES = ("S1", "S2")
ARCHITECTURES = ("B0", "B1-static", "B2-input", "B3-oracle", "B2-random-location")
TASKS = (
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
)
TASK_LABELS = {
    TASKS[0]: "565 CIRCA",
    TASKS[1]: "1603 SMCalFlow",
    TASKS[2]: "1501 DSTC3",
    TASKS[3]: "1384 Deal-or-No-Deal",
}
PRIMARY_FIELD = {
    TASKS[0]: "rouge_l_f1",
    TASKS[1]: "rouge_l_f1",
    TASKS[2]: "accuracy",
    TASKS[3]: "accuracy",
}
ROUTE_MODES = ("input", "random", "static_gate", "off")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    behavior = row.get("generation_behavior", {})
    nll = row.get("teacher_forced_nll") or row.get("teacher_forced_token_nll") or {}
    dominant = row.get("yes_no_dominant_output_share")
    if dominant is None:
        yes_share = behavior.get("exact_yes_share")
        no_share = behavior.get("exact_no_share")
        if yes_share is not None and no_share is not None:
            dominant = max(float(yes_share), float(no_share))
        else:
            dominant = behavior.get("yes_no_dominant_output_share")
    collapse = row.get("yes_no_single_label_collapse")
    if collapse is None:
        collapse = bool(dominant is not None and dominant >= 0.95)
    return {
        "task_type": metrics.get("task_type"),
        "primary_metric": metrics.get("primary_metric"),
        "primary_score": metrics.get("primary_score"),
        "rouge_l_f1": metrics.get("rouge_l_f1"),
        "token_f1": metrics.get("token_f1"),
        "teacher_forced_nll": nll.get("token_nll"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "yes_no_dominant_output_share": dominant,
        "yes_no_single_label_collapse": collapse,
        "evaluated": metrics.get("evaluated", behavior.get("examples")),
        "router_usage_by_layer": row.get("router_usage_by_layer", {}),
    }


def final_cl_metrics(payload: dict[str, Any], final_stage: str) -> dict[str, Any]:
    cl = payload.get("continual_learning_metrics", {})
    stage = cl.get("stage_metrics", {}).get(final_stage, {})
    return {
        "average_forgetting_nonnegative_normalized": stage.get(
            "average_forgetting_nonnegative_normalized",
            cl.get("average_forgetting_nonnegative_normalized"),
        ),
        "average_forgetting_nonnegative_raw_by_metric": stage.get(
            "average_forgetting_nonnegative_raw_by_metric",
            cl.get("average_forgetting_nonnegative_raw_by_metric", {}),
        ),
        "nonnegative_forgetting_by_task_normalized": stage.get(
            "nonnegative_forgetting_by_prior_task_normalized", {}
        ),
        "backward_transfer_normalized": stage.get(
            "backward_transfer_normalized", cl.get("backward_transfer_normalized")
        ),
        "backward_transfer_by_task_normalized": stage.get(
            "backward_transfer_by_task_normalized", {}
        ),
        "cross_stage_aaa_normalized": cl.get(
            "cross_stage_aaa_normalized_final", stage.get("cross_stage_aaa_normalized")
        ),
    }


def gradient_summary(path: Path) -> dict[str, Any]:
    trace = path / "gradient_diagnostics.jsonl"
    if not trace.is_file():
        return {"samples": 0}
    expert_slots = expert_slots_with_grad = expert_slots_nonzero = 0
    router_slots = router_slots_nonzero = 0
    branch_rms_values: list[float] = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        for layer in sample.get("paths", {}).values():
            for expert in layer.get("branch_expert_gradients", []):
                expert_slots += 1
                expert_slots_with_grad += int(bool(expert.get("has_gradient")))
                expert_slots_nonzero += int(float(expert.get("gradient_l2", 0.0)) > 0.0)
            if "router_has_gradient" in layer:
                router_slots += 1
                router_slots_nonzero += int(float(layer.get("router_gradient_l2", 0.0)) > 0.0)
            branch_rms_values.append(float(layer.get("branch_rms", 0.0)))
    return {
        "samples": len(trace.read_text(encoding="utf-8").splitlines()),
        "expert_gradient_slots": expert_slots,
        "expert_slots_with_gradient": expert_slots_with_grad,
        "expert_slots_nonzero_gradient": expert_slots_nonzero,
        "router_samples": router_slots,
        "router_samples_nonzero_gradient": router_slots_nonzero,
        "branch_rms_nonzero_samples": sum(value > 0 for value in branch_rms_values),
        "branch_rms_sample_count": len(branch_rms_values),
        "max_sampled_branch_rms": max(branch_rms_values, default=0.0),
    }


def load_run(sequence: str, architecture: str) -> dict[str, Any]:
    run_dir = OUT / "runs" / sequence / architecture / "seed_10"
    wrapper = read_json(run_dir / "metrics.json")
    if wrapper.get("status") != "complete":
        raise RuntimeError(f"Incomplete Phase 6B run: {sequence}/{architecture}")
    if wrapper.get("test_split_loaded") is not False:
        raise RuntimeError(f"Test split flag is not false: {sequence}/{architecture}")

    if architecture == "B0":
        source_path = storage_path("results") / "phase4b" / "runs" / "A0" / sequence / "seed_10" / "metrics.json"
        metric_source = read_json(source_path)
        if metric_source.get("test_split_loaded") is not False:
            raise RuntimeError(f"Baseline test split flag is not false: {source_path}")
        final_stage = metric_source["stage_order"][-1]
    else:
        metric_source = wrapper
        final_stage = metric_source["stage_order"][-1]

    rows = metric_source["task_metrics"][final_stage]
    tasks = {task: metric_values(rows[task]) for task in TASKS}
    score_map = metric_source["stage_scores_by_primary_metric"][final_stage]
    for task in TASKS:
        expected = tasks[task]["primary_score"]
        observed = score_map[task]
        if expected is not None and abs(float(expected) - float(observed)) > 1e-10:
            raise RuntimeError(f"Primary metric mismatch in {sequence}/{architecture}/{task}")

    if architecture == "B0":
        base_parameters = int(metric_source.get("parameter_counts", {}).get("total_parameters", 596_049_920))
        added_parameters = 0
        duration = wrapper.get("total_duration_seconds", metric_source.get("total_duration_seconds"))
        peak_bytes = wrapper.get("peak_cuda_allocated_bytes", metric_source.get("peak_cuda_allocated_bytes"))
        flops = {}
        initialization_checks = []
        reload_pass = wrapper.get("checkpoint_reload_audit", {}).get("passes_reload_audit")
    else:
        counts = wrapper.get("parameter_counts", {})
        base_parameters = int(counts.get("base_parameters", 596_049_920))
        added_parameters = int(counts.get("added_parameters", 0))
        duration = wrapper.get("duration_seconds")
        peak_bytes = wrapper.get("peak_cuda_allocated_bytes")
        flops = wrapper.get("flops", {})
        raw_initialization_checks = wrapper.get("initialization_check", [])
        initialization_checks = (
            [raw_initialization_checks]
            if isinstance(raw_initialization_checks, dict)
            else raw_initialization_checks
        )
        reload_pass = wrapper.get("checkpoint_reload_audit", {}).get("passes_reload_audit")

    cl = final_cl_metrics(wrapper, final_stage)
    final_scores = {task: float(score_map[task]) for task in TASKS}
    acquisition = wrapper.get("new_task_acquisition", {})
    if architecture == "B0" and not acquisition:
        acquisition = metric_source.get("new_task_acquisition", {})
    acquisition_gains = [float(v["normalized_acquisition_gain"]) for v in acquisition.values() if v.get("normalized_acquisition_gain") is not None]
    route_controls = wrapper.get("route_control_ablations", {})
    controls: dict[str, dict[str, Any]] = {}
    for mode, control in route_controls.items():
        controls[mode] = {}
        for task, row in control.get("task_metrics", {}).items():
            controls[mode][task] = metric_values(row)

    init_exact = all(bool(item.get("exact_output_preservation")) for item in initialization_checks) if initialization_checks else architecture == "B0"
    return {
        "sequence": sequence,
        "architecture": architecture,
        "seed": 10,
        "status": wrapper["status"],
        "source_model_sha256": wrapper.get("source_model_sha256", metric_source.get("source_model_sha256")),
        "task_order": wrapper.get("task_order", metric_source.get("task_order", [])),
        "test_split_loaded": False,
        "nas_started": bool(wrapper.get("nas_started", False)),
        "final_stage": final_stage,
        "tasks": tasks,
        "primary_scores": final_scores,
        "continual_learning": cl,
        "acquisition": acquisition,
        "mean_normalized_acquisition_gain": statistics.mean(acquisition_gains) if acquisition_gains else None,
        "base_parameters": base_parameters,
        "added_parameters": added_parameters,
        "added_fraction_of_base": added_parameters / base_parameters if base_parameters else None,
        "flops": flops,
        "duration_seconds": duration,
        "peak_cuda_allocated_bytes": peak_bytes,
        "peak_vram_gib": float(peak_bytes) / (1024**3) if peak_bytes else None,
        "gpu_name": wrapper.get("gpu_name", metric_source.get("gpu_name")),
        "initialization_exact": init_exact,
        "checkpoint_reload_pass": reload_pass,
        "route_controls": controls,
        "gradient_diagnostics": gradient_summary(run_dir),
        "route_policy_during_training": wrapper.get("architecture_definition", {}).get("route_policy_during_training", {}),
        "layer_indices_zero_based": wrapper.get("architecture_definition", {}).get("layers_zero_based", []),
        "historical_model_loaded": bool(wrapper.get("architecture_definition", {}).get("historical_model_loaded_during_pilot", False)),
        "replay_used": bool(wrapper.get("architecture_definition", {}).get("replay_used", False)),
        "sample_order_matches_A0": all(bool(row.get("sample_order_matches_A0", True)) for row in wrapper.get("training_metrics", {}).values()),
        "supervised_target_tokens_matches_A0": all(bool(row.get("supervised_target_tokens_matches_A0", True)) for row in wrapper.get("training_metrics", {}).values()),
        "wrapper_path": str((run_dir / "metrics.json").relative_to(ROOT)),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def route_distance_summary(runs: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str, str], list[float]] = {}
    entropy_values: list[float] = []
    dominant_values: list[float] = []
    for sequence in SEQUENCES:
        run = runs[(sequence, "B2-input")]
        for task, row in run["tasks"].items():
            layers = row.get("router_usage_by_layer", {})
            for layer, stats in layers.items():
                shares = stats.get("expert_token_shares", [])
                if shares:
                    values[(sequence, task, str(layer))] = [float(v) for v in shares]
                    entropy_values.append(float(stats.get("normalized_route_entropy", 0.0)))
                    dominant_values.append(max(float(v) for v in shares))

    same_task_tv: list[float] = []
    across_task_tv: list[float] = []
    for layer in ("25", "27"):
        for task in TASKS:
            one = values.get(("S1", task, layer))
            two = values.get(("S2", task, layer))
            if one and two:
                same_task_tv.append(0.5 * sum(abs(a - b) for a, b in zip(one, two)))
        for sequence in SEQUENCES:
            for i, task_a in enumerate(TASKS):
                for task_b in TASKS[i + 1 :]:
                    one = values.get((sequence, task_a, layer))
                    two = values.get((sequence, task_b, layer))
                    if one and two:
                        across_task_tv.append(0.5 * sum(abs(a - b) for a, b in zip(one, two)))
    return {
        "mean_normalized_entropy": statistics.mean(entropy_values) if entropy_values else None,
        "mean_dominant_expert_share": statistics.mean(dominant_values) if dominant_values else None,
        "mean_same_task_tv_across_orders": statistics.mean(same_task_tv) if same_task_tv else None,
        "mean_different_task_tv_within_order": statistics.mean(across_task_tv) if across_task_tv else None,
        "per_task_layer_shares": {
            f"{sequence}/{task}/layer{layer}": shares
            for (sequence, task, layer), shares in values.items()
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_panel_labels(axes: Any) -> None:
    from matplotlib.transforms import ScaledTranslation

    for label, axis in zip("abcd", axes.flat):
        offset = ScaledTranslation(-12 / 72, 5 / 72, axis.figure.dpi_scale_trans)
        axis.text(
            0,
            1,
            label,
            transform=axis.transAxes + offset,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def export_figure(fig: Any, stem: str, *, multi_panel: bool) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    base = FIGURES / stem
    if multi_panel:
        fig.canvas.draw()
        require_matplotlib_panel_alignment(
            fig,
            json_out=base.with_suffix(".alignment.json"),
            overlay_svg=base.with_suffix(".alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(runs: dict[tuple[str, str], dict[str, Any]], route_summary: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "B0": "#666666",
        "B1-static": "#4C78A8",
        "B2-input": "#F58518",
        "B3-oracle": "#54A24B",
        "B2-random-location": "#B279A2",
    }
    labels = {
        "B0": "B0",
        "B1-static": "B1 static",
        "B2-input": "B2 input",
        "B3-oracle": "B3 oracle",
        "B2-random-location": "B2 random location",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        x = np.arange(len(SEQUENCES))
        width = 0.15
        for i, architecture in enumerate(ARCHITECTURES):
            scores = [runs[(sequence, architecture)]["primary_scores"][task] for sequence in SEQUENCES]
            axis.bar(x + (i - 2) * width, scores, width, label=labels[architecture], color=colors[architecture])
        axis.set_xticks(x, SEQUENCES)
        axis.set_title(TASK_LABELS[task])
        axis.set_ylabel("ROUGE-L" if PRIMARY_FIELD[task] == "rouge_l_f1" else "Accuracy")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.text(0.5, -0.055, "Phase 6B · seed 10 · 50 locked dev examples per task", ha="center", fontsize=8)
    export_figure(fig, "pilot_final_primary_scores", multi_panel=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        deltas = [
            runs[(sequence, "B2-input")]["primary_scores"][task]
            - runs[(sequence, "B1-static")]["primary_scores"][task]
            for sequence in SEQUENCES
        ]
        axis.bar(SEQUENCES, deltas, color=["#4C78A8", "#F58518"], width=0.58)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(TASK_LABELS[task])
        axis.set_ylabel("Δ ROUGE-L" if PRIMARY_FIELD[task] == "rouge_l_f1" else "Δ Accuracy")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes)
    fig.text(0.5, -0.055, "Phase 6B · seed 10 · 50 locked dev examples per task · one paired run per order", ha="center", fontsize=8)
    fig.suptitle("Input routing versus parameter-matched static residual", fontsize=14)
    export_figure(fig, "pilot_b2_vs_b1_delta", multi_panel=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    mode_colors = {"input": "#F58518", "random": "#E45756", "static_gate": "#72B7B2", "off": "#777777"}
    for axis, task in zip(axes.flat, TASKS):
        x = np.arange(len(ROUTE_MODES) * len(SEQUENCES))
        tick_labels = []
        values = []
        colors_for_bars = []
        for sequence in SEQUENCES:
            run = runs[(sequence, "B2-input")]
            for mode in ROUTE_MODES:
                if mode == "input":
                    value = run["primary_scores"][task]
                else:
                    value = run["route_controls"].get(mode, {}).get(task, {}).get("primary_score")
                values.append(value)
                colors_for_bars.append(mode_colors[mode])
                tick_labels.append(mode.replace("_", "\n"))
        axis.bar(x, values, color=colors_for_bars, width=0.72)
        axis.axvline(3.5, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xticks(x, tick_labels, fontsize=8)
        axis.set_title(TASK_LABELS[task])
        axis.set_ylabel("ROUGE-L" if PRIMARY_FIELD[task] == "rouge_l_f1" else "Accuracy")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    add_panel_labels(axes)
    from matplotlib.patches import Patch

    mode_handles = [Patch(facecolor=mode_colors[mode], label=mode.replace("_", " ")) for mode in ROUTE_MODES]
    fig.legend(handles=mode_handles, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.text(0.5, -0.055, "Phase 6B · seed 10 · 50 locked dev examples per task · left group S1, right group S2", ha="center", fontsize=8)
    export_figure(fig, "pilot_b2_route_controls", multi_panel=True)

    share_map = route_summary["per_task_layer_shares"]
    rows = []
    row_labels = []
    for sequence in SEQUENCES:
        for task in TASKS:
            row_labels.append(f"{sequence} · {TASK_LABELS[task]}")
            vector: list[float] = []
            for layer in ("25", "27"):
                vector.extend(share_map.get(f"{sequence}/{task}/layer{layer}", [0.0] * 4))
            rows.append(vector)
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    image = axis.imshow(np.asarray(rows), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    column_labels = [f"L{layer}-E{expert}" for layer in (25, 27) for expert in range(4)]
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35, rotation_mode="anchor", ha="right")
    axis.set_title("B2 input-route expert shares on final dev · seed 10 · n=50/task")
    fig.colorbar(image, ax=axis, label="share")
    export_figure(fig, "pilot_b2_route_usage", multi_panel=False)


def write_reports(runs: dict[tuple[str, str], dict[str, Any]], route_summary: dict[str, Any]) -> None:
    primary_rows: dict[str, list[list[Any]]] = {sequence: [] for sequence in SEQUENCES}
    for sequence in SEQUENCES:
        for architecture in ARCHITECTURES:
            run = runs[(sequence, architecture)]
            primary_rows[sequence].append([
                architecture,
                *(fmt(run["primary_scores"][task], 4) for task in TASKS),
            ])

    detail_tables: dict[str, str] = {}
    for sequence in SEQUENCES:
        rows: list[list[Any]] = []
        for architecture in ARCHITECTURES:
            for task in TASKS:
                metric = runs[(sequence, architecture)]["tasks"][task]
                rows.append([
                    architecture,
                    TASK_LABELS[task],
                    metric["primary_metric"],
                    fmt(metric["primary_score"], 4),
                    fmt(metric["token_f1"], 4),
                    fmt(metric["teacher_forced_nll"], 4),
                    fmt(metric["accuracy"], 4),
                    fmt(metric["macro_f1"], 4),
                    fmt(metric["yes_no_dominant_output_share"], 3),
                    fmt(metric["yes_no_single_label_collapse"]),
                ])
        detail_tables[sequence] = md_table(
            ["架构", "任务", "主指标", "主指标值", "Token-F1", "Teacher NLL", "Accuracy", "Macro-F1", "Yes/No占比", "塌缩"],
            rows,
        )

    delta_rows = []
    stable_wins = 0
    total_pairs = 0
    per_task_deltas: dict[str, list[float]] = {task: [] for task in TASKS}
    for task in TASKS:
        deltas = []
        for sequence in SEQUENCES:
            delta = runs[(sequence, "B2-input")]["primary_scores"][task] - runs[(sequence, "B1-static")]["primary_scores"][task]
            deltas.append(delta)
            per_task_deltas[task].append(delta)
            stable_wins += int(delta > 1e-12)
            total_pairs += 1
        delta_rows.append([TASK_LABELS[task], PRIMARY_FIELD[task], fmt(deltas[0], 4), fmt(deltas[1], 4), fmt(statistics.mean(deltas), 4)])

    cl_rows = []
    resource_rows = []
    acquisition_rows = []
    collapse_counts: dict[str, tuple[int, int]] = {}
    control_rows = []
    route_rows = []
    gradient_lines = []
    for sequence in SEQUENCES:
        for architecture in ARCHITECTURES:
            run = runs[(sequence, architecture)]
            cl = run["continual_learning"]
            forget_by_metric = cl.get("average_forgetting_nonnegative_raw_by_metric", {})
            cl_rows.append([
                sequence,
                architecture,
                fmt(cl.get("average_forgetting_nonnegative_normalized"), 4),
                fmt(forget_by_metric.get("rouge_l_f1"), 4),
                fmt(forget_by_metric.get("accuracy"), 4),
                fmt(cl.get("backward_transfer_normalized"), 4),
                fmt(cl.get("cross_stage_aaa_normalized"), 4),
            ])
            flop_fraction = run["flops"].get("approx_extra_flop_fraction")
            resource_rows.append([
                sequence,
                architecture,
                f"{run['added_parameters']:,}",
                fmt(100 * float(run["added_fraction_of_base"]), 3),
                fmt(100 * float(flop_fraction), 3) if flop_fraction is not None else "0.000",
                fmt(run["duration_seconds"], 1),
                fmt(run["peak_vram_gib"], 2),
                fmt(run["mean_normalized_acquisition_gain"], 4),
            ])
            for stage, info in run["acquisition"].items():
                acquisition_rows.append([
                    sequence,
                    architecture,
                    TASK_LABELS.get(
                        info.get("task_id", info.get("new_task")),
                        info.get("task_id", info.get("new_task")),
                    ),
                    fmt(info.get("raw_acquisition_gain"), 4),
                    fmt(info.get("normalized_acquisition_gain"), 4),
                ])
            collapse_tasks = [task for task, metric in run["tasks"].items() if metric["yes_no_single_label_collapse"]]
            collapse_counts[f"{sequence}/{architecture}"] = (len(collapse_tasks), len(TASKS))
            gradient = run["gradient_diagnostics"]
            gradient_lines.append(
                f"- `{sequence}/{architecture}`: sampled gradient records {gradient.get('samples', 0)}, "
                f"branch slots with nonzero gradients {gradient.get('expert_slots_nonzero_gradient', 0)}/"
                f"{gradient.get('expert_gradient_slots', 0)}, router samples with nonzero gradient "
                f"{gradient.get('router_samples_nonzero_gradient', 0)}/"
                f"{gradient.get('router_samples', 0)}, branch RMS nonzero "
                f"{gradient.get('branch_rms_nonzero_samples', 0)}/"
                f"{gradient.get('branch_rms_sample_count', 0)}."
            )

        run = runs[(sequence, "B2-input")]
        for task in TASKS:
            row = run["tasks"][task]
            stats = row.get("router_usage_by_layer", {})
            layer_descriptions = []
            for layer in ("25", "27"):
                info = stats.get(layer, {})
                shares = info.get("expert_token_shares", [])
                if shares:
                    expert = int(np.argmax(shares))
                    layer_descriptions.append(f"L{layer}: E{expert} {float(shares[expert]):.2f}")
            entropies = [float(item.get("normalized_route_entropy", 0.0)) for item in stats.values()]
            route_rows.append([
                sequence,
                TASK_LABELS[task],
                "; ".join(layer_descriptions) or "—",
                fmt(statistics.mean(entropies), 3) if entropies else "—",
            ])

            for mode in ROUTE_MODES[1:]:
                control = run["route_controls"].get(mode, {}).get(task)
                if control:
                    control_rows.append([
                        sequence,
                        TASK_LABELS[task],
                        mode,
                        fmt(row["primary_score"], 4),
                        fmt(control["primary_score"], 4),
                        fmt(float(control["primary_score"]) - float(row["primary_score"]), 4),
                        fmt(row["teacher_forced_nll"], 4),
                        fmt(control["teacher_forced_nll"], 4),
                    ])

    result_summary = {
        "phase": "6B",
        "status": "complete",
        "seed": 10,
        "sequences": list(SEQUENCES),
        "architectures": list(ARCHITECTURES),
        "test_split_loaded": False,
        "nas_started": False,
        "paired_b2_vs_b1_positive_primary_pairs": stable_wins,
        "paired_b2_vs_b1_total_primary_pairs": total_pairs,
        "mean_b2_minus_b1_by_task": {
            task: statistics.mean(deltas) for task, deltas in per_task_deltas.items()
        },
        "route_usage": route_summary,
        "runs": [runs[(sequence, architecture)] for sequence in SEQUENCES for architecture in ARCHITECTURES],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pilot_summary.json").write_text(json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    detail_rows = []
    for sequence in SEQUENCES:
        for architecture in ARCHITECTURES:
            run = runs[(sequence, architecture)]
            for task, metric in run["tasks"].items():
                detail_rows.append({
                    "sequence": sequence,
                    "architecture": architecture,
                    "seed": 10,
                    "task_id": task,
                    "task_label": TASK_LABELS[task],
                    **{key: metric.get(key) for key in (
                        "task_type", "primary_metric", "primary_score", "rouge_l_f1", "token_f1",
                        "teacher_forced_nll", "accuracy", "macro_f1", "yes_no_dominant_output_share",
                        "yes_no_single_label_collapse", "evaluated",
                    )},
                    "average_forgetting_nonnegative_normalized": run["continual_learning"].get("average_forgetting_nonnegative_normalized"),
                    "backward_transfer_normalized": run["continual_learning"].get("backward_transfer_normalized"),
                    "added_parameters": run["added_parameters"],
                    "added_fraction_of_base": run["added_fraction_of_base"],
                    "duration_seconds": run["duration_seconds"],
                    "peak_vram_gib": run["peak_vram_gib"],
                })
    columns = list(detail_rows[0])
    write_csv(OUT / "pilot_task_metrics.csv", detail_rows, columns)
    write_csv(OUT / "pilot_resource_summary.csv", [
        {
            "sequence": sequence,
            "architecture": architecture,
            "added_parameters": runs[(sequence, architecture)]["added_parameters"],
            "added_fraction_of_base": runs[(sequence, architecture)]["added_fraction_of_base"],
            "extra_flop_fraction": runs[(sequence, architecture)]["flops"].get("approx_extra_flop_fraction", 0.0),
            "duration_seconds": runs[(sequence, architecture)]["duration_seconds"],
            "peak_vram_gib": runs[(sequence, architecture)]["peak_vram_gib"],
            "mean_normalized_acquisition_gain": runs[(sequence, architecture)]["mean_normalized_acquisition_gain"],
            "average_forgetting_nonnegative_normalized": runs[(sequence, architecture)]["continual_learning"].get("average_forgetting_nonnegative_normalized"),
            "backward_transfer_normalized": runs[(sequence, architecture)]["continual_learning"].get("backward_transfer_normalized"),
        }
        for sequence in SEQUENCES for architecture in ARCHITECTURES
    ], [
        "sequence", "architecture", "added_parameters", "added_fraction_of_base", "extra_flop_fraction",
        "duration_seconds", "peak_vram_gib", "mean_normalized_acquisition_gain",
        "average_forgetting_nonnegative_normalized", "backward_transfer_normalized",
    ])

    report = [
        "# Phase 6B Seed-10 Pilot Results",
        "",
        "## Scope and protocol",
        "",
        "The paired pilot completed both task orders (S1 and S2), seed 10, on 50 locked dev examples per task. B0 reuses the matching Phase 4B A0 run; all adapter arms start from the same local Qwen3-0.6B source weights. The run-time checks confirm matching task sample order and supervised target-token budgets against A0. The test split was not loaded. No NAS search was started.",
        "",
        "All recorded rows are from the final post-task-4 stage. Generation metrics are reported only for the two generation tasks; Accuracy and Macro-F1 are reported for the two classification tasks. Teacher-forced NLL is listed for all tasks. Yes/No collapse means one exact label accounts for at least 95% of outputs.",
        "",
        "## Final primary scores",
        "",
        f"| 架构 | 565 CIRCA ROUGE-L | 1603 SMCalFlow ROUGE-L | 1501 DSTC3 Accuracy | 1384 Deal-or-No-Deal Accuracy |\n| --- | ---: | ---: | ---: | ---: |\n" + "\n".join("| " + " | ".join(row) + " |" for row in primary_rows["S1"]),
        "",
        "**S2: 565 → 1501 → 1603 → 1384**",
        "",
        "| 架构 | 565 CIRCA ROUGE-L | 1603 SMCalFlow ROUGE-L | 1501 DSTC3 Accuracy | 1384 Deal-or-No-Deal Accuracy |\n| --- | ---: | ---: | ---: | ---: |\n" + "\n".join("| " + " | ".join(row) + " |" for row in primary_rows["S2"]),
        "",
        "S1 uses 565 → 1603 → 1501 → 1384. The first table is S1; the second table is S2. The detailed metric tables and full-resolution values are in `results/phase6b/pilot_task_metrics.csv`.",
        "",
        "## B2 versus parameter-matched B1",
        "",
        md_table(["任务", "主指标", "S1 Δ(B2−B1)", "S2 Δ(B2−B1)", "两顺序平均"], delta_rows),
        "",
        f"B2 was higher than B1 in {stable_wins}/{total_pairs} task-by-order primary-score comparisons. The direction is not stable: B2 loses on SMCalFlow in both orders, while the DSTC3 change reverses sign between orders. CIRCA's positive difference is small and both arms remain near zero. This single-seed, 50-example pilot does not establish a reliable B2 advantage.",
        "",
        "### Per-task final metrics",
        "",
    ]
    for sequence in SEQUENCES:
        report.extend([f"#### {sequence}", "", detail_tables[sequence], ""])

    report.extend([
        "## Continual-learning metrics",
        "",
        "Forgetting and backward transfer use the project’s normalized-gain definitions. Raw nonnegative forgetting remains separated by metric so Accuracy and ROUGE-L are not averaged together.",
        "",
        md_table(["顺序", "架构", "非负遗忘(归一化)", "ROUGE-L非负遗忘", "Accuracy非负遗忘", "BWT(归一化)", "Cross-stage AAA"], cl_rows),
        "",
        "### New-task acquisition",
        "",
        md_table(["顺序", "架构", "当前新任务", "原始主指标增益", "归一化增益"], acquisition_rows),
        "",
        "## Yes/No output collapse",
        "",
        md_table(["顺序/架构", "塌缩任务数", "检查任务数"], [[key, str(value[0]), str(value[1])] for key, value in collapse_counts.items()]),
        "",
        "The threshold is an exact Yes or No output share ≥0.95. This diagnostic catches degeneration that a task’s headline Accuracy or ROUGE-L alone can hide.",
        "",
        "## Routing mechanism and controls",
        "",
        "B2 takes the current token hidden state as its router input, selects one of four shared bottleneck experts per token, and receives no Task ID. At inference the route is deterministic top-1; training uses hard Gumbel selection with a straight-through gate gradient. The routed module therefore has input-dependent nonlinear dispatch, rather than a static linear reparameterization.",
        "",
        f"Across the 8 sequence/task evaluations, mean normalized route entropy was {fmt(route_summary.get('mean_normalized_entropy'), 3)} and mean dominant-expert share was {fmt(route_summary.get('mean_dominant_expert_share'), 3)}. Mean total-variation distance between the same task across S1/S2 was {fmt(route_summary.get('mean_same_task_tv_across_orders'), 3)}; between different tasks within an order it was {fmt(route_summary.get('mean_different_task_tv_within_order'), 3)}. These descriptive values mix task identity with order/training effects and are not a causal task-specialization test.",
        "",
        md_table(["顺序", "任务", "各层主导专家及占比", "平均归一化熵"], route_rows),
        "",
        "Primary metric and NLL results under final input/random/static-gate/off controls:",
        "",
        md_table(["顺序", "任务", "控制", "Input主指标", "控制主指标", "Δ控制−Input", "Input NLL", "控制 NLL"], control_rows),
        "",
        "On this 50-example dev set, random routing matched input routing on all 8 task/order primary scores. Static-gate and off controls also matched on most primary scores; the only visible primary-score change is on S2 SMCalFlow, where disabling the branch or using a static gate is better than input routing. NLL differences are small and task-dependent. This does not show that input-conditioned routing improves over random or static computation.",
        "",
        "### Gradient and branch participation traces",
        "",
        "The pilot sampled branch/router gradients at steps 1, 10, …, and the final step for each task. `branch_rms` is the residual delta magnitude at sampled forwards. The first zero-output initialization is expected; later nonzero branch RMS and parameter gradients show that branches participate in training. Full sampled traces are kept as `gradient_diagnostics.jsonl` in each run directory.",
        "",
        *gradient_lines,
        "",
        "## Capacity, compute, time, and memory",
        "",
        "FLOPs are linear multiply-add estimates for a 512-token prefill and exclude activation, token-dispatch, and cache costs. Wall time covers the recorded end-to-end run interval (training, dev evaluation, save/reload audit), not isolated GPU kernel time.",
        "",
        md_table(["顺序", "架构", "新增参数", "占主干", "估算额外FLOPs", "端到端秒", "峰值显存GiB", "平均新任务归一化增益"], resource_rows),
        "",
        "All adapter arms stayed below 1% added parameters and below the 24 GiB GPU limit. B1 adds 1,048,576 parameters; B2/B3 add 1,056,776 (0.177% of the 596,049,920-parameter backbone). The saved final states are full model states for replay/audit; no intermediate stage checkpoints or historical models were saved.",
        "",
        "## Decision",
        "",
        "Do not expand to seeds 20/30 or start NAS from this result. The predeclared seed-10 gate was not met: B2 is not consistently better than parameter-matched B1 across task orders, has no primary-score advantage over random routing in the measured controls, and sometimes performs worse than the static/off controls. The observed route distributions are descriptive and shift substantially with task order; they do not establish that routing mitigated representation interference.",
        "",
        "**Current evidence is insufficient to use Task-Conditional Residual Pathways as the architectural-search foundation of an ACL 2027 paper.** Keep it as an unvalidated hypothesis. A future replication should first use repeated seeds and a larger locked dev evaluation, and should require B2 to beat B1, random routing, and static/off controls on predeclared old-generation retention and new-classification metrics before NAS is considered.",
        "",
        "## Artifacts",
        "",
        "- `results/phase6b/pilot_summary.json` — consolidated S1/S2 run records and mechanism summary.",
        "- `results/phase6b/pilot_task_metrics.csv` — one row per task, sequence, and architecture.",
        "- `results/phase6b/pilot_resource_summary.csv` — parameter, FLOPs, runtime, memory, acquisition, forgetting, and BWT summary.",
        "- `figures/phase6b/pilot_final_primary_scores.png` — final dev primary scores.",
        "- `figures/phase6b/pilot_b2_vs_b1_delta.png` — paired primary-score deltas.",
        "- `figures/phase6b/pilot_b2_route_controls.png` — B2 route-mode controls.",
        "- `figures/phase6b/pilot_b2_route_usage.png` — per-task expert usage shares.",
    ])
    (REPORTS / "phase6b_pilot_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def consolidate_manifest(runs: dict[tuple[str, str], dict[str, Any]]) -> None:
    pilot_runs = {}
    baseline_reused = {}
    for sequence in SEQUENCES:
        a0_path = storage_path("results") / "phase4b" / "runs" / "A0" / sequence / "seed_10" / "metrics.json"
        a0 = read_json(a0_path)
        baseline_reused[sequence] = {
            "source_metrics": str(a0_path),
            "source_model_sha256": a0.get("source_model_sha256"),
            "task_order": a0.get("task_order"),
            "test_split_loaded": False,
        }
        for architecture in ARCHITECTURES[1:]:
            key = f"{sequence}/{architecture}/seed_10"
            run_path = OUT / "runs" / sequence / architecture / "seed_10" / "metrics.json"
            pilot_runs[key] = read_json(run_path)
    manifest = {
        "phase": "6B",
        "status": "complete",
        "sequences": list(SEQUENCES),
        "architectures": list(ARCHITECTURES[1:]),
        "seed": 10,
        "test_split_loaded": False,
        "nas_started": False,
        "pilot_runs": pilot_runs,
        "baseline_reused": baseline_reused,
        "consolidated_from_per_run_metrics": True,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    for name in ("pilot_manifest.json", "pilot_manifest_partial.json"):
        (OUT / name).write_text(text, encoding="utf-8")


def main() -> None:
    runs = {(sequence, architecture): load_run(sequence, architecture) for sequence in SEQUENCES for architecture in ARCHITECTURES}
    phase_config = yaml.safe_load((ROOT / "configs" / "phase6b.yaml").read_text(encoding="utf-8"))
    architecture_config = phase_config["architecture"]
    selected_random_layers = select_random_post_layers(
        list(architecture_config["random_location_candidate_layers"]),
        len(architecture_config["primary_layers"]),
        int(architecture_config["random_location_selection_seed"]),
        list(architecture_config["primary_layers"]),
    )
    declared_random_layers = sorted(int(layer) for layer in architecture_config["random_location_layers"])
    if selected_random_layers != declared_random_layers:
        raise RuntimeError(f"Configured random-location pair disagrees with seed: {selected_random_layers} != {declared_random_layers}")
    for sequence in SEQUENCES:
        stored_layers = runs[(sequence, "B2-random-location")]["layer_indices_zero_based"]
        if stored_layers != selected_random_layers:
            raise RuntimeError(f"Stored random-location run does not match configured draw: {sequence} {stored_layers} != {selected_random_layers}")
    hashes = {run["source_model_sha256"] for run in runs.values() if run["source_model_sha256"]}
    if len(hashes) != 1:
        raise RuntimeError(f"Runs do not share one base-model hash: {hashes}")
    for (sequence, architecture), run in runs.items():
        if architecture != "B0" and (not run["initialization_exact"] or run["checkpoint_reload_pass"] is not True):
            raise RuntimeError(f"Initialization/reload audit failed for {sequence}/{architecture}")
        if run["historical_model_loaded"] or run["replay_used"] or run["nas_started"]:
            raise RuntimeError(f"Forbidden pilot state recorded for {sequence}/{architecture}")
        if architecture != "B0" and (not run["sample_order_matches_A0"] or not run["supervised_target_tokens_matches_A0"]):
            raise RuntimeError(f"Training sample/token budget mismatch for {sequence}/{architecture}")
    route_summary = route_distance_summary(runs)
    route_summary["random_location_selection"] = {
        "candidate_layers_zero_based": architecture_config["random_location_candidate_layers"],
        "excluded_primary_layers_zero_based": architecture_config["primary_layers"],
        "seed": architecture_config["random_location_selection_seed"],
        "selected_layers_zero_based": selected_random_layers,
    }
    make_figures(runs, route_summary)
    write_reports(runs, route_summary)
    consolidate_manifest(runs)
    print(f"Consolidated {len(runs)} runs; 8 adapter runs and 2 B0 references.")
    print(f"B2 beats B1 on {sum(runs[(s, 'B2-input')]['primary_scores'][t] > runs[(s, 'B1-static')]['primary_scores'][t] for s in SEQUENCES for t in TASKS)}/8 paired task/order primary scores.")
    print(f"Wrote {REPORTS / 'phase6b_pilot_results.md'} and four figures under {FIGURES}.")


if __name__ == "__main__":
    main()
