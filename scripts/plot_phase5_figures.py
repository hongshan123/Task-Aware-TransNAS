"""Render publication-ready Phase 5 figures from cached experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from figure_qa.audit_panel_alignment import require_matplotlib_panel_alignment
from phase5_common import FIGURE_ROOT, PHASE5_ROOT, TASK_IDS


FINAL_CHECKPOINT = "phase4b_A0_S1_seed10"
STAGE_ORDER = [
    "pretrained",
    "homogeneous_stage_1",
    "homogeneous_stage_2",
    "homogeneous_stage_3",
    "homogeneous_stage_4",
]
STAGE_LABELS = ["Base", "565", "1603", "1501", "1384"]
TASK_LABELS = {
    "task565_circa_answer_generation": "565 CIRCA",
    "task1603_smcalflow_sentence_generation": "1603 SMCalFlow",
    "task1501_dstc3_answer_generation": "1501 DSTC3",
    "task1384_deal_or_no_dialog_classification": "1384 Deal/No Deal",
}
TASK_COLORS = {
    "task565_circa_answer_generation": "#0072B2",
    "task1603_smcalflow_sentence_generation": "#D55E00",
    "task1501_dstc3_answer_generation": "#009E73",
    "task1384_deal_or_no_dialog_classification": "#CC79A7",
}


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _panel_label(axis: Any, label: str) -> None:
    axis.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-17, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        annotation_clip=False,
    )


def _save_figure(fig: Any, stem: str, colorbar_axes: list[Any] | None = None) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    base = FIGURE_ROOT / stem
    require_matplotlib_panel_alignment(
        fig,
        json_out=base.with_suffix(".alignment.json"),
        overlay_svg=base.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=len(fig.axes) - len(colorbar_axes or []) > 1,
        strict=True,
        exclude_axes=colorbar_axes or [],
    )
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(
    final_id: str,
    final_rankings: dict[str, list[dict[str, Any]]],
    task_cosines: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    stage_nll_rows: list[dict[str, Any]],
) -> None:
    """Render figures and their traceable source tables from in-memory results."""
    _configure_style()
    figure_data_root = PHASE5_ROOT / "figure_data"

    # Figure 1: all tasks share one symmetric scale centered at zero.
    layers, heads = 28, 16
    matrices: dict[str, np.ndarray] = {}
    head_source_rows: list[dict[str, Any]] = []
    all_values: list[float] = []
    for task_id in TASK_IDS:
        matrix = np.full((layers, heads), np.nan, dtype=float)
        for row in final_rankings[f"{final_id}__{task_id}"]:
            layer = int(row["layer"])
            head = int(row["query_head"])
            gain = float(row["ce_mean_sequence_logp_gain"])
            matrix[layer, head] = gain
            all_values.append(gain)
            head_source_rows.append(
                {
                    "checkpoint_id": final_id,
                    "task_id": task_id,
                    "task_label": TASK_LABELS[task_id],
                    "layer": layer,
                    "query_head": head,
                    "mean_sequence_log_probability_gain": gain,
                }
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"Head ranking for {task_id} does not cover all {layers * heads} heads")
        matrices[task_id] = matrix
    _write_csv(
        figure_data_root / "phase5_head_importance.csv",
        [
            "checkpoint_id",
            "task_id",
            "task_label",
            "layer",
            "query_head",
            "mean_sequence_log_probability_gain",
        ],
        head_source_rows,
    )

    color_limit = max(float(np.max(np.abs(all_values))), 1e-12)
    shared_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    image = None
    for index, (axis, task_id) in enumerate(zip(axes.flat, TASK_IDS)):
        image = axis.imshow(
            matrices[task_id],
            aspect="auto",
            cmap="coolwarm",
            norm=shared_norm,
            interpolation="nearest",
        )
        axis.set_title(TASK_LABELS[task_id], pad=5)
        axis.set_xlabel("Query head")
        axis.set_ylabel("Layer")
        axis.set_xticks([0, 4, 8, 12, 15])
        axis.set_yticks([0, 4, 8, 12, 16, 20, 24, 27])
        _panel_label(axis, "abcd"[index])
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=list(axes.flat),
        fraction=0.035,
        pad=0.035,
        label="Mean sequence log-probability gain",
    )
    colorbar.ax.tick_params(labelsize=6)
    fig.suptitle("Causal importance of query heads at the Phase 4B A0 final checkpoint", fontsize=9)
    _save_figure(fig, "phase5_head_importance_heatmap", colorbar_axes=[colorbar.ax])

    # Figure 2: task-vector similarities at the Phase 4B endpoint.
    selected_cosines = [
        row for row in task_cosines if row["checkpoint_id"] == FINAL_CHECKPOINT
    ]
    order = list(TASK_IDS)
    index_by_task = {task_id: index for index, task_id in enumerate(order)}
    cosine_matrix = np.full((len(order), len(order)), np.nan, dtype=float)
    cosine_source_rows: list[dict[str, Any]] = []
    for row in selected_cosines:
        task_a, task_b = row["task_a"], row["task_b"]
        value = float(row["cosine"])
        cosine_matrix[index_by_task[task_a], index_by_task[task_b]] = value
        cosine_source_rows.append(
            {
                "checkpoint_id": row["checkpoint_id"],
                "task_a": task_a,
                "task_a_label": TASK_LABELS[task_a],
                "task_b": task_b,
                "task_b_label": TASK_LABELS[task_b],
                "cosine_similarity": value,
            }
        )
    if len(selected_cosines) != len(order) ** 2 or not np.isfinite(cosine_matrix).all():
        raise ValueError("The final-checkpoint Function Vector cosine matrix is incomplete")
    _write_csv(
        figure_data_root / "phase5_fv_cosine_matrix.csv",
        [
            "checkpoint_id",
            "task_a",
            "task_a_label",
            "task_b",
            "task_b_label",
            "cosine_similarity",
        ],
        cosine_source_rows,
    )
    fig, axis = plt.subplots(figsize=(3.5, 3.35), constrained_layout=True)
    image = axis.imshow(cosine_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    short_labels = ["565", "1603", "1501", "1384"]
    axis.set_xticks(range(len(order)), short_labels)
    axis.set_yticks(range(len(order)), short_labels)
    axis.set_xlabel("Task ID")
    axis.set_ylabel("Task ID")
    for row_index in range(len(order)):
        for column_index in range(len(order)):
            axis.text(
                column_index,
                row_index,
                f"{cosine_matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    axis.set_title("Phase 4B Function Vector cosine", pad=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Cosine")
    colorbar.ax.tick_params(labelsize=6)
    _save_figure(fig, "phase5_fv_cosine_matrix", colorbar_axes=[colorbar.ax])

    # Figure 3: stage-wise drift and held-out dev behavior.
    nll_by_key = {
        (row["checkpoint_id"], row["task_id"]): float(row["teacher_forced_token_nll"])
        for row in stage_nll_rows
        if row["checkpoint_id"] in STAGE_ORDER
    }
    drift_by_key = {
        (row["checkpoint_id"], row["task_id"]): row for row in drift_rows
    }
    stage_source_rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        for stage_index, checkpoint_id in enumerate(STAGE_ORDER):
            key = (checkpoint_id, task_id)
            if key not in nll_by_key or key not in drift_by_key:
                raise ValueError(f"Missing stage trajectory values for {checkpoint_id}, {task_id}")
            drift = drift_by_key[key]
            stage_source_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "stage_label": STAGE_LABELS[stage_index],
                    "task_id": task_id,
                    "task_label": TASK_LABELS[task_id],
                    "cosine_to_pretrained_function_vector": float(
                        drift["cosine_to_pretrained_fv"]
                    ),
                    "relative_l2_function_vector_drift": float(
                        drift["relative_l2_drift"]
                    ),
                    "held_out_dev_teacher_forced_token_nll": nll_by_key[key],
                    "dev_evaluation_count_per_task": 25,
                }
            )
    _write_csv(
        figure_data_root / "phase5_stage_drift_vs_nll.csv",
        [
            "checkpoint_id",
            "stage_label",
            "task_id",
            "task_label",
            "cosine_to_pretrained_function_vector",
            "relative_l2_function_vector_drift",
            "held_out_dev_teacher_forced_token_nll",
            "dev_evaluation_count_per_task",
        ],
        stage_source_rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    x = np.arange(len(STAGE_ORDER))
    for task_id in TASK_IDS:
        task_drift = [
            drift_by_key[(checkpoint_id, task_id)] for checkpoint_id in STAGE_ORDER
        ]
        task_nll = [nll_by_key[(checkpoint_id, task_id)] for checkpoint_id in STAGE_ORDER]
        axes[0].plot(
            x,
            [row["cosine_to_pretrained_fv"] for row in task_drift],
            marker="o",
            color=TASK_COLORS[task_id],
            label=TASK_LABELS[task_id],
        )
        axes[1].plot(
            x,
            task_nll,
            marker="o",
            color=TASK_COLORS[task_id],
            label=TASK_LABELS[task_id],
        )
    for axis, letter in zip(axes, "ab"):
        axis.set_xticks(x, STAGE_LABELS)
        axis.set_xlabel("Homogeneous A0 training stage")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
        _panel_label(axis, letter)
    axes[0].set_title("Function Vector cosine to pretrained", pad=7)
    axes[0].set_ylabel("Cosine")
    axes[0].set_ylim(-1.05, 1.05)
    axes[1].set_title("Held-out dev teacher-forced NLL", pad=7)
    axes[1].set_ylabel("Token NLL")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.18, 1, 0.94), w_pad=2.4)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        fontsize=6,
        columnspacing=1.8,
        handlelength=1.8,
    )
    fig.suptitle("Function Vector drift and held-out behavior across continual training", fontsize=9)
    _save_figure(fig, "phase5_stage_drift_vs_nll")


def main() -> None:
    fv_root = PHASE5_ROOT / "fv"
    rankings = json.loads((fv_root / "head_importance_ranked.json").read_text(encoding="utf-8"))
    cosines = json.loads((fv_root / "fv_cosines.json").read_text(encoding="utf-8"))
    drift = json.loads((fv_root / "fv_drift.json").read_text(encoding="utf-8"))
    stage_rows = json.loads(
        (PHASE5_ROOT / "stage_eval" / "stage_metrics.json").read_text(encoding="utf-8")
    )
    stage_nll = [
        {
            "checkpoint_id": row["checkpoint_id"],
            "task_id": row["task_id"],
            "teacher_forced_token_nll": row["teacher_forced_token_nll"],
        }
        for row in stage_rows
        if row["checkpoint_id"] in STAGE_ORDER and row["task_id"] in TASK_IDS
    ]
    save_figures(FINAL_CHECKPOINT, rankings, cosines, drift, stage_nll)
    print(f"Rendered three Phase 5 figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
