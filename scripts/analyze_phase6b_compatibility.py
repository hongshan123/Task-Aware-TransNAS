"""Build complete 4x4 Phase 6B compatibility matrices from locked Phase 6A evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
try:
    from .runtime_paths import storage_root
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import storage_root

load_dotenv(ROOT / ".env", override=False)
STORAGE_ROOT = storage_root(ROOT)
P6A = storage_path("results") / "phase6a"
OUT = storage_path("results") / "phase6b"
FIGURES = storage_path("figures") / "phase6b"
REPORT = storage_path("reports") / "phase6b_compatibility.md"
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
LABEL = dict(zip(TASKS, ("565 CIRCA", "1603 SMCalFlow", "1501 DSTC3", "1384 Deal-or-No-Deal")))
GENERATION = set(TASKS[:2])
TASK_RELATIONS = STORAGE_ROOT / "data" / "task_selection" / "task_relation_analysis.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        average_rank = (cursor + 1 + stop) / 2.0
        ranks[order[cursor:stop]] = average_rank
        cursor = stop
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _component_name(component: str) -> str:
    return {
        "transformer_block_output": "residual_stream",
        "attention_o_proj_output": "attention",
        "mlp_output": "mlp",
    }.get(component, component)


def _parameter_component(candidate_id: str) -> str:
    for marker in ("_global", "_group_", "_layer_"):
        if marker in candidate_id:
            candidate_id = candidate_id.split(marker, 1)[0]
    return candidate_id


def _baseline_metric(summary: dict[str, Any], task_id: str) -> tuple[str, float, float]:
    row = summary["activation"][task_id]["baseline"]
    primary = "rouge_l_f1" if task_id in GENERATION else "accuracy"
    return primary, float(row[primary]), float(row["teacher_forced_token_nll"])


def _matrix_row(
    kind: str,
    source: str,
    target: str,
    primary_delta: float,
    nll_gain: float,
    target_base_primary: float,
    target_base_nll: float,
    operation: str,
    condition: str,
    examples: int,
    provenance: str,
) -> dict[str, Any]:
    primary_metric = "rouge_l_f1" if target in GENERATION else "accuracy"
    return {
        "intervention_type": kind,
        "source_task": source,
        "target_task": target,
        "source_operation": operation,
        "condition": condition,
        "target_primary_metric": primary_metric,
        "target_baseline_primary": target_base_primary,
        "primary_delta_Qj": primary_delta,
        "primary_delta_normalized_by_baseline_headroom": primary_delta / max(1e-8, 1.0 - target_base_primary),
        "target_baseline_token_nll": target_base_nll,
        "token_nll_improvement": nll_gain,
        "examples": examples,
        "provenance": provenance,
    }


def _diag_rows(summary: dict[str, Any], activation_selection: dict, parameter_selection: dict) -> list[dict[str, Any]]:
    rows = []
    for task_id in TASKS:
        primary, baseline_primary, baseline_nll = _baseline_metric(summary, task_id)
        selected_act = next(
            row for row in activation_selection["all_candidates"]
            if row["task_id"] == task_id and row["candidate_id"] != "baseline"
            and row["scope"] not in ("baseline", "control")
            and row["candidate_type"] != "lm_head_logits_upper_bound"
            and row["candidate_id"] == max(
                (
                    candidate for candidate in activation_selection["all_candidates"]
                    if candidate["task_id"] == task_id and candidate["scope"] not in ("baseline", "control")
                    and candidate["candidate_type"] != "lm_head_logits_upper_bound"
                ),
                key=lambda candidate: float(candidate["nll_improvement_vs_baseline"]),
            )["candidate_id"]
        )
        act_result = summary["activation"][task_id][selected_act["candidate_id"]]
        rows.append(_matrix_row(
            "activation", task_id, task_id,
            float(act_result[primary]) - baseline_primary,
            baseline_nll - float(act_result["teacher_forced_token_nll"]),
            baseline_primary, baseline_nll,
            _component_name(str(selected_act["candidate_type"])),
            selected_act["candidate_id"], 25, "Phase 6A locked diagonal confirmation",
        ))

        selected_param = max(
            (
                row for row in parameter_selection["all_candidates"]
                if row["task_id"] == task_id and row["candidate_id"] != "baseline"
            ),
            key=lambda row: float(row["nll_improvement_vs_baseline"]),
        )
        param_result = summary["parameter_rollback"][task_id]
        parameter_key = selected_param["candidate_id"]
        if parameter_key not in param_result:
            # The all-layer MLP screen has equivalent `mlp_global` and
            # `mlp_group_28_00_27` spellings; Phase 6A confirmation picked one.
            parameter_key = next(
                key for key in param_result if key != "matched_random_delta"
                and ((key.startswith("mlp_") and selected_param["candidate_id"].startswith("mlp_"))
                     or (key == selected_param["candidate_id"]))
            )
        param_result = param_result[parameter_key]
        rows.append(_matrix_row(
            "parameter_rollback", task_id, task_id,
            float(param_result[primary]) - baseline_primary,
            baseline_nll - float(param_result["teacher_forced_token_nll"]),
            baseline_primary, baseline_nll,
            _parameter_component(parameter_key), parameter_key, 25,
            "Phase 6A locked diagonal confirmation",
        ))
    return rows


def _off_diagonal_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sources = _csv(P6A / "cross_task_transfer_summary.csv")
    sources += _csv(OUT / "compatibility_missing_activation_summary.csv")
    sources += _csv(OUT / "compatibility_missing_parameter_rollback_summary.csv")
    for row in sources:
        source, target = row["source_task"], row["target_task"]
        _, base_primary, base_nll = _baseline_metric(summary, target)
        condition = row["condition"]
        if row["intervention_type"] == "activation":
            candidate_id = condition.rsplit("_from_", 1)[0]
            candidate = next(
                item for item in _json(P6A / "activation_screen_selection.json")["all_candidates"]
                if item["task_id"] == source and item["candidate_id"] == candidate_id
            )
            operation = _component_name(str(candidate["candidate_type"]))
        else:
            candidate_id = condition.rsplit("_from_", 1)[0]
            operation = _parameter_component(candidate_id)
        rows.append(_matrix_row(
            row["intervention_type"], source, target,
            float(row["primary_metric_delta_vs_target_baseline"]),
            -float(row["teacher_forced_nll_delta_vs_target_baseline"]),
            base_primary, base_nll, operation, condition,
            int(row["examples"]),
            "Phase 6A cross-task result" if "compatibility_missing" not in row.get("source_checkpoint", "") else "Phase 6B missing edge",
        ))
    # Source file location is tracked by manifest, not source_checkpoint; repair
    # provenance labels for the eight new within-type measurements.
    for row in rows:
        if (OUT / "compatibility_missing_manifest.json").exists():
            missing = _json(OUT / "compatibility_missing_manifest.json")
            if any(item["source"] == row["source_task"] and item["target"] == row["target_task"] and item["type"] == row["intervention_type"] for item in missing["pairs"]):
                row["provenance"] = "Phase 6B missing same-type edge"
    return rows


def _compatibility_correlations(rows: list[dict[str, Any]], relation: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    rep = relation["representation_cosine_matrix"]
    grad = relation["gradient_cosine_matrix"]
    directed = [row for row in rows if row["source_task"] != row["target_task"]]
    for kind in ("activation", "parameter_rollback"):
        selected = [row for row in directed if row["intervention_type"] == kind]
        repr_sims = [float(rep[row["source_task"]][row["target_task"]]) for row in selected]
        gradient_cos = [float(grad[row["source_task"]][row["target_task"]]) for row in selected]
        nll_benefit = [float(row["token_nll_improvement"]) for row in selected]
        conflict = [max(0.0, -value) for value in gradient_cos]
        results[kind] = {
            "directed_off_diagonal_pairs": len(selected),
            "spearman_representation_cosine_vs_nll_compatibility": _spearman(repr_sims, nll_benefit),
            "spearman_gradient_cosine_vs_nll_compatibility": _spearman(gradient_cos, nll_benefit),
            "spearman_negative_gradient_conflict_magnitude_vs_nll_compatibility": _spearman(conflict, nll_benefit),
            "representation_cosines": repr_sims,
            "gradient_cosines": gradient_cos,
            "nll_compatibility_values": nll_benefit,
            "negative_gradient_cosine_pair_count": sum(value < 0 for value in gradient_cos) // 2,
            "interpretation": "Descriptive rank correlations over four tasks/12 directed pairs; paired directions and task-specific intervention types make these non-independent, exploratory evidence only.",
        }
    return results


def _make_heatmaps(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = ["565", "1603", "1501", "1384"]
    for metric, file_name, title, cmap in (
        ("primary_delta_Qj", "compatibility_primary_heatmap.png", "Δ task primary metric: patched − final A0", "RdBu"),
        ("token_nll_improvement", "compatibility_nll_heatmap.png", "Token NLL improvement: final A0 − patched", "RdBu"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, kind, title_prefix in zip(axes, ("activation", "parameter_rollback"), ("Activation recovery", "Parameter rollback")):
            matrix = np.full((4, 4), np.nan)
            for row in rows:
                if row["intervention_type"] == kind:
                    y, x = TASKS.index(row["source_task"]), TASKS.index(row["target_task"])
                    matrix[y, x] = float(row[metric])
            limit = max(0.05, float(np.nanmax(np.abs(matrix))))
            image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            for y in range(4):
                for x in range(4):
                    if np.isfinite(matrix[y, x]):
                        ax.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", fontsize=9,
                                color="white" if abs(matrix[y, x]) > limit * 0.6 else "black")
            ax.set_xticks(range(4), labels)
            ax.set_yticks(range(4), labels)
            ax.set_xlabel("Evaluated task j")
            ax.set_ylabel("Restoration source task i")
            ax.set_title(title_prefix)
            fig.colorbar(image, ax=ax, shrink=0.8)
        fig.suptitle(title)
        fig.savefig(FIGURES / file_name, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _make_relation_plot(rows: list[dict[str, Any]], relation: dict[str, Any]) -> None:
    rep = relation["representation_cosine_matrix"]
    grad = relation["gradient_cosine_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {"activation": "#2b8cbe", "parameter_rollback": "#e34a33"}
    markers = {"residual_stream": "s", "attention": "o", "mlp": "^", "attention_plus_mlp": "D"}
    for row in rows:
        if row["source_task"] == row["target_task"]:
            continue
        x = float(rep[row["source_task"]][row["target_task"]])
        y = float(row["token_nll_improvement"])
        conflict = max(0.0, -float(grad[row["source_task"]][row["target_task"]]))
        size = 35 + 260 * conflict
        for ax in axes:
            ax.scatter(x, y, color=colors[row["intervention_type"]], marker=markers.get(row["source_operation"], "o"), s=size, alpha=0.72)
    axes[0].set_xlabel("Frozen prompt representation cosine")
    axes[0].set_ylabel("NLL compatibility (positive = improves target)")
    axes[0].set_title("Representation similarity vs transfer effect")
    axes[1].set_xlabel("Frozen task gradient cosine")
    axes[1].set_ylabel("NLL compatibility (positive = improves target)")
    axes[1].set_title("Gradient alignment vs transfer effect")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[1].axhline(0, color="black", linewidth=0.7)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[kind], label=kind.replace("_", " "))
        for kind in colors
    ]
    handles.extend(
        plt.Line2D([0], [0], marker=marker, linestyle="", color="#555555", label=label)
        for label, marker in markers.items()
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(FIGURES / "similarity_gradient_compatibility.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = _json(P6A / "confirmation_summary.json")
    activation_selection = _json(P6A / "activation_screen_selection.json")
    parameter_selection = _json(P6A / "parameter_screen_selection.json")
    relation = _json(TASK_RELATIONS)
    rows = _diag_rows(summary, activation_selection, parameter_selection) + _off_diagonal_rows(summary)
    for kind in ("activation", "parameter_rollback"):
        matrix_rows = [row for row in rows if row["intervention_type"] == kind]
        if len(matrix_rows) != 16 or len({(row["source_task"], row["target_task"]) for row in matrix_rows}) != 16:
            raise RuntimeError(f"Incomplete 4x4 {kind} matrix: {len(matrix_rows)} rows")
    if not all(task in relation["representation_cosine_matrix"] and task in relation["gradient_cosine_matrix"] for task in TASKS):
        raise RuntimeError("The frozen task relation matrices do not contain all four pilot tasks")
    _write_csv(OUT / "compatibility_matrices.csv", rows)
    similarity_rows = []
    for source in TASKS:
        for target in TASKS:
            similarity_rows.append({
                "source_task": source,
                "target_task": target,
                "representation_cosine": relation["representation_cosine_matrix"][source][target],
                "gradient_cosine": relation["gradient_cosine_matrix"][source][target],
                "gradient_conflict_magnitude": max(0.0, -float(relation["gradient_cosine_matrix"][source][target])),
            })
    _write_csv(OUT / "task_similarity_gradient_matrix.csv", similarity_rows)
    correlations = _compatibility_correlations(rows, relation)
    components = {}
    for operation in ("residual_stream", "attention", "mlp", "attention_plus_mlp"):
        selected = [row for row in rows if row["intervention_type"] == "activation" and row["source_operation"] == operation and row["source_task"] != row["target_task"]]
        if selected:
            components[operation] = {
                "source_tasks": sorted({row["source_task"] for row in selected}),
                "directed_transfer_count": len(selected),
                "mean_target_token_nll_improvement": float(np.mean([row["token_nll_improvement"] for row in selected])),
                "positive_transfer_fraction": float(np.mean([row["token_nll_improvement"] > 0 for row in selected])),
                "mean_target_primary_delta": float(np.mean([row["primary_delta_Qj"] for row in selected])),
            }
    manifest = {
        "phase": "6B",
        "artifact": "cross-task recovery compatibility matrices",
        "matrix_definition": "C_ij = Q_j(Patch_i(M)) - Q_j(M); primary delta uses target-native ROUGE-L or accuracy; NLL compatibility is baseline token NLL minus patched token NLL.",
        "model_M": "final A0/S1/seed10 checkpoint after task1384",
        "source_references": {
            "generation_tasks": "historical after-task1603 checkpoint",
            "classification_tasks": "historical after-task1501 checkpoint",
        },
        "locked_dev_indices": [25, 49],
        "matrix_cells_per_intervention": 16,
        "complete_matrices": {kind: True for kind in ("activation", "parameter_rollback")},
        "diagonal_provenance": "Phase 6A locked confirmation; read-only",
        "cross_type_edge_provenance": "Phase 6A cross-task transfer; read-only",
        "same_type_edge_provenance": "Phase 6B supplement; 8 conditions and 200 per-example records",
        "task_similarity_source": str(TASK_RELATIONS),
        "task_similarity_split": "order1 train examples; no dev/test used in similarity computation",
        "task_relation_gradient_parameters": relation["gradient_parameter_group"],
        "correlations": correlations,
        "activation_component_compatibility": components,
        "historical_model_information": "The patch condition imports activations/weights from the historical boundary. It is not the target model's native retained capability. Phase 6B B1/B2/B3 pilot does not load or forward historical models.",
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "nas_started": False,
    }
    (OUT / "compatibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _make_heatmaps(rows)
    _make_relation_plot(rows, relation)

    activation_matrix = {(row["source_task"], row["target_task"]): row for row in rows if row["intervention_type"] == "activation"}
    parameter_matrix = {(row["source_task"], row["target_task"]): row for row in rows if row["intervention_type"] == "parameter_rollback"}
    table = [
        "| Source operation | Source task | Diagonal NLL recovery | Mean off-diagonal NLL benefit | Positive transfer targets |",
        "|---|---|---:|---:|---|",
    ]
    for source in TASKS:
        row = activation_matrix[(source, source)]
        offdiag = [activation_matrix[(source, target)] for target in TASKS if target != source]
        positive = ", ".join(LABEL[target] for target in TASKS if target != source and activation_matrix[(source, target)]["token_nll_improvement"] > 0) or "none"
        table.append(
            f"| {row['source_operation']} | {LABEL[source]} | {row['token_nll_improvement']:+.3f} | {np.mean([item['token_nll_improvement'] for item in offdiag]):+.3f} | {positive} |"
        )
    offdiag_rows = [row for row in rows if row["source_task"] != row["target_task"]]
    negative_activation = sum(row["token_nll_improvement"] < 0 for row in offdiag_rows if row["intervention_type"] == "activation")
    negative_parameter = sum(row["token_nll_improvement"] < 0 for row in offdiag_rows if row["intervention_type"] == "parameter_rollback")
    report = f"""# Phase 6B: Cross-Task Representation Compatibility

## Protocol and scope

This matrix uses final A0/S1/seed-10 as the target M and the historical Phase 6A boundaries as reference sources: the after-task1603 checkpoint for tasks565/1603 and after-task1501 for tasks1501/1384. `C_ij` is reported in two forms: target-native primary score difference (ROUGE-L for generation, accuracy for classification) and homogeneous token-NLL improvement (final baseline NLL minus patched NLL, positive means help). Every cell uses the same locked dev IDs 25–49. The four diagonals reuse the Phase 6A locked confirmation; Phase 6A cross-type edges are reused read-only; this Phase 6B supplement measured the remaining eight same-type off-diagonal conditions (200 per-example rows).

The historical source models provide the restoration information in each `Patch_i`. This score is not a claim that the final model natively retains that behavior. The subsequent conditional residual pilot starts from the final pretrained model and does not load or forward historical checkpoints. Only train/dev data were read; no optimizer steps, test split, or NAS were used for this compatibility analysis.

## Compatibility results

Activation patching produced negative off-diagonal NLL transfer in {negative_activation}/12 directed task pairs; parameter rollback did so in {negative_parameter}/12. Signs are defined per target; lower NLL is a positive compatibility effect. Primary metric changes remain target-native and should not be averaged across ROUGE-L and accuracy.

{chr(10).join(table)}

The selected activation source operations are task-specific: residual-stream block output for tasks565/1603, Attention output for task1501, and MLP output for task1384. No single operation is uniformly compatible across the full matrix. The diagonal values show how much historically sourced signal helps its own task; off-diagonal values show that signal's transfer cost/benefit to the other tasks. This supports sharing a small residual computation through conditional routing rather than sharing historical activations or weights directly.

## Similarity and gradient relation

The prompt-representation and gradient-cosine matrices come from the existing train-only task-relation analysis (frozen Qwen3-0.6B, 32 train prompts for representations and 8 for gradients of layer-27 Attention `o_proj` plus MLP `down_proj`). Negative gradient cosine marks first-order conflict. Over only 12 directed off-diagonal pairs, Spearman correlations between those similarities and NLL compatibility are: activation—representation {correlations['activation']['spearman_representation_cosine_vs_nll_compatibility']:.3f}, gradient cosine {correlations['activation']['spearman_gradient_cosine_vs_nll_compatibility']:.3f}, negative-conflict magnitude {correlations['activation']['spearman_negative_gradient_conflict_magnitude_vs_nll_compatibility']:.3f}; parameter rollback—representation {correlations['parameter_rollback']['spearman_representation_cosine_vs_nll_compatibility']:.3f}, gradient cosine {correlations['parameter_rollback']['spearman_gradient_cosine_vs_nll_compatibility']:.3f}, negative-conflict magnitude {correlations['parameter_rollback']['spearman_negative_gradient_conflict_magnitude_vs_nll_compatibility']:.3f}. These are exploratory, non-independent estimates; they do not establish that similarity or gradient alignment causes transfer compatibility.

The selected task pairs are not explained by one scalar relation: task565 and task1603 have high representation similarity ({float(relation['representation_cosine_matrix'][TASKS[0]][TASKS[1]]):.3f}) but modest positive gradient cosine ({float(relation['gradient_cosine_matrix'][TASKS[0]][TASKS[1]]):.3f}); task1501 and task1384 have high representation similarity ({float(relation['representation_cosine_matrix'][TASKS[2]][TASKS[3]]):.3f}) and strong gradient alignment ({float(relation['gradient_cosine_matrix'][TASKS[2]][TASKS[3]]):.3f}). Cross-task restoration outcomes still depend on the selected component and target metric.

## Artifacts and limits

- [Primary compatibility matrix](../figures/phase6b/compatibility_primary_heatmap.png)
- [Token-NLL compatibility matrix](../figures/phase6b/compatibility_nll_heatmap.png)
- [Similarity/gradient relation](../figures/phase6b/similarity_gradient_compatibility.png)
- Complete values: [compatibility_matrices.csv](../results/phase6b/compatibility_matrices.csv), [compatibility_manifest.json](../results/phase6b/compatibility_manifest.json)

The patch operations are selected screen winners and the dev sample is small. Within-task diagonal source operations vary by task, so the matrix is a compatibility audit of those selected recoveries rather than a fully crossed experiment over all components at all layers. Similarity estimates are computed from training examples; the transfer matrix is on dev examples. The results justify testing input-conditioned residual paths as a new architecture, but do not by themselves justify NAS.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"matrix_rows": len(rows), "complete_4x4": manifest["complete_matrices"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
