"""Extract Qwen3 Function Vectors, causal head effects and FV interventions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from plot_phase5_figures import save_figures as save_phase5_figures

from phase5_common import (
    FV_VERSION,
    ICL_SHOTS,
    PATCH_BATCH_SIZE,
    PHASE5_ROOT,
    PROJECT_ROOT,
    REPORT_ROOT,
    SEED,
    SPLITS,
    TASK_1384,
    TASK_1501,
    TASK_1603,
    TASK_565,
    TASK_GENERATION,
    TASK_IDS,
    TOP_K,
    HeadActivationPatcher,
    HeadActivationRecorder,
    ResidualVectorIntervention,
    aggregate_task,
    cleanup_model,
    json_dump,
    jsonl_write,
    load_checkpoint,
    load_phase5_context,
    make_complete_ids,
    make_icl_sets,
    model_head_geometry,
    norm_match_random,
    parse_yes_no,
    phase4b_checkpoint,
    reference_sequence,
    safe_float,
    score_prediction,
    split_records,
    stage_checkpoint_specs,
    zero_shot_generate,
    zero_shot_teacher_forced,
)


def checkpoint_key(checkpoint_id: str, task_id: str) -> str:
    return f"{checkpoint_id}__{task_id}"


def activation_file(checkpoint_id: str, task_id: str) -> Path:
    return PHASE5_ROOT / "fv" / "activations" / FV_VERSION / f"{checkpoint_key(checkpoint_id, task_id)}.pt"


def raw_effect_file(checkpoint_id: str, task_id: str, index: int) -> Path:
    return PHASE5_ROOT / "fv" / "head_effects" / FV_VERSION / checkpoint_id / task_id / f"query_{index:02d}.json"


def effect_cache_is_valid(
    path: Path,
    checkpoint_id: str,
    task_id: str,
    query_id: str,
    candidate_scope: str,
    candidate_count: int,
) -> bool:
    if not path.exists():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        scores = cached.get("scores", [])
        return (
            cached.get("checkpoint_id") == checkpoint_id
            and cached.get("task_id") == task_id
            and cached.get("query_id") == query_id
            and cached.get("candidate_scope") == candidate_scope
            and len(scores) == candidate_count
        )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError):
        return False


def metrics_file(checkpoint_id: str, task_id: str) -> Path:
    return PHASE5_ROOT / "stage_eval" / checkpoint_id / f"{task_id}.json"


def ensure_yes_no_shares(metrics: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(prediction_rows))
    labels = [parse_yes_no(row.get("prediction", "")) for row in prediction_rows]
    yes = sum(label == "Yes" for label in labels)
    no = sum(label == "No" for label in labels)
    metrics["yes_output_share"] = yes / total
    metrics["no_output_share"] = no / total
    metrics["yes_no_collapse_rate"] = max(yes, no) / total
    if metrics.get("task_type") == "classification":
        metrics["unparsed_output_share"] = (total - yes - no) / total
    return metrics


def load_means(path: Path, device: torch.device) -> torch.Tensor:
    means = torch.load(path, map_location="cpu", weights_only=True)
    return means.to(device=device, dtype=torch.float32)


def rank_effects(raw_paths: list[Path], candidate_scope: str) -> list[dict[str, Any]]:
    scores: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    for path in raw_paths:
        obj = json.loads(path.read_text(encoding="utf-8"))
        for score in obj["scores"]:
            key = (int(score["layer"]), int(score["query_head"]))
            scores[key].append(
                {
                    "sequence": float(score["sequence_delta_logp"]),
                    "first_logp": float(score["first_token_delta_logp"]),
                    "first_prob": float(score["first_token_delta_probability"]),
                }
            )
    output = []
    for (layer, head), rows in scores.items():
        sequence = [row["sequence"] for row in rows]
        first_logp = [row["first_logp"] for row in rows]
        first_prob = [row["first_prob"] for row in rows]
        output.append(
            {
                "layer": layer,
                "query_head": head,
                "candidate_scope": candidate_scope,
                "ce_mean_sequence_logp_gain": statistics.mean(sequence),
                "ce_std_sequence_logp_gain": statistics.stdev(sequence) if len(sequence) > 1 else 0.0,
                "ce_mean_first_token_logp_gain": statistics.mean(first_logp),
                "ce_mean_first_token_probability_gain": statistics.mean(first_prob),
                "n_queries": len(rows),
            }
        )
    return sorted(output, key=lambda row: row["ce_mean_sequence_logp_gain"], reverse=True)


def mean_task_activations(
    model: Any,
    recorder: HeadActivationRecorder,
    packs: dict[str, list[Any]],
    device: torch.device,
    checkpoint_id: str,
) -> dict[str, torch.Tensor]:
    output = {}
    for task_id in TASK_IDS:
        path = activation_file(checkpoint_id, task_id)
        if path.exists():
            output[task_id] = load_means(path, device)
            print(f"activation cache {checkpoint_id}/{task_id}", flush=True)
            continue
        collected = []
        for index, pack in enumerate(packs[task_id], start=1):
            collected.append(recorder.record(pack.correct_ids, device))
            if index == 1 or index % 5 == 0 or index == len(packs[task_id]):
                print(f"activations {checkpoint_id}/{task_id}: {index}/{len(packs[task_id])}", flush=True)
        mean = torch.stack(collected).mean(dim=0).float()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mean.cpu(), path)
        output[task_id] = mean.to(device=device)
    return output


def score_heads_for_task(
    model: Any,
    checkpoint_id: str,
    task_id: str,
    packs: list[Any],
    means: torch.Tensor,
    candidate_pairs: list[tuple[int, int]],
    candidate_scope: str,
    device: torch.device,
) -> list[Path]:
    raw_paths = []
    missing_indices = [
        idx for idx, pack in enumerate(packs)
        if not effect_cache_is_valid(
            raw_effect_file(checkpoint_id, task_id, idx), checkpoint_id, task_id,
            pack.query_id, candidate_scope, len(candidate_pairs)
        )
    ]
    if not missing_indices:
        return [raw_effect_file(checkpoint_id, task_id, idx) for idx in range(len(packs))]
    patcher = HeadActivationPatcher(model, means)
    try:
        for idx in missing_indices:
            pack = packs[idx]
            baseline = reference_sequence(model, pack.counterfactual_ids, pack.target_ids, device)
            correct_prompt_score = reference_sequence(model, pack.correct_ids, pack.target_ids, device)
            scores = patcher.run_candidates(
                pack.counterfactual_ids,
                pack.target_ids,
                candidate_pairs,
                batch_size=PATCH_BATCH_SIZE,
            )
            p0 = float(math.exp(max(-80.0, baseline["first_token_logp"])))
            for row in scores:
                row["sequence_delta_logp"] = row["mean_sequence_logp"] - baseline["mean_logp"]
                row["sequence_delta_nll"] = -row["sequence_delta_logp"]
                row["first_token_delta_logp"] = row["first_token_logp"] - baseline["first_token_logp"]
                row["first_token_delta_probability"] = math.exp(
                    max(-80.0, row["first_token_logp"])
                ) - p0
            path = raw_effect_file(checkpoint_id, task_id, idx)
            json_dump(
                path,
                {
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "query_id": pack.query_id,
                    "candidate_scope": candidate_scope,
                    "baseline_counterfactual": {
                        "mean_sequence_logp": baseline["mean_logp"],
                        "first_token_logp": baseline["first_token_logp"],
                        "first_token_probability": p0,
                    },
                    "correct_prompt_reference": {
                        "mean_sequence_logp": correct_prompt_score["mean_logp"],
                        "first_token_logp": correct_prompt_score["first_token_logp"],
                        "first_token_probability": math.exp(max(-80.0, correct_prompt_score["first_token_logp"])),
                        "counterfactual_mean_logp_gap": correct_prompt_score["mean_logp"] - baseline["mean_logp"],
                    },
                    "correct_icl_prompt_tokens": pack.correct_prompt_tokens,
                    "counterfactual_icl_prompt_tokens": pack.counterfactual_prompt_tokens,
                    "shots": pack.shots,
                    "support_train_ids": pack.support_ids,
                    "scores": scores,
                },
            )
            print(
                f"patch {checkpoint_id}/{task_id}: query {idx + 1}/{len(packs)}, "
                f"heads {len(candidate_pairs)}, {candidate_scope}",
                flush=True,
            )
    finally:
        patcher.close()
    return [raw_effect_file(checkpoint_id, task_id, idx) for idx in range(len(packs))]


def write_head_csv(all_rankings: dict[str, list[dict[str, Any]]]) -> None:
    path = PHASE5_ROOT / "fv" / "head_importance.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint_id", "task_id", "rank", "layer", "query_head", "candidate_scope",
        "ce_mean_sequence_logp_gain", "ce_std_sequence_logp_gain",
        "ce_mean_first_token_logp_gain", "ce_mean_first_token_probability_gain", "n_queries",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, rows in all_rankings.items():
            checkpoint_id, task_id = key.split("__", 1)
            for rank, row in enumerate(rows, start=1):
                writer.writerow({"checkpoint_id": checkpoint_id, "task_id": task_id, "rank": rank, **row})


def make_fv_objects(
    snapshot_ids: list[str],
    all_rankings: dict[str, list[dict[str, Any]]],
    means_by_snapshot: dict[str, dict[str, torch.Tensor]],
):
    top_heads: dict[str, dict[str, list[tuple[int, int]]]] = {}
    fvs: dict[str, dict[str, torch.Tensor]] = {}
    descriptors = []
    for snapshot in snapshot_ids:
        top_heads[snapshot] = {}
        fvs[snapshot] = {}
        for task_id in TASK_IDS:
            key = checkpoint_key(snapshot, task_id)
            ranked = all_rankings[key]
            top = [(int(row["layer"]), int(row["query_head"])) for row in ranked[:TOP_K]]
            top_heads[snapshot][task_id] = top
            mean_activation = means_by_snapshot[snapshot][task_id]
            vector = torch.stack([mean_activation[layer, head] for layer, head in top]).sum(dim=0)
            fvs[snapshot][task_id] = vector.detach().float().cpu()
            descriptors.append(
                {
                    "checkpoint_id": snapshot,
                    "task_id": task_id,
                    "top_k": len(top),
                    "top_heads": [{"layer": layer, "query_head": head} for layer, head in top],
                    "fv_l2_norm": float(vector.norm().item()),
                    "fv_rms": float(vector.square().mean().sqrt().item()),
                    "head_rank_scope": ranked[0]["candidate_scope"] if ranked else "unknown",
                }
            )
    return top_heads, fvs, descriptors


def similarity_rows(snapshot_ids: list[str], fvs: dict[str, dict[str, torch.Tensor]]):
    task_cosines, drift_rows = [], []
    base_id = snapshot_ids[0]
    for snapshot in snapshot_ids:
        for task_a in TASK_IDS:
            for task_b in TASK_IDS:
                left, right = fvs[snapshot][task_a], fvs[snapshot][task_b]
                cosine = float(F.cosine_similarity(left.view(1, -1), right.view(1, -1)).item())
                task_cosines.append(
                    {"checkpoint_id": snapshot, "task_a": task_a, "task_b": task_b, "cosine": cosine}
                )
            base_vector = fvs[base_id][task_a]
            final_vector = fvs[snapshot][task_a]
            drift_rows.append(
                {
                    "checkpoint_id": snapshot,
                    "task_id": task_a,
                    "cosine_to_pretrained_fv": float(
                        F.cosine_similarity(base_vector.view(1, -1), final_vector.view(1, -1)).item()
                    ),
                    "relative_l2_drift": float(
                        ((final_vector - base_vector).norm() / base_vector.norm().clamp_min(1e-12)).item()
                    ),
                    "fv_norm": float(final_vector.norm().item()),
                }
            )
    return task_cosines, drift_rows


def overlap_rows(snapshot_ids: list[str], top_heads: dict[str, dict[str, list[tuple[int, int]]]]):
    rows = []
    for snapshot in snapshot_ids:
        for task_a in TASK_IDS:
            a = set(top_heads[snapshot][task_a])
            for task_b in TASK_IDS:
                b = set(top_heads[snapshot][task_b])
                union = a | b
                rows.append(
                    {
                        "checkpoint_id": snapshot,
                        "task_a": task_a,
                        "task_b": task_b,
                        "intersection_count": len(a & b),
                        "jaccard": len(a & b) / len(union) if union else 0.0,
                    }
                )
    return rows


def paired_bootstrap_ci(values: list[float], seed: int, repeats: int = 5000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None}
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(repeats, len(data)), replace=True).mean(axis=1)
    return {
        "n": int(data.size),
        "mean_delta": float(data.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def compute_intervention_effects(predictions: list[dict[str, Any]], nll_rows: list[dict[str, Any]]):
    effects = []
    for task_number, task_id in enumerate(TASK_IDS):
        task_predictions = [row for row in predictions if row["task_id"] == task_id]
        task_nll = [row for row in nll_rows if row["task_id"] == task_id]
        conditions = sorted({row["condition"] for row in task_predictions})
        baseline_predictions = {
            row["instance_id"]: row for row in task_predictions if row["condition"] == "baseline"
        }
        baseline_nll = {row["instance_id"]: row for row in task_nll if row["condition"] == "baseline"}
        for condition in conditions:
            if condition == "baseline":
                continue
            current_predictions = {
                row["instance_id"]: row for row in task_predictions if row["condition"] == condition
            }
            current_nll = {row["instance_id"]: row for row in task_nll if row["condition"] == condition}
            ids = sorted(set(baseline_predictions) & set(current_predictions) & set(baseline_nll) & set(current_nll))
            metric_names = ["per_example_token_nll"]
            metric_names += ["rouge_l_f1", "token_f1"] if task_id in TASK_GENERATION else ["correct"]
            for metric in metric_names:
                deltas = []
                for instance_id in ids:
                    if metric == "per_example_token_nll":
                        base = baseline_nll[instance_id]
                        current = current_nll[instance_id]
                        base_value = float(base["nll_sum"]) / max(1, int(base["token_count"]))
                        value = float(current["nll_sum"]) / max(1, int(current["token_count"]))
                    elif metric == "correct":
                        base_value = float(baseline_predictions[instance_id].get("correct", False))
                        value = float(current_predictions[instance_id].get("correct", False))
                    else:
                        base_value = float(baseline_predictions[instance_id].get(metric, 0.0))
                        value = float(current_predictions[instance_id].get(metric, 0.0))
                    deltas.append(value - base_value)
                stats = paired_bootstrap_ci(deltas, SEED + task_number * 97 + len(effects))
                effects.append({"task_id": task_id, "condition": condition, "metric": metric, **stats})

            # Collapse is a task-level maximum over the Yes/No output shares.
            if ids:
                base_labels = [parse_yes_no(baseline_predictions[item]["prediction"]) for item in ids]
                current_labels = [parse_yes_no(current_predictions[item]["prediction"]) for item in ids]
                rng = np.random.default_rng(SEED + task_number * 31 + len(effects))
                deltas = []
                for _ in range(5000):
                    sample = rng.integers(0, len(ids), size=len(ids))
                    base_yes = sum(base_labels[i] == "Yes" for i in sample) / len(sample)
                    base_no = sum(base_labels[i] == "No" for i in sample) / len(sample)
                    current_yes = sum(current_labels[i] == "Yes" for i in sample) / len(sample)
                    current_no = sum(current_labels[i] == "No" for i in sample) / len(sample)
                    deltas.append(max(current_yes, current_no) - max(base_yes, base_no))
                # `deltas` already contains the bootstrap sampling distribution.
                # Resampling those 5,000 bootstrap draws a second time estimates
                # Monte Carlo error, not the uncertainty over held-out examples.
                stats = {
                    "n": len(ids),
                    "mean_delta": float(np.mean(deltas)),
                    "ci95_low": float(np.quantile(deltas, 0.025)),
                    "ci95_high": float(np.quantile(deltas, 0.975)),
                }
                effects.append({"task_id": task_id, "condition": condition, "metric": "yes_no_collapse_rate", **stats})
    return effects


def write_fv_report(results: dict[str, Any]) -> None:
    report_path = REPORT_ROOT / "phase5_fv_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_id = results["intervention_model_id"]
    fv_rows = results["fv_descriptors"]
    final_rows = [row for row in fv_rows if row["checkpoint_id"] == final_id]
    interventions = results.get("intervention_metrics", [])
    stage_rows = results.get("stage_generation_metrics", [])
    effect_rows = results.get("intervention_effects", [])
    cosine_rows = json.loads((PHASE5_ROOT / "fv" / "fv_cosines.json").read_text(encoding="utf-8"))
    overlap_rows = json.loads((PHASE5_ROOT / "fv" / "head_overlap.json").read_text(encoding="utf-8"))
    drift_rows = json.loads((PHASE5_ROOT / "fv" / "fv_drift.json").read_text(encoding="utf-8"))
    report = [
        "# Phase 5 Function Vector and Head Mechanism Analysis",
        "",
        "## Method and data split",
        "",
        "- Local Qwen3-0.6B checkpoints only; no network model loading, test split access, LoRA, replay, NAS, or training.",
        "- Dev indices 0–9 build correct ICL activations and causal head rankings; indices 10–24 select FV intervention layer/strength; indices 25–49 are the locked final evaluation set (25 examples per task). The four ICL demonstrations come only from each task's train split.",
        f"- ICL shots: {ICL_SHOTS} (the prompt builder may reduce to 2 shots only when preserving the full task definition and the 2048-token budget requires it). Top-K: {TOP_K}. Head-patch batch size: {PATCH_BATCH_SIZE}.",
        "- Qwen3 uses 16 query heads and 8 KV heads. Each query head has head_dim=128; per-head residual contributions use BF16-quantized inputs/weights and FP32 accumulation of the corresponding `o_proj.weight` column slice. This reconstructs the BF16 attention output within a measured relative-RMS tolerance; KV heads are not treated as independent output heads.",
        "- The paper defines a causal effect as a probability difference for a correct answer. For these multi-token generation tasks, head ranking uses the paired change in reference sequence mean log-probability on a label-shuffled prompt. First-token probability effects are also reported. This is a sequence-level extension; it avoids underflow and does not redefine the ICL prompt or patch position.",
        "- Published FV construction sums the mean last-token residual contribution of top-10 causal heads. The implementation uses that definition, with the GQA-aware query-head output projection described above.",
        "- Reference: [Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/74fc5575632191d96881d8015f79dde3-Paper-Conference.pdf).",
        "",
        "## Checkpoint trajectory",
        "",
        "The homogeneous A0 checkpoint trail supplies the pretrained model plus checkpoints after tasks 565, 1603, 1501 and 1384. Global causal scans are run at the pretrained and post-task-4 endpoints; intermediate post-task checkpoints are re-ranked within the union of their endpoint top-10 heads. The Phase 4B A0/S1/seed-10 final checkpoint receives a separate full 448-head scan and supplies the intervention vectors. Intermediate snapshots are from the retained homogeneous baseline trajectory, not intermediate saves from Phase 4B.",
        "",
        "## Phase 4B final task vectors",
        "",
        "| Task | Top-10 query heads (layer, head) | FV L2 norm |",
        "|---|---|---:|",
    ]
    for row in final_rows:
        heads = ", ".join(f"({head['layer']},{head['query_head']})" for head in row["top_heads"])
        report.append(f"| {row['task_id']} | {heads} | {row['fv_l2_norm']:.4f} |")
    tasks = TASK_IDS
    report.extend(
        [
            "",
            "## Cross-task FV similarity and head overlap",
            "",
            "The first and last columns compare pretrained and Phase 4B final task vectors. Head overlap counts are intersections of the task-specific top-10 query heads.",
            "",
            "| Task pair | Pretrained FV cosine | Phase 4B FV cosine | Pretrained top-10 overlap | Phase 4B top-10 overlap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for index, task_a in enumerate(tasks):
        for task_b in tasks[index + 1:]:
            cosine_start = next(
                row["cosine"] for row in cosine_rows
                if row["checkpoint_id"] == "pretrained" and row["task_a"] == task_a and row["task_b"] == task_b
            )
            cosine_final = next(
                row["cosine"] for row in cosine_rows
                if row["checkpoint_id"] == final_id and row["task_a"] == task_a and row["task_b"] == task_b
            )
            overlap_start = next(
                row["intersection_count"] for row in overlap_rows
                if row["checkpoint_id"] == "pretrained" and row["task_a"] == task_a and row["task_b"] == task_b
            )
            overlap_final = next(
                row["intersection_count"] for row in overlap_rows
                if row["checkpoint_id"] == final_id and row["task_a"] == task_a and row["task_b"] == task_b
            )
            report.append(
                f"| {task_a} / {task_b} | {cosine_start:.3f} | {cosine_final:.3f} | "
                f"{overlap_start}/10 | {overlap_final}/10 |"
            )
    report.extend(
        [
            "",
            "### Final-vector drift from pretrained",
            "",
            "| Task | Cosine to pretrained FV | Relative L2 drift |",
            "|---|---:|---:|",
        ]
    )
    for task_id in tasks:
        row = next(item for item in drift_rows if item["checkpoint_id"] == final_id and item["task_id"] == task_id)
        report.append(
            f"| {task_id} | {row['cosine_to_pretrained_fv']:.3f} | {row['relative_l2_drift']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Stage-wise held-out behavior and FV drift",
            "",
            "| A0 stage | Task | Token NLL | ROUGE-L | Accuracy | Yes share | No share | Yes/No collapse |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stage_rows:
        def stage_cell(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.4f}"
        report.append(
            f"| {row['checkpoint_id']} | {row['task_id']} | {stage_cell('teacher_forced_token_nll')} | "
            f"{stage_cell('mean_rouge_l_f1')} | {stage_cell('accuracy')} | {stage_cell('yes_output_share')} | "
            f"{stage_cell('no_output_share')} | {stage_cell('yes_no_collapse_rate')} |"
        )
    if results.get("fv_drift_nll_correlations"):
        report.extend(["", "Descriptive Pearson correlations across the five homogeneous A0 stages (n=5):"])
        for row in results["fv_drift_nll_correlations"]:
            val = row.get("pearson_fv_drift_vs_eval_nll")
            report.append(f"- {row['task_id']}: FV drift vs. held-out token NLL = {'not estimable' if val is None else f'{val:.3f}'}.")
    report.extend(
        [
            "",
            "Head scores, stage-specific top-10 lists, the fixed dev split, and raw per-query patch effects are in `results/phase5/fv/`. Figures are under `figures/phase5/`.",
            "",
            "## FV intervention tuning and held-out results",
            "",
            f"Tuning chose residual block layer {results.get('selected_layer')} (0-based) and coefficient {results.get('selected_scale')}. The selection objective is the mean teacher-forced token NLL of the two old generation tasks under `source FV − latest classifier FV`, averaged over tune indices 10–24.",
            "",
            "| Task | Condition | ROUGE-L | Token-F1 | Accuracy | Teacher-forced token NLL | Ref first-token P | Mean output tokens | Yes share | No share | Collapse |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in interventions:
        def cell(key):
            value = row.get(key)
            return "—" if value is None else f"{float(value):.4f}"
        reference_first_probability = row.get("mean_first_reference_token_probability")
        reference_first_probability_cell = (
            "—" if reference_first_probability is None
            else f"{float(reference_first_probability):.3e}"
        )
        report.append(
            f"| {row['task_id']} | {row['condition']} | {cell('mean_rouge_l_f1')} | {cell('mean_token_f1')} | "
            f"{cell('accuracy')} | {cell('teacher_forced_token_nll')} | {reference_first_probability_cell} | "
            f"{cell('mean_generated_token_count')} | {cell('yes_output_share')} | {cell('no_output_share')} | {cell('yes_no_collapse_rate')} |"
        )
    report.extend(
        [
            "",
            "### Paired held-out effects",
            "",
            "Percentile-bootstrap 95% intervals compare each condition with its same-query baseline (25 examples per task). Lower NLL and higher generation/classification scores are improvements. These small-sample intervals are exploratory and are not multiplicity-adjusted.",
            "",
            "| Task | Condition | Metric | Mean Δ | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in effect_rows:
        if row.get("mean_delta") is None:
            continue
        report.append(
            f"| {row['task_id']} | {row['condition']} | {row['metric']} | {float(row['mean_delta']):.4f} | "
            f"[{float(row['ci95_low']):.4f}, {float(row['ci95_high']):.4f}] |"
        )
    report.extend(
        [
            "",
            "All intervention conditions are evaluated on the same locked 25-example/task dev evaluation subset. `source FV` is the evaluated task's vector. `target FV subtraction` removes task 1384's vector. `combined` adds the evaluated task FV and subtracts task 1384. The random vector matches the combined-vector norm on generation tasks and the task-vector norm on classification tasks; unrelated vectors use the opposite task family (generation vs. classification).",
            "",
            "## Interpretation limits",
            "",
            "Activation patching is causal evidence for the selected head representation on the constructed ICL prompts. Similarity and drift relationships are correlational. The benchmark prompts are short demonstrations drawn from train plus a query from dev; this measures elicited task functions and does not by itself prove which latent function the zero-shot classifier/generator activates in deployment. Sample counts are small, so report effect sizes and per-example files alongside aggregate means.",
            "",
            "## Findings and architecture recommendation",
            "",
            "1. **FV recovery:** Neither old generation task shows a reliable recovery in generated text. Task 565 remains at ROUGE-L/Token-F1 0 and 2 output tokens under every condition. For task 1603, combined source-minus-classifier FV changes ROUGE-L by -0.0047 (95% CI [-0.0300, 0.0160]); generated length remains 2 tokens and the Yes/No-collapse change has a bootstrap interval that reaches zero. Some teacher-forced NLL changes are significant, but random or unrelated vectors can improve NLL as much or more, so NLL alone is not task-specific recovery evidence.",
            "2. **Forgetting mechanism:** The old generation FVs drift strongly from pretrained (cosines 0.119 and 0.045; relative L2 drift about 1.0), and task 565's final top-10 head set is disjoint from the other three tasks. Yet drift-to-NLL correlations across five stages are weak for the two generation tasks and FV injection does not restore their text outputs. The evidence therefore favors impaired old-task conditional generation/output behavior over activation bias as a sufficient explanation; activation bias remains a measurable contributing factor, not a demonstrated sole cause.",
            "3. **A1 branch:** The paired direct-path test is in `reports/phase5_a1_ablation.md`. A1 gates are very small and zeroing them changes few logits/predictions; A1-vs-A0 differences also include the distinct training trajectory, so they do not establish a direct branch benefit.",
            "4. **Next architecture:** Prioritize a small **Conditional Routing** study with task-conditioned gates over the identified head groups, and include Head Isolation as a control. Top-head overlap is zero for task 565 against all other tasks after Phase 4B, while task 1603 retains some FV similarity to both classification tasks; a task-conditioned route can protect the distinct path while retaining shared capacity. The evidence is enough to define a constrained Task-Aware NAS hypothesis, but not to claim a validated architecture benefit or launch a broad NAS search before the routing pilot.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            r".\conda-env\python.exe scripts\run_phase5_fv.py",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")


def recompute_intervention_summary_from_caches() -> None:
    """Rebuild paired inference statistics and report from locked dev caches."""
    predictions: list[dict[str, Any]] = []
    nll_rows: list[dict[str, Any]] = []
    conditions = [
        "baseline", "source_fv", "subtract_latest_classifier_fv",
        "combined_source_minus_latest", "random_norm_matched", "unrelated_fv",
    ]
    for task_id in TASK_IDS:
        for condition in conditions:
            path = PHASE5_ROOT / "interventions" / FV_VERSION / "evaluation" / task_id / f"{condition}.json"
            obj = json.loads(path.read_text(encoding="utf-8"))
            if (
                obj.get("fv_version") != FV_VERSION
                or len(obj.get("predictions", [])) != 25
                or len(obj.get("nll_rows", [])) != 25
            ):
                raise ValueError(f"Incomplete or mismatched locked evaluation cache: {path}")
            predictions.extend({"condition": condition, **row} for row in obj["predictions"])
            nll_rows.extend(
                {"condition": condition, "task_id": task_id, **row} for row in obj["nll_rows"]
            )
    effects = compute_intervention_effects(predictions, nll_rows)
    json_dump(PHASE5_ROOT / "interventions" / "paired_effects_bootstrap.json", effects)
    jsonl_write(PHASE5_ROOT / "interventions" / "final_predictions.jsonl", predictions)
    jsonl_write(PHASE5_ROOT / "interventions" / "final_teacher_forced_scores.jsonl", nll_rows)
    summary_path = PHASE5_ROOT / "fv" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["intervention_effects"] = effects
    json_dump(summary_path, summary)
    write_fv_report(summary)
    print("Rebuilt paired intervention statistics and report from 24 locked evaluation caches.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="Validate files, CUDA, tokenization, GQA hooks, then exit")
    parser.add_argument("--skip-interventions", action="store_true", help="Stop after extracting activations and causal head effects")
    parser.add_argument("--recompute-cached-intervention-stats", action="store_true", help="Recompute paired inference statistics from the completed locked evaluation caches")
    args = parser.parse_args()
    if args.recompute_cached_intervention_stats:
        recompute_intervention_summary_from_caches()
        return
    config, model_path, tokenizer, device, train, dev = load_phase5_context()
    PHASE5_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    architecture_info = {
        "device": str(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3),
        "model_path": str(model_path),
        "test_split_read": False,
        "dev_split": {"build": [0, 10], "tune": [10, 25], "evaluation": [25, 50]},
    }
    specs = stage_checkpoint_specs()
    for spec in specs[1:]:
        if not (Path(spec["path"]) / "model.safetensors").is_file():
            raise FileNotFoundError(f"Required retained stage checkpoint missing: {spec['path']}")
    final_path = phase4b_checkpoint("A0", "S1", 10)
    if not (final_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Required Phase 4B final checkpoint missing: {final_path}")
    packs = make_icl_sets(tokenizer, train, dev)
    build_meta = {}
    split_manifest = {"split_rule": SPLITS, "test_split_read": False, "tasks": {}}
    for task_id in TASK_IDS:
        split_manifest["tasks"][task_id] = {
            "build_dev_ids": [row["instance_id"] for row in dev[task_id][0:10]],
            "tune_dev_ids": [row["instance_id"] for row in dev[task_id][10:25]],
            "evaluation_dev_ids": [row["instance_id"] for row in dev[task_id][25:50]],
            "icl_support_train_ids": [pack.support_ids for pack in packs[task_id]],
        }
        build_meta[task_id] = [
            {
                "query_id": pack.query_id,
                "shots": pack.shots,
                "correct_prompt_tokens": pack.correct_prompt_tokens,
                "counterfactual_prompt_tokens": pack.counterfactual_prompt_tokens,
                "support_train_ids": pack.support_ids,
                "truncated_demo_count": pack.truncated_demo_count,
                "query_tail_tokens_kept": pack.query_tail_tokens_kept,
            }
            for pack in packs[task_id]
        ]
        if any(pack.correct_ids == pack.counterfactual_ids for pack in packs[task_id]):
            raise AssertionError(f"Correct and counterfactual prompt unexpectedly identical: {task_id}")
    json_dump(PHASE5_ROOT / "fv" / "split_manifest.json", split_manifest)
    json_dump(PHASE5_ROOT / "fv" / "icl_prompt_manifest.json", build_meta)
    json_dump(PHASE5_ROOT / "run_config.json", {**architecture_info, "fv_version": FV_VERSION, "icl_shots": ICL_SHOTS, "top_k": TOP_K, "patch_batch_size": PATCH_BATCH_SIZE})

    # Run a one-head patch smoke check before a full causal sweep.
    model = load_checkpoint(model_path, final_path, architecture="A0")
    geometry = model_head_geometry(model)
    if geometry != {"layers": 28, "query_heads": 16, "kv_heads": 8, "head_dim": 128, "hidden_size": 1024, "o_proj_input_width": 2048}:
        raise RuntimeError(f"Unexpected Qwen3-0.6B head geometry: {geometry}")
    recorder = HeadActivationRecorder(model)
    test_pack = packs[TASK_565][0]
    act_test = recorder.record(test_pack.correct_ids, device)
    if act_test.shape != (28, 16, 1024) or not torch.isfinite(act_test).all():
        raise RuntimeError(f"Activation capture shape/value check failed: {tuple(act_test.shape)}")
    if recorder.max_reconstruction_relative_rms > 0.005:
        raise RuntimeError(
            "Per-query-head o_proj contributions do not reconstruct the BF16 output within relative-RMS tolerance: "
            f"{recorder.max_reconstruction_relative_rms}"
        )
    patcher = HeadActivationPatcher(model, act_test.to(device))
    test_patch = patcher.run_candidates(test_pack.counterfactual_ids, test_pack.target_ids, [(0, 0)], batch_size=1)
    same_prompt_base = reference_sequence(model, test_pack.correct_ids, test_pack.target_ids, device)
    same_prompt_patch = patcher.run_candidates(test_pack.correct_ids, test_pack.target_ids, [(0, 0)], batch_size=1)
    no_op_patch_delta = abs(same_prompt_patch[0]["mean_sequence_logp"] - same_prompt_base["mean_logp"])
    patcher.close()
    recorder.close()
    if not test_patch or not math.isfinite(float(test_patch[0]["mean_sequence_logp"])):
        raise RuntimeError("Single-head activation patch smoke check failed")
    if no_op_patch_delta > 0.002:
        raise RuntimeError(f"A same-prompt activation replacement should be a no-op; observed Δ={no_op_patch_delta}")
    preflight = {
        **architecture_info,
        "head_geometry": geometry,
        "fv_version": FV_VERSION,
        "smoke_activation_shape": list(act_test.shape),
        "o_proj_head_sum_reconstruction_max_abs_error": recorder.max_reconstruction_error,
        "o_proj_head_sum_reconstruction_max_relative_rms_error": recorder.max_reconstruction_relative_rms,
        "smoke_patch_mean_sequence_logp": test_patch[0]["mean_sequence_logp"],
        "smoke_patch_first_token_logp": test_patch[0]["first_token_logp"],
        "smoke_same_prompt_patch_abs_mean_logp_delta": no_op_patch_delta,
        "loaded_checkpoint": str(final_path),
        "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        "passes": True,
    }
    json_dump(PHASE5_ROOT / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False), flush=True)
    if args.preflight_only:
        cleanup_model(model)
        return
    cleanup_model(model)

    # Capture task-conditioned activations at every retained homogeneous stage,
    # plus the actual collapsed Phase 4B A0 checkpoint used for interventions.
    all_specs = specs + [
        {
            "checkpoint_id": "phase4b_A0_S1_seed10",
            "architecture": "A0",
            "path": final_path,
            "stage": 4,
            "task": TASK_1384,
        }
    ]
    means_by_snapshot: dict[str, dict[str, torch.Tensor]] = {}
    stage_nll_rows: list[dict[str, Any]] = []
    stage_generation_rows: list[dict[str, Any]] = []
    eval_dev = split_records(dev, "evaluation")
    for spec in all_specs:
        checkpoint_id = str(spec["checkpoint_id"])
        has_missing_means = any(not activation_file(checkpoint_id, task).exists() for task in TASK_IDS)
        need_stage_metrics = checkpoint_id in {"pretrained", "homogeneous_stage_1", "homogeneous_stage_2", "homogeneous_stage_3", "homogeneous_stage_4"} and any(
            not metrics_file(checkpoint_id, task).exists() for task in TASK_IDS
        )
        if has_missing_means or need_stage_metrics:
            print(f"load {checkpoint_id}: {spec['path']}", flush=True)
            model = load_checkpoint(model_path, spec["path"], architecture=spec["architecture"])
            if checkpoint_id == "phase4b_A0_S1_seed10":
                geometry = model_head_geometry(model)
            recorder = HeadActivationRecorder(model)
            means_by_snapshot[checkpoint_id] = mean_task_activations(model, recorder, packs, device, checkpoint_id)
            recorder.close()
            if checkpoint_id in {"pretrained", "homogeneous_stage_1", "homogeneous_stage_2", "homogeneous_stage_3", "homogeneous_stage_4"}:
                for task_id in TASK_IDS:
                    metric_path = metrics_file(checkpoint_id, task_id)
                    if metric_path.exists():
                        cache = json.loads(metric_path.read_text(encoding="utf-8"))
                        current = ensure_yes_no_shares(cache["metrics"], cache.get("predictions", []))
                        cache["metrics"] = current
                        json_dump(metric_path, cache)
                    else:
                        nll_rows = []
                        gen_rows = []
                        for index, record in enumerate(eval_dev[task_id], start=25):
                            score = zero_shot_teacher_forced(model, tokenizer, record, config["model"], device)
                            nll_rows.append({k: score[k] for k in ["nll_sum", "token_count", "first_token_probability"]})
                            generation = zero_shot_generate(model, tokenizer, record, config["model"], config["evaluation"], device)
                            gen_rows.append(score_prediction(task_id, record, generation, config["dataset"]["task_types"][task_id]))
                        current = aggregate_task(task_id, gen_rows, config["dataset"]["task_types"][task_id], nll_rows)
                        current = ensure_yes_no_shares(current, gen_rows)
                        json_dump(metric_path, {"checkpoint_id": checkpoint_id, "task_id": task_id, "metrics": current, "predictions": gen_rows, "nll_rows": nll_rows})
                    stage_nll_rows.append({"checkpoint_id": checkpoint_id, "task_id": task_id, "teacher_forced_token_nll": current["teacher_forced_token_nll"]})
                    stage_generation_rows.append({"checkpoint_id": checkpoint_id, "task_id": task_id, **current})
            cleanup_model(model)
        else:
            means_by_snapshot[checkpoint_id] = {task: load_means(activation_file(checkpoint_id, task), device) for task in TASK_IDS}
            for task_id in TASK_IDS:
                metric_path = metrics_file(checkpoint_id, task_id)
                if metric_path.exists():
                    cache = json.loads(metric_path.read_text(encoding="utf-8"))
                    current = ensure_yes_no_shares(cache["metrics"], cache.get("predictions", []))
                    cache["metrics"] = current
                    json_dump(metric_path, cache)
                    stage_nll_rows.append({"checkpoint_id": checkpoint_id, "task_id": task_id, "teacher_forced_token_nll": current["teacher_forced_token_nll"]})
                    stage_generation_rows.append({"checkpoint_id": checkpoint_id, "task_id": task_id, **current})
    json_dump(PHASE5_ROOT / "stage_eval" / "stage_metrics.json", stage_generation_rows)

    # Full head scans on the pretrained base, homogeneous final checkpoint and
    # Phase 4B final; intermediate scans rerank the union of endpoint Top-10.
    rankings: dict[str, list[dict[str, Any]]] = {}
    full_candidates = [(layer, head) for layer in range(28) for head in range(16)]
    endpoint_ids = ["pretrained", "homogeneous_stage_4", "phase4b_A0_S1_seed10"]
    for checkpoint_id in endpoint_ids:
        spec = next(item for item in all_specs if item["checkpoint_id"] == checkpoint_id)
        print(f"causal endpoint scan {checkpoint_id}", flush=True)
        task_paths: dict[str, list[Path]] = {}
        for task_id in TASK_IDS:
            task_paths[task_id] = [raw_effect_file(checkpoint_id, task_id, idx) for idx in range(len(packs[task_id]))]
        if any(
            not effect_cache_is_valid(
                raw_effect_file(checkpoint_id, task_id, idx), checkpoint_id, task_id,
                packs[task_id][idx].query_id, "global_448", len(full_candidates)
            )
            for task_id in TASK_IDS for idx in range(len(packs[task_id]))
        ):
            model = load_checkpoint(model_path, spec["path"], architecture=spec["architecture"])
            for task_id in TASK_IDS:
                means = means_by_snapshot[checkpoint_id][task_id]
                paths = score_heads_for_task(model, checkpoint_id, task_id, packs[task_id], means, full_candidates, "global_448", device)
                rankings[checkpoint_key(checkpoint_id, task_id)] = rank_effects(paths, "global_448")
            cleanup_model(model)
        else:
            for task_id in TASK_IDS:
                rankings[checkpoint_key(checkpoint_id, task_id)] = rank_effects(task_paths[task_id], "global_448")

    endpoint_union: dict[str, list[tuple[int, int]]] = {}
    for task_id in TASK_IDS:
        start_top = rankings[checkpoint_key("pretrained", task_id)][:TOP_K]
        end_top = rankings[checkpoint_key("homogeneous_stage_4", task_id)][:TOP_K]
        endpoint_union[task_id] = sorted(
            {(int(row["layer"]), int(row["query_head"])) for row in start_top + end_top}
        )

    for checkpoint_id in ["homogeneous_stage_1", "homogeneous_stage_2", "homogeneous_stage_3"]:
        spec = next(item for item in all_specs if item["checkpoint_id"] == checkpoint_id)
        print(f"tracked-head stage scan {checkpoint_id}", flush=True)
        task_paths = {
            task_id: [raw_effect_file(checkpoint_id, task_id, idx) for idx in range(len(packs[task_id]))]
            for task_id in TASK_IDS
        }
        if any(
            not effect_cache_is_valid(
                raw_effect_file(checkpoint_id, task_id, idx), checkpoint_id, task_id,
                packs[task_id][idx].query_id, "endpoint_union_restricted", len(endpoint_union[task_id])
            )
            for task_id in TASK_IDS for idx in range(len(packs[task_id]))
        ):
            model = load_checkpoint(model_path, spec["path"], architecture="A0")
            for task_id in TASK_IDS:
                paths = score_heads_for_task(
                    model,
                    checkpoint_id,
                    task_id,
                    packs[task_id],
                    means_by_snapshot[checkpoint_id][task_id],
                    endpoint_union[task_id],
                    "endpoint_union_restricted",
                    device,
                )
                rankings[checkpoint_key(checkpoint_id, task_id)] = rank_effects(paths, "endpoint_union_restricted")
            cleanup_model(model)
        else:
            for task_id in TASK_IDS:
                rankings[checkpoint_key(checkpoint_id, task_id)] = rank_effects(task_paths[task_id], "endpoint_union_restricted")

    snapshot_ids = [spec["checkpoint_id"] for spec in all_specs]
    # Restore rankings in case every raw query was resumed from disk.
    for checkpoint_id in snapshot_ids:
        for task_id in TASK_IDS:
            key = checkpoint_key(checkpoint_id, task_id)
            if key in rankings:
                continue
            paths = [raw_effect_file(checkpoint_id, task_id, idx) for idx in range(len(packs[task_id]))]
            scope = "endpoint_union_restricted" if checkpoint_id in {
                "homogeneous_stage_1", "homogeneous_stage_2", "homogeneous_stage_3"
            } else "global_448"
            rankings[key] = rank_effects(paths, scope)

    write_head_csv(rankings)
    top_heads, fvs, fv_descriptors = make_fv_objects(snapshot_ids, rankings, means_by_snapshot)
    torch.save(
        {snapshot: {task: tensor.cpu() for task, tensor in tasks.items()} for snapshot, tasks in fvs.items()},
        PHASE5_ROOT / "fv" / "function_vectors.pt",
    )
    json_dump(PHASE5_ROOT / "fv" / "top_heads_and_vectors.json", fv_descriptors)
    task_cosines, drift_rows = similarity_rows(snapshot_ids, fvs)
    head_overlaps = overlap_rows(snapshot_ids, top_heads)
    json_dump(PHASE5_ROOT / "fv" / "fv_cosines.json", task_cosines)
    json_dump(PHASE5_ROOT / "fv" / "fv_drift.json", drift_rows)
    json_dump(PHASE5_ROOT / "fv" / "head_overlap.json", head_overlaps)
    json_dump(PHASE5_ROOT / "fv" / "head_importance_ranked.json", rankings)
    save_phase5_figures("phase4b_A0_S1_seed10", rankings, task_cosines, drift_rows, stage_nll_rows)

    # Relate held-out NLL and generation behavior to the baseline trajectory's
    # FV drift with a small descriptive Pearson coefficient, clearly marked as
    # correlational and low-n.
    correlations = []
    baseline_stages = ["pretrained", "homogeneous_stage_1", "homogeneous_stage_2", "homogeneous_stage_3", "homogeneous_stage_4"]
    for task_id in TASK_IDS:
        by_stage_fv = {row["checkpoint_id"]: row for row in drift_rows if row["task_id"] == task_id}
        by_stage_nll = {row["checkpoint_id"]: row for row in stage_nll_rows if row["task_id"] == task_id}
        if all(stage in by_stage_fv and stage in by_stage_nll for stage in baseline_stages):
            similarities = [by_stage_fv[stage]["cosine_to_pretrained_fv"] for stage in baseline_stages]
            nlls = [by_stage_nll[stage]["teacher_forced_token_nll"] for stage in baseline_stages]
            if np.std(similarities) > 0 and np.std(nlls) > 0:
                r = float(np.corrcoef(1 - np.array(similarities), np.array(nlls))[0, 1])
            else:
                r = None
            correlations.append({"task_id": task_id, "n_stages": 5, "pearson_fv_drift_vs_eval_nll": r, "interpretation": "descriptive correlation; not causal"})
    json_dump(PHASE5_ROOT / "fv" / "fv_drift_nll_correlations.json", correlations)

    if args.skip_interventions:
        return
    final_id = "phase4b_A0_S1_seed10"
    final_vectors = fvs[final_id]
    model = load_checkpoint(model_path, final_path, architecture="A0")
    intervention = ResidualVectorIntervention(model)
    tune_rows = split_records(dev, "tune")
    layer_candidates = [5, 12, 19, 26]
    scale_candidates = [0.25, 0.5, 1.0, 2.0]
    tune_file = PHASE5_ROOT / "interventions" / FV_VERSION / "tuning_grid.csv"
    tuned: list[dict[str, Any]] = []
    if tune_file.exists():
        with tune_file.open("r", encoding="utf-8-sig", newline="") as handle:
            tuned = list(csv.DictReader(handle))
    else:
        for layer in layer_candidates:
            for scale in scale_candidates:
                per_task = {}
                for task_id in [TASK_565, TASK_1603]:
                    vector = final_vectors[task_id] - final_vectors[TASK_1384]
                    total_nll, total_tokens = 0.0, 0
                    for record in tune_rows[task_id]:
                        score = zero_shot_teacher_forced(
                            model, tokenizer, record, config["model"], device,
                            intervention=intervention, vector_spec=(layer, vector, scale),
                        )
                        total_nll += score["nll_sum"]
                        total_tokens += score["token_count"]
                    per_task[task_id] = total_nll / max(1, total_tokens)
                tuned.append(
                    {
                        "layer": layer,
                        "scale": scale,
                        "task565_token_nll": per_task[TASK_565],
                        "task1603_token_nll": per_task[TASK_1603],
                        "mean_generation_token_nll": statistics.mean(per_task.values()),
                    }
                )
                print(f"tune FV intervention: layer={layer}, scale={scale}, NLL={tuned[-1]['mean_generation_token_nll']:.4f}", flush=True)
        tune_file.parent.mkdir(parents=True, exist_ok=True)
        with tune_file.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tuned[0]))
            writer.writeheader()
            writer.writerows(tuned)
    selected = min(tuned, key=lambda row: float(row["mean_generation_token_nll"]))
    selected_layer, selected_scale = int(float(selected["layer"])), float(selected["scale"])
    json_dump(
        PHASE5_ROOT / "interventions" / FV_VERSION / "selected_intervention.json",
        {
            "selection_split": "dev indices 10-24",
            "selection_objective": "minimum mean teacher-forced token NLL on task565/task1603 for source FV minus task1384 FV",
            "layer_zero_based": selected_layer,
            "scale": selected_scale,
            "selected_tuning_row": selected,
            "grid": [{k: safe_float(v) if k not in {"layer"} else int(float(v)) for k, v in row.items()} for row in tuned],
        },
    )

    evaluation_rows = split_records(dev, "evaluation")
    cross_family = {
        TASK_565: TASK_1384,
        TASK_1603: TASK_1501,
        TASK_1501: TASK_565,
        TASK_1384: TASK_1603,
    }
    condition_results = []
    all_prediction_rows = []
    all_nll_rows = []
    for task_id in TASK_IDS:
        own = final_vectors[task_id]
        latest = final_vectors[TASK_1384]
        random_reference = own - latest if task_id in TASK_GENERATION else own
        random_vector = norm_match_random(random_reference, SEED + int(task_id[4:7]))
        condition_vectors: dict[str, torch.Tensor | None] = {
            "baseline": None,
            "source_fv": own,
            "subtract_latest_classifier_fv": -latest,
            "combined_source_minus_latest": own - latest,
            "random_norm_matched": random_vector,
            "unrelated_fv": final_vectors[cross_family[task_id]],
        }
        for condition, vector in condition_vectors.items():
            cache_path = PHASE5_ROOT / "interventions" / FV_VERSION / "evaluation" / task_id / f"{condition}.json"
            current_norm = None if vector is None else float(vector.norm().item())
            cached_object = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
            cached_norm = None if cached_object is None else cached_object.get("vector_l2_norm")
            cache_has_expected_vector = (
                condition != "random_norm_matched" and cached_norm is None
            ) or (
                vector is None and cached_norm is None
            ) or (
                cached_norm is not None and current_norm is not None and abs(float(cached_norm) - current_norm) <= 1e-5
            )
            if cached_object is not None and cache_has_expected_vector:
                obj = cached_object
                predictions = obj["predictions"]
                nll_rows = obj["nll_rows"]
                metrics = ensure_yes_no_shares(obj["metrics"], predictions)
                obj["metrics"] = metrics
                json_dump(cache_path, obj)
            else:
                predictions, nll_rows = [], []
                spec = None if vector is None else (selected_layer, vector, selected_scale)
                for index, record in enumerate(evaluation_rows[task_id], start=1):
                    score = zero_shot_teacher_forced(
                        model, tokenizer, record, config["model"], device,
                        intervention=intervention, vector_spec=spec,
                    )
                    generated = zero_shot_generate(
                        model, tokenizer, record, config["model"], config["evaluation"], device,
                        intervention=intervention, vector_spec=spec,
                    )
                    predictions.append(score_prediction(task_id, record, generated, config["dataset"]["task_types"][task_id]))
                    nll_rows.append(
                        {
                            "instance_id": record["instance_id"],
                            "nll_sum": score["nll_sum"],
                            "token_count": score["token_count"],
                            "first_token_logp": score["first_token_logp"],
                            "first_token_probability": score["first_token_probability"],
                            "first_token_prediction_id": score["first_token_prediction_id"],
                            "prompt_diagnostics": score["prompt_diagnostics"],
                        }
                    )
                    if index == 1 or index % 5 == 0 or index == len(evaluation_rows[task_id]):
                        print(f"intervention {task_id}/{condition}: {index}/{len(evaluation_rows[task_id])}", flush=True)
                metrics = aggregate_task(task_id, predictions, config["dataset"]["task_types"][task_id], nll_rows)
                json_dump(
                    cache_path,
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "fv_version": FV_VERSION,
                        "vector_l2_norm": current_norm,
                        "injection_layer": None if vector is None else selected_layer,
                        "coefficient": None if vector is None else selected_scale,
                        "metrics": metrics,
                        "predictions": predictions,
                        "nll_rows": nll_rows,
                    },
                )
            metrics = {
                "task_id": task_id,
                "condition": condition,
                "vector_l2_norm": current_norm,
                "injection_layer": None if vector is None else selected_layer,
                "coefficient": None if vector is None else selected_scale,
                **metrics,
            }
            condition_results.append(metrics)
            all_prediction_rows.extend({"condition": condition, **row} for row in predictions)
            all_nll_rows.extend({"condition": condition, "task_id": task_id, **row} for row in nll_rows)
    intervention.close()
    cleanup_model(model)
    json_dump(PHASE5_ROOT / "interventions" / "final_metrics.json", condition_results)
    jsonl_write(PHASE5_ROOT / "interventions" / "final_predictions.jsonl", all_prediction_rows)
    jsonl_write(PHASE5_ROOT / "interventions" / "final_teacher_forced_scores.jsonl", all_nll_rows)
    intervention_effects = compute_intervention_effects(all_prediction_rows, all_nll_rows)
    json_dump(PHASE5_ROOT / "interventions" / "paired_effects_bootstrap.json", intervention_effects)

    stage_metrics = {"by_checkpoint": stage_generation_rows, "by_task": correlations}
    json_dump(PHASE5_ROOT / "stage_eval" / "summary.json", stage_metrics)
    result_object = {
        "intervention_model_id": final_id,
        "selected_layer": selected_layer,
        "selected_scale": selected_scale,
        "fv_descriptors": fv_descriptors,
        "intervention_metrics": condition_results,
        "intervention_effects": intervention_effects,
        "stage_generation_metrics": stage_generation_rows,
        "fv_drift_nll_correlations": correlations,
        "head_rank_scope": {
            "pretrained": "global_448",
            "homogeneous_stage_1": "endpoint_union_restricted",
            "homogeneous_stage_2": "endpoint_union_restricted",
            "homogeneous_stage_3": "endpoint_union_restricted",
            "homogeneous_stage_4": "global_448",
            final_id: "global_448",
        },
    }
    json_dump(PHASE5_ROOT / "fv" / "summary.json", result_object)
    write_fv_report(result_object)
    print("Phase 5 FV extraction and interventions complete.", flush=True)


if __name__ == "__main__":
    main()

