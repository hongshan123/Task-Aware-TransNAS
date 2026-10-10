"""Analyze full 448-query-head effects with identity-permutation nulls."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE5_ROOT = PROJECT_ROOT / "results" / "phase5" / "fv" / "head_effects" / "v2_bf16_dequantized_fp32" / "phase4b_A0_S1_seed10"
OUT_ROOT = PROJECT_ROOT / "results" / "phase5b" / "head_similarity"
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
TOP_KS = (5, 10, 20, 50)
REPEATS = 10_000
SEED = 314159


def _load_task(task_id: str) -> tuple[list[str], np.ndarray, list[tuple[int, int]]]:
    folder = PHASE5_ROOT / task_id
    files = sorted(folder.glob("query_*.json"))
    if len(files) != 10:
        raise RuntimeError(f"Expected 10 build query effects for {task_id}; found {len(files)}")
    query_ids: list[str] = []
    rows: list[dict[tuple[int, int], float]] = []
    expected: set[tuple[int, int]] | None = None
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("checkpoint_id") != "phase4b_A0_S1_seed10" or item.get("task_id") != task_id:
            raise RuntimeError(f"Unexpected endpoint scope in {path}")
        if item.get("candidate_scope") != "global_448" or len(item.get("scores", [])) != 448:
            raise RuntimeError(f"Not a complete global-448 effect file: {path}")
        scores = {
            (int(score["layer"]), int(score["query_head"])): float(score["sequence_delta_logp"])
            for score in item["scores"]
        }
        if len(scores) != 448:
            raise RuntimeError(f"Duplicate query head in {path}")
        if expected is None:
            expected = set(scores)
        elif set(scores) != expected:
            raise RuntimeError(f"Head set changed across queries for {task_id}")
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"Nonfinite causal score in {path}")
        rows.append(scores)
        query_ids.append(str(item["query_id"]))
    head_ids = sorted(expected or set())
    if len(head_ids) != 448:
        raise RuntimeError(f"Expected exactly 448 unique heads for {task_id}")
    matrix = np.asarray([[row[head] for head in head_ids] for row in rows], dtype=np.float64)
    return query_ids, matrix, head_ids


def _positive_mass(values: np.ndarray) -> tuple[np.ndarray, float]:
    positive = np.maximum(values, 0.0)
    total = float(positive.sum())
    if total <= 0:
        return np.zeros_like(positive), total
    return positive / total, total


def _top_indices(values: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(values, -k)[-k:]


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _empirical_p_upper(null: np.ndarray, observed: float) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def _empirical_p_two_sided(null: np.ndarray, observed: float) -> float:
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (null.size + 1))


def _bh_adjust(rows: list[dict[str, Any]], p_field: str, out_field: str) -> None:
    indexed = [(i, row[p_field]) for i, row in enumerate(rows) if row.get(p_field) is not None]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 1.0
    adjusted = [1.0] * m
    for rank_idx in range(m - 1, -1, -1):
        rank = rank_idx + 1
        running = min(running, float(indexed[rank_idx][1]) * m / rank)
        adjusted[rank_idx] = min(1.0, running)
    for (original_index, _), value in zip(indexed, adjusted):
        rows[original_index][out_field] = value


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    query_ids: dict[str, list[str]] = {}
    matrices: dict[str, np.ndarray] = {}
    head_ids: list[tuple[int, int]] | None = None
    for task in TASKS:
        qids, matrix, ids = _load_task(task)
        if head_ids is None:
            head_ids = ids
        elif ids != head_ids:
            raise RuntimeError("Task scans do not share one global 448-head universe")
        query_ids[task] = qids
        matrices[task] = matrix

    assert head_ids is not None
    means = {task: matrix.mean(axis=0) for task, matrix in matrices.items()}
    distributions = {task: _positive_mass(value)[0] for task, value in means.items()}
    top_sets = {
        task: {k: set(_top_indices(values, k).tolist()) for k in TOP_KS}
        for task, values in means.items()
    }
    stability = []
    for task, matrix in matrices.items():
        for k in TOP_KS:
            values = []
            sets = [_top_indices(row, k) for row in matrix]
            for left, right in itertools.combinations(sets, 2):
                a, b = set(left.tolist()), set(right.tolist())
                values.append(len(a & b) / len(a | b))
            stability.append({
                "task_id": task,
                "top_k": k,
                "query_pair_count": len(values),
                "mean_jaccard": float(np.mean(values)),
                "sd_jaccard": float(np.std(values, ddof=1)),
                "minimum_jaccard": float(np.min(values)),
                "maximum_jaccard": float(np.max(values)),
            })

    rng = np.random.default_rng(SEED)
    pair_rows = []
    all_null_details: dict[str, Any] = {}
    for task_a, task_b in itertools.combinations(TASKS, 2):
        a = means[task_a]
        b = means[task_b]
        pearson_observed = float(np.corrcoef(a, b)[0, 1])
        rank_a = rankdata(a, method="average")
        rank_b = rankdata(b, method="average")
        spearman_observed = float(np.corrcoef(rank_a, rank_b)[0, 1])
        p_a, pos_a = _positive_mass(a)
        p_b, pos_b = _positive_mass(b)
        weighted_observed = float(np.minimum(p_a, p_b).sum()) if pos_a and pos_b else None

        overlap_observed = {
            str(k): len(top_sets[task_a][k] & top_sets[task_b][k]) for k in TOP_KS
        }
        permutations = np.argsort(rng.random((REPEATS, 448)), axis=1)
        permuted_b = b[permutations]
        centered_a = a - a.mean()
        centered_b = permuted_b - permuted_b.mean(axis=1, keepdims=True)
        pearson_null = (centered_b @ centered_a) / np.sqrt(
            np.sum(centered_a**2) * np.sum(centered_b**2, axis=1)
        )
        ranks_b_perm = rank_b[permutations]
        centered_ranks_b = ranks_b_perm - ranks_b_perm.mean(axis=1, keepdims=True)
        spearman_null = (centered_ranks_b @ (rank_a - rank_a.mean())) / np.sqrt(
            np.sum((rank_a - rank_a.mean())**2) * np.sum(centered_ranks_b**2, axis=1)
        )
        if weighted_observed is not None:
            positive_b_perm = p_b[permutations]
            weighted_null = np.minimum(p_a[None, :], positive_b_perm).sum(axis=1)
        else:
            weighted_null = np.zeros(REPEATS, dtype=np.float64)

        inverse_permutations = np.argsort(permutations, axis=1)
        top_null: dict[str, np.ndarray] = {}
        for k in TOP_KS:
            b_top_original = _top_indices(b, k)
            new_positions = inverse_permutations[:, b_top_original]
            a_top = top_sets[task_a][k]
            a_mask = np.zeros(448, dtype=bool)
            a_mask[list(a_top)] = True
            overlaps = a_mask[new_positions].sum(axis=1)
            top_null[str(k)] = overlaps
        pair_key = f"{task_a}__{task_b}"
        null_summary: dict[str, Any] = {
            "randomization": "permute head identities of task B; retain its score distribution",
            "repeats": REPEATS,
            "seed": SEED,
            "pearson": {
                "observed": pearson_observed,
                "null_mean": float(pearson_null.mean()),
                "null_95_percentile_interval": _ci(pearson_null),
                "empirical_two_sided_p": _empirical_p_two_sided(pearson_null, pearson_observed),
            },
            "spearman": {
                "observed": spearman_observed,
                "null_mean": float(spearman_null.mean()),
                "null_95_percentile_interval": _ci(spearman_null),
                "empirical_two_sided_p": _empirical_p_two_sided(spearman_null, spearman_observed),
            },
            "positive_gain_weighted_overlap": {
                "observed": weighted_observed,
                "task_a_positive_mass": pos_a,
                "task_b_positive_mass": pos_b,
                "null_mean": float(weighted_null.mean()) if weighted_observed is not None else None,
                "null_95_percentile_interval": _ci(weighted_null) if weighted_observed is not None else None,
                "empirical_upper_tail_p": _empirical_p_upper(weighted_null, weighted_observed) if weighted_observed is not None else None,
            },
            "top_k_overlap": {},
        }
        for k in TOP_KS:
            null = top_null[str(k)]
            observed = overlap_observed[str(k)]
            null_summary["top_k_overlap"][str(k)] = {
                "observed_count": observed,
                "observed_jaccard": observed / (2 * k - observed),
                "uniform_random_expected_count": k * k / 448,
                "null_mean_count": float(null.mean()),
                "null_95_percentile_interval_count": _ci(null.astype(np.float64)),
                "empirical_upper_tail_p": _empirical_p_upper(null, observed),
            }
        pair_rows.append({
            "task_a": task_a,
            "task_b": task_b,
            "pearson": pearson_observed,
            "pearson_permutation_p": null_summary["pearson"]["empirical_two_sided_p"],
            "spearman": spearman_observed,
            "spearman_permutation_p": null_summary["spearman"]["empirical_two_sided_p"],
            "positive_gain_weighted_overlap": weighted_observed,
            "weighted_overlap_permutation_p": null_summary["positive_gain_weighted_overlap"]["empirical_upper_tail_p"],
            **{f"top_{k}_overlap": overlap_observed[str(k)] for k in TOP_KS},
            **{f"top_{k}_permutation_p": null_summary["top_k_overlap"][str(k)]["empirical_upper_tail_p"] for k in TOP_KS},
        })
        all_null_details[pair_key] = null_summary

    _bh_adjust(pair_rows, "pearson_permutation_p", "pearson_permutation_p_bh_6_pairs")
    _bh_adjust(pair_rows, "spearman_permutation_p", "spearman_permutation_p_bh_6_pairs")
    _bh_adjust(pair_rows, "weighted_overlap_permutation_p", "weighted_overlap_permutation_p_bh_6_pairs")
    for k in TOP_KS:
        _bh_adjust(pair_rows, f"top_{k}_permutation_p", f"top_{k}_permutation_p_bh_6_pairs")

    with (OUT_ROOT / "pairwise_similarity.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    with (OUT_ROOT / "within_task_topk_stability.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability[0]))
        writer.writeheader()
        writer.writerows(stability)
    payload = {
        "phase": "5B",
        "checkpoint_id": "phase4b_A0_S1_seed10",
        "candidate_scope": "global_448",
        "tasks": TASKS,
        "query_ids": query_ids,
        "head_ids_ordered": [{"layer": layer, "query_head": head} for layer, head in head_ids],
        "top_k": list(TOP_KS),
        "permutation_repeats": REPEATS,
        "random_seed": SEED,
        "permutation_null": "independently permute task B head identity, preserving its causal score distribution",
        "mean_sequence_delta_logp_by_task": {task: means[task].tolist() for task in TASKS},
        "pairwise_null_details": all_null_details,
        "within_task_topk_stability": stability,
        "test_split_loaded": False,
    }
    (OUT_ROOT / "full_448_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT_ROOT), "pair_count": len(pair_rows), "query_count_by_task": {key: len(value) for key, value in query_ids.items()}}, indent=2))


if __name__ == "__main__":
    main()
