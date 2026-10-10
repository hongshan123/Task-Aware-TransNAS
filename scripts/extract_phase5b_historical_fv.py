"""Extract same-trajectory task-boundary Function Vectors for Phase 5B."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase5_common import (  # noqa: E402
    FV_VERSION,
    PATCH_BATCH_SIZE,
    TASK_1603,
    TASK_565,
    HeadActivationPatcher,
    HeadActivationRecorder,
    cleanup_model,
    json_dump,
    load_checkpoint,
    load_phase5_context,
    make_icl_sets,
    model_head_geometry,
    reference_sequence,
)


PHASE5B_ROOT = PROJECT_ROOT / "results" / "phase5b"
PHASE5_FV_ROOT = PROJECT_ROOT / "results" / "phase5" / "fv"
BOUNDARY_MANIFEST = PHASE5B_ROOT / "boundary_replay_manifest.json"
SNAPSHOTS = [
    ("historical_after_task565", TASK_565, "task565_circa_answer_generation"),
    ("historical_after_task1603", TASK_1603, "task1603_smcalflow_sentence_generation"),
]
EXPECTED_GEOMETRY = {
    "layers": 28,
    "query_heads": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 1024,
    "o_proj_input_width": 2048,
}


def _effect_path(checkpoint_id: str, task_id: str, index: int) -> Path:
    return PHASE5B_ROOT / "extraction" / "head_effects" / checkpoint_id / task_id / f"query_{index:02d}.json"


def _cache_valid(path: Path, checkpoint_id: str, task_id: str, query_id: str) -> bool:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        scores = item["scores"]
        pairs = {(int(row["layer"]), int(row["query_head"])) for row in scores}
        return (
            item.get("checkpoint_id") == checkpoint_id
            and item.get("task_id") == task_id
            and item.get("query_id") == query_id
            and item.get("candidate_scope") == "global_448"
            and len(scores) == 448
            and len(pairs) == 448
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def _validate_build_packs(packs: dict[str, list[Any]], prompt_manifest: dict[str, Any]) -> None:
    for _snapshot, task_id, _manifest_task in SNAPSHOTS:
        expected = prompt_manifest[task_id]
        actual = packs[task_id]
        if len(expected) != 10 or len(actual) != 10:
            raise RuntimeError(f"Expected 10 Phase 5 build prompts for {task_id}")
        for row, pack in zip(expected, actual):
            if (
                row["query_id"] != pack.query_id
                or row["support_train_ids"] != pack.support_ids
                or row["shots"] != pack.shots
                or row["correct_prompt_tokens"] != pack.correct_prompt_tokens
            ):
                raise RuntimeError(f"Phase 5 prompt manifest mismatch for {task_id}/{pack.query_id}")
            if pack.shots != 4 or len(pack.support_ids) != 4:
                raise RuntimeError(f"Expected four train demonstrations for {task_id}/{pack.query_id}")


def _rank_and_save(
    model: Any,
    recorder: HeadActivationRecorder,
    model_path: Path,
    checkpoint_id: str,
    task_id: str,
    checkpoint_path: Path,
    packs: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    means_path = PHASE5B_ROOT / "extraction" / "activations" / f"{checkpoint_id}__{task_id}.pt"
    if means_path.is_file():
        means = torch.load(means_path, map_location="cpu", weights_only=True).to(device=device, dtype=torch.float32)
        if tuple(means.shape) != (28, 16, 1024) or not bool(torch.isfinite(means).all()):
            raise RuntimeError(f"Invalid activation cache: {means_path}")
    else:
        rows = []
        for index, pack in enumerate(packs, start=1):
            rows.append(recorder.record(pack.correct_ids, device))
            print(f"activations {checkpoint_id}/{task_id}: {index}/10", flush=True)
        means = torch.stack(rows).mean(dim=0).float().to(device)
        means_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(means.cpu(), means_path)

    missing = [
        i for i, pack in enumerate(packs)
        if not _cache_valid(_effect_path(checkpoint_id, task_id, i), checkpoint_id, task_id, pack.query_id)
    ]
    patcher = HeadActivationPatcher(model, means)
    candidates = [(layer, head) for layer in range(28) for head in range(16)]
    try:
        for index in missing:
            pack = packs[index]
            baseline = reference_sequence(model, pack.counterfactual_ids, pack.target_ids, device)
            correct = reference_sequence(model, pack.correct_ids, pack.target_ids, device)
            scores = patcher.run_candidates(
                pack.counterfactual_ids, pack.target_ids, candidates, batch_size=PATCH_BATCH_SIZE
            )
            p0 = math.exp(max(-80.0, baseline["first_token_logp"]))
            for score in scores:
                score["sequence_delta_logp"] = score["mean_sequence_logp"] - baseline["mean_logp"]
                score["sequence_delta_nll"] = -score["sequence_delta_logp"]
                score["first_token_delta_logp"] = score["first_token_logp"] - baseline["first_token_logp"]
                score["first_token_delta_probability"] = math.exp(
                    max(-80.0, score["first_token_logp"])
                ) - p0
            payload = {
                "phase": "5B",
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path),
                "model_source": str(model_path),
                "fv_version": FV_VERSION,
                "task_id": task_id,
                "query_id": pack.query_id,
                "candidate_scope": "global_448",
                "baseline_counterfactual": {
                    "mean_sequence_logp": baseline["mean_logp"],
                    "first_token_logp": baseline["first_token_logp"],
                },
                "correct_prompt_reference": {
                    "mean_sequence_logp": correct["mean_logp"],
                    "counterfactual_mean_logp_gap": correct["mean_logp"] - baseline["mean_logp"],
                },
                "correct_icl_prompt_tokens": pack.correct_prompt_tokens,
                "counterfactual_icl_prompt_tokens": pack.counterfactual_prompt_tokens,
                "shots": pack.shots,
                "support_train_ids": pack.support_ids,
                "scores": scores,
            }
            json_dump(_effect_path(checkpoint_id, task_id, index), payload)
            print(f"causal sweep {checkpoint_id}/{task_id}: query {index + 1}/10", flush=True)
    finally:
        patcher.close()

    score_lists: dict[tuple[int, int], list[dict[str, float]]] = {}
    for index, pack in enumerate(packs):
        path = _effect_path(checkpoint_id, task_id, index)
        if not _cache_valid(path, checkpoint_id, task_id, pack.query_id):
            raise RuntimeError(f"Incomplete global 448 causal effect file: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        for row in item["scores"]:
            key = (int(row["layer"]), int(row["query_head"]))
            score_lists.setdefault(key, []).append(row)

    rankings = []
    for (layer, head), rows in score_lists.items():
        gains = [float(row["sequence_delta_logp"]) for row in rows]
        rankings.append({
            "layer": layer,
            "query_head": head,
            "candidate_scope": "global_448",
            "ce_mean_sequence_logp_gain": statistics.mean(gains),
            "ce_std_sequence_logp_gain": statistics.stdev(gains),
            "ce_mean_first_token_logp_gain": statistics.mean(float(row["first_token_delta_logp"]) for row in rows),
            "n_queries": len(rows),
        })
    rankings.sort(key=lambda row: row["ce_mean_sequence_logp_gain"], reverse=True)
    rank_path = PHASE5B_ROOT / "extraction" / "rankings" / f"{checkpoint_id}__{task_id}.json"
    json_dump(rank_path, rankings)
    top = rankings[:10]
    vector = torch.stack([means[int(row["layer"]), int(row["query_head"])] for row in top]).sum(dim=0).detach().float().cpu()
    if tuple(vector.shape) != (1024,) or not bool(torch.isfinite(vector).all()):
        raise RuntimeError(f"Invalid historical FV for {checkpoint_id}/{task_id}")
    vector_path = PHASE5B_ROOT / "extraction" / "historical_function_vectors.pt"
    if vector_path.exists():
        vectors = torch.load(vector_path, map_location="cpu", weights_only=True)
    else:
        vectors = {}
    vectors.setdefault(checkpoint_id, {})[task_id] = vector
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, vector_path)
    return {
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "checkpoint_path": str(checkpoint_path),
        "activation_path": str(means_path),
        "rank_path": str(rank_path),
        "vector_path": str(vector_path),
        "vector_l2_norm": float(vector.norm().item()),
        "vector_rms": float(vector.square().mean().sqrt().item()),
        "top_heads": top,
        "queries": [pack.query_id for pack in packs],
        "support_train_ids": [pack.support_ids for pack in packs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5B extraction requires CUDA")
    replay = json.loads(BOUNDARY_MANIFEST.read_text(encoding="utf-8"))
    if replay.get("status") != "complete" or not replay.get("final_weight_comparison", {}).get("matches_within_tolerance"):
        raise RuntimeError("Same-trajectory boundary replay is not certified complete")
    config, model_path, tokenizer, device, train, dev = load_phase5_context()
    packs = make_icl_sets(tokenizer, train, dev)
    prompt_manifest = json.loads((PHASE5_FV_ROOT / "icl_prompt_manifest.json").read_text(encoding="utf-8"))
    _validate_build_packs(packs, prompt_manifest)
    geometry = EXPECTED_GEOMETRY
    if (
        int(config["model"]["max_sequence_length"]) < 256
        or not model_path.is_dir()
    ):
        raise RuntimeError("Local model/config preflight failed")
    boundary_rows = []
    for checkpoint_id, task_id, manifest_task in SNAPSHOTS:
        row = replay["boundary_checkpoints"][manifest_task]
        checkpoint_path = Path(row["checkpoint_path"])
        if not (checkpoint_path / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint_path / "model.safetensors")
        boundary_rows.append({"checkpoint_id": checkpoint_id, "task_id": task_id, "checkpoint_path": str(checkpoint_path)})
    report = {
        "phase": "5B",
        "status": "preflight_passed",
        "model_path": str(model_path),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0),
        "head_geometry_expected": geometry,
        "historical_checkpoints": boundary_rows,
        "build_queries_per_task": 10,
        "all_prompt_ids_and_support_ids_match_phase5": True,
        "test_split_loaded": False,
        "nas_started": False,
    }
    json_dump(PHASE5B_ROOT / "extraction" / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    source_vectors = torch.load(PHASE5_FV_ROOT / "function_vectors.pt", map_location="cpu", weights_only=True)
    final_id = "phase4b_A0_S1_seed10"
    baseline_id = "pretrained"
    if final_id not in source_vectors or baseline_id not in source_vectors:
        raise RuntimeError("Phase 5 pretrained/final Function Vectors are missing")
    final_vectors = source_vectors[final_id]
    baseline_vectors = source_vectors[baseline_id]
    summaries = []
    for checkpoint_id, task_id, manifest_task in SNAPSHOTS:
        boundary = replay["boundary_checkpoints"][manifest_task]
        checkpoint_path = Path(boundary["checkpoint_path"])
        model = load_checkpoint(model_path, checkpoint_path, architecture="A0")
        actual_geometry = model_head_geometry(model)
        if actual_geometry != geometry:
            cleanup_model(model)
            raise RuntimeError(f"Unexpected Qwen3 geometry: {actual_geometry}")
        recorder = HeadActivationRecorder(model)
        try:
            summary = _rank_and_save(
                model, recorder, model_path, checkpoint_id, task_id, checkpoint_path,
                packs[task_id], device,
            )
            summary["checkpoint_stage"] = int(boundary["stage"])
            summary["cosine_to_same_task_final_collapsed_fv"] = float(F.cosine_similarity(
                torch.load(summary["vector_path"], map_location="cpu", weights_only=True)[checkpoint_id][task_id].view(1, -1),
                final_vectors[task_id].float().view(1, -1), dim=1,
            ).item())
            summary["cosine_to_same_task_pretrained_fv"] = float(F.cosine_similarity(
                torch.load(summary["vector_path"], map_location="cpu", weights_only=True)[checkpoint_id][task_id].view(1, -1),
                baseline_vectors[task_id].float().view(1, -1), dim=1,
            ).item())
            summaries.append(summary)
            print(f"completed historical vector {checkpoint_id}/{task_id}", flush=True)
        finally:
            recorder.close()
            cleanup_model(model)

    descriptor_path = PHASE5B_ROOT / "extraction" / "historical_vector_descriptors.json"
    json_dump(descriptor_path, {
        "phase": "5B",
        "fv_definition": "sum of mean last-token per-query-head o_proj contributions for top-10 globally ranked by sequence log-probability gain, measured on 10 four-shot correct-vs-label-shuffled build prompts",
        "source_fv_version": FV_VERSION,
        "summaries": summaries,
        "test_split_loaded": False,
    })
    print(f"saved {descriptor_path}", flush=True)


if __name__ == "__main__":
    main()
