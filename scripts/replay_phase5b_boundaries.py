"""Replay one completed A0/S1 run to retain its task-1 and task-2 boundaries.

The replay writes only under results/phase5b. It reads the original Phase 4B
metrics and final checkpoint for provenance and comparison, and never edits the
completed Phase 4B experiment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from architecture_modules import apply_architecture, parameter_counts
from phase4_common import TASK_IDS, prepare_context, safe_task_seed, train_task_recorded
from run_formal_baseline import load_model, save_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
PHASE5B_ROOT = storage_path("results") / "phase5b"
ORIGINAL_RUN = storage_path("results") / "phase4b" / "runs" / "A0" / "S1" / "seed_10" / "metrics.json"
ORIGINAL_CHECKPOINT = (
    storage_path("results") / "phase4b" / "checkpoints" / "A0" / "S1" / "seed_10" / "model.safetensors"
)
REPLAY_ROOT = PHASE5B_ROOT / "checkpoints" / "historical_boundary_replay" / "A0" / "S1" / "seed_10"
MANIFEST_PATH = PHASE5B_ROOT / "boundary_replay_manifest.json"
STEPS_BY_TASK = {
    TASK_IDS[0]: 50,
    TASK_IDS[1]: 250,
    TASK_IDS[2]: 50,
    TASK_IDS[3]: 50,
}
BOUNDARY_STAGE_BY_TASK = {
    TASK_IDS[0]: 1,
    TASK_IDS[1]: 2,
}
MAX_GRADIENT_TOLERANCE = 1e-5


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest_instance_ids(records: list[dict[str, Any]]) -> str:
    ids = [str(record["instance_id"]) for record in records]
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode("utf-8")).hexdigest()


def boundary_path(stage: int) -> Path:
    task_id = TASK_IDS[stage - 1]
    return REPLAY_ROOT / f"after_{stage}_{task_id}"


def load_boundary_state(model: Any, checkpoint: Path) -> None:
    weights_path = checkpoint / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing resume weights: {weights_path}")
    state = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys) - {"lm_head.weight"}
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Boundary checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={incompatible.unexpected_keys[:12]}"
        )
    del state
    model.eval()
    model.config.use_cache = True


def compare_task_metrics(
    task_id: str,
    replay: dict[str, Any],
    original: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "sample_order_sha256_match": replay["sample_order_sha256"] == original["sample_order_sha256"],
        "supervised_target_tokens_match": replay["supervised_target_tokens"] == original["supervised_target_tokens"],
        "optimizer_steps_match": replay["optimizer_steps"] == original["optimizer_steps"],
        "effective_max_sequence_length_match": (
            replay["effective_max_sequence_length"] == original["effective_max_sequence_length"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Replay provenance mismatch for {task_id}: {checks}")
    numeric = {}
    for key in ("first_loss", "last_loss", "mean_loss"):
        replay_value = float(replay[key])
        original_value = float(original[key])
        numeric[key] = {
            "replay": replay_value,
            "original": original_value,
            "absolute_difference": abs(replay_value - original_value),
        }
    return {
        "checks": checks,
        "losses": numeric,
        "losses_match_within_tolerance": all(
            row["absolute_difference"] <= MAX_GRADIENT_TOLERANCE for row in numeric.values()
        ),
    }


def compare_final_weights(model: Any, reference_path: Path) -> dict[str, Any]:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing original Phase 4B checkpoint: {reference_path}")
    current_state = model.state_dict()
    count = 0
    max_abs_difference = 0.0
    squared_difference_sum = 0.0
    mismatch_count = 0
    max_difference_key = None
    with safe_open(str(reference_path), framework="pt", device="cpu") as reference:
        reference_keys = set(reference.keys())
        current_keys = set(current_state)
        unexpected = sorted(reference_keys - current_keys)
        if unexpected:
            raise RuntimeError(f"Original checkpoint has unexpected keys: {unexpected[:12]}")
        missing_from_reference = sorted(current_keys - reference_keys - {"lm_head.weight"})
        for key in sorted(reference_keys):
            target = current_state[key].detach().to(device="cpu", dtype=torch.float32)
            expected = reference.get_tensor(key).to(dtype=torch.float32)
            difference = (target - expected).abs()
            local_max = float(difference.max().item()) if difference.numel() else 0.0
            if local_max > max_abs_difference:
                max_abs_difference = local_max
                max_difference_key = key
            mismatch_count += int((difference > MAX_GRADIENT_TOLERANCE).sum().item())
            squared_difference_sum += float(difference.square().sum().item())
            count += difference.numel()
            del target, expected, difference
    return {
        "reference_checkpoint": str(reference_path),
        "compared_parameter_count": count,
        "max_abs_difference": max_abs_difference,
        "max_difference_parameter": max_difference_key,
        "mean_abs_difference": math.sqrt(squared_difference_sum / max(1, count)),
        "parameter_count_above_tolerance": mismatch_count,
        "missing_tied_keys_in_reference": missing_from_reference,
        "tolerance": MAX_GRADIENT_TOLERANCE,
        "matches_within_tolerance": mismatch_count == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume from the last saved task boundary")
    parser.add_argument("--preflight-only", action="store_true", help="validate data, checkpoint, and disk without training")
    args = parser.parse_args()

    config, model_path, tokenizer, device, train, _dev = prepare_context()
    if not ORIGINAL_RUN.is_file() or not ORIGINAL_CHECKPOINT.is_file():
        raise FileNotFoundError("The original A0/S1/seed-10 run and final weights are required")
    original = json.loads(ORIGINAL_RUN.read_text(encoding="utf-8"))
    if (
        original.get("architecture") != "A0"
        or original.get("sequence_name") != "S1"
        or original.get("seed") != 10
        or original.get("task_order") != TASK_IDS
        or original.get("test_split_loaded") is not False
    ):
        raise RuntimeError("The selected Phase 4B run does not match the expected A0/S1/seed-10 trajectory")
    id_hashes = {task_id: digest_instance_ids(train[task_id]) for task_id in TASK_IDS}
    if id_hashes != original.get("train_instance_id_sha256_by_task"):
        raise RuntimeError("Current train records do not match the original Phase 4B task data")

    required_bytes = 2 * ORIGINAL_CHECKPOINT.stat().st_size + 2 * 1024**3
    free_bytes = shutil.disk_usage(REPLAY_ROOT.anchor).free
    preflight = {
        "original_run": str(ORIGINAL_RUN),
        "original_checkpoint": str(ORIGINAL_CHECKPOINT),
        "architecture": "A0",
        "sequence": "S1",
        "run_seed": 10,
        "task_order_matches": original["task_order"] == TASK_IDS,
        "source_model_sha256": original["source_model_sha256"],
        "source_model_path": str(model_path),
        "train_instance_id_sha256_match": True,
        "checkpoint_bytes_each": ORIGINAL_CHECKPOINT.stat().st_size,
        "free_disk_bytes": free_bytes,
        "required_free_disk_bytes_with_margin": required_bytes,
        "test_split_loaded": False,
        "historical_experiment_will_be_modified": False,
    }
    if free_bytes < required_bytes:
        raise RuntimeError(f"Insufficient disk space for two boundary checkpoints: {preflight}")
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return
    if MANIFEST_PATH.exists() and not args.resume:
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
            return
        raise RuntimeError("An incomplete Phase 5B replay exists; pass --resume to continue it")

    source_model_sha = str(original["source_model_sha256"])
    replay_manifest: dict[str, Any] = {
        "phase": "5B",
        "status": "running",
        "architecture": "A0",
        "sequence": "S1",
        "run_seed": 10,
        "task_order": TASK_IDS,
        "source_model": str(model_path),
        "source_model_sha256": source_model_sha,
        "train_instance_id_sha256_by_task": id_hashes,
        "completed_stage": 0,
        "boundary_checkpoints": {},
        "replayed_training_metrics": {},
        "comparison_with_original_training": {},
        "dev_evaluated_during_replay": False,
        "test_split_loaded": False,
        "training_started": True,
        "nas_started": False,
    }
    start_stage = 0
    if args.resume and MANIFEST_PATH.is_file():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if previous.get("source_model_sha256") != source_model_sha or previous.get("train_instance_id_sha256_by_task") != id_hashes:
            raise RuntimeError("Cannot resume: replay source weights or train records differ")
        start_stage = int(previous.get("completed_stage", 0))
        if start_stage < 0 or start_stage > 2:
            raise RuntimeError(f"Unsupported replay boundary stage for resume: {start_stage}")
        if start_stage:
            replay_manifest.update(previous)
    if start_stage == 2 and replay_manifest.get("status") == "complete":
        print(json.dumps(replay_manifest, ensure_ascii=False, indent=2), flush=True)
        return

    model = load_model(model_path, local_only=True, training=True)
    torch.manual_seed(10)
    apply_architecture(model, "A0")
    counts = parameter_counts(model)
    if counts["total_parameters"] != original["parameter_counts"]["total_parameters"]:
        raise RuntimeError(f"Parameter count differs from original A0 run: {counts}")
    if start_stage:
        load_boundary_state(model, boundary_path(start_stage))
        print(f"Resuming after task stage {start_stage}", flush=True)

    for stage_index in range(start_stage + 1, len(TASK_IDS) + 1):
        task_id = TASK_IDS[stage_index - 1]
        task_seed = safe_task_seed(10, task_id)
        print(
            f"Replay A0/S1/seed-10 task {stage_index}/4: {task_id} "
            f"({STEPS_BY_TASK[task_id]} steps)",
            flush=True,
        )
        train_metrics = train_task_recorded(
            model,
            tokenizer,
            train[task_id],
            config["training"],
            config["model"],
            device,
            float(config["training"]["learning_rate"]),
            STEPS_BY_TASK[task_id],
            task_seed,
        )
        train_metrics["task_id"] = task_id
        train_metrics["training_seed"] = task_seed
        replay_manifest["replayed_training_metrics"][task_id] = train_metrics
        original_training = original["training_metrics"][task_id]
        comparison = compare_task_metrics(task_id, train_metrics, original_training)
        replay_manifest["comparison_with_original_training"][task_id] = comparison
        if stage_index in BOUNDARY_STAGE_BY_TASK.values():
            checkpoint = boundary_path(stage_index)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, tokenizer, checkpoint)
            replay_manifest["boundary_checkpoints"][task_id] = {
                "stage": stage_index,
                "checkpoint_path": str(checkpoint),
                "weights_path": str(checkpoint / "model.safetensors"),
                "weights_bytes": (checkpoint / "model.safetensors").stat().st_size,
                "train_metrics_match": comparison["checks"],
                "losses_match_within_tolerance": comparison["losses_match_within_tolerance"],
            }
            replay_manifest["completed_stage"] = stage_index
            replay_manifest["status"] = "boundaries_ready" if stage_index == 2 else "running"
            atomic_json(MANIFEST_PATH, replay_manifest)
        print(
            f"Finished {task_id}: order_match="
            f"{comparison['checks']['sample_order_sha256_match']}, "
            f"loss_match={comparison['losses_match_within_tolerance']}",
            flush=True,
        )

    replay_manifest["final_weight_comparison"] = compare_final_weights(model, ORIGINAL_CHECKPOINT)
    replay_manifest["status"] = "complete"
    replay_manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(MANIFEST_PATH, replay_manifest)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(replay_manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
