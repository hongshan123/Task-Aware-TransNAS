"""Recover the missing after-task1501 boundary from the verified Phase 5B stage-2 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from architecture_modules import apply_architecture, parameter_counts  # noqa: E402
from phase4_common import TASK_IDS, prepare_context  # noqa: E402
from run_formal_baseline import load_model, safe_task_seed, save_checkpoint  # noqa: E402
from phase4_common import train_task_recorded  # noqa: E402


ORIGINAL_RUN = PROJECT_ROOT / "results" / "phase4b" / "runs" / "A0" / "S1" / "seed_10" / "metrics.json"
STAGE2 = PROJECT_ROOT / "results" / "phase5b" / "checkpoints" / "historical_boundary_replay" / "A0" / "S1" / "seed_10" / f"after_2_{TASK_IDS[1]}"
OUTPUT = PROJECT_ROOT / "results" / "phase6a" / "checkpoints" / f"after_3_{TASK_IDS[2]}"
MANIFEST = PROJECT_ROOT / "results" / "phase6a" / "checkpoints" / "after_task1501_replay_manifest.json"
TASK_ID = TASK_IDS[2]
EXPECTED_STEPS = 50
TOLERANCE = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ids(records: list[dict[str, Any]]) -> str:
    values = [str(row["instance_id"]) for row in records]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    config, model_path, tokenizer, device, train, _dev = prepare_context()
    original = json.loads(ORIGINAL_RUN.read_text(encoding="utf-8"))
    phase5b_manifest = json.loads((PROJECT_ROOT / "results" / "phase5b" / "boundary_replay_manifest.json").read_text(encoding="utf-8"))
    if phase5b_manifest.get("status") != "complete" or not phase5b_manifest["final_weight_comparison"].get("matches_within_tolerance"):
        raise RuntimeError("Phase 5B same-trajectory replay is not certified complete")
    if original.get("architecture") != "A0" or original.get("sequence_name") != "S1" or original.get("seed") != 10:
        raise RuntimeError("Original run is not A0/S1/seed10")
    if not (STAGE2 / "model.safetensors").is_file():
        raise FileNotFoundError(STAGE2 / "model.safetensors")
    if sha256_file(model_path / "model.safetensors") != original["source_model_sha256"]:
        raise RuntimeError("Local base model weights do not match the original A0/S1 run")
    train_hashes = {task: digest_ids(train[task]) for task in TASK_IDS}
    if train_hashes != original.get("train_instance_id_sha256_by_task"):
        raise RuntimeError("Train IDs differ from the original A0/S1 run")
    if phase5b_manifest.get("train_instance_id_sha256_by_task") != train_hashes:
        raise RuntimeError("Phase 5B replay and current task stream have different training IDs")

    required_bytes = (PROJECT_ROOT / "results" / "phase4b" / "checkpoints" / "A0" / "S1" / "seed_10" / "model.safetensors").stat().st_size + 1024**3
    free_bytes = shutil.disk_usage(OUTPUT.anchor).free
    preflight = {
        "source_phase5b_replay_manifest": str(PROJECT_ROOT / "results" / "phase5b" / "boundary_replay_manifest.json"),
        "resume_checkpoint": str(STAGE2),
        "resume_checkpoint_sha256": sha256_file(STAGE2 / "model.safetensors"),
        "output_checkpoint": str(OUTPUT),
        "model_path": str(model_path),
        "source_model_sha256_match": True,
        "training_id_hashes_match": True,
        "task_id": TASK_ID,
        "task_steps": EXPECTED_STEPS,
        "task_seed": safe_task_seed(10, TASK_ID),
        "required_free_disk_bytes": required_bytes,
        "free_disk_bytes": free_bytes,
        "test_split_loaded": False,
        "nas_started": False,
    }
    if free_bytes < required_bytes:
        raise RuntimeError(f"Insufficient C: free space for one boundary checkpoint: {preflight}")
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        return

    if MANIFEST.is_file() and OUTPUT.joinpath("model.safetensors").is_file():
        cached = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if cached.get("status") == "complete" and cached.get("resume_checkpoint_sha256") == preflight["resume_checkpoint_sha256"]:
            print(json.dumps(cached, ensure_ascii=False, indent=2), flush=True)
            return
        raise RuntimeError("Existing Phase 6A stage-3 replay is incomplete or has different provenance")

    model = load_model(model_path, local_only=True, training=True)
    torch.manual_seed(10)
    apply_architecture(model, "A0")
    if parameter_counts(model)["total_parameters"] != original["parameter_counts"]["total_parameters"]:
        raise RuntimeError("A0 model parameter count differs from original trajectory")
    state = load_file(str(STAGE2 / "model.safetensors"), device="cpu")
    incompat = model.load_state_dict(state, strict=False)
    missing = set(incompat.missing_keys) - {"lm_head.weight"}
    if missing or incompat.unexpected_keys:
        raise RuntimeError(f"Task-2 checkpoint loading mismatch: missing={sorted(missing)}, unexpected={incompat.unexpected_keys[:8]}")
    del state

    metrics = train_task_recorded(
        model,
        tokenizer,
        train[TASK_ID],
        config["training"],
        config["model"],
        device,
        float(config["training"]["learning_rate"]),
        EXPECTED_STEPS,
        safe_task_seed(10, TASK_ID),
    )
    metrics["task_id"] = TASK_ID
    metrics["training_seed"] = safe_task_seed(10, TASK_ID)
    reference_metrics = original["training_metrics"][TASK_ID]
    checks = {
        "sample_order_sha256_match": metrics["sample_order_sha256"] == reference_metrics["sample_order_sha256"],
        "supervised_target_tokens_match": metrics["supervised_target_tokens"] == reference_metrics["supervised_target_tokens"],
        "optimizer_steps_match": metrics["optimizer_steps"] == reference_metrics["optimizer_steps"],
        "effective_max_sequence_length_match": metrics["effective_max_sequence_length"] == reference_metrics["effective_max_sequence_length"],
    }
    loss_diffs = {key: abs(float(metrics[key]) - float(reference_metrics[key])) for key in ("first_loss", "last_loss", "mean_loss")}
    if not all(checks.values()) or max(loss_diffs.values()) > TOLERANCE:
        raise RuntimeError(f"Replayed first-classifier stage differs from original: checks={checks}, loss_diffs={loss_diffs}")
    save_checkpoint(model, tokenizer, OUTPUT)
    manifest = {
        "phase": "6A",
        "status": "complete",
        **preflight,
        "source_model_sha256": original["source_model_sha256"],
        "train_instance_id_sha256_by_task": train_hashes,
        "replayed_training_metrics": metrics,
        "original_training_metrics": {key: reference_metrics[key] for key in ("first_loss", "last_loss", "mean_loss", "sample_order_sha256", "supervised_target_tokens", "optimizer_steps", "effective_max_sequence_length")},
        "checks": checks,
        "loss_absolute_differences": loss_diffs,
        "max_loss_absolute_difference": max(loss_diffs.values()),
        "test_split_loaded": False,
        "nas_started": False,
        "formal_new_sequence_training": False,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
