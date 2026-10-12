"""Verify recreated Linux CITB splits against the frozen Windows sample IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .runtime_paths import resolve_config_paths, resolve_runtime_path
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import resolve_config_paths, resolve_runtime_path


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def windows_crlf_sha256(path: Path) -> str:
    """Hash a Linux text file with the CRLF checkout convention used on Windows."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(content).hexdigest()


def verify_and_rebase_manifest(
    frozen: dict[str, Any],
    recreated: dict[str, Any],
    *,
    expected_task_ids: list[str],
    expected_source_commit: str,
    config_path: Path,
    config_sha256: str,
) -> dict[str, Any]:
    """Require identical ordered IDs and build a Linux-specific stream manifest."""
    frozen_tasks = {task["task_id"]: task for task in frozen.get("tasks", [])}
    recreated_tasks = {task["task_id"]: task for task in recreated.get("tasks", [])}
    frozen_order = [task.get("task_id") for task in frozen.get("tasks", [])]
    recreated_order = [task.get("task_id") for task in recreated.get("tasks", [])]
    if frozen_order != expected_task_ids or recreated_order != expected_task_ids:
        raise ValueError(
            f"Task order mismatch: expected {expected_task_ids}, frozen {frozen_order}, recreated {recreated_order}"
        )
    if recreated.get("source", {}).get("commit") != expected_source_commit:
        raise ValueError("Recreated manifest does not use the pinned CITB source commit")
    frozen_source_commit = frozen.get("data_source", {}).get("commit")
    if frozen_source_commit and frozen_source_commit != expected_source_commit:
        raise ValueError("Frozen manifest does not refer to the pinned CITB source commit")
    if recreated.get("leakage_check", {}).get("passed") is not True:
        raise ValueError("Recreated dataset leakage check did not pass")

    rebased = copy.deepcopy(frozen)
    rebased["formal_config"] = {"path": str(config_path), "sha256": config_sha256}
    rebased["data_source"] = copy.deepcopy(recreated["source"])
    rebased["leakage_check"] = copy.deepcopy(recreated["leakage_check"])
    rebased_tasks: list[dict[str, Any]] = []
    id_checks: list[dict[str, Any]] = []
    for task_id in expected_task_ids:
        old = frozen_tasks.get(task_id)
        new = recreated_tasks.get(task_id)
        if old is None or new is None:
            raise ValueError(f"Missing frozen or recreated task: {task_id}")
        if new.get("counts") != {"train": 500, "dev": 50, "test": 50}:
            raise ValueError(f"Unexpected 500/50/50 split counts for {task_id}: {new.get('counts')}")
        if old.get("counts") != new.get("counts"):
            raise ValueError(f"Split count mismatch for {task_id}")
        old_position = old.get("official_stream_position_1based", old.get("stream_position_1based"))
        if old_position is not None and old_position != new.get("stream_position_1based"):
            raise ValueError(f"Official CITB stream position changed for {task_id}")
        old_ids = old.get("instance_ids", {})
        new_ids = new.get("instance_ids", {})
        if old_ids != new_ids:
            differing = [split for split in ("train", "dev", "test") if old_ids.get(split) != new_ids.get(split)]
            raise ValueError(f"Frozen sample IDs or order changed for {task_id}: {differing}")
        sets = {split: set(new_ids[split]) for split in ("train", "dev", "test")}
        if sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"]:
            raise ValueError(f"Split leakage detected for {task_id}")
        historical_source_sha = old.get("source_json_sha256")
        recreated_source_sha = new.get("source_sha256")
        source_path = Path(new.get("source_json", ""))
        crlf_sha = windows_crlf_sha256(source_path) if source_path.is_file() else None
        source_bytes_match = historical_source_sha == recreated_source_sha
        windows_line_ending_match = bool(
            historical_source_sha and crlf_sha and historical_source_sha == crlf_sha
        )
        if historical_source_sha and not (source_bytes_match or windows_line_ending_match):
            raise ValueError(f"CITB source checksum differs beyond Windows CRLF normalization for {task_id}")

        updated = copy.deepcopy(old)
        updated["official_stream_position_1based"] = new["stream_position_1based"]
        updated["official_category"] = new.get("category", old.get("official_category"))
        updated["source_instance_count"] = new["source_instance_count"]
        updated["source_json_sha256"] = recreated_source_sha
        if historical_source_sha:
            updated["historical_windows_source_json_sha256"] = historical_source_sha
        updated["counts"] = copy.deepcopy(new["counts"])
        updated["instance_ids"] = copy.deepcopy(new_ids)
        updated["instance_id_sha256"] = {
            split: sha256_text(new_ids[split]) for split in ("train", "dev", "test")
        }
        updated["split_files"] = copy.deepcopy(new["split_files"])
        rebased_tasks.append(updated)
        id_checks.append(
            {
                "task_id": task_id,
                "counts": new["counts"],
                "instance_id_sha256": updated["instance_id_sha256"],
                "ordered_ids_match_frozen_windows_manifest": True,
                "source_json_sha256_match": source_bytes_match,
                "source_json_windows_crlf_normalization_match": windows_line_ending_match,
                "source_json_content_equivalent": source_bytes_match or windows_line_ending_match,
            }
        )
    rebased["tasks"] = rebased_tasks
    rebased["linux_migration_verification"] = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifest_task_ids": expected_task_ids,
        "frozen_manifest_sha256": None,
        "recreated_data_manifest_sha256": None,
        "checks": id_checks,
        "selection_analysis_note": (
            "Frozen task-selection values were carried forward from the tracked Windows manifest; "
            "the referenced raw Windows relation-analysis artifact was not present on this Linux host."
        ),
    }
    return rebased


def main() -> int:
    try:
        from dotenv import load_dotenv
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install requirements-linux.txt before verifying the Linux dataset") from exc

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/continual_baseline_full_linux.yaml")
    parser.add_argument("--frozen-manifest", type=Path, default=PROJECT_ROOT / "data/task_stream_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = resolve_runtime_path(args.config, project_root=PROJECT_ROOT)
    config = resolve_config_paths(yaml.safe_load(config_path.read_text(encoding="utf-8")), project_root=PROJECT_ROOT)
    recreated_path = Path(config["dataset"]["processed_dir"]) / "data_manifest.json"
    frozen_path = resolve_runtime_path(args.frozen_manifest, project_root=PROJECT_ROOT)
    output_path = resolve_runtime_path(
        args.output or config["dataset"].get("task_stream_manifest", "data/task_stream_manifest_linux.json"),
        project_root=PROJECT_ROOT,
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing migration manifest: {output_path}")

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    recreated = json.loads(recreated_path.read_text(encoding="utf-8"))
    rebased = verify_and_rebase_manifest(
        frozen,
        recreated,
        expected_task_ids=list(config["dataset"]["tasks"]),
        expected_source_commit=config["dataset"]["source_commit"],
        config_path=config_path,
        config_sha256=sha256_file(config_path),
    )
    verification = rebased["linux_migration_verification"]
    verification["frozen_manifest_sha256"] = sha256_file(frozen_path)
    verification["recreated_data_manifest_sha256"] = sha256_file(recreated_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rebased, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Linux task stream manifest: {output_path}")
    print("Frozen task order, ordered sample IDs, source commit, 500/50/50 counts, and leakage: PASS")
    for task in verification["checks"]:
        print(
            f"{task['task_id']}: {task['counts']} | train ID SHA-256={task['instance_id_sha256']['train']}"
            f" | source content match={task['source_json_content_equivalent']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
