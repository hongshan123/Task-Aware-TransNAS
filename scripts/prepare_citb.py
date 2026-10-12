"""Prepare a deterministic, leakage-checked subset of the official CITB stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .runtime_paths import resolve_config_paths, resolve_runtime_path
except ImportError:  # Running this file directly from scripts/.
    from runtime_paths import resolve_config_paths, resolve_runtime_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "continual_baseline.yaml")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config_path = resolve_runtime_path(args.config, project_root=PROJECT_ROOT)
    config = resolve_config_paths(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), project_root=PROJECT_ROOT
    )
    dataset_cfg = config["dataset"]
    source_repo = resolve_runtime_path(dataset_cfg["source_repo"], project_root=PROJECT_ROOT)
    source_data = resolve_runtime_path(dataset_cfg["source_data"], project_root=PROJECT_ROOT)
    output_root = resolve_runtime_path(dataset_cfg["processed_dir"], project_root=PROJECT_ROOT)
    seed = int(config["seed"])
    task_ids = list(dataset_cfg["tasks"])
    max_train = int(dataset_cfg["train_per_task"])
    eval_n = int(dataset_cfg["test_per_task"])
    dev_n = int(dataset_cfg["dev_per_task"])

    repo_commit = subprocess.check_output(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = dataset_cfg.get("source_commit")
    if expected_commit and repo_commit != expected_commit:
        raise ValueError(f"CITB source commit mismatch: expected {expected_commit}, got {repo_commit}")
    order_path = source_data / dataset_cfg["stream_order_file"]
    official_split_path = source_data / dataset_cfg["split_file"]
    order = [line.strip() for line in order_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    listed_tasks = [line.strip() for line in official_split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(order) != 19 or len(set(order)) != len(order):
        raise ValueError(f"Expected the official 19-task order1 stream; got {len(order)} entries")
    if any(task not in order or task not in listed_tasks for task in task_ids):
        raise ValueError("Configured tasks must be members of both official stream files")

    rng = random.Random(seed)
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/hyintell/CITB",
            "commit": repo_commit,
            "stream_order_file": str(order_path),
            "official_task_split_file": str(official_split_path),
            "stream_order": "order1",
            "split_rule": "Per official CITB code: first 50 examples test; next 50 dev; shuffle the remainder with seed 10 and take up to 500 for training.",
            "seed_note": "The upstream prepare_data.py declares seed=10 but does not seed Python random before its split helper. This preparation applies random.Random(10) explicitly for a deterministic training subset.",
        },
        "seed": seed,
        "caps": {"train": max_train, "dev": dev_n, "test": eval_n},
        "leakage_check": {"passed": True, "scope": "instance IDs are disjoint within every task across train/dev/test"},
        "tasks": [],
    }

    for task_id in task_ids:
        source_json = source_data / "tasks" / f"{task_id}.json"
        task = json.loads(source_json.read_text(encoding="utf-8"))
        instances = task["Instances"]
        if len(instances) <= eval_n + dev_n:
            raise ValueError(f"Task {task_id} has too few examples for the requested held-out splits")
        test_instances = instances[:eval_n]
        dev_instances = instances[eval_n : eval_n + dev_n]
        remaining = list(instances[eval_n + dev_n :])
        rng.shuffle(remaining)
        train_instances = remaining[:max_train]
        definition = text_value(task.get("Definition", []))

        def convert(instance: dict[str, Any], source_index: int, split: str) -> dict[str, Any]:
            outputs = instance.get("output", [])
            if isinstance(outputs, str):
                outputs = [outputs]
            return {
                "task_id": task_id,
                "category": task.get("Categories", []),
                "instance_id": instance["id"],
                "source_index": source_index,
                "split": split,
                "definition": definition,
                "input": text_value(instance.get("input", "")),
                "targets": [text_value(output) for output in outputs],
            }

        source_index_by_id = {instance["id"]: i for i, instance in enumerate(instances)}
        splits = {
            "train": [convert(row, source_index_by_id[row["id"]], "train") for row in train_instances],
            "dev": [convert(row, i, "dev") for i, row in enumerate(dev_instances, start=eval_n)],
            "test": [convert(row, i, "test") for i, row in enumerate(test_instances)],
        }
        id_sets = {name: {row["instance_id"] for row in rows} for name, rows in splits.items()}
        if id_sets["train"] & id_sets["dev"] or id_sets["train"] & id_sets["test"] or id_sets["dev"] & id_sets["test"]:
            raise AssertionError(f"Data leakage detected in {task_id}")

        task_dir = output_root / task_id
        for split_name, records in splits.items():
            jsonl_write(task_dir / f"{split_name}.jsonl", records)

        manifest["tasks"].append(
            {
                "task_id": task_id,
                "stream_position_1based": order.index(task_id) + 1,
                "category": task.get("Categories", []),
                "source_instance_count": len(instances),
                "counts": {name: len(rows) for name, rows in splits.items()},
                "source_json": str(source_json),
                "source_sha256": sha256(source_json),
                "instance_ids": {name: [row["instance_id"] for row in rows] for name, rows in splits.items()},
                "split_files": {name: str(task_dir / f"{name}.jsonl") for name in splits},
            }
        )
        print(f"{task_id}: train={len(train_instances)} dev={len(dev_instances)} test={len(test_instances)}")

    manifest_path = output_root / "data_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    project_manifest = resolve_runtime_path(
        dataset_cfg.get("manifest_project_path", "data_manifest.json"), project_root=PROJECT_ROOT
    )
    root_manifest = resolve_runtime_path(
        dataset_cfg.get("manifest_report_path", "reports/data_manifest.json"), project_root=PROJECT_ROOT
    )
    project_manifest.parent.mkdir(parents=True, exist_ok=True)
    serialized_manifest = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    project_manifest.write_text(serialized_manifest, encoding="utf-8")
    root_manifest.write_text(serialized_manifest, encoding="utf-8")
    print(f"manifest: {manifest_path}")
    print(f"tracked manifest: {project_manifest}")
    print(f"root manifest: {root_manifest}")
    print("leakage check: PASS")


if __name__ == "__main__":
    main()
