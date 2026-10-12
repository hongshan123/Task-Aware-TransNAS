from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase7a_parsp import load_phase7a_splits  # noqa: E402


def test_phase7a_split_loader_reads_only_train_and_dev(tmp_path: Path):
    task_root = tmp_path / "task-a"
    task_root.mkdir()
    train_path = task_root / "train.jsonl"
    dev_path = task_root / "dev.jsonl"
    train_path.write_text('{"split":"train","instance_id":"tr1"}\n', encoding="utf-8")
    dev_path.write_text('{"split":"dev","instance_id":"dv1"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "leakage_check": {"passed": True},
                "tasks": [
                    {
                        "task_id": "task-a",
                        "counts": {"train": 1, "dev": 1},
                        "split_files": {
                            "train": str(train_path),
                            "dev": str(dev_path),
                            "test": str(tmp_path / "must-never-be-opened.jsonl"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    train, dev = load_phase7a_splits(manifest_path, ["task-a"])

    assert train["task-a"][0]["instance_id"] == "tr1"
    assert dev["task-a"][0]["instance_id"] == "dv1"
