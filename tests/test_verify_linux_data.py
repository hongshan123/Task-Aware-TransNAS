import hashlib
import pytest

from scripts.verify_linux_data import verify_and_rebase_manifest


def _task(task_id):
    ids = {"train": ["tr1"], "dev": ["dv1"], "test": ["te1"]}
    return {
        "task_id": task_id,
        "counts": {"train": 500, "dev": 50, "test": 50},
        "instance_ids": ids,
        "source_json_sha256": "source-sha",
        "split_files": {split: f"/linux/{split}.jsonl" for split in ids},
        "stream_position_1based": 1,
        "source_instance_count": 600,
        "source_sha256": "source-sha",
        "category": ["dialogue"],
    }


def test_rebased_manifest_keeps_frozen_order_and_uses_linux_paths(tmp_path):
    task_id = "task-a"
    frozen = {
        "tasks": [
            {
                "task_id": task_id,
                "counts": {"train": 500, "dev": 50, "test": 50},
                "instance_ids": {"train": ["tr1"], "dev": ["dv1"], "test": ["te1"]},
                "source_json_sha256": "source-sha",
                "split_files": {"train": "D:/old/train.jsonl"},
            }
        ]
    }
    recreated = {
        "source": {"commit": "abc123"},
        "leakage_check": {"passed": True},
        "tasks": [_task(task_id)],
    }

    result = verify_and_rebase_manifest(
        frozen,
        recreated,
        expected_task_ids=[task_id],
        expected_source_commit="abc123",
        config_path=tmp_path / "linux.yaml",
        config_sha256="config-sha",
    )

    assert result["tasks"][0]["split_files"]["train"] == "/linux/train.jsonl"
    assert result["tasks"][0]["instance_ids"] == frozen["tasks"][0]["instance_ids"]
    assert result["linux_migration_verification"]["checks"][0]["ordered_ids_match_frozen_windows_manifest"]


def test_rebased_manifest_rejects_changed_sample_ids(tmp_path):
    frozen = {
        "tasks": [
            {
                "task_id": "task-a",
                "counts": {"train": 500, "dev": 50, "test": 50},
                "instance_ids": {"train": ["old"], "dev": ["dv1"], "test": ["te1"]},
                "source_json_sha256": "source-sha",
            }
        ]
    }
    recreated = {"source": {"commit": "abc123"}, "leakage_check": {"passed": True}, "tasks": [_task("task-a")]}

    with pytest.raises(ValueError, match="sample IDs or order changed"):
        verify_and_rebase_manifest(
            frozen,
            recreated,
            expected_task_ids=["task-a"],
            expected_source_commit="abc123",
            config_path=tmp_path / "linux.yaml",
            config_sha256="config-sha",
        )


def test_source_checksum_accounts_for_windows_crlf_checkout(tmp_path):
    source_file = tmp_path / "task.json"
    source_file.write_bytes(b'{\n  "task": 1\n}\n')
    linux_sha = hashlib.sha256(source_file.read_bytes()).hexdigest()
    windows_sha = hashlib.sha256(source_file.read_bytes().replace(b"\n", b"\r\n")).hexdigest()
    old_task = _task("task-a")
    old_task["source_json_sha256"] = windows_sha
    new_task = _task("task-a")
    new_task["source_sha256"] = linux_sha
    new_task["source_json"] = str(source_file)
    frozen = {"tasks": [old_task]}
    recreated = {"source": {"commit": "abc123"}, "leakage_check": {"passed": True}, "tasks": [new_task]}

    result = verify_and_rebase_manifest(
        frozen,
        recreated,
        expected_task_ids=["task-a"],
        expected_source_commit="abc123",
        config_path=tmp_path / "linux.yaml",
        config_sha256="config-sha",
    )

    check = result["linux_migration_verification"]["checks"][0]
    assert check["source_json_sha256_match"] is False
    assert check["source_json_windows_crlf_normalization_match"] is True
    assert check["source_json_content_equivalent"] is True
