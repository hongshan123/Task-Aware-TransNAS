import json
import importlib
import os
from pathlib import Path

import pytest

from scripts.run_training_smoke import _training_tensors, load_train_records_from_manifest, validate_step_count
from scripts import run_training_smoke


def test_smoke_loader_reads_only_first_task_training_split(tmp_path):
    train_path = tmp_path / "train.jsonl"
    train_record = {
        "task_id": "task-a",
        "instance_id": "train-1",
        "split": "train",
        "definition": "Do the task",
        "input": "input",
        "targets": ["answer"],
    }
    train_path.write_text(json.dumps(train_record) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "leakage_check": {"passed": True},
                "tasks": [
                    {
                        "task_id": "task-a",
                        "split_files": {
                            "train": str(train_path),
                            "dev": str(tmp_path / "missing-dev.jsonl"),
                            "test": str(tmp_path / "missing-test.jsonl"),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    task_id, records = load_train_records_from_manifest(manifest_path)

    assert task_id == "task-a"
    assert [row["instance_id"] for row in records] == ["train-1"]


def test_training_smoke_rejects_more_than_ten_optimizer_steps():
    assert validate_step_count(10) == 10
    with pytest.raises(ValueError, match="at most 10"):
        validate_step_count(11)


def test_importing_smoke_helpers_does_not_load_project_env(monkeypatch):
    monkeypatch.delenv("TRANSNAS_DATA_ROOT", raising=False)

    importlib.reload(run_training_smoke)

    assert "TRANSNAS_DATA_ROOT" not in os.environ


def test_training_batch_masks_prompt_and_supervises_answer_tokens():
    torch = pytest.importorskip("torch")

    class StubTokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return {"input_ids": [4, 5] if kwargs.get("add_special_tokens") is False else [1, 2, 3]}

    batch = _training_tensors(
        StubTokenizer(),
        {"definition": "definition", "input": "input", "targets": ["answer"]},
        max_length=16,
        device=torch.device("cpu"),
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 5, 99]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5, 99]]
