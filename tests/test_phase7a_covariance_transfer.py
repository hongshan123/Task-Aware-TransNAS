from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import phase7a_engine  # noqa: E402


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 1, bias=False)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        values = input_ids.float()
        values = torch.stack((values, values + 1.0), dim=-1)
        return self.proj(values)


def _covariance(tokens: list[int]) -> torch.Tensor:
    values = torch.tensor([[value, value + 1.0] for value in tokens])
    return values.T @ values


def test_covariance_capture_copies_once_per_task_instead_of_once_per_record(monkeypatch):
    model = _ToyModel()
    monkeypatch.setattr(
        phase7a_engine,
        "safe_prompt",
        lambda _tokenizer, row, _budget: (row["tokens"], "", {}),
    )
    cpu_copy_shapes = []
    original_cpu = torch.Tensor.cpu

    def track_cpu_copy(tensor, *args, **kwargs):
        cpu_copy_shapes.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu_copy)
    records = {
        "old_a": [
            {"instance_id": "a1", "tokens": [1, 2]},
            {"instance_id": "a2", "tokens": [3]},
        ],
        "old_b": [
            {"instance_id": "b1", "tokens": [4]},
            {"instance_id": "b2", "tokens": [5, 6]},
        ],
    }

    observed = phase7a_engine.capture_input_covariances(
        model,
        tokenizer=object(),
        records_by_task=records,
        module_names=["proj"],
        model_cfg={"max_sequence_length": 8},
        device=torch.device("cpu"),
        seed=10,
        sample_count_per_task=2,
    )["proj"]

    # One covariance transfer per task, regardless of the number of records.
    assert cpu_copy_shapes == [(2, 2), (2, 2)]
    task_a = _covariance([1, 2, 3])
    task_b = _covariance([4, 5, 6])
    expected_projector = (task_a / 3 + task_b / 3) / 2
    assert torch.allclose(observed["projector_covariance"], expected_projector)
    assert torch.equal(observed["activation_covariance_sum"], task_a + task_b)
    assert observed["anchor_tokens"] == 6
