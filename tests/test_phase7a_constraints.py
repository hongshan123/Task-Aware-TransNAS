from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase7a_parsp import map_query_channels_to_kv_channels, strict_project_update  # noqa: E402
from phase7a_engine import ConstraintAuditor, ModuleConstraint, _apply_compiled_strict_projection, _compile_strict_projection, _log_target_probability, _make_mask_specs, prepare_parsp_constraints  # noqa: E402
from phase4_common import perform_optimizer_step  # noqa: E402


def test_strict_projection_keeps_column_mask_and_removes_old_activation_component():
    covariance = torch.ones(3, 3)
    mask = torch.tensor([True, False, True])
    delta = torch.tensor([[3.0, 7.0, 1.0], [-2.0, 5.0, 4.0]])

    projected = strict_project_update(delta, mask, covariance, mask_axis=1)

    assert torch.equal(projected[:, 1], torch.zeros(2))
    assert torch.allclose(projected @ covariance, torch.zeros_like(projected), atol=1e-6)
    assert torch.allclose(projected[0, 0] + projected[0, 2], torch.tensor(0.0), atol=1e-6)


def test_strict_projection_keeps_row_mask_and_projects_each_active_row():
    covariance = torch.ones(3, 3)
    mask = torch.tensor([False, True])
    delta = torch.tensor([[2.0, 1.0, 1.0], [1.0, 3.0, -2.0]])

    projected = strict_project_update(delta, mask, covariance, mask_axis=0)

    assert torch.equal(projected[0], torch.zeros(3))
    assert torch.allclose(projected[1] @ covariance, torch.zeros(3), atol=1e-6)


def test_strict_projection_returns_zero_when_no_active_coordinate_is_in_the_exact_nullspace():
    covariance = torch.eye(2)
    mask = torch.tensor([True, False])
    delta = torch.tensor([[4.0, 9.0]])

    projected = strict_project_update(delta, mask, covariance, mask_axis=1)

    assert torch.equal(projected, torch.zeros_like(delta))


def test_strict_projection_rejects_low_variance_but_nonzero_directions():
    covariance = torch.diag(torch.tensor([1.0, 1e-8, 0.0]))
    delta = torch.tensor([[0.0, 1.0, 1.0]])

    projected = strict_project_update(delta, None, covariance)
    quadratic = (projected * (projected @ covariance)).sum().clamp_min(0.0)
    residual = torch.sqrt(quadratic) / (
        torch.linalg.vector_norm(projected) * torch.sqrt(torch.trace(covariance))
    )

    assert residual <= 1e-6
    assert torch.equal(projected, torch.tensor([[0.0, 0.0, 1.0]]))


def test_compiled_strict_projection_uses_the_audited_null_residual_tolerance():
    covariance = torch.diag(torch.tensor([1.0, 1e-8, 0.0]))
    layer = nn.Linear(3, 1, bias=False)
    spec = ModuleConstraint(
        name="q_proj",
        module=layer,
        mask=None,
        mask_axis=1,
        projector_covariance=covariance,
        activation_covariance_sum=covariance,
        anchor_tokens=3,
        official_null_rank=0,
        official_null_ratio=0.0,
        official_forced_ratio=False,
    )

    _compile_strict_projection(spec)
    projected = _apply_compiled_strict_projection(torch.tensor([[0.0, 1.0, 1.0]]), spec)
    residual = torch.sqrt((projected * (projected @ covariance)).sum().clamp_min(0.0)) / (
        torch.linalg.vector_norm(projected) * torch.sqrt(torch.trace(covariance))
    )

    assert spec.strict_null_rank == 1
    assert residual <= 1e-6
    assert torch.equal(projected, torch.tensor([[0.0, 0.0, 1.0]]))


def test_strict_auditor_rolls_back_any_quantized_or_approximate_residual_above_tolerance():
    covariance = torch.diag(torch.tensor([1.0, 1e-8, 0.0]))
    layer = nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    spec = ModuleConstraint(
        name="q_proj",
        module=layer,
        mask=None,
        mask_axis=1,
        projector_covariance=covariance,
        activation_covariance_sum=covariance,
        anchor_tokens=3,
        official_null_rank=0,
        official_null_ratio=0.0,
        official_forced_ratio=False,
    )
    _compile_strict_projection(spec, rcond=1e-6)
    auditor = ConstraintAuditor({"q_proj": spec}, strict=True)
    auditor.task_id = "task"
    layer.weight.grad = torch.tensor([[0.0, -1.0, 0.0]])
    before = layer.weight.detach().clone()

    auditor.before_step(1, layer, optimizer)
    optimizer.step()
    auditor.after_step(1, layer, optimizer)

    row = auditor.step_rows[0]
    assert row["strict_update_rolled_back"] is True
    assert row["normalized_history_activation_residual"] == 0.0
    assert row["weight_update_norm"] == 0.0
    assert torch.equal(layer.weight.detach(), before)


def test_strict_auditor_records_zero_update_without_dividing_by_zero():
    covariance = torch.eye(3)
    layer = nn.Linear(3, 1, bias=False)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    spec = ModuleConstraint(
        name="q_proj",
        module=layer,
        mask=None,
        mask_axis=1,
        projector_covariance=covariance,
        activation_covariance_sum=covariance,
        anchor_tokens=3,
        official_null_rank=0,
        official_null_ratio=0.0,
        official_forced_ratio=False,
    )
    _compile_strict_projection(spec)
    auditor = ConstraintAuditor({"q_proj": spec}, strict=True)
    auditor.task_id = "task"
    layer.weight.grad = torch.ones_like(layer.weight)

    auditor.before_step(1, layer, optimizer)
    optimizer.step()
    auditor.after_step(1, layer, optimizer)

    row = auditor.step_rows[0]
    assert row["weight_update_norm"] == 0.0
    assert row["normalized_history_activation_residual"] == 0.0


def test_strict_post_step_correction_removes_adamw_decay_and_adaptive_violation():
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0]]))
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1, weight_decay=0.2)
    mask = torch.tensor([True, False])
    covariance = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    spec = ModuleConstraint(
        name="down_proj",
        module=layer,
        mask=mask,
        mask_axis=1,
        projector_covariance=covariance,
        activation_covariance_sum=covariance,
        anchor_tokens=2,
        official_null_rank=1,
        official_null_ratio=0.5,
        official_forced_ratio=False,
    )
    _compile_strict_projection(spec)
    before = layer.weight.detach().clone()
    layer.weight.grad = torch.tensor([[1.0, 0.0]])
    optimizer.step()
    actual_delta = layer.weight.detach() - before
    corrected = _apply_compiled_strict_projection(actual_delta, spec)
    with torch.no_grad():
        layer.weight.copy_(before + corrected)
    final_delta = layer.weight.detach() - before

    assert final_delta[0, 1].item() == 0.0
    assert torch.allclose(final_delta @ covariance, torch.zeros_like(final_delta), atol=1e-6)


def test_query_channels_map_to_shared_gqa_key_value_channels():
    # Four query heads share each KV head; channel 2 of query heads 4 and 7
    # both map to channel 2 of KV head 1.
    mapped = map_query_channels_to_kv_channels(
        [4 * 8 + 2, 7 * 8 + 2, 0], num_query_heads=8, num_kv_heads=2, head_dim=8
    )

    assert mapped == [0, 10]


class _GqaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 64, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(64, 16, bias=False)


class _GqaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _GqaAttention()
        self.mlp = nn.Module()


class _GqaCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_GqaLayer()])


class _GqaQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _GqaCore()
        self.config = SimpleNamespace(
            _name_or_path="qwen3-gqa-test",
            hidden_size=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
        )


def test_attention_mask_maps_query_channels_to_gqa_v_rows_and_o_columns():
    model = _GqaQwen()
    selected = {
        "attention_layers": [0],
        "ffn_layers": [],
        "attention_neurons": {0: [34, 58]},
        "ffn_neurons": {},
    }

    names, masks = _make_mask_specs(model, selected, apply_mask=True)

    assert names == [
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert masks["model.layers.0.self_attn.q_proj"][0] is None
    assert masks["model.layers.0.self_attn.k_proj"][0] is None
    assert torch.nonzero(masks["model.layers.0.self_attn.v_proj"][0]).reshape(-1).tolist() == [10]
    assert torch.nonzero(masks["model.layers.0.self_attn.o_proj"][0]).reshape(-1).tolist() == [34, 58]


def test_optimizer_callbacks_bracket_the_actual_step():
    layer = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
    layer.weight.grad = torch.tensor([[2.0]])
    events = []

    def before(step, model, opt):
        events.append(("before", step, float(model.weight.item()), opt is optimizer))

    def after(step, model, opt):
        events.append(("after", step, float(model.weight.item()), opt is optimizer))

    perform_optimizer_step(optimizer, layer, 1, before=before, after=after)

    assert [event[0] for event in events] == ["before", "after"]
    assert events[0][2] != events[1][2]
    assert events[1][1:] == (1, float(layer.weight.item()), True)


def test_functional_importance_candidate_logits_do_not_build_autograd_graph():
    head = nn.Linear(4, 7, bias=False)
    hidden = torch.randn(3, 4)

    def vocabulary_projection(values, _model, _final_var):
        return head(values)

    score = _log_target_probability(hidden, 2, object(), torch.tensor(1.0), vocabulary_projection)

    assert not score.requires_grad
    assert score.grad_fn is None


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3, 3, bias=False)
        self.k_proj = nn.Linear(3, 3, bias=False)
        self.v_proj = nn.Linear(3, 3, bias=False)
        self.o_proj = nn.Linear(3, 1, bias=False)


class _FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(3, 3, bias=False)
        self.up_proj = nn.Linear(3, 3, bias=False)
        self.down_proj = nn.Linear(3, 1, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()
        self.mlp = _FakeMlp()


class _FakeCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer()])


class _FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeCore()
        self.config = SimpleNamespace(
            _name_or_path="qwen3-unit-test",
            hidden_size=3,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=3,
            intermediate_size=3,
        )


def test_upstream_mask_then_projection_can_refill_masked_columns(tmp_path: Path):
    official_root = Path("/home/hongzhijun/projects/PaRSP-official")
    if not official_root.is_dir():
        pytest.skip("official PaRSP clone is not available")
    sys.path.insert(0, str(official_root))
    from parsp.methods.masking import neuron_division_qwen
    from parsp.methods.projection import register_nullspace_hooks_qwen

    model = _FakeQwen()
    target = model.model.layers[0].mlp.down_proj
    target.covariance_matrix = torch.ones(3, 3)
    neuron_division_qwen(
        model,
        top_attn_layer_idx=[],
        top_ffn_layer_idx=[0],
        top_ffn_neurons=[[0, 0]],
        top_attn_neurons=[],
    )
    handles = register_nullspace_hooks_qwen(
        model,
        top_attn_layer_idx=[],
        top_ffn_layer_idx=[0],
        rcond=1e-6,
        min_ratio=0.2,
        dataset_name="unit",
        dataset_size=1,
        seed=10,
        cache_dir=str(tmp_path),
    )
    assert handles

    target.weight.grad = None
    with torch.no_grad():
        target.weight[0].copy_(torch.tensor([1.0, 0.0, 0.0]))
    target.weight.sum().backward()

    assert target.weight.grad[0, 1:].norm().item() > 0.1
    for handle in handles:
        handle.remove()


def test_phase7a_adapter_registers_every_qwen_projection_when_history_exists(tmp_path: Path):
    model = _FakeQwen()
    selected = {
        "attention_layers": [0],
        "ffn_layers": [0],
        "attention_neurons": {0: [0]},
        "ffn_neurons": {0: [0]},
    }
    names, _ = _make_mask_specs(model, selected, apply_mask=False)
    history = {}
    for name in names:
        module = dict(model.named_modules())[name]
        covariance = torch.ones(module.in_features, module.in_features)
        history[name] = {
            "projector_covariance": covariance,
            "activation_covariance_sum": covariance,
            "anchor_tokens": 1,
            "anchor_task_count": 1,
            "sample_ids": ["old-train-1"],
            "sample_ids_by_task": {"old-task": ["old-train-1"]},
        }

    auditor, handles, metadata = prepare_parsp_constraints(
        model,
        "C1",
        {"selected": selected},
        history,
        tmp_path,
        "S1",
        "current-task",
        10,
    )
    target = model.model.layers[0].mlp.down_proj
    target.weight.sum().backward()

    assert metadata["official_hooks_registered"] == len(names) == 7
    assert target.weight.grad[0, 1:].norm().item() > 0.1
    for handle in handles:
        handle.remove()
