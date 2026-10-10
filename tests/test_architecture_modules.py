from types import SimpleNamespace
import unittest

import torch
from torch import nn

from scripts.architecture_modules import (
    GroupedFFNRouter,
    apply_architecture,
    build_a2_lite_modules,
    estimate_forward_flops,
    parameter_counts,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.head_dim = 2


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=32,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer.self_attn.o_proj(x)
            x = x + layer.mlp(x)
        return x


def _adapter(model, name):
    return next(module for module_name, module in model.named_modules() if module_name.endswith(name))


def _set_scale(adapter, value=1.0):
    with torch.no_grad():
        adapter.residual_scale.fill_(value)


class ArchitectureModuleTests(unittest.TestCase):
    def test_a1_is_function_preserving_and_capacity_matched(self):
        torch.manual_seed(7)
        x = torch.randn(2, 5, 8)
        cross = TinyModel()
        control = TinyModel()
        control.load_state_dict(cross.state_dict())
        reference = cross(x)
        apply_architecture(cross, "A1")
        apply_architecture(control, "A1-control")
        torch.testing.assert_close(cross(x), reference, rtol=0, atol=0)
        torch.testing.assert_close(control(x), reference, rtol=0, atol=0)
        self.assertEqual(parameter_counts(cross)["total_parameters"] - parameter_counts(TinyModel())["total_parameters"], 33)
        self.assertEqual(parameter_counts(cross)["total_parameters"], parameter_counts(control)["total_parameters"])
        _set_scale(_adapter(cross, "ta_head_interaction"))
        _set_scale(_adapter(control, "ta_head_interaction"))
        self.assertFalse(torch.allclose(cross(x), control(x)))

    def test_a2_sparse_ffn_is_function_preserving_and_control_matches_parameters(self):
        torch.manual_seed(11)
        x = torch.randn(2, 4, 8)
        sparse = TinyModel()
        control = TinyModel()
        control.load_state_dict(sparse.state_dict())
        reference = sparse(x)
        base_keys = set(sparse.state_dict())
        apply_architecture(sparse, "A2")
        apply_architecture(control, "A2-control")
        torch.testing.assert_close(sparse(x), reference, rtol=0, atol=0)
        torch.testing.assert_close(control(x), reference, rtol=0, atol=0)
        self.assertEqual(parameter_counts(sparse)["total_parameters"], parameter_counts(control)["total_parameters"])
        self.assertEqual(set(key for key in sparse.state_dict() if ".ta_grouped_ffn." not in key), base_keys)
        router = _adapter(sparse, "ta_grouped_ffn")
        _set_scale(router)
        out = sparse(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertIsNotNone(router.router.weight.grad)
        self.assertTrue(any(expert.gate_proj.weight.grad is not None for expert in router.experts))

    def test_grouped_router_output_and_flop_estimates(self):
        module = GroupedFFNRouter(8, 16, groups=4, top_k=2)
        x = torch.randn(3, 2, 8, requires_grad=True)
        y = module(x)
        self.assertEqual(y.shape, x.shape)
        y.sum().backward()
        cfg = SimpleNamespace(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocab_size=151936,
        )
        a0 = estimate_forward_flops(cfg, "A0")
        a1 = estimate_forward_flops(cfg, "A1")
        a1_control = estimate_forward_flops(cfg, "A1-control")
        a2 = estimate_forward_flops(cfg, "A2")
        self.assertGreater(a2["approx_flops_per_sequence"], a0["approx_flops_per_sequence"])
        self.assertGreater(a1["approx_flops_per_sequence"], a1_control["approx_flops_per_sequence"])

    def test_zero_gate_gradient_opens_cross_head_branch(self):
        torch.manual_seed(31)
        model = TinyModel()
        apply_architecture(model, "A1")
        adapter = _adapter(model, "ta_head_interaction")
        x = torch.randn(2, 5, 8)
        model(x).square().mean().backward()
        self.assertEqual(adapter.residual_scale.item(), 0.0)
        self.assertGreater(float(adapter.residual_scale.grad.abs()), 0.0)
        self.assertEqual(float(adapter.query.weight.grad.norm()), 0.0)
        self.assertEqual(float(adapter.last_branch_rms), 0.0)

        with torch.no_grad():
            adapter.residual_scale.add_(-1e-2 * adapter.residual_scale.grad)
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        self.assertGreater(float(adapter.query.weight.grad.norm()), 0.0)
        self.assertGreater(float(adapter.last_branch_rms), 0.0)

    def test_a2_lite_32_meets_parameter_target_and_dense_control_matches(self):
        sparse = build_a2_lite_modules(1024, groups=4, num_layers=28, expert_intermediate_size=32, mode="sparse")
        dense = build_a2_lite_modules(1024, groups=4, num_layers=28, expert_intermediate_size=32, mode="dense_mix")
        sparse_params = sum(parameter.numel() for parameter in sparse.parameters())
        dense_params = sum(parameter.numel() for parameter in dense.parameters())
        self.assertEqual(sparse_params, 11_124_876)
        self.assertEqual(sparse_params, dense_params)
        self.assertGreaterEqual(100 * sparse_params / 596_049_920, 1.0)
        self.assertLessEqual(100 * sparse_params / 596_049_920, 2.0)

        torch.manual_seed(47)
        model = TinyModel()
        x = torch.randn(2, 4, 8)
        reference = model(x)
        apply_architecture(model, "A2-lite-16")
        torch.testing.assert_close(model(x), reference, rtol=0, atol=0)
        adapter = _adapter(model, "ta_grouped_ffn")
        model(x).square().mean().backward()
        self.assertGreater(float(adapter.residual_scale.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
