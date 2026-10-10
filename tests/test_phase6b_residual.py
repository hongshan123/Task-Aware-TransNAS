from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from scripts.phase6b_modules import (
    ConditionalResidualRouter,
    install_residual_paths,
    phase6b_added_parameter_count,
)


class TinyLayer(nn.Module):
    def forward(self, hidden):
        return (hidden * 1.5, "preserved-output")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer(), TinyLayer()])
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        for layer in self.model.layers:
            result = layer(hidden)
            hidden = result[0] if isinstance(result, tuple) else result
        return self.proj(hidden)


class Phase6BResidualTests(unittest.TestCase):
    def test_all_added_paths_start_as_exact_noops(self):
        for architecture in ("B1-static", "B2-input", "B3-oracle", "B2-random-location"):
            torch.manual_seed(13)
            model = TinyModel()
            values = torch.randn(2, 5, 8)
            before = model(values)
            controller = install_residual_paths(
                model, architecture, [1, 2], hidden_size=8, bottleneck_size=2, num_experts=4
            )
            if architecture == "B3-oracle":
                controller.set_condition("oracle", task_index=2)
            after = model(values)
            torch.testing.assert_close(after, before, rtol=0, atol=0)
            controller.close()

    def test_routed_parameter_budget_is_matched_and_below_one_percent(self):
        base_parameters = 596_049_920
        b1 = phase6b_added_parameter_count("B1-static", 1024, 2, 64, 4)
        b2 = phase6b_added_parameter_count("B2-input", 1024, 2, 64, 4)
        b3 = phase6b_added_parameter_count("B3-oracle", 1024, 2, 64, 4)
        self.assertEqual(b1, 1_048_576)
        self.assertEqual(b2, b3)
        self.assertLess((b2 - b1) / b1, 0.01)
        self.assertLess(b2 / base_parameters, 0.01)

    def test_input_router_selects_different_experts_for_different_hidden_states(self):
        torch.manual_seed(17)
        model = TinyModel()
        controller = install_residual_paths(model, "B2-input", [1], 8, bottleneck_size=2, num_experts=4)
        module = model.model.layers[1].ta_phase6b_residual
        self.assertIsInstance(module, ConditionalResidualRouter)
        with torch.no_grad():
            module.router.weight.zero_()
            module.router.bias.zero_()
            module.router.weight[0, 0] = 1.0
            module.router.weight[1, 0] = -1.0
        module.set_condition("input")
        hidden = torch.zeros(1, 2, 8)
        hidden[0, 0, 0] = -3.0
        hidden[0, 1, 0] = 3.0
        model(hidden)
        self.assertEqual(module.last_route_ids.tolist(), [1, 0])
        self.assertIsNone(module.task_index)
        self.assertFalse(module.route_statistics()["task_id_used_by_router"])
        controller.close()

    def test_oracle_and_control_modes_have_explicit_routing_semantics(self):
        module = ConditionalResidualRouter(hidden_size=8, bottleneck_size=2, num_experts=4)
        hidden = torch.randn(2, 3, 8)
        module.set_condition("oracle", task_index=3)
        module(hidden)
        self.assertEqual(module.last_route_ids.unique().tolist(), [3])
        self.assertTrue(module.route_statistics()["task_id_used_by_router"])
        module.set_condition("static_gate")
        module(hidden)
        self.assertEqual(module.last_route_ids.unique().numel(), 1)
        module.set_condition("off")
        self.assertEqual(float(module(hidden).abs().max()), 0.0)

    def test_routed_branch_and_router_receive_gradients(self):
        torch.manual_seed(19)
        model = TinyModel()
        controller = install_residual_paths(model, "B2-input", [1], 8, bottleneck_size=2, num_experts=4)
        adapter = model.model.layers[1].ta_phase6b_residual
        controller.set_condition("input_train")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        values = torch.randn(2, 32, 8)
        optimizer.zero_grad(set_to_none=True)
        model(values).square().mean().backward()
        self.assertTrue(any(expert.down_proj.weight.grad is not None for expert in adapter.experts))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(values * 1.3).square().mean().backward()
        self.assertIsNotNone(adapter.router.weight.grad)
        self.assertGreater(float(adapter.router.weight.grad.norm()), 0.0)
        gradients = controller.branch_gradient_statistics()
        self.assertTrue(any(item["has_gradient"] for item in gradients["1"]["branch_expert_gradients"]))
        controller.close()


if __name__ == "__main__":
    unittest.main()
