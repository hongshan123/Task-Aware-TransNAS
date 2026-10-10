"""Small post-block residual branches for the Phase 6B pilot.

The input-routed arm dispatches each token through exactly one bottleneck
expert. Routing uses the token's hidden state and does not receive a task ID.
No reference model or replay example is used by these modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class BottleneckResidual(nn.Module):
    """Two-layer nonlinear residual branch with an exact zero-output start."""

    def __init__(self, hidden_size: int, bottleneck_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.down_proj = nn.Linear(bottleneck_size, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(hidden)))


class StaticResidualAdapter(nn.Module):
    """A single wide, always-active residual adapter (B1)."""

    def __init__(self, hidden_size: int, bottleneck_size: int) -> None:
        super().__init__()
        self.branch = BottleneckResidual(hidden_size, bottleneck_size)
        self.last_branch_rms: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.branch(hidden)
        self.last_branch_rms = delta.detach().float().square().mean().sqrt()
        return delta

    def reset_route_statistics(self) -> None:
        self.last_branch_rms = None

    def route_statistics(self) -> dict[str, Any]:
        rms = None if self.last_branch_rms is None else float(self.last_branch_rms.item())
        return {"router_kind": "static_single_branch", "branch_rms": rms}


class ConditionalResidualRouter(nn.Module):
    """K-way token router over shared, fixed-capacity bottleneck paths.

    ``input_train`` samples hard paths with Gumbel noise and uses a straight-
    through selected probability for gate gradients. ``input`` is deterministic
    hard routing for inference. Static, random, oracle, and off policies exist
    solely for the declared mechanism controls.
    """

    VALID_POLICIES = {"input_train", "input", "oracle", "static_gate", "random", "off"}

    def __init__(self, hidden_size: int, bottleneck_size: int, num_experts: int = 4) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least 2")
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.router = nn.Linear(hidden_size, num_experts, bias=True)
        self.experts = nn.ModuleList(
            BottleneckResidual(hidden_size, bottleneck_size) for _ in range(num_experts)
        )
        self.policy = "input_train"
        self.task_index: int | None = None
        self._route_counts: torch.Tensor | None = None
        self._forward_calls = 0
        self.last_route_ids: torch.Tensor | None = None
        self.last_branch_rms: torch.Tensor | None = None

    def set_condition(self, policy: str, task_index: int | None = None) -> None:
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"Unsupported residual routing policy: {policy}")
        if policy == "oracle" and task_index is None:
            raise ValueError("Oracle routing requires a task index")
        self.policy = policy
        self.task_index = task_index

    def reset_route_statistics(self) -> None:
        self._route_counts = None
        self._forward_calls = 0
        self.last_route_ids = None
        self.last_branch_rms = None

    def _routes(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = flat.shape[0]
        device = flat.device
        if self.policy == "oracle":
            choices = torch.full((count,), int(self.task_index) % self.num_experts, device=device, dtype=torch.long)
            return choices, torch.ones((count,), device=device, dtype=torch.float32)
        if self.policy == "random":
            choices = torch.randint(self.num_experts, (count,), device=device)
            return choices, torch.ones((count,), device=device, dtype=torch.float32)
        if self.policy == "static_gate":
            logits = self.router.bias.detach().float().expand(count, -1)
            choices = logits.argmax(dim=-1)
            return choices, torch.ones((count,), device=device, dtype=torch.float32)
        logits = self.router(flat).float()
        probabilities = logits.softmax(dim=-1)
        if self.policy == "input_train" and self.training:
            uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(uniform))
            choices = (logits + gumbel).argmax(dim=-1)
        else:
            choices = logits.argmax(dim=-1)
        selected_probability = probabilities.gather(1, choices[:, None]).squeeze(1)
        # Hard forward gate (=1) with a continuous gradient into the router.
        gate = 1.0 + selected_probability - selected_probability.detach()
        return choices, gate

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.policy == "off":
            self.last_route_ids = None
            self.last_branch_rms = hidden.new_zeros(())
            return torch.zeros_like(hidden)
        original_shape = hidden.shape
        flat = hidden.reshape(-1, original_shape[-1])
        choices, gates = self._routes(flat)
        delta = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_ids = torch.nonzero(choices == expert_id, as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            values = expert(flat.index_select(0, token_ids)).to(dtype=delta.dtype)
            values = values * gates.index_select(0, token_ids).to(dtype=delta.dtype).unsqueeze(-1)
            delta = delta.index_add(0, token_ids, values)
        counts = torch.bincount(choices.detach(), minlength=self.num_experts)
        self._route_counts = counts if self._route_counts is None else self._route_counts + counts
        self._forward_calls += 1
        self.last_route_ids = choices.detach()
        shaped = delta.reshape(original_shape)
        self.last_branch_rms = shaped.detach().float().square().mean().sqrt()
        return shaped

    def route_statistics(self) -> dict[str, Any]:
        if self._route_counts is None:
            counts = [0] * self.num_experts
        else:
            counts = [int(value) for value in self._route_counts.detach().cpu().tolist()]
        total = sum(counts)
        probs = [count / total for count in counts] if total else [0.0] * self.num_experts
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        rms = None if self.last_branch_rms is None else float(self.last_branch_rms.item())
        return {
            "router_kind": "input_hidden_state",
            "policy": self.policy,
            "task_id_used_by_router": self.policy == "oracle",
            "expert_token_counts": counts,
            "expert_token_shares": probs,
            "normalized_route_entropy": entropy / math.log(self.num_experts),
            "active_experts": sum(count > 0 for count in counts),
            "forward_calls": self._forward_calls,
            "branch_rms_last_forward": rms,
        }


@dataclass
class _InstalledPath:
    layer_index: int
    module: nn.Module
    handle: Any


class ResidualPathController:
    """Manage hooks and route policies without adding task-specific parameters."""

    def __init__(self, paths: list[_InstalledPath], architecture: str, layer_indices: list[int]) -> None:
        self.paths = paths
        self.architecture = architecture
        self.layer_indices = list(layer_indices)

    def set_condition(self, policy: str | None = None, task_index: int | None = None) -> None:
        if self.architecture == "B1-static":
            if policy not in (None, "static"):
                raise ValueError("B1-static has no route policy to override")
            return
        for path in self.paths:
            path.module.set_condition(policy or "input", task_index)

    def reset_statistics(self) -> None:
        for path in self.paths:
            path.module.reset_route_statistics()

    def route_statistics(self) -> dict[str, Any]:
        return {
            str(path.layer_index): path.module.route_statistics()
            for path in self.paths
        }

    def branch_gradient_statistics(self) -> dict[str, Any]:
        rows = {}
        for path in self.paths:
            module = path.module
            branch_modules = [module.branch] if isinstance(module, StaticResidualAdapter) else list(module.experts)
            per_expert = []
            for branch in branch_modules:
                squared = 0.0
                has_gradient = False
                for parameter in branch.parameters():
                    if parameter.grad is not None:
                        has_gradient = True
                        squared += float(parameter.grad.detach().float().square().sum().item())
                per_expert.append({"gradient_l2": math.sqrt(squared), "has_gradient": has_gradient})
            router = getattr(module, "router", None)
            router_squared = 0.0
            router_has_gradient = False
            if router is not None:
                for parameter in router.parameters():
                    if parameter.grad is not None:
                        router_has_gradient = True
                        router_squared += float(parameter.grad.detach().float().square().sum().item())
            rows[str(path.layer_index)] = {
                "branch_expert_gradients": per_expert,
                "router_gradient_l2": math.sqrt(router_squared),
                "router_has_gradient": router_has_gradient,
                "branch_rms": None if module.last_branch_rms is None else float(module.last_branch_rms.detach().item()),
            }
        return rows

    def close(self) -> None:
        for path in self.paths:
            path.handle.remove()
        self.paths.clear()


def install_residual_paths(
    model: nn.Module,
    architecture: str,
    layer_indices: list[int],
    hidden_size: int,
    bottleneck_size: int = 64,
    num_experts: int = 4,
) -> ResidualPathController | None:
    """Install B1/B2/B3 branches on post-block hidden states.

    B1 uses one width-(K*r) nonlinear adapter. B2 and B3 split that same branch
    width into K routed experts; their router adds only K*H+K parameters/layer.
    The tied embedding/LM head and all trunk weights remain untouched here.
    """
    if architecture == "B0":
        return None
    if architecture not in {"B1-static", "B2-input", "B3-oracle", "B2-random-location"}:
        raise ValueError(f"Unknown Phase 6B architecture: {architecture}")
    layers = model.model.layers
    normalized = sorted(set(int(index) for index in layer_indices))
    if not normalized or normalized[0] < 0 or normalized[-1] >= len(layers):
        raise ValueError(f"Invalid decoder layer selection: {normalized}")
    parameter = next(model.parameters())
    installed = []
    for layer_index in normalized:
        layer = layers[layer_index]
        if hasattr(layer, "ta_phase6b_residual"):
            raise ValueError(f"Phase 6B residual path already exists at layer {layer_index}")
        if architecture == "B1-static":
            module: nn.Module = StaticResidualAdapter(hidden_size, bottleneck_size * num_experts)
        else:
            module = ConditionalResidualRouter(hidden_size, bottleneck_size, num_experts)
        module = module.to(device=parameter.device, dtype=parameter.dtype)
        layer.add_module("ta_phase6b_residual", module)

        def patch_output(_layer: nn.Module, _inputs: tuple[Any, ...], output: Any, adapter=module):
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden + adapter(hidden), *output[1:])
            if isinstance(output, list):
                hidden = output[0]
                return [hidden + adapter(hidden), *output[1:]]
            return output + adapter(output)

        handle = layer.register_forward_hook(patch_output)
        installed.append(_InstalledPath(layer_index, module, handle))
    return ResidualPathController(installed, architecture, normalized)


def phase6b_added_parameter_count(
    architecture: str,
    hidden_size: int,
    layer_count: int,
    bottleneck_size: int = 64,
    num_experts: int = 4,
) -> int:
    if architecture == "B0":
        return 0
    branch_params = layer_count * hidden_size * bottleneck_size * num_experts * 2
    if architecture == "B1-static":
        return branch_params
    if architecture in {"B2-input", "B3-oracle", "B2-random-location"}:
        return branch_params + layer_count * (hidden_size * num_experts + num_experts)
    raise ValueError(f"Unknown Phase 6B architecture: {architecture}")
