"""Function-preserving, full-parameter Qwen3 architecture variants for Phase 4A."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


A2_LITE_ARCHITECTURES = {
    "A2-lite-16": (16, "sparse"),
    "A2-lite-16-control": (16, "dense_mix"),
    "A2-lite-32": (32, "sparse"),
    "A2-lite-32-control": (32, "dense_mix"),
}


class HeadInteractionAdapter(nn.Module):
    """Input-conditioned interaction over the concatenated attention-head outputs."""

    def __init__(self, head_dim: int, num_heads: int, router_dim: int = 8, mode: str = "cross") -> None:
        super().__init__()
        if mode not in {"cross", "self_gate"}:
            raise ValueError(f"Unsupported head interaction mode: {mode}")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.mode = mode
        self.query = nn.Linear(head_dim, router_dim, bias=False)
        self.key = nn.Linear(head_dim, router_dim, bias=False)
        self.scale = router_dim**-0.5
        # Exact zero makes the initial model function-identical to the base model.
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.last_branch_rms: torch.Tensor | None = None

    def forward(self, flattened_heads: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = flattened_heads.shape
        expected = self.num_heads * self.head_dim
        if width != expected:
            raise ValueError(f"Expected attention output width {expected}; received {width}")
        heads = flattened_heads.reshape(batch, sequence, self.num_heads, self.head_dim)
        query = self.query(heads)
        key = self.key(heads)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if self.mode == "cross":
            weights = scores.float().softmax(dim=-1).to(dtype=heads.dtype)
            delta = torch.matmul(weights, heads)
        else:
            # Capacity-matched control: the same router parameters gate each head
            # independently, without connecting information between heads.
            gates = scores.diagonal(dim1=-2, dim2=-1).sigmoid().unsqueeze(-1)
            delta = heads * gates
        branch = self.residual_scale * delta.reshape(batch, sequence, width)
        if self.training:
            self.last_branch_rms = branch.detach().float().square().mean().sqrt()
        return flattened_heads + branch


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class GroupedFFNRouter(nn.Module):
    """Top-k token routing across grouped SwiGLU FFN experts.

    The sparse path evaluates only experts selected for each token. ``dense_mix``
    is a capacity-matched control that evaluates every expert and mixes them with
    the same input-conditioned router.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        groups: int = 4,
        top_k: int = 2,
        mode: str = "sparse",
        expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"sparse", "dense_mix"}:
            raise ValueError(f"Unsupported grouped FFN mode: {mode}")
        if groups < 2:
            raise ValueError("groups must be at least two")
        if expert_intermediate_size is None and intermediate_size % groups:
            raise ValueError("intermediate_size must be divisible by groups")
        if not 1 <= top_k <= groups:
            raise ValueError("top_k must be in [1, groups]")
        self.hidden_size = hidden_size
        self.groups = groups
        self.top_k = top_k
        self.mode = mode
        expert_width = intermediate_size // groups if expert_intermediate_size is None else expert_intermediate_size
        if expert_width < 1:
            raise ValueError("Expert intermediate width must be positive")
        self.expert_intermediate_size = expert_width
        self.router = nn.Linear(hidden_size, groups, bias=True)
        self.experts = nn.ModuleList(SwiGLUExpert(hidden_size, expert_width) for _ in range(groups))
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.last_branch_rms: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        original_shape = hidden.shape
        flat = hidden.reshape(-1, self.hidden_size)
        router_logits = self.router(flat).float()
        if self.mode == "dense_mix":
            weights = router_logits.softmax(dim=-1).to(dtype=flat.dtype)
            mixed = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                mixed = mixed + expert(flat) * weights[:, expert_id : expert_id + 1]
        else:
            selected_logits, selected_ids = torch.topk(router_logits, self.top_k, dim=-1)
            selected_weights = selected_logits.softmax(dim=-1).to(dtype=flat.dtype)
            mixed = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                token_and_slot = torch.nonzero(selected_ids == expert_id, as_tuple=False)
                if token_and_slot.numel() == 0:
                    continue
                token_ids, slots = token_and_slot[:, 0], token_and_slot[:, 1]
                expert_output = expert(flat.index_select(0, token_ids))
                weighted_output = expert_output * selected_weights[token_ids, slots].unsqueeze(-1)
                mixed = mixed.index_add(0, token_ids, weighted_output)
        branch = (self.residual_scale * mixed).reshape(original_shape)
        if self.training:
            self.last_branch_rms = branch.detach().float().square().mean().sqrt()
        return branch


def build_a2_lite_modules(
    hidden_size: int,
    groups: int,
    num_layers: int,
    expert_intermediate_size: int,
    mode: str,
) -> nn.ModuleList:
    """Instantiate an A2-lite module stack to audit exact added parameters."""
    return nn.ModuleList(
        GroupedFFNRouter(
            hidden_size=hidden_size,
            intermediate_size=expert_intermediate_size * groups,
            groups=groups,
            top_k=min(2, groups),
            mode=mode,
            expert_intermediate_size=expert_intermediate_size,
        )
        for _ in range(num_layers)
    )


def _register_pre_projection_adapter(attention: nn.Module, adapter: nn.Module) -> None:
    if hasattr(attention, "ta_head_interaction"):
        raise ValueError("Head interaction adapter is already installed")
    attention.add_module("ta_head_interaction", adapter)

    def adapt_o_proj_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (adapter(args[0]), *args[1:])

    attention.o_proj.register_forward_pre_hook(adapt_o_proj_input)


def _register_ffn_adapter(layer: nn.Module, adapter: nn.Module) -> None:
    if hasattr(layer, "ta_grouped_ffn"):
        raise ValueError("Grouped FFN adapter is already installed")
    layer.add_module("ta_grouped_ffn", adapter)
    original_forward = layer.mlp.forward
    layer.mlp._ta_transnas_original_forward = original_forward

    def routed_forward(hidden: torch.Tensor) -> torch.Tensor:
        return original_forward(hidden) + adapter(hidden)

    # Keep the original MLP module and its state-dict keys intact. The patch is
    # reinstalled by apply_architecture after checkpoint reload.
    layer.mlp.forward = routed_forward


def apply_architecture(model: nn.Module, architecture: str) -> nn.Module:
    """Attach A0/A1/A2 or a parameter-matched control to a Qwen3-style model."""
    valid = {"A0", "A1", "A2", "A1-control", "A2-control", *A2_LITE_ARCHITECTURES}
    if architecture not in valid:
        raise ValueError(f"architecture must be one of {sorted(valid)}")
    if architecture == "A0":
        return model

    config = model.config
    hidden_size = int(config.hidden_size)
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    for layer in model.model.layers:
        if architecture in {"A1", "A1-control"}:
            attention = layer.self_attn
            num_heads = int(config.num_attention_heads)
            head_dim = int(getattr(attention, "head_dim", hidden_size // num_heads))
            projection_input_width = int(attention.o_proj.in_features)
            if num_heads * head_dim != projection_input_width:
                raise ValueError("Attention head outputs must concatenate to o_proj.in_features")
            adapter = HeadInteractionAdapter(
                head_dim,
                num_heads,
                router_dim=8,
                mode="cross" if architecture == "A1" else "self_gate",
            ).to(device=device, dtype=dtype)
            _register_pre_projection_adapter(attention, adapter)
        elif architecture in {"A2", "A2-control"}:
            adapter = GroupedFFNRouter(
                hidden_size,
                int(config.intermediate_size),
                groups=4,
                top_k=2,
                mode="sparse" if architecture == "A2" else "dense_mix",
            ).to(device=device, dtype=dtype)
            _register_ffn_adapter(layer, adapter)
        else:
            expert_width, mode = A2_LITE_ARCHITECTURES[architecture]
            adapter = GroupedFFNRouter(
                hidden_size,
                int(config.intermediate_size),
                groups=4,
                top_k=2,
                mode=mode,
                expert_intermediate_size=expert_width,
            ).to(device=device, dtype=dtype)
            _register_ffn_adapter(layer, adapter)
    return model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def estimate_forward_flops(config: Any, architecture: str, sequence_length: int = 512) -> dict[str, int]:
    """Approximate multiply/add FLOPs; ignores normalization and nonlinearities."""
    valid_architectures = {"A0", "A1", "A2", "A1-control", "A2-control", *A2_LITE_ARCHITECTURES}
    if architecture not in valid_architectures:
        raise ValueError(f"Unknown architecture: {architecture}")
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    intermediate = int(config.intermediate_size)
    vocab = int(config.vocab_size)
    # Linear layers are counted as two FLOPs per multiply-accumulate.
    attention_width = heads * head_dim
    kv_width = kv_heads * head_dim
    qkv_macs = hidden * (attention_width + 2 * kv_width)
    attention_linear_macs = qkv_macs + attention_width * hidden
    base_mlp_macs = 3 * hidden * intermediate
    attention_matrix_flops = 4 * sequence_length * hidden
    base_layer_flops = 2 * (attention_linear_macs + base_mlp_macs) + attention_matrix_flops
    added_layer_flops = 0
    if architecture in {"A1", "A1-control"}:
        router_dim = 8
        # Q/K projections are shared across heads in both A1 and its control.
        adapter_macs_per_token = 2 * heads * head_dim * router_dim
        if architecture == "A1":
            # Pairwise head scores and weighted head-value mixing.
            adapter_macs_per_token += heads * heads * (router_dim + head_dim)
        else:
            # Diagonal score and per-head scaling, with no cross-head messages.
            adapter_macs_per_token += heads * (router_dim + head_dim)
        added_layer_flops = 2 * adapter_macs_per_token
    if architecture in {"A2", "A2-control"} or architecture in A2_LITE_ARCHITECTURES:
        groups, top_k = 4, 2
        expert_width = A2_LITE_ARCHITECTURES[architecture][0] if architecture in A2_LITE_ARCHITECTURES else intermediate // groups
        router_macs = hidden * groups
        sparse = architecture == "A2" or (
            architecture in A2_LITE_ARCHITECTURES and A2_LITE_ARCHITECTURES[architecture][1] == "sparse"
        )
        expert_macs = (top_k if sparse else groups) * 3 * hidden * expert_width
        added_layer_flops = 2 * (router_macs + expert_macs)
    lm_head_flops = 2 * hidden * vocab
    per_token = layers * (base_layer_flops + added_layer_flops) + lm_head_flops
    sequence_total = per_token * sequence_length
    return {
        "sequence_length": sequence_length,
        "approx_flops_per_prefill_token_at_sequence_length": int(per_token),
        "approx_flops_per_sequence": int(sequence_total),
        "attention_quadratic_component_per_sequence": int(layers * 4 * sequence_length * sequence_length * hidden),
    }
