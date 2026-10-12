"""PaRSP adaptation, activation statistics, and real-update constraint auditing."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from phase7a_parsp import (
    STRICT_NULLSPACE_RCOND,
    load_official_parsp,
    map_query_channels_to_kv_channels,
    strict_project_update,
)
from run_formal_baseline import encode_training_record, safe_prompt


ACTIVATION_RESIDUAL_TOLERANCE = 1e-6


def _sample(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("Calibration sample count must be positive")
    if len(records) <= count:
        return list(records)
    return random.Random(seed).sample(records, count)


def _record_prompt_and_target(
    tokenizer: Any,
    record: dict[str, Any],
    model_cfg: dict[str, Any],
) -> tuple[list[int], int]:
    ids, labels, _ = encode_training_record(
        tokenizer,
        record,
        int(model_cfg["max_sequence_length"]),
        int(model_cfg["max_target_tokens"]),
    )
    answer_positions = [index for index, label in enumerate(labels) if label != -100]
    if not answer_positions:
        raise ValueError(f"No supervised target token for {record.get('instance_id')}")
    target_positions = list(answer_positions)
    if tokenizer.eos_token_id is not None and ids[target_positions[-1]] == tokenizer.eos_token_id:
        target_positions.pop()
    trailing_punctuation = {".", '"', " ", ")", "\n", ",", "]"}
    while target_positions:
        last_text = tokenizer.decode([ids[target_positions[-1]]], skip_special_tokens=True)
        if last_text in trailing_punctuation:
            target_positions.pop()
        else:
            break
    if not target_positions:
        target_positions = list(answer_positions)
    target_position = target_positions[-1]
    if target_position == 0:
        raise ValueError("The localization target has no preceding context")
    return ids[:target_position], int(ids[target_position])


def _last_token(captured: list[Any], layer_index: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(captured[layer_index], dtype=torch.float32, device=device)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    return value.reshape(1, -1)


@torch.no_grad()
def _log_target_probability(
    hidden: torch.Tensor,
    target_id: int,
    wrapped_model: nn.Module,
    final_var: torch.Tensor,
    get_bsvalues: Callable[..., torch.Tensor],
) -> torch.Tensor:
    logits = get_bsvalues(hidden, wrapped_model, final_var).float()
    return F.log_softmax(logits, dim=-1)[:, target_id]


def compute_functional_importance(
    model: nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    sample_count: int = 256,
    layer_ratio: float = 0.38,
    neuron_ratio: float = 0.40,
    candidate_batch_size: int = 1024,
) -> dict[str, Any]:
    """Compute PaRSP's log-probability gain scores using its Qwen capture utilities.

    Layer scores follow the upstream `get_top_attribute_layers` equations. Fine
    scores follow `get_top_attribute_neurons_ffn` and
    `get_top_attribute_neurons_attn`; this adapter returns the raw scores in
    addition to the selected indices, and tokenizes the established CITB SFT
    prompt/target exactly once through the existing encoder.
    """
    if not 0.0 < layer_ratio <= 1.0 or not 0.0 < neuron_ratio <= 1.0:
        raise ValueError("PaRSP layer_ratio and neuron_ratio must be in (0, 1]")
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")
    official = load_official_parsp()
    localizer_class = official["localization"].NeuronLocalizationQwen
    wrapper_class = official["wrapper"].QwenIntermediatesWrapper
    transfer_output = official["utils"].transfer_output_qwen
    get_bsvalues = official["utils"].get_bsvalues_qwen

    selected_records = _sample(records, sample_count, seed)
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // n_heads))
    intermediate_size = int(model.config.intermediate_size)
    top_layer_count = max(1, round(n_layers * layer_ratio))
    n_ffn = max(1, round(intermediate_size * neuron_ratio))
    n_attn_channels = max(1, round(n_heads * head_dim * neuron_ratio))
    was_training = model.training
    model.eval()

    layer_attn_scores = torch.zeros(n_layers, dtype=torch.float64)
    layer_ffn_scores = torch.zeros(n_layers, dtype=torch.float64)
    with wrapper_class(model) as wrapped:
        # Instantiating the official Qwen localizer also validates its view of
        # Qwen3's layer/head/intermediate dimensions before score collection.
        localizer_class(wrapped, tokenizer, str(device))
        for row in selected_records:
            prefix, target_id = _record_prompt_and_target(tokenizer, row, model_cfg)
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = wrapped(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                )
            captured = transfer_output(outputs[2], n_layers)
            layer_input, attn_output, residual_output, ffn_output, layer_output = captured[:5]
            last_state = _last_token(layer_output, n_layers - 1, device)
            final_var = (torch.var(last_state, dim=-1, unbiased=False) + 1e-5).sqrt()
            for layer_index in range(n_layers):
                inp = _last_token(layer_input, layer_index, device)
                attn = _last_token(attn_output, layer_index, device)
                res = _last_token(residual_output, layer_index, device)
                ffn = _last_token(ffn_output, layer_index, device)
                base_attn = _log_target_probability(inp, target_id, wrapped, final_var, get_bsvalues)
                plus_attn = _log_target_probability(inp + attn, target_id, wrapped, final_var, get_bsvalues)
                base_ffn = _log_target_probability(res, target_id, wrapped, final_var, get_bsvalues)
                plus_ffn = _log_target_probability(res + ffn, target_id, wrapped, final_var, get_bsvalues)
                layer_attn_scores[layer_index] += float((plus_attn - base_attn).item())
                layer_ffn_scores[layer_index] += float((plus_ffn - base_ffn).item())

        attention_layers = torch.argsort(layer_attn_scores, descending=True)[:top_layer_count].tolist()
        ffn_layers = torch.argsort(layer_ffn_scores, descending=True)[:top_layer_count].tolist()
        attention_scores = {
            layer_index: torch.zeros(n_heads * head_dim, dtype=torch.float64)
            for layer_index in attention_layers
        }
        ffn_scores = {
            layer_index: torch.zeros(intermediate_size, dtype=torch.float64)
            for layer_index in ffn_layers
        }
        hf_layers = model.model.layers
        for row in selected_records:
            prefix, target_id = _record_prompt_and_target(tokenizer, row, model_cfg)
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = wrapped(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                )
            captured = transfer_output(outputs[2], n_layers)
            layer_input, _, residual_output = captured[:3]
            attn_subvalues, ffn_coefficients = captured[5], captured[6]
            last_state = _last_token(captured[4], n_layers - 1, device)
            final_var = (torch.var(last_state, dim=-1, unbiased=False) + 1e-5).sqrt()

            for layer_index in ffn_layers:
                coefficients = _last_token(ffn_coefficients, layer_index, device).reshape(-1)
                down_weight = hf_layers[layer_index].mlp.down_proj.weight.detach().T.float()
                residual = _last_token(residual_output, layer_index, device)
                base_logp = _log_target_probability(residual, target_id, wrapped, final_var, get_bsvalues)
                for start in range(0, intermediate_size, candidate_batch_size):
                    end = min(start + candidate_batch_size, intermediate_size)
                    contribution = down_weight[start:end] * coefficients[start:end, None]
                    candidate = residual + contribution
                    candidate_logp = _log_target_probability(
                        candidate, target_id, wrapped, final_var, get_bsvalues
                    )
                    ffn_scores[layer_index][start:end] += (candidate_logp - base_logp).double().cpu()

            for layer_index in attention_layers:
                values = _last_token(attn_subvalues, layer_index, device).reshape(-1)
                o_weight = hf_layers[layer_index].self_attn.o_proj.weight.detach().T.float()
                layer_inp = _last_token(layer_input, layer_index, device)
                base_logp = _log_target_probability(layer_inp, target_id, wrapped, final_var, get_bsvalues)
                for start in range(0, n_heads * head_dim, candidate_batch_size):
                    end = min(start + candidate_batch_size, n_heads * head_dim)
                    contribution = o_weight[start:end] * values[start:end, None]
                    candidate = layer_inp + contribution
                    candidate_logp = _log_target_probability(
                        candidate, target_id, wrapped, final_var, get_bsvalues
                    )
                    attention_scores[layer_index][start:end] += (candidate_logp - base_logp).double().cpu()

    if was_training:
        model.train()
    attention_neurons = {
        layer: torch.argsort(scores, descending=True)[:n_attn_channels].tolist()
        for layer, scores in attention_scores.items()
    }
    ffn_neurons = {
        layer: torch.argsort(scores, descending=True)[:n_ffn].tolist()
        for layer, scores in ffn_scores.items()
    }
    return {
        "sample_ids": [str(row["instance_id"]) for row in selected_records],
        "layer_scores": {
            "attention": [float(value) for value in layer_attn_scores.tolist()],
            "ffn": [float(value) for value in layer_ffn_scores.tolist()],
        },
        "attention_scores": {layer: values.tolist() for layer, values in attention_scores.items()},
        "ffn_scores": {layer: values.tolist() for layer, values in ffn_scores.items()},
        "selected": {
            "attention_layers": attention_layers,
            "ffn_layers": ffn_layers,
            "attention_neurons": attention_neurons,
            "ffn_neurons": ffn_neurons,
        },
        "ratios": {"layer_ratio": layer_ratio, "neuron_ratio": neuron_ratio},
        "source": {
            "layer_score": "official localization.py: Qwen layer log-probability gain",
            "ffn_score": "official localization.py: isolated down_proj neuron contribution",
            "attention_score": "official localization.py: isolated o_proj input-channel contribution",
        },
    }


def write_importance_artifacts(path: Path, task_id: str, importance: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "selection.json").write_text(
        json.dumps({"task_id": task_id, **importance["selected"], "ratios": importance["ratios"]}, indent=2),
        encoding="utf-8",
    )
    with (path / "importance_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "component_type", "layer", "unit", "score"])
        writer.writeheader()
        for kind, scores in importance["layer_scores"].items():
            for layer, score in enumerate(scores):
                writer.writerow({"task_id": task_id, "component_type": "layer_" + kind, "layer": layer, "unit": "all", "score": score})
        for kind, mapping in (("attention_channel", importance["attention_scores"]), ("ffn_neuron", importance["ffn_scores"])):
            for layer_text, scores in mapping.items():
                layer = int(layer_text)
                for unit, score in enumerate(scores):
                    writer.writerow({"task_id": task_id, "component_type": kind, "layer": layer, "unit": unit, "score": score})
        for layer_text, scores in importance["attention_scores"].items():
            layer = int(layer_text)
            width = len(scores) // 16
            for head in range(16):
                head_score = float(sum(scores[head * width : (head + 1) * width]))
                writer.writerow({"task_id": task_id, "component_type": "attention_head", "layer": layer, "unit": head, "score": head_score})


@dataclass
class ModuleConstraint:
    name: str
    module: nn.Linear
    mask: torch.Tensor | None
    mask_axis: int
    projector_covariance: torch.Tensor
    activation_covariance_sum: torch.Tensor
    anchor_tokens: int
    official_null_rank: int
    official_null_ratio: float
    official_forced_ratio: bool
    projector_enabled: bool = False
    strict_projector: torch.Tensor | None = None
    strict_active_indices: torch.Tensor | None = None
    strict_null_rank: int | None = None
    audit_covariance_device: torch.Tensor | None = None


def _active_indices(mask: torch.Tensor | None, size: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.arange(size, device=device)
    return torch.nonzero(mask.to(device=device, dtype=torch.bool), as_tuple=False).reshape(-1)


def _compile_strict_projection(spec: ModuleConstraint, rcond: float = STRICT_NULLSPACE_RCOND) -> None:
    covariance = spec.activation_covariance_sum.to(dtype=torch.float64)
    device = spec.module.weight.device
    covariance = covariance.to(device=device)
    axis_size = spec.module.weight.shape[spec.mask_axis]
    active = _active_indices(spec.mask, axis_size, device)
    spec.strict_active_indices = active
    if active.numel() == 0:
        spec.strict_projector = torch.zeros((0, 0), device=device)
        return
    if spec.mask_axis == 1 and spec.mask is not None:
        restricted = covariance.index_select(0, active).index_select(1, active)
    else:
        restricted = covariance
    restricted = (restricted + restricted.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(restricted)
    max_eigenvalue = max(float(eigenvalues[-1].item()), 0.0)
    null = eigenvalues <= rcond * max_eigenvalue if max_eigenvalue else torch.ones_like(eigenvalues, dtype=torch.bool)
    if bool(null.any()):
        basis = eigenvectors[:, null]
        spec.strict_projector = basis @ basis.T
    else:
        spec.strict_projector = torch.zeros_like(restricted)
    spec.strict_null_rank = int(null.sum().item())


def _apply_compiled_strict_projection(delta: torch.Tensor, spec: ModuleConstraint) -> torch.Tensor:
    result = torch.zeros_like(delta)
    active = spec.strict_active_indices
    projector = spec.strict_projector
    if active is None or projector is None or active.numel() == 0:
        return result
    if spec.mask_axis == 0 and spec.mask is not None:
        result[active, :] = (delta[active, :].to(torch.float64) @ projector).to(delta.dtype)
    elif spec.mask_axis == 1 and spec.mask is not None:
        result[:, active] = (delta[:, active].to(torch.float64) @ projector).to(delta.dtype)
    elif spec.mask_axis == 0:
        result[active, :] = (delta[active, :].to(torch.float64) @ projector).to(delta.dtype)
    else:
        result[:, active] = (delta[:, active].to(torch.float64) @ projector).to(delta.dtype)
    return result


class ConstraintAuditor:
    """Measure gradients and true per-step linear-weight changes for active modules."""

    def __init__(self, specs: dict[str, ModuleConstraint], strict: bool = False):
        self.specs = specs
        self.strict = strict
        self.grad_stats: dict[str, dict[str, list[float]]] = {
            name: {"raw": [], "masked": [], "projected": []} for name in specs
        }
        self.step_rows: list[dict[str, Any]] = []
        self.weight_before: dict[str, torch.Tensor] = {}
        self.bias_before: dict[str, torch.Tensor] = {}
        self.grad_before: dict[str, torch.Tensor] = {}
        self.task_id = ""

    def reset_attempt(self, _step: int) -> None:
        self.grad_stats = {name: {"raw": [], "masked": [], "projected": []} for name in self.specs}

    def raw_hook(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self.grad_stats[name]["raw"].append(float(torch.linalg.vector_norm(grad.detach()).item()))
            return grad
        return hook

    def after_mask_hook(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self.grad_stats[name]["masked"].append(float(torch.linalg.vector_norm(grad.detach()).item()))
            return grad
        return hook

    def after_projection_hook(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self.grad_stats[name]["projected"].append(float(torch.linalg.vector_norm(grad.detach()).item()))
            return grad
        return hook

    def before_step(self, step: int, _model: nn.Module, _optimizer: torch.optim.Optimizer) -> None:
        self.weight_before = {}
        self.bias_before = {}
        self.grad_before = {}
        for name, spec in self.specs.items():
            self.weight_before[name] = spec.module.weight.detach().clone()
            if spec.module.bias is not None and spec.module.bias.requires_grad:
                self.bias_before[name] = spec.module.bias.detach().clone()
            if spec.module.weight.grad is not None:
                self.grad_before[name] = spec.module.weight.grad.detach().clone()

    def after_step(self, step: int, _model: nn.Module, _optimizer: torch.optim.Optimizer) -> None:
        for name, spec in self.specs.items():
            current = spec.module.weight.detach()
            before = self.weight_before[name]
            raw_delta = current - before
            correction_norm = 0.0
            strict_update_rolled_back = False
            if self.strict:
                corrected_delta = _apply_compiled_strict_projection(raw_delta, spec)
                correction_norm = float(torch.linalg.vector_norm(raw_delta - corrected_delta).item())
                with torch.no_grad():
                    current.copy_(before + corrected_delta)
            delta = current - before
            update_norm = float(torch.linalg.vector_norm(delta).item())
            mask_out_norm: float | None = None
            mask_out_ratio: float | None = None
            if spec.mask is not None:
                excluded = ~spec.mask.to(device=delta.device, dtype=torch.bool)
                masked_delta = delta[excluded, :] if spec.mask_axis == 0 else delta[:, excluded]
                mask_out_norm = float(torch.linalg.vector_norm(masked_delta).item())
                mask_out_ratio = mask_out_norm / max(update_norm, 1e-30)
            covariance = spec.audit_covariance_device
            if covariance is None:
                covariance = spec.activation_covariance_sum.to(device=delta.device, dtype=torch.float32)
            if update_norm == 0.0 or spec.anchor_tokens == 0:
                residual_norm = 0.0
            else:
                quadratic = (delta.float() * (delta.float() @ covariance)).sum().clamp_min(0.0)
                residual_norm = float(torch.sqrt(quadratic).item())
            normalization = update_norm * max(float(torch.trace(covariance).clamp_min(0.0).sqrt().item()), 1e-30)
            normalized_residual = (
                residual_norm / normalization if update_norm > 0.0 and spec.anchor_tokens > 0 else 0.0
            )
            if (
                self.strict
                and spec.anchor_tokens > 0
                and normalized_residual > ACTIVATION_RESIDUAL_TOLERANCE
            ):
                # Fail closed if finite-precision projection still leaves an
                # update outside the declared old-activation tolerance.
                with torch.no_grad():
                    current.copy_(before)
                delta = current - before
                update_norm = 0.0
                correction_norm = float(torch.linalg.vector_norm(raw_delta).item())
                mask_out_norm = 0.0 if spec.mask is not None else None
                mask_out_ratio = 0.0 if spec.mask is not None else None
                residual_norm = 0.0
                normalized_residual = 0.0
                strict_update_rolled_back = True
            grad = self.grad_before.get(name)
            if grad is not None and update_norm > 0.0:
                grad_norm = float(torch.linalg.vector_norm(grad).item())
                cosine = float(F.cosine_similarity(delta.reshape(1, -1), (-grad).reshape(1, -1), dim=-1).item())
                direction_error = float(
                    torch.linalg.vector_norm(
                        delta / max(update_norm, 1e-30) + grad / max(grad_norm, 1e-30)
                    ).item()
                )
            else:
                grad_norm, cosine, direction_error = 0.0, None, None
            bias_norm = None
            if name in self.bias_before and spec.module.bias is not None:
                bias_norm = float(torch.linalg.vector_norm(spec.module.bias.detach() - self.bias_before[name]).item())
            raw_values = self.grad_stats.get(name, {}).get("raw", [])
            masked_values = self.grad_stats.get(name, {}).get("masked", [])
            projected_values = self.grad_stats.get(name, {}).get("projected", [])
            row = {
                "task_id": self.task_id,
                "optimizer_step": step,
                "module": name,
                "mask_axis": spec.mask_axis if spec.mask is not None else None,
                "mask_active_fraction": float(spec.mask.float().mean().item()) if spec.mask is not None else 1.0,
                "mask_out_update_norm": mask_out_norm,
                "mask_out_update_ratio": mask_out_ratio,
                "weight_update_norm": update_norm,
                "history_activation_residual_norm": residual_norm,
                "normalized_history_activation_residual": normalized_residual,
                "anchor_tokens": spec.anchor_tokens,
                "projector_null_rank": spec.official_null_rank,
                "projector_null_ratio": spec.official_null_ratio,
                "projector_forced_min_ratio": spec.official_forced_ratio,
                "projector_enabled": spec.projector_enabled,
                "grad_pre_mask_mean_microbatch_norm": sum(raw_values) / max(1, len(raw_values)),
                "grad_post_mask_mean_microbatch_norm": sum(masked_values) / max(1, len(masked_values)),
                "grad_post_projector_mean_microbatch_norm": sum(projected_values) / max(1, len(projected_values)),
                "grad_accumulated_after_clip_norm": grad_norm,
                "update_direction_cosine_vs_negative_projected_gradient": cosine,
                "update_direction_l2_difference": direction_error,
                "bias_update_norm": bias_norm,
                "strict_correction_norm": correction_norm,
                "strict_null_rank": spec.strict_null_rank,
                "strict_update_rolled_back": strict_update_rolled_back,
            }
            self.step_rows.append(row)
        self.weight_before = {}
        self.bias_before = {}
        self.grad_before = {}


def _official_projector_stats(covariance: torch.Tensor, rcond: float, min_ratio: float) -> tuple[int, float, bool]:
    eigenvalues = torch.linalg.eigvalsh(covariance.to(device="cpu", dtype=torch.float32))
    maximum = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    if maximum <= 0:
        count = eigenvalues.numel()
        return count, 1.0, False
    count = int((eigenvalues <= rcond * maximum).sum().item())
    minimum = int(eigenvalues.numel() * min_ratio)
    forced = count < minimum
    if forced:
        count = minimum
    return count, count / max(1, eigenvalues.numel()), forced


def capture_input_covariances(
    model: nn.Module,
    tokenizer: Any,
    records_by_task: dict[str, list[dict[str, Any]]],
    module_names: list[str],
    model_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    sample_count_per_task: int = 256,
) -> dict[str, dict[str, Any]]:
    """Capture train-prompt input covariances without gradients or optimizer updates."""
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules or not isinstance(modules[name], nn.Linear)]
    if missing:
        raise KeyError(f"Selected PaRSP linear modules are unavailable: {missing[:5]}")
    tasks = list(records_by_task.items())
    if not tasks:
        return {
            name: {
                "projector_covariance": torch.zeros(modules[name].in_features, modules[name].in_features),
                "activation_covariance_sum": torch.zeros(modules[name].in_features, modules[name].in_features),
                "anchor_tokens": 0,
                "sample_ids": [],
            }
            for name in module_names
        }

    projector_sum = {name: torch.zeros(modules[name].in_features, modules[name].in_features, dtype=torch.float32) for name in module_names}
    activation_sum = {name: torch.zeros_like(projector_sum[name]) for name in module_names}
    task_cov_count = {name: 0 for name in module_names}
    total_tokens = {name: 0 for name in module_names}
    sample_ids: list[str] = []
    sample_ids_by_task: dict[str, list[str]] = {}
    was_training = model.training
    model.eval()
    previous_use_cache = bool(getattr(model.config, "use_cache", True))
    model.config.use_cache = False

    for task_index, (task_id, records) in enumerate(tasks):
        selected = _sample(records, sample_count_per_task, seed)
        sample_ids_by_task[task_id] = [str(row["instance_id"]) for row in selected]
        task_cov = {
            name: torch.zeros_like(projector_sum[name], device=device)
            for name in module_names
        }
        task_tokens = {name: 0 for name in module_names}
        handles = []
        for name in module_names:
            def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...], *, key: str = name) -> None:
                values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                if not torch.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite anchor activation in {key}")
                task_cov[key].add_(values.T @ values)
                task_tokens[key] += values.shape[0]
            handles.append(modules[name].register_forward_pre_hook(capture))
        try:
            with torch.inference_mode():
                for row in selected:
                    prompt, _, _ = safe_prompt(tokenizer, row, int(model_cfg["max_sequence_length"]))
                    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
                    sample_ids.append(str(row["instance_id"]))
        finally:
            for handle in handles:
                handle.remove()
        for name in module_names:
            count = task_tokens[name]
            if count == 0:
                continue
            # Keep the per-record covariance accumulation on the model device;
            # transfer each module's matrix to CPU only once after this task.
            task_cov_cpu = task_cov[name].cpu()
            task_mean = task_cov_cpu / count
            projector_sum[name].add_(task_mean)
            activation_sum[name].add_(task_cov_cpu)
            task_cov_count[name] += 1
            total_tokens[name] += count

    model.config.use_cache = previous_use_cache
    if was_training:
        model.train()
    output = {}
    for name in module_names:
        divisor = max(1, task_cov_count[name])
        output[name] = {
            "projector_covariance": projector_sum[name] / divisor,
            "activation_covariance_sum": activation_sum[name],
            "anchor_tokens": total_tokens[name],
            "sample_ids": sample_ids,
            "sample_ids_by_task": sample_ids_by_task,
            "anchor_task_count": task_cov_count[name],
        }
    return output


def _make_mask_specs(
    model: nn.Module, selected: dict[str, Any], apply_mask: bool
) -> tuple[list[str], dict[str, tuple[torch.Tensor | None, int]]]:
    modules = dict(model.named_modules())
    selected_attn = selected["attention_neurons"]
    selected_ffn = selected["ffn_neurons"]
    attn_layers = list(selected["attention_layers"])
    ffn_layers = list(selected["ffn_layers"])
    target_names: list[str] = []
    masks: dict[str, tuple[torch.Tensor | None, int]] = {}

    def add(name: str, axis: int, indices: list[int] | None = None) -> None:
        module = modules[name]
        target_names.append(name)
        mask = None
        if apply_mask and indices is not None:
            mask = torch.zeros(module.weight.shape[axis], dtype=torch.bool)
            valid = [int(index) for index in indices if 0 <= int(index) < module.weight.shape[axis]]
            if valid:
                mask[valid] = True
        masks[name] = (mask, axis)

    for layer_index in attn_layers:
        prefix = f"model.layers.{layer_index}.self_attn."
        q_channels = [int(value) for value in selected_attn.get(layer_index, selected_attn.get(str(layer_index), []))]
        q_proj = modules[prefix + "q_proj"]
        head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        q_heads = int(model.config.num_attention_heads)
        kv_heads = int(getattr(model.config, "num_key_value_heads", q_heads))
        kv_channels = map_query_channels_to_kv_channels(q_channels, q_heads, kv_heads, head_dim)
        add(prefix + "q_proj", 0)
        add(prefix + "k_proj", 0)
        add(prefix + "v_proj", 0, kv_channels)
        add(prefix + "o_proj", 1, q_channels)
        if q_proj.weight.shape[0] != q_heads * head_dim:
            raise ValueError("Qwen query projection width is inconsistent with attention heads and head_dim")
    for layer_index in ffn_layers:
        prefix = f"model.layers.{layer_index}.mlp."
        neurons = [int(value) for value in selected_ffn.get(layer_index, selected_ffn.get(str(layer_index), []))]
        add(prefix + "gate_proj", 0, neurons)
        add(prefix + "up_proj", 0, neurons)
        add(prefix + "down_proj", 1, neurons)
    return sorted(set(target_names)), masks


def _gradient_mask_hook(mask: torch.Tensor, axis: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def hook(grad: torch.Tensor) -> torch.Tensor:
        local = mask.to(device=grad.device, dtype=grad.dtype)
        if axis == 0:
            local = local.reshape(-1, 1)
        else:
            local = local.reshape(1, -1)
        return grad * local
    return hook


def prepare_parsp_constraints(
    model: nn.Module,
    condition: str,
    importance: dict[str, Any],
    history_covariances: dict[str, dict[str, Any]],
    run_cache_dir: Path,
    sequence_name: str,
    task_id: str,
    seed: int,
    auditor_strict: bool = False,
) -> tuple[ConstraintAuditor, list[Any], dict[str, Any]]:
    """Install original PaRSP layer freezing, adapted GQA masks, and official projectors."""
    if condition not in {"C1", "C2", "C3", "C4"}:
        raise ValueError(f"Unsupported PaRSP condition {condition}")
    official = load_official_parsp()
    selected = importance["selected"]
    attn_layers = list(selected["attention_layers"])
    ffn_layers = list(selected["ffn_layers"])
    official["masking"].layer_division_qwen(model, attn_layers, ffn_layers)
    use_mask = condition in {"C1", "C2", "C4"}
    use_projection = condition in {"C1", "C3", "C4"}
    target_names, raw_masks = _make_mask_specs(model, selected, use_mask)
    modules = dict(model.named_modules())
    specs: dict[str, ModuleConstraint] = {}
    has_history = bool(history_covariances)
    for name in target_names:
        module = modules[name]
        mask, axis = raw_masks[name]
        covariance_data = history_covariances.get(name)
        if covariance_data is None:
            zeros = torch.zeros(module.in_features, module.in_features, dtype=torch.float32)
            covariance_data = {
                "projector_covariance": zeros,
                "activation_covariance_sum": torch.zeros_like(zeros),
                "anchor_tokens": 0,
            }
        covariance = covariance_data["projector_covariance"].float().cpu()
        covariance_sum = covariance_data["activation_covariance_sum"].float().cpu()
        null_rank, null_ratio, forced = _official_projector_stats(covariance, 1e-6, 0.2)
        spec = ModuleConstraint(
            name=name,
            module=module,
            mask=mask,
            mask_axis=axis,
            projector_covariance=covariance,
            activation_covariance_sum=covariance_sum,
            anchor_tokens=int(covariance_data["anchor_tokens"]),
            official_null_rank=null_rank,
            official_null_ratio=null_ratio,
            official_forced_ratio=forced,
            projector_enabled=bool(use_projection and has_history),
            audit_covariance_device=covariance_sum.to(device=module.weight.device, dtype=torch.float32),
        )
        if auditor_strict:
            _compile_strict_projection(spec)
        specs[name] = spec

    handles: list[Any] = []
    auditor = ConstraintAuditor(specs, strict=auditor_strict)
    auditor.task_id = task_id
    for name, spec in specs.items():
        handles.append(spec.module.weight.register_hook(auditor.raw_hook(name)))
        if spec.mask is not None:
            handles.append(spec.module.weight.register_hook(_gradient_mask_hook(spec.mask, spec.mask_axis)))
        handles.append(spec.module.weight.register_hook(auditor.after_mask_hook(name)))
        if use_projection and has_history:
            spec.module.covariance_matrix = spec.projector_covariance

    projection_handles = []
    if use_projection and has_history:
        projection_handles = official["projection"].register_nullspace_hooks_qwen(
            model,
            attn_layers,
            ffn_layers,
            rcond=1e-6,
            min_ratio=0.2,
            dataset_name=f"{sequence_name}_{condition}_{task_id}",
            dataset_size=max(1, sum(int(item["anchor_task_count"]) for item in history_covariances.values()) // max(1, len(history_covariances)) * 256),
            seed=seed,
            cache_dir=str(run_cache_dir.resolve()),
        )
        if len(projection_handles) != len(specs):
            raise RuntimeError(
                f"Official PaRSP projector silently skipped modules: registered {len(projection_handles)} of {len(specs)}"
            )
        handles.extend(projection_handles)

    for name, spec in specs.items():
        handles.append(spec.module.weight.register_hook(auditor.after_projection_hook(name)))

    if auditor_strict and has_history:
        for spec in specs.values():
            if spec.module.bias is not None:
                spec.module.bias.requires_grad_(False)

    projection_metadata = {
        "official_hooks_registered": len(projection_handles),
        "projector_modules_expected": len(specs) if use_projection and has_history else 0,
        "history_task_count": len(history_covariances[next(iter(history_covariances))].get("sample_ids_by_task", {}))
        if history_covariances
        else 0,
        "mask_semantics": "official row/column gradient masks; attention V indices mapped from query channels to shared GQA KV channels",
        "projection_semantics": "unmodified official register_nullspace_hooks_qwen; mask hook is registered first",
    }
    return auditor, handles, projection_metadata


def audit_rows_to_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

