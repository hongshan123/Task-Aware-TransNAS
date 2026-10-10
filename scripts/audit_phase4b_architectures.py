"""Real-Qwen3 implementation, cache, padding, initialization, and gate audit."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from architecture_modules import apply_architecture, parameter_counts
from phase4_common import TASK_IDS, prepare_context, write_json
from run_formal_baseline import encode_training_record, load_model, safe_prompt


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def selected_weight_fingerprints(model: Any) -> dict[str, str]:
    keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.13.self_attn.q_proj.weight",
        "model.layers.27.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    state = model.state_dict()
    return {
        key: hashlib.sha256(state[key].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for key in keys
    }


def repeated_ids(ids: list[int], size: int) -> list[int]:
    if not ids:
        raise ValueError("Audit prompt cannot be empty")
    repeats = math.ceil(size / len(ids))
    return (ids * repeats)[:size]


def last_attended_logits(model: Any, ids: list[int], device: torch.device) -> torch.Tensor:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids=x, attention_mask=mask, use_cache=False)
    return output.logits[0, -1].float().cpu()


def padded_batch_logits(
    model: Any, tokenizer: Any, long_ids: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    short_ids = long_ids[: max(8, len(long_ids) // 2)]
    max_len = len(long_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((2, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    input_ids[0, : len(long_ids)] = torch.tensor(long_ids, dtype=torch.long, device=device)
    input_ids[1, : len(short_ids)] = torch.tensor(short_ids, dtype=torch.long, device=device)
    attention_mask[0, : len(long_ids)] = 1
    attention_mask[1, : len(short_ids)] = 1
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
    selected = torch.stack([logits[0, len(long_ids) - 1], logits[1, len(short_ids) - 1]]).cpu()
    return selected, input_ids, {"long_tokens": len(long_ids), "short_tokens": len(short_ids), "padding_tokens": max_len - len(short_ids)}


def incremental_decode_logits(model: Any, ids: list[int], device: torch.device) -> dict[str, Any]:
    prefix = torch.tensor([ids[:32]], dtype=torch.long, device=device)
    next_token = torch.tensor([[ids[32]]], dtype=torch.long, device=device)
    full_ids = torch.cat([prefix, next_token], dim=1)
    full_mask = torch.ones_like(full_ids)
    with torch.inference_mode():
        full = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False).logits[0, -1].float()
        prefill = model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
        step = model(
            input_ids=next_token,
            attention_mask=full_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        ).logits[0, -1].float()
    return {
        "full_logits": full.cpu(),
        "incremental_logits": step.cpu(),
        "max_full_vs_incremental_abs_difference": float((full - step).abs().max().item()),
        "relative_full_vs_incremental_l2_difference": float(
            ((full - step).norm() / full.norm().clamp_min(1e-12)).item()
        ),
        "full_argmax_token_id": int(full.argmax().item()),
        "incremental_argmax_token_id": int(step.argmax().item()),
        "full_vs_incremental_argmax_equal": int(full.argmax().item()) == int(step.argmax().item()),
    }


def generation_tokens(model: Any, tokenizer: Any, ids: list[int], device: torch.device) -> list[int]:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=8,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return generated[0, x.shape[1] :].cpu().tolist()


def functional_audit(
    architecture: str,
    base_path: Path,
    tokenizer: Any,
    prompt_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model = load_model(base_path, local_only=True, training=False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    fingerprints = selected_weight_fingerprints(model)
    test_lengths = [1, 17, 75, 160, 256]
    inputs = {str(length): repeated_ids(prompt_ids, length) for length in test_lengths}
    base_logits = {length: last_attended_logits(model, ids, device) for length, ids in inputs.items()}
    base_padded, padded_ids, padding_info = padded_batch_logits(model, tokenizer, inputs["160"], device)
    base_generation = generation_tokens(model, tokenizer, inputs["75"], device)
    base_incremental = incremental_decode_logits(model, inputs["75"], device)

    torch.manual_seed(9091)
    apply_architecture(model, architecture)
    model.eval()
    differences = {
        length: float((last_attended_logits(model, inputs[length], device) - reference).abs().max().item())
        for length, reference in base_logits.items()
    }
    adapted_padded, _, _ = padded_batch_logits(model, tokenizer, inputs["160"], device)
    padded_difference = float((adapted_padded - base_padded).abs().max().item())
    adapted_generation = generation_tokens(model, tokenizer, inputs["75"], device)
    adapted_incremental = incremental_decode_logits(model, inputs["75"], device)
    incremental_difference = float(
        (adapted_incremental["incremental_logits"] - base_incremental["full_logits"]).abs().max().item()
    )
    full_vs_incremental = adapted_incremental["max_full_vs_incremental_abs_difference"]
    result = {
        "architecture": architecture,
        "parameter_counts": parameter_counts(model),
        "base_state_key_count": len([key for key in model.state_dict() if "ta_head_interaction" not in key]),
        "selected_local_weight_fingerprints": fingerprints,
        "max_abs_logit_difference_by_input_length": differences,
        "padded_batch": {
            **padding_info,
            "max_abs_logit_difference": padded_difference,
        },
        "incremental_decode": {
            "prefix_tokens": 32,
            "incremental_tokens": 1,
            "base_full_vs_incremental_abs_difference": base_incremental[
                "max_full_vs_incremental_abs_difference"
            ],
            "base_full_vs_incremental_relative_l2_difference": base_incremental[
                "relative_full_vs_incremental_l2_difference"
            ],
            "base_full_vs_incremental_argmax_equal": base_incremental["full_vs_incremental_argmax_equal"],
            "adapted_full_vs_incremental_abs_difference": full_vs_incremental,
            "adapted_full_vs_incremental_relative_l2_difference": adapted_incremental[
                "relative_full_vs_incremental_l2_difference"
            ],
            "adapted_full_vs_incremental_argmax_equal": adapted_incremental["full_vs_incremental_argmax_equal"],
            "adapted_incremental_vs_base_incremental_abs_difference": float(
                (
                    adapted_incremental["incremental_logits"]
                    - base_incremental["incremental_logits"]
                ).abs().max().item()
            ),
            "adapted_incremental_vs_base_full_abs_difference": incremental_difference,
        },
        "greedy_generation_tokens_match_base": adapted_generation == base_generation,
        "base_generation_token_ids": base_generation,
        "adapted_generation_token_ids": adapted_generation,
        "passes_all_output_checks": (
            max(differences.values()) <= 1e-6
            and padded_difference <= 1e-6
            and float(
                (
                    adapted_incremental["incremental_logits"]
                    - base_incremental["incremental_logits"]
                ).abs().max().item()
            )
            <= 1e-6
            and base_incremental["full_vs_incremental_argmax_equal"]
            and adapted_incremental["full_vs_incremental_argmax_equal"]
            and base_incremental["max_full_vs_incremental_abs_difference"] <= 5e-5
            and adapted_incremental["max_full_vs_incremental_abs_difference"] <= 5e-5
            and adapted_generation == base_generation
        ),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def gate_gradient_audit(
    architecture: str,
    base_path: Path,
    tokenizer: Any,
    record: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = load_model(base_path, local_only=True, training=True)
    torch.manual_seed(9091)
    apply_architecture(model, architecture)
    adapters = [layer.self_attn.ta_head_interaction for layer in model.model.layers]
    ids, labels, _ = encode_training_record(
        tokenizer,
        record,
        int(config["model"]["max_sequence_length"]),
        int(config["model"]["max_target_tokens"]),
    )
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([labels], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, foreach=False)
    steps: list[dict[str, Any]] = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y, use_cache=False).loss
        loss.backward()
        gate_grad = [float(adapter.residual_scale.grad.detach().abs().item()) for adapter in adapters]
        query_key_grad = [
            float(
                torch.sqrt(
                    adapter.query.weight.grad.detach().float().square().sum()
                    + adapter.key.weight.grad.detach().float().square().sum()
                ).item()
            )
            for adapter in adapters
        ]
        branch_rms = [
            float(adapter.last_branch_rms.item()) if adapter.last_branch_rms is not None else 0.0
            for adapter in adapters
        ]
        optimizer.step()
        gate_values_after_update = [float(adapter.residual_scale.detach().item()) for adapter in adapters]
        steps.append(
            {
                "step": step,
                "loss": float(loss.detach().float().item()),
                "gate_values_after_update": gate_values_after_update,
                "gate_gradient_l2_after_clip": gate_grad,
                "query_key_gradient_l2": query_key_grad,
                "added_branch_rms": branch_rms,
                "gate_gradient_nonzero_layers": sum(value > 0 for value in gate_grad),
                "query_key_gradient_nonzero_layers": sum(value > 0 for value in query_key_grad),
                "branch_rms_nonzero_layers": sum(value > 0 for value in branch_rms),
            }
        )
    summary = {
        "architecture": architecture,
        "layer_count": len(adapters),
        "steps": steps,
        "first_step_gate_gradient_all_finite": all(math.isfinite(x) for x in steps[0]["gate_gradient_l2_after_clip"]),
        "first_step_gates_receive_gradient": steps[0]["gate_gradient_nonzero_layers"] == len(adapters),
        "second_step_router_receives_gradient": steps[1]["query_key_gradient_nonzero_layers"] == len(adapters),
        "second_step_added_branch_active": steps[1]["branch_rms_nonzero_layers"] == len(adapters),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    started = time.perf_counter()
    config, base_path, tokenizer, device, _, dev = prepare_context()
    output_root = Path(config["storage"]["root"]) / "results" / "phase4b" / "audit"
    output_root.mkdir(parents=True, exist_ok=True)
    model_file = base_path / "model.safetensors"
    source_hash = sha256_file(model_file)
    prompt, _, _ = safe_prompt(tokenizer, dev[TASK_IDS[0]][0], int(config["model"]["max_sequence_length"]))
    prompt = repeated_ids(prompt, max(256, len(prompt)))
    architectures = ["A0", "A1", "A1-control"]
    functional = [functional_audit(name, base_path, tokenizer, prompt, device) for name in architectures]
    reference_fingerprints = functional[0]["selected_local_weight_fingerprints"]
    same_weights = all(
        result["selected_local_weight_fingerprints"] == reference_fingerprints for result in functional[1:]
    )
    gradient = [
        gate_gradient_audit(name, base_path, tokenizer, dev[TASK_IDS[0]][0], config, device)
        for name in ("A1", "A1-control")
    ]
    result = {
        "source_model": str(base_path),
        "source_model_safetensors_sha256": source_hash,
        "local_files_only": True,
        "test_split_loaded": False,
        "base_weights_same_across_A0_A1_A1_control": same_weights,
        "functional_audits": functional,
        "gate_gradient_audits": gradient,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "architecture_audit.json", result)
    print(f"Phase 4B architecture audit complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
