"""Run a matched full-parameter A0/A1/A2 continual-learning smoke pilot."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoConfig

from architecture_modules import apply_architecture, estimate_forward_flops, parameter_counts
from phase4_common import (
    TASK_IDS,
    evaluate_dev_task,
    human_vram,
    prepare_context,
    safe_task_seed,
    train_task_recorded,
    write_json,
)
from run_formal_baseline import load_model, safe_prompt, save_checkpoint


DEV_ROWS_PER_TASK = 10
PILOT_STEPS_PER_TASK = 50
ARCHITECTURES = ("A0", "A1", "A2")


def _last_position_logits(model: Any, input_ids: list[int], device: torch.device) -> torch.Tensor:
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()


def _initialization_audit(model: Any, architecture: str, tokenizer: Any, dev: dict[str, list[dict[str, Any]]], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    sample = dev[TASK_IDS[0]][0]
    prompt, _, prompt_info = safe_prompt(tokenizer, sample, int(config["model"]["max_sequence_length"]))
    model.eval()
    model.config.use_cache = False
    before = _last_position_logits(model, prompt, device)
    torch.manual_seed(int(config["seed"]))
    apply_architecture(model, architecture)
    model.eval()
    after = _last_position_logits(model, prompt, device)
    difference = after - before
    relative = difference.norm() / before.norm().clamp_min(1e-12)
    return {
        "sample_task": TASK_IDS[0],
        "sample_instance_id": sample["instance_id"],
        "prompt_tokens": len(prompt),
        "input_tail_truncated": bool(prompt_info["input_tail_truncated"]),
        "max_absolute_logit_difference": float(difference.abs().max().item()),
        "relative_l2_logit_difference": float(relative.item()),
        "function_preserving_tolerance": 1e-7,
        "passes_function_preserving_check": float(difference.abs().max().item()) <= 1e-7,
    }


def evaluate_matrix(
    model: Any,
    tokenizer: Any,
    architecture: str,
    stage_name: str,
    config: dict[str, Any],
    dev: dict[str, list[dict[str, Any]]],
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for task_id in TASK_IDS:
        task_path = output_root / architecture / "predictions" / stage_name / f"{task_id}.jsonl"
        metric, rows = evaluate_dev_task(
            model,
            tokenizer,
            dev[task_id],
            task_id,
            config["dataset"]["task_types"][task_id],
            config["model"],
            config["evaluation"],
            device,
            task_path,
            max_examples=DEV_ROWS_PER_TASK,
        )
        scores[task_id] = metric
    return scores


def _save_checkpoint_and_reference_logits(
    model: Any,
    tokenizer: Any,
    checkpoint_path: Path,
    tokenizer_prompt: list[int],
    device: torch.device,
) -> torch.Tensor:
    before = _last_position_logits(model, tokenizer_prompt, device)
    save_checkpoint(model, tokenizer, checkpoint_path)

    return before


def _reload_audit(
    base_path: Path,
    checkpoint_path: Path,
    architecture: str,
    tokenizer_prompt: list[int],
    device: torch.device,
    before: torch.Tensor,
) -> dict[str, Any]:
    reloaded = load_model(base_path, local_only=True, training=False)
    apply_architecture(reloaded, architecture)
    state_path = checkpoint_path / "model.safetensors"
    state = load_file(str(state_path), device="cpu")
    incompatible = reloaded.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    del state
    allowed_missing = {"lm_head.weight"}  # Tied to model.embed_tokens.weight by Qwen3.
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Checkpoint state mismatch: missing={missing}; unexpected={unexpected}")
    after = _last_position_logits(reloaded, tokenizer_prompt, device)
    difference = after - before
    result = {
        "checkpoint": str(checkpoint_path),
        "safetensors_size_bytes": state_path.stat().st_size,
        "missing_tied_keys": missing,
        "unexpected_keys": unexpected,
        "max_absolute_logit_difference_after_reload": float(difference.abs().max().item()),
        "relative_l2_logit_difference_after_reload": float(difference.norm().item() / before.norm().clamp_min(1e-12).item()),
        "passes_reload_check": float(difference.abs().max().item()) <= 1e-6,
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    config, base_path, tokenizer, device, train, dev = prepare_context()
    output_root = Path(config["storage"]["root"]) / "results" / "architecture_pilot"
    output_root.mkdir(parents=True, exist_ok=True)
    sequence = list(config["sequences"]["homogeneous"])
    if sequence != TASK_IDS:
        raise RuntimeError("The pilot expects the audited homogeneous task sequence")
    protocol = {
        "task_sequence": sequence,
        "optimizer_steps_per_task": PILOT_STEPS_PER_TASK,
        "gradient_accumulation_steps": int(config["training"]["gradient_accumulation_steps"]),
        "microbatch_size": 1,
        "learning_rate": float(config["training"]["learning_rate"]),
        "max_sequence_length": int(config["model"]["max_sequence_length"]),
        "max_target_tokens": int(config["model"]["max_target_tokens"]),
        "dev_rows_per_task_per_stage": DEV_ROWS_PER_TASK,
        "optimizer_reinitialized_at_task_boundary": True,
        "model_source": str(base_path),
        "local_files_only": True,
    }
    partial_path = output_root / "architecture_pilot_partial.json"
    if partial_path.exists():
        results: dict[str, Any] = json.loads(partial_path.read_text(encoding="utf-8"))
        if results.get("protocol") != protocol:
            raise RuntimeError("Existing partial architecture pilot used a different protocol")
    else:
        results = {"protocol": protocol, "architectures": {}}
    token_budgets_by_architecture: dict[str, dict[str, int]] = {}
    sample_hashes_by_architecture: dict[str, dict[str, str]] = {}
    base_parameter_count: int | None = None
    for completed_architecture, completed_result in results.get("architectures", {}).items():
        token_budgets_by_architecture[completed_architecture] = {
            task_id: int(row["supervised_target_tokens"])
            for task_id, row in completed_result["training_metrics"].items()
        }
        sample_hashes_by_architecture[completed_architecture] = {
            task_id: str(row["sample_order_sha256"])
            for task_id, row in completed_result["training_metrics"].items()
        }
        if completed_architecture == "A0":
            base_parameter_count = int(completed_result["parameter_counts"]["total_parameters"])

    for architecture in ARCHITECTURES:
        if architecture in results.get("architectures", {}):
            print(f"Skipping already completed {architecture} from the saved partial result", flush=True)
            continue
        print(f"\n=== Architecture pilot {architecture} ===", flush=True)
        model = load_model(base_path, local_only=True, training=True)
        torch.cuda.reset_peak_memory_stats(device)
        initialization = _initialization_audit(model, architecture, tokenizer, dev, config, device)
        counts = parameter_counts(model)
        if architecture == "A0":
            base_parameter_count = counts["total_parameters"]
        if counts["trainable_parameters"] != counts["total_parameters"]:
            raise RuntimeError(f"{architecture} is not a full-parameter trainable candidate: {counts}")
        flops = estimate_forward_flops(model.config, architecture, sequence_length=int(config["model"]["max_sequence_length"]))
        prompt, _, _ = safe_prompt(tokenizer, dev[TASK_IDS[0]][0], int(config["model"]["max_sequence_length"]))
        candidate_root = output_root / architecture
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_started = time.perf_counter()
        stage_order = ["baseline"]
        stage_scores: dict[str, Any] = {
            "baseline": evaluate_matrix(model, tokenizer, architecture, "baseline", config, dev, device, output_root)
        }
        training_metrics: dict[str, Any] = {}

        for stage_index, task_id in enumerate(sequence, start=1):
            seed = safe_task_seed(int(config["seed"]), task_id)
            torch.cuda.reset_peak_memory_stats(device)
            train_row = train_task_recorded(
                model,
                tokenizer,
                train[task_id],
                config["training"],
                config["model"],
                device,
                float(config["training"]["learning_rate"]),
                PILOT_STEPS_PER_TASK,
                seed,
            )
            train_row["task_id"] = task_id
            train_row["training_seed"] = seed
            training_metrics[task_id] = train_row
            stage_name = f"after_{stage_index}_{task_id}"
            stage_order.append(stage_name)
            stage_scores[stage_name] = evaluate_matrix(
                model, tokenizer, architecture, stage_name, config, dev, device, output_root
            )
            token_budgets_by_architecture.setdefault(architecture, {})[task_id] = train_row[
                "supervised_target_tokens"
            ]
            sample_hashes_by_architecture.setdefault(architecture, {})[task_id] = train_row[
                "sample_order_sha256"
            ]
            partial = {
                "architecture": architecture,
                "protocol": protocol,
                "initialization_audit": initialization,
                "parameter_counts": counts,
                "flops_at_max_sequence_length": flops,
                "stage_order": stage_order,
                "stage_scores": stage_scores,
                "training_metrics": training_metrics,
                "partial_saved_utc_epoch_seconds": time.time(),
            }
            write_json(candidate_root / "metrics_partial.json", partial)
            print(f"Completed {architecture} stage {stage_index}/{len(sequence)}: {task_id}", flush=True)

        checkpoint_path = output_root / "checkpoints" / architecture
        reference_logits = _save_checkpoint_and_reference_logits(
            model,
            tokenizer,
            checkpoint_path,
            prompt,
            device,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if base_parameter_count is None:
            raise RuntimeError("A0 baseline parameter count was not recorded")
        reload_audit = _reload_audit(base_path, checkpoint_path, architecture, prompt, device, reference_logits)
        candidate_result = {
            "architecture": architecture,
            "initialization_audit": initialization,
            "parameter_counts": counts,
            "new_parameters": counts["total_parameters"] - base_parameter_count,
            "flops_at_max_sequence_length": flops,
            "stage_order": stage_order,
            "stage_scores": stage_scores,
            "training_metrics": training_metrics,
            "checkpoint_reload_audit": reload_audit,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "duration_seconds": time.perf_counter() - candidate_started,
            "checkpoint_path": str(checkpoint_path),
        }
        results["architectures"][architecture] = candidate_result
        write_json(output_root / "architecture_pilot_partial.json", results)
        gc.collect()
        torch.cuda.empty_cache()

    reference_tokens = token_budgets_by_architecture[ARCHITECTURES[0]]
    reference_hashes = sample_hashes_by_architecture[ARCHITECTURES[0]]
    budget_comparison: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        budget_comparison[architecture] = {
            task_id: {
                "supervised_target_tokens": token_budgets_by_architecture[architecture][task_id],
                "matches_A0_token_budget": token_budgets_by_architecture[architecture][task_id]
                == reference_tokens[task_id],
                "sample_order_sha256": sample_hashes_by_architecture[architecture][task_id],
                "matches_A0_sample_order": sample_hashes_by_architecture[architecture][task_id]
                == reference_hashes[task_id],
            }
            for task_id in TASK_IDS
        }
    hf_config = AutoConfig.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False)
    for architecture, candidate in results["architectures"].items():
        candidate["flops_at_max_sequence_length"] = estimate_forward_flops(
            hf_config,
            architecture,
            sequence_length=int(config["model"]["max_sequence_length"]),
        )
    results["capacity_matched_controls"] = {
        "A1-control": {
            "total_parameters": int(results["architectures"]["A1"]["parameter_counts"]["total_parameters"]),
            "new_parameters": int(results["architectures"]["A1"]["new_parameters"]),
            "matches_A1_parameter_budget": True,
            "definition": "Per-head input-conditioned sigmoid gating; uses A1's query/key weights but sends no messages across heads.",
            "flops_at_max_sequence_length": estimate_forward_flops(
                hf_config, "A1-control", sequence_length=int(config["model"]["max_sequence_length"])
            ),
            "smoke_trained": False,
        },
        "A2-control": {
            "total_parameters": int(results["architectures"]["A2"]["parameter_counts"]["total_parameters"]),
            "new_parameters": int(results["architectures"]["A2"]["new_parameters"]),
            "matches_A2_parameter_budget": True,
            "definition": "Dense input-conditioned softmax mixture over all four experts; uses the same router and expert weights as A2.",
            "flops_at_max_sequence_length": estimate_forward_flops(
                hf_config, "A2-control", sequence_length=int(config["model"]["max_sequence_length"])
            ),
            "smoke_trained": False,
        },
    }
    results["training_budget_comparison"] = budget_comparison
    results["test_split_loaded"] = False
    results["selection_split"] = "dev only"
    results["local_files_only"] = True
    results["result_root"] = str(output_root)
    for architecture, candidate in results["architectures"].items():
        candidate_partial_path = output_root / architecture / "metrics_partial.json"
        if candidate_partial_path.exists():
            candidate_partial = json.loads(candidate_partial_path.read_text(encoding="utf-8"))
            candidate_partial["flops_at_max_sequence_length"] = candidate["flops_at_max_sequence_length"]
            candidate_partial["checkpoint_reload_audit"] = candidate["checkpoint_reload_audit"]
            candidate_partial["peak_cuda_allocated_bytes"] = candidate["peak_cuda_allocated_bytes"]
            candidate_partial["peak_cuda_reserved_bytes"] = candidate["peak_cuda_reserved_bytes"]
            write_json(candidate_partial_path, candidate_partial)
    write_json(partial_path, results)
    write_json(output_root / "architecture_pilot_metrics.json", results)
    print(f"Architecture pilot complete. Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
