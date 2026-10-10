"""Run the seed-10 Phase 6B matched continual-learning architecture pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from architecture_modules import estimate_forward_flops
from phase4_common import (
    TASK_IDS,
    evaluate_dev_task,
    prepare_context,
    safe_task_seed,
    teacher_forced_nll,
    train_task_recorded,
    write_json,
)
from phase6b_modules import (
    ConditionalResidualRouter,
    ResidualPathController,
    install_residual_paths,
    phase6b_added_parameter_count,
    select_random_post_layers,
)
from run_formal_baseline import load_model, safe_prompt
from run_phase4b_comparison import SEQUENCES, TASK_CALIBRATED_STEPS, summarize_predictions
from task_aware_transnas.cl.metrics import continual_learning_metrics


OUT = ROOT / "results" / "phase6b"
ARCHITECTURES = ("B1-static", "B2-input", "B3-oracle", "B2-random-location")
PRIMARY_LAYERS = [25, 27]
BOTTLENECK_PER_EXPERT = 64
NUM_EXPERTS = 4
MAX_NEW_TOKENS = 64
ALLOWED_MISSING_TIED = {"lm_head.weight"}
SOURCE_MODEL_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_write(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _primary_scores(stage_rows: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    return {
        task: float(stage_rows[task]["metrics"][config["dataset"]["task_types"][task]["primary_metric"]])
        for task in TASK_IDS
    }


def _save_state(model: Any, path: Path) -> None:
    """Atomically write one full current state; tied LM-head alias is omitted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.safetensors")
    state = {
        key: tensor.detach().to(device="cpu", copy=True).contiguous()
        for key, tensor in model.state_dict().items()
        if key != "lm_head.weight"
    }
    save_file(state, str(temporary))
    del state
    os.replace(temporary, path)


def _restore_state(model: Any, path: Path) -> dict[str, Any]:
    state = load_file(str(path), device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    del state
    if set(missing) - ALLOWED_MISSING_TIED or unexpected:
        raise RuntimeError(f"Phase 6B state mismatch: missing={missing}; unexpected={unexpected}")
    return {"missing_tied_keys": missing, "unexpected_keys": unexpected}


def _condition_for(architecture: str, task_id: str, mode_override: str | None = None) -> tuple[str | None, int | None]:
    task_index = TASK_IDS.index(task_id)
    if architecture == "B1-static":
        return "static", None
    if architecture == "B3-oracle":
        return "oracle", task_index
    return (mode_override or "input"), None


def evaluate_matrix(
    model: Any,
    controller: ResidualPathController | None,
    architecture: str,
    config: dict[str, Any],
    tokenizer: Any,
    dev: dict[str, list[dict[str, Any]]],
    device: torch.device,
    run_root: Path,
    stage_name: str,
    route_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    task_rows: dict[str, Any] = {}
    scores: dict[str, float] = {}
    eval_started = time.perf_counter()
    for task_id in TASK_IDS:
        if controller is not None:
            policy, task_index = _condition_for(architecture, task_id, route_override)
            controller.set_condition(policy, task_index)
        nll = teacher_forced_nll(model, tokenizer, dev[task_id], config["model"], device)
        if controller is not None:
            controller.reset_statistics()
        pred_path = run_root / "predictions" / stage_name / (route_override or "primary") / f"{task_id}.jsonl"
        metrics, predictions = evaluate_dev_task(
            model,
            tokenizer,
            dev[task_id],
            task_id,
            config["dataset"]["task_types"][task_id],
            config["model"],
            config["evaluation"],
            device,
            pred_path,
            max_examples=len(dev[task_id]),
        )
        metric_name = config["dataset"]["task_types"][task_id]["primary_metric"]
        scores[task_id] = float(metrics[metric_name])
        pred_summary = summarize_predictions(predictions)
        yes_share = float(pred_summary["exact_yes_share"])
        no_share = float(pred_summary["exact_no_share"])
        task_rows[task_id] = {
            "metrics": metrics,
            "teacher_forced_nll": nll,
            "generation_behavior": pred_summary,
            "yes_no_dominant_output_share": max(yes_share, no_share),
            "yes_no_single_label_collapse": bool(max(yes_share, no_share) >= 0.95),
            "router_usage_by_layer": {} if controller is None else controller.route_statistics(),
            "prediction_file": str(pred_path),
        }
        print(
            f"eval {architecture}/{stage_name}/{task_id}: {metric_name}={scores[task_id]:.4f}; "
            f"NLL={nll['token_nll']:.4f}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    summary = {
        "duration_seconds": time.perf_counter() - eval_started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "dev_examples_per_task": len(dev[TASK_IDS[0]]),
        "route_override": route_override,
    }
    return task_rows, scores, summary


def _load_a0_metrics(sequence_name: str) -> dict[str, Any]:
    path = ROOT / "results" / "phase4b" / "runs" / "A0" / sequence_name / "seed_10" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Existing matched A0 seed-10 baseline is required: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if int(row.get("seed", -1)) != 10 or row.get("sequence_name") != sequence_name:
        raise RuntimeError(f"A0 baseline does not match seed/order: {path}")
    if row.get("test_split_loaded") is not False or row.get("split") != "dev":
        raise RuntimeError(f"A0 baseline is not a verified dev-only run: {path}")
    if row.get("source_model_sha256") != SOURCE_MODEL_SHA256:
        raise RuntimeError(f"A0 baseline source model hash mismatch: {path}")
    if row.get("task_order") != SEQUENCES[sequence_name]:
        raise RuntimeError(f"A0 task order differs from Phase 6B plan: {path}")
    for task_id, steps in TASK_CALIBRATED_STEPS.items():
        if int(row["training_metrics"][task_id]["optimizer_steps"]) != steps:
            raise RuntimeError(f"A0 optimizer budget mismatch for {task_id}: {path}")
    return row


def _initialization_check(
    model: Any,
    controller: ResidualPathController,
    architecture: str,
    task_id: str,
    record: dict[str, Any],
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prompt, _, info = safe_prompt(tokenizer, record, int(config["model"]["max_sequence_length"]))
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    model.eval()
    model.config.use_cache = False
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        before = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
    policy, task_index = _condition_for(architecture, task_id)
    controller.set_condition(policy, task_index)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        after = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float()
    difference = (after - before).abs()
    return {
        "task_id": task_id,
        "instance_id": record["instance_id"],
        "prompt_tokens": len(prompt),
        "input_tail_truncated": info["input_tail_truncated"],
        "max_absolute_logit_difference": float(difference.max().item()),
        "exact_output_preservation": bool(float(difference.max().item()) == 0.0),
        "task_id_visible_to_route": architecture == "B3-oracle",
    }


def _build_flop_report(model_config: Any, architecture: str, selected_layers: list[int]) -> dict[str, Any]:
    base = estimate_forward_flops(model_config, "A0", sequence_length=512)
    hidden = int(model_config.hidden_size)
    layer_count = len(selected_layers)
    base_tokens = base["approx_flops_per_prefill_token_at_sequence_length"]
    if architecture == "B1-static":
        extra_per_token = layer_count * 4 * hidden * BOTTLENECK_PER_EXPERT * NUM_EXPERTS
    else:
        expert_flops = 4 * hidden * BOTTLENECK_PER_EXPERT
        router_flops = 2 * hidden * NUM_EXPERTS if architecture in {"B2-input", "B2-random-location"} else 0
        extra_per_token = layer_count * (expert_flops + router_flops)
    return {
        "base_approx_flops_per_512_token_prefill": base["approx_flops_per_sequence"],
        "extra_approx_flops_per_active_token": extra_per_token,
        "extra_approx_flops_per_512_token_prefill": extra_per_token * 512,
        "total_approx_flops_per_512_token_prefill": base["approx_flops_per_sequence"] + extra_per_token * 512,
        "approx_extra_flop_fraction": extra_per_token / max(1, base_tokens),
        "flop_note": "Linear multiply-add estimate; excludes activation, dispatch, and cache costs. B2 top-1 evaluates one expert per token.",
    }


def run_variant(
    architecture: str,
    sequence_name: str,
    config: dict[str, Any],
    model_path: Path,
    tokenizer: Any,
    train: dict[str, list[dict[str, Any]]],
    dev: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> dict[str, Any]:
    sequence = list(SEQUENCES[sequence_name])
    run_root = OUT / "runs" / sequence_name / architecture / "seed_10"
    run_root.mkdir(parents=True, exist_ok=True)
    metrics_path = run_root / "metrics.json"
    checkpoint_root = OUT / "checkpoints" / sequence_name / architecture / "seed_10"
    final_state_path = checkpoint_root / "final_state.safetensors"
    if metrics_path.is_file() and final_state_path.is_file():
        print(f"Skipping complete run {sequence_name}/{architecture}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    baseline = _load_a0_metrics(sequence_name)
    run_started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc)
    torch.manual_seed(10)
    model = load_model(model_path, local_only=True, training=True)
    if architecture == "B2-random-location":
        location = select_random_post_layers(
            list(config["architecture"]["random_location_candidate_layers"]),
            len(PRIMARY_LAYERS),
            int(config["architecture"]["random_location_selection_seed"]),
            PRIMARY_LAYERS,
        )
    else:
        location = PRIMARY_LAYERS
    controller = install_residual_paths(
        model,
        architecture,
        location,
        hidden_size=int(model.config.hidden_size),
        bottleneck_size=BOTTLENECK_PER_EXPERT,
        num_experts=NUM_EXPERTS,
    )
    if controller is None:
        raise RuntimeError("Non-baseline Phase 6B variant unexpectedly has no residual controller")
    base_parameter_count = int(baseline["parameter_counts"]["total_parameters"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    added_parameters = total_parameters - base_parameter_count
    expected_added = phase6b_added_parameter_count(
        architecture,
        int(model.config.hidden_size),
        len(location),
        BOTTLENECK_PER_EXPERT,
        NUM_EXPERTS,
    )
    if added_parameters != expected_added or added_parameters / base_parameter_count >= 0.01:
        raise RuntimeError(f"Unexpected Phase 6B parameter budget: {added_parameters=} {expected_added=}")
    if trainable_parameters != total_parameters:
        raise RuntimeError(f"Expected the base and residual branches to receive full-parameter updates: {trainable_parameters}/{total_parameters}")
    if baseline["task_order"] != sequence:
        raise RuntimeError(f"Baseline and candidate sequences differ: {sequence_name}")

    initialization = _initialization_check(
        model, controller, architecture, sequence[0], dev[sequence[0]][0], tokenizer, config, device
    )
    if not initialization["exact_output_preservation"]:
        raise RuntimeError(f"Residual architecture changed the initialized model function: {initialization}")

    controller.set_condition(_condition_for(architecture, sequence[0])[0], _condition_for(architecture, sequence[0])[1])
    stage_order = ["baseline"]
    scores_by_stage: dict[str, dict[str, float]] = {
        "baseline": {task: float(score) for task, score in baseline["stage_scores_by_primary_metric"]["baseline"].items()}
    }
    task_metrics: dict[str, Any] = {"baseline": baseline["task_metrics"]["baseline"]}
    eval_times: dict[str, Any] = {"baseline": {"reused_from": str(ROOT / "results" / "phase4b" / "runs" / "A0" / sequence_name / "seed_10" / "metrics.json"), "duration_seconds": 0.0}}
    training_metrics: dict[str, Any] = {}
    gradient_trace_path = run_root / "gradient_diagnostics.jsonl"
    gradient_trace_path.write_text("", encoding="utf-8")
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    partial = {
        "status": "running",
        "architecture": architecture,
        "sequence_name": sequence_name,
        "seed": 10,
        "task_order": sequence,
        "training_protocol": baseline["training_protocol"],
        "test_split_loaded": False,
        "nas_started": False,
        "stage_order": stage_order,
        "stage_scores_by_primary_metric": scores_by_stage,
        "task_metrics": task_metrics,
        "training_metrics": training_metrics,
        "evaluation_metrics": eval_times,
        "initialization_check": initialization,
        "parameter_counts": {
            "base_parameters": base_parameter_count,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "added_parameters": added_parameters,
            "added_fraction_of_base": added_parameters / base_parameter_count,
        },
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "data_hashes": {
            "train_instance_id_sha256_by_task": baseline["train_instance_id_sha256_by_task"],
            "dev_instance_id_sha256_by_task": baseline["dev_instance_id_sha256_by_task"],
        },
    }
    write_json(run_root / "metrics_partial.json", partial)

    for stage_index, task_id in enumerate(sequence, start=1):
        task_seed = safe_task_seed(10, task_id)
        policy, task_index = _condition_for(architecture, task_id)
        if architecture in {"B2-input", "B2-random-location"}:
            policy = "input_train"
        controller.set_condition(policy, task_index)
        controller.reset_statistics()
        torch.cuda.reset_peak_memory_stats(device)
        sampled_steps = TASK_CALIBRATED_STEPS[task_id]
        trace_rows: list[dict[str, Any]] = []

        def on_step(step: int, _current_model: Any) -> None:
            if step == 1 or step % 10 == 0 or step == sampled_steps:
                trace_rows.append({
                    "architecture": architecture,
                    "sequence": sequence_name,
                    "seed": 10,
                    "task_id": task_id,
                    "optimizer_step": step,
                    "route_policy": policy,
                    "paths": controller.branch_gradient_statistics(),
                })

        print(f"\n=== Phase 6B {sequence_name}/{architecture}: {task_id} ({sampled_steps} steps) ===", flush=True)
        training_row = train_task_recorded(
            model,
            tokenizer,
            train[task_id],
            config["training"],
            config["model"],
            device,
            float(baseline["training_protocol"]["learning_rate"]),
            sampled_steps,
            task_seed,
            on_optimizer_step=on_step,
        )
        training_row.update({"task_id": task_id, "task_seed": task_seed, "route_policy": policy})
        training_row["max_cuda_allocated_bytes"] = max(
            int(training_row["peak_cuda_allocated_bytes"]), int(torch.cuda.max_memory_allocated(device))
        )
        training_row["max_cuda_reserved_bytes"] = max(
            int(training_row["peak_cuda_reserved_bytes"]), int(torch.cuda.max_memory_reserved(device))
        )
        training_row["route_usage_during_training"] = controller.route_statistics()
        base_train = baseline["training_metrics"][task_id]
        if training_row["sample_order_sha256"] != base_train["sample_order_sha256"]:
            raise RuntimeError(f"Sample order mismatch with A0 for {sequence_name}/{architecture}/{task_id}")
        if int(training_row["supervised_target_tokens"]) != int(base_train["supervised_target_tokens"]):
            raise RuntimeError(f"Supervised target token budget mismatch with A0 for {sequence_name}/{architecture}/{task_id}")
        training_row["sample_order_matches_A0"] = True
        training_row["supervised_target_tokens_matches_A0"] = True
        training_metrics[task_id] = training_row
        peak_allocated = max(peak_allocated, training_row["max_cuda_allocated_bytes"])
        peak_reserved = max(peak_reserved, training_row["max_cuda_reserved_bytes"])
        _jsonl_write(gradient_trace_path, trace_rows, append=True)

        stage_name = f"after_{stage_index}_{task_id}"
        stage_order.append(stage_name)
        torch.cuda.reset_peak_memory_stats(device)
        current_rows, current_scores, eval_summary = evaluate_matrix(
            model, controller, architecture, config, tokenizer, dev, device, run_root, stage_name
        )
        task_metrics[stage_name] = current_rows
        scores_by_stage[stage_name] = current_scores
        eval_times[stage_name] = eval_summary
        peak_allocated = max(peak_allocated, int(eval_summary["peak_cuda_allocated_bytes"]))
        peak_reserved = max(peak_reserved, int(eval_summary["peak_cuda_reserved_bytes"]))
        partial.update({
            "stage_order": stage_order,
            "stage_scores_by_primary_metric": scores_by_stage,
            "task_metrics": task_metrics,
            "training_metrics": training_metrics,
            "evaluation_metrics": eval_times,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        })
        write_json(run_root / "metrics_partial.json", partial)
        print(f"Finished and recorded {stage_name}", flush=True)

    metric_by_task = {
        task_id: str(config["dataset"]["task_types"][task_id]["primary_metric"]) for task_id in TASK_IDS
    }
    cl_metrics = continual_learning_metrics(scores_by_stage, sequence, metric_by_task)
    acquisition = {}
    for index, task_id in enumerate(sequence, start=1):
        stage_name = stage_order[index]
        base_score = scores_by_stage["baseline"][task_id]
        learned_score = scores_by_stage[stage_name][task_id]
        acquisition[stage_name] = {
            "task_id": task_id,
            "primary_metric": metric_by_task[task_id],
            "baseline_score": base_score,
            "score_after_acquisition": learned_score,
            "raw_acquisition_gain": learned_score - base_score,
            "normalized_acquisition_gain": cl_metrics["normalized_gain_matrix"][stage_name][task_id],
        }

    route_controls = {}
    if architecture == "B2-input":
        for mode in ("random", "static_gate", "off"):
            torch.cuda.reset_peak_memory_stats(device)
            ctrl_rows, ctrl_scores, ctrl_summary = evaluate_matrix(
                model, controller, architecture, config, tokenizer, dev, device, run_root,
                f"final_route_control_{mode}", route_override=mode,
            )
            route_controls[mode] = {"task_metrics": ctrl_rows, "primary_scores": ctrl_scores, "evaluation": ctrl_summary}
            peak_allocated = max(peak_allocated, int(ctrl_summary["peak_cuda_allocated_bytes"]))
            peak_reserved = max(peak_reserved, int(ctrl_summary["peak_cuda_reserved_bytes"]))
        write_json(run_root / "final_route_controls.json", route_controls)

    final_layer_set = location
    sample = dev[TASK_IDS[0]][0]
    prompt, _, _ = safe_prompt(tokenizer, sample, int(config["model"]["max_sequence_length"]))
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    policy, task_index = _condition_for(architecture, TASK_IDS[0])
    controller.set_condition(policy, task_index)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference_logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _save_state(model, final_state_path)
    final_state_sha256 = _sha256_file(final_state_path)
    del model
    controller.close()
    del controller
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = load_model(model_path, local_only=True, training=False)
    reload_controller = install_residual_paths(
        reloaded,
        architecture,
        final_layer_set,
        hidden_size=int(reloaded.config.hidden_size),
        bottleneck_size=BOTTLENECK_PER_EXPERT,
        num_experts=NUM_EXPERTS,
    )
    reload_keys = _restore_state(reloaded, final_state_path)
    reload_policy, reload_task = _condition_for(architecture, TASK_IDS[0])
    reload_controller.set_condition(reload_policy, reload_task)
    reloaded.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reload_logits = reloaded(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()
    reload_difference = float((reload_logits - reference_logits).abs().max().item())
    reload_audit = {
        "state_file": str(final_state_path),
        "state_bytes": final_state_path.stat().st_size,
        "state_sha256": final_state_sha256,
        **reload_keys,
        "max_absolute_logit_difference": reload_difference,
        "passes_reload_audit": reload_difference <= 1e-6,
    }
    if not reload_audit["passes_reload_audit"]:
        raise RuntimeError(f"Saved final model reload changed outputs: {reload_audit}")
    reload_controller.close()
    del reloaded, reload_controller
    gc.collect()
    torch.cuda.empty_cache()

    flop_info = _build_flop_report(load_model_config(model_path), architecture, location)
    final_stage = stage_order[-1]
    final_scores = scores_by_stage[final_stage]
    for task_id in TASK_IDS:
        if not math.isfinite(final_scores[task_id]):
            raise RuntimeError(f"Non-finite final primary score for {task_id}")
    finished = datetime.now(timezone.utc)
    result = {
        **partial,
        "status": "complete",
        "started_utc": run_started_utc.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": time.perf_counter() - run_started,
        "architecture_definition": {
            "variant": architecture,
            "layers_zero_based": location,
            "random_location_candidate_layers": config["architecture"].get("random_location_candidate_layers") if architecture == "B2-random-location" else None,
            "random_location_selection_seed": config["architecture"].get("random_location_selection_seed") if architecture == "B2-random-location" else None,
            "bottleneck_size_per_routed_path": BOTTLENECK_PER_EXPERT,
            "number_of_fixed_shared_routes": NUM_EXPERTS,
            "route_policy_during_training": {task: training_metrics[task]["route_policy"] for task in sequence},
            "route_inference_uses_task_id": architecture == "B3-oracle",
            "per_task_parameters_added": False,
            "historical_model_loaded_during_pilot": False,
            "replay_used": False,
        },
        "parameter_counts": partial["parameter_counts"],
        "flops": flop_info,
        "continual_learning_metrics": cl_metrics,
        "new_task_acquisition": acquisition,
        "route_control_ablations": route_controls,
        "checkpoint_reload_audit": reload_audit,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "gpu_name": torch.cuda.get_device_name(device),
        "free_vram_bytes_at_finish": torch.cuda.mem_get_info(device)[0],
        "test_split_loaded": False,
        "nas_started": False,
    }
    write_json(metrics_path, result)
    metadata = {
        "architecture": architecture,
        "sequence": sequence_name,
        "seed": 10,
        "checkpoint_format": "state_dict safetensors; tied lm_head alias omitted",
        "base_model_path": str(model_path),
        "base_model_sha256": SOURCE_MODEL_SHA256,
        "model_config_path": str(model_path / "config.json"),
        "layers": location,
        "bottleneck_per_route": BOTTLENECK_PER_EXPERT,
        "routes": NUM_EXPERTS,
        "added_parameters": added_parameters,
        "test_split_loaded": False,
        "historical_models_saved": False,
    }
    write_json(checkpoint_root / "checkpoint_metadata.json", metadata)
    write_json(run_root / "metrics_partial.json", result)
    print(
        f"Complete {sequence_name}/{architecture}; score={final_scores}; "
        f"added={added_parameters:,} ({100*added_parameters/base_parameter_count:.3f}%); "
        f"peak={peak_allocated/1024**3:.2f} GiB; GPU time={result['duration_seconds']:.1f}s",
        flush=True,
    )
    return result


def load_model_config(model_path: Path) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)


def _disk_preflight(model_path: Path) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    drive = shutil.disk_usage(OUT.resolve())
    actual_model_sha256 = _sha256_file(model_path / "model.safetensors")
    if actual_model_sha256 != SOURCE_MODEL_SHA256:
        raise RuntimeError(
            f"Local Qwen3 source hash differs from the paired A0 baseline: {actual_model_sha256}"
        )
    model_parameters = 596_049_920
    expected_checkpoint_bytes = model_parameters * 4 - 4096
    total_final_bytes = expected_checkpoint_bytes * (len(ARCHITECTURES) * len(SEQUENCES))
    active_temp_bytes = expected_checkpoint_bytes
    safety_bytes = 8 * 1024**3
    required = total_final_bytes + active_temp_bytes + safety_bytes
    if drive.free < required:
        raise RuntimeError(
            f"Phase 6B checkpoint preflight failed: need {required/1024**3:.1f} GiB, "
            f"available {drive.free/1024**3:.1f} GiB. No GPU run started."
        )
    return {
        "disk_root": str(OUT.resolve()),
        "source_model_path": str(model_path),
        "source_model_sha256": actual_model_sha256,
        "available_bytes_before_run": drive.free,
        "expected_one_final_state_bytes": expected_checkpoint_bytes,
        "expected_nonbaseline_final_checkpoint_count": len(ARCHITECTURES) * len(SEQUENCES),
        "estimated_all_final_states_gib": total_final_bytes / 1024**3,
        "active_atomic_save_allowance_gib": active_temp_bytes / 1024**3,
        "safety_reserve_gib": safety_bytes / 1024**3,
        "estimated_required_free_gib": required / 1024**3,
        "preflight_passed": True,
        "saved_stage_checkpoint_count": 0,
        "historical_checkpoints_saved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument("--architectures", nargs="+", choices=list(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config, model_path, tokenizer, device, train, dev = prepare_context()
    torch.cuda.set_device(device)
    if config["training"]["steps_per_task"] != 50 or config["training"]["learning_rate"] != 1e-5:
        raise RuntimeError("The Phase 6B pilot expects the frozen Phase 4B training configuration")
    sampled_random_layers = select_random_post_layers(
        list(config["architecture"]["random_location_candidate_layers"]),
        len(PRIMARY_LAYERS),
        int(config["architecture"]["random_location_selection_seed"]),
        PRIMARY_LAYERS,
    )
    declared_random_layers = sorted(int(layer) for layer in config["architecture"]["random_location_layers"])
    if sampled_random_layers != declared_random_layers:
        raise RuntimeError(
            f"Random location layers do not match seed-based selection: "
            f"sampled={sampled_random_layers}; declared={declared_random_layers}"
        )
    disk = _disk_preflight(model_path)
    write_json(OUT / "disk_preflight.json", disk)
    if args.preflight_only:
        print(json.dumps(disk, indent=2), flush=True)
        return
    for task_id in TASK_IDS:
        if len(dev[task_id]) != 50 or any(record.get("task_id") != task_id for record in dev[task_id]):
            raise RuntimeError(f"Expected the audited 50-row dev split for {task_id}")
    results: dict[str, Any] = {"disk_preflight": disk, "baseline_reused": {}, "pilot_runs": {}}
    for sequence_name in args.sequences:
        baseline = _load_a0_metrics(sequence_name)
        results["baseline_reused"][sequence_name] = {
            "source_metrics": str(ROOT / "results" / "phase4b" / "runs" / "A0" / sequence_name / "seed_10" / "metrics.json"),
            "training_seed": 10,
            "task_order": baseline["task_order"],
            "sample_order_sha256_by_task": {task: baseline["training_metrics"][task]["sample_order_sha256"] for task in TASK_IDS},
            "dev_id_sha256_by_task": baseline["dev_instance_id_sha256_by_task"],
            "test_split_loaded": False,
        }
        b0_root = OUT / "runs" / sequence_name / "B0" / "seed_10"
        b0_root.mkdir(parents=True, exist_ok=True)
        write_json(b0_root / "metrics.json", {
            "status": "complete",
            "architecture": "B0",
            "sequence_name": sequence_name,
            "seed": 10,
            "task_order": baseline["task_order"],
            "source_metrics_reused_from": str(ROOT / "results" / "phase4b" / "runs" / "A0" / sequence_name / "seed_10" / "metrics.json"),
            "stage_scores_by_primary_metric": baseline["stage_scores_by_primary_metric"],
            "task_metrics": baseline["task_metrics"],
            "training_metrics": baseline["training_metrics"],
            "continual_learning_metrics": baseline["continual_learning_metrics"],
            "source_model_sha256": baseline["source_model_sha256"],
            "test_split_loaded": False,
            "nas_started": False,
        })
        for architecture in args.architectures:
            key = f"{sequence_name}/{architecture}/seed_10"
            results["pilot_runs"][key] = run_variant(
                architecture, sequence_name, config, model_path, tokenizer, train, dev, device
            )
            write_json(OUT / "pilot_manifest_partial.json", results)
    manifest = {
        "phase": "6B",
        "status": "complete" if len(results["pilot_runs"]) == len(args.sequences) * len(args.architectures) else "partial",
        "sequences": args.sequences,
        "architectures": args.architectures,
        "seed": 10,
        "test_split_loaded": False,
        "nas_started": False,
        "pilot_runs": results["pilot_runs"],
        "baseline_reused": results["baseline_reused"],
        "disk_preflight": disk,
    }
    write_json(OUT / "pilot_manifest.json", manifest)
    write_json(OUT / "pilot_manifest_partial.json", {**results, "status": manifest["status"]})
    print(json.dumps({"status": manifest["status"], "runs": list(manifest["pilot_runs"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
