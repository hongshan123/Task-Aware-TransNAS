"""GPU smoke for B2 zero-init, input routing, gradients, and state round-trip."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from phase4_common import prepare_context, safe_task_seed, train_task_recorded, write_json
from phase6b_modules import install_residual_paths
from run_formal_baseline import load_model, safe_prompt
from run_phase6b_pilot import _restore_state, _save_state


def main() -> None:
    config, model_path, tokenizer, device, train, dev = prepare_context()
    torch.cuda.set_device(device)
    task_id = "task565_circa_answer_generation"
    model = load_model(model_path, local_only=True, training=True)
    controller = install_residual_paths(model, "B2-input", [25, 27], 1024, 64, 4)
    prompt, _, _ = safe_prompt(tokenizer, dev[task_id][0], int(config["model"]["max_sequence_length"]))
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        before = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()
    controller.set_condition("input")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        initial_after = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()
    init_diff = float((before - initial_after).abs().max())
    if init_diff != 0.0:
        raise RuntimeError(f"B2 did not preserve the initial function exactly: {init_diff}")

    trace = []
    controller.set_condition("input_train")
    controller.reset_statistics()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    train_row = train_task_recorded(
        model,
        tokenizer,
        train[task_id],
        config["training"],
        config["model"],
        device,
        1e-5,
        2,
        safe_task_seed(10, task_id),
        on_optimizer_step=lambda step, _model: trace.append({
            "step": step,
            "paths": controller.branch_gradient_statistics(),
        }),
    )
    branch_stats = controller.branch_gradient_statistics()
    route_stats = controller.route_statistics()
    branch_updated = any(
        float(parameter.detach().abs().sum().item()) > 0.0
        for path in controller.paths for parameter in path.module.parameters()
        if parameter.ndim >= 2 and "down_proj" in next(
            name for name, candidate in path.module.named_parameters() if candidate is parameter
        )
    )
    router_gradient_present = any(row["router_has_gradient"] and row["router_gradient_l2"] > 0 for row in branch_stats.values())
    branch_gradient_present = any(
        expert["has_gradient"] and expert["gradient_l2"] > 0
        for row in branch_stats.values() for expert in row["branch_expert_gradients"]
    )
    if not branch_updated or not branch_gradient_present or not router_gradient_present:
        raise RuntimeError(f"B2 did not show branch/router learning: {branch_stats}")

    model.eval()
    controller.set_condition("input")
    controller.reset_statistics()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reference = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()
    eval_routes = controller.route_statistics()
    if any(info["task_id_used_by_router"] for info in eval_routes.values()):
        raise RuntimeError("B2 inference unexpectedly consumed a Task ID")

    state_path = ROOT / "results" / "phase6b" / ".phase6b_state_roundtrip.safetensors"
    _save_state(model, state_path)
    controller.close()
    del model, controller
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_model(model_path, local_only=True, training=False)
    reload_controller = install_residual_paths(reloaded, "B2-input", [25, 27], 1024, 64, 4)
    restore_keys = _restore_state(reloaded, state_path)
    reload_controller.set_condition("input")
    reloaded.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        restored = reloaded(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1].float().cpu()
    reload_diff = float((reference - restored).abs().max())
    state_size = state_path.stat().st_size
    state_path.unlink()
    if reload_diff > 1e-6:
        raise RuntimeError(f"B2 state-dict round-trip changed logits: {reload_diff}")
    reload_controller.close()
    del reloaded, reload_controller
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "status": "passed",
        "architecture": "B2-input",
        "task_id": task_id,
        "diagnostic_optimizer_steps": 2,
        "formal_pilot_steps_included": False,
        "test_split_loaded": False,
        "historical_model_loaded": False,
        "initial_max_logit_difference": init_diff,
        "route_does_not_use_task_id": True,
        "training_route_statistics": route_stats,
        "gradient_trace": trace,
        "branch_parameters_updated": branch_updated,
        "branch_gradient_present": branch_gradient_present,
        "router_gradient_present": router_gradient_present,
        "evaluation_route_statistics": eval_routes,
        "checkpoint_state_bytes": state_size,
        "checkpoint_reload_max_logit_difference": reload_diff,
        "checkpoint_restore_keys": restore_keys,
        "training_metrics": {key: train_row[key] for key in ("optimizer_steps", "sample_order_sha256", "supervised_target_tokens", "peak_cuda_allocated_bytes", "duration_seconds")},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "gpu_name": torch.cuda.get_device_name(device),
    }
    write_json(ROOT / "results" / "phase6b" / "gpu_smoke.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
