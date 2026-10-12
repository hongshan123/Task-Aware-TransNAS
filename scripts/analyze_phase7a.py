"""Offline, dev-only functional-importance and preservation-risk diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
load_dotenv(PROJECT_ROOT / ".env", override=False)

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy.stats import kendalltau, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase7a_parsp import load_phase7a_task_split, official_repo_metadata
from run_formal_baseline import encode_training_record, load_project_config, resolved_model_path
from phase4_common import teacher_forced_nll, write_json


RESULT_ROOT = PROJECT_ROOT / "results" / "phase7a"
FIGURE_ROOT = PROJECT_ROOT / "figures" / "phase7a"


def gradient_dot_delta(gradient: torch.Tensor, delta: torch.Tensor) -> float:
    """First-order old-task NLL change from a component's actual parameter delta."""
    if gradient.shape != delta.shape:
        raise ValueError("Gradient and parameter delta must have matching shapes")
    return float((gradient.detach().float() * delta.detach().float()).sum().item())


def positive_risk_part(signed_risk: float) -> float:
    """Keep only the loss-increasing part while retaining signed risk separately."""
    return max(float(signed_risk), 0.0)


def rank_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        (float(row["importance"]), float(row["risk"]))
        for row in rows
        if row.get("importance") is not None
        and row.get("risk") is not None
        and float(row.get("update_norm", 0.0)) > 1e-12
        and math.isfinite(float(row["importance"]))
        and math.isfinite(float(row["risk"]))
    ]
    if len(values) < 2:
        return {"n": len(values), "spearman_rho": None, "spearman_p": None, "kendall_tau": None, "kendall_p": None}
    importance, risk = zip(*values)
    spearman = spearmanr(importance, risk)
    kendall = kendalltau(importance, risk)
    return {
        "n": len(values),
        "spearman_rho": float(spearman.statistic) if math.isfinite(float(spearman.statistic)) else None,
        "spearman_p": float(spearman.pvalue) if math.isfinite(float(spearman.pvalue)) else None,
        "kendall_tau": float(kendall.statistic) if math.isfinite(float(kendall.statistic)) else None,
        "kendall_p": float(kendall.pvalue) if math.isfinite(float(kendall.pvalue)) else None,
    }


def select_discordant_components(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eligible = [
        row
        for row in rows
        if row.get("importance") is not None
        and row.get("risk") is not None
        and float(row.get("update_norm", 0.0)) > 1e-12
        and math.isfinite(float(row["importance"]))
        and math.isfinite(float(row["risk"]))
    ]
    if not eligible:
        return None, None
    by_importance = sorted(eligible, key=lambda row: float(row["importance"]))
    quartile_count = max(1, math.ceil(len(by_importance) * 0.25))
    low_importance = by_importance[:quartile_count]
    high_importance = by_importance[-quartile_count:]
    high_important_low_risk = min(
        high_importance,
        key=lambda row: (float(row["risk"]), -float(row["importance"]), str(row["component_id"])),
    )
    low_important_high_risk = max(
        low_importance,
        key=lambda row: (float(row["risk"]), -float(row["importance"]), str(row["component_id"])),
    )
    return high_important_low_risk, low_important_high_risk


def matched_control(
    candidates: list[dict[str, Any]],
    target: dict[str, Any],
    exclude_ids: set[str],
    seed: int,
) -> dict[str, Any] | None:
    """Choose a random unit with equal parameter count and near-matched update norm."""
    same_size = [
        row
        for row in candidates
        if str(row.get("component_id")) not in exclude_ids
        and int(row["parameter_count"]) == int(target["parameter_count"])
        and float(row.get("update_norm", 0.0)) > 1e-12
    ]
    if not same_size:
        return None
    target_norm = max(float(target["update_norm"]), 1e-30)
    distances = {
        str(row["component_id"]): abs(float(row["update_norm"]) - target_norm) / target_norm
        for row in same_size
    }
    close = [row for row in same_size if distances[str(row["component_id"])] <= 0.10]
    if not close:
        nearest = sorted(same_size, key=lambda row: (distances[str(row["component_id"])], str(row["component_id"])))
        close = nearest[: min(3, len(nearest))]
    return random.Random(seed).choice(close)


def _parameter_slices(model: torch.nn.Module, kind: str, layer: int, unit: int | None) -> dict[str, tuple[slice, ...]]:
    named = dict(model.named_parameters())
    layer_prefix = f"model.layers.{layer}."
    if kind in {"layer_attention", "layer_ffn"}:
        branch = "self_attn." if kind == "layer_attention" else "mlp."
        return {
            name: tuple(slice(None) for _ in range(parameter.ndim))
            for name, parameter in named.items()
            if name.startswith(layer_prefix + branch)
        }
    if unit is None:
        raise ValueError(f"Component kind {kind} requires a unit index")
    if kind == "ffn_neuron":
        selectors = {
            f"{layer_prefix}mlp.gate_proj.weight": (slice(unit, unit + 1), slice(None)),
            f"{layer_prefix}mlp.gate_proj.bias": (slice(unit, unit + 1),),
            f"{layer_prefix}mlp.up_proj.weight": (slice(unit, unit + 1), slice(None)),
            f"{layer_prefix}mlp.up_proj.bias": (slice(unit, unit + 1),),
            f"{layer_prefix}mlp.down_proj.weight": (slice(None), slice(unit, unit + 1)),
        }
    elif kind == "attention_head":
        width = int(model.config.head_dim)
        start, end = unit * width, (unit + 1) * width
        selectors = {
            f"{layer_prefix}self_attn.q_proj.weight": (slice(start, end), slice(None)),
            f"{layer_prefix}self_attn.q_proj.bias": (slice(start, end),),
            f"{layer_prefix}self_attn.o_proj.weight": (slice(None), slice(start, end)),
        }
    elif kind == "kv_group":
        width = int(model.config.head_dim)
        start, end = unit * width, (unit + 1) * width
        selectors = {}
        for projection in ("k_proj", "v_proj"):
            selectors[f"{layer_prefix}self_attn.{projection}.weight"] = (slice(start, end), slice(None))
            selectors[f"{layer_prefix}self_attn.{projection}.bias"] = (slice(start, end),)
    else:
        raise ValueError(f"Unsupported component kind: {kind}")
    return {name: slices for name, slices in selectors.items() if name in named}


def _component_measure(
    slices: dict[str, tuple[slice, ...]],
    gradients: dict[str, torch.Tensor],
    deltas: dict[str, torch.Tensor],
) -> tuple[float, float, int]:
    risk = 0.0
    squared_norm = 0.0
    parameter_count = 0
    for name, index in slices.items():
        if name not in gradients or name not in deltas:
            continue
        gradient_part = gradients[name][index]
        delta_part = deltas[name][index]
        risk += gradient_dot_delta(gradient_part, delta_part)
        squared_norm += float(delta_part.float().square().sum().item())
        parameter_count += delta_part.numel()
    return risk, math.sqrt(squared_norm), parameter_count


def _component_row(
    model: torch.nn.Module,
    kind: str,
    layer: int,
    unit: int | None,
    importance: float | None,
    gradients: dict[str, torch.Tensor],
    deltas: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, tuple[slice, ...]]]:
    slices = _parameter_slices(model, kind, layer, unit)
    risk, norm, count = _component_measure(slices, gradients, deltas)
    component_id = f"{kind}:{layer}:{unit if unit is not None else 'all'}"
    row = {
        "component_id": component_id,
        "component_type": kind,
        "layer": layer,
        "unit": unit if unit is not None else "all",
        "importance": importance,
        "risk": risk,
        "positive_risk": positive_risk_part(risk),
        "update_norm": norm,
        "parameter_count": count,
    }
    return row, slices


def _load_importance(path: Path) -> dict[tuple[str, int, int | None], float]:
    output = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row["component_type"]
            layer = int(row["layer"])
            unit = None if row["unit"] == "all" else int(row["unit"])
            output[(kind, layer, unit)] = float(row["score"])
    return output


def _dev_nll_gradient(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    encoded = [
        encode_training_record(
            tokenizer,
            row,
            int(model_cfg["max_sequence_length"]),
            int(model_cfg["max_target_tokens"]),
        )
        for row in records
    ]
    total_tokens = sum(int(torch.tensor(labels[1:]).ne(-100).sum().item()) for _, labels, _ in encoded)
    if total_tokens <= 0:
        raise ValueError("Old-task dev split has no supervised target tokens")
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
    total_nll = 0.0
    for ids, labels, _ in encoded:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = torch.tensor([labels], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        shifted_logits = logits[:, :-1, :].float().reshape(-1, logits.shape[-1])
        shifted_labels = y[:, 1:].reshape(-1)
        loss_sum = F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100, reduction="sum")
        total_nll += float(loss_sum.detach().item())
        (loss_sum / total_tokens).backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    model.config.use_cache = True
    return gradients, total_nll / total_tokens


def _rollback_nll(
    model: torch.nn.Module,
    before_state: dict[str, torch.Tensor],
    slices: dict[str, tuple[slice, ...]],
    tokenizer: Any,
    records: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    device: torch.device,
) -> float:
    named = dict(model.named_parameters())
    saved = {}
    with torch.no_grad():
        for name, index in slices.items():
            if name not in before_state:
                continue
            saved[name] = named[name][index].detach().clone()
            named[name][index].copy_(before_state[name][index].to(device=device, dtype=named[name].dtype))
    try:
        return float(teacher_forced_nll(model, tokenizer, records, model_cfg, device)["token_nll"])
    finally:
        with torch.no_grad():
            for name, index in slices.items():
                if name in saved:
                    named[name][index].copy_(saved[name])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_sequence(sequence_name: str, device: torch.device) -> dict[str, Any]:
    phase_cfg = json.loads((RESULT_ROOT / "phase7a_resolved_config.json").read_text()) if (RESULT_ROOT / "phase7a_resolved_config.json").exists() else None
    baseline_cfg = load_project_config(PROJECT_ROOT / "configs" / "continual_baseline_full.yaml")
    phase_cfg = phase_cfg or json.loads((RESULT_ROOT / "phase7a_config.json").read_text())
    sequence = list(phase_cfg["sequences"][sequence_name])
    model_path = resolved_model_path(baseline_cfg)
    manifest_path = Path(baseline_cfg["dataset"]["processed_dir"]) / "data_manifest.json"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_cfg = baseline_cfg["model"]
    access_log: list[str] = []
    component_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    rollback_rows: list[dict[str, Any]] = []
    run_dir = RESULT_ROOT / sequence_name / "C1"
    training = json.loads((run_dir / "metrics_training.json").read_text(encoding="utf-8"))

    for stage_index in range(2, len(sequence) + 1):
        learned_task = sequence[stage_index - 1]
        before_task = sequence[stage_index - 2]
        before_path = Path(training["task_training"][before_task]["checkpoint"])
        after_path = Path(training["task_training"][learned_task]["checkpoint"])
        importance_path = run_dir / "importance" / learned_task / "importance_scores.csv"
        importance_lookup = _load_importance(importance_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(after_path), local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        post_state = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
        before_state = load_file(str(before_path / "model.safetensors"), device="cpu")
        missing = [name for name in post_state if name not in before_state]
        if missing:
            raise KeyError(f"Pre-task checkpoint misses parameters: {missing[:5]}")
        deltas = {name: post_state[name] - before_state[name] for name in post_state}
        layer_count = int(model.config.num_hidden_layers)
        attention_heads = int(model.config.num_attention_heads)
        kv_heads = int(getattr(model.config, "num_key_value_heads", attention_heads))
        selected_layers = sorted(
            {layer for kind, layer, unit in importance_lookup if kind in {"attention_head", "ffn_neuron"}}
        )
        ffn_layers = sorted({layer for kind, layer, unit in importance_lookup if kind == "ffn_neuron"})
        head_layers = sorted({layer for kind, layer, unit in importance_lookup if kind == "attention_head"})
        for old_task in sequence[: stage_index - 1]:
            dev_records = load_phase7a_task_split(manifest_path, old_task, "dev", access_log)
            gradients, post_nll = _dev_nll_gradient(model, tokenizer, dev_records, model_cfg, device)
            local_rows: list[dict[str, Any]] = []
            local_slices: dict[str, dict[str, tuple[slice, ...]]] = {}
            for layer in range(layer_count):
                for kind in ("layer_attention", "layer_ffn"):
                    row, slices = _component_row(
                        model,
                        kind,
                        layer,
                        None,
                        importance_lookup.get((kind, layer, None)),
                        gradients,
                        deltas,
                    )
                    local_rows.append(row)
                    local_slices[row["component_id"]] = slices
            for layer in ffn_layers:
                for unit in range(int(model.config.intermediate_size)):
                    kind = "ffn_neuron"
                    row, slices = _component_row(
                        model,
                        kind,
                        layer,
                        unit,
                        importance_lookup.get((kind, layer, unit)),
                        gradients,
                        deltas,
                    )
                    local_rows.append(row)
                    local_slices[row["component_id"]] = slices
            for layer in head_layers:
                for unit in range(attention_heads):
                    kind = "attention_head"
                    row, slices = _component_row(
                        model,
                        kind,
                        layer,
                        unit,
                        importance_lookup.get((kind, layer, unit)),
                        gradients,
                        deltas,
                    )
                    local_rows.append(row)
                    local_slices[row["component_id"]] = slices
                for unit in range(kv_heads):
                    kind = "kv_group"
                    row, slices = _component_row(model, kind, layer, unit, None, gradients, deltas)
                    local_rows.append(row)
                    local_slices[row["component_id"]] = slices
            for row in local_rows:
                row.update(
                    {
                        "sequence": sequence_name,
                        "stage_index": stage_index,
                        "learned_task": learned_task,
                        "old_task": old_task,
                        "learned_task_type": baseline_cfg["dataset"]["task_types"][learned_task]["type"],
                        "old_task_type": baseline_cfg["dataset"]["task_types"][old_task]["type"],
                        "post_update_old_dev_nll": post_nll,
                    }
                )
                component_rows.append(row)

            for kind in ("layer_attention", "layer_ffn", "attention_head", "ffn_neuron"):
                selected = [row for row in local_rows if row["component_type"] == kind and row.get("importance") is not None]
                correlation_rows.append(
                    {
                        "sequence": sequence_name,
                        "stage_index": stage_index,
                        "learned_task": learned_task,
                        "old_task": old_task,
                        "component_type": kind,
                        **rank_correlations(selected),
                    }
                )

            for kind in ("attention_head", "ffn_neuron"):
                candidates = [
                    row
                    for row in local_rows
                    if row["component_type"] == kind and row.get("importance") is not None
                ]
                high, low = select_discordant_components(candidates)
                used = {row["component_id"] for row in (high, low) if row is not None}
                selections = [
                    ("high_importance_low_risk", high),
                    ("low_importance_high_risk", low),
                ]
                for role, target in selections:
                    if target is None:
                        continue
                    control = matched_control(
                        candidates,
                        target,
                        exclude_ids=used,
                        seed=int(phase_cfg["seed"]) + stage_index + len(rollback_rows),
                    )
                    if control is None:
                        chosen = [(role, target, None)]
                    else:
                        used.add(control["component_id"])
                        chosen = [(role, target, None), ("matched_random_control", control, target)]
                    for selection_role, candidate, matched_to in chosen:
                        slices = local_slices[candidate["component_id"]]
                        rollback_nll = _rollback_nll(
                            model,
                            before_state,
                            slices,
                            tokenizer,
                            dev_records,
                            model_cfg,
                            device,
                        )
                        rollback_rows.append(
                            {
                                "sequence": sequence_name,
                                "stage_index": stage_index,
                                "learned_task": learned_task,
                                "old_task": old_task,
                                "component_type": kind,
                                "selection_role": selection_role,
                                "component_id": candidate["component_id"],
                                "importance": candidate["importance"],
                                "first_order_preservation_risk": candidate["risk"],
                                "positive_first_order_preservation_risk": candidate["positive_risk"],
                                "update_norm": candidate["update_norm"],
                                "parameter_count": candidate["parameter_count"],
                                "matched_to_component": matched_to["component_id"] if matched_to else None,
                                "matched_parameter_count": matched_to["parameter_count"] if matched_to else None,
                                "matched_update_norm_relative_gap": abs(
                                    float(candidate["update_norm"]) - float(matched_to["update_norm"])
                                ) / max(float(candidate["update_norm"]), 1e-30)
                                if matched_to
                                else None,
                                "post_update_old_dev_nll": post_nll,
                                "rollback_old_dev_nll": rollback_nll,
                                "exact_rollback_preservation_risk": post_nll - rollback_nll,
                                "positive_exact_rollback_preservation_risk": positive_risk_part(post_nll - rollback_nll),
                                "dev_examples": len(dev_records),
                            }
                        )
            model.zero_grad(set_to_none=True)
            del gradients
            gc.collect()
            torch.cuda.empty_cache()
        del model, before_state, post_state, deltas
        gc.collect()
        torch.cuda.empty_cache()

    if any("/test" in path.casefold() or "\\test" in path.casefold() for path in access_log):
        raise RuntimeError("Functional-risk analysis opened a test split path")
    return {
        "sequence": sequence_name,
        "component_rows": component_rows,
        "correlation_rows": correlation_rows,
        "rollback_rows": rollback_rows,
        "opened_split_paths": access_log,
        "test_split_files_opened": [],
        "official_parsp": official_repo_metadata(),
    }


def _plot_importance_risk(rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for axis, kind, title in (
        (axes[0], "attention_head", "Attention head"),
        (axes[1], "ffn_neuron", "FFN neuron"),
    ):
        selected = [row for row in rows if row["component_type"] == kind and row.get("importance") is not None]
        axis.scatter(
            [row["importance"] for row in selected],
            [row["risk"] for row in selected],
            s=5,
            alpha=0.25,
            c=[row["stage_index"] for row in selected],
            cmap="viridis",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("PaRSP functional importance (log-probability gain)")
        axis.set_ylabel("Old-task dev NLL first-order risk (g · Δθ)")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "functional_importance_vs_preservation_risk.png")
    plt.close(fig)


def _read_phase_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / "configs" / "phase7a.yaml").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", choices=["S1", "S2", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for offline Qwen3-0.6B dev diagnostics")
    device = torch.device(args.device)
    phase_cfg = _read_phase_config()
    write_json(RESULT_ROOT / "phase7a_config.json", phase_cfg)
    sequences = ["S1", "S2"] if args.sequence == "both" else [args.sequence]
    all_components, all_correlations, all_rollbacks, access_log = [], [], [], []
    for index, sequence in enumerate(sequences):
        local_device = device if args.sequence != "both" else torch.device(f"cuda:{index}")
        summary = analyze_sequence(sequence, local_device)
        all_components.extend(summary["component_rows"])
        all_correlations.extend(summary["correlation_rows"])
        all_rollbacks.extend(summary["rollback_rows"])
        access_log.extend(summary["opened_split_paths"])
        write_json(RESULT_ROOT / sequence / "functional_risk_analysis.json", {key: value for key, value in summary.items() if key not in {"component_rows", "correlation_rows", "rollback_rows"}})
    _write_rows(RESULT_ROOT / "functional_importance_preservation.csv", all_components)
    _write_rows(RESULT_ROOT / "functional_importance_preservation_correlations.csv", all_correlations)
    _write_rows(RESULT_ROOT / "functional_importance_rollback_controls.csv", all_rollbacks)
    _plot_importance_risk(all_components)
    if any("/test" in path.casefold() or "\\test" in path.casefold() for path in access_log):
        raise RuntimeError("Functional-risk analysis opened a test split path")
    write_json(
        RESULT_ROOT / "functional_importance_preservation_summary.json",
        {
            "sequences": sequences,
            "correlation_groups": len(all_correlations),
            "component_measurements": len(all_components),
            "exact_rollback_interventions": len(all_rollbacks),
            "opened_split_paths": access_log,
            "evaluation_splits_loaded": ["dev"],
            "test_split_files_opened": [],
            "risk_definition": "old-task dev token-NLL gradient at the post-task checkpoint dotted with the actual task parameter delta; exact rollback is post-update NLL minus NLL after restoring one component and matched controls",
        },
    )


if __name__ == "__main__":
    main()
