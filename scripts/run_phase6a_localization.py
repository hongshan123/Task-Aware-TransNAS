"""Phase 6A component-level activation replacement and parameter rollback.

All model loads are local-only. The script reads train/dev manifests only and
never loads a test split or performs an optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from phase4_common import prepare_context  # noqa: E402
from phase5_common import cleanup_model, load_checkpoint  # noqa: E402
from phase5b_common import generated_label_metrics  # noqa: E402
from run_formal_baseline import (  # noqa: E402
    answer_tokens_for,
    load_model,
    parse_yes_no,
    rouge_l_f1,
    safe_prompt,
    token_f1,
)


OUT_ROOT = storage_path("results") / "phase6a"
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase6a_component_localization.yaml"
STAGE2 = storage_path("results") / "phase5b" / "checkpoints" / "historical_boundary_replay" / "A0" / "S1" / "seed_10" / "after_2_task1603_smcalflow_sentence_generation"
STAGE3 = storage_path("results") / "phase6a" / "checkpoints" / "after_3_task1501_dstc3_answer_generation"
FINAL = storage_path("results") / "phase4b" / "checkpoints" / "A0" / "S1" / "seed_10"
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
GENERATION_TASKS = set(TASKS[:2])
COMPONENTS = ("attention_o_proj_output", "mlp_output", "transformer_block_output")
COMP_TO_SHORT = {
    "attention_o_proj_output": "attention",
    "mlp_output": "mlp",
    "transformer_block_output": "block",
    "token_embedding_output": "embedding",
    "final_norm_output": "norm",
    "lm_head_logits_upper_bound": "logits",
}
MAX_NEW_TOKENS = 64
MAX_SEQUENCE_LENGTH = 512
MAX_TARGET_TOKENS = 64
DEVICE = torch.device("cuda:0")


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha_ids(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps([str(row["instance_id"]) for row in rows], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_checkpoint(model_path: Path, checkpoint_path: Path | None = None):
    return load_checkpoint(model_path, checkpoint_path, architecture="A0")


def _prepare_records(tokenizer: Any, task_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packed = []
    for row in rows:
        prompt_ids, input_used, diagnostics = safe_prompt(tokenizer, row, MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS)
        target_ids = answer_tokens_for(tokenizer, row, input_used, str(row["targets"][0]))
        if not target_ids:
            raise RuntimeError(f"Empty encoded target for {task_id}/{row['instance_id']}")
        if len(target_ids) >= MAX_TARGET_TOKENS:
            target_ids = target_ids[: MAX_TARGET_TOKENS - 1] + [int(tokenizer.eos_token_id)]
        if len(prompt_ids) + len(target_ids) > MAX_SEQUENCE_LENGTH:
            raise RuntimeError(f"Packed prompt and target exceed {MAX_SEQUENCE_LENGTH}: {task_id}/{row['instance_id']}")
        packed.append({
            "task_id": task_id,
            "record": row,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "full_ids": prompt_ids + target_ids,
            "prompt_diagnostics": diagnostics,
        })
    return packed


def _component_modules(model: Any):
    modules = [("token_embedding_output", None, model.model.embed_tokens)]
    for layer_id, layer in enumerate(model.model.layers):
        modules.extend([
            ("attention_o_proj_output", layer_id, layer.self_attn.o_proj),
            ("mlp_output", layer_id, layer.mlp),
            ("transformer_block_output", layer_id, layer),
        ])
    modules.extend([
        ("final_norm_output", None, model.model.norm),
        ("lm_head_logits_upper_bound", None, model.lm_head),
    ])
    return modules


def _out_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unexpected hooked module output type: {type(output)}")


def _replace_out(output: Any, tensor: torch.Tensor):
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    raise TypeError(f"Unexpected hooked module output type: {type(output)}")


class ActivationHooks:
    """Capture a reference suffix or patch per-row candidate suffixes."""

    def __init__(self, model: Any, mode: str):
        self.mode = mode
        self.start = 0
        self.bank: dict[tuple[str, int | None], torch.Tensor] = {}
        self.candidates: list[dict[str, Any]] = []
        self.handles = []
        for component, layer, module in _component_modules(model):
            self.handles.append(module.register_forward_hook(self._hook(component, layer)))

    def _hook(self, component: str, layer: int | None):
        def hook(_module, _args, output):
            values = _out_tensor(output)
            key = (component, layer)
            if self.mode == "capture":
                start = 0 if component == "lm_head_logits_upper_bound" else self.start
                self.bank[key] = values[0, start:].detach().to(device="cpu", dtype=torch.float16).contiguous()
                return output
            if self.mode != "patch":
                raise RuntimeError(f"Invalid activation hook mode {self.mode}")
            if not self.candidates:
                return output
            changed = None
            logits = component == "lm_head_logits_upper_bound"
            start = 0 if logits else self.start
            for row_index, candidate in enumerate(self.candidates):
                if candidate.get("operation") == "zero_no_op":
                    continue
                candidate_components = candidate.get("components", [candidate["component"]])
                if component not in candidate_components:
                    continue
                layers = candidate.get("layers")
                if layers is not None and layer not in layers:
                    continue
                if layers is None and layer is not None:
                    continue
                source = self.bank[(component, layer)].to(device=values.device, dtype=values.dtype)
                if changed is None:
                    changed = values.clone()
                if candidate.get("operation", "swap") == "swap":
                    changed[row_index, start:] = source
                elif candidate["operation"] == "random_delta_same_l2":
                    current = values[row_index, start:].float()
                    delta = source.float() - current
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(int(candidate["noise_seed"]))
                    noise = torch.randn(tuple(delta.shape), generator=generator, dtype=torch.float32).to(delta.device)
                    noise *= delta.norm() / noise.norm().clamp_min(1e-12)
                    changed[row_index, start:] = (current + noise).to(values.dtype)
                elif candidate["operation"] == "reference_norm_random":
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(int(candidate["noise_seed"]))
                    noise = torch.randn(tuple(source.shape), generator=generator, dtype=torch.float32).to(values.device)
                    noise *= source.float().norm() / noise.norm().clamp_min(1e-12)
                    changed[row_index, start:] = noise.to(values.dtype)
                else:
                    raise ValueError(f"Unknown activation operation: {candidate['operation']}")
            return output if changed is None else _replace_out(output, changed)

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _positions(pack: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.arange(len(pack["target_ids"]), device=device) + len(pack["prompt_ids"]) - 1


def _source_activation_bank(source: Any, pack: dict[str, Any]) -> dict[tuple[str, int | None], torch.Tensor]:
    hooks = ActivationHooks(source, "capture")
    hooks.start = len(pack["prompt_ids"]) - 1
    x = torch.tensor([pack["full_ids"]], dtype=torch.long, device=DEVICE)
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            body = source.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
            selected = body.last_hidden_state.index_select(1, _positions(pack, DEVICE))
            _ = source.lm_head(selected)
        return hooks.bank
    finally:
        hooks.close()


def _score_logits(logits: torch.Tensor, packs: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    output = []
    vocab_size = logits.shape[-1]
    for row_index, pack in enumerate(packs):
        labels = torch.tensor(pack["target_ids"], dtype=torch.long, device=logits.device)
        row_logits = logits[row_index, : labels.numel()].float()
        nll = F.cross_entropy(row_logits, labels, reduction="sum")
        first_logp = F.log_softmax(row_logits[0], dim=-1)
        yes_ids = set(tokenizer.encode("Yes", add_special_tokens=False)) | set(tokenizer.encode(" Yes", add_special_tokens=False))
        no_ids = set(tokenizer.encode("No", add_special_tokens=False)) | set(tokenizer.encode(" No", add_special_tokens=False))
        yes_ids.discard(tokenizer.unk_token_id)
        no_ids.discard(tokenizer.unk_token_id)
        yes_values = [int(value) for value in yes_ids if 0 <= int(value) < vocab_size]
        no_values = [int(value) for value in no_ids if 0 <= int(value) < vocab_size]
        margin = None
        if yes_values and no_values:
            margin = float((torch.logsumexp(first_logp[yes_values], dim=0) - torch.logsumexp(first_logp[no_values], dim=0)).item())
        output.append({
            "nll_sum": float(nll.item()),
            "token_count": int(labels.numel()),
            "first_reference_token_log_probability": float(first_logp[int(labels[0].item())].item()),
            "yes_minus_no_log_probability_margin": margin,
        })
    return output


def _baseline_logits(model: Any, pack: dict[str, Any]) -> torch.Tensor:
    x = torch.tensor([pack["full_ids"]], dtype=torch.long, device=DEVICE)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        body = model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True)
        selected = body.last_hidden_state.index_select(1, _positions(pack, DEVICE))
        return model.lm_head(selected).detach()


def _patch_candidate_logits(
    model: Any,
    pack: dict[str, Any],
    candidate_chunk: list[dict[str, Any]],
    bank: dict[tuple[str, int | None], torch.Tensor],
    hooks: ActivationHooks,
) -> torch.Tensor:
    hooks.start = len(pack["prompt_ids"]) - 1
    hooks.bank = bank
    hooks.candidates = candidate_chunk
    x = torch.tensor([pack["full_ids"]], dtype=torch.long, device=DEVICE).expand(len(candidate_chunk), -1)
    attention = torch.ones_like(x)
    positions = _positions(pack, DEVICE)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        body = model.model(input_ids=x, attention_mask=attention, use_cache=False, return_dict=True)
        selected = body.last_hidden_state.index_select(1, positions)
        logits = model.lm_head(selected)
    hooks.candidates = []
    return logits.detach()


def _candidate(component: str, layers: list[int] | None, name: str, scope: str = "global", operation: str = "swap"):
    return {
        "candidate_id": name,
        "component": component,
        "layers": layers,
        "scope": scope,
        "operation": operation,
    }


def activation_candidates(num_layers: int) -> list[dict[str, Any]]:
    candidates = []
    for component in COMPONENTS:
        short = COMP_TO_SHORT[component]
        for layer in range(num_layers):
            candidates.append(_candidate(component, [layer], f"{short}_layer_{layer:02d}", "single_layer"))
        for width in (2, 4, 8, 14, num_layers):
            for start in range(num_layers - width + 1):
                end = start + width - 1
                candidates.append(_candidate(component, list(range(start, end + 1)), f"{short}_group_{width:02d}_{start:02d}_{end:02d}", "contiguous_group"))
    for component in ("token_embedding_output", "final_norm_output", "lm_head_logits_upper_bound"):
        candidates.append(_candidate(component, None, COMP_TO_SHORT[component] + "_global", "global"))
    candidates.append(_candidate(COMPONENTS[0], [13], "control_zero_no_op", "control", "zero_no_op"))
    for component in COMPONENTS:
        short = COMP_TO_SHORT[component]
        for operation, suffix in (("random_delta_same_l2", "random_delta_l2"), ("reference_norm_random", "reference_norm_random")):
            candidates.append({
                **_candidate(component, [13], f"control_{short}_layer_13_{suffix}", "control", operation),
                "noise_seed": 314159 + (13 * 1009) + len(candidates),
            })
    return candidates


def run_activation_screen(model_path: Path, tokenizer: Any, train: dict[str, list[dict[str, Any]]], dev: dict[str, list[dict[str, Any]]], model_cfg: dict[str, Any], chunk_size: int):
    path = OUT_ROOT / "activation_screen.jsonl"
    if path.exists():
        path.unlink()
    target = _model_checkpoint(model_path, FINAL)
    layers = int(target.config.num_hidden_layers)
    candidates = activation_candidates(layers)
    target_hooks = ActivationHooks(target, "patch")
    all_rows = []
    try:
        for task_id in TASKS:
            source_path = STAGE2 if task_id in GENERATION_TASKS else STAGE3
            source = _model_checkpoint(model_path, source_path)
            rows = dev[task_id][5:15]
            packed = _prepare_records(tokenizer, task_id, rows)
            print(f"activation screen {task_id}: {len(packed)} dev rows, {len(candidates)} candidates, source={source_path.name}", flush=True)
            for example_i, item in enumerate(packed, start=1):
                bank = _source_activation_bank(source, item)
                baseline = _score_logits(_baseline_logits(target, item), [item], tokenizer)[0]
                if example_i == 1:
                    print(f"  reference capture shapes: {len(bank)} components/layers; baseline NLL={baseline['nll_sum']/baseline['token_count']:.4f}", flush=True)
                all_rows.append({
                    "candidate_id": "baseline", "task_id": task_id,
                    "instance_id": str(item["record"]["instance_id"]),
                    "nll_sum": baseline["nll_sum"], "token_count": baseline["token_count"],
                    "mean_token_nll": baseline["nll_sum"] / baseline["token_count"],
                    "first_reference_token_log_probability": baseline["first_reference_token_log_probability"],
                    "yes_minus_no_log_probability_margin": baseline["yes_minus_no_log_probability_margin"],
                    "candidate_type": "baseline", "scope": "baseline",
                })
                for start in range(0, len(candidates), chunk_size):
                    chunk = candidates[start : start + chunk_size]
                    # A paired unpatched row in every candidate batch controls
                    # small batch-kernel roundoff differences versus B=1 runs.
                    batch_baseline = {
                        "candidate_id": "within_batch_baseline",
                        "component": COMPONENTS[0], "layers": None,
                        "operation": "zero_no_op", "scope": "control",
                    }
                    logits = _patch_candidate_logits(target, item, [*chunk, batch_baseline], bank, target_hooks)
                    scores = _score_logits(logits, [item] * (len(chunk) + 1), tokenizer)
                    paired_baseline = scores[-1]
                    for candidate, score in zip(chunk, scores):
                        all_rows.append({
                            "candidate_id": candidate["candidate_id"], "task_id": task_id,
                            "instance_id": str(item["record"]["instance_id"]),
                            "nll_sum": score["nll_sum"], "token_count": score["token_count"],
                            "mean_token_nll": score["nll_sum"] / score["token_count"],
                            "first_reference_token_log_probability": score["first_reference_token_log_probability"],
                            "yes_minus_no_log_probability_margin": score["yes_minus_no_log_probability_margin"],
                            "candidate_type": candidate["component"], "scope": candidate["scope"],
                            "operation": candidate.get("operation", "swap"),
                            "within_batch_baseline_mean_token_nll": paired_baseline["nll_sum"] / paired_baseline["token_count"],
                        })
                    del logits
                    if (start // chunk_size + 1) % 10 == 0 or start + chunk_size >= len(candidates):
                        print(f"  {example_i}/{len(packed)} examples; {min(start + chunk_size, len(candidates))}/{len(candidates)} candidates", flush=True)
                del bank
            cleanup_model(source)
            source = None
        _jsonl_write(path, all_rows)
    finally:
        target_hooks.close()
        cleanup_model(target)
    _write_activation_summary(path, candidates)


def _write_activation_summary(path: Path, candidates: list[dict[str, Any]] | None = None):
    rows = _jsonl_read(path)
    aggregates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        aggregates.setdefault((row["task_id"], row["candidate_id"]), []).append(row)
    output = []
    for (task_id, candidate_id), values in aggregates.items():
        baseline_by_id = {
            row["instance_id"]: row for row in rows
            if row["task_id"] == task_id and row["candidate_id"] == "baseline"
        }
        nlls = [value["mean_token_nll"] for value in values]
        base = [
            value.get("within_batch_baseline_mean_token_nll", baseline_by_id[value["instance_id"]]["mean_token_nll"])
            for value in values
        ]
        delta = [b - a for b, a in zip(base, nlls)]
        output.append({
            "task_id": task_id,
            "candidate_id": candidate_id,
            "candidate_type": values[0]["candidate_type"],
            "scope": values[0]["scope"],
            "operation": values[0].get("operation", "baseline"),
            "examples": len(values),
            "mean_token_nll": sum(nlls) / max(1, len(nlls)),
            "baseline_mean_token_nll": sum(base) / max(1, len(base)),
            "nll_improvement_vs_baseline": sum(delta) / max(1, len(delta)),
            "mean_first_reference_token_log_probability": sum(float(v["first_reference_token_log_probability"]) for v in values) / max(1, len(values)),
            "mean_yes_minus_no_log_probability_margin": (
                sum(float(v["yes_minus_no_log_probability_margin"]) for v in values if v["yes_minus_no_log_probability_margin"] is not None)
                / max(1, sum(v["yes_minus_no_log_probability_margin"] is not None for v in values))
            ),
        })
    output.sort(key=lambda row: (row["task_id"], -row["nll_improvement_vs_baseline"]))
    csv_path = path.with_name("activation_screen_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    selected = {}
    for task_id in TASKS:
        valid = [row for row in output if row["task_id"] == task_id and row["candidate_id"] != "baseline" and row["scope"] not in ("control", "baseline")]
        selected[task_id] = sorted(valid, key=lambda row: (-row["nll_improvement_vs_baseline"], row["candidate_id"]))[:5]
    _json_dump(OUT_ROOT / "activation_screen_selection.json", {
        "ranking_metric": "mean per-example token NLL improvement vs unpatched final model on dev indices 5-14; larger is better",
        "screen_ids_sha256": {task: _sha_ids([{"instance_id": r["instance_id"]} for r in _jsonl_read(path) if r["task_id"] == task and r["candidate_id"] == "baseline"]) for task in TASKS},
        "selected_top5_per_task": selected,
        "all_candidates": output,
        "test_split_loaded": False,
    })
    print(f"wrote {csv_path} and activation_screen_selection.json", flush=True)


def _param_candidates(num_layers: int) -> list[dict[str, Any]]:
    candidates = []
    types = ("attention", "mlp", "norm")
    for comp in types:
        for layer in range(num_layers):
            candidates.append({"candidate_id": f"{comp}_layer_{layer:02d}", "component": comp, "layers": [layer], "scope": "single_layer"})
        for width in (2, 4, 8, 14, num_layers):
            for start in range(num_layers - width + 1):
                end = start + width - 1
                candidates.append({"candidate_id": f"{comp}_group_{width:02d}_{start:02d}_{end:02d}", "component": comp, "layers": list(range(start, end + 1)), "scope": "contiguous_group"})
    for comp in ("attention_global", "mlp_global", "attention_plus_mlp_global", "norm_global", "embedding_lm_head_global", "backbone_global"):
        candidates.append({"candidate_id": comp, "component": comp.replace("_global", ""), "layers": None, "scope": "global"})
    projection_specs = {
        "attention_q_proj": ("attention", "q_proj"),
        "attention_k_proj": ("attention", "k_proj"),
        "attention_v_proj": ("attention", "v_proj"),
        "attention_o_proj": ("attention", "o_proj"),
        "mlp_gate_proj": ("mlp", "gate_proj"),
        "mlp_up_proj": ("mlp", "up_proj"),
        "mlp_down_proj": ("mlp", "down_proj"),
    }
    for component, (_parent, projection) in projection_specs.items():
        for layer in range(num_layers):
            candidates.append({
                "candidate_id": f"{component}_layer_{layer:02d}",
                "component": component, "projection": projection,
                "layers": [layer], "scope": "single_projection_layer",
            })
        candidates.append({
            "candidate_id": f"{component}_global", "component": component,
            "projection": projection, "layers": None, "scope": "global_projection",
        })
    return candidates


def _param_keys(named_parameters: dict[str, torch.nn.Parameter], candidate: dict[str, Any]) -> list[str]:
    comp = candidate["component"]
    layers = candidate.get("layers")
    keys = []
    projection_specs = {
        "attention_q_proj": ("attention", "q_proj"),
        "attention_k_proj": ("attention", "k_proj"),
        "attention_v_proj": ("attention", "v_proj"),
        "attention_o_proj": ("attention", "o_proj"),
        "mlp_gate_proj": ("mlp", "gate_proj"),
        "mlp_up_proj": ("mlp", "up_proj"),
        "mlp_down_proj": ("mlp", "down_proj"),
    }
    for name in named_parameters:
        match = re.match(r"^model\.layers\.(\d+)\.", name)
        layer = int(match.group(1)) if match else None
        if comp in projection_specs and layer is not None:
            if layers is not None and layer not in layers:
                continue
            parent, projection = projection_specs[comp]
            if parent == "attention" and re.search(rf"\.self_attn\.{projection}\.(weight|bias)$", name):
                keys.append(name)
            elif parent == "mlp" and re.search(rf"\.mlp\.{projection}\.(weight|bias)$", name):
                keys.append(name)
        elif comp in ("attention", "mlp", "norm") and layer is not None:
            if layers is not None and layer not in layers:
                continue
            if comp == "attention" and re.search(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$", name):
                keys.append(name)
            elif comp == "mlp" and re.search(r"\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$", name):
                keys.append(name)
            elif comp == "norm" and re.search(r"\.(input_layernorm|post_attention_layernorm)\.(weight|bias)$", name):
                keys.append(name)
        elif comp == "norm" and layer is None and name in ("model.norm.weight", "model.norm.bias"):
            keys.append(name)
        elif comp == "embedding_lm_head" and name == "model.embed_tokens.weight":
            keys.append(name)
        elif comp == "attention_plus_mlp" and layer is not None:
            if re.search(r"\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$", name) or re.search(r"\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$", name):
                keys.append(name)
        elif comp == "backbone" and name.startswith("model."):
            keys.append(name)
    return keys


def _pack_batch(packs: list[dict[str, Any]], tokenizer: Any):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(len(pack["full_ids"]) for pack in packs)
    ids = torch.full((len(packs), width), int(pad_id), dtype=torch.long, device=DEVICE)
    mask = torch.zeros_like(ids)
    for row_index, pack in enumerate(packs):
        length = len(pack["full_ids"])
        ids[row_index, :length] = torch.tensor(pack["full_ids"], dtype=torch.long, device=DEVICE)
        mask[row_index, :length] = 1
    return ids, mask


def _score_pack_batch(model: Any, packs: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    ids, attention = _pack_batch(packs, tokenizer)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        body = model.model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
        max_targets = max(len(pack["target_ids"]) for pack in packs)
        hidden = torch.zeros((len(packs), max_targets, body.last_hidden_state.shape[-1]), dtype=body.last_hidden_state.dtype, device=DEVICE)
        for row_index, pack in enumerate(packs):
            positions = _positions(pack, DEVICE)
            hidden[row_index, :len(positions)] = body.last_hidden_state[row_index].index_select(0, positions)
        logits = model.lm_head(hidden)
    output = []
    for row_index, pack in enumerate(packs):
        targets = torch.tensor(pack["target_ids"], dtype=torch.long, device=DEVICE)
        selected = logits[row_index, :targets.numel()].float()
        loss = F.cross_entropy(selected, targets, reduction="sum")
        first_logp = F.log_softmax(selected[0], dim=-1)
        yes_ids = set(tokenizer.encode("Yes", add_special_tokens=False)) | set(tokenizer.encode(" Yes", add_special_tokens=False))
        no_ids = set(tokenizer.encode("No", add_special_tokens=False)) | set(tokenizer.encode(" No", add_special_tokens=False))
        yes_ids.discard(tokenizer.unk_token_id)
        no_ids.discard(tokenizer.unk_token_id)
        yes_ids = [int(value) for value in yes_ids if 0 <= int(value) < first_logp.numel()]
        no_ids = [int(value) for value in no_ids if 0 <= int(value) < first_logp.numel()]
        margin = float((torch.logsumexp(first_logp[yes_ids], 0) - torch.logsumexp(first_logp[no_ids], 0)).item()) if yes_ids and no_ids else None
        output.append({
            "nll_sum": float(loss.item()),
            "token_count": int(targets.numel()),
            "mean_token_nll": float(loss.item() / targets.numel()),
            "first_reference_token_log_probability": float(first_logp[int(targets[0].item())].item()),
            "yes_minus_no_log_probability_margin": margin,
        })
    del ids, attention, body, hidden, logits
    return output


def run_parameter_screen(model_path: Path, tokenizer: Any, train: dict[str, list[dict[str, Any]]], dev: dict[str, list[dict[str, Any]]]):
    path = OUT_ROOT / "parameter_screen.jsonl"
    if path.exists():
        path.unlink()
    target = _model_checkpoint(model_path, FINAL)
    params = dict(target.named_parameters())
    candidates = _param_candidates(int(target.config.num_hidden_layers))
    reference_states = {
        "stage2": load_file(str(STAGE2 / "model.safetensors"), device="cpu"),
        "stage3": load_file(str(STAGE3 / "model.safetensors"), device="cpu"),
    }
    all_rows = []
    try:
        packs_by_task = {
            task_id: _prepare_records(tokenizer, task_id, dev[task_id][15:25])
            for task_id in TASKS
        }
        baselines = {task: _score_pack_batch(target, packs_by_task[task], tokenizer) for task in TASKS}
        for task_id in TASKS:
            for pack, score in zip(packs_by_task[task_id], baselines[task_id]):
                all_rows.append({"candidate_id": "baseline", "task_id": task_id, "instance_id": str(pack["record"]["instance_id"]), **score, "scope": "baseline", "reference_stage": "final"})
        for index, candidate in enumerate(candidates, start=1):
            key_names = _param_keys(params, candidate)
            if not key_names:
                print(f"skip empty parameter group: {candidate['candidate_id']}", flush=True)
                continue
            for task_id in TASKS:
                source_stage = "stage2" if task_id in GENERATION_TASKS else "stage3"
                source_state = reference_states[source_stage]
                missing = [name for name in key_names if name not in source_state]
                if missing:
                    raise RuntimeError(f"Reference checkpoint missing expected parameter names: {missing[:8]}")
                backup = {name: params[name].detach().clone() for name in key_names}
                start_time = time.perf_counter()
                try:
                    with torch.no_grad():
                        for name in key_names:
                            params[name].copy_(source_state[name].to(device=DEVICE, dtype=params[name].dtype))
                    scores = _score_pack_batch(target, packs_by_task[task_id], tokenizer)
                finally:
                    with torch.no_grad():
                        for name in key_names:
                            params[name].copy_(backup[name])
                    del backup
                state_delta_sq = 0.0
                state_count = 0
                for name in key_names:
                    difference = source_state[name].float() - params[name].detach().cpu().float()
                    state_delta_sq += float(difference.square().sum().item())
                    state_count += difference.numel()
                delta_norm = math.sqrt(state_delta_sq)
                elapsed = time.perf_counter() - start_time
                for pack, score in zip(packs_by_task[task_id], scores):
                    baseline_score = baselines[task_id][packs_by_task[task_id].index(pack)]
                    all_rows.append({
                        "candidate_id": candidate["candidate_id"], "task_id": task_id,
                        "instance_id": str(pack["record"]["instance_id"]), **score,
                        "scope": candidate["scope"], "reference_stage": source_stage,
                        "parameter_count": state_count,
                        "rollback_fraction_of_model": state_count / sum(parameter.numel() for parameter in params.values()),
                        "update_l2_norm": delta_norm,
                        "update_rms": delta_norm / math.sqrt(max(1, state_count)),
                        "condition_elapsed_seconds": elapsed,
                        "nll_improvement_vs_baseline": baseline_score["mean_token_nll"] - score["mean_token_nll"],
                    })
            if index % 10 == 0 or index == len(candidates):
                print(f"parameter screen {index}/{len(candidates)}: {candidate['candidate_id']}", flush=True)
        _jsonl_write(path, all_rows)
    finally:
        cleanup_model(target)
    _write_parameter_summary(path)


def _write_parameter_summary(path: Path):
    rows = _jsonl_read(path)
    aggregates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        aggregates.setdefault((row["task_id"], row["candidate_id"]), []).append(row)
    output = []
    for (task_id, candidate_id), values in aggregates.items():
        baselines = {(row["task_id"], row["instance_id"]): row for row in rows if row["candidate_id"] == "baseline"}
        nlls = [row["mean_token_nll"] for row in values]
        base = [baselines[(task_id, row["instance_id"])]["mean_token_nll"] for row in values]
        improvement = sum(b - a for a, b in zip(nlls, base)) / max(1, len(nlls))
        output.append({
            "task_id": task_id, "candidate_id": candidate_id,
            "scope": values[0]["scope"], "reference_stage": values[0]["reference_stage"],
            "examples": len(values), "mean_token_nll": sum(nlls) / max(1, len(nlls)),
            "baseline_mean_token_nll": sum(base) / max(1, len(base)),
            "nll_improvement_vs_baseline": improvement,
            "parameter_count": values[0].get("parameter_count", 0),
            "rollback_fraction_of_model": values[0].get("rollback_fraction_of_model", 0),
            "update_l2_norm": values[0].get("update_l2_norm", 0),
            "update_rms": values[0].get("update_rms", 0),
            "condition_elapsed_seconds": values[0].get("condition_elapsed_seconds", 0),
        })
    output.sort(key=lambda row: (row["task_id"], -row["nll_improvement_vs_baseline"]))
    csv_path = path.with_name("parameter_screen_summary.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    selected = {}
    for task_id in TASKS:
        valid = [row for row in output if row["task_id"] == task_id and row["candidate_id"] != "baseline"]
        selected[task_id] = sorted(valid, key=lambda row: (-row["nll_improvement_vs_baseline"], row["candidate_id"]))[:5]
    _json_dump(OUT_ROOT / "parameter_screen_selection.json", {
        "ranking_metric": "mean per-example token NLL improvement on dev indices 15-24; larger is better",
        "selected_top5_per_task": selected,
        "all_candidates": output,
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "tied_embedding_lm_head_restored_as_single_model.embed_tokens.weight": True,
    })
    print(f"wrote {csv_path} and parameter_screen_selection.json", flush=True)


def run_activation_synergy(model_path: Path, tokenizer: Any, dev: dict[str, list[dict[str, Any]]]):
    """Compare attention, MLP, their joint patch, and whole-block patch at one location."""
    selection = json.loads((OUT_ROOT / "activation_screen_selection.json").read_text(encoding="utf-8"))
    metadata = {row["candidate_id"]: row for row in activation_candidates(28)}
    selected_rows = selection["all_candidates"]
    target = _model_checkpoint(model_path, FINAL)
    hooks = ActivationHooks(target, "patch")
    output_rows = []
    try:
        for task_id in TASKS:
            internal = [
                row for row in selected_rows
                if row["task_id"] == task_id
                and row["scope"] not in ("control", "baseline")
                and row["candidate_type"] in COMPONENTS
            ]
            winner = max(internal, key=lambda row: row["nll_improvement_vs_baseline"])
            base = metadata[winner["candidate_id"]]
            layers = base.get("layers") or [13]
            candidates = [
                {"candidate_id": f"attention_at_{winner['candidate_id']}", "component": COMPONENTS[0], "layers": layers, "scope": "single_component"},
                {"candidate_id": f"mlp_at_{winner['candidate_id']}", "component": COMPONENTS[1], "layers": layers, "scope": "single_component"},
                {"candidate_id": f"attention_plus_mlp_at_{winner['candidate_id']}", "component": "attention_plus_mlp_output", "components": [COMPONENTS[0], COMPONENTS[1]], "layers": layers, "scope": "joint_attention_mlp"},
                {"candidate_id": f"block_at_{winner['candidate_id']}", "component": COMPONENTS[2], "layers": layers, "scope": "whole_block"},
            ]
            reference_path = STAGE2 if task_id in GENERATION_TASKS else STAGE3
            source = _model_checkpoint(model_path, reference_path)
            for split_name, start, end in (("screen", 5, 15), ("locked_confirmation", 25, 50)):
                packs = _prepare_records(tokenizer, task_id, dev[task_id][start:end])
                print(f"activation synergy {task_id}/{split_name}: layers={layers}, n={len(packs)}", flush=True)
                for pack in packs:
                    bank = _source_activation_bank(source, pack)
                    batch = [*candidates, {
                        "candidate_id": "within_batch_baseline", "component": COMPONENTS[0],
                        "layers": None, "operation": "zero_no_op", "scope": "control",
                    }]
                    logits = _patch_candidate_logits(target, pack, batch, bank, hooks)
                    scores = _score_logits(logits, [pack] * len(batch), tokenizer)
                    baseline = scores[-1]
                    for candidate, score in zip(candidates, scores[:-1]):
                        output_rows.append({
                            "task_id": task_id, "split": split_name,
                            "instance_id": str(pack["record"]["instance_id"]),
                            "location_candidate": winner["candidate_id"],
                            "candidate_id": candidate["candidate_id"],
                            "components": candidate.get("components", [candidate["component"]]),
                            "layers": layers,
                            "nll_sum": score["nll_sum"], "token_count": score["token_count"],
                            "mean_token_nll": score["nll_sum"] / score["token_count"],
                            "nll_improvement_vs_within_batch_baseline": baseline["nll_sum"] / baseline["token_count"] - score["nll_sum"] / score["token_count"],
                            "yes_minus_no_log_probability_margin": score["yes_minus_no_log_probability_margin"],
                        })
            cleanup_model(source)
            source = None
        _jsonl_write(OUT_ROOT / "activation_synergy.jsonl", output_rows)
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in output_rows:
            groups.setdefault((row["task_id"], row["split"], row["candidate_id"]), []).append(row)
        summary = []
        for (task_id, split_name, candidate_id), values in groups.items():
            summary.append({
                "task_id": task_id, "split": split_name, "candidate_id": candidate_id,
                "location_candidate": values[0]["location_candidate"],
                "components": "+".join(COMP_TO_SHORT.get(value, value) for value in values[0]["components"]),
                "layers": json.dumps(values[0]["layers"]), "examples": len(values),
                "mean_token_nll": sum(row["mean_token_nll"] for row in values) / len(values),
                "nll_improvement_vs_within_batch_baseline": sum(row["nll_improvement_vs_within_batch_baseline"] for row in values) / len(values),
            })
        csv_path = OUT_ROOT / "activation_synergy_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"wrote {csv_path}", flush=True)
    finally:
        hooks.close()
        cleanup_model(target)


def run_checkpoint_trajectory(model_path: Path, tokenizer: Any, dev: dict[str, list[dict[str, Any]]]):
    """Measure same locked-dev reference NLL across the retained task boundaries."""
    checkpoints = (("after_task1603", STAGE2), ("after_task1501", STAGE3), ("final_after_task1384", FINAL))
    raw = []
    summaries = []
    for checkpoint_name, checkpoint_path in checkpoints:
        model = _model_checkpoint(model_path, checkpoint_path)
        try:
            for task_id in TASKS:
                packs = _prepare_records(tokenizer, task_id, dev[task_id][25:50])
                scores = _score_pack_batch(model, packs, tokenizer)
                total_nll = sum(row["nll_sum"] for row in scores)
                token_count = sum(row["token_count"] for row in scores)
                margins = [row["yes_minus_no_log_probability_margin"] for row in scores if row["yes_minus_no_log_probability_margin"] is not None]
                summaries.append({
                    "checkpoint": checkpoint_name, "task_id": task_id,
                    "examples": len(packs), "teacher_forced_token_nll": total_nll / max(1, token_count),
                    "mean_per_example_nll": sum(row["mean_token_nll"] for row in scores) / max(1, len(scores)),
                    "mean_first_reference_token_log_probability": sum(row["first_reference_token_log_probability"] for row in scores) / max(1, len(scores)),
                    "mean_yes_minus_no_log_probability_margin": sum(margins) / max(1, len(margins)),
                })
                for pack, score in zip(packs, scores):
                    raw.append({
                        "checkpoint": checkpoint_name, "task_id": task_id,
                        "instance_id": str(pack["record"]["instance_id"]), **score,
                    })
            print(f"checkpoint trajectory {checkpoint_name}: all four tasks scored", flush=True)
        finally:
            cleanup_model(model)
            model = None
    _jsonl_write(OUT_ROOT / "checkpoint_trajectory.jsonl", raw)
    path = OUT_ROOT / "checkpoint_trajectory_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote {path}", flush=True)


class PairedActivationGenerator:
    """Greedy decode on target while source sees the same generated tokens."""

    def __init__(self, source: Any, target: Any, candidate: dict[str, Any]):
        self.source = source
        self.target = target
        self.candidate = candidate
        self.reference_values: dict[tuple[str, int | None], torch.Tensor] = {}
        self.handles = []
        for component, layer, module in _component_modules(source):
            if self._matches(component, layer):
                self.handles.append(module.register_forward_hook(self._source_hook(component, layer)))
        for component, layer, module in _component_modules(target):
            if self._matches(component, layer):
                self.handles.append(module.register_forward_hook(self._target_hook(component, layer)))

    def _matches(self, component: str, layer: int | None) -> bool:
        if component not in self.candidate.get("components", [self.candidate["component"]]):
            return False
        layers = self.candidate.get("layers")
        return (layers is None and layer is None) or (layers is not None and layer in layers)

    def _source_hook(self, component: str, layer: int | None):
        def hook(_module, _args, output):
            values = _out_tensor(output)
            self.reference_values[(component, layer)] = values[:, -1:, :].detach()
            return output
        return hook

    def _target_hook(self, component: str, layer: int | None):
        def hook(_module, _args, output):
            key = (component, layer)
            if key not in self.reference_values:
                raise RuntimeError(f"Reference activation was not produced before target site {key}")
            values = _out_tensor(output)
            patched = values.clone()
            patched[:, -1:, :] = self.reference_values[key].to(device=values.device, dtype=values.dtype)
            return _replace_out(output, patched)
        return hook

    def generate(self, prompt_ids: list[int], tokenizer: Any) -> dict[str, Any]:
        self.reference_values.clear()
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
        mask = torch.ones_like(current_ids)
        source_cache = None
        target_cache = None
        generated: list[int] = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step in range(MAX_NEW_TOKENS):
                source_output = self.source(
                    input_ids=current_ids, attention_mask=mask,
                    past_key_values=source_cache, use_cache=True, return_dict=True,
                )
                target_output = self.target(
                    input_ids=current_ids, attention_mask=mask,
                    past_key_values=target_cache, use_cache=True, return_dict=True,
                )
                source_cache = source_output.past_key_values
                target_cache = target_output.past_key_values
                next_token = int(target_output.logits[0, -1].argmax().item())
                generated.append(next_token)
                if tokenizer.eos_token_id is not None and next_token == int(tokenizer.eos_token_id):
                    break
                current_ids = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
                mask = torch.ones((1, len(prompt_ids) + len(generated)), dtype=torch.long, device=DEVICE)
                self.reference_values.clear()
                del source_output, target_output
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "prediction": text,
            "generated_token_count_including_eos": len(generated),
            "eos_terminated": bool(generated and tokenizer.eos_token_id is not None and generated[-1] == int(tokenizer.eos_token_id)),
            "generated_token_ids": generated,
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.reference_values.clear()


def _plain_generate(model: Any, tokenizer: Any, pack: dict[str, Any]) -> dict[str, Any]:
    x = torch.tensor([pack["prompt_ids"]], dtype=torch.long, device=DEVICE)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.generate(
            input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
    completion = output[0, len(pack["prompt_ids"]):]
    return {
        "prediction": tokenizer.decode(completion, skip_special_tokens=True).strip(),
        "generated_token_count_including_eos": int(completion.numel()),
        "eos_terminated": bool(completion.numel() and int(completion[-1]) == int(tokenizer.eos_token_id)),
        "generated_token_ids": [int(token) for token in completion.tolist()],
    }


def _generation_metrics(task_id: str, packs: list[dict[str, Any]], generations: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [str(item["prediction"]) for item in generations]
    base = generated_label_metrics(predictions)
    lengths = [int(item["generated_token_count_including_eos"]) for item in generations]
    result: dict[str, Any] = {
        "task_id": task_id,
        "evaluation_count": len(packs),
        "mean_generated_token_count_including_eos": sum(lengths) / max(1, len(lengths)),
        "eos_terminated_rate": sum(bool(item["eos_terminated"]) for item in generations) / max(1, len(generations)),
        **base,
    }
    records = [pack["record"] for pack in packs]
    if task_id in GENERATION_TASKS:
        result["rouge_l_f1"] = sum(max(rouge_l_f1(pred, ref) for ref in row.get("targets", [""]) or [""]) for pred, row in zip(predictions, records)) / max(1, len(records))
        result["token_f1"] = sum(max(token_f1(pred, ref) for ref in row.get("targets", [""]) or [""]) for pred, row in zip(predictions, records)) / max(1, len(records))
        result["exact_match"] = sum(any(pred.strip().casefold() == str(ref).strip().casefold() for ref in row.get("targets", [])) for pred, row in zip(predictions, records)) / max(1, len(records))
    else:
        golds = [parse_yes_no(str(row["targets"][0])) for row in records]
        parsed = [parse_yes_no(pred) for pred in predictions]
        result["accuracy"] = sum(pred is not None and pred == gold for pred, gold in zip(parsed, golds)) / max(1, len(records))
        labels = ("Yes", "No")
        per_label = []
        for label in labels:
            tp = sum(gold == label and pred == label for gold, pred in zip(golds, parsed))
            fp = sum(gold != label and pred == label for gold, pred in zip(golds, parsed))
            fn = sum(gold == label and pred != label for gold, pred in zip(golds, parsed))
            per_label.append(2 * tp / max(1, 2 * tp + fp + fn))
        result["macro_f1"] = sum(per_label) / len(per_label)
    return result


def _save_generations(path: Path, condition: str, task_id: str, packs: list[dict[str, Any]], generations: list[dict[str, Any]]):
    rows = []
    for pack, generated in zip(packs, generations):
        rows.append({
            "condition": condition, "task_id": task_id,
            "instance_id": str(pack["record"]["instance_id"]),
            "prediction": generated["prediction"],
            "references": list(pack["record"].get("targets", [])),
            "generated_token_count_including_eos": generated["generated_token_count_including_eos"],
            "eos_terminated": generated["eos_terminated"],
        })
    _jsonl_write(path, _jsonl_read(path) + rows)


def _apply_parameter_rollback(params: dict[str, torch.nn.Parameter], keys: list[str], state: dict[str, torch.Tensor]):
    backup = {name: params[name].detach().clone() for name in keys}
    delta_sq = 0.0
    with torch.no_grad():
        for name in keys:
            reference = state[name].to(device=DEVICE, dtype=params[name].dtype)
            delta_sq += float((reference.float() - params[name].float()).square().sum().item())
            params[name].copy_(reference)
    return backup, math.sqrt(delta_sq)


def _apply_matched_random_delta(params: dict[str, torch.nn.Parameter], keys: list[str], backup: dict[str, torch.Tensor], target_norm: float, seed: int):
    noise_sq = 0.0
    for name in keys:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little"))
        noise = torch.randn(params[name].shape, dtype=torch.float32, device=DEVICE, generator=generator)
        noise_sq += float(noise.square().sum().item())
        del noise
    scale = target_norm / math.sqrt(max(noise_sq, 1e-20))
    with torch.no_grad():
        for name in keys:
            generator = torch.Generator(device=DEVICE)
            generator.manual_seed(seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little"))
            noise = torch.randn(params[name].shape, dtype=torch.float32, device=DEVICE, generator=generator)
            params[name].copy_(backup[name] + (noise * scale).to(dtype=params[name].dtype))
            del noise
    return scale


def _restore_parameters(params: dict[str, torch.nn.Parameter], backup: dict[str, torch.Tensor]):
    with torch.no_grad():
        for name, value in backup.items():
            params[name].copy_(value)
    del backup


def run_confirmation(model_path: Path, tokenizer: Any, train: dict[str, list[dict[str, Any]]], dev: dict[str, list[dict[str, Any]]]):
    activation_selection = json.loads((OUT_ROOT / "activation_screen_selection.json").read_text(encoding="utf-8"))
    parameter_selection = json.loads((OUT_ROOT / "parameter_screen_selection.json").read_text(encoding="utf-8"))
    activation_by_id = {item["candidate_id"]: item for item in activation_candidates(28)}
    parameter_by_id = {item["candidate_id"]: item for item in _param_candidates(28)}
    target = _model_checkpoint(model_path, FINAL)
    params = dict(target.named_parameters())
    activation_screen_rows = activation_selection["all_candidates"]
    parameter_screen_rows = parameter_selection["all_candidates"]
    confirmation_summary: dict[str, Any] = {
        "activation": {}, "parameter_rollback": {},
        "confirmation_dev_indices": [25, 49], "test_split_loaded": False,
        "optimizer_steps": 0, "formal_new_sequence_training": False,
    }
    prediction_path = OUT_ROOT / "confirmation_predictions.jsonl"
    activation_score_rows = []
    parameter_score_rows = []
    if prediction_path.exists():
        prediction_path.unlink()
    for task_id in TASKS:
        packs = _prepare_records(tokenizer, task_id, dev[task_id][25:50])
        baseline_scores = _score_pack_batch(target, packs, tokenizer)
        baseline_generations = [_plain_generate(target, tokenizer, pack) for pack in packs]
        _save_generations(prediction_path, "baseline", task_id, packs, baseline_generations)
        for pack, score in zip(packs, baseline_scores):
            activation_score_rows.append({"task_id": task_id, "candidate_id": "baseline", "instance_id": str(pack["record"]["instance_id"]), **score})
            parameter_score_rows.append({"task_id": task_id, "candidate_id": "baseline", "condition": "baseline", "instance_id": str(pack["record"]["instance_id"]), **score})
        confirmation_summary["activation"].setdefault(task_id, {})["baseline"] = {
            **_generation_metrics(task_id, packs, baseline_generations),
            "teacher_forced_token_nll": sum(x["nll_sum"] for x in baseline_scores) / max(1, sum(x["token_count"] for x in baseline_scores)),
            "mean_first_reference_token_log_probability": sum(x["first_reference_token_log_probability"] for x in baseline_scores) / max(1, len(baseline_scores)),
            "mean_yes_minus_no_log_probability_margin": sum(x["yes_minus_no_log_probability_margin"] for x in baseline_scores if x["yes_minus_no_log_probability_margin"] is not None) / max(1, sum(x["yes_minus_no_log_probability_margin"] is not None for x in baseline_scores)),
        }
        # Select the strongest internal activation candidate; retain the logit
        # swap separately only when it was the screen winner, as an upper bound.
        screened = [row for row in activation_screen_rows if row["task_id"] == task_id and row["scope"] not in ("control", "baseline")]
        internal = [row for row in screened if row["candidate_type"] != "lm_head_logits_upper_bound"]
        best_internal = max(internal, key=lambda row: row["nll_improvement_vs_baseline"])
        activation_candidates_to_confirm = [activation_by_id[best_internal["candidate_id"]]]
        best_any = max(screened, key=lambda row: row["nll_improvement_vs_baseline"])
        if best_any["candidate_type"] == "lm_head_logits_upper_bound" and best_any["candidate_id"] != best_internal["candidate_id"]:
            activation_candidates_to_confirm.append(activation_by_id[best_any["candidate_id"]])
        reference_path = STAGE2 if task_id in GENERATION_TASKS else STAGE3
        source = _model_checkpoint(model_path, reference_path)
        for activation_candidate in activation_candidates_to_confirm:
            condition = activation_candidate["candidate_id"]
            target_hooks = ActivationHooks(target, "patch")
            try:
                scores = []
                generated = []
                for pack in packs:
                    bank = _source_activation_bank(source, pack)
                    logits = _patch_candidate_logits(target, pack, [activation_candidate], bank, target_hooks)
                    scores.extend(_score_logits(logits, [pack], tokenizer))
                    target_hooks.candidates = []
                paired = PairedActivationGenerator(source, target, activation_candidate)
                try:
                    for pack in packs:
                        generated.append(paired.generate(pack["prompt_ids"], tokenizer))
                finally:
                    paired.close()
                metrics = _generation_metrics(task_id, packs, generated)
                metrics.update({
                    "condition": condition,
                    "teacher_forced_token_nll": sum(x["nll_sum"] for x in scores) / max(1, sum(x["token_count"] for x in scores)),
                    "mean_first_reference_token_log_probability": sum(x["first_reference_token_log_probability"] for x in scores) / max(1, len(scores)),
                    "mean_yes_minus_no_log_probability_margin": sum(x["yes_minus_no_log_probability_margin"] for x in scores if x["yes_minus_no_log_probability_margin"] is not None) / max(1, sum(x["yes_minus_no_log_probability_margin"] is not None for x in scores)),
                    "source_checkpoint": reference_path.relative_to(PROJECT_ROOT).as_posix(),
                    "component": activation_candidate["component"],
                    "layers": activation_candidate.get("layers"),
                    "paired_autoregressive_generation": True,
                })
                confirmation_summary["activation"][task_id][condition] = metrics
                for pack, score in zip(packs, scores):
                    activation_score_rows.append({"task_id": task_id, "candidate_id": condition, "instance_id": str(pack["record"]["instance_id"]), **score})
                _save_generations(prediction_path, condition, task_id, packs, generated)
                print(f"activation confirmation {task_id}/{condition}: TF NLL={metrics['teacher_forced_token_nll']:.5f}, generated metric={metrics.get('rouge_l_f1', metrics.get('accuracy'))}", flush=True)
            finally:
                target_hooks.close()
        cleanup_model(source)
        source = None

        screened_params = [row for row in parameter_screen_rows if row["task_id"] == task_id and row["candidate_id"] != "baseline"]
        best_param_row = max(screened_params, key=lambda row: row["nll_improvement_vs_baseline"])
        param_candidate = parameter_by_id[best_param_row["candidate_id"]]
        source_stage = "stage2" if task_id in GENERATION_TASKS else "stage3"
        source_dir = STAGE2 if source_stage == "stage2" else STAGE3
        source_state = load_file(str(source_dir / "model.safetensors"), device="cpu")
        key_names = _param_keys(params, param_candidate)
        backup, update_norm = _apply_parameter_rollback(params, key_names, source_state)
        try:
            rollback_scores = _score_pack_batch(target, packs, tokenizer)
            rollback_generations = [_plain_generate(target, tokenizer, pack) for pack in packs]
            _save_generations(prediction_path, "parameter_rollback_" + param_candidate["candidate_id"], task_id, packs, rollback_generations)
            rollback_metrics = _generation_metrics(task_id, packs, rollback_generations)
            rollback_metrics.update({
                "condition": param_candidate["candidate_id"],
                "teacher_forced_token_nll": sum(x["nll_sum"] for x in rollback_scores) / max(1, sum(x["token_count"] for x in rollback_scores)),
                "mean_first_reference_token_log_probability": sum(x["first_reference_token_log_probability"] for x in rollback_scores) / max(1, len(rollback_scores)),
                "mean_yes_minus_no_log_probability_margin": sum(x["yes_minus_no_log_probability_margin"] for x in rollback_scores if x["yes_minus_no_log_probability_margin"] is not None) / max(1, sum(x["yes_minus_no_log_probability_margin"] is not None for x in rollback_scores)),
                "reference_checkpoint": source_dir.relative_to(PROJECT_ROOT).as_posix(),
                "parameter_count": len([]) + sum(params[name].numel() for name in key_names),
                "rollback_fraction_of_model": sum(params[name].numel() for name in key_names) / sum(parameter.numel() for parameter in params.values()),
                "update_l2_norm": update_norm,
                "update_rms": update_norm / math.sqrt(max(1, sum(params[name].numel() for name in key_names))),
                "tied_embedding_lm_head_restored_once": "model.embed_tokens.weight" in key_names,
            })
            confirmation_summary["parameter_rollback"].setdefault(task_id, {})[param_candidate["candidate_id"]] = rollback_metrics
            for pack, score in zip(packs, rollback_scores):
                parameter_score_rows.append({"task_id": task_id, "candidate_id": param_candidate["candidate_id"], "condition": "rollback", "instance_id": str(pack["record"]["instance_id"]), **score})
            print(f"parameter confirmation {task_id}/{param_candidate['candidate_id']}: TF NLL={rollback_metrics['teacher_forced_token_nll']:.5f}", flush=True)

            # Matched update control: same selected tensors and exact L2 norm,
            # but a deterministic random direction.
            _restore_parameters(params, backup)
            backup = None
            matched_backup = {name: params[name].detach().clone() for name in key_names}
            scale = _apply_matched_random_delta(params, key_names, matched_backup, update_norm, 314159 + int(task_id[4:7]))
            random_scores = _score_pack_batch(target, packs, tokenizer)
            random_generations = [_plain_generate(target, tokenizer, pack) for pack in packs]
            _save_generations(prediction_path, "matched_random_delta_" + param_candidate["candidate_id"], task_id, packs, random_generations)
            random_metrics = _generation_metrics(task_id, packs, random_generations)
            random_metrics.update({
                "condition": "matched_random_delta_" + param_candidate["candidate_id"],
                "teacher_forced_token_nll": sum(x["nll_sum"] for x in random_scores) / max(1, sum(x["token_count"] for x in random_scores)),
                "update_l2_norm": update_norm,
                "update_rms": update_norm / math.sqrt(max(1, sum(params[name].numel() for name in key_names))),
                "noise_scale": scale,
                "parameter_count": sum(params[name].numel() for name in key_names),
                "rollback_fraction_of_model": sum(params[name].numel() for name in key_names) / sum(parameter.numel() for parameter in params.values()),
                "reference_checkpoint": "random_direction_control",
            })
            confirmation_summary["parameter_rollback"][task_id]["matched_random_delta"] = random_metrics
            for pack, score in zip(packs, random_scores):
                parameter_score_rows.append({"task_id": task_id, "candidate_id": param_candidate["candidate_id"], "condition": "matched_random_delta", "instance_id": str(pack["record"]["instance_id"]), **score})
            _restore_parameters(params, matched_backup)
            backup = None
        finally:
            if backup is not None:
                _restore_parameters(params, backup)
        del source_state
    cleanup_model(target)
    _jsonl_write(OUT_ROOT / "activation_confirmation.jsonl", activation_score_rows)
    _jsonl_write(OUT_ROOT / "parameter_confirmation.jsonl", parameter_score_rows)
    _json_dump(OUT_ROOT / "confirmation_summary.json", confirmation_summary)
    print(f"wrote locked confirmation outputs to {OUT_ROOT}", flush=True)


def run_cross_task_transfer(model_path: Path, tokenizer: Any, dev: dict[str, list[dict[str, Any]]]):
    """Apply screen-selected conditions to other task types on locked dev IDs."""
    activation_selection = json.loads((OUT_ROOT / "activation_screen_selection.json").read_text(encoding="utf-8"))
    parameter_selection = json.loads((OUT_ROOT / "parameter_screen_selection.json").read_text(encoding="utf-8"))
    activation_by_id = {row["candidate_id"]: row for row in activation_candidates(28)}
    parameter_by_id = {row["candidate_id"]: row for row in _param_candidates(28)}
    confirmation = json.loads((OUT_ROOT / "confirmation_summary.json").read_text(encoding="utf-8"))
    activation_screen_rows = activation_selection["all_candidates"]
    parameter_screen_rows = parameter_selection["all_candidates"]
    target = _model_checkpoint(model_path, FINAL)
    params = dict(target.named_parameters())
    raw_rows = []
    summary_rows = []
    output_path = OUT_ROOT / "cross_task_transfer.jsonl"
    if output_path.exists():
        output_path.unlink()

    def append_results(kind: str, source_task: str, target_task: str, condition: str, packs: list[dict[str, Any]], scores: list[dict[str, Any]], generations: list[dict[str, Any]], extras: dict[str, Any]):
        metrics = _generation_metrics(target_task, packs, generations)
        token_nll = sum(row["nll_sum"] for row in scores) / max(1, sum(row["token_count"] for row in scores))
        baseline = confirmation["activation"][target_task]["baseline"]
        primary_key = "rouge_l_f1" if target_task in GENERATION_TASKS else "accuracy"
        row = {
            "intervention_type": kind,
            "source_task": source_task,
            "target_task": target_task,
            "condition": condition,
            "source_checkpoint": (STAGE2 if source_task in GENERATION_TASKS else STAGE3).relative_to(PROJECT_ROOT).as_posix(),
            "examples": len(packs),
            "teacher_forced_token_nll": token_nll,
            "teacher_forced_nll_delta_vs_target_baseline": token_nll - baseline["teacher_forced_token_nll"],
            "primary_metric": primary_key,
            "primary_metric_value": metrics[primary_key],
            "primary_metric_delta_vs_target_baseline": metrics[primary_key] - baseline[primary_key],
            "mean_generated_token_count_including_eos": metrics["mean_generated_token_count_including_eos"],
            "exact_yes_output_rate": metrics["exact_yes_output_rate"],
            "exact_no_output_rate": metrics["exact_no_output_rate"],
            "exact_yes_no_output_rate": metrics["exact_yes_no_output_rate"],
            "invalid_answer_format_rate": metrics["invalid_answer_format_rate"],
            **extras,
        }
        summary_rows.append(row)
        for pack, score, generated in zip(packs, scores, generations):
            item = {
                "intervention_type": kind, "source_task": source_task,
                "target_task": target_task, "condition": condition,
                "instance_id": str(pack["record"]["instance_id"]),
                "prediction": generated["prediction"],
                "references": list(pack["record"].get("targets", [])),
                "generated_token_count_including_eos": generated["generated_token_count_including_eos"],
                "eos_terminated": generated["eos_terminated"], **score,
            }
            if target_task in GENERATION_TASKS:
                refs = pack["record"].get("targets", [""]) or [""]
                item["rouge_l_f1"] = max(rouge_l_f1(generated["prediction"], ref) for ref in refs)
                item["token_f1"] = max(token_f1(generated["prediction"], ref) for ref in refs)
            else:
                label = parse_yes_no(generated["prediction"])
                gold = parse_yes_no(str(pack["record"]["targets"][0]))
                item["parsed_prediction"] = label
                item["gold_label"] = gold
                item["correct"] = bool(label is not None and label == gold)
            raw_rows.append(item)

    # Source-task-selected activation patches cross between generation and
    # classification tasks. Source/target always receive the identical prompt
    # and target-generated continuation during autoregressive transfer.
    for source_task in TASKS:
        selected = [row for row in activation_screen_rows if row["task_id"] == source_task and row["scope"] not in ("baseline", "control") and row["candidate_type"] != "lm_head_logits_upper_bound"]
        selected_row = max(selected, key=lambda row: row["nll_improvement_vs_baseline"])
        candidate = activation_by_id[selected_row["candidate_id"]]
        reference_path = STAGE2 if source_task in GENERATION_TASKS else STAGE3
        source = _model_checkpoint(model_path, reference_path)
        for target_task in TASKS:
            if (source_task in GENERATION_TASKS) == (target_task in GENERATION_TASKS):
                continue
            packs = _prepare_records(tokenizer, target_task, dev[target_task][25:50])
            hooks = ActivationHooks(target, "patch")
            try:
                scores = []
                for pack in packs:
                    bank = _source_activation_bank(source, pack)
                    logits = _patch_candidate_logits(target, pack, [candidate], bank, hooks)
                    scores.extend(_score_logits(logits, [pack], tokenizer))
                    hooks.candidates = []
                pair = PairedActivationGenerator(source, target, candidate)
                try:
                    generations = [pair.generate(pack["prompt_ids"], tokenizer) for pack in packs]
                finally:
                    pair.close()
                condition = f"{candidate['candidate_id']}_from_{source_task}"
                append_results("activation", source_task, target_task, condition, packs, scores, generations, {
                    "component": candidate["component"], "layers": candidate.get("layers"),
                })
                print(f"cross-task activation {source_task} -> {target_task}: {condition}", flush=True)
            finally:
                hooks.close()
        cleanup_model(source)
        source = None

    # Parameter conditions use the source task's selected group and boundary.
    for source_task in TASKS:
        if source_task in GENERATION_TASKS:
            candidates = [row for row in parameter_screen_rows if row["task_id"] == source_task and row["candidate_id"] != "baseline"]
        else:
            candidates = [row for row in parameter_screen_rows if row["task_id"] == source_task and row["candidate_id"] != "baseline"]
        selected_row = max(candidates, key=lambda row: row["nll_improvement_vs_baseline"])
        candidate = parameter_by_id[selected_row["candidate_id"]]
        reference_path = STAGE2 if source_task in GENERATION_TASKS else STAGE3
        source_state = load_file(str(reference_path / "model.safetensors"), device="cpu")
        key_names = _param_keys(params, candidate)
        backup, update_norm = _apply_parameter_rollback(params, key_names, source_state)
        try:
            for target_task in TASKS:
                if (source_task in GENERATION_TASKS) == (target_task in GENERATION_TASKS):
                    continue
                packs = _prepare_records(tokenizer, target_task, dev[target_task][25:50])
                scores = _score_pack_batch(target, packs, tokenizer)
                generations = [_plain_generate(target, tokenizer, pack) for pack in packs]
                condition = f"{candidate['candidate_id']}_from_{source_task}"
                append_results("parameter_rollback", source_task, target_task, condition, packs, scores, generations, {
                    "parameter_count": sum(params[name].numel() for name in key_names),
                    "rollback_fraction_of_model": sum(params[name].numel() for name in key_names) / sum(parameter.numel() for parameter in params.values()),
                    "update_l2_norm": update_norm,
                    "update_rms": update_norm / math.sqrt(max(1, sum(params[name].numel() for name in key_names))),
                })
                print(f"cross-task parameter rollback {source_task} -> {target_task}: {condition}", flush=True)
        finally:
            _restore_parameters(params, backup)
            del source_state

    _jsonl_write(output_path, raw_rows)
    summary_path = OUT_ROOT / "cross_task_transfer_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in summary_rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    _json_dump(OUT_ROOT / "cross_task_transfer_manifest.json", {
        "source_to_target_task_pairs": len(summary_rows),
        "evaluation_dev_indices": [25, 49],
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "nas_started": False,
    })
    cleanup_model(target)
    print(f"wrote cross-task transfer results: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("activation-screen", "parameter-screen", "synergy", "trajectory", "confirm", "cross-task"), required=True)
    parser.add_argument("--candidate-batch-size", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6A requires CUDA")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    config, model_path, tokenizer, _device, train, dev = prepare_context()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if args.stage == "activation-screen":
        run_activation_screen(model_path, tokenizer, train, dev, config["model"], args.candidate_batch_size)
    elif args.stage == "parameter-screen":
        run_parameter_screen(model_path, tokenizer, train, dev)
    elif args.stage == "synergy":
        run_activation_synergy(model_path, tokenizer, dev)
    elif args.stage == "trajectory":
        run_checkpoint_trajectory(model_path, tokenizer, dev)
    elif args.stage == "cross-task":
        run_cross_task_transfer(model_path, tokenizer, dev)
    else:
        run_confirmation(model_path, tokenizer, train, dev)
    memory = {
        "stage": args.stage,
        "cuda_device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(DEVICE)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(DEVICE)),
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "formal_new_sequence_training": False,
        "nas_started": False,
    }
    _json_dump(OUT_ROOT / f"{args.stage.replace('-', '_')}_runtime.json", memory)
    print(json.dumps(memory, indent=2), flush=True)


if __name__ == "__main__":
    main()
