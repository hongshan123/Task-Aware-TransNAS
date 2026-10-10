"""Tune and evaluate Phase 5B FV recovery, format, and correct-ICL controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase5_common import (  # noqa: E402
    PHASE5_ROOT,
    SEED,
    SPLITS,
    TASK_1384,
    TASK_1603,
    TASK_565,
    TASK_IDS,
    cleanup_model,
    json_dump,
    load_checkpoint,
    load_phase5_context,
    model_head_geometry,
    norm_match_random,
    parse_yes_no,
    phase4b_checkpoint,
    rouge_l_f1,
    split_records,
    token_f1,
)
from phase5b_common import (  # noqa: E402
    FORMAT_HINT,
    Phase5BResidualIntervention,
    generated_label_metrics,
    make_exact_four_shot_prompt,
    prepare_zero_shot_ids,
    with_format_hint,
)
from phase5_common import answer_tokens_for_custom, make_complete_ids, token_probabilities  # noqa: E402
from run_formal_baseline import answer_tokens_for, safe_prompt  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "results" / "phase5b" / "interventions"
VECTOR_PATH = PROJECT_ROOT / "results" / "phase5b" / "extraction" / "historical_function_vectors.pt"
PHASE5_VECTOR_PATH = PHASE5_ROOT / "fv" / "function_vectors.pt"
FINAL_CHECKPOINT = phase4b_checkpoint("A0", "S1", 10)
LAYERS = (5, 12, 19, 26)
SCALES = (0.25, 0.5, 1.0, 2.0)
MAX_SEQUENCE_LENGTH = 512
MAX_TARGET_TOKENS = 64
MAX_NEW_TOKENS = 64
BOOTSTRAP_REPEATS = 5000
RANDOM_SEED = 314159
FINAL_ID = "phase4b_A0_S1_seed10"
INJECTION_SITE = "selected_decoder_layer.self_attn.o_proj.output.additive_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_score(model: Any, prompt_ids: list[int], target_ids: list[int], device: torch.device,
                intervention: Phase5BResidualIntervention | None = None,
                vector_spec: tuple[int, torch.Tensor, float] | None = None) -> dict[str, Any]:
    if not prompt_ids or not target_ids:
        raise ValueError("Need nonempty prompt and reference answer tokens")
    if intervention is not None:
        if vector_spec is None:
            intervention.deactivate()
        else:
            layer, vector, scale = vector_spec
            intervention.activate(layer, vector, scale, "suffix", len(prompt_ids) - 1)
    full = make_complete_ids(prompt_ids, target_ids, device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        values, _ = token_probabilities(model, full, len(prompt_ids), target_ids)
    values = values[0].float()
    result = {
        "nll_sum": float(-values.sum().item()),
        "token_count": int(values.numel()),
        "mean_token_nll": float(-values.mean().item()),
        "prompt_token_count": len(prompt_ids),
        "reference_token_ids": target_ids,
    }
    if intervention is not None and vector_spec is not None:
        events = intervention.events
        expected_patches = len(target_ids) + 1
        if len(events) != 1 or events[0]["patched_token_count"] != expected_patches:
            raise RuntimeError(
                f"Teacher-forced suffix hook patched {events[0]['patched_token_count'] if events else 0} "
                f"positions; expected {expected_patches}"
            )
        result["intervention_hook"] = {
            "hook_point": events[0]["hook_point"],
            "mode": "suffix",
            "hook_calls": len(events),
            "sequence_length": events[0]["sequence_length"],
            "hidden_size": events[0]["hidden_size"],
            "patched_token_count": sum(event["patched_token_count"] for event in events),
            "requested_delta_rms": events[0]["requested_delta_rms"],
            "applied_delta_rms": events[0]["applied_delta_rms"],
            "max_applied_delta": events[0]["max_applied_delta"],
        }
        intervention.deactivate()
    return result


def _standard_pack(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    # Leave room for the full 64-token greedy completion and use this identical
    # prompt for generation and teacher-forced answer scoring.
    prompt_ids, input_used, prompt_diag = safe_prompt(
        tokenizer, record, MAX_SEQUENCE_LENGTH - MAX_NEW_TOKENS
    )
    target_ids = answer_tokens_for(tokenizer, record, input_used, str(record["targets"][0]))
    target_ids = target_ids[:MAX_TARGET_TOKENS - 1] + [int(tokenizer.eos_token_id)] if len(target_ids) >= MAX_TARGET_TOKENS else target_ids
    if len(prompt_ids) + len(target_ids) > MAX_SEQUENCE_LENGTH:
        raise AssertionError("Zero-shot prompt/target exceeds the 512 token sequence cap")
    return {
        "prompt_ids": prompt_ids,
        "target_ids": target_ids,
        "prompt_diagnostics": prompt_diag,
        "shots": 0,
        "support_train_ids": [],
    }


def _icl_pack(tokenizer: Any, task_id: str, record: dict[str, Any], train_rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    pack = make_exact_four_shot_prompt(
        tokenizer, task_id, record, train_rows, index,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        max_target_tokens=MAX_TARGET_TOKENS,
    )
    if pack.shots != 4 or len(pack.support_ids) != 4 or len(set(pack.support_ids)) != 4:
        raise RuntimeError(f"Four-shot ICL control lost a demonstration for {task_id}/{record['instance_id']}")
    train_ids = {str(row["instance_id"]) for row in train_rows}
    if not set(pack.support_ids).issubset(train_ids):
        raise RuntimeError("ICL demonstrations must come from train only")
    return {
        "prompt_ids": pack.input_ids,
        "target_ids": pack.target_ids,
        "prompt_diagnostics": {
            "prompt_tokens": pack.prompt_tokens,
            "query_input_tokens_kept": pack.query_input_tokens_kept,
            "demonstration_input_tokens_kept": pack.demonstration_input_tokens_kept,
        },
        "shots": 4,
        "support_train_ids": pack.support_ids,
    }


def _encode_tune_record(tokenizer: Any, record: dict[str, Any]) -> tuple[list[int], list[int]]:
    item = _standard_pack(tokenizer, record)
    return item["prompt_ids"], item["target_ids"]


def _make_vector_specs(historical: dict[str, dict[str, torch.Tensor]], final: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    specs: dict[str, dict[str, torch.Tensor]] = {}
    gen_tasks = (TASK_565, TASK_1603)
    unrelated_map = {TASK_565: TASK_1603, TASK_1603: TASK_565}
    all_historical = {task: historical[f"historical_after_{'task565' if task == TASK_565 else 'task1603'}"][task].float() for task in gen_tasks}
    for task_id in TASK_IDS:
        own_ref = all_historical.get(task_id, final[task_id].float())
        unrelated_task = unrelated_map.get(task_id, TASK_565)
        specs[task_id] = {
            "final_collapsed_checkpoint_fv": final[task_id].float(),
            "norm_matched_random_fv": norm_match_random(own_ref.float(), RANDOM_SEED + int(task_id[4:7])),
            "unrelated_historical_task_fv": all_historical[unrelated_task].float(),
            "subtract_latest_classifier_fv": -final[TASK_1384].float(),
        }
        if task_id in all_historical:
            specs[task_id]["historical_task_boundary_fv"] = all_historical[task_id].float()
    return specs


def _tune(
    model: Any,
    tokenizer: Any,
    train: dict[str, list[dict[str, Any]]],
    dev: dict[str, list[dict[str, Any]]],
    vector_specs: dict[str, dict[str, torch.Tensor]],
    intervention: Phase5BResidualIntervention,
) -> dict[str, dict[str, Any]]:
    tune_rows = split_records(dev, "tune")
    chosen: dict[str, dict[str, Any]] = {}
    grid_all = []
    for task_id in TASK_IDS:
        key = f"{task_id}"
        cache = OUT_ROOT / "tuning" / f"{task_id}.json"
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("tune_ids") == [row["instance_id"] for row in tune_rows[task_id]] and cached.get("layers") == list(LAYERS) and cached.get("scales") == list(SCALES) and cached.get("injection_site") == INJECTION_SITE:
                chosen[key] = cached["selected"]
                grid_all.extend(cached["grid"])
                print(f"using validated tuning cache {task_id}", flush=True)
                continue
        encoded = [_encode_tune_record(tokenizer, row) for row in tune_rows[task_id]]
        task_grid = []
        for condition, vector in vector_specs[task_id].items():
            for layer in LAYERS:
                for scale in SCALES:
                    nll_sum, token_count = 0.0, 0
                    for prompt_ids, target_ids in encoded:
                        score = _mean_score(model, prompt_ids, target_ids, next(model.parameters()).device,
                                            intervention, (layer, vector, scale))
                        nll_sum += score["nll_sum"]
                        token_count += score["token_count"]
                    task_grid.append({
                        "task_id": task_id,
                        "condition": condition,
                        "layer": layer,
                        "scale": scale,
                        "nll_sum": nll_sum,
                        "reference_tokens": token_count,
                        "token_weighted_mean_nll": nll_sum / max(1, token_count),
                        "mean_per_example_nll": nll_sum / max(1, len(encoded)),
                    })
            print(f"tuning {task_id}/{condition}: completed {len(LAYERS) * len(SCALES)} layer-scale pairs", flush=True)
        selected = {}
        for condition in vector_specs[task_id]:
            candidates = [row for row in task_grid if row["condition"] == condition]
            best = min(candidates, key=lambda row: (row["token_weighted_mean_nll"], row["layer"], row["scale"]))
            selected[condition] = {
                "layer": int(best["layer"]),
                "scale": float(best["scale"]),
                "tune_token_weighted_mean_nll": float(best["token_weighted_mean_nll"]),
                "tune_mean_per_example_nll": float(best["mean_per_example_nll"]),
            }
        cache_payload = {
            "task_id": task_id,
            "injection_site": INJECTION_SITE,
            "tune_split": [10, 25],
            "tune_ids": [row["instance_id"] for row in tune_rows[task_id]],
            "objective": "token-weighted teacher-forced reference NLL; lower is better",
            "layers": list(LAYERS),
            "scales": list(SCALES),
            "vector_conditions": list(vector_specs[task_id]),
            "grid": task_grid,
            "selected": selected,
        }
        json_dump(cache, cache_payload)
        chosen[key] = selected
        grid_all.extend(task_grid)
    json_dump(OUT_ROOT / "tuning_grid.json", {
        "split": {"dev_indices": [10, 24], "per_task_count": 15},
        "objective": "token-weighted teacher-forced reference NLL; lower is better",
        "layers": list(LAYERS),
        "scales": list(SCALES),
        "grid_rows": grid_all,
        "selected": chosen,
        "test_split_loaded": False,
    })
    return chosen


def _generate(
    model: Any,
    tokenizer: Any,
    pack: dict[str, Any],
    device: torch.device,
    intervention: Phase5BResidualIntervention,
    vector_spec: tuple[int, torch.Tensor, float] | None,
) -> dict[str, Any]:
    if vector_spec is None:
        intervention.deactivate()
    else:
        layer, vector, scale = vector_spec
        intervention.activate(layer, vector, scale, "last")
    prompt = torch.tensor([pack["prompt_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    completion = generated[0, len(pack["prompt_ids"]):]
    prediction = tokenizer.decode(completion, skip_special_tokens=True).strip()
    output = {
        "prediction": prediction,
        "generated_token_count_including_eos": int(completion.numel()),
        "eos_terminated": bool(completion.numel() and int(completion[-1]) == int(tokenizer.eos_token_id)),
    }
    if vector_spec is not None:
        events = list(intervention.events)
        lengths = [int(event["sequence_length"]) for event in events]
        if not events or lengths[0] <= 1:
            raise RuntimeError(f"Generation intervention did not hit the prefill sequence: {lengths}")
        # A completion that ends with EOS as its first token has no cached
        # single-token decode call. The dedicated preflight above already
        # verifies that the hook follows all decode calls when they occur.
        if output["generated_token_count_including_eos"] > 1 and not any(length == 1 for length in lengths[1:]):
            raise RuntimeError(f"Generation intervention missed cached decode after prefill: {lengths}")
        output["intervention_hook"] = {
            "hook_point": events[0]["hook_point"],
            "mode": "last_each_cached_step",
            "hook_calls": len(events),
            "sequence_lengths": lengths,
            "hidden_size": int(events[0]["hidden_size"]),
            "patched_token_count": sum(event["patched_token_count"] for event in events),
            "requested_delta_rms_mean": float(np.mean([event["requested_delta_rms"] for event in events])),
            "applied_delta_rms_mean": float(np.mean([event["applied_delta_rms"] for event in events])),
            "max_applied_delta": float(max(event["max_applied_delta"] for event in events)),
        }
        intervention.deactivate()
    else:
        intervention.deactivate()
    return output


def _condition_summary(task_id: str, condition: str, records: list[dict[str, Any]], nll_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preds = [str(row["prediction"]) for row in records]
    labels = generated_label_metrics(preds)
    nll_values = [row["nll_sum"] / max(1, row["token_count"]) for row in nll_rows]
    target_tokens = sum(int(row["token_count"]) for row in nll_rows)
    nll_sum = sum(float(row["nll_sum"]) for row in nll_rows)
    rouge_values = [float(row["rouge_l_f1"]) for row in records]
    token_f1_values = [float(row["token_f1"]) for row in records]
    exact_values = [float(row["exact_match"]) for row in records]
    parsed = [parse_yes_no(value) for value in preds]
    row = {
        "task_id": task_id,
        "condition": condition,
        "evaluation_count": len(records),
        "teacher_forced_nll_sum": nll_sum,
        "teacher_forced_target_tokens": target_tokens,
        "teacher_forced_token_nll": nll_sum / max(1, target_tokens),
        "mean_per_example_nll": float(np.mean(nll_values)),
        "mean_generated_token_count_including_eos": float(np.mean([row["generated_token_count_including_eos"] for row in records])),
        "eos_terminated_rate": float(np.mean([row["eos_terminated"] for row in records])),
        "rouge_l_f1": float(np.mean(rouge_values)),
        "token_f1": float(np.mean(token_f1_values)),
        "exact_match": float(np.mean(exact_values)),
        "parsed_accuracy": float(np.mean([
            parsed[index] is not None and parsed[index] == parse_yes_no(records[index]["references"][0])
            for index in range(len(records))
        ])),
        **labels,
    }
    return row


def _bootstrap_delta(
    baseline: list[dict[str, Any]],
    condition: list[dict[str, Any]],
    task_id: str,
    condition_name: str,
) -> list[dict[str, Any]]:
    by_base = {row["instance_id"]: row for row in baseline}
    by_cond = {row["instance_id"]: row for row in condition}
    if set(by_base) != set(by_cond):
        raise RuntimeError(f"Paired locked IDs differ for {task_id}/{condition_name}")
    ids = sorted(by_base)
    scalar_extractors = {
        "rouge_l_f1": lambda row: float(row["rouge_l_f1"]),
        "token_f1": lambda row: float(row["token_f1"]),
        "mean_per_example_nll": lambda row: float(row["nll_sum"]) / max(1, int(row["token_count"])),
        "generated_token_count_including_eos": lambda row: float(row["generated_token_count_including_eos"]),
        "exact_yes_output_indicator": lambda row: float(row["strict_exact_yes"]),
        "exact_no_output_indicator": lambda row: float(row["strict_exact_no"]),
        "exact_yes_no_output_indicator": lambda row: float(row["strict_exact_yes"] or row["strict_exact_no"]),
    }
    rng = np.random.default_rng(RANDOM_SEED + sum(map(ord, task_id + condition_name)))
    indexes = rng.integers(0, len(ids), size=(BOOTSTRAP_REPEATS, len(ids)))
    output = []
    for metric, extractor in scalar_extractors.items():
        base = np.asarray([extractor(by_base[item]) for item in ids], dtype=np.float64)
        cond = np.asarray([extractor(by_cond[item]) for item in ids], dtype=np.float64)
        deltas = cond - base
        samples = deltas[indexes].mean(axis=1)
        output.append({
            "task_id": task_id,
            "condition": condition_name,
            "metric": metric,
            "n": len(ids),
            "mean_delta": float(deltas.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
        })
    # Bootstrap parse-based format metrics as paired sample-level statistics.
    for metric in ("most_common_label_concentration_among_parseable", "invalid_answer_format_rate", "parseable_yes_no_label_rate"):
        sample_results = []
        for selected in indexes:
            base_metric = generated_label_metrics([str(by_base[ids[index]]["prediction"]) for index in selected])[metric]
            cond_metric = generated_label_metrics([str(by_cond[ids[index]]["prediction"]) for index in selected])[metric]
            if base_metric is not None and cond_metric is not None:
                sample_results.append(cond_metric - base_metric)
        if sample_results:
            output.append({
                "task_id": task_id,
                "condition": condition_name,
                "metric": metric,
                "n": len(ids),
                "mean_delta": float((generated_label_metrics([by_cond[item]["prediction"] for item in ids])[metric] or 0.0) - (generated_label_metrics([by_base[item]["prediction"] for item in ids])[metric] or 0.0)),
                "ci95_low": float(np.quantile(sample_results, 0.025)),
                "ci95_high": float(np.quantile(sample_results, 0.975)),
            })
    return output


def _bootstrap_recovery_contrasts(
    task_id: str,
    eval_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Directly compare each own historical FV with the required controls."""
    if "historical_task_boundary_fv" not in eval_rows:
        return []
    historical = {row["instance_id"]: row for row in eval_rows["historical_task_boundary_fv"]}
    contrasts = []
    for comparator in (
        "final_collapsed_checkpoint_fv",
        "norm_matched_random_fv",
        "unrelated_historical_task_fv",
    ):
        other = {row["instance_id"]: row for row in eval_rows[comparator]}
        if set(historical) != set(other):
            raise RuntimeError(f"Historical/control locked IDs differ for {task_id}/{comparator}")
        ids = sorted(historical)
        rng = np.random.default_rng(RANDOM_SEED + sum(map(ord, task_id + comparator)))
        indexes = rng.integers(0, len(ids), size=(BOOTSTRAP_REPEATS, len(ids)))
        measures = {
            "mean_per_example_nll_historical_minus_control": lambda row: float(row["nll_sum"]) / max(1, int(row["token_count"])),
            "rouge_l_f1_historical_minus_control": lambda row: float(row["rouge_l_f1"]),
            "generated_length_historical_minus_control": lambda row: float(row["generated_token_count_including_eos"]),
            "exact_yes_no_output_historical_minus_control": lambda row: float(row["strict_exact_yes"] or row["strict_exact_no"]),
        }
        for metric, get_value in measures.items():
            historical_values = np.asarray([get_value(historical[item]) for item in ids], dtype=np.float64)
            control_values = np.asarray([get_value(other[item]) for item in ids], dtype=np.float64)
            differences = historical_values - control_values
            samples = differences[indexes].mean(axis=1)
            contrasts.append({
                "task_id": task_id,
                "contrast": f"historical_task_boundary_fv_minus_{comparator}",
                "metric": metric,
                "n": len(ids),
                "mean_delta": float(differences.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
                "interpretation": "negative NLL favors historical FV; positive ROUGE favors historical FV",
            })
    return contrasts


def _preflight(model: Any, tokenizer: Any, train: dict[str, list[dict[str, Any]]], dev: dict[str, list[dict[str, Any]]], device: torch.device,
               intervention: Phase5BResidualIntervention, historical: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    geometry = model_head_geometry(model)
    expected = {"layers": 28, "query_heads": 16, "kv_heads": 8, "head_dim": 128, "hidden_size": 1024, "o_proj_input_width": 2048}
    if geometry != expected:
        raise RuntimeError(f"Unexpected model head geometry: {geometry}")
    task = TASK_565
    record = dev[task][25]
    pack = _standard_pack(tokenizer, record)
    baseline = _mean_score(model, pack["prompt_ids"], pack["target_ids"], device)
    zero = torch.zeros(1024, dtype=torch.float32)
    patched = _mean_score(model, pack["prompt_ids"], pack["target_ids"], device, intervention, (5, zero, 1.0))
    no_op_nll_delta = abs(baseline["nll_sum"] - patched["nll_sum"])
    no_op_events = patched["intervention_hook"]
    if no_op_nll_delta > 1e-6:
        raise RuntimeError(f"Zero-vector intervention is not a no-op: NLL delta {no_op_nll_delta}")
    expected_count = len(pack["target_ids"]) + 1
    if no_op_events["patched_token_count"] != expected_count:
        raise RuntimeError(f"Teacher-forced suffix patch count mismatch: {no_op_events}")
    if no_op_events["hook_point"] != INJECTION_SITE or no_op_events["hidden_size"] != 1024:
        raise RuntimeError(f"Intervention did not patch the 1024-wide Qwen3 attention output: {no_op_events}")
    sample_vector = historical["historical_after_task565"][TASK_565].float()
    generate_spec = (5, sample_vector, 0.25)
    generation_pack = {"prompt_ids": pack["prompt_ids"]}
    if not sample_vector.norm().isfinite() or sample_vector.norm().item() <= 0:
        raise RuntimeError("Historical vector is invalid during generation-hook preflight")
    # Two new tokens are required so the test covers prefill and cached decode
    # even when greedy decoding would otherwise emit EOS immediately.
    intervention.activate(*generate_spec, "last")
    prompt = torch.tensor([generation_pack["prompt_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=False,
            max_new_tokens=4,
            min_new_tokens=2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    event_lengths = [int(item["sequence_length"]) for item in intervention.events]
    if (
        not event_lengths
        or event_lengths[0] <= 1
        or not any(value == 1 for value in event_lengths[1:])
        or any(item.get("hook_point") != INJECTION_SITE or item.get("hidden_size") != 1024 for item in intervention.events)
    ):
        raise RuntimeError(f"Generation hook missed prefill or cached single-token decode: {event_lengths}")
    intervention.deactivate()
    fourshot = _icl_pack(tokenizer, task, record, train[task], 25)
    if fourshot["shots"] != 4 or len(fourshot["support_train_ids"]) != 4:
        raise RuntimeError("Correct four-shot ICL preflight did not preserve all four train demonstrations")
    return {
        "head_geometry": geometry,
        "injection_site": INJECTION_SITE,
        "attention_output_hidden_size": int(no_op_events["hidden_size"]),
        "zero_vector_intervention_max_nll_sum_delta": no_op_nll_delta,
        "teacher_forced_patch_positions_expected_and_observed": [expected_count, no_op_events["patched_token_count"]],
        "generation_hook_sequence_lengths": event_lengths,
        "generation_generated_tokens": int(generated.shape[1] - prompt.shape[1]),
        "generation_hook_checked_prefill_and_cached_decode": True,
        "four_shot_count": fourshot["shots"],
        "four_shot_support_ids": fourshot["support_train_ids"],
        "four_shot_support_from_train_only": True,
        "injection_site": INJECTION_SITE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-retune", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 5B intervention evaluation requires CUDA")
    config, model_path, tokenizer, device, train, dev = load_phase5_context()
    if not FINAL_CHECKPOINT.joinpath("model.safetensors").is_file():
        raise FileNotFoundError(FINAL_CHECKPOINT / "model.safetensors")
    if not VECTOR_PATH.is_file() or not PHASE5_VECTOR_PATH.is_file():
        raise FileNotFoundError("Phase 5B historical and Phase 5 final vectors must exist before interventions")
    historical = torch.load(VECTOR_PATH, map_location="cpu", weights_only=True)
    final_all = torch.load(PHASE5_VECTOR_PATH, map_location="cpu", weights_only=True)
    if FINAL_ID not in final_all:
        raise RuntimeError(f"Missing final Phase 5 FV snapshot {FINAL_ID}")
    final = final_all[FINAL_ID]
    expected_historical = {
        "historical_after_task565": TASK_565,
        "historical_after_task1603": TASK_1603,
    }
    for snapshot, task_id in expected_historical.items():
        vector = historical.get(snapshot, {}).get(task_id)
        if vector is None or tuple(vector.shape) != (1024,) or not bool(torch.isfinite(vector).all()):
            raise RuntimeError(f"Missing/invalid historical task FV: {snapshot}/{task_id}")
    if set(final) != set(TASK_IDS):
        raise RuntimeError("Final collapsed vectors do not cover all four tasks")
    for task_id in TASK_IDS:
        if tuple(final[task_id].shape) != (1024,) or not bool(torch.isfinite(final[task_id]).all()):
            raise RuntimeError(f"Invalid final collapsed task vector: {task_id}")
    vector_specs = _make_vector_specs(historical, final)

    model = load_checkpoint(model_path, FINAL_CHECKPOINT, architecture="A0")
    intervention = Phase5BResidualIntervention(model)
    try:
        preflight = _preflight(model, tokenizer, train, dev, device, intervention, historical)
        preflight.update({
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "model_path": str(model_path),
            "final_checkpoint": str(FINAL_CHECKPOINT),
            "final_checkpoint_sha256": _sha256_file(FINAL_CHECKPOINT / "model.safetensors"),
            "train_ids_used_for_demos_only": True,
            "tune_dev_indices": [10, 24],
            "locked_evaluation_dev_indices": [25, 49],
            "test_split_loaded": False,
            "nas_started": False,
        })
        json_dump(OUT_ROOT / "preflight.json", preflight)
        print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            return

        tune_dir = OUT_ROOT / "tuning"
        if args.force_retune and tune_dir.exists():
            for cache in tune_dir.glob("*.json"):
                cache.unlink()
        selected = _tune(model, tokenizer, train, dev, vector_specs, intervention)
        eval_root = OUT_ROOT / "evaluation"
        eval_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        nll_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
        all_conditions = sorted({
            "baseline", "explicit_format_hint", "correct_four_shot_icl",
            *[condition for task in TASK_IDS for condition in vector_specs[task]],
        })
        locked = split_records(dev, "evaluation")
        start_index = SPLITS["evaluation"][0]
        eval_cache_manifest: dict[str, Any] = {}

        for task_id in TASK_IDS:
            for condition in all_conditions:
                if condition not in {"baseline", "explicit_format_hint", "correct_four_shot_icl"} and condition not in vector_specs[task_id]:
                    continue
                condition_path = eval_root / task_id / f"{condition}.json"
                expected_ids = [row["instance_id"] for row in locked[task_id]]
                if condition_path.exists() and not args.force_evaluation:
                    cached = json.loads(condition_path.read_text(encoding="utf-8"))
                    if cached.get("evaluation_ids") == expected_ids and cached.get("injection_site") == INJECTION_SITE and len(cached.get("predictions", [])) == 25 and len(cached.get("nll_rows", [])) == 25:
                        cached_predictions = cached["predictions"]
                        cached_scores = cached["nll_rows"]
                        for prediction_row, score_row in zip(cached_predictions, cached_scores):
                            if prediction_row["instance_id"] != score_row["instance_id"]:
                                raise RuntimeError(f"Cached prediction/NLL IDs differ for {task_id}/{condition}")
                            prediction_row["nll_sum"] = float(score_row["nll_sum"])
                            prediction_row["token_count"] = int(score_row["token_count"])
                        eval_rows.setdefault(task_id, {})[condition] = cached_predictions
                        nll_rows.setdefault(task_id, {})[condition] = cached_scores
                        eval_cache_manifest[f"{task_id}/{condition}"] = "validated_cache"
                        continue
                predictions: list[dict[str, Any]] = []
                scores: list[dict[str, Any]] = []
                for local_index, record in enumerate(locked[task_id]):
                    absolute_index = start_index + local_index
                    if condition == "correct_four_shot_icl":
                        pack = _icl_pack(tokenizer, task_id, record, train[task_id], absolute_index)
                    else:
                        source = with_format_hint(record) if condition == "explicit_format_hint" else record
                        pack = _standard_pack(tokenizer, source)
                    spec = None
                    if condition in vector_specs[task_id]:
                        tuned = selected[task_id][condition]
                        spec = (int(tuned["layer"]), vector_specs[task_id][condition], float(tuned["scale"]))
                    generated = _generate(model, tokenizer, pack, device, intervention, spec)
                    scored = _mean_score(model, pack["prompt_ids"], pack["target_ids"], device, intervention, spec)
                    prediction = generated["prediction"]
                    references = list(record.get("targets", []))
                    parsed = parse_yes_no(prediction)
                    gold_parsed = parse_yes_no(str(references[0])) if references else None
                    pred = {
                        "task_id": task_id,
                        "instance_id": record["instance_id"],
                        "condition": condition,
                        "prediction": prediction,
                        "references": references,
                        "generated_token_count_including_eos": generated["generated_token_count_including_eos"],
                        "eos_terminated": generated["eos_terminated"],
                        "rouge_l_f1": max((rouge_l_f1(prediction, ref) for ref in references), default=0.0),
                        "token_f1": max((token_f1(prediction, ref) for ref in references), default=0.0),
                        "exact_match": float(any(prediction.strip().casefold() == ref.strip().casefold() for ref in references)),
                        "parsed_prediction": parsed,
                        "gold_parsed_label": gold_parsed,
                        "parsed_correct": bool(parsed is not None and parsed == gold_parsed),
                        "strict_exact_yes": unicodedata.normalize("NFKC", prediction).strip().casefold() == "yes",
                        "strict_exact_no": unicodedata.normalize("NFKC", prediction).strip().casefold() == "no",
                        "nll_sum": scored["nll_sum"],
                        "token_count": scored["token_count"],
                        "shots": pack["shots"],
                        "support_train_ids": pack["support_train_ids"],
                        "prompt_token_count": len(pack["prompt_ids"]),
                        "prompt_diagnostics": pack["prompt_diagnostics"],
                        "generation_intervention_hook": generated.get("intervention_hook"),
                        "tuning": selected.get(task_id, {}).get(condition),
                    }
                    nll = {
                        "task_id": task_id,
                        "instance_id": record["instance_id"],
                        "condition": condition,
                        "nll_sum": scored["nll_sum"],
                        "token_count": scored["token_count"],
                        "mean_token_nll": scored["mean_token_nll"],
                        "teacher_forced_intervention_hook": scored.get("intervention_hook"),
                    }
                    predictions.append(pred)
                    scores.append(nll)
                    if local_index == 0 or (local_index + 1) % 5 == 0 or local_index == len(locked[task_id]) - 1:
                        print(f"evaluation {task_id}/{condition}: {local_index + 1}/25", flush=True)
                payload = {
                    "phase": "5B",
                    "checkpoint_id": FINAL_ID,
                    "task_id": task_id,
                    "condition": condition,
                    "injection_site": INJECTION_SITE,
                    "evaluation_split": {"dev_indices": [25, 49], "count": 25},
                    "evaluation_ids": expected_ids,
                    "predictions": predictions,
                    "nll_rows": scores,
                    "test_split_loaded": False,
                }
                json_dump(condition_path, payload)
                eval_rows.setdefault(task_id, {})[condition] = predictions
                nll_rows.setdefault(task_id, {})[condition] = scores
                eval_cache_manifest[f"{task_id}/{condition}"] = "computed"

        metrics = []
        bootstrap = []
        recovery_contrasts = []
        for task_id in TASK_IDS:
            for condition, rows in eval_rows[task_id].items():
                summary = _condition_summary(task_id, condition, rows, nll_rows[task_id][condition])
                metrics.append(summary)
                if condition != "baseline":
                    bootstrap.extend(_bootstrap_delta(eval_rows[task_id]["baseline"], rows, task_id, condition))
            recovery_contrasts.extend(_bootstrap_recovery_contrasts(task_id, eval_rows[task_id]))
        json_dump(OUT_ROOT / "metrics.json", {
            "phase": "5B",
            "metric_definitions": {
                "exact_yes_output_rate": "NFKC/casefold/trim output exactly equals yes, divided by all generations",
                "exact_no_output_rate": "NFKC/casefold/trim output exactly equals no, divided by all generations",
                "exact_yes_no_output_rate": "exact yes plus exact no, divided by all generations",
                "most_common_label_concentration_among_parseable": "larger unambiguous parsed Yes/No count divided by parseable generations only",
                "parseable_yes_no_label_rate": "unambiguous parsed Yes/No generations divided by all generations",
                "invalid_answer_format_rate": "existing unambiguous parser returns None, divided by all generations",
                "generated_length": "completion token count including EOS when emitted",
                "teacher_forced_token_nll": "summed reference NLL divided by all reference tokens in the condition's prompt setup",
            },
            "rows": metrics,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_vs_baseline": bootstrap,
            "direct_historical_fv_vs_control_bootstrap": recovery_contrasts,
            "evaluation_cache_manifest": eval_cache_manifest,
            "test_split_loaded": False,
        })
        with (OUT_ROOT / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        json_dump(OUT_ROOT / "run_config.json", {
            "phase": "5B",
            "model_path": str(model_path),
            "model_config_sha256": _sha256_file(model_path / "config.json"),
            "final_checkpoint": str(FINAL_CHECKPOINT),
            "final_checkpoint_sha256": _sha256_file(FINAL_CHECKPOINT / "model.safetensors"),
            "historical_vector_file": str(VECTOR_PATH),
            "historical_vector_file_sha256": _sha256_file(VECTOR_PATH),
            "phase5_final_vector_file": str(PHASE5_VECTOR_PATH),
            "phase5_final_vector_file_sha256": _sha256_file(PHASE5_VECTOR_PATH),
            "layer_candidates_zero_based": list(LAYERS),
            "scale_candidates": list(SCALES),
            "injection_site": INJECTION_SITE,
            "selected_parameters_by_task": selected,
            "tune_dev_indices": [10, 24],
            "locked_evaluation_dev_indices": [25, 49],
            "tune_objective": "token-weighted teacher-forced reference NLL",
            "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": 0, "use_cache": True},
            "format_instruction": FORMAT_HINT,
            "vector_condition_policy": {
                "historical_own_vector": "available only for task565 and task1603; other tasks omit this condition",
                "unrelated_historical_vector": "swap task565/task1603 for each other; task1501/task1384 receive task565 historical vector",
                "norm_matched_random_reference": "historical own FV when available; final task FV otherwise",
                "subtract_latest_classifier": "negative final task1384 Function Vector",
            },
            "icl": {"shots": 4, "support_source": "train only", "support_labels_correct": True, "support_ids_saved": True},
            "test_split_loaded": False,
            "nas_started": False,
        })
        print(f"saved locked Phase 5B intervention metrics: {OUT_ROOT}", flush=True)
    finally:
        intervention.close()
        cleanup_model(model)


if __name__ == "__main__":
    main()
