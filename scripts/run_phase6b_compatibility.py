"""Fill the same-task-type edges missing from the Phase 6A 4x4 matrices.

Existing Phase 6A cross-task rows are consumed read-only. This script evaluates
the four remaining directed off-diagonal task pairs with the exact same selected
activation and parameter rollback operations on locked dev IDs 25--49.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_phase6a_localization as phase6a
from phase4_common import prepare_context

OUT = ROOT / "results" / "phase6b"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _operation_selections() -> tuple[dict[str, dict], dict[str, dict]]:
    activation = json.loads((phase6a.OUT_ROOT / "activation_screen_selection.json").read_text(encoding="utf-8"))
    parameters = json.loads((phase6a.OUT_ROOT / "parameter_screen_selection.json").read_text(encoding="utf-8"))
    activation_rows = activation["all_candidates"]
    parameter_rows = parameters["all_candidates"]
    activation_selected = {}
    parameter_selected = {}
    for task in phase6a.TASKS:
        options = [row for row in activation_rows if row["task_id"] == task and row["scope"] not in ("baseline", "control") and row["candidate_type"] != "lm_head_logits_upper_bound"]
        activation_selected[task] = max(options, key=lambda row: float(row["nll_improvement_vs_baseline"]))
        options = [row for row in parameter_rows if row["task_id"] == task and row["candidate_id"] != "baseline"]
        parameter_selected[task] = max(options, key=lambda row: float(row["nll_improvement_vs_baseline"]))
    return activation_selected, parameter_selected


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6B compatibility check requires CUDA")
    config, model_path, tokenizer, device, _train, dev = prepare_context()
    torch.cuda.set_device(device)
    activation_selected, parameter_selected = _operation_selections()
    pairs = [
        (source, target)
        for source in phase6a.TASKS
        for target in phase6a.TASKS
        if source != target
        and (source in phase6a.GENERATION_TASKS) == (target in phase6a.GENERATION_TASKS)
    ]
    missing_activation_raw = OUT / "compatibility_missing_activation.jsonl"
    missing_parameter_raw = OUT / "compatibility_missing_parameter.jsonl"
    summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for kind, raw_path in (("activation", missing_activation_raw), ("parameter_rollback", missing_parameter_raw)):
        rows_out: list[dict[str, Any]] = []
        summaries_kind: list[dict[str, Any]] = []
        for source_task, target_task in pairs:
            source_checkpoint = phase6a.STAGE2 if source_task in phase6a.GENERATION_TASKS else phase6a.STAGE3
            packs = phase6a._prepare_records(tokenizer, target_task, dev[target_task][25:50])
            target = phase6a._model_checkpoint(model_path, phase6a.FINAL)
            params = dict(target.named_parameters())
            if kind == "activation":
                candidate_row = activation_selected[source_task]
                candidate = {row["candidate_id"]: row for row in phase6a.activation_candidates(28)}[candidate_row["candidate_id"]]
                source = phase6a._model_checkpoint(model_path, source_checkpoint)
                hooks = phase6a.ActivationHooks(target, "patch")
                scores = []
                generations = []
                try:
                    pair_generator = phase6a.PairedActivationGenerator(source, target, candidate)
                    try:
                        for index, pack in enumerate(packs, start=1):
                            bank = phase6a._source_activation_bank(source, pack)
                            logits = phase6a._patch_candidate_logits(target, pack, [candidate], bank, hooks)
                            scores.extend(phase6a._score_logits(logits, [pack], tokenizer))
                            generations.append(pair_generator.generate(pack["prompt_ids"], tokenizer))
                            if index in (1, len(packs)):
                                print(f"compat activation {source_task} -> {target_task}: {index}/{len(packs)}", flush=True)
                        summaries_kind.append(_aggregate(kind, source_task, target_task, candidate_row["candidate_id"], packs, scores, generations))
                    finally:
                        pair_generator.close()
                finally:
                    hooks.close()
                    phase6a.cleanup_model(source)
                    source = None
            else:
                candidate_row = parameter_selected[source_task]
                candidate = next(row for row in phase6a._param_candidates(28) if row["candidate_id"] == candidate_row["candidate_id"])
                state = load_file(str(source_checkpoint / "model.safetensors"), device="cpu")
                keys = phase6a._param_keys(params, candidate)
                backup, update_norm = phase6a._apply_parameter_rollback(params, keys, state)
                try:
                    scores = phase6a._score_pack_batch(target, packs, tokenizer)
                    generations = [phase6a._plain_generate(target, tokenizer, pack) for pack in packs]
                    result = _aggregate(kind, source_task, target_task, candidate_row["candidate_id"], packs, scores, generations)
                    result.update({
                        "parameter_count": sum(params[name].numel() for name in keys),
                        "rollback_fraction_of_model": sum(params[name].numel() for name in keys) / sum(parameter.numel() for parameter in params.values()),
                        "update_l2_norm": update_norm,
                    })
                    summaries_kind.append(result)
                finally:
                    phase6a._restore_parameters(params, backup)
                    del state
            summary = summaries_kind[-1]
            for pack, score, generated in zip(packs, scores, generations):
                row = {
                    "intervention_type": kind,
                    "source_task": source_task,
                    "target_task": target_task,
                    "condition": summary["condition"],
                    "instance_id": str(pack["record"]["instance_id"]),
                    "prediction": generated["prediction"],
                    "references": list(pack["record"].get("targets", [])),
                    "generated_token_count_including_eos": generated["generated_token_count_including_eos"],
                    "eos_terminated": generated["eos_terminated"],
                    **score,
                }
                if target_task in phase6a.GENERATION_TASKS:
                    refs = pack["record"].get("targets", [""]) or [""]
                    row["rouge_l_f1"] = max(phase6a.rouge_l_f1(generated["prediction"], ref) for ref in refs)
                    row["token_f1"] = max(phase6a.token_f1(generated["prediction"], ref) for ref in refs)
                else:
                    prediction = phase6a.parse_yes_no(generated["prediction"])
                    gold = phase6a.parse_yes_no(str(pack["record"]["targets"][0]))
                    row["parsed_prediction"] = prediction
                    row["gold_label"] = gold
                    row["correct"] = bool(prediction is not None and prediction == gold)
                rows_out.append(row)
            phase6a.cleanup_model(target)
            target = None
            del params
            torch.cuda.empty_cache()
            _write_csv(OUT / f"compatibility_missing_{kind}_summary.csv", summaries_kind)
            phase6a._jsonl_write(raw_path, rows_out)
        summaries.extend(summaries_kind)

    manifest = {
        "scope": "same_type_off_diagonal_edges_only",
        "evaluated_conditions": len(summaries),
        "raw_example_rows": 4 * 25 * 2,
        "dev_indices": [25, 49],
        "historical_source_checkpoints_loaded_for_activation_capture_only": True,
        "historical_model_forward_used_only_for_compatibility_analysis": True,
        "historical_model_forward_used_by_conditional_architecture_pilot": False,
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "nas_started": False,
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "pairs": [{"source": row["source_task"], "target": row["target_task"], "type": row["intervention_type"]} for row in summaries],
    }
    (OUT / "compatibility_missing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def _aggregate(kind: str, source_task: str, target_task: str, candidate_id: str, packs: list[dict[str, Any]], scores: list[dict[str, Any]], generations: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = phase6a._generation_metrics(target_task, packs, generations)
    token_nll = sum(row["nll_sum"] for row in scores) / max(1, sum(row["token_count"] for row in scores))
    confirmation = json.loads((phase6a.OUT_ROOT / "confirmation_summary.json").read_text(encoding="utf-8"))
    baseline = confirmation["activation"][target_task]["baseline"]
    primary_key = "rouge_l_f1" if target_task in phase6a.GENERATION_TASKS else "accuracy"
    condition = f"{candidate_id}_from_{source_task}"
    return {
        "intervention_type": kind,
        "source_task": source_task,
        "target_task": target_task,
        "condition": condition,
        "candidate_id": candidate_id,
        "source_checkpoint": (phase6a.STAGE2 if source_task in phase6a.GENERATION_TASKS else phase6a.STAGE3).relative_to(ROOT).as_posix(),
        "examples": len(packs),
        "teacher_forced_token_nll": token_nll,
        "teacher_forced_nll_delta_vs_target_baseline": token_nll - float(baseline["teacher_forced_token_nll"]),
        "primary_metric": primary_key,
        "primary_metric_value": metrics[primary_key],
        "primary_metric_delta_vs_target_baseline": metrics[primary_key] - float(baseline[primary_key]),
        "mean_generated_token_count_including_eos": metrics["mean_generated_token_count_including_eos"],
        "exact_yes_output_rate": metrics["exact_yes_output_rate"],
        "exact_no_output_rate": metrics["exact_no_output_rate"],
        "exact_yes_no_output_rate": metrics["exact_yes_no_output_rate"],
        "invalid_answer_format_rate": metrics["invalid_answer_format_rate"],
    }


if __name__ == "__main__":
    main()
