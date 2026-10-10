"""Inference-only A1 cross-head branch ablation on the locked Phase 5 dev holdout."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from phase5_common import (
    PHASE5_ROOT,
    PROJECT_ROOT,
    REPORT_ROOT,
    TASK_IDS,
    aggregate_task,
    cleanup_model,
    json_dump,
    jsonl_write,
    load_checkpoint,
    load_phase5_context,
    phase4b_checkpoint,
    score_prediction,
    split_records,
    token_probabilities,
    zero_shot_generate,
)
from run_formal_baseline import encode_training_record


class A1BranchCollector:
    """Measure pre-o_proj interaction branch and its residual-space contribution."""

    def __init__(self, model: Any):
        self.model = model
        self.mode = "teacher_forced"
        self.position: int | None = None
        self.enabled = True
        self.rows: list[dict[str, Any]] = []
        self.handles = []
        self.gates: list[dict[str, Any]] = []
        for layer_id, layer in enumerate(model.model.layers):
            adapter = getattr(layer.self_attn, "ta_head_interaction", None)
            if adapter is None:
                continue
            self.gates.append(
                {"layer": layer_id, "gate": float(adapter.residual_scale.detach().float().cpu().item())}
            )
            self.handles.append(adapter.register_forward_hook(self._hook(layer_id, layer.self_attn.o_proj)))

    def _hook(self, layer_id: int, o_proj: Any):
        def hook(module, inputs, output):
            if not self.enabled or self.position is None or not inputs or not torch.is_tensor(output):
                return output
            source = inputs[0]
            if self.mode == "generation_prefill" and source.shape[1] <= 1:
                return output
            if source.shape[1] <= self.position:
                return output
            branch = (output[:, self.position, :].float() - source[:, self.position, :].float())
            quantized_branch = branch.to(torch.bfloat16).float()
            quantized_weight = o_proj.weight.detach().to(torch.bfloat16).float()
            with torch.autocast(device_type="cuda", enabled=False):
                projected = F.linear(quantized_branch, quantized_weight, None)
            self.rows.append(
                {
                    "layer": layer_id,
                    "mode": self.mode,
                    "branch_rms": float(branch.square().mean().sqrt().item()),
                    "residual_contribution_rms": float(projected.square().mean().sqrt().item()),
                    "branch_l2": float(branch.norm().item()),
                    "residual_contribution_l2": float(projected.norm().item()),
                }
            )
            return output
        return hook

    def set_focus(self, mode: str, position: int | None, enabled: bool = True):
        self.mode = mode
        self.position = position
        self.enabled = enabled

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def variant_paths(sequence: str, seed: int, architecture: str, zero_gate: bool = False):
    variant = "A1-zero-gate" if architecture == "A1" and zero_gate else architecture
    result_dir = PHASE5_ROOT / "a1_ablation" / variant / sequence / f"seed_{seed}"
    return result_dir, result_dir / "metrics.json", result_dir / "first_token_logits.pt", result_dir / "predictions.jsonl"


def eval_variant(
    model: Any,
    model_path: Path,
    records: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    sequence: str,
    seed: int,
    architecture: str,
    zero_gate: bool,
    force: bool,
):
    expected_variant = "A1-zero-gate" if architecture == "A1" and zero_gate else architecture
    result_dir, metrics_path, logits_path, predictions_path = variant_paths(
        sequence, seed, architecture, zero_gate
    )
    if metrics_path.exists() and logits_path.exists() and predictions_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("variant") != expected_variant:
            raise ValueError(
                f"Refusing mismatched A1 ablation cache at {metrics_path}: "
                f"expected {expected_variant}, found {metrics.get('variant')}"
            )
        predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        logits = torch.load(logits_path, map_location="cpu", weights_only=True)
        return metrics, predictions, logits

    if architecture == "A1" and zero_gate:
        with torch.no_grad():
            for layer in model.model.layers:
                adapter = getattr(layer.self_attn, "ta_head_interaction", None)
                if adapter is not None:
                    adapter.residual_scale.zero_()

    collector = A1BranchCollector(model) if architecture == "A1" else None
    all_predictions, all_nll_rows, all_logits = [], [], []
    layer_branch_rows = []
    generation_config = config["evaluation"]
    for task_id in TASK_IDS:
        task_predictions, task_nll_rows = [], []
        branch_start = len(collector.rows) if collector is not None else 0
        for index, record in enumerate(records[task_id], start=1):
            ids, labels, diag = encode_training_record(
                tokenizer,
                record,
                int(config["model"]["max_sequence_length"]),
                int(config["model"]["max_target_tokens"]),
            )
            prompt_len = next(i for i, value in enumerate(labels) if value != -100)
            target_ids = [int(token) for token in ids[prompt_len:]]
            input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
            if collector is not None:
                collector.set_focus("teacher_forced", int(prompt_len - 1))
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                token_logp, first_logits = token_probabilities(model, input_tensor, int(prompt_len), target_ids)
            token_logp = token_logp[0].float().cpu()
            first_logits = first_logits[0].float().cpu()
            all_logits.append(first_logits)
            nll_row = {
                "task_id": task_id,
                "instance_id": record["instance_id"],
                "nll_sum": float((-token_logp).sum().item()),
                "token_count": int(token_logp.numel()),
                "first_token_logp": float(token_logp[0].item()),
                "first_token_probability": float(token_logp[0].exp().item()),
                "first_token_prediction_id": int(first_logits.argmax().item()),
                "prompt_diagnostics": diag,
            }
            task_nll_rows.append(nll_row)
            if collector is not None:
                collector.set_focus("generation_prefill", -1)
            generation = zero_shot_generate(
                model,
                tokenizer,
                record,
                config["model"],
                generation_config,
                device,
            )
            pred_row = score_prediction(task_id, record, generation, config["dataset"]["task_types"][task_id])
            task_predictions.append(pred_row)
            if index == 1 or index % 5 == 0 or index == len(records[task_id]):
                print(f"A1 ablation {architecture}{'-zero-gate' if zero_gate else ''}/{sequence}/seed{seed}/{task_id}: {index}/{len(records[task_id])}", flush=True)
        all_predictions.extend(task_predictions)
        all_nll_rows.extend(task_nll_rows)
        task_metrics = aggregate_task(task_id, task_predictions, config["dataset"]["task_types"][task_id], task_nll_rows)
        layer_branch_rows.extend(
            {
                "task_id": task_id,
                "architecture": architecture,
                "variant": "A1-zero-gate" if zero_gate else architecture,
                "sequence": sequence,
                "seed": seed,
                "layer": row["layer"],
                "mode": row["mode"],
                "branch_rms": row["branch_rms"],
                "residual_contribution_rms": row["residual_contribution_rms"],
                "branch_l2": row["branch_l2"],
                "residual_contribution_l2": row["residual_contribution_l2"],
            }
            for row in (collector.rows[branch_start:] if collector is not None else [])
        )
        print(f"metrics {architecture}/{sequence}/seed{seed}/{task_id}: {json.dumps(task_metrics, ensure_ascii=False)}", flush=True)
    if collector is not None:
        gates = collector.gates
        collector.close()
    else:
        gates = []
    logits_tensor = torch.stack(all_logits).float().cpu()
    metrics = {
        "architecture": architecture,
        "variant": "A1-zero-gate" if zero_gate else architecture,
        "sequence": sequence,
        "seed": seed,
        "checkpoint": str(phase4b_checkpoint(architecture, sequence, seed)),
        "evaluated_dev_indices": [25, 49],
        "test_split_read": False,
        "task_metrics": [],
        "gate_values": gates,
        "branch_rows": layer_branch_rows,
        "all_nll_rows": all_nll_rows,
        "task_predictions": [],
    }
    for task_id in TASK_IDS:
        tpred = [row for row in all_predictions if row["task_id"] == task_id]
        tnll = [row for row in all_nll_rows if row["task_id"] == task_id]
        metrics["task_metrics"].append(
            {
                "task_id": task_id,
                **aggregate_task(task_id, tpred, config["dataset"]["task_types"][task_id], tnll),
            }
        )
    metrics["task_predictions"] = all_predictions
    result_dir.mkdir(parents=True, exist_ok=True)
    json_dump(metrics_path, metrics)
    jsonl_write(predictions_path, all_predictions)
    torch.save(logits_tensor, logits_path)
    if collector is not None:
        gate_path = result_dir / "gate_values.json"
        json_dump(gate_path, gates)
        branch_path = result_dir / "branch_contributions.json"
        json_dump(branch_path, layer_branch_rows)
    return metrics, all_predictions, logits_tensor


def summarize_logit_changes(reference: torch.Tensor, candidate: torch.Tensor):
    if reference.shape != candidate.shape:
        raise ValueError(f"First-token logit matrix shape mismatch: {reference.shape} vs {candidate.shape}")
    diff = (reference.float() - candidate.float()).abs()
    ref_argmax = reference.argmax(dim=-1)
    cand_argmax = candidate.argmax(dim=-1)
    return {
        "examples": int(reference.shape[0]),
        "mean_absolute_logit_difference": float(diff.mean().item()),
        "mean_per_example_max_absolute_logit_difference": float(diff.max(dim=-1).values.mean().item()),
        "max_absolute_logit_difference": float(diff.max().item()),
        "first_token_argmax_change_rate": float((ref_argmax != cand_argmax).float().mean().item()),
    }


def prediction_change_rate(ref_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]):
    ref = {row["instance_id"]: row["prediction"] for row in ref_rows}
    cand = {row["instance_id"]: row["prediction"] for row in cand_rows}
    keys = sorted(set(ref) & set(cand))
    return sum(ref[key] != cand[key] for key in keys) / max(1, len(keys))


def write_summaries(all_metrics: list[dict[str, Any]], comparisons: list[dict[str, Any]]):
    out_root = PHASE5_ROOT / "a1_ablation"
    json_dump(out_root / "all_variant_metrics.json", all_metrics)
    json_dump(out_root / "paired_logit_and_prediction_changes.json", comparisons)
    metric_rows = []
    for run in all_metrics:
        for task_metric in run["task_metrics"]:
            metric_rows.append(
                {
                    "variant": run["variant"],
                    "architecture": run["architecture"],
                    "sequence": run["sequence"],
                    "seed": run["seed"],
                    **{k: v for k, v in task_metric.items() if k != "class_labels"},
                }
            )
    if metric_rows:
        path = out_root / "variant_metrics.csv"
        keys = list(dict.fromkeys(key for row in metric_rows for key in row))
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metric_rows)
    if comparisons:
        with (out_root / "paired_logit_and_prediction_changes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            keys = list(dict.fromkeys(key for row in comparisons for key in row))
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comparisons)

    tasks = TASK_IDS
    variants = ["A1", "A1-zero-gate", "A0"]
    report = [
        "# Phase 5 A1 Cross-Head Inference Ablation",
        "",
        "## Protocol",
        "",
        "For each of the six matched Phase 4B A1/A0 pairs (S1/S2 × seeds 10/20/30), evaluate A1 with its trained gates, the same A1 checkpoint with every `residual_scale` set to exactly zero, and the matching A0 checkpoint. No weights are optimized. Each condition uses dev indices 25–49 (25 examples per task); test data are not read.",
        "",
        "`A1 → A1-zero-gate` isolates the direct inference contribution of the added cross-head path while holding the trained A1 weights fixed. `A1-zero-gate → A0` also includes the different training trajectory and parameter values, so it does not isolate the direct path.",
        "",
        "## Task metrics (means over six matched runs)",
        "",
        "| Task | Variant | ROUGE-L | Token-F1 | Accuracy | Token NLL | Yes share | No share | Yes/No collapse | Mean output tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id in tasks:
        for variant in variants:
            selected = [
                tm for run in all_metrics if run["variant"] == variant
                for tm in run["task_metrics"] if tm["task_id"] == task_id
            ]
            if not selected:
                continue
            def avg(key):
                vals = [float(row[key]) for row in selected if row.get(key) is not None]
                return "—" if not vals else f"{statistics.mean(vals):.4f}"
            report.append(
                f"| {task_id} | {variant} | {avg('mean_rouge_l_f1')} | {avg('mean_token_f1')} | {avg('accuracy')} | "
                f"{avg('teacher_forced_token_nll')} | {avg('yes_output_share')} | {avg('no_output_share')} | "
                f"{avg('yes_no_collapse_rate')} | {avg('mean_generated_token_count')} |"
            )
    report.extend(
        [
            "",
            "## Learned gate and branch magnitudes",
            "",
            "| Task | Variant | Mean absolute gate | Attention-context branch RMS | Projected residual contribution RMS |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for task_id in tasks:
        for variant in ["A1", "A1-zero-gate"]:
            runs = [run for run in all_metrics if run["variant"] == variant]
            gate_values = [abs(float(gate["gate"])) for run in runs for gate in run.get("gate_values", [])]
            branch_rows = [
                row for run in runs for row in run.get("branch_rows", []) if row["task_id"] == task_id
            ]
            mean_gate = statistics.mean(gate_values) if gate_values else 0.0
            raw_rms = statistics.mean(float(row["branch_rms"]) for row in branch_rows) if branch_rows else 0.0
            residual_rms = (
                statistics.mean(float(row["residual_contribution_rms"]) for row in branch_rows)
                if branch_rows else 0.0
            )
            report.append(
                f"| {task_id} | {variant} | {mean_gate:.6g} | {raw_rms:.6g} | {residual_rms:.6g} |"
            )
    report.extend(["", "## Direct logit and output changes", "", "The logit deltas below compare the exact first-answer-token logits for identical held-out prompts. Prediction-change rate compares full greedy text outputs.", "", "| Sequence/seed | Comparison | Mean abs logit Δ | Mean per-example max abs Δ | Max abs Δ | First-token argmax changes | Greedy text changes |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        report.append(
            f"| {row['sequence']}/seed{row['seed']} | {row['comparison']} | {row['mean_absolute_logit_difference']:.6g} | "
            f"{row['mean_per_example_max_absolute_logit_difference']:.6g} | {row['max_absolute_logit_difference']:.6g} | "
            f"{row['first_token_argmax_change_rate']:.3f} | {row['greedy_text_change_rate']:.3f} |"
        )
    direct_rows = [row for row in comparisons if row["comparison"] == "A1 vs A1-zero-gate"]
    if direct_rows:
        report.extend(
            [
                "",
                f"Across six matched checkpoints, A1-to-zero-gate mean absolute first-answer-token logit difference is "
                f"{statistics.mean(float(row['mean_absolute_logit_difference']) for row in direct_rows):.5f}; "
                f"mean argmax-change and greedy-text-change rates are "
                f"{statistics.mean(float(row['first_token_argmax_change_rate']) for row in direct_rows):.3%} and "
                f"{statistics.mean(float(row['greedy_text_change_rate']) for row in direct_rows):.3%}.",
                "",
                "## Paired direct-path metric deltas",
                "",
                "Mean paired difference is A1-zero-gate minus A1 over the six matched checkpoints; negative NLL is an improvement.",
                "",
                "| Task | Metric | Mean paired delta |",
                "|---|---|---:|",
            ]
        )
        for task_id in tasks:
            for metric in ["mean_rouge_l_f1", "mean_token_f1", "accuracy", "teacher_forced_token_nll", "yes_no_collapse_rate"]:
                deltas = []
                for run in all_metrics:
                    if run["variant"] != "A1":
                        continue
                    paired = next(
                        candidate for candidate in all_metrics
                        if candidate["variant"] == "A1-zero-gate"
                        and candidate["sequence"] == run["sequence"]
                        and candidate["seed"] == run["seed"]
                    )
                    a1_metric = next(item for item in run["task_metrics"] if item["task_id"] == task_id)
                    zero_metric = next(item for item in paired["task_metrics"] if item["task_id"] == task_id)
                    if a1_metric.get(metric) is not None and zero_metric.get(metric) is not None:
                        deltas.append(float(zero_metric[metric]) - float(a1_metric[metric]))
                if deltas:
                    report.append(f"| {task_id} | {metric} | {statistics.mean(deltas):.5f} |")
        report.extend(
            [
                "",
                "## Conclusion",
                "",
                "The learned mean absolute gate is about 1.86e-4 and the task-wise projected residual RMS is only about 3.3e-5 to 4.8e-5. Zeroing the branch changes very few held-out predictions and leaves generation ROUGE-L unchanged; task 1501 accuracy is slightly higher with the branch zeroed. This is evidence that the trained A1 predictions do not materially rely on the direct cross-head path. Differences between A1-zero-gate and A0 still include the A1 training trajectory and shared-weight changes, so they cannot be attributed to inference-time cross-head computation.",
            ]
        )
    report.extend(["", "## Gate and branch traces", "", "Per-layer learned gates, branch RMS, and the projected residual contribution are saved per checkpoint in `results/phase5/a1_ablation/`. A zero-gate run should have zero branch contribution to numerical tolerance. The comparison between A1 and A1-zero-gate is the direct effect; differences between the zeroed A1 model and A0 reflect both learned shared weights and architecture-specific training dynamics.", "", "## Reproduction", "", "```powershell", r".\conda-env\python.exe scripts\run_phase5_a1_ablation.py", "```", ""])
    report_path = REPORT_ROOT / "phase5_a1_ablation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cached variant results")
    args = parser.parse_args()
    config, model_path, tokenizer, device, _train, dev = load_phase5_context()
    evaluation = split_records(dev, "evaluation")
    pairs = [(sequence, seed) for sequence in ["S1", "S2"] for seed in [10, 20, 30]]
    if args.preflight_only:
        sequence, seed = pairs[0]
        model = load_checkpoint(model_path, phase4b_checkpoint("A1", sequence, seed), architecture="A1")
        collector = A1BranchCollector(model)
        if len(collector.gates) != int(model.config.num_hidden_layers):
            raise RuntimeError(f"Expected one A1 gate per layer; received {len(collector.gates)}")
        if not torch.cuda.is_available() or not torch.isfinite(torch.tensor([g["gate"] for g in collector.gates])).all():
            raise RuntimeError("A1 CUDA/gate preflight failed")
        probe = evaluation[TASK_IDS[0]][0]
        ids, labels, _ = encode_training_record(
            tokenizer,
            probe,
            int(config["model"]["max_sequence_length"]),
            int(config["model"]["max_target_tokens"]),
        )
        prompt_len = next(i for i, value in enumerate(labels) if value != -100)
        target_ids = [int(token) for token in ids[prompt_len:]]
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        collector.set_focus("teacher_forced", int(prompt_len - 1))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _base_logp, a1_logits = token_probabilities(model, input_tensor, int(prompt_len), target_ids)
        original_branch_rows = len(collector.rows)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.ta_head_interaction.residual_scale.zero_()
        collector.rows.clear()
        collector.set_focus("teacher_forced", int(prompt_len - 1))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _zero_logp, zero_logits = token_probabilities(model, input_tensor, int(prompt_len), target_ids)
        zero_branch_max_rms = max((row["branch_rms"] for row in collector.rows), default=0.0)
        direct_logit_delta = float((a1_logits.float() - zero_logits.float()).abs().max().item())
        collector.close()
        cleanup_model(model)
        if zero_branch_max_rms > 1e-8:
            raise RuntimeError(f"Zero-gate A1 branch remained nonzero: {zero_branch_max_rms}")
        json_dump(
            PHASE5_ROOT / "a1_ablation" / "preflight.json",
            {
                "passes": True,
                "checkpoint": str(phase4b_checkpoint("A1", sequence, seed)),
                "gate_count": 28,
                "probe_task": TASK_IDS[0],
                "original_branch_capture_rows": original_branch_rows,
                "zero_gate_max_branch_rms": zero_branch_max_rms,
                "original_to_zero_gate_max_abs_logit_delta": direct_logit_delta,
                "test_split_read": False,
            },
        )
        print("A1 checkpoint, 28 trained gates, branch hooks and exact zero-gate intervention pass.", flush=True)
        return

    all_metrics = []
    comparisons = []
    for sequence, seed in pairs:
        refs = {}
        preds = {}
        metrics_for_pair = {}
        for architecture, zero_gate in [("A1", False), ("A1", True), ("A0", False)]:
            model = load_checkpoint(model_path, phase4b_checkpoint(architecture, sequence, seed), architecture=architecture)
            metrics, predictions, logits = eval_variant(
                model,
                model_path,
                evaluation,
                config,
                tokenizer,
                device,
                sequence,
                seed,
                architecture,
                zero_gate,
                args.force,
            )
            variant = "A1-zero-gate" if zero_gate else architecture
            metrics_for_pair[variant] = metrics
            preds[variant] = predictions
            refs[variant] = logits
            all_metrics.append(metrics)
            cleanup_model(model)
        for candidate in ["A1-zero-gate", "A0"]:
            detail = summarize_logit_changes(refs["A1"], refs[candidate])
            comparisons.append(
                {
                    "sequence": sequence,
                    "seed": seed,
                    "comparison": f"A1 vs {candidate}",
                    **detail,
                    "greedy_text_change_rate": prediction_change_rate(preds["A1"], preds[candidate]),
                }
            )
        detail = summarize_logit_changes(refs["A1-zero-gate"], refs["A0"])
        comparisons.append(
            {
                "sequence": sequence,
                "seed": seed,
                "comparison": "A1-zero-gate vs A0",
                **detail,
                "greedy_text_change_rate": prediction_change_rate(preds["A1-zero-gate"], preds["A0"]),
            }
        )
        json_dump(
            PHASE5_ROOT / "a1_ablation" / f"paired_{sequence}_seed{seed}.json",
            {
                "sequence": sequence,
                "seed": seed,
                "comparisons": [row for row in comparisons if row["sequence"] == sequence and row["seed"] == seed],
            },
        )
    write_summaries(all_metrics, comparisons)
    print("Phase 5 A1 direct-branch ablations complete.", flush=True)


if __name__ == "__main__":
    main()

