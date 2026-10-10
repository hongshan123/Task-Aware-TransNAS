"""Rebuild cross-task transfer summaries from saved per-example outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from phase5b_common import generated_label_metrics
from run_formal_baseline import parse_yes_no
from run_phase6a_localization import TASKS, GENERATION_TASKS, activation_candidates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results" / "phase6a"
STAGE2 = "results/phase5b/checkpoints/historical_boundary_replay/A0/S1/seed_10/after_2_task1603_smcalflow_sentence_generation"
STAGE3 = "results/phase6a/checkpoints/after_3_task1501_dstc3_answer_generation"


def main():
    raw_path = OUT_ROOT / "cross_task_transfer.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["intervention_type"], row["source_task"], row["target_task"], row["condition"])].append(row)
    baseline_summary = json.loads((OUT_ROOT / "confirmation_summary.json").read_text(encoding="utf-8"))
    param_screen = list(csv.DictReader((OUT_ROOT / "parameter_screen_summary.csv").open(encoding="utf-8", newline="")))
    activation_by_id = {row["candidate_id"]: row for row in activation_candidates(28)}
    output = []
    for (kind, source_task, target_task, condition), values in groups.items():
        source_checkpoint = STAGE2 if source_task in GENERATION_TASKS else STAGE3
        baseline = baseline_summary["activation"][target_task]["baseline"]
        nll = sum(float(row["nll_sum"]) for row in values) / max(1, sum(int(row["token_count"]) for row in values))
        predictions = [str(row["prediction"]) for row in values]
        labels = generated_label_metrics(predictions)
        if target_task in GENERATION_TASKS:
            primary_metric = "rouge_l_f1"
            primary = sum(float(row.get("rouge_l_f1", 0.0)) for row in values) / max(1, len(values))
        else:
            primary_metric = "accuracy"
            primary = sum(bool(row.get("correct", False)) for row in values) / max(1, len(values))
        result = {
            "intervention_type": kind, "source_task": source_task,
            "target_task": target_task, "condition": condition,
            "source_checkpoint": source_checkpoint,
            "examples": len(values),
            "teacher_forced_token_nll": nll,
            "teacher_forced_nll_delta_vs_target_baseline": nll - float(baseline["teacher_forced_token_nll"]),
            "primary_metric": primary_metric,
            "primary_metric_value": primary,
            "primary_metric_delta_vs_target_baseline": primary - float(baseline[primary_metric]),
            "mean_generated_token_count_including_eos": sum(int(row["generated_token_count_including_eos"]) for row in values) / max(1, len(values)),
            "exact_yes_output_rate": labels["exact_yes_output_rate"],
            "exact_no_output_rate": labels["exact_no_output_rate"],
            "exact_yes_no_output_rate": labels["exact_yes_no_output_rate"],
            "invalid_answer_format_rate": labels["invalid_answer_format_rate"],
        }
        if kind == "activation":
            candidate_id = condition.rsplit("_from_", 1)[0]
            metadata = activation_by_id[candidate_id]
            result["component"] = metadata["component"]
            result["layers"] = metadata.get("layers")
        else:
            candidate_id = condition.rsplit("_from_", 1)[0]
            screen_row = next(row for row in param_screen if row["task_id"] == source_task and row["candidate_id"] == candidate_id)
            result.update({
                "parameter_count": int(float(screen_row["parameter_count"])),
                "rollback_fraction_of_model": float(screen_row["rollback_fraction_of_model"]),
                "update_l2_norm": float(screen_row["update_l2_norm"]),
                "update_rms": float(screen_row["update_rms"]),
            })
        output.append(result)
    output.sort(key=lambda row: (row["intervention_type"], row["source_task"], row["target_task"]))
    with (OUT_ROOT / "cross_task_transfer_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(dict.fromkeys(key for row in output for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    manifest = {
        "source_to_target_condition_count": len(output),
        "raw_example_rows": len(rows),
        "evaluation_dev_indices": [25, 49],
        "test_split_loaded": False,
        "optimizer_steps": 0,
        "nas_started": False,
        "source_to_target_pairs": [
            {"type": row["intervention_type"], "source": row["source_task"], "target": row["target_task"], "condition": row["condition"]}
            for row in output
        ],
        "reconstructed_summary_from_saved_raw_rows": True,
    }
    (OUT_ROOT / "cross_task_transfer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"reconstructed {len(output)} cross-task condition summaries from {len(rows)} saved rows")


if __name__ == "__main__":
    main()
