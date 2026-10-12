"""Audit Phase 5B artifacts, split boundaries, hooks, and output metric semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase5_common import TASK_1384, TASK_1501, TASK_1603, TASK_565, TASK_IDS, load_phase5_context  # noqa: E402
from phase5b_common import generated_label_metrics  # noqa: E402


ROOT = storage_path("results") / "phase5b"
EVAL_ROOT = ROOT / "interventions" / "evaluation"


def _close(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10)


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"Phase 5B audit failed: {name}: {detail}")

    replay = json.loads((ROOT / "boundary_replay_manifest.json").read_text(encoding="utf-8"))
    check("same_trajectory_boundary_replay_complete", replay.get("status") == "complete")
    check("replayed_final_weights_match_original", replay["final_weight_comparison"].get("matches_within_tolerance") is True,
          replay["final_weight_comparison"])
    check("replay_did_not_load_test", replay.get("test_split_loaded") is False)
    check("no_nas_started", replay.get("nas_started") is False)

    vectors = __import__("torch").load(ROOT / "extraction" / "historical_function_vectors.pt", map_location="cpu", weights_only=True)
    expected_vectors = {
        "historical_after_task565": TASK_565,
        "historical_after_task1603": TASK_1603,
    }
    for checkpoint, task_id in expected_vectors.items():
        vector = vectors[checkpoint][task_id]
        check(f"historical_fv_shape_{task_id}", tuple(vector.shape) == (1024,) and bool(vector.isfinite().all()))
        folder = ROOT / "extraction" / "head_effects" / checkpoint / task_id
        effects = sorted(folder.glob("query_*.json"))
        complete = len(effects) == 10
        for path in effects:
            item = json.loads(path.read_text(encoding="utf-8"))
            heads = {(int(row["layer"]), int(row["query_head"])) for row in item.get("scores", [])}
            complete &= item.get("candidate_scope") == "global_448" and len(heads) == 448
        check(f"historical_global_448_queries_{task_id}", complete, {"query_files": len(effects)})

    head = json.loads((ROOT / "head_similarity" / "full_448_analysis.json").read_text(encoding="utf-8"))
    check("full_448_tasks_and_pairs", len(head.get("pairwise_null_details", {})) == 6 and len(head.get("tasks", [])) == 4)
    check("head_null_repeats", head.get("permutation_repeats") == 10000)
    check("head_analysis_test_not_loaded", head.get("test_split_loaded") is False)

    preflight = json.loads((ROOT / "interventions" / "preflight.json").read_text(encoding="utf-8"))
    check("injection_site_is_qwen3_attention_o_proj", preflight.get("injection_site") == "selected_decoder_layer.self_attn.o_proj.output.additive_v1", preflight.get("injection_site"))
    check("attention_output_hidden_size_1024", preflight.get("attention_output_hidden_size") == 1024)

    _, _, _, _, train, dev = load_phase5_context()
    train_ids = {task: {str(row["instance_id"]) for row in train[task]} for task in TASK_IDS}
    metrics = json.loads((ROOT / "interventions" / "metrics.json").read_text(encoding="utf-8"))
    summaries = {(row["task_id"], row["condition"]): row for row in metrics["rows"]}
    expected_conditions = {
        TASK_565: {"baseline", "historical_task_boundary_fv", "final_collapsed_checkpoint_fv", "norm_matched_random_fv", "unrelated_historical_task_fv", "subtract_latest_classifier_fv", "explicit_format_hint", "correct_four_shot_icl"},
        TASK_1603: {"baseline", "historical_task_boundary_fv", "final_collapsed_checkpoint_fv", "norm_matched_random_fv", "unrelated_historical_task_fv", "subtract_latest_classifier_fv", "explicit_format_hint", "correct_four_shot_icl"},
        TASK_1501: {"baseline", "final_collapsed_checkpoint_fv", "norm_matched_random_fv", "unrelated_historical_task_fv", "subtract_latest_classifier_fv", "explicit_format_hint", "correct_four_shot_icl"},
        TASK_1384: {"baseline", "final_collapsed_checkpoint_fv", "norm_matched_random_fv", "unrelated_historical_task_fv", "subtract_latest_classifier_fv", "explicit_format_hint", "correct_four_shot_icl"},
    }
    for task_id in TASK_IDS:
        files = {path.stem: path for path in (EVAL_ROOT / task_id).glob("*.json")}
        check(f"evaluation_condition_coverage_{task_id}", set(files) == expected_conditions[task_id], sorted(files))
        expected_ids = [row["instance_id"] for row in dev[task_id][25:50]]
        for condition in sorted(expected_conditions[task_id]):
            payload = json.loads(files[condition].read_text(encoding="utf-8"))
            check(f"locked_ids_{task_id}_{condition}", payload.get("evaluation_ids") == expected_ids)
            preds = payload["predictions"]
            scores = payload["nll_rows"]
            check(f"locked_row_counts_{task_id}_{condition}", len(preds) == 25 and len(scores) == 25)
            check(f"paired_nll_ids_{task_id}_{condition}", [p["instance_id"] for p in preds] == [s["instance_id"] for s in scores])
            if condition == "correct_four_shot_icl":
                valid_supports = all(
                    row.get("shots") == 4
                    and len(row.get("support_train_ids", [])) == 4
                    and len(set(row["support_train_ids"])) == 4
                    and set(row["support_train_ids"]).issubset(train_ids[task_id])
                    for row in preds
                )
                check(f"four_correct_train_shots_{task_id}", valid_supports)
            if condition not in {"baseline", "explicit_format_hint", "correct_four_shot_icl"}:
                hook_ok = all(
                    isinstance(row.get("generation_intervention_hook"), dict)
                    and row["generation_intervention_hook"].get("mode") == "last_each_cached_step"
                    and row["generation_intervention_hook"].get("hook_point") == "selected_decoder_layer.self_attn.o_proj.output.additive_v1"
                    and score.get("teacher_forced_intervention_hook", {}).get("mode") == "suffix"
                    and score.get("teacher_forced_intervention_hook", {}).get("hook_point") == "selected_decoder_layer.self_attn.o_proj.output.additive_v1"
                    for row, score in zip(preds, scores)
                )
                check(f"generation_and_teacher_forcing_hook_audit_{task_id}_{condition}", hook_ok)
            computed = generated_label_metrics([str(row["prediction"]) for row in preds])
            summary = summaries[(task_id, condition)]
            for key in (
                "exact_yes_output_rate", "exact_no_output_rate", "exact_yes_no_output_rate",
                "most_common_label_concentration_among_parseable", "parseable_yes_no_label_rate",
                "invalid_answer_format_rate",
            ):
                check(f"metric_recomputed_{task_id}_{condition}_{key}", _close(summary[key], computed[key]),
                      {"summary": summary[key], "recomputed": computed[key]})

    check("phase5b_metrics_have_no_ambiguous_legacy_key", all("yes_no_collapse_rate" not in row for row in metrics["rows"]))
    check("phase5b_only_dev_eval", metrics.get("test_split_loaded") is False)
    check("direct_specificity_contrasts_present", len(metrics.get("direct_historical_fv_vs_control_bootstrap", [])) == 24)
    for task_id in (TASK_565, TASK_1603):
        baseline_payload = json.loads((EVAL_ROOT / task_id / "baseline.json").read_text(encoding="utf-8"))
        baseline_predictions = {row["instance_id"]: row["prediction"] for row in baseline_payload["predictions"]}
        all_identical = True
        for path in (EVAL_ROOT / task_id).glob("*.json"):
            if path.stem in {"baseline", "final_collapsed_checkpoint_fv"} and task_id == TASK_1603:
                continue
            if path.stem == "baseline":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_identical &= all(
                row["prediction"] == baseline_predictions[row["instance_id"]]
                for row in payload["predictions"]
            )
        check(f"old_generation_predictions_identical_to_baseline_{task_id}", all_identical)
    task1603_base = json.loads((EVAL_ROOT / TASK_1603 / "baseline.json").read_text(encoding="utf-8"))["predictions"]
    task1603_final = json.loads((EVAL_ROOT / TASK_1603 / "final_collapsed_checkpoint_fv.json").read_text(encoding="utf-8"))["predictions"]
    final_flips = sum(left["prediction"] != right["prediction"] for left, right in zip(task1603_base, task1603_final))
    check("task1603_final_fv_one_label_flip", final_flips == 1, {"changed_predictions": final_flips})

    report_path = storage_path("reports") / "phase5b_fv_recovery.md"
    check("final_report_exists", report_path.is_file())
    check("canonical_full_run_config_exists", (ROOT / "run_config.json").is_file())
    output = {"phase": "5B", "status": "passed", "check_count": len(checks), "checks": checks, "test_split_loaded": False}
    (ROOT / "audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "check_count": len(checks), "audit": str(ROOT / "audit.json")}, indent=2))


if __name__ == "__main__":
    main()
