"""Audit Phase 6A result completeness, split use, controls, and figure outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
RESULTS = storage_path("results") / "phase6a"
FIGURES = storage_path("figures") / "phase6a"
REPORTS = storage_path("reports")
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    checks = []

    def check(name: str, passed: bool, detail: str):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    replay = json.loads((RESULTS / "checkpoints" / "after_task1501_replay_manifest.json").read_text(encoding="utf-8"))
    check("historical task1501 checkpoint replay", replay.get("status") == "complete" and replay.get("checks", {}).get("sample_order_sha256_match") and replay.get("checks", {}).get("supervised_target_tokens_match") and replay.get("checks", {}).get("optimizer_steps_match") and replay.get("loss_absolute_differences", {}).get("mean_loss") == 0.0, "50-step replay matches recorded order, token count, step count, and losses")
    check("no test split loaded", replay.get("test_split_loaded") is False, "replay manifest marks test split as not loaded")
    unit_tests = json.loads((RESULTS / "unit_test_results.json").read_text(encoding="utf-8"))
    check("unit test suite", unit_tests.get("exit_code") == 0 and unit_tests.get("result", "").split(" ")[0].isdigit() and unit_tests.get("focused_phase6a_result", "").split(" ")[0].isdigit(), f"repository suite: {unit_tests.get('result')}; focused Phase 6A suite: {unit_tests.get('focused_phase6a_result')}")

    expected_counts = {
        "activation_screen.jsonl": 14480,
        "parameter_screen.jsonl": 22440,
        "activation_confirmation.jsonl": 225,
        "parameter_confirmation.jsonl": 300,
        "activation_synergy.jsonl": 560,
        "checkpoint_trajectory.jsonl": 300,
        "cross_task_transfer.jsonl": 400,
    }
    counts = {}
    for name, expected in expected_counts.items():
        actual = len(read_jsonl(RESULTS / name))
        counts[name] = actual
        check(f"row count: {name}", actual == expected, f"{actual} rows; expected {expected}")

    screen_summary = read_csv(RESULTS / "activation_screen_summary.csv")
    no_ops = [row for row in screen_summary if row["candidate_id"] == "control_zero_no_op"]
    check("activation zero-operation control", len(no_ops) == len(TASKS) and all(abs(float(row["nll_improvement_vs_baseline"])) < 1e-9 for row in no_ops), "paired NLL change is zero for all four tasks")
    param_summary = json.loads((RESULTS / "confirmation_summary.json").read_text(encoding="utf-8"))
    random_matches = []
    for task in TASKS:
        conditions = param_summary["parameter_rollback"][task]
        rollback_key = next(key for key in conditions if key != "matched_random_delta")
        rollback_l2 = float(conditions[rollback_key]["update_l2_norm"])
        random_l2 = float(conditions["matched_random_delta"]["update_l2_norm"])
        random_matches.append(abs(rollback_l2 - random_l2) <= max(1e-7, rollback_l2 * 1e-7))
    check("parameter random-direction update norm", all(random_matches), "matched control preserves each selected rollback group's L2 update norm")
    model_config = json.loads((RESULTS / "checkpoints" / "after_3_task1501_dstc3_answer_generation" / "config.json").read_text(encoding="utf-8"))
    parameter_screen = read_csv(RESULTS / "parameter_screen_summary.csv")
    parameter_candidate_ids = {row["candidate_id"] for row in parameter_screen}
    projection_ids = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    projection_count = sum(any(proj in candidate_id for proj in projection_ids) for candidate_id in parameter_candidate_ids)
    check("projection rollback screen", len(parameter_screen) == 2244 and projection_count == 203, f"{len(parameter_screen)} task/candidate summary rows; {projection_count} distinct projection groups per task set")
    tied_groups = [row for row in parameter_screen if row["candidate_id"] == "embedding_lm_head_global"]
    tied_count_ok = len(tied_groups) == len(TASKS) and all(int(float(row["parameter_count"])) == 155_582_464 for row in tied_groups)
    check("tied embedding/LM Head handling", model_config.get("tie_word_embeddings") is True and tied_count_ok, "checkpoint config ties the weights; rollback screen contains one 155,582,464-parameter shared group")

    runtimes = [
        json.loads((RESULTS / f"{stage}_runtime.json").read_text(encoding="utf-8"))
        for stage in ("activation_screen", "parameter_screen", "synergy", "trajectory", "confirm", "cross_task")
    ]
    check("all causal runs are inference-only", all(row.get("optimizer_steps") == 0 and not row.get("test_split_loaded") and not row.get("nas_started") for row in runtimes), "no causal-stage optimizer steps, test split loads, or NAS start")
    peak = max(row["peak_allocated_bytes"] for row in runtimes)
    check("RTX 4090 memory headroom", peak < 24 * 1024**3, f"peak allocated {peak / 1024**3:.2f} GiB < 24 GiB")
    cross_summary = read_csv(RESULTS / "cross_task_transfer_summary.csv")
    cross_manifest = json.loads((RESULTS / "cross_task_transfer_manifest.json").read_text(encoding="utf-8"))
    cross_count_ok = len(cross_summary) == 16 and len(read_jsonl(RESULTS / "cross_task_transfer.jsonl")) == 400
    manifest_count = cross_manifest.get("source_to_target_pairs", cross_manifest.get("source_to_target_task_pairs", cross_manifest.get("source_to_target_condition_count")))
    if isinstance(manifest_count, list):
        manifest_count = len(manifest_count)
    check("cross-task transfer completeness", cross_count_ok and manifest_count == 16 and cross_manifest.get("test_split_loaded") is False and cross_manifest.get("optimizer_steps") == 0 and cross_manifest.get("nas_started") is False, "16 transfer conditions and 400 locked-dev per-example rows; test/NAS/optimizer flags clear")

    expected_outputs = [
        RESULTS / "activation_screen_selection.json", RESULTS / "parameter_screen_selection.json",
        RESULTS / "confirmation_summary.json", RESULTS / "confirmation_paired_deltas.csv",
        RESULTS / "activation_synergy_summary.csv", RESULTS / "checkpoint_trajectory_summary.csv", RESULTS / "unit_test_results.json",
        RESULTS / "cross_task_transfer_summary.csv", RESULTS / "cross_task_transfer_manifest.json",
        REPORTS / "phase6a_component_localization.md", REPORTS / "phase6a_architecture_implications.md",
        FIGURES / "activation_single_layer_heatmap.png", FIGURES / "parameter_rollback_tradeoff.png",
        FIGURES / "locked_confirmation_metrics.png", FIGURES / "attention_mlp_synergy.png",
        FIGURES / "checkpoint_trajectory_nll.png", FIGURES / "binary_margin_layer_shift.png",
        FIGURES / "cross_task_transfer_heatmap.png",
    ]
    missing_or_empty = [str(path.relative_to(PROJECT_ROOT)) for path in expected_outputs if not path.is_file() or path.stat().st_size == 0]
    check("reports, summaries, and figures exist", not missing_or_empty, "all listed artifacts are nonempty" if not missing_or_empty else f"missing/empty: {missing_or_empty}")

    result = {
        "phase": "6A",
        "status": "complete" if all(row["passed"] for row in checks) else "audit_failed",
        "test_split_loaded": False,
        "new_task_sequence_training": False,
        "optimizer_steps_in_causal_analysis": 0,
        "nas_started": False,
        "peak_allocated_bytes": peak,
        "raw_row_counts": counts,
        "checks": checks,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase6a_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Phase 6A Audit",
        "",
        f"**Status:** {result['status']}",
        "",
        "| Check | Result | Evidence |",
        "|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
    lines.extend([
        "",
        "Causal interventions used no optimizer updates and loaded no test split. A separate exact replay of the original task1501 50-step stage supplied the missing historical boundary; the replay manifest records exact matching losses and training order.",
        "",
        "Machine-readable evidence: [phase6a_audit.json](../results/phase6a/phase6a_audit.json).",
    ])
    (REPORTS / "phase6a_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
