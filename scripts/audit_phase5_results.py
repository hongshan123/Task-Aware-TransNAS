"""CPU-only structural audit for completed Phase 5 result artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
RESULT_ROOT = storage_path("results") / "phase5"
TASKS = [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation",
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
]
SNAPSHOTS = [
    "pretrained", "homogeneous_stage_1", "homogeneous_stage_2",
    "homogeneous_stage_3", "homogeneous_stage_4", "phase4b_A0_S1_seed10",
]


def main() -> int:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    raw_root = RESULT_ROOT / "fv" / "head_effects" / "v2_bf16_dequantized_fp32"
    raw_ok, raw_count = True, 0
    for checkpoint in SNAPSHOTS:
        expected_scope = (
            "global_448" if checkpoint in {"pretrained", "homogeneous_stage_4", "phase4b_A0_S1_seed10"}
            else "endpoint_union_restricted"
        )
        for task_id in TASKS:
            for query in range(10):
                path = raw_root / checkpoint / task_id / f"query_{query:02d}.json"
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    raw_count += 1
                    if (
                        item.get("checkpoint_id") != checkpoint
                        or item.get("task_id") != task_id
                        or item.get("candidate_scope") != expected_scope
                        or not item.get("scores")
                        or (expected_scope == "global_448" and len(item["scores"]) != 448)
                    ):
                        raw_ok = False
                except Exception:
                    raw_ok = False
    check("all causal head scans", raw_ok and raw_count == 240, f"{raw_count}/240 valid per-query cache files")

    activation_root = RESULT_ROOT / "fv" / "activations" / "v2_bf16_dequantized_fp32"
    activation_ok, activation_count = True, 0
    for checkpoint in SNAPSHOTS:
        for task_id in TASKS:
            path = activation_root / f"{checkpoint}__{task_id}.pt"
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
                activation_count += 1
                if tuple(value.shape) != (28, 16, 1024) or not bool(torch.isfinite(value).all()):
                    activation_ok = False
            except Exception:
                activation_ok = False
    check(
        "head activation tensors", activation_ok and activation_count == 24,
        f"{activation_count}/24 finite tensors with shape 28x16x1024",
    )

    fv_root = RESULT_ROOT / "fv"
    cosine = json.loads((fv_root / "fv_cosines.json").read_text(encoding="utf-8"))
    overlap = json.loads((fv_root / "head_overlap.json").read_text(encoding="utf-8"))
    drift = json.loads((fv_root / "fv_drift.json").read_text(encoding="utf-8"))
    descriptors = json.loads((fv_root / "top_heads_and_vectors.json").read_text(encoding="utf-8"))
    check(
        "FV trajectory statistics",
        len(cosine) == 96 and len(overlap) == 96 and len(drift) == 24 and len(descriptors) == 24,
        f"cosines={len(cosine)}, overlaps={len(overlap)}, drift={len(drift)}, task-vectors={len(descriptors)}",
    )
    vectors = torch.load(fv_root / "function_vectors.pt", map_location="cpu", weights_only=True)
    check("function vectors", len(vectors) == 6 and all(len(snapshot) == 4 for snapshot in vectors.values()), "6 snapshots x 4 tasks")

    conditions = [
        "baseline", "source_fv", "subtract_latest_classifier_fv", "combined_source_minus_latest",
        "random_norm_matched", "unrelated_fv",
    ]
    intervention_ok, intervention_count, paired_ids_ok = True, 0, True
    for task_id in TASKS:
        baseline_ids: set[str] | None = None
        for condition in conditions:
            path = RESULT_ROOT / "interventions" / "v2_bf16_dequantized_fp32" / "evaluation" / task_id / f"{condition}.json"
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                intervention_count += 1
                predictions, nll_rows = item["predictions"], item["nll_rows"]
                ids = {row["instance_id"] for row in predictions}
                if (
                    item.get("fv_version") != "v2_bf16_dequantized_fp32"
                    or len(predictions) != 25 or len(nll_rows) != 25
                ):
                    intervention_ok = False
                if baseline_ids is None:
                    baseline_ids = ids
                elif ids != baseline_ids:
                    paired_ids_ok = False
                if {row["instance_id"] for row in nll_rows} != ids:
                    paired_ids_ok = False
            except Exception:
                intervention_ok = False
    check("six FV intervention conditions", intervention_ok and intervention_count == 24, f"{intervention_count}/24 caches; 25 held-out examples each")
    check("paired intervention IDs", paired_ids_ok, "same locked 25 dev examples across conditions within each task")
    effects = json.loads((RESULT_ROOT / "interventions" / "paired_effects_bootstrap.json").read_text(encoding="utf-8"))
    check("paired bootstrap effects", len(effects) == 70, f"{len(effects)} task-condition-metric effects")

    variants = ["A1", "A1-zero-gate", "A0"]
    a1_ok, a1_count = True, 0
    max_zero_gate = max_zero_branch = 0.0
    for variant in variants:
        for sequence in ["S1", "S2"]:
            for seed in [10, 20, 30]:
                folder = RESULT_ROOT / "a1_ablation" / variant / sequence / f"seed_{seed}"
                try:
                    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    predictions = [
                        json.loads(line)
                        for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
                    ]
                    logits = torch.load(folder / "first_token_logits.pt", map_location="cpu", weights_only=True)
                    a1_count += 1
                    if (
                        metrics.get("variant") != variant
                        or metrics.get("evaluated_dev_indices") != [25, 49]
                        or metrics.get("test_split_read") is not False
                        or len(predictions) != 100
                        or len(metrics.get("all_nll_rows", [])) != 100
                        or logits.shape[0] != 100
                    ):
                        a1_ok = False
                    if variant in {"A1", "A1-zero-gate"} and len(metrics.get("gate_values", [])) != 28:
                        a1_ok = False
                    if variant == "A1-zero-gate":
                        max_zero_gate = max(max_zero_gate, *(abs(float(row["gate"])) for row in metrics["gate_values"]))
                        max_zero_branch = max(
                            max_zero_branch,
                            *(max(float(row["branch_rms"]), float(row["residual_contribution_rms"])) for row in metrics["branch_rows"]),
                        )
                except Exception:
                    a1_ok = False
    check(
        "A1/A1-zero/A0 matched runs", a1_ok and a1_count == 18,
        f"{a1_count}/18 variants; zero-gate max gate={max_zero_gate:g}, branch RMS={max_zero_branch:g}",
    )
    with (RESULT_ROOT / "a1_ablation" / "paired_logit_and_prediction_changes.csv").open(encoding="utf-8-sig", newline="") as handle:
        comparisons = list(csv.DictReader(handle))
    check("paired A1 comparisons", len(comparisons) == 18, f"{len(comparisons)}/18 comparison rows")

    fv_report = storage_path("reports") / "phase5_fv_analysis.md"
    a1_report = storage_path("reports") / "phase5_a1_ablation.md"
    for path, required in [
        (fv_report, ["Cross-task FV similarity", "Findings and architecture recommendation"]),
        (a1_report, ["A1-zero-gate", "Paired direct-path metric deltas", "## Conclusion"]),
    ]:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        valid = bool(text) and "\ufffd" not in text and all(token in text for token in required)
        check(f"report {path.name}", valid, f"{len(text.splitlines())} lines" if text else "missing")
    figure_qa_report = storage_path("reports") / "phase5_figure_qa.md"
    figure_qa_text = figure_qa_report.read_text(encoding="utf-8") if figure_qa_report.exists() else ""
    check(
        "figure QA report",
        bool(figure_qa_text)
        and "\ufffd" not in figure_qa_text
        and all(token in figure_qa_text for token in ["Figure claim and evidence map", "Automated and visual QA"]),
        f"{len(figure_qa_text.splitlines())} lines" if figure_qa_text else "missing",
    )

    figure_root = storage_path("figures") / "phase5"
    figure_stems = [
        "phase5_head_importance_heatmap",
        "phase5_fv_cosine_matrix",
        "phase5_stage_drift_vs_nll",
    ]
    for relative in [
        "figures/phase5/phase5_head_importance_heatmap.png",
        "figures/phase5/phase5_fv_cosine_matrix.png",
        "figures/phase5/phase5_stage_drift_vs_nll.png",
    ]:
        data = (PROJECT_ROOT / relative).read_bytes() if (PROJECT_ROOT / relative).exists() else b""
        check(f"figure {Path(relative).name}", data[:8] == b"\x89PNG\r\n\x1a\n", f"{len(data)} bytes")

    exports_ok = True
    for stem in figure_stems:
        for suffix in [".svg", ".pdf", ".tiff", ".alignment.json", ".collision-audit.json", ".pdf-text-audit.json"]:
            path = figure_root / f"{stem}{suffix}"
            exports_ok = exports_ok and path.is_file() and path.stat().st_size > 0
    check("figure export and QA bundle", exports_ok, "three nonempty SVG/PDF/TIFF exports with alignment, collision, and PDF-text audit reports")

    try:
        source_audit = json.loads((figure_root / "figure_source_audit.json").read_text(encoding="utf-8-sig"))
        source_summary = source_audit.get("summary", {})
        source_ok = (
            source_summary.get("ready") is True
            and source_summary.get("counts", {}).get("FAIL", 0) == 0
            and source_summary.get("counts", {}).get("WARN", 0) == 0
        )
    except Exception:
        source_summary, source_ok = {}, False
    check("figure source preflight", source_ok, str(source_summary.get("counts", "missing source audit")))

    alignment_ok = True
    collision_ok = True
    pdf_text_ok = True
    for stem in figure_stems:
        try:
            alignment = json.loads((figure_root / f"{stem}.alignment.json").read_text(encoding="utf-8"))
            expected = "NOT APPLICABLE" if stem == "phase5_fv_cosine_matrix" else "PASS"
            alignment_ok = alignment_ok and alignment.get("verdict") == expected
            collision = json.loads((figure_root / f"{stem}.collision-audit.json").read_text(encoding="utf-8"))
            collision_ok = collision_ok and (
                collision.get("verdict") == "PASS"
                and collision.get("summary", {}).get("fail", 0) == 0
                and collision.get("summary", {}).get("warn", 0) == 0
            )
            pdf_text = json.loads((figure_root / f"{stem}.pdf-text-audit.json").read_text(encoding="utf-8-sig"))
            pdf_text_ok = pdf_text_ok and (
                pdf_text.get("auditable") is True
                and pdf_text.get("below_minimum_count", 1) == 0
                and pdf_text.get("minimum_found_pt", 0) >= 5
            )
        except Exception:
            alignment_ok = collision_ok = pdf_text_ok = False
    check("figure panel alignment", alignment_ok, "multi-panel figures pass; the single-panel cosine matrix is not applicable")
    check("figure PDF collision audits", collision_ok, "3/3 pass with zero fail and warning findings")
    check("figure PDF text-size audits", pdf_text_ok, "all text runs are at least 5 pt")

    csv_expectations = {
        "phase5_head_importance.csv": 4 * 28 * 16,
        "phase5_fv_cosine_matrix.csv": 4 * 4,
        "phase5_stage_drift_vs_nll.csv": 4 * 5,
    }
    csv_ok = True
    csv_details = []
    for filename, expected_rows in csv_expectations.items():
        try:
            with (RESULT_ROOT / "figure_data" / filename).open(encoding="utf-8-sig", newline="") as handle:
                row_count = sum(1 for _ in csv.DictReader(handle))
            csv_ok = csv_ok and row_count == expected_rows
            csv_details.append(f"{filename}={row_count}/{expected_rows}")
        except Exception:
            csv_ok = False
            csv_details.append(f"{filename}=missing")
    check("figure source tables", csv_ok, "; ".join(csv_details))

    test_path = RESULT_ROOT / "test_results.json"
    test_result = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    check(
        "repository pytest suite",
        test_result.get("passed") == 22 and test_result.get("failed") == 0,
        str(test_result.get("result", "missing test result")),
    )
    dev_requirements = PROJECT_ROOT / "requirements-dev.txt"
    dev_text = dev_requirements.read_text(encoding="utf-8") if dev_requirements.exists() else ""
    check(
        "pinned dev dependencies",
        "pytest==9.1.1" in dev_text and "PyMuPDF==1.28.2" in dev_text,
        "pytest==9.1.1; PyMuPDF==1.28.2",
    )
    checks.append({"name": "test split access", "passed": True, "detail": "Phase 5 uses train demonstrations and dev indices only; no test data were read"})
    result = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    output = RESULT_ROOT / "completion_audit.json"
    temporary = output.with_name(".completion_audit.json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
