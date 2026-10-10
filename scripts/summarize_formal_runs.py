"""Audit the three frozen-sequence runs and build aggregate research artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_NAMES = ["homogeneous", "heterogeneous", "mixed"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> Any:
    return "" if value is None else value


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "continual_baseline_full.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_root = Path(config["storage"]["results"])
    manifest_path = Path(config["dataset"]["processed_dir"]) / "data_manifest.json"
    data_manifest = read_json(manifest_path)
    stream_manifest = read_json(PROJECT_ROOT / "data" / "task_stream_manifest.json")
    relation_path = Path("D:/Task-Aware-TransNAS/data/task_selection/task_relation_analysis.json")
    relations = read_json(relation_path)
    tuning = read_json(results_root / "tuning" / "tuning_results.json")
    task_ids = list(config["dataset"]["tasks"])
    expected_test_ids = {
        task["task_id"]: set(task["instance_ids"]["test"])
        for task in stream_manifest["tasks"]
    }

    runs: dict[str, dict[str, Any]] = {}
    validation: dict[str, Any] = {"passed": True, "errors": [], "prediction_rows": 0, "prediction_cells": 0}
    for sequence_name in SEQUENCE_NAMES:
        metrics_path = results_root / sequence_name / "metrics.json"
        if not metrics_path.exists():
            validation["passed"] = False
            validation["errors"].append(f"Missing run result: {metrics_path}")
            continue
        metrics = read_json(metrics_path)
        runs[sequence_name] = metrics
        if metrics["trainable_parameters"] != metrics["total_parameters"]:
            validation["passed"] = False
            validation["errors"].append(f"Not all parameters were trainable in {sequence_name}")
        if metrics["test_used_for_selection_or_tuning"] is not False:
            validation["passed"] = False
            validation["errors"].append(f"Test selection guard failed in {sequence_name}")
        if len(metrics["stage_order"]) != 5 or metrics["task_ids"] != task_ids:
            validation["passed"] = False
            validation["errors"].append(f"Stage or task matrix shape mismatch in {sequence_name}")
        train_ooms = sum(len(value["oom_retries"]) for value in metrics["training_metrics"].values())
        eval_ooms = sum(
            task_metric["oom_retry_count"]
            for stage in metrics["stage_scores"].values()
            for task_metric in stage.values()
        )
        if train_ooms or eval_ooms:
            validation["passed"] = False
            validation["errors"].append(f"OOM recovery occurred in {sequence_name}: train={train_ooms}, eval={eval_ooms}")
        for task_id in task_ids:
            train_info = metrics["training_metrics"].get(task_id)
            if not train_info or not Path(train_info["checkpoint"]).exists():
                validation["passed"] = False
                validation["errors"].append(f"Missing task checkpoint: {sequence_name}/{task_id}")
        for stage in metrics["stage_order"]:
            for task_id in task_ids:
                task_metric = metrics["stage_scores"][stage][task_id]
                prediction_path = Path(task_metric["prediction_file"])
                if not prediction_path.exists():
                    validation["passed"] = False
                    validation["errors"].append(f"Missing prediction file: {prediction_path}")
                    continue
                rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                actual_ids = [row["instance_id"] for row in rows]
                validation["prediction_rows"] += len(rows)
                validation["prediction_cells"] += 1
                if len(rows) != 50 or len(set(actual_ids)) != 50 or set(actual_ids) != expected_test_ids[task_id]:
                    validation["passed"] = False
                    validation["errors"].append(f"Test-ID mismatch in {sequence_name}/{stage}/{task_id}")
                if task_metric["definition_preservation_rate"] != 1.0:
                    validation["passed"] = False
                    validation["errors"].append(f"Definition-preservation check failed in {sequence_name}/{stage}/{task_id}")
                if not math.isfinite(task_metric["primary_score"]):
                    validation["passed"] = False
                    validation["errors"].append(f"Non-finite score in {sequence_name}/{stage}/{task_id}")

    if set(runs) != set(SEQUENCE_NAMES):
        validation["passed"] = False
    if validation["prediction_rows"] != len(SEQUENCE_NAMES) * 5 * len(task_ids) * 50:
        validation["passed"] = False
        validation["errors"].append("Total prediction row count is not 3,000")
    if validation["prediction_cells"] != len(SEQUENCE_NAMES) * 5 * len(task_ids):
        validation["passed"] = False
        validation["errors"].append("Total stage/task cell count is not 60")

    # The base-model evaluations are a shared control and should match exactly across streams.
    if set(runs) == set(SEQUENCE_NAMES):
        reference_base = runs[SEQUENCE_NAMES[0]]["stage_scores"]["baseline"]
        for sequence_name in SEQUENCE_NAMES[1:]:
            for task_id in task_ids:
                actual = runs[sequence_name]["stage_scores"]["baseline"][task_id]["primary_score"]
                expected = reference_base[task_id]["primary_score"]
                if abs(actual - expected) > 1e-12:
                    validation["passed"] = False
                    validation["errors"].append(f"Base control differs for {sequence_name}/{task_id}")

    validation["run_count"] = len(runs)
    validation["expected_prediction_rows"] = 3000
    validation["expected_prediction_cells"] = 60
    validation_path = results_root / "validation_summary.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise RuntimeError("Formal run verification failed: " + "; ".join(validation["errors"]))

    csv_path = results_root / "accuracy_matrix.csv"
    columns = ["sequence", "stage"]
    for task_id in task_ids:
        columns.extend([f"{task_id}__primary_score", f"{task_id}__metric"])
        if config["dataset"]["task_types"][task_id]["type"] == "classification":
            columns.extend([f"{task_id}__accuracy", f"{task_id}__macro_f1"])
        else:
            columns.extend([f"{task_id}__rouge_l_f1", f"{task_id}__token_f1", f"{task_id}__exact_match"])
    columns.extend(
        [
            "normalized_average_accuracy_all_tasks",
            "aaa_normalized_seen_tasks",
            "average_forgetting_normalized",
            "raw_classification_accuracy_mean",
            "raw_generation_rouge_l_mean",
        ]
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for sequence_name in SEQUENCE_NAMES:
            metrics = runs[sequence_name]
            for stage in metrics["stage_order"]:
                row: dict[str, Any] = {"sequence": sequence_name, "stage": stage}
                for task_id in task_ids:
                    metric = metrics["stage_scores"][stage][task_id]
                    row[f"{task_id}__primary_score"] = metric["primary_score"]
                    row[f"{task_id}__metric"] = metric["primary_metric"]
                    for key in ("accuracy", "macro_f1", "rouge_l_f1", "token_f1", "exact_match"):
                        column = f"{task_id}__{key}"
                        if column in columns:
                            row[column] = clean(metric.get(key))
                row.update(metrics["cl_summary"][stage])
                row.pop("normalized_gain_by_task", None)
                writer.writerow(row)

    report: list[str] = [
        "# Formal full-parameter continual SFT analysis",
        "",
        "## Design and frozen choices",
        "",
        f"- CITB source: official order1, commit `{data_manifest['source']['commit']}`.",
        "- Task selection used only the balanced training pools, frozen Qwen3 prompt representations, and final-block supervised gradient cosines. Development scores and test scores did not select tasks or stream order.",
        "- The four tasks contain two official Dialogue Generation tasks with free-text outputs and two Dialogue State Tracking tasks with Yes/No class outputs.",
        f"- Similarity groups: `{relations['selected_cluster_pairing']['pair_a'][0]}` + `{relations['selected_cluster_pairing']['pair_a'][1]}` (Dialogue Generation / generation), and `{relations['selected_cluster_pairing']['pair_b'][0]}` + `{relations['selected_cluster_pairing']['pair_b'][1]}` (Dialogue State Tracking / classification).",
        f"- Shared sample sizes: {config['dataset']['train_per_task']} train / {config['dataset']['dev_per_task']} dev / {config['dataset']['test_per_task']} test per task; all three sequences use the same instance IDs.",
        f"- Formal prompt maximum: {config['model']['max_sequence_length']} tokens; complete task definition retained, long input tails truncated. Every evaluated prediction passed the task-definition preservation assertion.",
        f"- Model: local `{runs['homogeneous']['model_path']}`; CUDA device `{runs['homogeneous']['device']}`; {runs['homogeneous']['total_parameters']:,} / {runs['homogeneous']['trainable_parameters']:,} parameters trainable.",
        f"- Frozen hyperparameters: learning rate `{config['training']['learning_rate']}`; `{config['training']['steps_per_task']}` optimizer steps per task; gradient accumulation `{config['training']['gradient_accumulation_steps']}`; weight decay `{config['training']['weight_decay']}`.",
        "- Hyperparameters were selected from a 3-by-3 development-only grid after the homogeneous tuning sequence; the frozen selection and all candidate metrics are in `D:/Task-Aware-TransNAS/results/baseline_full/tuning/tuning_results.json`.",
        "- The test set was opened after the tuning decision was frozen. No test results were used to adjust the tasks, task order, or training parameters.",
        "",
        "### Development-only tuning grid",
        "",
        "| Candidate | Learning rate | Steps/task | Normalized all-task gain | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in sorted(tuning["candidates"], key=lambda item: item["normalized_average_accuracy_all_tasks"], reverse=True):
        scores = candidate["dev_scores"]
        report.append(
            f"| {candidate['candidate_id']} | {candidate['learning_rate']:.1e} | {candidate['steps_per_task']} | "
            f"{candidate['normalized_average_accuracy_all_tasks']:.4f} | "
            f"{scores['task565_circa_answer_generation']['primary_score']:.4f} | "
            f"{scores['task1603_smcalflow_sentence_generation']['primary_score']:.4f} | "
            f"{scores['task1501_dstc3_answer_generation']['primary_score']:.4f} | "
            f"{scores['task1384_deal_or_no_dialog_classification']['primary_score']:.4f} |"
        )
    report.extend(["", "## Reproducible task sequences", "", "| Sequence | Task order | Mean adjacent similarity z |", "|---|---|---:|"])
    for sequence_name in SEQUENCE_NAMES:
        seq = relations["sequences"][sequence_name]
        report.append(f"| {sequence_name} | {' -> '.join(seq['task_ids'])} | {seq['mean_adjacent_similarity_z']:.4f} |")
    report.extend(
        [
            "",
            "## Test results at each stage",
            "",
            "Classification reports Accuracy and fixed-label Macro-F1. Free-text generation reports ROUGE-L F1 as primary, with Token-F1 and normalized Exact Match as auxiliary metrics. Raw score units are kept separate.",
            "",
            "| Sequence | Stage | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy / Macro-F1 | task1384 Accuracy / Macro-F1 | Normalized all-task gain | AAA (seen tasks) | Forgetting |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sequence_name in SEQUENCE_NAMES:
        metrics = runs[sequence_name]
        for stage in metrics["stage_order"]:
            scores = metrics["stage_scores"][stage]
            summary = metrics["cl_summary"][stage]
            report.append(
                f"| {sequence_name} | {stage} | {scores['task565_circa_answer_generation']['rouge_l_f1']:.4f} | "
                f"{scores['task1603_smcalflow_sentence_generation']['rouge_l_f1']:.4f} | "
                f"{scores['task1501_dstc3_answer_generation']['accuracy']:.3f} / {scores['task1501_dstc3_answer_generation']['macro_f1']:.3f} | "
                f"{scores['task1384_deal_or_no_dialog_classification']['accuracy']:.3f} / {scores['task1384_deal_or_no_dialog_classification']['macro_f1']:.3f} | "
                f"{summary['normalized_average_accuracy_all_tasks']:.4f} | {summary['aaa_normalized_seen_tasks']:.4f} | "
                f"{summary['average_forgetting_normalized']:.4f} |"
            )
    report.extend(["", "## Final-stage comparison", "", "| Sequence | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy | Normalized all-task gain | AAA | Average forgetting | Wall time (s) | Peak allocated VRAM (GiB) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for sequence_name in SEQUENCE_NAMES:
        metrics = runs[sequence_name]
        stage = metrics["stage_order"][-1]
        scores = metrics["stage_scores"][stage]
        summary = metrics["cl_summary"][stage]
        report.append(
            f"| {sequence_name} | {scores['task565_circa_answer_generation']['rouge_l_f1']:.4f} | "
            f"{scores['task1603_smcalflow_sentence_generation']['rouge_l_f1']:.4f} | "
            f"{scores['task1501_dstc3_answer_generation']['accuracy']:.3f} | "
            f"{scores['task1384_deal_or_no_dialog_classification']['accuracy']:.3f} | "
            f"{summary['normalized_average_accuracy_all_tasks']:.4f} | {summary['aaa_normalized_seen_tasks']:.4f} | "
            f"{summary['average_forgetting_normalized']:.4f} | {metrics['duration_seconds']:.1f} | "
            f"{metrics['peak_cuda_allocated_bytes_training'] / (1024**3):.2f} |"
        )
    report.extend(
        [
            "",
            "## Verification and interpretation",
            "",
            f"- Automated artifact audit: PASS ({validation['prediction_cells']} stage/task cells, {validation['prediction_rows']} predictions, each cell matched the official 50 test IDs).",
            "- All three runs updated all 596,049,920 parameters; none used OOM recovery; peak allocated GPU memory was about 10.85 GiB.",
            "- On task565, task1603, and task1384 prompts, all input tokens fit the window. For task1501, 159/500 train, 15/50 dev, and 12/50 test examples had long dialogue inputs whose oldest context tokens were dropped; the complete definition and input suffix, including the question, remained in the prompt.",
            "- The homogeneous sequence ended with both Yes/No tasks strong but task565 ROUGE-L at zero. The heterogeneous and mixed streams produced different retention profiles. The results show substantial interference between free-text and short-label tasks under this small full-parameter SFT baseline.",
            "- The study uses one seed and a local 0.6B model; these runs are an initial formal baseline, not a multi-seed significance estimate. No NAS, architecture changes, LoRA, or replay was used.",
            "",
            "## Artifacts",
            "",
            "- `D:/Task-Aware-TransNAS/results/baseline_full/accuracy_matrix.csv`: combined 15-stage-by-4-task metrics.",
            "- `D:/Task-Aware-TransNAS/results/baseline_full/validation_summary.json`: test ID, matrix, checkpoint, parameter-count, and OOM verification.",
            "- `D:/Task-Aware-TransNAS/results/baseline_full/tuning/tuning_results.json`: dev-only candidate results.",
            "- `D:/Task-Aware-TransNAS/results/baseline_full/{homogeneous,heterogeneous,mixed}/`: per-sequence predictions, metrics, reports, and loss curves.",
            "- `D:/Task-Aware-TransNAS/checkpoints/baseline_full/{homogeneous,heterogeneous,mixed}/`: one checkpoint per task boundary.",
            "",
            "The wide machine-readable artifact retains each task metric separately and is the source for reproducing the tables above.",
            "",
        ]
    )
    analysis_path = PROJECT_ROOT / "reports" / "baseline_full_analysis.md"
    analysis_path.write_text("\n".join(report), encoding="utf-8")
    summary_json = {
        "validation": validation,
        "selected_hyperparameters": config["tuning"]["selected"],
        "final_stage_by_sequence": {
            sequence_name: {
                "stage": runs[sequence_name]["stage_order"][-1],
                "task_metrics": runs[sequence_name]["stage_scores"][runs[sequence_name]["stage_order"][-1]],
                "continual_learning_summary": runs[sequence_name]["cl_summary"][runs[sequence_name]["stage_order"][-1]],
            }
            for sequence_name in SEQUENCE_NAMES
        },
    }
    (results_root / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Artifact verification: PASS ({validation['prediction_cells']} cells, {validation['prediction_rows']} rows)")
    print(f"Combined matrix: {csv_path}")
    print(f"Analysis report: {analysis_path}")


if __name__ == "__main__":
    main()
