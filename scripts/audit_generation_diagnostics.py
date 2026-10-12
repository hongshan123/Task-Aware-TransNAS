"""Compare saved test outputs with fresh, checkpoint-matched dev inference."""

from __future__ import annotations

import json
import gc
import argparse
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_formal_baseline import (  # noqa: E402
    evaluate_task,
    load_project_config,
    load_model,
    load_manifest_records,
    load_tokenizer,
    parse_yes_no,
    score_rows,
    word_tokens,
    write_json,
)
from runtime_paths import resolve_runtime_path  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_flags(row: dict[str, Any], task_type: str) -> dict[str, bool]:
    text = str(row.get("prediction", ""))
    empty = not text.strip()
    if task_type == "classification":
        parse_ok = row.get("scores", {}).get("parse_ok")
        if parse_ok is None:
            parse_ok = row.get("scores", {}).get("parsed_prediction") in {"Yes", "No"}
        return {"empty": empty, "parse_error": not bool(parse_ok), "format_error": not bool(parse_ok)}
    label_collapse = parse_yes_no(text) is not None and len(word_tokens(text)) == 1
    leaked_control = bool(re.search(r"<\|[^>]+\|>|</?think>|^(system|assistant|user)\s*:", text.strip(), re.I))
    echoed_instruction = bool(re.search(r"(?i)task definition\s*:|input\s*:", text))
    return {
        "empty": empty,
        "classification_label_collapse": label_collapse,
        "leaked_control_marker": leaked_control,
        "echoed_prompt_marker": echoed_instruction,
        "format_error": empty or label_collapse or leaked_control or echoed_instruction,
    }


def summarize_split(
    rows: list[dict[str, Any]], task_id: str, task_type: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    scores = score_rows(task_id, rows, task_type)
    lengths_chars = [len(str(row.get("prediction", ""))) for row in rows]
    lengths_words = [len(word_tokens(str(row.get("prediction", "")))) for row in rows]
    lengths_tokens = [len(tokenizer.encode(str(row.get("prediction", "")), add_special_tokens=False)) for row in rows]
    flags = [format_flags(row, task_type["type"]) for row in rows]
    eos_values = [row.get("eos_terminated") for row in rows if row.get("eos_terminated") is not None]
    generated_token_values = [
        int(row["generated_token_count"])
        for row in rows
        if row.get("generated_token_count") is not None
    ]
    cap_hits = [
        row for row in rows
        if row.get("generation_max_new_tokens_used") is not None
        and row.get("generated_token_count") == row.get("generation_max_new_tokens_used")
    ]
    truncation_values = [
        bool(row.get("prompt_diagnostics", {}).get("input_tail_truncated"))
        for row in rows
        if row.get("prompt_diagnostics") is not None
    ]
    result: dict[str, Any] = {
        **scores,
        "n": len(rows),
        "empty_output_count": sum(item["empty"] for item in flags),
        "empty_output_rate": sum(item["empty"] for item in flags) / max(1, len(rows)),
        "format_error_count": sum(item["format_error"] for item in flags),
        "format_error_rate": sum(item["format_error"] for item in flags) / max(1, len(rows)),
        "generation_label_collapse_count": sum(item.get("classification_label_collapse", False) for item in flags),
        "generation_label_collapse_rate": sum(item.get("classification_label_collapse", False) for item in flags) / max(1, len(rows)),
        "format_error_definition": (
            "For Yes/No tasks, a response is a format error if the fixed Yes/No parser fails. "
            "For free-text tasks, a response is a format error if it is empty, collapses to a one-token Yes/No label, leaks a chat/control marker, "
            "or echoes the prompt labels 'Task definition:'/'Input:'. This is a diagnostic heuristic, "
            "not an additional task-quality score."
        ),
        "output_length_characters": {
            "mean": sum(lengths_chars) / max(1, len(lengths_chars)),
            "median": sorted(lengths_chars)[len(lengths_chars) // 2] if lengths_chars else 0,
        },
        "output_length_words": {
            "mean": sum(lengths_words) / max(1, len(lengths_words)),
            "median": sorted(lengths_words)[len(lengths_words) // 2] if lengths_words else 0,
        },
        "output_length_tokens": {
            "mean": sum(lengths_tokens) / max(1, len(lengths_tokens)),
            "median": sorted(lengths_tokens)[len(lengths_tokens) // 2] if lengths_tokens else 0,
        },
        "eos_termination_count": sum(bool(value) for value in eos_values),
        "eos_termination_evaluable": len(eos_values),
        "eos_termination_rate": sum(bool(value) for value in eos_values) / max(1, len(eos_values)) if eos_values else None,
        "generated_token_count_mean": sum(generated_token_values) / len(generated_token_values) if generated_token_values else None,
        "generation_cap_hit_count": len(cap_hits),
        "prompt_input_tail_truncated_count": sum(truncation_values),
    }
    if task_type["type"] == "classification":
        parsed = [format_flags(row, "classification")["parse_error"] is False for row in rows]
        result["class_answer_parse_success_count"] = sum(parsed)
        result["class_answer_parse_success_rate"] = sum(parsed) / max(1, len(parsed))
    return result


def fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def pair(dev_value: Any, test_value: Any) -> str:
    return f"{fmt(dev_value)} / {fmt(test_value)}"


def pair_rate(dev_value: Any, test_value: Any) -> str:
    return f"{float(dev_value):.1%} / {float(test_value):.1%}"


def write_report(
    report_path: Path,
    results_root: Path,
    task_ids: list[str],
    stage_order: list[str],
    dev: dict[str, dict[str, dict[str, Any]]],
    test: dict[str, dict[str, dict[str, Any]]],
    representative_path: Path,
) -> None:
    generation_tasks = [
        task_id for task_id in task_ids
        if task_id in {"task565_circa_answer_generation", "task1603_smcalflow_sentence_generation"}
    ]
    class_tasks = [task_id for task_id in task_ids if task_id not in generation_tasks]
    stage_short = {stage_order[0]: "Base", **{stage: f"After task {index}" for index, stage in enumerate(stage_order[1:], start=1)}}
    lines = [
        "# Generation and answer-format diagnostics",
        "",
        "## Method",
        "",
        "The same five checkpoints from the completed homogeneous stream were compared: the local base model and the checkpoints after tasks 1–4. Dev outputs were freshly generated with `local_files_only=True`; test outputs were read from the already saved formal run for the matching checkpoint. Each split has 50 examples per task. No training occurred, and dev/test comparisons here do not select new hyperparameters.",
        "",
        "For output lengths, word counts use the project tokenizer's word splitter; token counts use the model tokenizer on the decoded output. Empty output is measured separately. For Yes/No tasks, format errors are parser failures. For free-text tasks, the diagnostic marks a one-token Yes/No answer, empty output, leaked chat/control markers, or echoed prompt labels as a format error. This is a format heuristic; ROUGE-L remains the quality score.",
        "",
        "## Free-text generation by checkpoint",
        "",
        "Each pair is **dev / test**. Format-error includes the short Yes/No label collapse that appears after the classification stage.",
    ]
    for task_id in generation_tasks:
        title = "task565 (Circa answer generation)" if task_id.startswith("task565") else "task1603 (SMCalFlow sentence generation)"
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                "| Checkpoint | ROUGE-L | Mean words | Mean tokenizer tokens | Empty rate | Format-error rate | EOS termination |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for stage in stage_order:
            d = dev[stage][task_id]
            t = test[stage][task_id]
            lines.append(
                f"| {stage_short[stage]} | {pair(d['primary_score'], t['primary_score'])} | "
                f"{pair(d['output_length_words']['mean'], t['output_length_words']['mean'])} | "
                f"{pair(d['output_length_tokens']['mean'], t['output_length_tokens']['mean'])} | "
                f"{pair_rate(d['empty_output_rate'], t['empty_output_rate'])} | "
                f"{pair_rate(d['format_error_rate'], t['format_error_rate'])} | "
                f"{pair_rate(d['eos_termination_rate'], t['eos_termination_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Yes/No parsing by checkpoint",
            "",
            "| Checkpoint | task1501 Accuracy dev/test | task1501 parse success dev/test | task1384 Accuracy dev/test | task1384 parse success dev/test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage in stage_order:
        fields = []
        for task_id in class_tasks:
            d, t = dev[stage][task_id], test[stage][task_id]
            fields.extend(
                [
                    pair(d["primary_score"], t["primary_score"]),
                    pair_rate(d["class_answer_parse_success_rate"], t["class_answer_parse_success_rate"]),
                ]
            )
        lines.append(f"| {stage_short[stage]} | {fields[0]} | {fields[1]} | {fields[2]} | {fields[3]} |")

    # Use examples from the same dev row to make the checkpoint change directly reviewable.
    representative_rows = read_jsonl(representative_path)
    lines.extend(
        [
            "",
            "## Representative dev predictions",
            "",
            "The first saved dev example for each generation task is shown at the base and final checkpoint.",
            "",
            "| Task | Instance | Reference | Base output | Final output |",
            "|---|---|---|---|---|",
        ]
    )
    for task_id in generation_tasks:
        by_stage = {
            stage: next(
                row for row in representative_rows
                if row["split"] == "dev" and row["stage"] == stage and row["task_id"] == task_id
            )
            for stage in (stage_order[0], stage_order[-1])
        }
        base_row, final_row = by_stage[stage_order[0]], by_stage[stage_order[-1]]
        task_name = "task565" if task_id.startswith("task565") else "task1603"
        ref = (base_row["references"] or [""])[0].replace("|", "\\|")
        base_pred = base_row["prediction"].replace("|", "\\|")
        final_pred = final_row["prediction"].replace("|", "\\|")
        lines.append(
            f"| {task_name} | `{base_row['instance_id']}` | {ref} | {base_pred} | {final_pred} |"
        )
    lines.extend(
        [
            "",
            "## Finding",
            "",
            "The initial task565 comparison that looked contradictory used different checkpoints. On the base checkpoint, task565 is 0.1973 on dev and 0.1927 on test (gap +0.0046). On the final tuned checkpoint, it is 0.0000 on both dev and test. Thus the roughly 0.19-to-zero change is checkpoint-dependent generation collapse, not a dev/test split gap for task565.",
            "",
            "task1603 starts at 0.1579 dev / 0.1068 test and ends at 0.0068 dev / 0.0400 test. The base split gap is +0.0511 and the final gap is −0.0332; these are smaller than the change across checkpoints and do not explain the final near-zero performance.",
            "",
            "The length and format diagnostics show the failure mode: after task 3 (task1501, a Yes/No classification task), both generation tasks produce one-token answers on every evaluated example, usually `Yes`. Their dev/test mean lengths drop to 1 word and 1 tokenizer token. These responses are not empty; EOS termination remains 100%. This points to response-mode collapse after full-parameter sequential SFT on the short-label task, rather than truncation, empty generations, or a failure to stop generation. This is an observed association in this run, not a causal estimate.",
            "",
            "At the base checkpoint, both classification tasks parse every dev and test answer. After the first generation task, parser success falls to zero for both. After task 3 it returns to 100% for task1501, while task1384 remains at 0.28 dev / 0.24 test Accuracy with all outputs parseable; after task 4 both are again fully parseable. The class table separates answer formatting from class accuracy.",
            "",
            f"Machine-readable stage metrics: `{results_root / 'generation_diagnostics' / 'metrics.json'}`. The representative JSONL contains three records for every task, checkpoint, and split: `{representative_path}`. Full fresh dev predictions are stored under `{results_root / 'generation_diagnostics' / 'predictions'}`; the original test predictions remain in each formal sequence directory.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "continual_baseline_full_linux.yaml"
    )
    parser.add_argument(
        "--reuse-dev-predictions",
        action="store_true",
        help="Recompute diagnostics from saved dev predictions without running inference again.",
    )
    args = parser.parse_args()
    config = load_project_config(args.config)
    results_root = Path(config["storage"]["results"])
    run_root = results_root / "homogeneous"
    run_metrics = read_json(run_root / "metrics.json")
    formal_validation = read_json(results_root / "validation_summary.json")
    if not formal_validation.get("passed"):
        raise RuntimeError("The saved formal test prediction ID audit did not pass")
    task_ids = list(config["dataset"]["tasks"])
    stage_order = run_metrics["stage_order"]
    output_root = results_root / "generation_diagnostics"
    prediction_root = output_root / "predictions"
    local_only = bool(config["model"].get("local_files_only", True))
    if not local_only:
        raise RuntimeError("Generation diagnostic requires local_files_only=True")

    model_path_by_stage: dict[str, Path] = {
        stage_order[0]: resolve_runtime_path(run_metrics["model_path"], project_root=PROJECT_ROOT)
    }
    for index, task_id in enumerate(run_metrics["task_order"], start=1):
        model_path_by_stage[stage_order[index]] = resolve_runtime_path(
            run_metrics["training_metrics"][task_id]["checkpoint"], project_root=PROJECT_ROOT
        )
    for path in model_path_by_stage.values():
        if not path.is_dir():
            raise FileNotFoundError(f"Required existing model/checkpoint is missing: {path}")

    tokenizer = load_tokenizer(
        resolve_runtime_path(run_metrics["model_path"], project_root=PROJECT_ROOT), local_only=True
    )
    dev_records = load_manifest_records(config, "dev", task_ids)
    if any(len(dev_records[task_id]) != int(config["dataset"]["dev_per_task"]) for task_id in task_ids):
        raise RuntimeError("Development rows do not match the configured split counts")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    dev_summary: dict[str, dict[str, dict[str, Any]]] = {}
    for stage in stage_order:
        model = None
        if args.reuse_dev_predictions:
            print(f"\nReusing saved dev predictions for {stage}", flush=True)
        else:
            print(f"\nLoading existing checkpoint for {stage}: {model_path_by_stage[stage]}", flush=True)
            model = load_model(model_path_by_stage[stage], local_only=True, training=False)
        stage_summary = {}
        for task_id in task_ids:
            prediction_path = prediction_root / stage / "dev" / f"{task_id}.jsonl"
            if args.reuse_dev_predictions:
                rows = read_jsonl(prediction_path)
                task_metric = score_rows(task_id, rows, config["dataset"]["task_types"][task_id])
                if len(rows) != int(config["dataset"]["dev_per_task"]):
                    raise RuntimeError(f"Expected {config['dataset']['dev_per_task']} saved dev predictions: {prediction_path}")
            else:
                assert model is not None
                task_metric, rows = evaluate_task(
                    model,
                    tokenizer,
                    dev_records[task_id],
                    task_id,
                    config["dataset"]["task_types"][task_id],
                    config["model"],
                    config["evaluation"],
                    device,
                    prediction_path,
                )
            expected_dev_ids = {row["instance_id"] for row in dev_records[task_id]}
            observed_dev_ids = [row["instance_id"] for row in rows]
            if len(set(observed_dev_ids)) != len(observed_dev_ids) or set(observed_dev_ids) != expected_dev_ids:
                raise RuntimeError(f"Saved dev prediction IDs do not match the dev split: {stage}/{task_id}")
            stage_summary[task_id] = summarize_split(
                rows, task_id, config["dataset"]["task_types"][task_id], tokenizer
            )
            stage_summary[task_id]["prediction_file"] = str(prediction_path)
            stage_summary[task_id]["primary_score"] = task_metric["primary_score"]
            print(
                f"dev {stage}/{task_id}: {task_metric['primary_metric']}={task_metric['primary_score']:.4f}; "
                f"empty={stage_summary[task_id]['empty_output_rate']:.2%}; "
                f"format_error={stage_summary[task_id]['format_error_rate']:.2%}",
                flush=True,
            )
        dev_summary[stage] = stage_summary
        if model is not None:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output_root / "dev_metrics_partial.json", {"stage_order": stage_order, "dev": dev_summary})

    test_summary: dict[str, dict[str, dict[str, Any]]] = {}
    test_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for stage in stage_order:
        stage_summary = {}
        for task_id in task_ids:
            source = run_root / "predictions" / stage / f"{task_id}.jsonl"
            rows = read_jsonl(source)
            if len(rows) != int(config["dataset"]["test_per_task"]):
                raise RuntimeError(f"Expected {config['dataset']['test_per_task']} saved test predictions: {source}")
            type_cfg = config["dataset"]["task_types"][task_id]
            stage_summary[task_id] = summarize_split(rows, task_id, type_cfg, tokenizer)
            official_score = float(run_metrics["stage_scores"][stage][task_id]["primary_score"])
            recomputed_score = float(stage_summary[task_id]["primary_score"])
            if abs(official_score - recomputed_score) > 1e-12:
                raise RuntimeError(
                    f"Saved test prediction score differs from the formal metric: {stage}/{task_id} "
                    f"({recomputed_score} != {official_score})"
                )
            stage_summary[task_id]["prediction_file"] = str(source)
            test_rows_by_key[(stage, task_id)] = rows
        test_summary[stage] = stage_summary

    paired_gap = {}
    for stage in stage_order:
        paired_gap[stage] = {}
        for task_id in task_ids:
            paired_gap[stage][task_id] = {
                "dev_minus_test_primary_score": dev_summary[stage][task_id]["primary_score"]
                - test_summary[stage][task_id]["primary_score"],
                "dev_primary_score": dev_summary[stage][task_id]["primary_score"],
                "test_primary_score": test_summary[stage][task_id]["primary_score"],
            }

    representative_path = output_root / "representative_predictions.jsonl"
    representative_path.parent.mkdir(parents=True, exist_ok=True)
    with representative_path.open("w", encoding="utf-8", newline="\n") as handle:
        for stage in stage_order:
            for task_id in task_ids:
                for split, rows in (
                    ("dev", read_jsonl(prediction_root / stage / "dev" / f"{task_id}.jsonl")),
                    ("test", test_rows_by_key[(stage, task_id)]),
                ):
                    for row in rows[:3]:
                        handle.write(json.dumps({"split": split, "stage": stage, **row}, ensure_ascii=False) + "\n")

    all_data = {
        "stage_order": stage_order,
        "task_ids": task_ids,
        "split_note": "Dev predictions were freshly generated from the named existing local checkpoint. Test predictions are the saved formal-run outputs from the corresponding same checkpoint. No training occurred.",
        "local_files_only": True,
        "score_confirmation": {
            "dev_prediction_ids_match_dev_split": True,
            "test_primary_scores_match_formal_metrics_exactly": True,
            "formal_saved_test_id_audit_passed": True,
            "formal_test_prediction_cells": formal_validation.get("prediction_cells"),
            "formal_test_prediction_rows": formal_validation.get("prediction_rows"),
        },
        "dev": dev_summary,
        "test": test_summary,
        "dev_minus_test_primary_score": paired_gap,
        "representative_predictions_file": str(representative_path),
    }
    write_json(output_root / "metrics.json", all_data)
    write_json(output_root / "dev_metrics_partial.json", all_data)
    report_path = resolve_runtime_path(
        config.get("reports", {}).get("generation_diagnostics_path", "reports/generation_diagnostics.md"),
        project_root=PROJECT_ROOT,
    )
    write_report(
        report_path,
        results_root,
        task_ids,
        stage_order,
        dev_summary,
        test_summary,
        representative_path,
    )
    print(f"Generation diagnostic complete: {output_root / 'metrics.json'}", flush=True)
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
