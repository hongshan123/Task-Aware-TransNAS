"""Re-evaluate every saved pilot stage with task-appropriate metrics and trace metadata."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path("D:/Task-Aware-TransNAS")
os.environ.setdefault("HF_HOME", str(STORAGE_ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(STORAGE_ROOT / "cache" / "huggingface" / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer


CLASS_LABELS = {
    "task848_pubmedqa_classification": ["0", "1"],
    "task611_mutual_multi_turn_dialogue": ["A", "B", "C", "D"],
    "task1500_dstc3_classification": ["Cheap", "Moderate", "Expensive", "Don't Care"],
}
GENERATION_TASKS = {"task1590_diplomacy_text_generation"}
MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = 512
AUDIT_SAMPLE_N = 15
AUDIT_SAMPLE_SEED = 20261009


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalized_text(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n.,!?;:'\"`“”‘’()[]{}")


def canonical_gold(task_id: str, references: list[str]) -> list[str]:
    labels = CLASS_LABELS[task_id]
    mapping = {normalized_text(label): label for label in labels}
    return [mapping[normalized_text(ref)] for ref in references if normalized_text(ref) in mapping]


def parse_classification(task_id: str, prediction: str) -> str | None:
    text = prediction.replace("’", "'").replace("‘", "'").strip()
    if task_id == "task848_pubmedqa_classification":
        patterns = [
            r"^\s*(?:(?:the\s+)?(?:answer|output|label|class)\s*(?:is|=|:|-)?\s*)?['\"`(]*([01])['\"`)]?(?=\s|$|[.,;:!?])",
            r"^\s*(?:the\s+)?(?:answer|output|label|class)\s+(?:is\s+)?['\"`(]*([01])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    if task_id == "task611_mutual_multi_turn_dialogue":
        pattern = r"^\s*(?:(?:the\s+)?(?:answer|choice|option)\s*(?:is|=|:|-)?\s*)?[\(\[]?([A-D])[\)\]]?(?=\s|$|[.,;:!?-])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None
    if task_id == "task1500_dstc3_classification":
        pattern = (
            r"^\s*(?:(?:the\s+)?(?:answer|price(?:\s+range)?|range)\s*(?:is|=|:|-)?\s*)?"
            r"['\"`]*(cheap|moderate|expensive|don't\s+care)['\"`]*(?=\s|$|[.,;:!?])"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        normalized = normalized_text(match.group(1))
        return {normalized_text(label): label for label in CLASS_LABELS[task_id]}.get(normalized)
    raise ValueError(f"No classification parser configured for {task_id}")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’][\w]+)?", text.casefold(), flags=re.UNICODE)


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred, ref = word_tokens(prediction), word_tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for pred_token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    lcs = previous[-1]
    return (2 * lcs) / (len(pred) + len(ref))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = Counter(word_tokens(prediction)), Counter(word_tokens(reference))
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def generation_scores(prediction: str, references: list[str]) -> dict[str, float]:
    refs = references or [""]
    return {
        "exact_match": float(any(normalized_text(prediction) == normalized_text(ref) for ref in refs)),
        "rouge_l_f1": max(rouge_l_f1(prediction, ref) for ref in refs),
        "token_f1": max(token_f1(prediction, ref) for ref in refs),
    }


def classification_scores(task_id: str, prediction: str, references: list[str]) -> dict[str, Any]:
    parsed = parse_classification(task_id, prediction)
    gold = canonical_gold(task_id, references)
    return {
        "parsed_prediction": parsed,
        "gold_labels": gold,
        "parse_ok": parsed is not None,
        "accuracy": float(parsed is not None and parsed in gold),
    }


def get_token_ids(value: Any) -> list[int]:
    if hasattr(value, "keys"):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def make_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    system = "You are an instruction-following assistant. Follow the task definition and return only the requested answer."
    user = "Task definition:\n" + record["definition"] + "\n\nInput:\n" + record["input"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_diagnostics(tokenizer: Any, record: dict[str, Any], training: bool) -> dict[str, Any]:
    messages = make_messages(record)
    prompt_ids = get_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    )
    target_tokens = 0
    target_capped = False
    if training:
        target = record["targets"][0] if record["targets"] else ""
        complete_ids = get_token_ids(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": target}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if complete_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Chat-template prefix mismatch for {record['instance_id']}")
        target_ids = complete_ids[len(prompt_ids) :]
        target_tokens = len(target_ids)
        if len(target_ids) >= MAX_INPUT_TOKENS:
            kept_prompt = []
            target_capped = True
        else:
            kept_count = MAX_INPUT_TOKENS - len(target_ids)
            kept_prompt = prompt_ids[-kept_count:]
    else:
        kept_prompt = prompt_ids[-MAX_INPUT_TOKENS:]
    decoded = tokenizer.decode(kept_prompt, skip_special_tokens=True)
    decoded_norm = normalized_text(decoded)
    definition_norm = normalized_text(record["definition"])
    definition_prefix = normalized_text(record["definition"][:80])
    return {
        "prompt_tokens_before_truncation": len(prompt_ids),
        "prompt_tokens_retained": len(kept_prompt),
        "prompt_tokens_dropped": len(prompt_ids) - len(kept_prompt),
        "task_definition_marker_retained": "task definition" in decoded.casefold(),
        "task_definition_prefix_retained": bool(definition_prefix and definition_prefix in decoded_norm),
        "full_task_definition_retained": bool(definition_norm and definition_norm in decoded_norm),
        "assistant_target_tokens": target_tokens,
        "assistant_target_capped": target_capped,
    }


def macro_f1(golds: list[str], predictions: list[str | None], labels: list[str]) -> float:
    per_label = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, predictions))
        fp = sum(g != label and p == label for g, p in zip(golds, predictions))
        fn = sum(g == label and p != label for g, p in zip(golds, predictions))
        denom = 2 * tp + fp + fn
        per_label.append((2 * tp / denom) if denom else 0.0)
    return sum(per_label) / len(per_label)


def score_stage_task(task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if task_id in CLASS_LABELS:
        golds: list[str] = []
        preds: list[str | None] = []
        for row in rows:
            targets = canonical_gold(task_id, row["reference"])
            if not targets:
                raise ValueError(f"No canonical class reference for {task_id}/{row['instance_id']}")
            golds.append(targets[0])
            preds.append(row["scores"]["parsed_prediction"])
        correct = sum(p is not None and p in canonical_gold(task_id, row["reference"]) for p, row in zip(preds, rows))
        return {
            "task_type": "classification",
            "primary_metric": "accuracy",
            "primary_score": correct / len(rows),
            "accuracy": correct / len(rows),
            "macro_f1": macro_f1(golds, preds, CLASS_LABELS[task_id]),
            "parse_success_rate": sum(p is not None for p in preds) / len(preds),
            "evaluated": len(rows),
            "class_labels": CLASS_LABELS[task_id],
        }
    scores = [row["scores"] for row in rows]
    return {
        "task_type": "generation",
        "primary_metric": "rouge_l_f1",
        "primary_score": sum(s["rouge_l_f1"] for s in scores) / len(scores),
        "rouge_l_f1": sum(s["rouge_l_f1"] for s in scores) / len(scores),
        "token_f1": sum(s["token_f1"] for s in scores) / len(scores),
        "exact_match": sum(s["exact_match"] for s in scores) / len(scores),
        "evaluated": len(rows),
    }


def normalized_gain(score: float, base_score: float) -> float:
    headroom = max(1.0 - base_score, 1e-8)
    return (score - base_score) / headroom


def build_cl_summary(
    stages: list[str], task_ids: list[str], scores: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    base = {task: scores[stages[0]][task]["primary_score"] for task in task_ids}
    normalized = {
        stage: {task: normalized_gain(scores[stage][task]["primary_score"], base[task]) for task in task_ids}
        for stage in stages
    }
    matrix: dict[str, Any] = {}
    seen: list[str] = []
    post_stage_seen: dict[str, list[str]] = {}
    for index, stage in enumerate(stages):
        if index > 0:
            seen.append(task_ids[index - 1])
        if stage == "baseline":
            seen_now: list[str] = []
        else:
            seen_now = list(seen)
        post_stage_seen[stage] = seen_now
        if seen_now:
            best_prev: dict[str, float] = {}
            for task in seen_now[:-1]:
                prior_values = [normalized[s][task] for s in stages[1:index] if task in post_stage_seen.get(s, [])]
                best_prev[task] = max(prior_values) if prior_values else 0.0
            forgetting_by_task = {
                task: best_prev[task] - normalized[stage][task] for task in seen_now[:-1]
            }
            average_forgetting = sum(forgetting_by_task.values()) / len(forgetting_by_task) if forgetting_by_task else 0.0
            aaa = sum(normalized[stage][task] for task in seen_now) / len(seen_now)
        else:
            forgetting_by_task = {}
            average_forgetting = 0.0
            aaa = 0.0
        metric_groups: dict[str, list[float]] = {}
        for task in task_ids:
            metric = scores[stage][task]["primary_metric"]
            metric_groups.setdefault(metric, []).append(scores[stage][task]["primary_score"])
        matrix[stage] = {
            "tasks": scores[stage],
            "normalized_gain_vs_base": normalized[stage],
            "average_accuracy_by_metric": {key: sum(vals) / len(vals) for key, vals in metric_groups.items()},
            "normalized_average_accuracy_all_tasks": sum(normalized[stage].values()) / len(task_ids),
            "aaa_normalized_seen_tasks": aaa,
            "average_forgetting_normalized": average_forgetting,
            "forgetting_by_prior_task_normalized": forgetting_by_task,
            "tasks_seen": seen_now,
        }
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-diagnostics-only", action="store_true", help="Refresh token/truncation metadata without rerunning generation")
    args = parser.parse_args()
    start_wall = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "continual_baseline.yaml").read_text(encoding="utf-8"))
    data_root = Path(config["dataset"]["processed_dir"])
    manifest = json.loads((data_root / "data_manifest.json").read_text(encoding="utf-8"))
    task_infos = manifest["tasks"]
    task_ids = [task["task_id"] for task in task_infos]
    result_root = Path(config["storage"]["results"]) / "pilot_audit"
    result_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(config["storage"]["checkpoints"]) / "baseline_pilot"
    stage_specs = [("baseline", checkpoint_root / "initial_base_model")]
    stage_specs.extend(
        (f"after_{i}_{task_id}", checkpoint_root / f"stage_{i}_{task_id}")
        for i, task_id in enumerate(task_ids, start=1)
    )

    tests = {task["task_id"]: read_jsonl(Path(task["split_files"]["test"])) for task in task_infos}
    sample_ids: dict[str, set[str]] = {}
    for index, task_id in enumerate(task_ids):
        sample_indices = random.Random(AUDIT_SAMPLE_SEED + index).sample(range(len(tests[task_id])), AUDIT_SAMPLE_N)
        sample_ids[task_id] = {tests[task_id][i]["instance_id"] for i in sample_indices}

    model_value = os.environ.get(config["model"]["env_var"])
    if not model_value:
        raise EnvironmentError("MODEL_DIR is missing from the project .env")
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if args.refresh_diagnostics_only:
        refresh_existing_diagnostics(tokenizer, task_infos, data_root, config)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    eos_ids = set(tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id])
    original_predictions_root = Path(config["storage"]["results"]) / "baseline_pilot" / "predictions"
    all_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    consistency: dict[str, dict[str, Any]] = {}
    stage_memory: dict[str, dict[str, int]] = {}

    for stage, checkpoint_path in stage_specs:
        print(f"Loading checkpoint {stage}: {checkpoint_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        model.config.use_cache = True
        torch.cuda.reset_peak_memory_stats()
        metrics[stage] = {}
        consistency[stage] = {}
        for task_id in task_ids:
            task_rows: list[dict[str, Any]] = []
            old_path = original_predictions_root / stage / f"{task_id}.jsonl"
            old_by_id = {row["instance_id"]: row for row in read_jsonl(old_path)} if old_path.is_file() else {}
            for index, record in enumerate(tests[task_id], start=1):
                messages = make_messages(record)
                tokenized = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
                input_ids_all = get_token_ids(tokenized)
                prompt_tokens = input_ids_all[-MAX_INPUT_TOKENS:]
                dropped = len(input_ids_all) - len(prompt_tokens)
                decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                prompt_norm = normalized_text(decoded_prompt)
                definition_norm = normalized_text(record["definition"])
                definition_prefix_norm = normalized_text(record["definition"][:80])
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
                attention = torch.ones_like(input_tensor)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        input_ids=input_tensor,
                        attention_mask=attention,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                generated_ids = output_ids[0, input_tensor.shape[1]:].tolist()
                raw_generation = tokenizer.decode(generated_ids, skip_special_tokens=False)
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                finished_with_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
                finish_reason = "eos" if finished_with_eos else ("max_new_tokens" if len(generated_ids) >= MAX_NEW_TOKENS else "other_stop")
                if task_id in CLASS_LABELS:
                    scores = classification_scores(task_id, prediction, record["targets"])
                    scores["exact_match"] = float(any(normalized_text(prediction) == normalized_text(ref) for ref in record["targets"]))
                    score = scores["accuracy"]
                else:
                    scores = generation_scores(prediction, record["targets"])
                    score = scores["rouge_l_f1"]
                previous = old_by_id.get(record["instance_id"])
                consistent = None if previous is None else (previous["prediction"].strip() == prediction)
                row = {
                    "stage": stage,
                    "task_id": task_id,
                    "task_type": "classification" if task_id in CLASS_LABELS else "generation",
                    "instance_id": record["instance_id"],
                    "source_index": record["source_index"],
                    "input": record["input"],
                    "reference": record["targets"],
                    "prediction": prediction,
                    "raw_generation": raw_generation,
                    "score": score,
                    "scores": scores,
                    "format_parse_ok": scores.get("parse_ok", True),
                    "parsed_prediction": scores.get("parsed_prediction"),
                    "finish_reason": finish_reason,
                    "generated_tokens": len(generated_ids),
                    "finished_with_eos": finished_with_eos,
                    "hit_generation_cap": finish_reason == "max_new_tokens",
                    "prompt_tokens_before_truncation": len(input_ids_all),
                    "prompt_tokens_used": len(prompt_tokens),
                    "prompt_tokens_dropped": dropped,
                    "task_definition_marker_retained": "task definition" in decoded_prompt.casefold(),
                    "task_definition_prefix_retained": bool(definition_prefix_norm and definition_prefix_norm in prompt_norm),
                    "full_task_definition_retained": bool(definition_norm and definition_norm in prompt_norm),
                    "generated_think_marker_present": "<think>" in raw_generation or "</think>" in raw_generation,
                    "stored_prediction_match": consistent,
                    "chat_template_enable_thinking": False,
                }
                task_rows.append(row)
                all_rows.append(row)
                if index % 25 == 0 or index == len(tests[task_id]):
                    print(f"{stage} / {task_id}: {index}/{len(tests[task_id])}", flush=True)
            metrics[stage][task_id] = score_stage_task(task_id, task_rows)
            existing_matches = [r["stored_prediction_match"] for r in task_rows if r["stored_prediction_match"] is not None]
            consistency[stage][task_id] = {
                "existing_rows_compared": len(existing_matches),
                "prediction_match_rate": sum(existing_matches) / len(existing_matches) if existing_matches else None,
            }
        stage_memory[stage] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        del model
        torch.cuda.empty_cache()

    # Every stage/task cell has 50 outputs; the requested 15-row audit subset is a fixed
    # seeded sample per task, reused across all stages.
    audit_rows = [row for row in all_rows if row["instance_id"] in sample_ids[row["task_id"]]]
    with (result_root / "all_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (result_root / "audit_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ordered_stages = [stage for stage, _ in stage_specs]
    task_scores = {
        stage: {task: metrics[stage][task] for task in task_ids}
        for stage in ordered_stages
    }
    cl_summary = build_cl_summary(ordered_stages, task_ids, task_scores)
    n_audit_pairs = len(ordered_stages) * len(task_ids)
    sample_counts = Counter((r["stage"], r["task_id"]) for r in audit_rows)
    format_failures = {
        stage: {
            task: sum(not row["format_parse_ok"] for row in audit_rows if row["stage"] == stage and row["task_id"] == task)
            for task in task_ids if task in CLASS_LABELS
        }
        for stage in ordered_stages
    }
    cap_summary = {
        task: {
            "generated_count": sum(r["task_id"] == task for r in all_rows),
            "hit_cap_count": sum(r["task_id"] == task and r["hit_generation_cap"] for r in all_rows),
            "eos_count": sum(r["task_id"] == task and r["finished_with_eos"] for r in all_rows),
            "prompt_truncated_count": sum(r["task_id"] == task and r["prompt_tokens_dropped"] > 0 for r in all_rows),
            "task_definition_marker_lost_count": sum(r["task_id"] == task and not r["task_definition_marker_retained"] for r in all_rows),
            "task_definition_prefix_lost_count": sum(r["task_id"] == task and not r["task_definition_prefix_retained"] for r in all_rows),
            "generated_think_marker_count": sum(r["task_id"] == task and r["generated_think_marker_present"] for r in all_rows),
        }
        for task in task_ids
    }
    total_seconds = time.perf_counter() - start_wall
    audit_metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_pilot_metrics": str(Path(config["storage"]["results"]) / "baseline_pilot" / "metrics.json"),
        "data_manifest": str(data_root / "data_manifest.json"),
        "source_commit": manifest["source"]["commit"],
        "seed_for_audit_sampling": AUDIT_SAMPLE_SEED,
        "sample_count_per_task_stage": AUDIT_SAMPLE_N,
        "task_ids": task_ids,
        "stage_ids": ordered_stages,
        "task_types": {task: "classification" if task in CLASS_LABELS else "generation" for task in task_ids},
        "task_metrics": metrics,
        "cl_summary": cl_summary,
        "prediction_consistency_with_saved_pilot": consistency,
        "generation_diagnostics": cap_summary,
        "classification_parse_failures_in_audit_sample": format_failures,
        "audit_sample_counts": {f"{stage}/{task}": sample_counts[(stage, task)] for stage in ordered_stages for task in task_ids},
        "stage_gpu_memory": stage_memory,
        "peak_cuda_allocated_bytes": max(v["peak_allocated_bytes"] for v in stage_memory.values()),
        "peak_cuda_reserved_bytes": max(v["peak_reserved_bytes"] for v in stage_memory.values()),
        "audit_predictions": len(all_rows),
        "audit_sample_predictions": len(audit_rows),
        "duration_seconds": total_seconds,
        "metric_definitions": {
            "classification": "Parse a canonical class/choice from the start of the assistant response; report Accuracy and macro-F1 across fixed task labels.",
            "generation_primary": "ROUGE-L F1, best over official references; Token-F1 and normalized Exact Match are also reported.",
            "aaa_normalized_seen_tasks": "Mean per-task baseline-headroom-normalized primary-score gain over tasks learned through that stage.",
            "normalized_average_accuracy_all_tasks": "Mean baseline-headroom-normalized primary-score gain over all tasks evaluated at that stage.",
            "average_forgetting_normalized": "For each prior learned task, best earlier post-training normalized gain minus current normalized gain; average across prior tasks.",
            "normalized_gain": "(stage primary score - base primary score) / max(1 - base primary score, 1e-8); raw task scores remain separately available and are never averaged across task types.",
        },
    }
    write_json(result_root / "metrics.json", audit_metrics)
    write_csv_matrix(result_root / "accuracy_matrix.csv", ordered_stages, task_ids, metrics, cl_summary)
    write_audit_report(audit_metrics, task_infos, result_root)
    print(f"Audit complete: {result_root}", flush=True)
    print(f"Saved {len(all_rows)} full predictions and {len(audit_rows)} audited samples", flush=True)
    print(f"Audit runtime: {total_seconds:.1f}s; peak allocated={audit_metrics['peak_cuda_allocated_bytes'] / (1024 ** 3):.2f} GiB", flush=True)


def refresh_existing_diagnostics(
    tokenizer: Any,
    task_infos: list[dict[str, Any]],
    data_root: Path,
    config: dict[str, Any],
) -> None:
    result_root = Path(config["storage"]["results"]) / "pilot_audit"
    all_path = result_root / "all_predictions.jsonl"
    samples_path = result_root / "audit_samples.jsonl"
    metrics_path = result_root / "metrics.json"
    all_rows = read_jsonl(all_path)
    sample_rows = read_jsonl(samples_path)
    split_diagnostics: dict[str, Any] = {}
    test_rows_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_rows:
        raw = row.get("raw_generation", "")
        row["generated_think_marker_present"] = "<think>" in raw or "</think>" in raw
    for row in sample_rows:
        raw = row.get("raw_generation", "")
        row["generated_think_marker_present"] = "<think>" in raw or "</think>" in raw

    for task in task_infos:
        task_id = task["task_id"]
        split_diagnostics[task_id] = {}
        for split in ("train", "dev", "test"):
            data_rows = read_jsonl(Path(task["split_files"][split]))
            stats = {
                "examples": len(data_rows),
                "prompt_truncated_count": 0,
                "task_definition_prefix_lost_count": 0,
                "full_task_definition_lost_count": 0,
                "assistant_target_capped_count": 0,
                "prompt_entirely_dropped_count": 0,
            }
            diagnostics_by_id: dict[str, dict[str, Any]] = {}
            for record in data_rows:
                diagnostics = prompt_diagnostics(tokenizer, record, training=(split == "train"))
                diagnostics_by_id[record["instance_id"]] = diagnostics
                stats["prompt_truncated_count"] += diagnostics["prompt_tokens_dropped"] > 0
                stats["task_definition_prefix_lost_count"] += not diagnostics["task_definition_prefix_retained"]
                stats["full_task_definition_lost_count"] += not diagnostics["full_task_definition_retained"]
                stats["assistant_target_capped_count"] += diagnostics["assistant_target_capped"]
                stats["prompt_entirely_dropped_count"] += diagnostics["prompt_tokens_retained"] == 0
                if split == "test":
                    test_rows_by_id[(task_id, record["instance_id"])] = diagnostics
            split_diagnostics[task_id][split] = stats
            if split == "test":
                for row in all_rows:
                    if row["task_id"] == task_id:
                        row.update(diagnostics_by_id[row["instance_id"]])
                for row in sample_rows:
                    if row["task_id"] == task_id:
                        row.update(diagnostics_by_id[row["instance_id"]])

    with all_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = json.loads(metrics_path.read_text(encoding="utf-8"))
    audit["input_truncation_diagnostics_by_split"] = split_diagnostics
    for task_id, stats in audit["generation_diagnostics"].items():
        test_stats = split_diagnostics[task_id]["test"]
        stage_count = len(audit["stage_ids"])
        stats["task_definition_marker_lost_count"] = sum(
            not row["task_definition_marker_retained"] for row in all_rows if row["task_id"] == task_id
        )
        stats["task_definition_prefix_lost_count"] = sum(
            not row["task_definition_prefix_retained"] for row in all_rows if row["task_id"] == task_id
        )
        stats["full_task_definition_lost_count"] = sum(
            not row["full_task_definition_retained"] for row in all_rows if row["task_id"] == task_id
        )
        stats["generated_think_marker_count"] = sum(
            row["generated_think_marker_present"] for row in all_rows if row["task_id"] == task_id
        )
        stats["definition_prefix_lost_test_instances"] = test_stats["task_definition_prefix_lost_count"]
        stats["generation_stage_count"] = stage_count
    metrics_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_audit_report(audit, task_infos, result_root)
    print("Refreshed prompt truncation diagnostics from existing predictions; no generation or training was run.")


def write_csv_matrix(
    path: Path,
    stages: list[str],
    task_ids: list[str],
    metrics: dict[str, dict[str, dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    import csv

    columns = ["stage"]
    for task in task_ids:
        columns.extend([f"{task}__primary_score", f"{task}__primary_metric"])
        if task in CLASS_LABELS:
            columns.extend([f"{task}__accuracy", f"{task}__macro_f1"])
        else:
            columns.extend([f"{task}__rouge_l_f1", f"{task}__token_f1", f"{task}__exact_match"])
    columns.extend(["normalized_average_accuracy_all_tasks", "aaa_normalized_seen_tasks", "average_forgetting_normalized"])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for stage in stages:
            row: dict[str, Any] = {"stage": stage}
            for task in task_ids:
                metric = metrics[stage][task]
                row[f"{task}__primary_score"] = metric["primary_score"]
                row[f"{task}__primary_metric"] = metric["primary_metric"]
                if task in CLASS_LABELS:
                    row[f"{task}__accuracy"] = metric["accuracy"]
                    row[f"{task}__macro_f1"] = metric["macro_f1"]
                else:
                    row[f"{task}__rouge_l_f1"] = metric["rouge_l_f1"]
                    row[f"{task}__token_f1"] = metric["token_f1"]
                    row[f"{task}__exact_match"] = metric["exact_match"]
            stage_summary = summary[stage]
            row["normalized_average_accuracy_all_tasks"] = stage_summary["normalized_average_accuracy_all_tasks"]
            row["aaa_normalized_seen_tasks"] = stage_summary["aaa_normalized_seen_tasks"]
            row["average_forgetting_normalized"] = stage_summary["average_forgetting_normalized"]
            writer.writerow(row)


def write_audit_report(audit: dict[str, Any], task_infos: list[dict[str, Any]], result_root: Path) -> None:
    task_names = {item["task_id"]: item for item in task_infos}
    lines = [
        "# Pilot Evaluation Audit",
        "",
        f"Audit run: {audit['created_utc']}",
        f"Source CITB commit: {audit['source_commit']}",
        f"Re-evaluated predictions: {audit['audit_predictions']} (50 per task per stage)",
        f"Saved audit sample: {audit['audit_sample_predictions']} (15 per task per stage)",
        f"Runtime: {audit['duration_seconds']:.1f} seconds; peak CUDA allocated: {audit['peak_cuda_allocated_bytes'] / (1024 ** 3):.2f} GiB.",
        "",
        "## Task formats and scoring decisions",
        "",
        "| Task | Official CITB category | Observed reference format | Scoring type |",
        "|---|---|---|---|",
    ]
    reference_examples = {
        "task848_pubmedqa_classification": "binary label 0 / 1",
        "task611_mutual_multi_turn_dialogue": "choice label A / B / C / D",
        "task1590_diplomacy_text_generation": "free-form next dialogue message",
        "task1500_dstc3_classification": "Cheap / Moderate / Expensive / Don't Care",
    }
    for task, info in task_names.items():
        scoring = "Accuracy + fixed-label Macro-F1; anchored answer parser"
        if task in GENERATION_TASKS:
            scoring = "ROUGE-L F1 primary; Token-F1 and normalized Exact Match auxiliary"
        lines.append(f"| {task} | {', '.join(info['category'])} | {reference_examples[task]} | {scoring} |")
    lines.extend(
        [
            "",
            "The task category alone does not determine the output type: task611 is categorized as Dialogue Generation by CITB, but its instances expect a multiple-choice letter, so it is evaluated as classification after parsing the selected option.",
            "",
            "## Pilot scoring defect found and corrected",
            "",
            "The previous Exact Match evaluator compared the complete generated string with the canonical reference. On task611, the base model often returned a complete option such as ‘(C) ...’ while the reference was only ‘A’. The old evaluator marked every base prediction wrong (0.00) even when the selected option could be parsed. Task611 after later tasks could also emit prose or a different task's label; these are counted as parser failures rather than guessed from arbitrary letters in the text.",
            "",
            "The previous evaluator used exact string match as the sole score for task1590, an open-ended generation task. This is too strict to measure semantic overlap. The pilot has been re-evaluated over the full 50-example test split at all five checkpoints with ROUGE-L F1 and Token-F1 as primary/secondary overlap measures, while retaining Exact Match as auxiliary. No test result was used to select tasks, sequences, or hyperparameters.",
            "",
            "## Full task-by-stage matrix",
            "",
            "See the companion accuracy_matrix.csv for a wide machine-readable matrix. Raw classification and generation scores are kept separate by metric name.",
            "",
            "| Stage | Task | Primary metric | Score | Accuracy | Macro-F1 | ROUGE-L | Token-F1 | Exact Match | Parse success |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in audit["stage_ids"]:
        for task in audit["task_ids"]:
            m = audit["task_metrics"][stage][task]
            lines.append(
                f"| {stage} | {task} | {m['primary_metric']} | {m['primary_score']:.4f} | "
                f"{m.get('accuracy', '')} | {m.get('macro_f1', '')} | {m.get('rouge_l_f1', '')} | "
                f"{m.get('token_f1', '')} | {m.get('exact_match', '')} | {m.get('parse_success_rate', '')} |"
            )
    lines.extend(
        [
            "",
            "## Answer parsing, format, EOS, and truncation diagnostics",
            "",
            "Each per-example audit record contains the original input, all references, visible and raw token-decoded prediction, per-example score, parsed class, parse status, prompt token counts, EOS termination, max-token-cap status, and whether the regenerated visible output matched the stored pilot output.",
            "",
            "| Task | Outputs | EOS terminated | Hit 64-token cap | Prompt truncated at 512 | Definition prefix lost | Raw think marker |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, item in audit["generation_diagnostics"].items():
        lines.append(
            f"| {task} | {item['generated_count']} | {item['eos_count']} | {item['hit_cap_count']} | "
            f"{item['prompt_truncated_count']} | {item['task_definition_prefix_lost_count']} | {item['generated_think_marker_count']} |"
        )
    lines.extend(
        [
            "",
            "Training-time input diagnostics (the SFT encoder reserves space for the target, so its retained prompt may be shorter than 512 tokens):",
            "",
            "| Task | Split | Examples | Prompt truncated | Definition prefix lost | Target capped | Prompt fully dropped |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, splits in audit.get("input_truncation_diagnostics_by_split", {}).items():
        for split, stats in splits.items():
            lines.append(
                f"| {task} | {split} | {stats['examples']} | {stats['prompt_truncated_count']} | "
                f"{stats['task_definition_prefix_lost_count']} | {stats['assistant_target_capped_count']} | {stats['prompt_entirely_dropped_count']} |"
            )
    lines.extend(
        [
            "",
            "Classification parse failures in the 15-row audit samples by stage:",
            "",
            "```json",
            json.dumps(audit["classification_parse_failures_in_audit_sample"], ensure_ascii=False, indent=2),
            "```",
            "",
            "Stored-pilot prediction consistency rates for comparable rows are in metrics.json. Chat template generation was rerun with enable_thinking=False; prompt truncation always retained the final 512 prompt tokens, matching the pilot runner. The definition-prefix counts identify cases where tail truncation removed the task instructions; those cases are a prompt construction defect to fix before the formal run. Raw generated think markers are counted separately from template-provided assistant prefixes.",
            "",
            "## Cross-task Average Accuracy, AAA, and forgetting",
            "",
            "Primary task scores have different interpretations. They are never averaged raw across task types. Per-task normalized gain is (stage primary score - base primary score) / max(1 - base primary score, 1e-8). The all-task Average Accuracy is the mean normalized gain over all four tasks. AAA is the mean normalized gain over tasks seen through that stage. Forgetting is the mean, over prior tasks, of the best earlier post-training normalized gain minus the current normalized gain. Raw averages remain grouped by metric in metrics.json.",
            "",
            "| Stage | Normalized average accuracy (all tasks) | AAA (seen tasks) | Normalized average forgetting |",
            "|---|---:|---:|---:|",
        ]
    )
    for stage in audit["stage_ids"]:
        s = audit["cl_summary"][stage]
        lines.append(
            f"| {stage} | {s['normalized_average_accuracy_all_tasks']:.4f} | "
            f"{s['aaa_normalized_seen_tasks']:.4f} | {s['average_forgetting_normalized']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Audit artifacts and next step",
            "",
            f"- Full re-evaluation predictions: {result_root / 'all_predictions.jsonl'}",
            f"- Deterministic 15-per-task/per-stage input/reference/prediction/score sample: {result_root / 'audit_samples.jsonl'}",
            f"- Complete task-by-stage metric matrix: {result_root / 'metrics.json'} and {result_root / 'accuracy_matrix.csv'}",
            "",
            "The original pilot's task611 exact-match score is invalid as a choice-classification measure, and task1590 exact match alone is inadequate as a generation measure. Use the corrected metrics above for pilot interpretation. Only after this audit is complete should development-set comparisons be used to choose the new balanced tasks and formal-run hyperparameters; the test split must remain excluded from those choices.",
            "",
        ]
    )
    (PROJECT_ROOT / "reports" / "pilot_evaluation_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
