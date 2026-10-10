"""Compare eligible CITB tasks using frozen Qwen representations and supervised gradients."""

from __future__ import annotations

import itertools
import hashlib
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path("D:/Task-Aware-TransNAS")
os.environ.setdefault("HF_HOME", str(STORAGE_ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(STORAGE_ROOT / "cache" / "huggingface" / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer


SAMPLE_N = 32
GRADIENT_N = 8
TRAIN_CAP = 500
MAX_SEQUENCE_LENGTH = 384
MAX_TARGET_TOKENS = 64
ANALYSIS_SEED = 314159


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def ids_from(result: Any) -> list[int]:
    if hasattr(result, "keys"):
        result = result["input_ids"]
    if isinstance(result, torch.Tensor):
        result = result.tolist()
    if result and isinstance(result[0], list):
        result = result[0]
    return [int(value) for value in result]


def build_messages(record: dict[str, Any], input_text: str | None = None) -> list[dict[str, str]]:
    system = "You are an instruction-following assistant. Follow the task definition and return only the requested answer."
    text = "Task definition:\n" + record["definition"] + "\n\nInput:\n" + (record["input"] if input_text is None else input_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def prompt_ids(tokenizer: Any, record: dict[str, Any], budget: int) -> tuple[list[int], int, bool, str]:
    messages = build_messages(record)
    full_ids = ids_from(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    if len(full_ids) <= budget:
        return full_ids, 0, False, record["input"]
    instruction_only = build_messages(record, input_text="")
    instruction_ids = ids_from(tokenizer.apply_chat_template(instruction_only, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    if len(instruction_ids) + 8 > budget:
        raise ValueError(f"Task definition alone does not fit the prompt budget: {record['task_id']} {len(instruction_ids)}>{budget}")
    input_tokens = tokenizer.encode(record["input"], add_special_tokens=False)
    keep = max(0, min(len(input_tokens), budget - len(instruction_ids) - 8))
    was_truncated = keep < len(input_tokens)
    for _ in range(8):
        input_tail = tokenizer.decode(input_tokens[-keep:], skip_special_tokens=True) if keep else ""
        candidate = ids_from(
            tokenizer.apply_chat_template(
                build_messages(record, input_text=input_tail),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if len(candidate) <= budget:
            return candidate, len(full_ids) - len(candidate), was_truncated, input_tail
        keep = max(0, keep - (len(candidate) - budget + 4))
    raise RuntimeError(f"Could not fit prompt while preserving task instructions for {record['instance_id']}")


def encode_sft(tokenizer: Any, record: dict[str, Any]) -> tuple[list[int], list[int], int, bool, bool]:
    target = record["targets"][0] if record["targets"] else ""
    messages = build_messages(record)
    original_prompt = ids_from(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))
    original_full = ids_from(
        tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target}], tokenize=True, add_generation_prompt=False, enable_thinking=False)
    )
    if original_full[: len(original_prompt)] != original_prompt:
        raise ValueError(f"Prompt/answer prefix mismatch in {record['instance_id']}")
    original_target = original_full[len(original_prompt):]
    target_capped = len(original_target) >= MAX_TARGET_TOKENS
    if target_capped:
        answer_tokens = original_target[: MAX_TARGET_TOKENS - 1] + [tokenizer.eos_token_id]
    else:
        answer_tokens = original_target
    prompt_budget = MAX_SEQUENCE_LENGTH - len(answer_tokens)
    if prompt_budget < 16:
        raise ValueError(f"Not enough sequence budget for prompt/answer in {record['instance_id']}")
    prompt, _, prompt_input_truncated, shortened_input = prompt_ids(tokenizer, record, prompt_budget)
    full = ids_from(
        tokenizer.apply_chat_template(
            build_messages(record, input_text=shortened_input) + [{"role": "assistant", "content": target}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )
    if full[: len(prompt)] != prompt:
        raise ValueError(f"Retained-prompt/answer prefix mismatch in {record['instance_id']}")
    actual_target = full[len(prompt):]
    if len(actual_target) >= MAX_TARGET_TOKENS:
        actual_target = actual_target[: MAX_TARGET_TOKENS - 1] + [tokenizer.eos_token_id]
        target_capped = True
    if len(prompt) + len(actual_target) > MAX_SEQUENCE_LENGTH:
        prompt_budget = MAX_SEQUENCE_LENGTH - len(actual_target)
        prompt, _, prompt_input_truncated, shortened_input = prompt_ids(tokenizer, record, prompt_budget)
        full = ids_from(
            tokenizer.apply_chat_template(
                build_messages(record, input_text=shortened_input) + [{"role": "assistant", "content": target}],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        )
        if full[: len(prompt)] != prompt:
            raise ValueError(f"Retained-prompt/answer prefix mismatch after budget adjustment in {record['instance_id']}")
        actual_target = full[len(prompt):]
        if len(actual_target) >= MAX_TARGET_TOKENS:
            actual_target = actual_target[: MAX_TARGET_TOKENS - 1] + [tokenizer.eos_token_id]
            target_capped = True
    input_ids = prompt + actual_target
    labels = [-100] * len(prompt) + actual_target
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError(f"Invalid gradient sample encoding for {record['instance_id']}")
    return input_ids, labels, len(prompt), prompt_input_truncated, target_capped


def clean_record(task: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    outputs = instance.get("output", [])
    if isinstance(outputs, str):
        outputs = [outputs]
    return {
        "task_id": task["task_id"],
        "category": task["Categories"],
        "instance_id": instance["id"],
        "definition": as_text(task.get("Definition", [])),
        "input": as_text(instance.get("input", "")),
        "targets": [as_text(value) for value in outputs],
    }


def output_type(records: list[dict[str, Any]]) -> dict[str, Any]:
    refs = [target for row in records for target in row["targets"]]
    unique = len(set(refs))
    median_chars = statistics.median([len(ref) for ref in refs]) if refs else 0
    is_classification = median_chars <= 12 and unique <= 20
    return {
        "heuristic_type": "classification" if is_classification else "generation",
        "unique_reference_strings_in_analysis_sample": unique,
        "median_reference_chars_in_analysis_sample": median_chars,
    }


def batch_pad(rows: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    input_ids = [row + [pad_id] * (width - len(row)) for row in rows]
    mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
    return torch.tensor(input_ids, dtype=torch.long, device=device), torch.tensor(mask, dtype=torch.long, device=device)


def zscore_pair_matrix(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    indices = torch.triu_indices(n, n, offset=1)
    values = matrix[indices[0], indices[1]]
    std = values.std(unbiased=False).clamp_min(1e-8)
    normalized = (matrix - values.mean()) / std
    normalized.fill_diagonal_(0.0)
    return normalized


def pairings(four: tuple[int, int, int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    a, b, c, d = four
    return [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]


def find_best_two_pair_clusters(
    score: torch.Tensor, task_ids: list[str], task_docs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    n = len(task_ids)
    doc_by_id = {document["task_id"]: document for document in task_docs}
    for four in itertools.combinations(range(n), 4):
        selected_docs = [doc_by_id[task_ids[index]] for index in four]
        categories = [document["official_category"] for document in selected_docs]
        output_types = [document["response_format_analysis"]["heuristic_type"] for document in selected_docs]
        if any(len(category) != 1 for category in categories):
            continue
        category_labels = [category[0] for category in categories]
        if len(set(category_labels)) != 2 or sorted(Counter(category_labels).values()) != [2, 2]:
            continue
        if sorted(Counter(output_types).values()) != [2, 2]:
            continue
        for first, second in pairings(four):
            first_docs = [doc_by_id[task_ids[index]] for index in first]
            second_docs = [doc_by_id[task_ids[index]] for index in second]
            first_categories = [document["official_category"][0] for document in first_docs]
            second_categories = [document["official_category"][0] for document in second_docs]
            first_types = [document["response_format_analysis"]["heuristic_type"] for document in first_docs]
            second_types = [document["response_format_analysis"]["heuristic_type"] for document in second_docs]
            if len(set(first_categories)) != 1 or len(set(second_categories)) != 1:
                continue
            if first_categories[0] == second_categories[0] or first_types[0] != first_types[1] or second_types[0] != second_types[1]:
                continue
            if first_types[0] == second_types[0]:
                continue
            within_values = [float(score[a, b]) for a, b in (first, second)]
            cross_values = [float(score[a, b]) for a in first for b in second]
            within = sum(within_values) / 2
            cross = sum(cross_values) / 4
            min_within = min(within_values)
            separation = within - cross
            candidates.append(
                {
                    "pair_a": [task_ids[first[0]], task_ids[first[1]]],
                    "pair_b": [task_ids[second[0]], task_ids[second[1]]],
                    "within_similarity_z": within,
                    "minimum_pair_similarity_z": min_within,
                    "cross_similarity_z": cross,
                    "separation_z": separation,
                    "selection_score": separation + 0.25 * min_within,
                    "category_pair_a": first_categories[0],
                    "category_pair_b": second_categories[0],
                    "output_type_pair_a": first_types[0],
                    "output_type_pair_b": second_types[0],
                }
            )
    if not candidates:
        raise RuntimeError("No four-task selection satisfies the 2x2 category/output-type balance constraints")
    return sorted(candidates, key=lambda row: row["selection_score"], reverse=True)


def main() -> None:
    start = time.perf_counter()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "continual_baseline.yaml").read_text(encoding="utf-8"))
    citb_data = Path(config["dataset"]["source_data"])
    task_order = [line.strip() for line in (citb_data / "CIT_data/task_orders/stream=cl_dialogue_tasks/order1.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    split_file = citb_data / "splits/CIT_splits/cl_dialogue_tasks.txt"
    official_stream = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    task_docs = []
    chosen_records: dict[str, list[dict[str, Any]]] = {}
    analysis_metadata = {}
    for position, task_id in enumerate(task_order, start=1):
        if task_id not in official_stream:
            raise ValueError(f"Task missing from official cl_dialogue_tasks split: {task_id}")
        task = json.loads((citb_data / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        instances = task["Instances"]
        train_pool = instances[100:]
        train_count = min(TRAIN_CAP, len(train_pool))
        if train_count < TRAIN_CAP:
            continue
        rng = random.Random(ANALYSIS_SEED + position)
        selected = rng.sample(train_pool, TRAIN_CAP)
        sample_rows = [clean_record(task | {"task_id": task_id}, row) for row in selected[:SAMPLE_N]]
        gradient_rows = [clean_record(task | {"task_id": task_id}, row) for row in selected[:GRADIENT_N]]
        info = {
            "task_id": task_id,
            "official_stream_position_1based": position,
            "official_category": task.get("Categories", []),
            "raw_instance_count": len(instances),
            "official_test_count": 50,
            "official_dev_count": 50,
            "available_train_pool_count": len(train_pool),
            "balanced_train_cap": TRAIN_CAP,
            "selection_train_ids_sha": hashlib.sha256(
                "\n".join(sorted(instance["id"] for instance in selected)).encode("utf-8")
            ).hexdigest(),
            "balanced_training_ids": [instance["id"] for instance in selected],
            "analysis_sample_ids": [row["instance_id"] for row in sample_rows],
            "gradient_sample_ids": [row["instance_id"] for row in gradient_rows],
            "definition": as_text(task.get("Definition", [])),
            "response_format_analysis": output_type(sample_rows),
        }
        task_docs.append(info)
        chosen_records[task_id] = sample_rows
        analysis_metadata[task_id] = {"task": task, "gradient_records": gradient_rows}
    candidate_ids = [entry["task_id"] for entry in task_docs]
    if len(candidate_ids) < 4:
        raise RuntimeError("Fewer than four eligible tasks have 500 training instances after official holdouts")

    model_value = os.environ.get(config["model"]["env_var"])
    if not model_value:
        raise EnvironmentError("MODEL_DIR is missing from .env")
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen representation and gradient analysis")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, dtype=torch.float32, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    representations = []
    truncation = {}
    for task_id in candidate_ids:
        rows = chosen_records[task_id]
        encoded = []
        n_truncated = 0
        for record in rows:
            tokens, _, was_truncated, _ = prompt_ids(tokenizer, record, MAX_SEQUENCE_LENGTH)
            encoded.append(tokens)
            n_truncated += was_truncated
        vector_list = []
        for start_idx in range(0, len(encoded), 4):
            batch_tokens = encoded[start_idx : start_idx + 4]
            input_ids, attention = batch_pad(batch_tokens, tokenizer.pad_token_id, device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids=input_ids, attention_mask=attention, use_cache=False, output_hidden_states=True)
            hidden = output.hidden_states[-1].float()
            lengths = attention.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=device), lengths]
            vector_list.append(F.normalize(pooled, p=2, dim=-1).cpu())
        centroid = F.normalize(torch.cat(vector_list, dim=0).mean(dim=0), p=2, dim=0)
        representations.append(centroid)
        truncation[task_id] = {"representation_prompts": len(rows), "input_tail_truncated_count": n_truncated}
        print(f"representation {task_id}: {len(rows)} train prompts, truncated={n_truncated}", flush=True)
    rep_matrix = torch.stack(representations) @ torch.stack(representations).T

    selected_names = []
    final_layer_index = len(model.model.layers) - 1
    final_layer_prefix = f"model.layers.{final_layer_index}."
    for name, parameter in model.named_parameters():
        if name.startswith(final_layer_prefix) and (name.endswith("self_attn.o_proj.weight") or name.endswith("mlp.down_proj.weight")):
            selected_names.append(name)
            parameter.requires_grad_(True)
    if len(selected_names) != 2:
        raise RuntimeError(f"Expected final-block o_proj and down_proj weights, found {selected_names}")
    selected_parameters = [dict(model.named_parameters())[name] for name in selected_names]
    gradient_vectors = []
    gradient_norms = {}
    gradient_truncation = {}
    for task_id in candidate_ids:
        accum = None
        for record in analysis_metadata[task_id]["gradient_records"]:
            input_ids, labels, prompt_length, was_truncated, target_capped = encode_sft(tokenizer, record)
            if len(input_ids) > MAX_SEQUENCE_LENGTH:
                raise AssertionError("Gradient example exceeds the agreed analysis token budget")
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            attention = torch.ones_like(x)
            for parameter in selected_parameters:
                parameter.grad = None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, attention_mask=attention, labels=y, use_cache=False).loss
            loss.backward()
            vector = torch.cat([parameter.grad.detach().float().reshape(-1).cpu() for parameter in selected_parameters])
            accum = vector if accum is None else accum + vector
            gradient_truncation.setdefault(task_id, []).append(
                {"prompt_tokens": prompt_length, "prompt_input_truncated": was_truncated, "target_capped": target_capped, "loss": float(loss.detach().float().item())}
            )
        mean_gradient = accum / len(analysis_metadata[task_id]["gradient_records"])
        gradient_norms[task_id] = float(mean_gradient.norm().item())
        gradient_vectors.append(F.normalize(mean_gradient, p=2, dim=0))
        for parameter in selected_parameters:
            parameter.grad = None
        torch.cuda.empty_cache()
        print(f"gradient {task_id}: samples={GRADIENT_N}, norm={gradient_norms[task_id]:.6g}", flush=True)
    grad_matrix = torch.stack(gradient_vectors) @ torch.stack(gradient_vectors).T
    combined = 0.5 * zscore_pair_matrix(rep_matrix) + 0.5 * zscore_pair_matrix(grad_matrix)

    pair_rows = []
    for i, j in itertools.combinations(range(len(candidate_ids)), 2):
        pair_rows.append(
            {
                "task_a": candidate_ids[i],
                "task_b": candidate_ids[j],
                "representation_cosine": float(rep_matrix[i, j]),
                "gradient_cosine": float(grad_matrix[i, j]),
                "combined_similarity_z": float(combined[i, j]),
                "same_official_category": bool(set(task_docs[i]["official_category"]) & set(task_docs[j]["official_category"])),
            }
        )
    best_matchings = find_best_two_pair_clusters(combined, candidate_ids, task_docs)
    best = best_matchings[0]
    pair_a, pair_b = best["pair_a"], best["pair_b"]
    sequences = {
        "homogeneous": [pair_a[0], pair_a[1], pair_b[0], pair_b[1]],
        "heterogeneous": [pair_a[0], pair_b[0], pair_a[1], pair_b[1]],
        "mixed": [pair_a[0], pair_b[0], pair_b[1], pair_a[1]],
    }
    idx = {task_id: i for i, task_id in enumerate(candidate_ids)}
    sequence_diagnostics = {}
    for name, ordered in sequences.items():
        edges = []
        for left, right in zip(ordered, ordered[1:]):
            i, j = idx[left], idx[right]
            edges.append(
                {
                    "left": left,
                    "right": right,
                    "representation_cosine": float(rep_matrix[i, j]),
                    "gradient_cosine": float(grad_matrix[i, j]),
                    "combined_similarity_z": float(combined[i, j]),
                }
            )
        sequence_diagnostics[name] = {"task_ids": ordered, "adjacent_pairs": edges, "mean_adjacent_similarity_z": statistics.mean(edge["combined_similarity_z"] for edge in edges)}

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_seed": ANALYSIS_SEED,
        "data_source_commit": __import__("subprocess").check_output(
            ["git", "-C", str(Path(config["dataset"]["source_repo"])), "rev-parse", "HEAD"], text=True
        ).strip(),
        "selection_guard": "Only the official order1 train pool (source indices >=100) was used for task relationships. First 50 test records were not loaded into representations or gradient batches; the next 50 dev records were not used for task choice.",
        "training_cap_per_candidate": TRAIN_CAP,
        "representation_sample_per_candidate": SAMPLE_N,
        "gradient_sample_per_candidate": GRADIENT_N,
        "gradient_parameter_group": selected_names,
        "gradient_parameter_count": sum(p.numel() for p in selected_parameters),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "max_target_tokens": MAX_TARGET_TOKENS,
        "candidates": task_docs,
        "representation_truncation": truncation,
        "gradient_sample_diagnostics": gradient_truncation,
        "gradient_norms": gradient_norms,
        "representation_cosine_matrix": {candidate_ids[i]: {candidate_ids[j]: float(rep_matrix[i, j]) for j in range(len(candidate_ids))} for i in range(len(candidate_ids))},
        "gradient_cosine_matrix": {candidate_ids[i]: {candidate_ids[j]: float(grad_matrix[i, j]) for j in range(len(candidate_ids))} for i in range(len(candidate_ids))},
        "combined_similarity_z_matrix": {candidate_ids[i]: {candidate_ids[j]: float(combined[i, j]) for j in range(len(candidate_ids))} for i in range(len(candidate_ids))},
        "top_pairwise_similarities": sorted(pair_rows, key=lambda row: row["combined_similarity_z"], reverse=True)[:20],
        "top_cluster_matchings": best_matchings[:20],
        "selected_task_set": best["pair_a"] + best["pair_b"],
        "selected_cluster_pairing": best,
        "sequences": sequence_diagnostics,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "analysis_duration_seconds": time.perf_counter() - start,
    }
    output_root = STORAGE_ROOT / "data" / "task_selection"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "task_relation_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matrix_rows = []
    for row in sorted(pair_rows, key=lambda item: item["combined_similarity_z"], reverse=True):
        matrix_rows.append(
            f"| {row['task_a']} | {row['task_b']} | {row['representation_cosine']:.4f} | {row['gradient_cosine']:.4f} | {row['combined_similarity_z']:.3f} | {row['same_official_category']} |"
        )
    report = [
        "# CITB Task Relation Analysis",
        "",
        "This selection analysis used only order1 training records after the official 50-test / 50-dev holdouts. Each eligible candidate has exactly 500 training examples after the cap. No test scores or test generations were used.",
        "",
        f"Frozen Qwen3-0.6B representations: mean last-token prompt embeddings over {SAMPLE_N} training records/task. Supervised gradient similarity: cosine of mean gradients over {GRADIENT_N} training records/task for the final-block o_proj and down_proj weights. Each analysis prompt retained the full task definition and truncated only the input tail to fit {MAX_SEQUENCE_LENGTH} tokens.",
        "",
        "## Selection constraints and measured pairing",
        "",
        "The selected set is constrained to contain two tasks from each of two official CITB categories and two tasks of each reference format (classification or generation). Each similarity pair is homogeneous in both category and reference format; the two pairs differ on both. Within-pair and cross-pair ordering is then chosen from the measured combined similarity.",
        "",
        json.dumps(best, ensure_ascii=False, indent=2),
        "",
        "## Three reproducible sequences",
        "",
    ]
    for name, detail in sequence_diagnostics.items():
        report.append(f"### {name.title()}")
        report.append("")
        report.append(" -> ".join(detail["task_ids"]))
        report.append("")
        report.append(f"Mean adjacent combined similarity z: {detail['mean_adjacent_similarity_z']:.3f}")
        report.append("")
    report.extend(
        [
            "## Highest measured pairwise similarities",
            "",
            "| Task A | Task B | Frozen representation cosine | Gradient cosine | Combined z | Same official category |",
            "|---|---|---:|---:|---:|---|",
            *matrix_rows[:20],
            "",
            f"Full pair matrices, candidate counts/categories, sampled training IDs, gradient norms, and token truncation diagnostics are in {output_root / 'task_relation_analysis.json'}.",
            "",
        ]
    )
    report_path = PROJECT_ROOT / "reports" / "task_relation_analysis.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Selected set: {result['selected_task_set']}", flush=True)
    for name, details in sequence_diagnostics.items():
        print(f"{name}: {details['task_ids']} | mean adjacent z={details['mean_adjacent_similarity_z']:.3f}", flush=True)
    print(f"Analysis report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
