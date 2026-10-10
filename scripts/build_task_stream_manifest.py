"""Combine the frozen task-similarity selection and deterministic CITB split into one manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "continual_baseline_full.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["dataset"]["processed_dir"])
    data_manifest = json.loads((data_root / "data_manifest.json").read_text(encoding="utf-8"))
    relation_path = Path("D:/Task-Aware-TransNAS/data/task_selection/task_relation_analysis.json")
    relations = json.loads(relation_path.read_text(encoding="utf-8"))

    task_infos = {entry["task_id"]: entry for entry in data_manifest["tasks"]}
    relation_docs = {entry["task_id"]: entry for entry in relations["candidates"]}
    selected = relations["selected_task_set"]
    if set(selected) != set(config["dataset"]["tasks"]):
        raise ValueError("Formal config task set differs from the frozen similarity selection")

    output_tasks = []
    for task_id in selected:
        info = task_infos[task_id]
        relation = relation_docs[task_id]
        task_type = config["dataset"]["task_types"][task_id]
        if info["counts"] != {"train": 500, "dev": 50, "test": 50}:
            raise ValueError(f"Task {task_id} is not balanced at 500/50/50: {info['counts']}")
        ids = info["instance_ids"]
        if set(ids["train"]) & set(ids["dev"]) or set(ids["train"]) & set(ids["test"]) or set(ids["dev"]) & set(ids["test"]):
            raise ValueError(f"Split leakage detected for {task_id}")
        output_tasks.append(
            {
                "task_id": task_id,
                "official_stream_position_1based": info["stream_position_1based"],
                "official_category": info["category"],
                "output_type": task_type,
                "source_instance_count": info["source_instance_count"],
                "source_json_sha256": info["source_sha256"],
                "counts": info["counts"],
                "instance_ids": ids,
                "instance_id_sha256": {name: sha256_text(values) for name, values in ids.items()},
                "split_files": info["split_files"],
                "relation_analysis_train_ids_sha256": relation["selection_train_ids_sha"],
                "relation_analysis_sample_ids": relation["analysis_sample_ids"],
                "relation_gradient_sample_ids": relation["gradient_sample_ids"],
                "relation_embedding_truncation": relations["representation_truncation"][task_id],
                "relation_gradient_truncation": relations["gradient_sample_diagnostics"][task_id],
            }
        )

    leakage_ok = all(
        not (set(task["instance_ids"]["train"]) & set(task["instance_ids"]["dev"]))
        and not (set(task["instance_ids"]["train"]) & set(task["instance_ids"]["test"]))
        and not (set(task["instance_ids"]["dev"]) & set(task["instance_ids"]["test"]))
        for task in output_tasks
    )
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Task-Aware-TransNAS",
        "formal_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "selection_analysis": {
            "artifact": str(relation_path),
            "analysis_seed": relations["analysis_seed"],
            "data_source_commit": relations["data_source_commit"],
            "method": "Equal-weight mean z-score of frozen final-token prompt-representation cosine and supervised gradient cosine for final-block attention o_proj and MLP down_proj weights.",
            "representation_examples_per_candidate": relations["representation_sample_per_candidate"],
            "gradient_examples_per_candidate": relations["gradient_sample_per_candidate"],
            "training_pool_cap": relations["training_cap_per_candidate"],
            "selection_guard": relations["selection_guard"],
            "balance_constraints": "Two tasks per official CITB category and two tasks per reference format; each similarity pair is homogeneous by both category and format, and the pairs differ on both.",
            "selected_cluster_pairing": relations["selected_cluster_pairing"],
            "selected_task_set": selected,
            "sequence_similarity_diagnostics": relations["sequences"],
        },
        "data_source": data_manifest["source"],
        "split_rule": data_manifest["source"]["split_rule"],
        "seed": config["seed"],
        "training_seeds": config.get("training_seeds", [config["seed"]]),
        "caps": data_manifest["caps"],
        "leakage_check": {"passed": leakage_ok, "scope": "train/dev/test IDs are disjoint within every task"},
        "prompt_policy": {
            "chat_template_thinking": False,
            "instruction_preservation": "Always retain complete system message and task definition; only retain a suffix of long input context when the sequence budget is exceeded.",
            "max_sequence_length": config["model"]["max_sequence_length"],
            "max_target_tokens": config["model"]["max_target_tokens"],
        },
        "sequences": config["sequences"],
        "tuning_policy": {
            "split": "development only",
            "sequence": config["tuning"]["sequence"],
            "learning_rates": config["tuning"]["learning_rates"],
            "steps_per_task": config["tuning"]["steps_per_task"],
            "selection_metric": config["tuning"]["selection_metric"],
            "test_access_before_freezing": False,
            "selected": config["tuning"].get("selected"),
            "frozen_utc": config["tuning"].get("frozen_utc"),
        },
        "tasks": output_tasks,
    }
    if not leakage_ok:
        raise ValueError("Split leakage check failed")

    destination = PROJECT_ROOT / "data" / "task_stream_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (data_root / "task_stream_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Task stream manifest: {destination}")
    print(f"Leakage check: {'PASS' if leakage_ok else 'FAIL'}")
    for task in output_tasks:
        print(f"{task['task_id']}: {task['counts']} | train ID SHA-256={task['instance_id_sha256']['train']}")


if __name__ == "__main__":
    main()
