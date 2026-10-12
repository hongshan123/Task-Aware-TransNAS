"""Run a bounded, train-only CUDA fine-tuning and checkpoint round-trip check."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv  # noqa: E402
except ModuleNotFoundError:  # Pure helper tests can run before requirements are installed.
    def load_dotenv(*_args, **_kwargs):
        return False

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "citb_formal_4tasks" / "data_manifest.json"


def load_train_records_from_manifest(manifest_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Load only the first task's train file; dev and test paths are never opened."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leakage = manifest.get("leakage_check", {})
    if leakage is True:
        passed = True
    else:
        passed = isinstance(leakage, dict) and leakage.get("passed") is True
    if not passed:
        raise ValueError("Training smoke requires a passing manifest leakage_check")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest has no tasks")
    task = tasks[0]
    task_id = str(task.get("task_id", ""))
    train_path = Path(task.get("split_files", {}).get("train", ""))
    if not train_path.is_absolute():
        train_path = (manifest_path.parent / train_path).resolve()
    if not task_id or not train_path.is_file():
        raise FileNotFoundError(f"First task train split is unavailable: {train_path}")
    records: list[dict[str, Any]] = []
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-training record at {train_path}:{line_number}")
            records.append(row)
    if not records:
        raise ValueError(f"First task training split is empty: {train_path}")
    return task_id, records


def validate_step_count(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10:
        raise ValueError("Training smoke accepts 1 to at most 10 optimizer steps")
    return steps


def _training_tensors(tokenizer: Any, record: dict[str, Any], max_length: int, device: torch.device):
    import torch

    prompt = "\n".join(
        part for part in (str(record.get("definition", "")).strip(), str(record.get("input", "")).strip()) if part
    )
    targets = record.get("targets") or [""]
    answer = str(targets[0]).strip()
    if not answer:
        raise ValueError(f"Training record {record.get('instance_id')} has no target text")
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids[: max_length - 2] + [tokenizer.eos_token_id]
    else:
        answer_ids = answer_ids[: max_length - 2]
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max(1, max_length - len(answer_ids)),
    )["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    if not answer_ids or len(input_ids) <= len(prompt_ids):
        raise ValueError("Tokenization produced no supervised target tokens")
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids) :]
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
    }


def _logits(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    import torch

    model.eval()
    with torch.inference_mode():
        result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return result.logits[:, -1, :].detach().float().cpu()


def run_smoke(
    model_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    steps: int,
    seed: int,
    max_length: int = 128,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    steps = validate_step_count(steps)
    if max_length < 16:
        raise ValueError("max_length must be at least 16")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty smoke output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id, records = load_train_records_from_manifest(manifest_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing local model config: {model_dir / 'config.json'}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    batch_cache = [
        _training_tensors(tokenizer, records[index % len(records)], max_length, device)
        for index in range(steps)
    ]

    started = time.perf_counter()
    losses: list[float] = []
    for index, batch in enumerate(batch_cache):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {index + 1}: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    expected_logits = _logits(model, batch_cache[-1])
    checkpoint_dir = output_dir / "checkpoint"
    model_dir_out = checkpoint_dir / "model"
    model.save_pretrained(model_dir_out, safe_serialization=True)
    tokenizer.save_pretrained(model_dir_out)
    torch.save({"optimizer": optimizer.state_dict(), "global_step": steps}, checkpoint_dir / "optimizer.pt")
    rng_state = {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device)}
    torch.save(rng_state, checkpoint_dir / "rng_state.pt")

    del model, optimizer
    torch.cuda.empty_cache()
    restored = AutoModelForCausalLM.from_pretrained(
        model_dir_out,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-6, weight_decay=0.0)
    optimizer_payload = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=True)
    restored_optimizer.load_state_dict(optimizer_payload["optimizer"])
    if int(optimizer_payload["global_step"]) != steps:
        raise RuntimeError("Restored optimizer step count does not match the requested smoke steps")
    restored_batch = batch_cache[-1]
    actual_logits = _logits(restored, restored_batch)
    max_logit_diff = float((expected_logits - actual_logits).abs().max())
    del restored_optimizer, restored
    torch.cuda.empty_cache()
    if max_logit_diff > 1e-6:
        raise RuntimeError(f"Saved/restored model logits differ: max_abs_diff={max_logit_diff}")

    rng_payload = torch.load(checkpoint_dir / "rng_state.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng_payload["torch_rng"])
    torch.cuda.set_rng_state(rng_payload["cuda_rng"], device=device)
    expected_cpu_random = torch.rand(4)
    expected_cuda_random = torch.rand(4, device=device).cpu()
    torch.set_rng_state(rng_payload["torch_rng"])
    torch.cuda.set_rng_state(rng_payload["cuda_rng"], device=device)
    rng_roundtrip_passed = torch.equal(expected_cpu_random, torch.rand(4)) and torch.equal(
        expected_cuda_random, torch.rand(4, device=device).cpu()
    )
    if not rng_roundtrip_passed:
        raise RuntimeError("Saved/restored random generator states differ")

    report = {
        "status": "PASS",
        "task_id": task_id,
        "train_records_available": len(records),
        "evaluation_splits_loaded": [],
        "optimizer_steps": steps,
        "seed": seed,
        "model_dir": str(model_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "losses": losses,
        "elapsed_seconds": elapsed,
        "checkpoint_restore": {
            "optimizer_global_step": int(optimizer_payload["global_step"]),
            "max_abs_logit_diff": max_logit_diff,
            "rng_state_roundtrip": rng_roundtrip_passed,
            "passed": True,
        },
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    (output_dir / "smoke_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "model/qwen3-0.6b")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    model_dir = args.model_dir if args.model_dir.is_absolute() else PROJECT_ROOT / args.model_dir
    manifest = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    report = run_smoke(model_dir.resolve(), manifest.resolve(), output_dir.resolve(), args.steps, args.seed, args.max_length)
    print(json.dumps(report, indent=2))
    print("TRAINING_SMOKE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
