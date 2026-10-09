"""Offline-only local model load and short CUDA generation check."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# Explicitly prohibit Hub access during this check.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def main() -> int:
    model_value = os.environ.get("MODEL_DIR", "../model/qwen3-0.6b")
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"No config.json found in local model directory: {model_path}")

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model path: {model_path}")
    print("Offline mode: enabled (local_files_only=True)")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = "In one sentence, explain continual learning for language models."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    print(f"Generated text: {generated}")
    print(f"Generated tokens: {output.shape[1] - inputs['input_ids'].shape[1]}")
    print(f"Generation time: {elapsed:.2f} s")
    print(f"GPU memory total: {total_bytes / 2**30:.2f} GiB")
    print(f"GPU memory free after generation: {free_bytes / 2**30:.2f} GiB")
    print(f"GPU memory allocated: {allocated / 2**30:.2f} GiB")
    print(f"GPU peak allocated during generation: {peak / 2**30:.2f} GiB")
    if not generated:
        raise RuntimeError("The model loaded but generated an empty response")
    print("SMOKE_TEST=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
