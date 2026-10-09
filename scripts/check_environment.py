"""Print a concise local environment and model-file diagnostic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> int:
    model_value = os.environ.get("MODEL_DIR", "../model/qwen3-0.6b")
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"MODEL_DIR: {model_path}")
    if not model_path.is_dir():
        print("MODEL_DIR_EXISTS=False")
        return 1
    files = list(model_path.iterdir())
    total_bytes = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
    print(f"Model top-level entries: {len(files)}")
    print(f"Model size: {total_bytes / 2**30:.3f} GiB")
    print(f"config.json: {(model_path / 'config.json').is_file()}")
    print(f"tokenizer.json: {(model_path / 'tokenizer.json').is_file()}")
    print(f"model.safetensors: {(model_path / 'model.safetensors').is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
