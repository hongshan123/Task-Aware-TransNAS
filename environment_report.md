# Environment Report — Task-Aware-TransNAS

**Date:** 2026-10-09  
**Status:** Ready for local CUDA inference; formal training has not been started.

## Machine and storage

| Check | Result |
|---|---|
| OS / architecture | Windows, win32, 64-bit Python |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU memory | 24,564 MiB reported (23.99 GiB) |
| NVIDIA driver | 610.60; `nvidia-smi` reports CUDA UMD 13.3 |
| Installed CUDA Toolkit | 12.1 (`nvcc`) |
| PyTorch CUDA runtime | 13.2, bundled with the PyTorch wheel |
| C: free space after setup | 41,591,533,568 bytes (about 38.74 GiB) |
| D: free space after setup | 201,848,094,720 bytes (about 187.99 GiB) |

PyTorch reports CUDA available and identifies the RTX 4090 with compute capability 8.9. The locally installed CUDA Toolkit is not used by this PyTorch wheel; no system-level CUDA changes or administrator access were needed.

## Local model

- Project setting: `.env` sets `MODEL_DIR=../model/qwen3-0.6b` relative to the project root.
- Resolved model path: `C:\Users\Lenovo\Desktop\nas-cl-transformer\model\qwen3-0.6b`.
- Model files: 11 files, 1,519,209,873 bytes (about 1.415 GiB).
- Required files present: `config.json`, `tokenizer.json`, and `model.safetensors`.
- The shell environment did not expose a `MODEL_DIR` value during inspection; the project-local `.env` now supplies the path.

## Python environment and packages

- Environment: project-local Conda prefix at `conda-env/`.
- Python: 3.11.2, 64-bit; pip: 26.2.1.
- Main packages:

| Package | Installed version |
|---|---:|
| PyTorch | 2.14.1+cu132 |
| Transformers | 5.19.0 |
| Accelerate | 1.15.0 |
| Datasets | 5.1.0 |
| TRL | 1.15.0 |
| Evaluate | 0.4.6 |
| PEFT | 0.21.2 |
| Safetensors | 0.8.0 |
| NumPy | 2.4.6 |
| Pandas | 3.0.6 |
| scikit-learn | 1.9.1 |

`pip check` passed with **no broken requirements**. The environment uses Windows wheels. Linux-only extensions such as FlashAttention were not installed.

The installed project package imports successfully; `SearchSpace.validate()` and a basic replay-buffer insertion also passed.

Conda's configured default channel returned TLS connection errors. To keep setup moving without changing system configuration, Python 3.11.2 was linked from the machine's existing Conda package cache in offline mode; pip package installation then succeeded through the configured package index. Python 3.11.2 meets the requested minor version, but it is an older patch release. Once the Conda channel is reachable, consider rebuilding with a current Python 3.11 patch release and reinstalling from `requirements.txt`.

## Offline CUDA inference

The smoke script set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and both `from_pretrained` calls use `local_files_only=True` and `trust_remote_code=False`.

- Result: **PASS** (`SMOKE_TEST=PASS`).
- Model: local Qwen3-0.6B; no model download was attempted.
- Generation: 32 tokens in 0.97 seconds.
- PyTorch peak allocated memory during generation: 1.12 GiB.
- Free GPU memory after generation while the model remained loaded: 20.86 GiB.
- After the smoke process exited, `nvidia-smi` reported 1,556 MiB in use by the desktop and other processes.

## Project status

The project structure, `pyproject.toml` metadata, editable package install (`task-aware-transnas==0.1.0`), local model configuration, replay-buffer scaffold, NAS search-space scaffold, smoke script, and environment checker are in place. The project Git repository is initialized on `main`. No dataset preparation, NAS search, continual-learning experiment, or formal training has run.

An incomplete, unused `.venv/` directory remains from an earlier bootstrap attempt. It is Git-ignored and the working environment is `conda-env/`. Automatic policy review rejected the recursive deletion of that generated directory, so it was left in place.

## Next commands

From PowerShell or Command Prompt:

```powershell
cd C:\Users\Lenovo\Desktop\nas-cl-transformer\Task-Aware-TransNAS
conda activate .\conda-env
python -m pip install -e . --no-deps
python .\scripts\check_environment.py
python .\scripts\smoke_inference.py
```

The resolved package lock is in `requirements.txt`. To recreate the pip layer inside a matching Python 3.11 environment, run `python -m pip install -r requirements.txt`, then install the local source package with `python -m pip install -e . --no-deps`.

When ready for experiments, first add versioned dataset manifests and task-specific configuration, then implement the training/search entry points; no formal training command is provided yet.

## References

- [PyTorch Windows/CUDA wheel installation matrix](https://pytorch.org/get-started/previous-versions/)
- [Transformers installation and local model loading](https://huggingface.co/docs/transformers/installation)
