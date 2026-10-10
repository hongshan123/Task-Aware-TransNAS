# Environment Report — Task-Aware-TransNAS

**Date:** 2026-10-09  
**Status:** Windows CUDA environment, local-model inference, deterministic CITB preparation, one-task smoke run, and bounded four-task pilot completed.

## Machine and storage

| Check | Result |
|---|---|
| OS / architecture | Windows, 64-bit Python |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU memory | 24,564 MiB (about 24 GiB) |
| Driver | 610.60; nvidia-smi reported CUDA UMD 13.3 |
| Installed CUDA Toolkit | 12.1; PyTorch uses its bundled CUDA 13.2 runtime |
| Python | 3.11.2, project-local Conda prefix conda-env/ |
| D: free space after data and checkpoints | 180,783,091,712 bytes (about 168.4 GiB) |
| GPU after pilot process exited | 1,614 MiB reported in use by desktop/other processes |

The CUDA wheel recognized the RTX 4090 (compute capability 8.9). No administrator access or system CUDA changes were required. All experiment data, caches, logs, results, and checkpoints were directed to D:/Task-Aware-TransNAS. Project-root results and checkpoints are Windows directory junctions to those D: directories.

## Python packages

The Windows x64 environment passed pip check with no broken requirements. Installed main versions:

| Package | Version |
|---|---:|
| PyTorch | 2.14.1+cu132 |
| Transformers | 5.19.0 |
| Accelerate | 1.15.0 |
| Datasets | 5.1.0 |
| TRL | 1.15.0 |
| Evaluate | 0.4.6 |
| PEFT | 0.21.2 |
| PyYAML | 6.0.3 |
| python-dotenv | 1.2.4 |

Windows-compatible wheels were used. No Linux-only extensions such as FlashAttention were installed.

## Local model and CUDA validation

- .env sets MODEL_DIR=../model/qwen3-0.6b relative to the project root. Resolved path: C:\Users\Lenovo\Desktop\nas-cl-transformer\model\qwen3-0.6b.
- The local directory contains 11 model/tokenizer files, including config.json, tokenizer files, and model.safetensors. The model was only read; training checkpoints were written to D:.
- The environment smoke inference passed with local_files_only=True and offline flags. It generated 32 tokens in 0.97 seconds; peak CUDA allocation was 1.12 GiB.
- The continual runner also used local_files_only=True and HF offline flags; no model download was requested.

## CITB baseline validation

The official CITB data from the authors' [repository](https://github.com/hyintell/CITB) is stored outside the project checkout at D:/Task-Aware-TransNAS/data/CITB-official. The selected stream order1 subset covers intent identification, dialogue generation, and dialogue state tracking. There are 1,558 train, 200 dev, and 200 test examples. Test/dev IDs are disjoint from each other and train IDs; the exact IDs, checksums, source commit, counts, and seed note are in reports/data_manifest.json.

Assistant-only labels, EOS, padding, and split disjointness were validated. The single-task smoke completed 20 full-parameter optimizer updates. The four-task pilot completed 20 updates per task and saved a checkpoint after each task. All 596,049,920 parameters were trainable; peak CUDA allocation during training was 10.86 GiB. No OOM or automatic fallback occurred.

The corrected base scores are 0.42 task848 Accuracy, 0.48 task611 Accuracy, 0.0627 task1590 ROUGE-L, and 0.18 task1500 Accuracy. After the four-task sequence they are 0.38, 0.00, 0.0058, and 0.82. The old task611 Exact Match score was invalid because the model often returned a complete option while the reference was only a letter. See the [pilot evaluation audit](reports/pilot_evaluation_audit.md) and [pilot report](reports/baseline_pilot_report.md); the corrected matrix is at D:/Task-Aware-TransNAS/results/baseline_pilot/accuracy_matrix.csv.

The audit found left truncation removed the task definition for 64/500 task848 training examples and 160/500 task1500 training examples. Formal runs must preserve the complete instruction and truncate only input context. This short pilot is not a full benchmark.

## Next commands

In Windows PowerShell from the project directory:

1. Activate the environment: conda activate .\conda-env
2. Rebuild deterministic data splits if needed: python .\scripts\prepare_citb.py
3. Run a one-task smoke: python .\scripts\run_continual_baseline.py --mode smoke
4. Repeat the four-task pilot: python .\scripts\run_continual_baseline.py --mode pilot

The runner starts from the configured local base model each time. The project [README](README.md) and [baseline protocol](docs/continual_baseline.md) describe the setup.
