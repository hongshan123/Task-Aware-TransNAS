# Task-Aware-TransNAS

Windows research workspace for task-aware Transformer neural architecture search (NAS) and language-model continual learning (CL).

## Local setup

The project uses a project-local Conda environment at `conda-env/` (Python 3.11). Activate it from PowerShell or Command Prompt with:

```powershell
conda activate .\conda-env
python -m pip install -e . --no-deps
```

The local Qwen3-0.6B checkpoint is configured in `.env`; model loading is offline-only. Run the GPU smoke inference with:

```powershell
python .\scripts\smoke_inference.py
```

No training is started by setup or by the smoke script.

## Layout

- `src/task_aware_transnas/cl/`: continual-learning components, including replay memory.
- `src/task_aware_transnas/nas/`: Transformer candidate search-space definitions.
- `configs/`: experiment configuration.
- `data/`: raw, processed, and replay data (kept out of version control).
- `artifacts/`: checkpoints and NAS results (kept out of version control).
- `scripts/`: environment and local inference checks.
- `environment_report.md`: machine and verification report.
- `requirements.txt`: resolved package versions for this environment.

## Research status

Environment preparation and a single local inference smoke test only. Dataset preparation, NAS search, continual-learning experiments, and formal training have not been run.
