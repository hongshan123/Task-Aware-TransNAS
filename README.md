# Task-Aware-TransNAS

Windows research workspace for task-aware Transformer neural architecture search (NAS) and language-model continual learning (CL).

## Local setup

The project uses a project-local Conda environment at `conda-env/` (Python 3.11). Activate it from PowerShell or Command Prompt with:

```powershell
conda activate .\conda-env
python -m pip install -e . --no-deps
```

The local Qwen3-0.6B checkpoint is configured in `.env`; model loading is offline-only. Run the original GPU inference check with:

```powershell
python .\scripts\smoke_inference.py
```

The bounded full-parameter continual instruction-tuning pilot uses the official CITB task stream. Dataset, caches, checkpoints, predictions, and metrics are written to `D:\Task-Aware-TransNAS`. See [docs/continual_baseline.md](docs/continual_baseline.md) for the split policy and commands. Run:

```powershell
python .\scripts\prepare_citb.py
python .\scripts\run_continual_baseline.py --mode smoke
python .\scripts\run_continual_baseline.py --mode pilot
```

The smoke run fine-tunes one task for 20 optimizer steps. The pilot runs four tasks in sequence for 20 steps each; it is an initial baseline, not a full benchmark.

## Formal continual SFT baseline

The formal run uses four CITB tasks balanced at 500 train / 50 dev / 50 test examples each. Task categories and output formats are balanced, with three repeatable orders built from frozen prompt-representation and gradient-similarity measurements. Development data selected the shared learning rate and optimizer-step count; the test split was opened only after those choices were frozen. The runner updates all model parameters and preserves complete task definitions when long dialogue inputs need truncation.

The selected configuration is stored in `configs/continual_baseline_full.yaml`. Validate the train/dev encodings, rerun a dev-only hyperparameter search when needed, reproduce each stream, and rebuild the aggregate report with:

```powershell
python .\scripts\prepare_citb.py --config .\configs\continual_baseline_full.yaml
python .\scripts\analyze_citb_task_relations.py
python .\scripts\run_formal_baseline.py --phase validate
python .\scripts\run_formal_baseline.py --phase tune --reset-tuning
python .\scripts\build_task_stream_manifest.py
python .\scripts\run_formal_baseline.py --phase run --sequence homogeneous
python .\scripts\run_formal_baseline.py --phase run --sequence heterogeneous
python .\scripts\run_formal_baseline.py --phase run --sequence mixed
python .\scripts\summarize_formal_runs.py
```

The frozen selection and split IDs are in [data/task_stream_manifest.json](data/task_stream_manifest.json). Final metrics and limitations are in [reports/baseline_full_analysis.md](reports/baseline_full_analysis.md); raw predictions, metrics, tuning candidates, and checkpoints are stored on `D:\Task-Aware-TransNAS`.

## Layout

- `src/task_aware_transnas/cl/`: continual-learning components, including replay memory.
- `src/task_aware_transnas/nas/`: Transformer candidate search-space definitions.
- `configs/`: experiment configuration.
- `data/`: raw, processed, and replay data (kept out of version control).
- `artifacts/`: checkpoints and NAS results (kept out of version control).
- `scripts/`: environment checks, CITB preparation, and local training/evaluation.
- `environment_report.md`: machine and verification report.
- `requirements.txt`: resolved package versions for this environment.

## Research status

The CITB manifests record source commit, task order, checksums, sample IDs, and split leakage checks. The pilot evaluation was corrected and the formal full-parameter continual SFT baseline has been completed. Transformer NAS search remains a separate, not-yet-run experiment.
