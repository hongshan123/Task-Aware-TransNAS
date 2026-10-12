# Task-Aware-TransNAS

Research workspace for task-aware Transformer neural architecture search (NAS) and language-model continual learning (CL). Windows remains supported; the Linux setup below targets the RTX PRO 6000 Blackwell workstation.

## Linux setup (RTX PRO 6000 Blackwell)

Create the Python 3.11 environment, install the Linux CUDA 13.2 PyTorch build and project dependencies, then install the repository package:

```bash
conda env create -f environment-linux.yml
conda run -n transnas python -m pip install -r requirements-linux.txt
conda run -n transnas python -m pip install -e . --no-deps
```

Download the public Qwen3-0.6B files, verify the weight checksum, then copy the offline environment template and confirm `MODEL_DIR` points to `model/qwen3-0.6b`. The model is loaded locally by the verification scripts:

```bash
conda run -n transnas hf download Qwen/Qwen3-0.6B --local-dir model/qwen3-0.6b
sha256sum model/qwen3-0.6b/model.safetensors
cp .env.linux.example .env
conda run -n transnas python scripts/smoke_inference.py
```

Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` in `.env` after the download. Prepare the deterministic four-task dataset and run the isolated, train-only GPU smoke checks with:

```bash
conda run -n transnas python scripts/prepare_citb.py --config configs/continual_baseline_full_linux.yaml
conda run -n transnas python scripts/run_training_smoke.py --steps 10 --output-dir runs/smoke/manual
bash scripts/run_dual_gpu_experiments.sh
```

The dual-GPU launcher runs two independent one-GPU jobs, each capped at ten optimizer steps, and writes separate logs, metrics, and save/restore checkpoints below `runs/parallel/`. It does not start the formal baseline or NAS search. The root `requirements.txt` remains the historical Windows lock; use `requirements-linux.txt` on Linux.

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

## Phase 7A：PaRSP 复现与功能保护分析

Phase 7A 已在 Qwen3-0.6B、seed 10、两个 CITB 顺序上完成 C0–C4 pilot，每任务 50 optimizer steps。评价只使用 dev。安装 Linux 环境与准备数据后，在项目旁边保留指定版本的官方代码：

```bash
git clone https://github.com/JinhuiBot/PaRSP.git ../PaRSP-official
git -C ../PaRSP-official checkout 00685fe4f9086f03ac1d1a725dcb280714e6125a
conda run -n transnas python scripts/run_phase7a_pilot.py --mode smoke --sequence S1 --device cuda:0 --smoke-steps 5
```

S1/S2 可在两个终端分别运行，输出目录互相独立。C4 只在 C1 实际更新审计触发后启用：

```bash
conda run -n transnas python scripts/run_phase7a_pilot.py --mode pilot --sequence S1 --device cuda:0
conda run -n transnas python scripts/run_phase7a_pilot.py --mode pilot --sequence S2 --device cuda:1
conda run -n transnas python scripts/analyze_phase7a.py --sequence both
conda run -n transnas python scripts/plot_phase7a_results.py
```

复现说明、真实更新审计、风险/随机回滚对照和获取–保留取舍分别见 [复现报告](reports/phase7a_parsp_reproduction.md)、[约束审计](reports/phase7a_constraint_audit.md)、[功能风险报告](reports/phase7a_functional_importance_vs_preservation.md)及 [Pilot 结果](reports/phase7a_pilot_results.md)。C1 在最终平均分上优于 Full SFT，但 task1603 的新任务获取不足；C4 的严格保护也明显压缩更新空间。结果仍是单 seed 的机制诊断，未启动 NAS。

Git 跟踪 `results/phase7a/` 下的紧凑指标、评价摘要与访问/恢复记录。完整组件测量、逐步审计、模型和边界 checkpoint 保留本地；离线风险分析需要这些本地 checkpoint。
