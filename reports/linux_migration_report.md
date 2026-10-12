# Task-Aware-TransNAS Linux 迁移报告

执行日期：2026-10-11  
项目目录：`/home/hongzhijun/projects/Task-Aware-TransNAS`  
迁移范围：环境、代码路径、数据清单、本地模型、双卡冒烟；未启动正式训练或 NAS。

| 步骤 | 状态 | 核验结果 |
|---|---|---|
| 1. Linux、GPU、驱动、网络和磁盘检查 | PASS | Ubuntu 24.04.3、x86_64；两张 RTX PRO 6000 Blackwell Server Edition，各 97,887 MiB；Driver 595.99.02；GitHub/Hugging Face 可访问；最终剩余磁盘约 247 GiB。未使用 sudo，未修改系统驱动。 |
| 2. 仓库与 CITB 迁移 | PASS | 仓库 `main` commit 为 `5f756d834db83c8df1f9384285649de25cdb3314`；官方 CITB checkout 为 `bf50533b5bced4c388691ecc75e26773da96b3fd`，工作树干净。 |
| 3. Python/CUDA 环境 | PASS | Conda 环境 `transnas` 使用 Python 3.11.16；PyTorch 2.14.1+cu132 / CUDA runtime 13.2；Transformers 5.19.0、Accelerate 1.15.0、Datasets 5.1.0、TRL 1.15.0、PEFT 0.21.2、Evaluate 0.4.6。`pip check` 无依赖冲突；双卡 CUDA 可用且 BF16 可用。 |
| 4. Qwen3-0.6B 下载与离线加载 | PASS | 官方 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 文件位于 `model/qwen3-0.6b`；`model.safetensors` 为 1,503,300,328 bytes，SHA256 与要求一致：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。config、tokenizer、generation config 和权重文件齐全；`.env` 已启用 Hub、Transformers、Datasets 离线模式。 |
| 5. Linux 路径与运行配置 | PASS | 增加 `TRANSNAS_DATA_ROOT` / `TRANSNAS_STORAGE_ROOT` 路径解析、Phase 4B–6B 统一产物路径和独立 Linux 配置；保留 Windows 原配置，并把旧 `D:/Task-Aware-TransNAS/...` 路径在 Linux 上映射到本地数据/存储根。Linux 数据和历史清单写入独立文件，没有覆盖原始 Windows 清单。 |
| 6. CITB 数据集复现 | PASS | 四项任务各为 train/dev/test = 500/50/50；冻结任务顺序及每个 split 的有序样本 ID 与 Windows manifest 完全一致，泄漏检查通过；训练冒烟只打开 train 文件，不读取 dev/test。原 Windows source JSON 的 SHA256 对应 CRLF 字节，Linux checkout 为 LF；逐文件转换换行后四项源文件 SHA256 均匹配。 |
| 7. 双卡推理和独立训练冒烟 | PASS | GPU0、GPU1 均以本地离线模型生成 32 个 token；两项独立单卡训练各完成 10 个 optimizer steps，峰值显存各 11.19 GiB。 |
| 8. Phase 4B–6B 历史产物迁移 | FAIL（等待旧机文件） | 已检查 Git 文件和本机目录并统一脚本路径，但 Windows 上未提交的历史 raw runs/checkpoints 不在当前 Linux checkout。Git 仅包含 `results/phase6b/` 下 7 个历史汇总文件，Phase 4B、5、5B、6A 原始目录及 Phase 6B 每次运行的 checkpoint 均缺失；不能据此声称历史实验已复现。需从旧机按下文目录补拷。 |
| 9. pytest、训练保存/恢复及代码验证 | PASS | `pytest -q`：47 passed；20 个 Phase 4B–6B 模块导入成功；Python 编译、双卡启动脚本语法和 `git diff --check` 均通过。两张 GPU 的 checkpoint 恢复 logits 最大绝对差为 0，optimizer step 恢复为 10，CPU/CUDA RNG 状态恢复通过。 |
| 10. 限制遵守情况 | PASS | 未使用 sudo、未修改驱动；未启动正式 baseline、Phase 4B–6B 正式实验或 NAS；仅运行每卡 10 步的短训练冒烟。 |

## 环境与数据证据

Conda 环境可通过 `conda activate transnas` 使用。当前环境核验值为 Python `3.11.16`、PyTorch `2.14.1+cu132`、PyTorch CUDA runtime `13.2`、GPU compute capability `(12, 0)`，两卡均支持 BF16。系统未安装 `nvcc`；本次 CUDA 运算使用 PyTorch wheel 自带的 CUDA runtime，实际推理和训练均已在两卡上执行。

CITB 使用固定 commit `bf50533b5bced4c388691ecc75e26773da96b3fd`。重建后的 Linux 数据清单位于 `data/processed/citb_formal_4tasks/data_manifest.json`，独立迁移清单位于 `data/linux_data_manifest.json`、`data/task_stream_manifest_linux.json` 和 `reports/linux_data_manifest.json`。原 `data_manifest.json` 与 `data/task_stream_manifest.json` 保持未修改。

| 顺序 | 任务 | 官方流位置 | train ID SHA256 |
|---:|---|---:|---|
| 1 | `task565_circa_answer_generation` | 3 | `fb066bdaaa8c2238d08505b8657d5e6878b56abe7e581675167d477b5c620baa` |
| 2 | `task1603_smcalflow_sentence_generation` | 17 | `845c757020f694687e32cb1f952957493517e19c38d2beeacc971ad94eb35752` |
| 3 | `task1501_dstc3_answer_generation` | 14 | `0ac6a059b17aa34119d01ead351eacc765e894561509b4e92c49eebc4f439f33` |
| 4 | `task1384_deal_or_no_dialog_classification` | 18 | `6367fb6efa17f7504d2d1d644deb5e785c63a002d3be12c1675721cf02ba8bd4` |

样本 ID 的 train/dev/test 次序均与冻结 manifest 相同；新旧 source JSON 原始字节哈希因 Windows CRLF / Linux LF 不同，但规范化换行后的哈希相同。详细逐 split ID 哈希和核验结果保存在 Linux 清单中。

## 双卡冒烟产物

推理日志：

- GPU0：`runs/inference/gpu0/inference.log`
- GPU1：`runs/inference/gpu1/inference.log`

成功的独立并行运行目录为 `runs/parallel/dual_gpu_smoke_20261011T053525Z_306303/`。每张卡分别保留 `logs/training.log`、`results/smoke_metrics.json`、`results/checkpoint/model/model.safetensors`、`results/checkpoint/optimizer.pt` 和 RNG state。每个模型 checkpoint 为 2,384,234,968 bytes，optimizer checkpoint 为 4,768,661,459 bytes。指标记录明确标记 `evaluation_splits_loaded: []`。

## 缺少的 Windows 历史产物

目前可见的历史文件只有 Git 中的 `results/phase6b/compatibility_manifest.json`、`compatibility_matrices.csv`、`gpu_smoke.json`、`pilot_resource_summary.csv`、`pilot_summary.json`、`pilot_task_metrics.csv`、`task_similarity_gradient_matrix.csv`。这些汇总表不足以继续 Phase 5/6 的回放或审计。

请把旧机 `D:\Task-Aware-TransNAS` 下以下目录复制到本机仓库的 `runs/` 下对应位置；当前 Phase 4B–6B 脚本已统一从 `TRANSNAS_STORAGE_ROOT` 读取这些路径：

```text
D:\Task-Aware-TransNAS\checkpoints\baseline_full\homogeneous\
  -> runs/checkpoints/baseline_full/homogeneous/

D:\Task-Aware-TransNAS\results\phase4b\
  -> runs/results/phase4b/
D:\Task-Aware-TransNAS\results\phase5\
  -> runs/results/phase5/
D:\Task-Aware-TransNAS\results\phase5b\
  -> runs/results/phase5b/
D:\Task-Aware-TransNAS\results\phase6a\
  -> runs/results/phase6a/
D:\Task-Aware-TransNAS\results\phase6b\
  -> runs/results/phase6b/
```

关键依赖包括 Phase 4B 的 `runs/A0/S1/seed_10/metrics.json` 与 `checkpoints/A0/S1/seed_10/model.safetensors`（Phase 5B 和 Phase 6A 使用），Phase 4B 的 S1/S2 A0 seed-10 metrics（Phase 6B baseline 对照使用），Phase 5 的 `fv/head_effects/`、`fv/activations/`，Phase 5B 的 `boundary_replay_manifest.json` 与 `checkpoints/historical_boundary_replay/A0/S1/seed_10/after_2_task1603_smcalflow_sentence_generation/`，Phase 6A 的 after-task1501 checkpoint、screen selection 和 replay manifest，以及 Phase 6B `runs/`、`checkpoints/` 下的逐次运行记录。建议复制整个阶段目录，保留其中审计文件、raw predictions、metrics 和 checkpoint sidecar。

## 常用命令

```bash
conda activate transnas

# 重建数据并与冻结 Windows manifest 比对
python scripts/prepare_citb.py --config configs/continual_baseline_full_linux.yaml
python scripts/verify_linux_data.py --config configs/continual_baseline_full_linux.yaml

# 本地离线推理，分别选择一张卡
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_inference.py
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_inference.py

# 两张卡各自启动一个独立的最多 10 步训练冒烟，日志和结果分别保存
CONDA_ENV=transnas bash scripts/run_dual_gpu_experiments.sh

# 重新执行测试
python -m pytest -q
```

正式 Phase 4B runner 的 `--preflight-only` 只做资源预检；实际 runner 会启动正式实验。本次按要求没有启动。补齐历史文件并确认要继续研究实验后，可先运行 `python scripts/audit_phase4b_architectures.py` 和 `python scripts/run_phase4b_comparison.py --preflight-only`，再按项目协议单独安排正式实验。
