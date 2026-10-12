# Phase 7A：PaRSP 复现记录

## 复现对象与来源

本阶段以 ACL 2026 Main 论文 [PaRSP: Harmonizing the Past, Present, and Future](https://aclanthology.org/2026.acl-long.1244/) 及其[官方仓库](https://github.com/JinhuiBot/PaRSP)为方法来源。独立克隆位于 `/home/hongzhijun/projects/PaRSP-official`，固定 commit 为 `00685fe4f9086f03ac1d1a725dcb280714e6125a`。复现结束时官方仓库工作树干净；所有 CITB/Qwen3 适配都在本项目 Phase 7A 文件中完成，没有修改上游实现。

实验使用现有 `transnas` Conda 环境、Qwen3-0.6B（596,049,920 个参数）及两张 RTX PRO 6000 Blackwell GPU。运行配置见 [phase7a.yaml](../configs/phase7a.yaml)。训练入口是 [run_phase7a_pilot.py](../scripts/run_phase7a_pilot.py)，上游 attribution、mask 和 null-space hook 的调用与结构适配位于 [phase7a_parsp.py](../scripts/phase7a_parsp.py) 与 [phase7a_engine.py](../scripts/phase7a_engine.py)。

## 固定实验协议

- 随机种子为 10；S1 顺序为 `565 → 1603 → 1501 → 1384`，S2 顺序为 `565 → 1501 → 1603 → 1384`。
- 每个任务训练 50 个 optimizer steps，沿用现有 CITB 样本 ID、prompt/target 编码、学习率 `1e-5`、梯度累积 4、AdamW 与 weight decay `0.01`。所有任务有效最大序列长度均为 512。
- 任务归因每个新任务抽取 256 条 train 样本；协方差校准每个已学任务抽取 256 条 train 样本。历史样本只用于无梯度前向校准，不参与 replay 或优化。
- C0 为 Full SFT；C1 调用官方 PaRSP mask 与 projection；C2 仅 mask；C3 仅 projection；C4 是在 C1 的实际更新审计触发后才运行的严格约束对照。
- 先完成全部条件的训练，再统一加载 dev 做基础模型和所有任务边界 checkpoint 的评价。dev 评价每个任务 50 条记录，分类与生成使用项目既有评价函数。每个序列的 `evaluation_summary.json` 记载只加载 `dev`；`test_split_files_opened` 为空，训练及离线风险分析也都没有打开 test 文件。

Qwen3 使用 16 个 query heads 和 8 个 KV heads。GQA 下每个 KV group 服务两个 query heads，因此适配器把选中的 query channel 映射到共享 KV group，再用于 V 输出通道；O 投影仍按 query channel 施加输入列 mask。官方 Qwen 实现中选中的 attention layer 解冻 Q/K/V/O，但稀疏通道 mask 只落在 V 行与 O 输入列，Q/K 仍遵循官方的层级冻结策略。FFN mask 作用于 gate/up 输出行及 down 输入列。该映射在真实 Qwen3 形状和线性层单测中核对。

C1–C4 都保留官方 attribution 选层与层级冻结。这里 C3 的“仅 projection”指所选层不使用层内 neuron/channel mask，并不代表全模型所有层都施加 projection。消融的归因边界见 pilot 报告的“获取与保留的取舍”。

## 与论文设置的边界

这是指定项目协议下的方法复现，不是论文表格复刻：模型为 Qwen3-0.6B、CITB 四任务、单 seed、每任务 50 步；论文主结果使用更大模型、更多基准和多 seed。因此本报告只比较本项目 C0–C4，不把本地分数与论文数值横向比较。当前设置的 layer ratio 为 0.38、层内 neuron/channel ratio 为 0.40；smoke 观察到的可训练比例约 29%，正式 C1/C2/C3 实际变化比例约 29.03%。

## Pilot 完成状态与产物

S1、S2 的 C0–C4 全部完成，每个条件每条序列含 4 个任务、共 200 optimizer steps。先运行了单任务 5 步和双任务每任务 1 步 smoke，并验证 checkpoint 可重载。条件训练摘要与任务边界评估分别保存在 `results/phase7a/S{1,2}/{C0…C4}/`。

正式 pilot 的最终指标、资源数据与研究结论见 [phase7a_pilot_results.md](phase7a_pilot_results.md)。逐阶段指标在 `results/phase7a/pilot_stage_summary.csv`，最终逐任务指标（ROUGE-L、Token-F1、Accuracy、Macro-F1、Yes/No 错误率、teacher-forced NLL）在 `results/phase7a/pilot_final_task_metrics.csv`；AAA 曲线见 [pilot_aaa_trajectory.png](../figures/phase7a/pilot_aaa_trajectory.png)。功能风险离线数据与图表见 [功能重要性分析报告](phase7a_functional_importance_vs_preservation.md)。

此前 Phase 4–6B 的结果与图表均保留，Phase 7A 只写入 `results/phase7a/` 和 `figures/phase7a/`。检查没有找到 Phase 4–6A 正式原始结果或边界 checkpoint（`artifacts/checkpoints/` 只有 `.gitkeep`）；本阶段未伪造或补写这些历史数据。
