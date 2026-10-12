# Phase 7A 实施计划与完成记录

## 目标

在现有 Qwen3-0.6B CITB 任务流复现官方 PaRSP mask/projection 语义，审计 optimizer 后的真实参数更新，并检验当前任务功能重要性与旧任务保留风险之间的关系。

## 全局约束

- 官方 PaRSP 固定在 commit `00685fe4f9086f03ac1d1a725dcb280714e6125a`，原仓库保持干净。
- 使用 seed 10、S1/S2 CITB 四任务顺序及每任务 50 步。
- 训练期只读 train；全部训练结束后再读取 dev；不打开 test。
- 不做历史 replay，不启动 NAS，不覆盖 Phase 4–6B 结果，不提交已有迁移更改。
- 先完成单/双任务 smoke，再执行完整 pilot；C4 仅在 C1 审计发现违规后运行。

## 已完成工作

- [x] 克隆并记录官方仓库和 commit；检查官方工作树干净。
- [x] 阅读论文、官方 localization/masking/projection 实现与训练入口。
- [x] 适配 Qwen3 GQA 的 Q/K/V/O、FFN gate/up/down 结构；保留 C1 官方 hook 与调用语义。
- [x] 为 mask/project 顺序、真实 AdamW 后更新、GQA 映射、严格交集、空秩和数据访问编写单元测试。
- [x] 在共享训练循环增加 opt-in optimizer step 前/后 callback，验证真实参数差审计。
- [x] 加入 train-only 校准、stage checkpoints、C0–C4 runner、资源摘要和 dev-only 边界评价。
- [x] 单任务 5 步与双任务 smoke 成功；checkpoint 可重载。
- [x] S1/S2 C0–C3 正式 pilot 完成；两条 C1 审计均触发 C4。
- [x] 修复严格约束审计的零更新除零问题；S1/S2 C4 均完成，违规残差低于阈值。
- [x] 完成离线 functional importance/risk 分析、分组 Spearman/Kendall、精确组件回滚和匹配随机对照。
- [x] 生成 `pilot_stage_summary.csv`、`pilot_final_task_metrics.csv` 和 AAA/importance-risk 图表。
- [x] 完成四份 Phase 7A 报告；明确记录缺失的 Phase 4–6A 原始结果/检查点。
- [x] 最终完整 pytest、文件/manifest/test-access 核对与结果路径复核（69 passed；正式运行与分析均未打开 test）。

## 主要文件

- 适配和严格审计：[phase7a_parsp.py](../scripts/phase7a_parsp.py)、[phase7a_engine.py](../scripts/phase7a_engine.py)、[run_phase7a_pilot.py](../scripts/run_phase7a_pilot.py)
- 离线风险分析：[analyze_phase7a.py](../scripts/analyze_phase7a.py)
- 指标汇总和图：[plot_phase7a_results.py](../scripts/plot_phase7a_results.py)
- 配置：[phase7a.yaml](../configs/phase7a.yaml)
- 报告：[复现](phase7a_parsp_reproduction.md)、[约束审计](phase7a_constraint_audit.md)、[功能风险](phase7a_functional_importance_vs_preservation.md)、[Pilot](phase7a_pilot_results.md)

## 解释范围

试验只有一个种子、两条指定顺序和每任务 50 步。误差条与跨 seed 显著性尚未估计；功能风险和 generation quality 的组件级关系仍需独立复验。因此目前可据此决定是否开展小规模风险感知参数分配研究，不能据此宣称已经得到通用持续学习方法或直接进入架构搜索。
