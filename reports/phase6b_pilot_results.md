# Phase 6B Seed-10 Pilot Results

## Scope and protocol

The paired pilot completed both task orders (S1 and S2), seed 10, on 50 locked dev examples per task. B0 reuses the matching Phase 4B A0 run; all adapter arms start from the same local Qwen3-0.6B source weights. The run-time checks confirm matching task sample order and supervised target-token budgets against A0. The test split was not loaded. No NAS search was started.

All recorded rows are from the final post-task-4 stage. Generation metrics are reported only for the two generation tasks; Accuracy and Macro-F1 are reported for the two classification tasks. Teacher-forced NLL is listed for all tasks. Yes/No collapse means one exact label accounts for at least 95% of outputs.

## Final primary scores

| 架构 | 565 CIRCA ROUGE-L | 1603 SMCalFlow ROUGE-L | 1501 DSTC3 Accuracy | 1384 Deal-or-No-Deal Accuracy |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.0000 | 0.0068 | 0.6800 | 0.9200 |
| B1-static | 0.0000 | 0.0108 | 0.8000 | 0.9000 |
| B2-input | 0.0089 | 0.0068 | 0.5800 | 0.9400 |
| B3-oracle | 0.0000 | 0.0068 | 0.7400 | 0.9200 |
| B2-random-location | 0.0000 | 0.0139 | 0.7600 | 0.9200 |

**S2: 565 → 1501 → 1603 → 1384**

| 架构 | 565 CIRCA ROUGE-L | 1603 SMCalFlow ROUGE-L | 1501 DSTC3 Accuracy | 1384 Deal-or-No-Deal Accuracy |
| --- | ---: | ---: | ---: | ---: |
| B0 | 0.0067 | 0.0677 | 0.5000 | 0.9000 |
| B1-static | 0.0073 | 0.1112 | 0.4800 | 0.9400 |
| B2-input | 0.0089 | 0.0605 | 0.8400 | 0.9200 |
| B3-oracle | 0.0062 | 0.0377 | 0.4800 | 0.9200 |
| B2-random-location | 0.0080 | 0.0939 | 0.6000 | 0.9400 |

S1 uses 565 → 1603 → 1501 → 1384. The first table is S1; the second table is S2. The detailed metric tables and full-resolution values are in `results/phase6b/pilot_task_metrics.csv`.

## B2 versus parameter-matched B1

| 任务 | 主指标 | S1 Δ(B2−B1) | S2 Δ(B2−B1) | 两顺序平均 |
| --- | --- | --- | --- | --- |
| 565 CIRCA | rouge_l_f1 | 0.0089 | 0.0016 | 0.0053 |
| 1603 SMCalFlow | rouge_l_f1 | -0.0040 | -0.0508 | -0.0274 |
| 1501 DSTC3 | accuracy | -0.2200 | 0.3600 | 0.0700 |
| 1384 Deal-or-No-Deal | accuracy | 0.0400 | -0.0200 | 0.0100 |

B2 was higher than B1 in 4/8 task-by-order primary-score comparisons. The direction is not stable: B2 loses on SMCalFlow in both orders, while the DSTC3 change reverses sign between orders. CIRCA's positive difference is small and both arms remain near zero. This single-seed, 50-example pilot does not establish a reliable B2 advantage.

### Per-task final metrics

#### S1

| 架构 | 任务 | 主指标 | 主指标值 | Token-F1 | Teacher NLL | Accuracy | Macro-F1 | Yes/No占比 | 塌缩 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 565 CIRCA | rouge_l_f1 | 0.0000 | 0.0000 | 5.3113 | — | — | 1.000 | 是 |
| B0 | 1603 SMCalFlow | rouge_l_f1 | 0.0068 | 0.0068 | 2.2206 | — | — | 0.940 | 否 |
| B0 | 1501 DSTC3 | accuracy | 0.6800 | — | 0.2021 | 0.6800 | 0.6667 | 0.560 | 否 |
| B0 | 1384 Deal-or-No-Deal | accuracy | 0.9200 | — | 0.0931 | 0.9200 | 0.8834 | 0.820 | 否 |
| B1-static | 565 CIRCA | rouge_l_f1 | 0.0000 | 0.0000 | 5.0681 | — | — | 0.900 | 否 |
| B1-static | 1603 SMCalFlow | rouge_l_f1 | 0.0108 | 0.0108 | 2.2427 | — | — | 0.860 | 否 |
| B1-static | 1501 DSTC3 | accuracy | 0.8000 | — | 0.1685 | 0.8000 | 0.7259 | 0.760 | 否 |
| B1-static | 1384 Deal-or-No-Deal | accuracy | 0.9000 | — | 0.1076 | 0.9000 | 0.8493 | 0.840 | 否 |
| B2-input | 565 CIRCA | rouge_l_f1 | 0.0089 | 0.0089 | 4.6891 | — | — | 0.980 | 是 |
| B2-input | 1603 SMCalFlow | rouge_l_f1 | 0.0068 | 0.0068 | 1.9668 | — | — | 0.980 | 是 |
| B2-input | 1501 DSTC3 | accuracy | 0.5800 | — | 0.3292 | 0.5800 | 0.5758 | 0.660 | 否 |
| B2-input | 1384 Deal-or-No-Deal | accuracy | 0.9400 | — | 0.0892 | 0.9400 | 0.9153 | 0.800 | 否 |
| B3-oracle | 565 CIRCA | rouge_l_f1 | 0.0000 | 0.0000 | 4.8713 | — | — | 0.960 | 是 |
| B3-oracle | 1603 SMCalFlow | rouge_l_f1 | 0.0068 | 0.0068 | 2.0527 | — | — | 0.880 | 否 |
| B3-oracle | 1501 DSTC3 | accuracy | 0.7400 | — | 0.1738 | 0.7400 | 0.7060 | 0.580 | 否 |
| B3-oracle | 1384 Deal-or-No-Deal | accuracy | 0.9200 | — | 0.0953 | 0.9200 | 0.8834 | 0.820 | 否 |
| B2-random-location | 565 CIRCA | rouge_l_f1 | 0.0000 | 0.0000 | 4.8854 | — | — | 0.880 | 否 |
| B2-random-location | 1603 SMCalFlow | rouge_l_f1 | 0.0139 | 0.0139 | 1.9820 | — | — | 0.800 | 否 |
| B2-random-location | 1501 DSTC3 | accuracy | 0.7600 | — | 0.1721 | 0.7600 | 0.7243 | 0.600 | 否 |
| B2-random-location | 1384 Deal-or-No-Deal | accuracy | 0.9200 | — | 0.0950 | 0.9200 | 0.8904 | 0.780 | 否 |

#### S2

| 架构 | 任务 | 主指标 | 主指标值 | Token-F1 | Teacher NLL | Accuracy | Macro-F1 | Yes/No占比 | 塌缩 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 565 CIRCA | rouge_l_f1 | 0.0067 | 0.0067 | 3.8714 | — | — | 0.980 | 是 |
| B0 | 1603 SMCalFlow | rouge_l_f1 | 0.0677 | 0.0677 | 1.3729 | — | — | 0.840 | 否 |
| B0 | 1501 DSTC3 | accuracy | 0.5000 | — | 0.3660 | 0.5000 | 0.4998 | 0.740 | 否 |
| B0 | 1384 Deal-or-No-Deal | accuracy | 0.9000 | — | 0.1117 | 0.9000 | 0.8588 | 0.800 | 否 |
| B1-static | 565 CIRCA | rouge_l_f1 | 0.0073 | 0.0073 | 3.7202 | — | — | 0.980 | 是 |
| B1-static | 1603 SMCalFlow | rouge_l_f1 | 0.1112 | 0.1112 | 1.3780 | — | — | 0.600 | 否 |
| B1-static | 1501 DSTC3 | accuracy | 0.4800 | — | 0.3218 | 0.4800 | 0.4800 | 0.760 | 否 |
| B1-static | 1384 Deal-or-No-Deal | accuracy | 0.9400 | — | 0.0805 | 0.9400 | 0.9153 | 0.800 | 否 |
| B2-input | 565 CIRCA | rouge_l_f1 | 0.0089 | 0.0089 | 4.7246 | — | — | 0.760 | 否 |
| B2-input | 1603 SMCalFlow | rouge_l_f1 | 0.0605 | 0.0605 | 1.9557 | — | — | 0.660 | 否 |
| B2-input | 1501 DSTC3 | accuracy | 0.8400 | — | 0.1408 | 0.8400 | 0.8162 | 0.600 | 否 |
| B2-input | 1384 Deal-or-No-Deal | accuracy | 0.9200 | — | 0.0915 | 0.9200 | 0.8834 | 0.820 | 否 |
| B3-oracle | 565 CIRCA | rouge_l_f1 | 0.0062 | 0.0062 | 4.3150 | — | — | 0.980 | 是 |
| B3-oracle | 1603 SMCalFlow | rouge_l_f1 | 0.0377 | 0.0377 | 1.6146 | — | — | 0.900 | 否 |
| B3-oracle | 1501 DSTC3 | accuracy | 0.4800 | — | 0.4303 | 0.4800 | 0.4800 | 0.760 | 否 |
| B3-oracle | 1384 Deal-or-No-Deal | accuracy | 0.9200 | — | 0.0953 | 0.9200 | 0.8904 | 0.780 | 否 |
| B2-random-location | 565 CIRCA | rouge_l_f1 | 0.0080 | 0.0080 | 4.3077 | — | — | 0.980 | 是 |
| B2-random-location | 1603 SMCalFlow | rouge_l_f1 | 0.0939 | 0.0939 | 1.7302 | — | — | 0.820 | 否 |
| B2-random-location | 1501 DSTC3 | accuracy | 0.6000 | — | 0.3025 | 0.6000 | 0.5942 | 0.640 | 否 |
| B2-random-location | 1384 Deal-or-No-Deal | accuracy | 0.9400 | — | 0.0871 | 0.9400 | 0.9153 | 0.800 | 否 |

## Continual-learning metrics

Forgetting and backward transfer use the project’s normalized-gain definitions. Raw nonnegative forgetting remains separated by metric so Accuracy and ROUGE-L are not averaged together.

| 顺序 | 架构 | 非负遗忘(归一化) | ROUGE-L非负遗忘 | Accuracy非负遗忘 | BWT(归一化) | Cross-stage AAA |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | B0 | 0.3818 | 0.2670 | 0.1200 | -0.3818 | -0.0291 |
| S1 | B1-static | 0.2057 | 0.2550 | 0.0000 | -0.1779 | -0.0092 |
| S1 | B2-input | 0.5732 | 0.2633 | 0.2600 | -0.5732 | -0.0321 |
| S1 | B3-oracle | 0.3766 | 0.2604 | 0.1200 | -0.3766 | 0.0084 |
| S1 | B2-random-location | 0.2814 | 0.2805 | 0.0400 | -0.2814 | 0.0043 |
| S2 | B0 | 0.3347 | 0.2427 | 0.1000 | -0.3069 | -0.2490 |
| S2 | B1-static | 0.5408 | 0.1872 | 0.2800 | -0.5408 | -0.1365 |
| S2 | B2-input | 0.1879 | 0.2327 | 0.0000 | -0.1323 | 0.0385 |
| S2 | B3-oracle | 0.6476 | 0.2520 | 0.3200 | -0.6476 | -0.1469 |
| S2 | B2-random-location | 0.3730 | 0.1859 | 0.1600 | -0.3730 | -0.0600 |

### New-task acquisition

| 顺序 | 架构 | 当前新任务 | 原始主指标增益 | 归一化增益 |
| --- | --- | --- | --- | --- |
| S1 | B0 | 565 CIRCA | -0.0037 | -0.0046 |
| S1 | B0 | 1603 SMCalFlow | 0.1892 | 0.2247 |
| S1 | B0 | 1501 DSTC3 | 0.0400 | 0.1667 |
| S1 | B0 | 1384 Deal-or-No-Deal | 0.0400 | 0.3333 |
| S1 | B1-static | 565 CIRCA | -0.0008 | -0.0010 |
| S1 | B1-static | 1603 SMCalFlow | 0.1665 | 0.1977 |
| S1 | B1-static | 1501 DSTC3 | 0.0200 | 0.0833 |
| S1 | B1-static | 1384 Deal-or-No-Deal | 0.0200 | 0.1667 |
| S1 | B2-input | 565 CIRCA | 0.0003 | 0.0004 |
| S1 | B2-input | 1603 SMCalFlow | 0.1868 | 0.2218 |
| S1 | B2-input | 1501 DSTC3 | 0.0800 | 0.3333 |
| S1 | B2-input | 1384 Deal-or-No-Deal | 0.0600 | 0.5000 |
| S1 | B3-oracle | 565 CIRCA | -0.0039 | -0.0049 |
| S1 | B3-oracle | 1603 SMCalFlow | 0.1763 | 0.2094 |
| S1 | B3-oracle | 1501 DSTC3 | 0.1000 | 0.4167 |
| S1 | B3-oracle | 1384 Deal-or-No-Deal | 0.0400 | 0.3333 |
| S1 | B2-random-location | 565 CIRCA | -0.0028 | -0.0035 |
| S1 | B2-random-location | 1603 SMCalFlow | 0.2226 | 0.2643 |
| S1 | B2-random-location | 1501 DSTC3 | 0.0400 | 0.1667 |
| S1 | B2-random-location | 1384 Deal-or-No-Deal | 0.0400 | 0.3333 |
| S2 | B0 | 565 CIRCA | -0.0037 | -0.0046 |
| S2 | B0 | 1501 DSTC3 | -0.1800 | -0.7500 |
| S2 | B0 | 1603 SMCalFlow | 0.2083 | 0.2473 |
| S2 | B0 | 1384 Deal-or-No-Deal | 0.0200 | 0.1667 |
| S2 | B1-static | 565 CIRCA | -0.0008 | -0.0010 |
| S2 | B1-static | 1501 DSTC3 | 0.0000 | 0.0000 |
| S2 | B1-static | 1603 SMCalFlow | 0.1385 | 0.1645 |
| S2 | B1-static | 1384 Deal-or-No-Deal | 0.0600 | 0.5000 |
| S2 | B2-input | 565 CIRCA | 0.0003 | 0.0004 |
| S2 | B2-input | 1501 DSTC3 | 0.0400 | 0.1667 |
| S2 | B2-input | 1603 SMCalFlow | 0.1792 | 0.2128 |
| S2 | B2-input | 1384 Deal-or-No-Deal | 0.0400 | 0.3333 |
| S2 | B3-oracle | 565 CIRCA | -0.0039 | -0.0049 |
| S2 | B3-oracle | 1501 DSTC3 | 0.0400 | 0.1667 |
| S2 | B3-oracle | 1603 SMCalFlow | 0.1966 | 0.2334 |
| S2 | B3-oracle | 1384 Deal-or-No-Deal | 0.0400 | 0.3333 |
| S2 | B2-random-location | 565 CIRCA | -0.0028 | -0.0035 |
| S2 | B2-random-location | 1501 DSTC3 | 0.0000 | 0.0000 |
| S2 | B2-random-location | 1603 SMCalFlow | 0.1214 | 0.1441 |
| S2 | B2-random-location | 1384 Deal-or-No-Deal | 0.0600 | 0.5000 |

## Yes/No output collapse

| 顺序/架构 | 塌缩任务数 | 检查任务数 |
| --- | --- | --- |
| S1/B0 | 1 | 4 |
| S1/B1-static | 0 | 4 |
| S1/B2-input | 2 | 4 |
| S1/B3-oracle | 1 | 4 |
| S1/B2-random-location | 0 | 4 |
| S2/B0 | 1 | 4 |
| S2/B1-static | 1 | 4 |
| S2/B2-input | 0 | 4 |
| S2/B3-oracle | 1 | 4 |
| S2/B2-random-location | 1 | 4 |

The threshold is an exact Yes or No output share ≥0.95. This diagnostic catches degeneration that a task’s headline Accuracy or ROUGE-L alone can hide.

## Routing mechanism and controls

B2 takes the current token hidden state as its router input, selects one of four shared bottleneck experts per token, and receives no Task ID. At inference the route is deterministic top-1; training uses hard Gumbel selection with a straight-through gate gradient. The routed module therefore has input-dependent nonlinear dispatch, rather than a static linear reparameterization.

Across the 8 sequence/task evaluations, mean normalized route entropy was 0.373 and mean dominant-expert share was 0.777. Mean total-variation distance between the same task across S1/S2 was 0.303; between different tasks within an order it was 0.047. These descriptive values mix task identity with order/training effects and are not a causal task-specialization test.

| 顺序 | 任务 | 各层主导专家及占比 | 平均归一化熵 |
| --- | --- | --- | --- |
| S1 | 565 CIRCA | L25: E2 0.95; L27: E2 0.91 | 0.194 |
| S1 | 1603 SMCalFlow | L25: E2 0.94; L27: E2 0.89 | 0.247 |
| S1 | 1501 DSTC3 | L25: E2 0.93; L27: E2 0.87 | 0.260 |
| S1 | 1384 Deal-or-No-Deal | L25: E2 0.97; L27: E2 0.92 | 0.167 |
| S2 | 565 CIRCA | L25: E2 0.76; L27: E2 0.58 | 0.536 |
| S2 | 1603 SMCalFlow | L25: E2 0.75; L27: E2 0.55 | 0.520 |
| S2 | 1501 DSTC3 | L25: E2 0.69; L27: E0 0.52 | 0.531 |
| S2 | 1384 Deal-or-No-Deal | L25: E2 0.70; L27: E0 0.51 | 0.525 |

Primary metric and NLL results under final input/random/static-gate/off controls:

| 顺序 | 任务 | 控制 | Input主指标 | 控制主指标 | Δ控制−Input | Input NLL | 控制 NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 565 CIRCA | random | 0.0089 | 0.0089 | 0.0000 | 4.6891 | 4.6707 |
| S1 | 565 CIRCA | static_gate | 0.0089 | 0.0089 | 0.0000 | 4.6891 | 4.6661 |
| S1 | 565 CIRCA | off | 0.0089 | 0.0089 | 0.0000 | 4.6891 | 4.6622 |
| S1 | 1603 SMCalFlow | random | 0.0068 | 0.0068 | 0.0000 | 1.9668 | 1.9557 |
| S1 | 1603 SMCalFlow | static_gate | 0.0068 | 0.0068 | 0.0000 | 1.9668 | 1.9549 |
| S1 | 1603 SMCalFlow | off | 0.0068 | 0.0068 | 0.0000 | 1.9668 | 1.9536 |
| S1 | 1501 DSTC3 | random | 0.5800 | 0.5800 | 0.0000 | 0.3292 | 0.3320 |
| S1 | 1501 DSTC3 | static_gate | 0.5800 | 0.5800 | 0.0000 | 0.3292 | 0.3342 |
| S1 | 1501 DSTC3 | off | 0.5800 | 0.5800 | 0.0000 | 0.3292 | 0.3342 |
| S1 | 1384 Deal-or-No-Deal | random | 0.9400 | 0.9400 | 0.0000 | 0.0892 | 0.0898 |
| S1 | 1384 Deal-or-No-Deal | static_gate | 0.9400 | 0.9400 | 0.0000 | 0.0892 | 0.0898 |
| S1 | 1384 Deal-or-No-Deal | off | 0.9400 | 0.9400 | 0.0000 | 0.0892 | 0.0898 |
| S2 | 565 CIRCA | random | 0.0089 | 0.0089 | 0.0000 | 4.7246 | 4.6823 |
| S2 | 565 CIRCA | static_gate | 0.0089 | 0.0089 | 0.0000 | 4.7246 | 4.6626 |
| S2 | 565 CIRCA | off | 0.0089 | 0.0089 | 0.0000 | 4.7246 | 4.6583 |
| S2 | 1603 SMCalFlow | random | 0.0605 | 0.0605 | 0.0000 | 1.9557 | 1.9277 |
| S2 | 1603 SMCalFlow | static_gate | 0.0605 | 0.0722 | 0.0117 | 1.9557 | 1.9218 |
| S2 | 1603 SMCalFlow | off | 0.0605 | 0.0722 | 0.0117 | 1.9557 | 1.9177 |
| S2 | 1501 DSTC3 | random | 0.8400 | 0.8400 | 0.0000 | 0.1408 | 0.1432 |
| S2 | 1501 DSTC3 | static_gate | 0.8400 | 0.8400 | 0.0000 | 0.1408 | 0.1433 |
| S2 | 1501 DSTC3 | off | 0.8400 | 0.8400 | 0.0000 | 0.1408 | 0.1432 |
| S2 | 1384 Deal-or-No-Deal | random | 0.9200 | 0.9200 | 0.0000 | 0.0915 | 0.0900 |
| S2 | 1384 Deal-or-No-Deal | static_gate | 0.9200 | 0.9200 | 0.0000 | 0.0915 | 0.0911 |
| S2 | 1384 Deal-or-No-Deal | off | 0.9200 | 0.9200 | 0.0000 | 0.0915 | 0.0901 |

On this 50-example dev set, random routing matched input routing on all 8 task/order primary scores. Static-gate and off controls also matched on most primary scores; the only visible primary-score change is on S2 SMCalFlow, where disabling the branch or using a static gate is better than input routing. NLL differences are small and task-dependent. This does not show that input-conditioned routing improves over random or static computation.

### Gradient and branch participation traces

The pilot sampled branch/router gradients at steps 1, 10, …, and the final step for each task. `branch_rms` is the residual delta magnitude at sampled forwards. The first zero-output initialization is expected; later nonzero branch RMS and parameter gradients show that branches participate in training. Full sampled traces are kept as `gradient_diagnostics.jsonl` in each run directory.

- `S1/B0`: sampled gradient records 0, branch slots with nonzero gradients 0/0, router samples with nonzero gradient 0/0, branch RMS nonzero 0/0.
- `S1/B1-static`: sampled gradient records 44, branch slots with nonzero gradients 88/88, router samples with nonzero gradient 0/88, branch RMS nonzero 86/88.
- `S1/B2-input`: sampled gradient records 44, branch slots with nonzero gradients 272/352, router samples with nonzero gradient 86/88, branch RMS nonzero 86/88.
- `S1/B3-oracle`: sampled gradient records 44, branch slots with nonzero gradients 88/352, router samples with nonzero gradient 0/88, branch RMS nonzero 80/88.
- `S1/B2-random-location`: sampled gradient records 44, branch slots with nonzero gradients 348/352, router samples with nonzero gradient 86/88, branch RMS nonzero 86/88.
- `S2/B0`: sampled gradient records 0, branch slots with nonzero gradients 0/0, router samples with nonzero gradient 0/0, branch RMS nonzero 0/0.
- `S2/B1-static`: sampled gradient records 44, branch slots with nonzero gradients 88/88, router samples with nonzero gradient 0/88, branch RMS nonzero 86/88.
- `S2/B2-input`: sampled gradient records 44, branch slots with nonzero gradients 270/352, router samples with nonzero gradient 86/88, branch RMS nonzero 86/88.
- `S2/B3-oracle`: sampled gradient records 44, branch slots with nonzero gradients 88/352, router samples with nonzero gradient 0/88, branch RMS nonzero 80/88.
- `S2/B2-random-location`: sampled gradient records 44, branch slots with nonzero gradients 348/352, router samples with nonzero gradient 86/88, branch RMS nonzero 86/88.

## Capacity, compute, time, and memory

FLOPs are linear multiply-add estimates for a 512-token prefill and exclude activation, token-dispatch, and cache costs. Wall time covers the recorded end-to-end run interval (training, dev evaluation, save/reload audit), not isolated GPU kernel time.

| 顺序 | 架构 | 新增参数 | 占主干 | 估算额外FLOPs | 端到端秒 | 峰值显存GiB | 平均新任务归一化增益 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | B0 | 0 | 0.000 | 0.000 | 574.0 | 10.85 | 0.1800 |
| S1 | B1-static | 1,048,576 | 0.176 | 0.168 | 459.3 | 10.87 | 0.1117 |
| S1 | B2-input | 1,056,776 | 0.177 | 0.043 | 526.5 | 10.88 | 0.2639 |
| S1 | B3-oracle | 1,056,776 | 0.177 | 0.042 | 467.7 | 10.86 | 0.2386 |
| S1 | B2-random-location | 1,056,776 | 0.177 | 0.043 | 477.7 | 10.87 | 0.1902 |
| S2 | B0 | 0 | 0.000 | 0.000 | 517.5 | 10.85 | -0.0852 |
| S2 | B1-static | 1,048,576 | 0.176 | 0.168 | 464.9 | 10.87 | 0.1659 |
| S2 | B2-input | 1,056,776 | 0.177 | 0.043 | 550.1 | 10.87 | 0.1783 |
| S2 | B3-oracle | 1,056,776 | 0.177 | 0.042 | 495.8 | 10.86 | 0.1821 |
| S2 | B2-random-location | 1,056,776 | 0.177 | 0.043 | 494.6 | 10.87 | 0.1602 |

All adapter arms stayed below 1% added parameters and below the 24 GiB GPU limit. B1 adds 1,048,576 parameters; B2/B3 add 1,056,776 (0.177% of the 596,049,920-parameter backbone). The saved final states are full model states for replay/audit; no intermediate stage checkpoints or historical models were saved.

## Decision

Do not expand to seeds 20/30 or start NAS from this result. The predeclared seed-10 gate was not met: B2 is not consistently better than parameter-matched B1 across task orders, has no primary-score advantage over random routing in the measured controls, and sometimes performs worse than the static/off controls. The observed route distributions are descriptive and shift substantially with task order; they do not establish that routing mitigated representation interference.

**Current evidence is insufficient to use Task-Conditional Residual Pathways as the architectural-search foundation of an ACL 2027 paper.** Keep it as an unvalidated hypothesis. A future replication should first use repeated seeds and a larger locked dev evaluation, and should require B2 to beat B1, random routing, and static/off controls on predeclared old-generation retention and new-classification metrics before NAS is considered.

## Artifacts

- `results/phase6b/pilot_summary.json` — consolidated S1/S2 run records and mechanism summary.
- `results/phase6b/pilot_task_metrics.csv` — one row per task, sequence, and architecture.
- `results/phase6b/pilot_resource_summary.csv` — parameter, FLOPs, runtime, memory, acquisition, forgetting, and BWT summary.
- `figures/phase6b/pilot_final_primary_scores.png` — final dev primary scores.
- `figures/phase6b/pilot_b2_vs_b1_delta.png` — paired primary-score deltas.
- `figures/phase6b/pilot_b2_route_controls.png` — B2 route-mode controls.
- `figures/phase6b/pilot_b2_route_usage.png` — per-task expert usage shares.
