# Formal full-parameter continual SFT analysis

> Metric audit update: The `AAA (seen tasks)` and `Forgetting` columns below are the legacy stage-local AAA and signed, unclipped forgetting fields. The corrected nonnegative average forgetting, BWT, cross-stage AAA, and raw per-task score matrices are in [metric_audit.md](metric_audit.md) and `D:/Task-Aware-TransNAS/results/baseline_full/metrics.json`.

## Design and frozen choices

- CITB source: official order1, commit `bf50533b5bced4c388691ecc75e26773da96b3fd`.
- Task selection used only the balanced training pools, frozen Qwen3 prompt representations, and final-block supervised gradient cosines. Development scores and test scores did not select tasks or stream order.
- The four tasks contain two official Dialogue Generation tasks with free-text outputs and two Dialogue State Tracking tasks with Yes/No class outputs.
- Similarity groups: `task565_circa_answer_generation` + `task1603_smcalflow_sentence_generation` (Dialogue Generation / generation), and `task1501_dstc3_answer_generation` + `task1384_deal_or_no_dialog_classification` (Dialogue State Tracking / classification).
- Shared sample sizes: 500 train / 50 dev / 50 test per task; all three sequences use the same instance IDs.
- Formal prompt maximum: 512 tokens; complete task definition retained, long input tails truncated. Every evaluated prediction passed the task-definition preservation assertion.
- Model: local `C:\Users\Lenovo\Desktop\nas-cl-transformer\model\qwen3-0.6b`; CUDA device `NVIDIA GeForce RTX 4090`; 596,049,920 / 596,049,920 parameters trainable.
- Frozen hyperparameters: learning rate `1e-05`; `50` optimizer steps per task; gradient accumulation `4`; weight decay `0.01`.
- Hyperparameters were selected from a 3-by-3 development-only grid after the homogeneous tuning sequence; the frozen selection and all candidate metrics are in `D:/Task-Aware-TransNAS/results/baseline_full/tuning/tuning_results.json`.
- The test set was opened after the tuning decision was frozen. No test results were used to adjust the tasks, task order, or training parameters.

### Development-only tuning grid

| Candidate | Learning rate | Steps/task | Normalized all-task gain | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr_1e-05_steps_50 | 1.0e-05 | 50 | 0.0395 | 0.0000 | 0.0068 | 0.8600 | 0.9000 |
| lr_2e-05_steps_100 | 2.0e-05 | 100 | 0.0199 | 0.0000 | 0.0108 | 0.8000 | 0.9200 |
| lr_5e-06_steps_100 | 5.0e-06 | 100 | 0.0199 | 0.0000 | 0.0108 | 0.7600 | 0.9400 |
| lr_5e-06_steps_50 | 5.0e-06 | 50 | -0.0563 | 0.0114 | 0.0229 | 0.7600 | 0.9000 |
| lr_1e-05_steps_100 | 1.0e-05 | 100 | -0.1896 | 0.0000 | 0.0068 | 0.7600 | 0.8400 |
| lr_5e-06_steps_20 | 5.0e-06 | 20 | -0.2477 | 0.0089 | 0.0125 | 0.5400 | 0.9200 |
| lr_1e-05_steps_20 | 1.0e-05 | 20 | -0.2730 | 0.0000 | 0.0068 | 0.5200 | 0.9200 |
| lr_2e-05_steps_50 | 2.0e-05 | 50 | -0.3563 | 0.0000 | 0.0068 | 0.4800 | 0.9000 |
| lr_2e-05_steps_20 | 2.0e-05 | 20 | -0.5230 | 0.0000 | 0.0068 | 0.2400 | 0.9400 |

## Reproducible task sequences

| Sequence | Task order | Mean adjacent similarity z |
|---|---|---:|
| homogeneous | task565_circa_answer_generation -> task1603_smcalflow_sentence_generation -> task1501_dstc3_answer_generation -> task1384_deal_or_no_dialog_classification | 1.5344 |
| heterogeneous | task565_circa_answer_generation -> task1501_dstc3_answer_generation -> task1603_smcalflow_sentence_generation -> task1384_deal_or_no_dialog_classification | 0.0803 |
| mixed | task565_circa_answer_generation -> task1501_dstc3_answer_generation -> task1384_deal_or_no_dialog_classification -> task1603_smcalflow_sentence_generation | 1.0486 |

## Test results at each stage

Classification reports Accuracy and fixed-label Macro-F1. Free-text generation reports ROUGE-L F1 as primary, with Token-F1 and normalized Exact Match as auxiliary metrics. Raw score units are kept separate.

| Sequence | Stage | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy / Macro-F1 | task1384 Accuracy / Macro-F1 | Normalized all-task gain | AAA (seen tasks) | Forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | baseline | 0.1927 | 0.1068 | 0.640 / 0.438 | 0.840 / 0.729 | 0.0000 | 0.0000 | 0.0000 |
| homogeneous | after_1_task565_circa_answer_generation | 0.2003 | 0.0954 | 0.000 / 0.000 | 0.000 / 0.000 | -1.7578 | 0.0094 | 0.0000 |
| homogeneous | after_2_task1603_smcalflow_sentence_generation | 0.0954 | 0.2352 | 0.160 / 0.288 | 0.000 / 0.000 | -1.6400 | 0.0116 | 0.1299 |
| homogeneous | after_3_task1501_dstc3_answer_generation | 0.0000 | 0.0224 | 0.740 / 0.653 | 0.240 / 0.194 | -0.9513 | -0.0185 | 0.2431 |
| homogeneous | after_4_task1384_deal_or_no_dialog_classification | 0.0000 | 0.0400 | 0.860 / 0.842 | 0.820 / 0.681 | 0.0432 | 0.0432 | 0.0444 |
| heterogeneous | baseline | 0.1927 | 0.1068 | 0.640 / 0.438 | 0.840 / 0.729 | 0.0000 | 0.0000 | 0.0000 |
| heterogeneous | after_1_task565_circa_answer_generation | 0.2003 | 0.0954 | 0.000 / 0.000 | 0.000 / 0.000 | -1.7578 | 0.0094 | 0.0000 |
| heterogeneous | after_2_task1501_dstc3_answer_generation | 0.0000 | 0.0224 | 0.740 / 0.653 | 0.240 / 0.194 | -0.9513 | 0.0195 | 0.2481 |
| heterogeneous | after_3_task1603_smcalflow_sentence_generation | 0.0407 | 0.2816 | 0.800 / 0.762 | 0.380 / 0.468 | -0.6058 | 0.1506 | 0.0155 |
| heterogeneous | after_4_task1384_deal_or_no_dialog_classification | 0.0000 | 0.0400 | 0.760 / 0.760 | 0.820 / 0.647 | -0.0263 | -0.0263 | 0.2099 |
| mixed | baseline | 0.1927 | 0.1068 | 0.640 / 0.438 | 0.840 / 0.729 | 0.0000 | 0.0000 | 0.0000 |
| mixed | after_1_task565_circa_answer_generation | 0.2003 | 0.0954 | 0.000 / 0.000 | 0.000 / 0.000 | -1.7578 | 0.0094 | 0.0000 |
| mixed | after_2_task1501_dstc3_answer_generation | 0.0000 | 0.0224 | 0.740 / 0.653 | 0.240 / 0.194 | -0.9513 | 0.0195 | 0.2481 |
| mixed | after_3_task1384_deal_or_no_dialog_classification | 0.0000 | 0.0400 | 0.740 / 0.740 | 0.840 / 0.702 | -0.0089 | 0.0130 | 0.1240 |
| mixed | after_4_task1603_smcalflow_sentence_generation | 0.0619 | 0.3055 | 0.820 / 0.790 | 0.500 / 0.617 | -0.3911 | -0.3911 | 0.6914 |

## Final-stage comparison

| Sequence | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy | Normalized all-task gain | AAA | Average forgetting | Wall time (s) | Peak allocated VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | 0.0000 | 0.0400 | 0.860 | 0.820 | 0.0432 | 0.0432 | 0.0444 | 353.3 | 10.85 |
| heterogeneous | 0.0000 | 0.0400 | 0.760 | 0.820 | -0.0263 | -0.0263 | 0.2099 | 338.0 | 10.85 |
| mixed | 0.0619 | 0.3055 | 0.820 | 0.500 | -0.3911 | -0.3911 | 0.6914 | 333.3 | 10.85 |

## Verification and interpretation

- Automated artifact audit: PASS (60 stage/task cells, 3000 predictions, each cell matched the official 50 test IDs).
- All three runs updated all 596,049,920 parameters; none used OOM recovery; peak allocated GPU memory was about 10.85 GiB.
- On task565, task1603, and task1384 prompts, all input tokens fit the window. For task1501, 159/500 train, 15/50 dev, and 12/50 test examples had long dialogue inputs whose oldest context tokens were dropped; the complete definition and input suffix, including the question, remained in the prompt.
- The homogeneous sequence ended with both Yes/No tasks strong but task565 ROUGE-L at zero. The heterogeneous and mixed streams produced different retention profiles. The results show substantial interference between free-text and short-label tasks under this small full-parameter SFT baseline.
- The study uses one seed and a local 0.6B model; these runs are an initial formal baseline, not a multi-seed significance estimate. No NAS, architecture changes, LoRA, or replay was used.

## Artifacts

- `D:/Task-Aware-TransNAS/results/baseline_full/accuracy_matrix.csv`: combined 15-stage-by-4-task metrics.
- `D:/Task-Aware-TransNAS/results/baseline_full/validation_summary.json`: test ID, matrix, checkpoint, parameter-count, and OOM verification.
- `D:/Task-Aware-TransNAS/results/baseline_full/tuning/tuning_results.json`: dev-only candidate results.
- `D:/Task-Aware-TransNAS/results/baseline_full/{homogeneous,heterogeneous,mixed}/`: per-sequence predictions, metrics, reports, and loss curves.
- `D:/Task-Aware-TransNAS/checkpoints/baseline_full/{homogeneous,heterogeneous,mixed}/`: one checkpoint per task boundary.

The wide machine-readable artifact retains each task metric separately and is the source for reproducing the tables above.
