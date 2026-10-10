# Continual learning metric audit

This audit recalculates the three complete formal streams from their saved 5-stage × 4-task primary-score matrices. The partial `homogeneous/seed_314160` run is excluded because it has no complete `metrics.json`. No model was trained by this audit.

## Metric definitions

- **Raw primary score matrix:** each task retains its own primary score and unit: ROUGE-L F1 for task565/task1603 and Accuracy for task1501/task1384. The matrix is preserved per stream in `D:/Task-Aware-TransNAS/results/baseline_full/metrics.json`; the CSV has one raw-score column per task. Raw Accuracy and raw ROUGE-L are never averaged together.
- **Normalized gain:** for task *t* at stage *s*, `(score[t,s] − baseline[t]) / max(1 − baseline[t], 1e−8)`. The per-task matrix is preserved for every stage. The existing all-task mean normalized gain is also preserved, including unlearned tasks, under its historical field name and the clearer `mean_normalized_gain_all_tasks` alias.
- **Nonnegative average forgetting:** for each previously learned task, take `max(0, best post-acquisition normalized gain so far − current normalized gain)`, then average the task-level clipped drops. An improvement on one task cannot cancel forgetting on another.
- **Backward transfer (BWT):** for each previously learned task, current normalized gain minus its normalized gain immediately after that task was learned; average across prior tasks. Positive values indicate later stages improved the earlier task relative to its acquisition-stage score.
- **Per-stage mean seen-task score:** average raw scores among tasks already learned, grouped separately by primary metric (`accuracy` and `rouge_l_f1`). The baseline has no seen tasks.
- **Cross-stage AAA:** mean of the stage-local mean normalized gain over seen tasks for every trained stage up to and including the current stage. The per-stage legacy AAA remains the current stage's seen-task mean only.
- **Legacy forgetting:** the prior implementation averaged `best earlier normalized gain − current normalized gain` without clipping each task at zero. It is retained in the saved files as `legacy_cl_summary` and `average_forgetting_signed_normalized_legacy` for comparison.

## Per-stage metrics

Raw means are split by metric. Baseline-stage entries have no seen-task raw mean.

| Sequence | Stage | Seen tasks | Mean seen Accuracy | Mean seen ROUGE-L | Mean seen normalized gain | Cross-stage AAA | Nonnegative forgetting | BWT | Legacy forgetting |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | baseline | 0 | — | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| homogeneous | after_1_task565_circa_answer_generation | 1 | — | 0.2003 | 0.0094 | 0.0094 | 0.0000 | 0.0000 | 0.0000 |
| homogeneous | after_2_task1603_smcalflow_sentence_generation | 2 | — | 0.1653 | 0.0116 | 0.0105 | 0.1299 | -0.1299 | 0.1299 |
| homogeneous | after_3_task1501_dstc3_answer_generation | 3 | 0.7400 | 0.0112 | -0.0185 | 0.0008 | 0.2431 | -0.2431 | 0.2431 |
| homogeneous | after_4_task1384_deal_or_no_dialog_classification | 4 | 0.8400 | 0.0200 | 0.0432 | 0.0114 | 0.1555 | -0.0444 | 0.0444 |
| heterogeneous | baseline | 0 | — | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heterogeneous | after_1_task565_circa_answer_generation | 1 | — | 0.2003 | 0.0094 | 0.0094 | 0.0000 | 0.0000 | 0.0000 |
| heterogeneous | after_2_task1501_dstc3_answer_generation | 2 | 0.7400 | 0.0000 | 0.0195 | 0.0145 | 0.2481 | -0.2481 | 0.2481 |
| heterogeneous | after_3_task1603_smcalflow_sentence_generation | 3 | 0.8000 | 0.1612 | 0.1506 | 0.0598 | 0.0988 | -0.0155 | 0.0155 |
| heterogeneous | after_4_task1384_deal_or_no_dialog_classification | 4 | 0.7900 | 0.0200 | -0.0263 | 0.0383 | 0.2099 | -0.1543 | 0.2099 |
| mixed | baseline | 0 | — | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mixed | after_1_task565_circa_answer_generation | 1 | — | 0.2003 | 0.0094 | 0.0094 | 0.0000 | 0.0000 | 0.0000 |
| mixed | after_2_task1501_dstc3_answer_generation | 2 | 0.7400 | 0.0000 | 0.0195 | 0.0145 | 0.2481 | -0.2481 | 0.2481 |
| mixed | after_3_task1384_deal_or_no_dialog_classification | 3 | 0.7900 | 0.0000 | 0.0130 | 0.0140 | 0.1240 | -0.1240 | 0.1240 |
| mixed | after_4_task1603_smcalflow_sentence_generation | 4 | 0.6600 | 0.1837 | -0.3911 | -0.0873 | 0.7655 | -0.6914 | 0.6914 |

## Final raw primary scores and summary

The score columns remain grouped by their primary metric; no raw cross-metric mean is reported.

| Sequence | task565 ROUGE-L | task1603 ROUGE-L | task1501 Accuracy | task1384 Accuracy | Mean all-task normalized gain | Cross-stage AAA | Nonnegative forgetting | BWT | Legacy forgetting |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous | 0.0000 | 0.0400 | 0.8600 | 0.8200 | 0.0432 | 0.0114 | 0.1555 | -0.0444 | 0.0444 |
| heterogeneous | 0.0000 | 0.0400 | 0.7600 | 0.8200 | -0.0263 | 0.0383 | 0.2099 | -0.1543 | 0.2099 |
| mixed | 0.0619 | 0.3055 | 0.8200 | 0.5000 | -0.3911 | -0.0873 | 0.7655 | -0.6914 | 0.6914 |

The final-score columns are written in canonical task order: task565 ROUGE-L, task1603 ROUGE-L, task1501 Accuracy, task1384 Accuracy, irrespective of stream order.

## Interpretation and artifacts

The new nonnegative forgetting measure is higher than the legacy signed value for homogeneous (0.1555 vs 0.0444) and mixed (0.7655 vs 0.6914), because task improvements no longer offset drops on another prior task. Heterogeneous is unchanged at 0.2099 because its final prior-task drops are all nonnegative. BWT uses the acquisition-stage anchor, so it can differ from forgetting, which compares against the best post-acquisition score observed so far.

The implementation is in `src/task_aware_transnas/cl/metrics.py`; its 9 focused tests pass via `python -m unittest discover -s tests -p "test_cl_metrics.py"`.

Machine-readable outputs: `D:/Task-Aware-TransNAS/results/baseline_full/metrics.json`, each sequence's `metrics.json` (`cl_metric_audit` plus unchanged `legacy_cl_summary`), and the expanded `accuracy_matrix.csv`.
