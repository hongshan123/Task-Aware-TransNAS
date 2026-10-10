# Type-switch effects across all 24 task orders

## Design

The experiment covers every permutation of the four fixed tasks. All orders use training seed 10, the same 500 train IDs and 50 dev IDs per task, the same local Qwen checkpoint, learning rate, optimizer steps, sequence/target budgets, and generation budget. Each task uses `safe_task_seed(10, task_id)`, so its row order and Torch random seed are held constant across permutations. Evaluation used dev only; the runner did not load test records. The full settings and ID lists are in `results/order_sensitivity/experiment_config.json`; stage scores and loss curves are in `results/order_sensitivity/order_metrics.json`.

Type-switch count is the number of adjacent task-type changes in an order. The three groups contain 8 orders each (1, 2, or 3 switches). Each task ID is the final task in 6 orders.

`mean_new_task_learning_normalized_gain` averages each task's normalized gain at the stage where that task is first learned. `final_nonnegative_forgetting_normalized` averages per-task clipped drops from the best post-acquisition score to the final stage. Task-specific metrics are normalized against each task's own base score; raw Accuracy and ROUGE-L remain separate in the CSV.

## Descriptive means by switch count

Values are mean (sample SD), with eight orders per group.

| Type switches | Orders | Mean new-task learning gain | Mean final nonnegative forgetting | Mean final BWT |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0.1339 (0.1025) | 1.2112 (1.0798) | -1.2077 (1.0813) |
| 2 | 8 | 0.0604 (0.1410) | 1.1958 (1.1981) | -1.1889 (1.2035) |
| 3 | 8 | 0.0542 (0.1504) | 0.6244 (0.7589) | -0.5793 (0.7103) |

## Controlling for final task identity

The adjusted slope is an ordinary least-squares slope for `outcome ~ type_switch_count + final_task_id fixed effects`. It is calculated by demeaning switch count and outcome within each final-task ID before fitting one common slope. Since task ID determines task type, this controls both final identity and type. The coefficient is the descriptive change in normalized outcome per additional type switch among orders with the same final task.

- Final nonnegative forgetting slope: **-0.2934** per additional type switch.
- Mean new-task learning gain slope: **-0.0399** per additional type switch.

| Final task ID | Type | Orders | Mean switches | Mean new-task gain | Mean forgetting | Within-task gain slope | Within-task forgetting slope |
|---|---|---:|---:|---:|---:|---:|---:|
| task1384_deal_or_no_dialog_classification | classification | 6 | 2.00 | 0.0861 | 0.3936 | -0.0708 | 0.0583 |
| task1501_dstc3_answer_generation | classification | 6 | 2.00 | 0.1009 | 0.6939 | -0.0154 | -0.0674 |
| task1603_smcalflow_sentence_generation | generation | 6 | 2.00 | 0.0428 | 0.5486 | -0.0375 | -0.3945 |
| task565_circa_answer_generation | generation | 6 | 2.00 | 0.1015 | 2.4058 | -0.0357 | -0.7699 |

## Per-order results

| Order | Type sequence | Switches | Final task | Mean new-task gain | Final forgetting |
|---|---|---:|---|---:|---:|
| O01 | generation -> generation -> classification -> classification | 1 | task1384_deal_or_no_dialog_classification | 0.1171 | 0.3256 |
| O02 | generation -> generation -> classification -> classification | 1 | task1501_dstc3_answer_generation | 0.1587 | 0.6820 |
| O03 | generation -> classification -> generation -> classification | 3 | task1384_deal_or_no_dialog_classification | -0.0638 | 0.4733 |
| O04 | generation -> classification -> classification -> generation | 2 | task1603_smcalflow_sentence_generation | -0.1807 | 0.6142 |
| O05 | generation -> classification -> generation -> classification | 3 | task1501_dstc3_answer_generation | 0.0759 | 0.4605 |
| O06 | generation -> classification -> classification -> generation | 2 | task1603_smcalflow_sentence_generation | 0.1645 | 0.1529 |
| O07 | generation -> generation -> classification -> classification | 1 | task1384_deal_or_no_dialog_classification | 0.1376 | 0.3808 |
| O08 | generation -> generation -> classification -> classification | 1 | task1501_dstc3_answer_generation | 0.2209 | 0.4594 |
| O09 | generation -> classification -> generation -> classification | 3 | task1384_deal_or_no_dialog_classification | 0.0352 | 0.4665 |
| O10 | generation -> classification -> classification -> generation | 2 | task565_circa_answer_generation | 0.1648 | 2.8302 |
| O11 | generation -> classification -> generation -> classification | 3 | task1501_dstc3_answer_generation | 0.2423 | 0.4112 |
| O12 | generation -> classification -> classification -> generation | 2 | task565_circa_answer_generation | 0.1368 | 3.1035 |
| O13 | classification -> generation -> generation -> classification | 2 | task1384_deal_or_no_dialog_classification | 0.0582 | 0.3271 |
| O14 | classification -> generation -> classification -> generation | 3 | task1603_smcalflow_sentence_generation | 0.2076 | 0.0541 |
| O15 | classification -> generation -> generation -> classification | 2 | task1384_deal_or_no_dialog_classification | 0.2323 | 0.3881 |
| O16 | classification -> generation -> classification -> generation | 3 | task565_circa_answer_generation | 0.1899 | 0.2431 |
| O17 | classification -> classification -> generation -> generation | 1 | task1603_smcalflow_sentence_generation | 0.2288 | 0.8122 |
| O18 | classification -> classification -> generation -> generation | 1 | task565_circa_answer_generation | 0.2396 | 2.8241 |
| O19 | classification -> generation -> generation -> classification | 2 | task1501_dstc3_answer_generation | -0.0216 | 0.4137 |
| O20 | classification -> generation -> classification -> generation | 3 | task1603_smcalflow_sentence_generation | -0.1459 | 0.4190 |
| O21 | classification -> generation -> generation -> classification | 2 | task1501_dstc3_answer_generation | -0.0710 | 1.7368 |
| O22 | classification -> generation -> classification -> generation | 3 | task565_circa_answer_generation | -0.1076 | 2.4678 |
| O23 | classification -> classification -> generation -> generation | 1 | task1603_smcalflow_sentence_generation | -0.0172 | 1.2390 |
| O24 | classification -> classification -> generation -> generation | 1 | task565_circa_answer_generation | -0.0145 | 2.9664 |

## Conclusion and limits

Mean acquisition gain has a negative within-final-task slope for all four final task IDs. The adjusted acquisition slope is -0.0399 per additional type switch. Final forgetting has mixed within-final-task slopes: positive for task1384_deal_or_no_dialog_classification and negative for task1501_dstc3_answer_generation, task1603_smcalflow_sentence_generation, task565_circa_answer_generation. The adjusted forgetting slope is -0.2934. The switch-group means decrease as switch count rises for both outcomes, but their standard deviations are large. This is a descriptive pattern in one seed, not evidence of a stable general effect.

The controlled slopes and switch-group means describe these exact four tasks, one training seed, and one dev split. They are not significance estimates and do not establish a causal effect of type switching. The experiment contains no test evaluation and makes no architecture or NAS claims.

Artifacts: `results/order_sensitivity/all_24_orders.csv`, `results/order_sensitivity/order_metrics.json`, `results/order_sensitivity/experiment_config.json`, and two final review checkpoints under `D:/Task-Aware-TransNAS/checkpoints/order_sensitivity/review/`.
