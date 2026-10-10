# Phase 6A: Component-Level Causal Localization

## Scope and safeguards

This phase localizes the final A0/S1/seed-10 model's behavior using same-trajectory checkpoints. It reads only train and development data. The test split was not loaded; the causal screens and confirmations used no optimizer steps and no NAS search. Because the original after-task1501 weights were missing, a prerequisite replay ran that task's original 50 optimizer steps from the retained task1603 boundary. Its training order, supervised-token count, losses, and resulting weights matched the original final trajectory audit; no new task sequence was trained.

The target checkpoint is the final after-task1384 Phase 4B model. Historical activation/weight references are after task1603 for the two generation tasks and after task1501 for the two classification tasks. The Embedding and LM Head are tied and were treated as a single `model.embed_tokens.weight` parameter for rollback. Logit replacement is reported as an output-level upper bound, not a deployable internal component.

All new artifacts are under `results/phase6a/` and `figures/phase6a/`; prior Phase 4B/5 artifacts were read only. The CUDA peak across the Phase 6A runs was 8.62 GiB allocated (9.21 GiB reserved) on the RTX 4090.

The machine-readable completion checks are summarized in [phase6a_audit.md](phase6a_audit.md) and `results/phase6a/phase6a_audit.json`.

The focused Phase 6A tests passed (5 passed in 5.28s); the full repository suite passed (27 passed in 5.46s).

## Method

The activation screen used dev indices 5–14 (10 examples/task). For each task, reference activations were captured on the exact same prompt-plus-reference-token sequence, then substituted into the final model at `self_attn.o_proj`, MLP output, whole decoder-block output, token embedding output, final norm output, or selected logits. It scanned all 28 single layers and all sliding contiguous groups of widths 2, 4, 8, 14, and 28. The objective was per-example teacher-forced completion NLL. Each candidate batch included an unpatched row; the zero-operation control had exactly zero paired change.

The parameter screen used dev indices 15–24 (10 examples/task). It restored matching tensors from the appropriate earlier boundary without an optimizer step. It scanned single layers, contiguous layer groups, global Attention, global MLP, combined Attention+MLP, norms, tied embedding/LM Head, and the full backbone. It recorded parameter count, fraction of model weights, rollback update L2/RMS, and runtime. The best screen condition per task was confirmed against a seeded random-direction update with the same selected parameter count and update L2 norm.

Locked confirmation used dev indices 25–49 (25 examples/task). Teacher-forced NLL and greedy autoregressive generation (up to 64 new tokens) were evaluated for the baseline and screen-selected activation/parameter conditions. Activation generation used paired decoding: source and target received the same prompt and target-generated tokens, with reference activations substituted at the selected site at prefill and every cached decode step. The activation synergy check compared Attention-only, MLP-only, joint Attention+MLP, and whole-block replacement at the same selected layer/group. The output includes raw per-example JSONL and paired bootstrap intervals over 10,000 resamples of the 25 locked IDs.

## Locked confirmation

| Task | NLL baseline → activation | Paired NLL improvement (95% bootstrap CI) | Primary baseline → activation | Parameter rollback and primary | Matched random NLL |
|---|---:|---:|---:|---|---:|
| 565 CIRCA | 5.108 → 2.860 | 2.290 [2.012, 2.590] | 0.000 → 0.106 | attention_plus_mlp_global (73.9% weights), 0.096 rouge_l_f1 | 5.115 |
| 1603 SMCalFlow | 2.522 → 1.067 | 1.553 [1.358, 1.754] | 0.014 → 0.272 | attention_plus_mlp_global (73.9% weights), 0.262 rouge_l_f1 | 2.534 |
| 1501 DSTC3 | 0.230 → 0.133 | 0.098 [0.053, 0.149] | 0.640 → 0.760 | mlp_group_28_00_27 (44.3% weights), 0.760 accuracy | 0.228 |
| 1384 Deal-or-No-Deal | 0.072 → 0.056 | 0.015 [-0.001, 0.027] | 0.960 → 0.960 | attention_group_14_05_18 (14.8% weights), 0.960 accuracy | 0.072 |

The generation primary metric is ROUGE-L F1; the classifier primary metric is parsed Yes/No accuracy. The NLL interval is a paired bootstrap interval for the mean per-example token NLL improvement (baseline minus intervention); token-weighted NLL is shown in the table. The per-task detailed metrics also include Token-F1/macro-F1, generated length, EOS termination, exact Yes/No rates, parseable/invalid-format rates, first-reference-token log probability, and the Yes-minus-No first-token log-probability margin.

## Localization and interaction results

- **565 CIRCA**: selected activation condition `block_layer_25`. On the locked same-location swap, NLL improvements were Attention 0.552, MLP 0.143, Attention+MLP 0.651, and whole block 2.293. Parameter screening selected `attention_plus_mlp_global` (73.9% of model parameters).
- **1603 SMCalFlow**: selected activation condition `block_layer_27`. On the locked same-location swap, NLL improvements were Attention 0.166, MLP -0.252, Attention+MLP -0.176, and whole block 1.549. Parameter screening selected `attention_plus_mlp_global` (73.9% of model parameters).
- **1501 DSTC3**: selected activation condition `attention_group_14_13_26`. On the locked same-location swap, NLL improvements were Attention 0.097, MLP 0.096, Attention+MLP 0.123, and whole block 0.126. Parameter screening selected `mlp_group_28_00_27` (44.3% of model parameters).
- **1384 Deal-or-No-Deal**: selected activation condition `mlp_group_14_14_27`. On the locked same-location swap, NLL improvements were Attention -0.047, MLP 0.016, Attention+MLP -0.154, and whole block -0.210. Parameter screening selected `attention_group_14_05_18` (14.8% of model parameters).

## Task-stage trajectory

The same 25 locked development IDs were scored under the three retained boundaries. NLL shifts show that the representation changes are tied to the task sequence: task565 and task1603 deteriorate after later classifier stages, task1501 sharply improves after its own stage and partially regresses after task1384, while task1384 improves across the later boundaries.

| Task | After task1603 | After task1501 | Final after task1384 |
|---|---:|---:|---:|
| 565 CIRCA | 2.690 | 4.129 | 5.108 |
| 1603 SMCalFlow | 1.071 | 1.991 | 2.522 |
| 1501 DSTC3 | 6.327 | 0.099 | 0.230 |
| 1384 Deal-or-No-Deal | 5.801 | 0.293 | 0.072 |

See [checkpoint_trajectory_nll.png](../figures/phase6a/checkpoint_trajectory_nll.png) and the per-example [checkpoint_trajectory.jsonl](../results/phase6a/checkpoint_trajectory.jsonl).

### Yes/No margin and output bias

Single-layer activation swaps shift the first-token log P(Yes) − log P(No) margin in different directions across tasks and components; there is no shared layer at which both classifiers acquire the same label bias. These are teacher-forced first-token margin changes, while the locked autoregressive checks below give the actual output shares.

- **565 CIRCA**: baseline emitted exact Yes/No strings for 100.0% of outputs (Yes 100.0%, No 0.0%) at mean length 2.00. The selected activation yielded exact Yes/No for 0.0% at length 7.48; it began producing longer text, while ROUGE-L remained 0.106.
- **1603 SMCalFlow**: baseline emitted exact Yes/No strings for 100.0% of outputs (Yes 92.0%, No 8.0%) at mean length 2.00. The selected activation yielded exact Yes/No for 0.0% at length 16.60; it began producing longer text, while ROUGE-L remained 0.272.
- **1501 DSTC3**: the locked Yes-minus-No first-token margin changed -0.195 → -1.645; exact output shares changed Yes 56.0%/44.0% → Yes 44.0%/No 56.0%, with accuracy 0.64 → 0.76.
- **1384 Deal-or-No-Deal**: the locked Yes-minus-No first-token margin changed 1.055 → 1.695; exact output shares changed Yes 84.0%/16.0% → Yes 84.0%/No 16.0%, with accuracy 0.96 → 0.96.

The per-layer margin shifts appear in [binary_margin_layer_shift.png](../figures/phase6a/binary_margin_layer_shift.png).

The late-layer single-component heatmap is in [activation_single_layer_heatmap.png](../figures/phase6a/activation_single_layer_heatmap.png). The rollback-size tradeoff is in [parameter_rollback_tradeoff.png](../figures/phase6a/parameter_rollback_tradeoff.png). Locked generation/classification outcomes are in [locked_confirmation_metrics.png](../figures/phase6a/locked_confirmation_metrics.png), and same-location Attention/MLP interactions are in [attention_mlp_synergy.png](../figures/phase6a/attention_mlp_synergy.png).

## Interpretation limits

The 10-example screens search hundreds of conditions, so winner estimates are selection-biased; the 25-example locked development slice reduces but does not remove that risk. Bootstrap intervals quantify ID-level variability conditional on this fixed task stream and checkpoints, not across seeds or datasets. Activation substitution is a causal intervention on the measured prompt/reference trajectory; autoregressive results are stronger behavioral evidence, but a single fixed stream does not establish a general architecture benefit. Parameter rollback changes large groups of weights and may restore multiple correlated functions at once. The large global rollback for old generation tasks is not evidence that a small module alone is sufficient.

No held-out test conclusions are made. Before a formal architecture decision, repeat the locked analysis with additional seeds/orderings and evaluate once on a test split under a frozen plan.


## Projection-level parameter rollback

The expanded screen rolled back each Attention Q/K/V/O and MLP gate/up/down projection separately at every layer and globally (203 projection groups per task, within a 560-candidate parameter screen). The strongest individual projection candidates are screening results from 10 dev examples, not locked confirmations.

| Task | Best individual projection rollback | NLL gain | Parameters | Best scanned group | NLL gain |
|---|---|---:|---:|---|---:|
| 565 CIRCA | `mlp_down_proj_global` | 0.907 | 14.777% | `attention_plus_mlp_global` | 2.403 |
| 1603 SMCalFlow | `mlp_down_proj_global` | 0.546 | 14.777% | `attention_plus_mlp_global` | 1.318 |
| 1501 DSTC3 | `attention_o_proj_global` | 0.027 | 9.852% | `mlp_group_28_00_27` | 0.049 |
| 1384 Deal-or-No-Deal | `attention_o_proj_layer_16` | 0.005 | 0.352% | `attention_group_14_05_18` | 0.011 |

## Cross-task transfer and retention

Each task's selected activation and parameter condition was applied to both tasks of the other task type using the same 25 locked dev IDs. Negative Δ token NLL means lower NLL than the target task's final-model baseline; a negative primary-metric change means lower ROUGE-L or classifier accuracy.

| Source task | Target task | Intervention | Δ token NLL | Primary metric, baseline → condition | Mean output tokens | Exact Yes/No outputs |
|---|---|---|---:|---:|---:|---:|
| 565 CIRCA | 1501 DSTC3 | activation | +3.223 | 0.640 → 0.200 (-0.440) | 14.6 | 0% |
| 565 CIRCA | 1384 Deal-or-No-Deal | activation | +2.611 | 0.960 → 0.760 (-0.200) | 7.6 | 0% |
| 1603 SMCalFlow | 1501 DSTC3 | activation | +5.912 | 0.640 → 0.200 (-0.440) | 19.0 | 0% |
| 1603 SMCalFlow | 1384 Deal-or-No-Deal | activation | +5.529 | 0.960 → 0.000 (-0.960) | 13.2 | 0% |
| 1501 DSTC3 | 565 CIRCA | activation | -0.477 | 0.000 → 0.000 (+0.000) | 2.0 | — |
| 1501 DSTC3 | 1603 SMCalFlow | activation | -0.187 | 0.014 → 0.030 (+0.017) | 2.0 | — |
| 1384 Deal-or-No-Deal | 565 CIRCA | activation | -1.085 | 0.000 → 0.000 (+0.000) | 2.0 | — |
| 1384 Deal-or-No-Deal | 1603 SMCalFlow | activation | -0.535 | 0.014 → 0.030 (+0.017) | 2.0 | — |
| 565 CIRCA | 1501 DSTC3 | parameter rollback | +5.839 | 0.640 → 0.200 (-0.440) | 20.0 | 0% |
| 565 CIRCA | 1384 Deal-or-No-Deal | parameter rollback | +5.462 | 0.960 → 0.000 (-0.960) | 13.2 | 0% |
| 1603 SMCalFlow | 1501 DSTC3 | parameter rollback | +5.839 | 0.640 → 0.200 (-0.440) | 20.0 | 0% |
| 1603 SMCalFlow | 1384 Deal-or-No-Deal | parameter rollback | +5.462 | 0.960 → 0.000 (-0.960) | 13.2 | 0% |
| 1501 DSTC3 | 565 CIRCA | parameter rollback | -0.538 | 0.000 → 0.000 (+0.000) | 2.0 | — |
| 1501 DSTC3 | 1603 SMCalFlow | parameter rollback | -0.282 | 0.014 → 0.027 (+0.014) | 2.0 | — |
| 1384 Deal-or-No-Deal | 565 CIRCA | parameter rollback | -0.214 | 0.000 → 0.000 (+0.000) | 2.0 | — |
| 1384 Deal-or-No-Deal | 1603 SMCalFlow | parameter rollback | -0.109 | 0.014 → 0.004 (-0.010) | 2.0 | — |

Across generation-to-classification transfers, accuracy changed from -0.96 to -0.20, while token-NLL change ranged from +2.611 to +5.912. Classifier-to-generation transfers changed ROUGE-L from -0.010 to +0.017; generated lengths ranged from 2.0 to 2.0 tokens. These selected restoration conditions are not task-agnostic: generation-selected restorations can damage classification, and classifier-selected conditions do not recover normal free-form generation. See [cross-task transfer heatmap](../figures/phase6a/cross_task_transfer_heatmap.png) and [per-example results](../results/phase6a/cross_task_transfer.jsonl).
