# Phase 6A: Architecture Implications

## Decision from the measured interventions

The evidence supports testing **task-conditional late residual pathways** as the next small architecture pilot. It does not justify broad NAS yet. In particular, a fixed policy that restores Attention and MLP together is not supported across tasks.

## Task-specific evidence

- **565 CIRCA**: selected activation condition `block_layer_25`. On the locked same-location swap, NLL improvements were Attention 0.552, MLP 0.143, Attention+MLP 0.651, and whole block 2.293. Parameter screening selected `attention_plus_mlp_global` (73.9% of model parameters).
- **1603 SMCalFlow**: selected activation condition `block_layer_27`. On the locked same-location swap, NLL improvements were Attention 0.166, MLP -0.252, Attention+MLP -0.176, and whole block 1.549. Parameter screening selected `attention_plus_mlp_global` (73.9% of model parameters).
- **1501 DSTC3**: selected activation condition `attention_group_14_13_26`. On the locked same-location swap, NLL improvements were Attention 0.097, MLP 0.096, Attention+MLP 0.123, and whole block 0.126. Parameter screening selected `mlp_group_28_00_27` (44.3% of model parameters).
- **1384 Deal-or-No-Deal**: selected activation condition `mlp_group_14_14_27`. On the locked same-location swap, NLL improvements were Attention -0.047, MLP 0.016, Attention+MLP -0.154, and whole block -0.210. Parameter screening selected `attention_group_14_05_18` (14.8% of model parameters).

The locked same-location interaction comparison shows why a single global module rule is risky:

- task565 and task1603 gain much more from whole-block output replacement than from Attention+MLP replacement. The residual stream carries a useful integrated state beyond the sum of restoring just those two module outputs.
- task1501 benefits from a broad Attention activation patch; joint Attention+MLP and block output are comparable or slightly stronger on its locked NLL. A classifier path may need both branches to preserve label evidence.
- task1384 benefits from MLP output replacement alone, while the joint Attention+MLP patch is harmful on NLL. Adding a branch can undo the useful MLP correction.

The parameter screen gives a second, distinct result. Rollback toward the task1603 boundary improves old generation NLL, but the best group contains roughly 74% of all model parameters. On task1501 a full MLP rollback (44.3%) improves accuracy and NLL; for task1384 the best screened rollback improves neither locked accuracy nor NLL relative to baseline. Matched-L2 random directions do not reproduce the old-generation or task1501 effects. This indicates checkpoint-specific directions matter, while the parameter groups remain too large to recommend as selective storage or routing units.

The checkpoint trajectory adds a task-stage effect: generation NLL rises after the classifier tasks; task1501 loses part of its boundary likelihood after the last task; task1384 obtains its largest improvement at its own final stage. This is consistent with interference along this one sequence, though NLL alone does not measure decoded task quality.

## Recommended follow-up pilot

1. Compare a shared 28-layer trunk with a small task-conditioned late-block adapter/router. Include a residual-preserving block route and separate Attention/MLP routes as distinct arms.
2. Keep task565 and task1603 generation plus both classifier tasks in every evaluation batch; report old-task retention and new-task accuracy together.
3. Freeze routing from train/dev only, repeat at least three seeds and one alternate task order, and include no-op, random-layer, random-direction, and parameter-count-matched controls.
4. Limit the added state to a small fraction of the 596.05M-parameter model. Use the current 73.9% global rollback only as a recovery upper bound, not as a proposed deployment design.
5. After selection is frozen, evaluate a test set once and report ROUGE-L/Token-F1, classifier accuracy/macro-F1, exact Yes/No proportions, invalid-format rate, and generated length.

## What this phase does not establish

The study uses one sequence, one seed, one Qwen3-0.6B checkpoint, and small development slices selected from a larger screen. It identifies causal locations for these prompts and this trajectory; it does not prove that a router, adapter, or NAS-selected topology will improve continual learning generally. The causal analyses used no optimizer steps. A separate exact replay of the original task1501 boundary supplied missing historical weights; no new task sequence or NAS search was run.

Raw results: [results/phase6a](../results/phase6a/). Main report: [phase6a_component_localization.md](phase6a_component_localization.md).


## Cross-task retention and projection screen

Transferring generation-selected activations or weights onto the classifiers reduced accuracy and raised NLL in the locked development evaluation. Transferring classifier-selected conditions to the generation tasks left output lengths at 2.0–2.0 tokens and produced only -0.010–+0.017 ROUGE-L change. The screen's strongest projection also varied by task; no common Q/K/V/O or gate/up/down rollback is supported across all tasks. The next pilot should use task-conditioned late routes and evaluate retention and incoming-task quality together.
