# Pilot Evaluation Audit

Audit run: 2026-10-09T07:25:20.184602+00:00
Source CITB commit: bf50533b5bced4c388691ecc75e26773da96b3fd
Re-evaluated predictions: 1000 (50 per task per stage)
Saved audit sample: 300 (15 per task per stage)
Runtime: 219.9 seconds; peak CUDA allocated: 2.64 GiB.

## Task formats and scoring decisions

| Task | Official CITB category | Observed reference format | Scoring type |
|---|---|---|---|
| task848_pubmedqa_classification | Intent Identification | binary label 0 / 1 | Accuracy + fixed-label Macro-F1; anchored answer parser |
| task611_mutual_multi_turn_dialogue | Dialogue Generation | choice label A / B / C / D | Accuracy + fixed-label Macro-F1; anchored answer parser |
| task1590_diplomacy_text_generation | Dialogue Generation | free-form next dialogue message | ROUGE-L F1 primary; Token-F1 and normalized Exact Match auxiliary |
| task1500_dstc3_classification | Dialogue State Tracking | Cheap / Moderate / Expensive / Don't Care | Accuracy + fixed-label Macro-F1; anchored answer parser |

The task category alone does not determine the output type: task611 is categorized as Dialogue Generation by CITB, but its instances expect a multiple-choice letter, so it is evaluated as classification after parsing the selected option.

## Pilot scoring defect found and corrected

The previous Exact Match evaluator compared the complete generated string with the canonical reference. On task611, the base model often returned a complete option such as ‘(C) ...’ while the reference was only ‘A’. The old evaluator marked every base prediction wrong (0.00) even when the selected option could be parsed. Task611 after later tasks could also emit prose or a different task's label; these are counted as parser failures rather than guessed from arbitrary letters in the text.

The previous evaluator used exact string match as the sole score for task1590, an open-ended generation task. This is too strict to measure semantic overlap. The pilot has been re-evaluated over the full 50-example test split at all five checkpoints with ROUGE-L F1 and Token-F1 as primary/secondary overlap measures, while retaining Exact Match as auxiliary. No test result was used to select tasks, sequences, or hyperparameters.

## Full task-by-stage matrix

See the companion accuracy_matrix.csv for a wide machine-readable matrix. Raw classification and generation scores are kept separate by metric name.

| Stage | Task | Primary metric | Score | Accuracy | Macro-F1 | ROUGE-L | Token-F1 | Exact Match | Parse success |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | task848_pubmedqa_classification | accuracy | 0.4200 | 0.42 | 0.32983193277310924 |  |  |  | 0.92 |
| baseline | task611_mutual_multi_turn_dialogue | accuracy | 0.4800 | 0.48 | 0.4480654761904762 |  |  |  | 0.94 |
| baseline | task1590_diplomacy_text_generation | rouge_l_f1 | 0.0627 |  |  | 0.06266144058872582 | 0.0820564538657186 | 0.0 |  |
| baseline | task1500_dstc3_classification | accuracy | 0.1800 | 0.18 | 0.08653846153846154 |  |  |  | 0.82 |
| after_1_task848_pubmedqa_classification | task848_pubmedqa_classification | accuracy | 0.5200 | 0.52 | 0.34210526315789475 |  |  |  | 1.0 |
| after_1_task848_pubmedqa_classification | task611_mutual_multi_turn_dialogue | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_1_task848_pubmedqa_classification | task1590_diplomacy_text_generation | rouge_l_f1 | 0.0000 |  |  | 0.0 | 0.0 | 0.0 |  |
| after_1_task848_pubmedqa_classification | task1500_dstc3_classification | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_2_task611_mutual_multi_turn_dialogue | task848_pubmedqa_classification | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_2_task611_mutual_multi_turn_dialogue | task611_mutual_multi_turn_dialogue | accuracy | 0.1000 | 0.1 | 0.05555555555555555 |  |  |  | 1.0 |
| after_2_task611_mutual_multi_turn_dialogue | task1590_diplomacy_text_generation | rouge_l_f1 | 0.0000 |  |  | 0.0 | 0.0 | 0.0 |  |
| after_2_task611_mutual_multi_turn_dialogue | task1500_dstc3_classification | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_3_task1590_diplomacy_text_generation | task848_pubmedqa_classification | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.02 |
| after_3_task1590_diplomacy_text_generation | task611_mutual_multi_turn_dialogue | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_3_task1590_diplomacy_text_generation | task1590_diplomacy_text_generation | rouge_l_f1 | 0.0500 |  |  | 0.05000491768685325 | 0.05433740765752302 | 0.0 |  |
| after_3_task1590_diplomacy_text_generation | task1500_dstc3_classification | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_4_task1500_dstc3_classification | task848_pubmedqa_classification | accuracy | 0.3800 | 0.38 | 0.3717948717948718 |  |  |  | 0.6 |
| after_4_task1500_dstc3_classification | task611_mutual_multi_turn_dialogue | accuracy | 0.0000 | 0.0 | 0.0 |  |  |  | 0.0 |
| after_4_task1500_dstc3_classification | task1590_diplomacy_text_generation | rouge_l_f1 | 0.0058 |  |  | 0.005832752613240419 | 0.005832752613240417 | 0.0 |  |
| after_4_task1500_dstc3_classification | task1500_dstc3_classification | accuracy | 0.8200 | 0.82 | 0.6606625980819529 |  |  |  | 1.0 |

## Answer parsing, format, EOS, and truncation diagnostics

Each per-example audit record contains the original input, all references, visible and raw token-decoded prediction, per-example score, parsed class, parse status, prompt token counts, EOS termination, max-token-cap status, and whether the regenerated visible output matched the stored pilot output.

| Task | Outputs | EOS terminated | Hit 64-token cap | Prompt truncated at 512 | Definition prefix lost | Raw think marker |
|---|---:|---:|---:|---:|---:|---:|
| task848_pubmedqa_classification | 250 | 241 | 9 | 65 | 35 | 0 |
| task611_mutual_multi_turn_dialogue | 250 | 249 | 1 | 10 | 5 | 0 |
| task1590_diplomacy_text_generation | 250 | 246 | 4 | 10 | 5 | 0 |
| task1500_dstc3_classification | 250 | 250 | 0 | 80 | 70 | 0 |

Training-time input diagnostics (the SFT encoder reserves space for the target, so its retained prompt may be shorter than 512 tokens):

| Task | Split | Examples | Prompt truncated | Definition prefix lost | Target capped | Prompt fully dropped |
|---|---|---:|---:|---:|---:|---:|
| task848_pubmedqa_classification | train | 500 | 90 | 64 | 0 | 0 |
| task848_pubmedqa_classification | dev | 50 | 9 | 7 | 0 | 0 |
| task848_pubmedqa_classification | test | 50 | 13 | 7 | 0 | 0 |
| task611_mutual_multi_turn_dialogue | train | 500 | 7 | 3 | 0 | 0 |
| task611_mutual_multi_turn_dialogue | dev | 50 | 2 | 1 | 0 | 0 |
| task611_mutual_multi_turn_dialogue | test | 50 | 2 | 1 | 0 | 0 |
| task1590_diplomacy_text_generation | train | 58 | 2 | 1 | 0 | 0 |
| task1590_diplomacy_text_generation | dev | 50 | 2 | 2 | 0 | 0 |
| task1590_diplomacy_text_generation | test | 50 | 2 | 1 | 0 | 0 |
| task1500_dstc3_classification | train | 500 | 186 | 160 | 0 | 0 |
| task1500_dstc3_classification | dev | 50 | 18 | 15 | 0 | 0 |
| task1500_dstc3_classification | test | 50 | 16 | 14 | 0 | 0 |

Classification parse failures in the 15-row audit samples by stage:

```json
{
  "baseline": {
    "task848_pubmedqa_classification": 1,
    "task611_mutual_multi_turn_dialogue": 1,
    "task1500_dstc3_classification": 2
  },
  "after_1_task848_pubmedqa_classification": {
    "task848_pubmedqa_classification": 0,
    "task611_mutual_multi_turn_dialogue": 15,
    "task1500_dstc3_classification": 15
  },
  "after_2_task611_mutual_multi_turn_dialogue": {
    "task848_pubmedqa_classification": 15,
    "task611_mutual_multi_turn_dialogue": 0,
    "task1500_dstc3_classification": 15
  },
  "after_3_task1590_diplomacy_text_generation": {
    "task848_pubmedqa_classification": 15,
    "task611_mutual_multi_turn_dialogue": 15,
    "task1500_dstc3_classification": 15
  },
  "after_4_task1500_dstc3_classification": {
    "task848_pubmedqa_classification": 4,
    "task611_mutual_multi_turn_dialogue": 15,
    "task1500_dstc3_classification": 0
  }
}
```

Stored-pilot prediction consistency rates for comparable rows are in metrics.json. Chat template generation was rerun with enable_thinking=False; prompt truncation always retained the final 512 prompt tokens, matching the pilot runner. The definition-prefix counts identify cases where tail truncation removed the task instructions; those cases are a prompt construction defect to fix before the formal run. Raw generated think markers are counted separately from template-provided assistant prefixes.

## Cross-task Average Accuracy, AAA, and forgetting

Primary task scores have different interpretations. They are never averaged raw across task types. Per-task normalized gain is (stage primary score - base primary score) / max(1 - base primary score, 1e-8). The all-task Average Accuracy is the mean normalized gain over all four tasks. AAA is the mean normalized gain over tasks seen through that stage. Forgetting is the mean, over prior tasks, of the best earlier post-training normalized gain minus the current normalized gain. Raw averages remain grouped by metric in metrics.json.

| Stage | Normalized average accuracy (all tasks) | AAA (seen tasks) | Normalized average forgetting |
|---|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.0000 |
| after_1_task848_pubmedqa_classification | -0.2593 | 0.1724 | 0.0000 |
| after_2_task611_mutual_multi_turn_dialogue | -0.4353 | -0.7275 | 0.8966 |
| after_3_task1590_diplomacy_text_generation | -0.4701 | -0.5536 | 0.5444 |
| after_4_task1500_dstc3_classification | -0.0680 | -0.0680 | 0.1603 |

## Audit artifacts and next step

- Full re-evaluation predictions: D:\Task-Aware-TransNAS\results\pilot_audit\all_predictions.jsonl
- Deterministic 15-per-task/per-stage input/reference/prediction/score sample: D:\Task-Aware-TransNAS\results\pilot_audit\audit_samples.jsonl
- Complete task-by-stage metric matrix: D:\Task-Aware-TransNAS\results\pilot_audit\metrics.json and D:\Task-Aware-TransNAS\results\pilot_audit\accuracy_matrix.csv

The original pilot's task611 exact-match score is invalid as a choice-classification measure, and task1590 exact match alone is inadequate as a generation measure. Use the corrected metrics above for pilot interpretation. Only after this audit is complete should development-set comparisons be used to choose the new balanced tasks and formal-run hyperparameters; the test split must remain excluded from those choices.
