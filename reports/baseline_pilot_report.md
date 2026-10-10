# Continual Instruction-Tuning Baseline Pilot

**Run date:** 2026-10-09  
**Status:** bounded pipeline pilot completed; not a full benchmark

## Training run

- Local Qwen3-0.6B, RTX 4090 24GB, full parameter updates (596,049,920 trainable parameters).
- Four tasks followed CITB order1 positions 1, 2, 6, and 11. The pilot included one task with only 58 training examples; it will be replaced before the formal baseline.
- 20 optimizer steps per task, micro-batch 1, gradient accumulation 4, constant LR 2e-5, 512 maximum tokens, FP32 model weights with CUDA BF16 autocast, gradient checkpointing.
- Training took 59.9 seconds total; peak CUDA allocated memory during training was 10.86 GiB. No OOM or automatic reduction occurred.
- The original exact-match evaluation is superseded by the post-hoc full matrix in the evaluation audit. The corrected audit re-evaluated all four tasks at all five checkpoints using 1,000 predictions and sampled 15 examples for each of the 20 task-stage cells.

## Corrected pilot results

See [pilot_evaluation_audit.md](pilot_evaluation_audit.md) for the full 5×4 matrix, classification Accuracy and Macro-F1, generation ROUGE-L / Token-F1 / Exact Match, answer-parse failures, EOS termination, and per-sample examples. The main correction is task611: the base model often produced a full option string, while the reference was only a choice letter. Parsing the leading option yields 0.48 accuracy instead of the original false 0.00. For task1590, normalized Exact Match is 0.00 at every stage, but ROUGE-L and Token-F1 now provide generation-overlap measures.

The corrected baseline primary scores are 0.42 accuracy for task848, 0.48 accuracy for task611, 0.0627 ROUGE-L for task1590, and 0.18 accuracy for task1500. After the full sequence the corresponding scores are 0.38, 0.00, 0.0058, and 0.82.

The audit also found the pilot's tail truncation removed the task definition for 64/500 task848 training examples and 160/500 task1500 training examples. At test time it removed the definition prefix in 7/50 task848 and 14/50 task1500 examples. The official instructions must be preserved while truncating only long input context in the formal runner.

## Metric definitions

Classification outputs use anchored task-specific parsing, Accuracy, and Macro-F1 over fixed class labels. Task611 is a multiple-choice classification task despite CITB's broader Dialogue Generation category. Task1590 uses ROUGE-L F1 as primary, Token-F1 and normalized Exact Match as auxiliaries.

Raw classification and generation scores are not averaged together. Cross-task AAA and Average Accuracy use baseline-headroom-normalized primary-score gain, (stage score - base score) / max(1 - base score, 1e-8). AAA averages seen tasks; Average Accuracy averages all evaluated tasks. Forgetting is the average drop from each prior task's best earlier post-training normalized gain to its current normalized gain.

## Artifacts and commands

- Corrected complete matrix and metrics: D:/Task-Aware-TransNAS/results/baseline_pilot/accuracy_matrix.csv and metrics.json.
- Per-sample audit material: D:/Task-Aware-TransNAS/results/pilot_audit/audit_samples.jsonl (300 rows) and all_predictions.jsonl (1,000 rows).
- Raw saved checkpoints: D:/Task-Aware-TransNAS/checkpoints/baseline_pilot.
- Audit log: D:/Task-Aware-TransNAS/logs/pilot_audit.log.
- Task stream preparation report: reports/data_manifest.json and data_manifest.json.

To repeat the full post-hoc scoring audit, run python .\scripts\audit_pilot.py. To refresh token/truncation diagnostics without generating again, run python .\scripts\audit_pilot.py --refresh-diagnostics-only. Both use local checkpoints and do not train.

The detected prompt truncation is a methodology defect that must be fixed before the formal baseline. Pilot scores remain diagnostic and must not be presented as final study results.
