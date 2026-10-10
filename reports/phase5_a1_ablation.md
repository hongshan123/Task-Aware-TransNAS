# Phase 5 A1 Cross-Head Inference Ablation

## Protocol

For each of the six matched Phase 4B A1/A0 pairs (S1/S2 × seeds 10/20/30), evaluate A1 with its trained gates, the same A1 checkpoint with every `residual_scale` set to exactly zero, and the matching A0 checkpoint. No weights are optimized. Each condition uses dev indices 25–49 (25 examples per task); test data are not read.

`A1 → A1-zero-gate` isolates the direct inference contribution of the added cross-head path while holding the trained A1 weights fixed. `A1-zero-gate → A0` also includes the different training trajectory and parameter values, so it does not isolate the direct path.

## Task metrics (means over six matched runs)

| Task | Variant | ROUGE-L | Token-F1 | Accuracy | Token NLL | Yes share | No share | Yes/No collapse | Mean output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| task565_circa_answer_generation | A1 | 0.0000 | 0.0000 | — | 4.3674 | 0.6400 | 0.3600 | 0.8533 | 2.0000 |
| task565_circa_answer_generation | A1-zero-gate | 0.0000 | 0.0000 | — | 4.3684 | 0.6400 | 0.3600 | 0.8533 | 2.0000 |
| task565_circa_answer_generation | A0 | 0.0000 | 0.0000 | — | 4.6651 | 0.7133 | 0.2867 | 0.9400 | 2.0000 |
| task1603_smcalflow_sentence_generation | A1 | 0.0651 | 0.0651 | — | 2.2402 | 0.5000 | 0.3867 | 0.7133 | 2.6200 |
| task1603_smcalflow_sentence_generation | A1-zero-gate | 0.0651 | 0.0651 | — | 2.2394 | 0.5067 | 0.3800 | 0.7067 | 2.6200 |
| task1603_smcalflow_sentence_generation | A0 | 0.0531 | 0.0531 | — | 2.3697 | 0.6267 | 0.2733 | 0.7333 | 2.4467 |
| task1501_dstc3_answer_generation | A1 | — | — | 0.6267 | 0.2691 | 0.4667 | 0.5333 | 0.7333 | 2.0000 |
| task1501_dstc3_answer_generation | A1-zero-gate | — | — | 0.6467 | 0.2678 | 0.4600 | 0.5400 | 0.7133 | 2.0000 |
| task1501_dstc3_answer_generation | A0 | — | — | 0.6400 | 0.2627 | 0.5200 | 0.4800 | 0.6800 | 2.0000 |
| task1384_deal_or_no_dialog_classification | A1 | — | — | 0.9067 | 0.0907 | 0.8133 | 0.1867 | 0.8133 | 2.0000 |
| task1384_deal_or_no_dialog_classification | A1-zero-gate | — | — | 0.9067 | 0.0913 | 0.8133 | 0.1867 | 0.8133 | 2.0000 |
| task1384_deal_or_no_dialog_classification | A0 | — | — | 0.9133 | 0.1057 | 0.8333 | 0.1667 | 0.8333 | 2.0000 |

## Learned gate and branch magnitudes

| Task | Variant | Mean absolute gate | Attention-context branch RMS | Projected residual contribution RMS |
|---|---|---:|---:|---:|
| task565_circa_answer_generation | A1 | 0.000185965 | 2.89529e-05 | 3.29866e-05 |
| task565_circa_answer_generation | A1-zero-gate | 0 | 0 | 0 |
| task1603_smcalflow_sentence_generation | A1 | 0.000185965 | 3.51811e-05 | 4.01378e-05 |
| task1603_smcalflow_sentence_generation | A1-zero-gate | 0 | 0 | 0 |
| task1501_dstc3_answer_generation | A1 | 0.000185965 | 3.88381e-05 | 4.25527e-05 |
| task1501_dstc3_answer_generation | A1-zero-gate | 0 | 0 | 0 |
| task1384_deal_or_no_dialog_classification | A1 | 0.000185965 | 4.24608e-05 | 4.80495e-05 |
| task1384_deal_or_no_dialog_classification | A1-zero-gate | 0 | 0 | 0 |

## Direct logit and output changes

The logit deltas below compare the exact first-answer-token logits for identical held-out prompts. Prediction-change rate compares full greedy text outputs.

| Sequence/seed | Comparison | Mean abs logit Δ | Mean per-example max abs Δ | Max abs Δ | First-token argmax changes | Greedy text changes |
|---|---|---:|---:|---:|---:|---:|
| S1/seed10 | A1 vs A1-zero-gate | 0.0119196 | 0.0871094 | 0.125 | 0.010 | 0.010 |
| S1/seed10 | A1 vs A0 | 0.240071 | 1.5061 | 3.49219 | 0.050 | 0.040 |
| S1/seed10 | A1-zero-gate vs A0 | 0.240261 | 1.50597 | 3.53516 | 0.040 | 0.050 |
| S1/seed20 | A1 vs A1-zero-gate | 0.0156261 | 0.0941541 | 0.15625 | 0.010 | 0.010 |
| S1/seed20 | A1 vs A0 | 1.04385 | 5.90614 | 8.77539 | 0.310 | 0.310 |
| S1/seed20 | A1-zero-gate vs A0 | 1.04417 | 5.90393 | 8.77539 | 0.300 | 0.300 |
| S1/seed30 | A1 vs A1-zero-gate | 0.0119567 | 0.0799707 | 0.125 | 0.000 | 0.000 |
| S1/seed30 | A1 vs A0 | 0.478356 | 3.17148 | 6.625 | 0.100 | 0.100 |
| S1/seed30 | A1-zero-gate vs A0 | 0.478663 | 3.17365 | 6.625 | 0.100 | 0.100 |
| S2/seed10 | A1 vs A1-zero-gate | 0.0140915 | 0.0898059 | 0.140625 | 0.000 | 0.000 |
| S2/seed10 | A1 vs A0 | 1.0122 | 5.49723 | 7.67188 | 0.080 | 0.090 |
| S2/seed10 | A1-zero-gate vs A0 | 1.01307 | 5.49725 | 7.67188 | 0.080 | 0.090 |
| S2/seed20 | A1 vs A1-zero-gate | 0.0139752 | 0.0868994 | 0.15625 | 0.000 | 0.010 |
| S2/seed20 | A1 vs A0 | 0.951194 | 5.42891 | 15.875 | 0.110 | 0.130 |
| S2/seed20 | A1-zero-gate vs A0 | 0.949735 | 5.42331 | 15.7656 | 0.110 | 0.140 |
| S2/seed30 | A1 vs A1-zero-gate | 0.0181012 | 0.107639 | 0.195312 | 0.020 | 0.010 |
| S2/seed30 | A1 vs A0 | 0.729497 | 4.3302 | 8.36328 | 0.420 | 0.440 |
| S2/seed30 | A1-zero-gate vs A0 | 0.728699 | 4.33128 | 8.42578 | 0.420 | 0.430 |

Across six matched checkpoints, A1-to-zero-gate mean absolute first-answer-token logit difference is 0.01428; mean argmax-change and greedy-text-change rates are 0.667% and 0.667%.

## Paired direct-path metric deltas

Mean paired difference is A1-zero-gate minus A1 over the six matched checkpoints; negative NLL is an improvement.

| Task | Metric | Mean paired delta |
|---|---|---:|
| task565_circa_answer_generation | mean_rouge_l_f1 | 0.00000 |
| task565_circa_answer_generation | mean_token_f1 | 0.00000 |
| task565_circa_answer_generation | teacher_forced_token_nll | 0.00102 |
| task565_circa_answer_generation | yes_no_collapse_rate | 0.00000 |
| task1603_smcalflow_sentence_generation | mean_rouge_l_f1 | 0.00000 |
| task1603_smcalflow_sentence_generation | mean_token_f1 | 0.00000 |
| task1603_smcalflow_sentence_generation | teacher_forced_token_nll | -0.00087 |
| task1603_smcalflow_sentence_generation | yes_no_collapse_rate | -0.00667 |
| task1501_dstc3_answer_generation | accuracy | 0.02000 |
| task1501_dstc3_answer_generation | teacher_forced_token_nll | -0.00128 |
| task1501_dstc3_answer_generation | yes_no_collapse_rate | -0.02000 |
| task1384_deal_or_no_dialog_classification | accuracy | 0.00000 |
| task1384_deal_or_no_dialog_classification | teacher_forced_token_nll | 0.00053 |
| task1384_deal_or_no_dialog_classification | yes_no_collapse_rate | 0.00000 |

## Conclusion

The learned mean absolute gate is about 1.86e-4 and the task-wise projected residual RMS is only about 3.3e-5 to 4.8e-5. Zeroing the branch changes very few held-out predictions and leaves generation ROUGE-L unchanged; task 1501 accuracy is slightly higher with the branch zeroed. This is evidence that the trained A1 predictions do not materially rely on the direct cross-head path. Differences between A1-zero-gate and A0 still include the A1 training trajectory and shared-weight changes, so they cannot be attributed to inference-time cross-head computation.

## Gate and branch traces

Per-layer learned gates, branch RMS, and the projected residual contribution are saved per checkpoint in `results/phase5/a1_ablation/`. A zero-gate run should have zero branch contribution to numerical tolerance. The comparison between A1 and A1-zero-gate is the direct effect; differences between the zeroed A1 model and A0 reflect both learned shared weights and architecture-specific training dynamics.

## Reproduction

```powershell
.\conda-env\python.exe scripts\run_phase5_a1_ablation.py
```
