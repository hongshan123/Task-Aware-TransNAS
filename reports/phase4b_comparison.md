# Phase 4B Controlled Attention Comparison

## Scope and completion

All 18 planned runs completed: three architectures x two orders x three seeds. Every run used the same local Qwen3-0.6B source weights, dev-only evaluation, and a final checkpoint reload audit. The test split was not loaded.

All task-by-stage metrics, predictions, teacher-forced NLL, generation behavior, gate traces, timing, VRAM, parameter counts, and reload records are retained under `D:/Task-Aware-TransNAS/results/phase4b/`.

## Main outcomes

Values are means +/- sample SD across six runs (three seeds in each sequence). Accuracy and ROUGE-L stay in separate columns.

| Architecture | Final generation ROUGE-L | Final classification accuracy | Mean normalized acquisition gain | Final cross-stage AAA | Nonnegative forgetting | Backward transfer | Peak allocated VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.027 +/- 0.032 | 0.770 +/- 0.089 | -0.011 +/- 0.168 | -0.133 +/- 0.146 | 0.289 +/- 0.067 | -0.248 +/- 0.120 | 10.851 +/- 0.003 |
| A1 | 0.035 +/- 0.034 | 0.773 +/- 0.055 | 0.078 +/- 0.082 | -0.068 +/- 0.039 | 0.341 +/- 0.155 | -0.331 +/- 0.142 | 10.838 +/- 0.002 |
| A1-control | 0.027 +/- 0.022 | 0.795 +/- 0.058 | 0.002 +/- 0.155 | -0.114 +/- 0.068 | 0.339 +/- 0.139 | -0.168 +/- 0.322 | 10.838 +/- 0.002 |

## Paired architecture effects

Effects are candidate minus reference. Overall intervals first average the S1 and S2 paired deltas within each seed, then summarize across three seeds (nominal paired t 95% interval, df=2). Order-specific intervals use three paired seeds per order. These are descriptive intervals for a small exploratory study.

### A1 - A0

| Metric | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---:|---:|---:|---:|
| `acquisition_normalized_mean` | 0.088 (SD 0.028; 95% CI 0.019 to 0.158) | -0.028 | 0.205 | 0.233 (-0.254 to 0.720) |
| `final_generation_rouge_l_mean` | 0.008 (SD 0.005; 95% CI -0.004 to 0.020) | 0.001 | 0.015 | 0.013 (-0.006 to 0.032) |
| `final_classification_accuracy_mean` | 0.003 (SD 0.028; 95% CI -0.065 to 0.072) | -0.077 | 0.083 | 0.160 (-0.192 to 0.512) |
| `cross_stage_aaa_normalized_final` | 0.065 (SD 0.006; 95% CI 0.050 to 0.080) | -0.056 | 0.186 | 0.242 (-0.045 to 0.529) |
| `average_forgetting_nonnegative_normalized` | 0.051 (SD 0.094; 95% CI -0.182 to 0.285) | 0.082 | 0.021 | -0.061 (-0.375 to 0.254) |
| `backward_transfer_normalized` | -0.084 (SD 0.088; 95% CI -0.302 to 0.135) | -0.156 | -0.012 | 0.144 (-0.431 to 0.719) |
| `final_generation_token_nll_mean` | -0.195 (SD 0.172; 95% CI -0.623 to 0.232) | -0.022 | -0.369 | -0.347 (-2.609 to 1.915) |
| `final_classification_token_nll_mean` | -0.007 (SD 0.043; 95% CI -0.113 to 0.099) | 0.036 | -0.050 | -0.087 (-0.239 to 0.065) |

### A1-control - A0

| Metric | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---:|---:|---:|---:|
| `acquisition_normalized_mean` | 0.013 (SD 0.090; 95% CI -0.212 to 0.237) | -0.173 | 0.199 | 0.372 (0.040 to 0.705) |
| `final_generation_rouge_l_mean` | -0.000 (SD 0.015; 95% CI -0.039 to 0.038) | -0.000 | -0.001 | -0.001 (-0.077 to 0.075) |
| `final_classification_accuracy_mean` | 0.025 (SD 0.041; 95% CI -0.077 to 0.127) | -0.020 | 0.070 | 0.090 (0.024 to 0.156) |
| `cross_stage_aaa_normalized_final` | 0.019 (SD 0.014; 95% CI -0.015 to 0.054) | -0.072 | 0.111 | 0.183 (0.076 to 0.290) |
| `average_forgetting_nonnegative_normalized` | 0.050 (SD 0.098; 95% CI -0.193 to 0.292) | 0.010 | 0.090 | 0.080 (-0.243 to 0.403) |
| `backward_transfer_normalized` | 0.080 (SD 0.221; 95% CI -0.468 to 0.628) | 0.194 | -0.034 | -0.228 (-0.876 to 0.419) |
| `final_generation_token_nll_mean` | -0.092 (SD 0.127; 95% CI -0.406 to 0.223) | 0.011 | -0.195 | -0.206 (-1.907 to 1.495) |
| `final_classification_token_nll_mean` | -0.013 (SD 0.009; 95% CI -0.036 to 0.010) | 0.024 | -0.050 | -0.074 (-0.240 to 0.093) |

### A1 - A1-control

| Metric | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---:|---:|---:|---:|
| `acquisition_normalized_mean` | 0.076 (SD 0.099; 95% CI -0.170 to 0.321) | 0.145 | 0.006 | -0.139 (-0.852 to 0.574) |
| `final_generation_rouge_l_mean` | 0.008 (SD 0.011; 95% CI -0.019 to 0.035) | 0.001 | 0.015 | 0.014 (-0.044 to 0.072) |
| `final_classification_accuracy_mean` | -0.022 (SD 0.052; 95% CI -0.151 to 0.108) | -0.057 | 0.013 | 0.070 (-0.223 to 0.363) |
| `cross_stage_aaa_normalized_final` | 0.046 (SD 0.009; 95% CI 0.022 to 0.069) | 0.016 | 0.075 | 0.059 (-0.308 to 0.426) |
| `average_forgetting_nonnegative_normalized` | 0.002 (SD 0.062; 95% CI -0.153 to 0.157) | 0.072 | -0.069 | -0.141 (-0.304 to 0.022) |
| `backward_transfer_normalized` | -0.164 (SD 0.209; 95% CI -0.684 to 0.357) | -0.350 | 0.022 | 0.372 (-0.159 to 0.903) |
| `final_generation_token_nll_mean` | -0.104 (SD 0.167; 95% CI -0.519 to 0.312) | -0.033 | -0.174 | -0.141 (-0.734 to 0.451) |
| `final_classification_token_nll_mean` | 0.006 (SD 0.035; 95% CI -0.080 to 0.093) | 0.013 | -0.000 | -0.013 (-0.124 to 0.098) |

## Per-task final metrics and acquisition

Raw acquisition is reported in the task's own primary-metric units. Normalized gain divides by the task's baseline headroom.

| Architecture | Task | Final primary score (mean +/- SD) | Normalized acquisition gain | Raw acquisition | Token NLL | Exact Yes share | Exact No share | Mean generated tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | `task565_circa_answer_generation` | 0.005 +/- 0.006 | 0.023 +/- 0.026 | 0.019 +/- 0.021 | 4.844 +/- 0.582 | 0.710 +/- 0.429 | 0.280 +/- 0.437 | 2.040 +/- 0.079 |
| A0 | `task1603_smcalflow_sentence_generation` | 0.049 +/- 0.060 | 0.226 +/- 0.054 | 0.190 +/- 0.045 | 1.964 +/- 0.345 | 0.710 +/- 0.216 | 0.207 +/- 0.228 | 2.403 +/- 0.601 |
| A0 | `task1501_dstc3_answer_generation` | 0.660 +/- 0.167 | -0.292 +/- 0.560 | -0.070 +/- 0.134 | 0.246 +/- 0.107 | 0.520 +/- 0.236 | 0.480 +/- 0.236 | 2.000 +/- 0.000 |
| A0 | `task1384_deal_or_no_dialog_classification` | 0.880 +/- 0.031 | 0.000 +/- 0.258 | 0.000 +/- 0.031 | 0.124 +/- 0.024 | 0.807 +/- 0.052 | 0.193 +/- 0.052 | 2.000 +/- 0.000 |
| A1 | `task565_circa_answer_generation` | 0.005 +/- 0.005 | 0.014 +/- 0.029 | 0.011 +/- 0.023 | 4.565 +/- 0.770 | 0.610 +/- 0.363 | 0.380 +/- 0.366 | 2.033 +/- 0.048 |
| A1 | `task1603_smcalflow_sentence_generation` | 0.066 +/- 0.063 | 0.255 +/- 0.041 | 0.215 +/- 0.034 | 1.853 +/- 0.476 | 0.513 +/- 0.301 | 0.373 +/- 0.305 | 2.757 +/- 0.927 |
| A1 | `task1501_dstc3_answer_generation` | 0.653 +/- 0.121 | -0.069 +/- 0.359 | -0.017 +/- 0.086 | 0.248 +/- 0.089 | 0.433 +/- 0.280 | 0.567 +/- 0.280 | 2.000 +/- 0.000 |
| A1 | `task1384_deal_or_no_dialog_classification` | 0.893 +/- 0.030 | 0.111 +/- 0.251 | 0.013 +/- 0.030 | 0.108 +/- 0.021 | 0.787 +/- 0.045 | 0.213 +/- 0.045 | 2.000 +/- 0.000 |
| A1-control | `task565_circa_answer_generation` | 0.005 +/- 0.006 | 0.031 +/- 0.028 | 0.025 +/- 0.022 | 4.730 +/- 0.669 | 0.420 +/- 0.384 | 0.570 +/- 0.382 | 2.033 +/- 0.045 |
| A1-control | `task1603_smcalflow_sentence_generation` | 0.048 +/- 0.039 | 0.227 +/- 0.056 | 0.191 +/- 0.048 | 1.895 +/- 0.388 | 0.460 +/- 0.344 | 0.433 +/- 0.298 | 2.423 +/- 0.481 |
| A1-control | `task1501_dstc3_answer_generation` | 0.690 +/- 0.123 | -0.417 +/- 0.624 | -0.100 +/- 0.150 | 0.224 +/- 0.064 | 0.483 +/- 0.184 | 0.517 +/- 0.184 | 2.000 +/- 0.000 |
| A1-control | `task1384_deal_or_no_dialog_classification` | 0.900 +/- 0.028 | 0.167 +/- 0.236 | 0.020 +/- 0.028 | 0.120 +/- 0.033 | 0.753 +/- 0.079 | 0.247 +/- 0.079 | 2.000 +/- 0.000 |

### Task-specific paired effects: A1 - A0

| Task | Outcome | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---|---:|---:|---:|---:|
| `task565_circa_answer_generation` | `final_score` | -0.000 (SD 0.002; 95% CI -0.004 to 0.004) | 0.000 | -0.001 | -0.001 (-0.009 to 0.007) |
| `task565_circa_answer_generation` | `acquisition_normalized` | -0.009 (SD 0.013; 95% CI -0.042 to 0.024) | -0.009 | -0.009 | 0.000 (0.000 to 0.000) |
| `task565_circa_answer_generation` | `acquisition_raw` | -0.007 (SD 0.011; 95% CI -0.034 to 0.019) | -0.007 | -0.007 | 0.000 (0.000 to 0.000) |
| `task1603_smcalflow_sentence_generation` | `final_score` | 0.016 (SD 0.011; 95% CI -0.010 to 0.043) | 0.003 | 0.030 | 0.027 (-0.015 to 0.070) |
| `task1603_smcalflow_sentence_generation` | `acquisition_normalized` | 0.029 (SD 0.032; 95% CI -0.049 to 0.108) | 0.035 | 0.024 | -0.011 (-0.232 to 0.209) |
| `task1603_smcalflow_sentence_generation` | `acquisition_raw` | 0.025 (SD 0.027; 95% CI -0.042 to 0.091) | 0.029 | 0.020 | -0.010 (-0.195 to 0.176) |
| `task1501_dstc3_answer_generation` | `final_score` | -0.007 (SD 0.061; 95% CI -0.158 to 0.145) | -0.167 | 0.153 | 0.320 (-0.233 to 0.873) |
| `task1501_dstc3_answer_generation` | `acquisition_normalized` | 0.222 (SD 0.134; 95% CI -0.111 to 0.555) | -0.250 | 0.694 | 0.944 (0.514 to 1.375) |
| `task1501_dstc3_answer_generation` | `acquisition_raw` | 0.053 (SD 0.032; 95% CI -0.027 to 0.133) | -0.060 | 0.167 | 0.227 (0.123 to 0.330) |
| `task1384_deal_or_no_dialog_classification` | `final_score` | 0.013 (SD 0.025; 95% CI -0.049 to 0.076) | 0.013 | 0.013 | 0.000 (-0.179 to 0.179) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_normalized` | 0.111 (SD 0.210; 95% CI -0.410 to 0.632) | 0.111 | 0.111 | 0.000 (-1.493 to 1.493) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_raw` | 0.013 (SD 0.025; 95% CI -0.049 to 0.076) | 0.013 | 0.013 | 0.000 (-0.179 to 0.179) |

### Task-specific paired effects: A1-control - A0

| Task | Outcome | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---|---:|---:|---:|---:|
| `task565_circa_answer_generation` | `final_score` | 0.000 (SD 0.000; 95% CI -0.000 to 0.001) | 0.000 | 0.000 | 0.000 (-0.000 to 0.001) |
| `task565_circa_answer_generation` | `acquisition_normalized` | 0.008 (SD 0.013; 95% CI -0.023 to 0.039) | 0.008 | 0.008 | 0.000 (0.000 to 0.000) |
| `task565_circa_answer_generation` | `acquisition_raw` | 0.007 (SD 0.010; 95% CI -0.018 to 0.032) | 0.007 | 0.007 | 0.000 (0.000 to 0.000) |
| `task1603_smcalflow_sentence_generation` | `final_score` | -0.001 (SD 0.031; 95% CI -0.077 to 0.075) | -0.000 | -0.002 | -0.002 (-0.153 to 0.150) |
| `task1603_smcalflow_sentence_generation` | `acquisition_normalized` | 0.001 (SD 0.052; 95% CI -0.128 to 0.130) | -0.008 | 0.009 | 0.017 (-0.205 to 0.238) |
| `task1603_smcalflow_sentence_generation` | `acquisition_raw` | 0.001 (SD 0.044; 95% CI -0.108 to 0.109) | -0.006 | 0.008 | 0.014 (-0.172 to 0.201) |
| `task1501_dstc3_answer_generation` | `final_score` | 0.030 (SD 0.066; 95% CI -0.133 to 0.193) | -0.053 | 0.113 | 0.167 (0.063 to 0.270) |
| `task1501_dstc3_answer_generation` | `acquisition_normalized` | -0.125 (SD 0.470; 95% CI -1.291 to 1.041) | -0.806 | 0.556 | 1.361 (-0.193 to 2.915) |
| `task1501_dstc3_answer_generation` | `acquisition_raw` | -0.030 (SD 0.113; 95% CI -0.310 to 0.250) | -0.193 | 0.133 | 0.327 (-0.046 to 0.700) |
| `task1384_deal_or_no_dialog_classification` | `final_score` | 0.020 (SD 0.017; 95% CI -0.023 to 0.063) | 0.013 | 0.027 | 0.013 (-0.063 to 0.089) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_normalized` | 0.167 (SD 0.144; 95% CI -0.192 to 0.525) | 0.111 | 0.222 | 0.111 (-0.521 to 0.744) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_raw` | 0.020 (SD 0.017; 95% CI -0.023 to 0.063) | 0.013 | 0.027 | 0.013 (-0.063 to 0.089) |

### Task-specific paired effects: A1 - A1-control

| Task | Outcome | Overall paired delta (SD; 95% CI) | S1 mean delta | S2 mean delta | Order interaction S2-S1 (95% CI) |
|---|---|---:|---:|---:|---:|
| `task565_circa_answer_generation` | `final_score` | -0.001 (SD 0.002; 95% CI -0.005 to 0.004) | 0.000 | -0.001 | -0.001 (-0.009 to 0.007) |
| `task565_circa_answer_generation` | `acquisition_normalized` | -0.017 (SD 0.026; 95% CI -0.081 to 0.046) | -0.017 | -0.017 | 0.000 (0.000 to 0.000) |
| `task565_circa_answer_generation` | `acquisition_raw` | -0.014 (SD 0.021; 95% CI -0.065 to 0.037) | -0.014 | -0.014 | 0.000 (0.000 to 0.000) |
| `task1603_smcalflow_sentence_generation` | `final_score` | 0.017 (SD 0.020; 95% CI -0.033 to 0.068) | 0.003 | 0.032 | 0.029 (-0.080 to 0.138) |
| `task1603_smcalflow_sentence_generation` | `acquisition_normalized` | 0.029 (SD 0.025; 95% CI -0.035 to 0.092) | 0.043 | 0.014 | -0.028 (-0.149 to 0.092) |
| `task1603_smcalflow_sentence_generation` | `acquisition_raw` | 0.024 (SD 0.021; 95% CI -0.029 to 0.077) | 0.036 | 0.012 | -0.024 (-0.125 to 0.078) |
| `task1501_dstc3_answer_generation` | `final_score` | -0.037 (SD 0.108; 95% CI -0.305 to 0.231) | -0.113 | 0.040 | 0.153 (-0.369 to 0.675) |
| `task1501_dstc3_answer_generation` | `acquisition_normalized` | 0.347 (SD 0.339; 95% CI -0.496 to 1.190) | 0.556 | 0.139 | -0.417 (-2.111 to 1.278) |
| `task1501_dstc3_answer_generation` | `acquisition_raw` | 0.083 (SD 0.081; 95% CI -0.119 to 0.286) | 0.133 | 0.033 | -0.100 (-0.507 to 0.307) |
| `task1384_deal_or_no_dialog_classification` | `final_score` | -0.007 (SD 0.012; 95% CI -0.035 to 0.022) | 0.000 | -0.013 | -0.013 (-0.165 to 0.138) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_normalized` | -0.056 (SD 0.096; 95% CI -0.295 to 0.183) | 0.000 | -0.111 | -0.111 (-1.376 to 1.154) |
| `task1384_deal_or_no_dialog_classification` | `acquisition_raw` | -0.007 (SD 0.012; 95% CI -0.035 to 0.022) | 0.000 | -0.013 | -0.013 (-0.165 to 0.138) |

## Gate and branch activity

| Architecture | Runs | Optimizer updates traced | Branch-active layer-step share | Mean absolute gate | Mean clipped gate-gradient L2 | Mean branch RMS | Maximum final absolute gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 6 | 2400 | 99.75% | 0.0001204 | 0.0002762 | 0.0000540 | 0.0005810 |
| A1-control | 6 | 2400 | 99.75% | 0.0001285 | 0.0004311 | 0.0000449 | 0.0010644 |

## Answers to the four study questions

1. **New-task acquisition.** A1-A0 mean normalized acquisition delta: 0.088 (95% CI 0.019 to 0.158); A1-A1-control: 0.076 (-0.170 to 0.321). Task-specific acquisition deltas are listed above.
2. **Retention.** A1-A0 change in final nonnegative forgetting: 0.051 (95% CI -0.182 to 0.285); A1-A1-control: 0.002 (-0.153 to 0.157). Lower forgetting is better; backward transfer and final task scores are also shown.
3. **Order dependence.** S2-S1 interaction in the A1-A0 final cross-stage AAA effect: 0.242 (95% CI -0.045 to 0.529). See the per-order effects and task-specific order interactions above.
4. **Gate participation.** Branch-active layer-step share: 99.75% for A1 and 99.75% for A1-control across 400 updates/run and 28 layers. Nonzero branch RMS indicates path activity; it does not establish that gate changes caused metric differences.

## Limits

This is a small exploratory comparison with three independent seeds. The t intervals describe uncertainty in these runs and cannot establish broad generalization. The task-stream manifest defines a 50-row dev partition per task; every row was evaluated at every stage. No test data were used. The equal-token reference was specified but not run, and A2-lite was not formally trained.

## Reproducibility artifacts

- Aggregated machine-readable summary: `D:/Task-Aware-TransNAS/results/phase4b/comparison_summary.json`.
- Per-run stage metrics, predictions, NLL, gate traces, and checkpoint audits: `D:/Task-Aware-TransNAS/results/phase4b/runs/`.
- Final full-parameter checkpoints: `D:/Task-Aware-TransNAS/results/phase4b/checkpoints/`.
- Frozen protocol: `reports/phase4b_training_protocol.md`.
