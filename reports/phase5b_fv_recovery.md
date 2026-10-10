# Phase 5B: Function Vector Recovery Verification

## Scope and provenance

This analysis tests whether same-trajectory historical Function Vectors restore old task behavior after the A0/S1 task565 → task1603 → task1501 → task1384 stream. The historical task565 and task1603 checkpoints had not been retained by the original run, so the minimum required S1 sequence was replayed into the Phase 5B output directory. The replay manifest records exact agreement with the original final checkpoint across **596,049,920 parameters** (maximum absolute difference 0), and exact equality of sample order, supervised tokens, optimizer steps, and recorded losses at every task stage. Original Phase 4B/5 outputs were only read; all new experiment results are under `results/phase5b/`.

The replay used the same task order, dataset ID hashes, task seeds, optimizer settings, and training rows. It did not load dev during training or load test; NAS was not run. The two historical vectors use the model immediately after task565 and immediately after task1603, respectively.

## Method

* Historical vectors: for each task, average the last-token, per-query-head `o_proj` contributions on the same 10 four-shot correct-ICL build prompts used by Phase 5. Rank all 448 `(layer, query_head)` candidates by sequence log-probability gain over the label-shuffled ICL counterfactual, then sum the top 10 mean head contributions. Each historical scan stores all 448 per-query effects and train support IDs.
* Checkpoint controls: compare the historical vector with the same task's Phase 5 final collapsed vector, a seeded norm-matched random vector, the other generation task's historical vector, and subtraction of the final task1384 classifier vector.
* Tuning: each available injection vector was tuned separately on dev indices 10–24 (15 examples/task), over layers `[5, 12, 19, 26]` and scales `[0.25, 0.5, 1, 2]`, minimizing token-weighted teacher-forced reference NLL. The selected setting was then locked.
* Locked evaluation: all 25 examples/task at dev indices 25–49. Baseline, vector conditions, a format-instruction condition, and a correct-label four-shot ICL condition were evaluated. Four ICL demonstrations came only from train and all four labels were preserved.
* Injection path: at the selected Qwen3 layer, the FV is added directly to `self_attn.o_proj` output, the same 1024-wide attention-output space used to construct the per-head contributions. Teacher forcing patches from the final prompt position through reference tokens; cached generation patches the final output position on prefill and on every single-token decode call. Greedy generation is capped at 64 tokens. Paired bootstrap intervals use 5,000 resamples of the 25 locked IDs.

The intervention preflight verified Qwen3 geometry `[28 layers, 16 query heads, 8 KV heads, 128 head_dim, 1024 hidden]`, the selected `self_attn.o_proj` output hook and its 1024-wide tensor, a zero-vector no-op (NLL-sum difference 0), the expected teacher-forced patch count, and both prefill and cached decode hook events. The test split was not loaded.

## Historical vectors

| Historical boundary | Task | L2 norm | Cosine to same-task final FV | Cosine to same-task pretrained FV |
|---|---|---:|---:|---:|
| After task 565 | task565_circa_answer_generation | 8.368 | -0.019 | 0.112 |
| After task 1603 | task1603_smcalflow_sentence_generation | 51.366 | 0.016 | 0.087 |

The vectors point in directions distinct from the final collapsed vectors (cosines near zero). A low cosine alone does not establish recovery; the controlled locked evaluation below is the behavioral check.

## Locked development results

ROUGE-L is shown for every task; it is the primary generation measure for the two old generation tasks. `NLL` is token-weighted reference NLL for the exact prompt condition. Generated length includes EOS when emitted.

| Task | Condition | ROUGE-L | Parsed accuracy | Token NLL | Generated tokens | Exact Yes | Exact No | Exact Yes/No | Most-common label among parseable | Parseable | Invalid format |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1384 | baseline | 0.960 | 0.960 | 0.073 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | correct_four_shot_icl | 0.960 | 0.960 | 0.098 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | explicit_format_hint | 0.960 | 0.960 | 0.073 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | final_collapsed_checkpoint_fv | 0.960 | 0.960 | 0.094 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | norm_matched_random_fv | 0.960 | 0.960 | 0.075 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | subtract_latest_classifier_fv | 0.960 | 0.960 | 0.095 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1384 | unrelated_historical_task_fv | 0.960 | 0.960 | 0.070 | 2.00 | 84.0% | 16.0% | 100.0% | 84.0% | 100.0% | 0.0% |
| 1501 | baseline | 0.640 | 0.640 | 0.231 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1501 | correct_four_shot_icl | 0.520 | 0.520 | 0.235 | 2.00 | 60.0% | 40.0% | 100.0% | 60.0% | 100.0% | 0.0% |
| 1501 | explicit_format_hint | 0.640 | 0.640 | 0.232 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1501 | final_collapsed_checkpoint_fv | 0.640 | 0.640 | 0.195 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1501 | norm_matched_random_fv | 0.640 | 0.640 | 0.228 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1501 | subtract_latest_classifier_fv | 0.640 | 0.640 | 0.235 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1501 | unrelated_historical_task_fv | 0.640 | 0.640 | 0.231 | 2.00 | 56.0% | 44.0% | 100.0% | 56.0% | 100.0% | 0.0% |
| 1603 | baseline | 0.014 | — | 2.525 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | correct_four_shot_icl | 0.014 | — | 2.658 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | explicit_format_hint | 0.014 | — | 2.563 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | final_collapsed_checkpoint_fv | 0.014 | — | 2.218 | 2.00 | 88.0% | 12.0% | 100.0% | 88.0% | 100.0% | 0.0% |
| 1603 | historical_task_boundary_fv | 0.014 | — | 2.441 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | norm_matched_random_fv | 0.014 | — | 2.480 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | subtract_latest_classifier_fv | 0.014 | — | 2.555 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 1603 | unrelated_historical_task_fv | 0.014 | — | 2.445 | 2.00 | 92.0% | 8.0% | 100.0% | 92.0% | 100.0% | 0.0% |
| 565 | baseline | 0.000 | — | 5.109 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | correct_four_shot_icl | 0.000 | — | 5.103 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | explicit_format_hint | 0.000 | — | 5.153 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | final_collapsed_checkpoint_fv | 0.000 | — | 5.062 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | historical_task_boundary_fv | 0.000 | — | 4.953 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | norm_matched_random_fv | 0.000 | — | 5.114 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | subtract_latest_classifier_fv | 0.000 | — | 5.179 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| 565 | unrelated_historical_task_fv | 0.000 | — | 5.026 | 2.00 | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |

### What the output-format measures mean

* **Exact Yes/No output rate** counts only outputs that equal `Yes` or `No` after NFKC normalization, trimming, and case folding, divided by all outputs.
* **Most-common label concentration** is the larger parsed Yes/No count divided by parseable outputs only.
* **Invalid answer-format rate** is the share for which the existing unambiguous Yes/No parser returns no label.

The legacy Phase 5 field `yes_no_collapse_rate` is parser-based: it took the larger parsed label count divided by all outputs. It is neither a strict exact-string rate nor the new concentration denominator when some outputs are unparseable. Phase 5 artifacts were left unchanged. In this Phase 5B locked sample, all outputs parsed and were exactly Yes/No (100% exact and 0% invalid), but label concentration differs: task565 was 100% one label, task1603 was 92%, task1384 was 84%, and task1501 was 56%. Thus a 100% exact Yes/No rate alone does not mean all examples collapsed to the same label.

### Direct historical-vector contrasts

Paired deltas below are historical FV minus the named control. For NLL, negative values favor the historical FV. For ROUGE-L, positive values favor the historical FV.

| Task | Contrast | Mean per-example NLL delta (95% CI) | ROUGE-L delta (95% CI) |
|---|---|---:|---:|
| 1603 | final_collapsed_checkpoint_fv | 0.230 [0.165, 0.294] | 0.000 [0.000, 0.000] |
| 1603 | norm_matched_random_fv | -0.050 [-0.086, -0.019] | 0.000 [0.000, 0.000] |
| 1603 | unrelated_historical_task_fv | -0.026 [-0.058, 0.000] | 0.000 [0.000, 0.000] |
| 565 | final_collapsed_checkpoint_fv | -0.110 [-0.148, -0.074] | 0.000 [0.000, 0.000] |
| 565 | norm_matched_random_fv | -0.158 [-0.197, -0.124] | 0.000 [0.000, 0.000] |
| 565 | unrelated_historical_task_fv | -0.065 [-0.113, -0.021] | 0.000 [0.000, 0.000] |

For task565, the historical vector improves teacher-forced NLL over the final, random, and unrelated controls, while all 25 generated answers remain the same collapsed `Yes` response (ROUGE-L stays 0). This supports a task-specific effect on reference likelihood, but not recovery of generated text. For task1603, the historical vector improves paired NLL over random, but the interval against the unrelated historical vector reaches zero and it performs worse than the final collapsed vector. The historical injection, format hint, and true ICL preserve baseline outputs on all 25 IDs; the final vector changes one label but leaves ROUGE-L at 0.014 and length at two tokens. Evidence for task1603-specific recovery is weak.

The explicit format instruction and true four-shot ICL did not change the old tasks' generated outputs or ROUGE-L. They did not restore the missing generation behavior. A few NLL shifts are measurable, but NLL changes alone do not recover the output function.

## Full 448-head causal-effect similarity

For each task, this compares the mean sequence-log-probability gain vector over all 448 query heads from 10 Phase 5 build prompts on the final collapsed checkpoint. Pearson/Spearman correlations and positive-gain weighted overlap are tested against 10,000 permutations that independently shuffle one task's head identities while preserving its score distribution. P values are BH-adjusted across the six task pairs within each statistic.

| Pair | Pearson r (BH p) | Spearman ρ (BH p) | Positive-gain weighted overlap (BH p) | Top-10 overlap (permutation p) |
|---|---:|---:|---:|---:|
| 565–1603 | 0.198 (0.0003) | 0.154 (0.0024) | 0.566 (0.0006) | 0 (1.0000) |
| 565–1501 | -0.159 (0.0008) | -0.169 (0.0003) | 0.085 (0.9093) | 0 (1.0000) |
| 565–1384 | 0.006 (0.8902) | 0.039 (0.4064) | 0.718 (0.1572) | 0 (1.0000) |
| 1603–1501 | -0.212 (0.0003) | -0.264 (0.0003) | 0.068 (0.9246) | 1 (0.2035) |
| 1603–1384 | 0.082 (0.1186) | 0.084 (0.1051) | 0.535 (0.0819) | 1 (0.1923) |
| 1501–1384 | -0.051 (0.3252) | -0.074 (0.1387) | 0.084 (0.9093) | 1 (0.1969) |

Top-10 query-head sets are also unstable across build prompts. Mean within-task pairwise Top-10 Jaccard over the 45 pairs of prompts is:

| Task | Mean Top-10 Jaccard | Range |
|---|---:|---:|
| 565 | 0.014 | [0.000, 0.111] |
| 1603 | 0.024 | [0.000, 0.111] |
| 1501 | 0.088 | [0.000, 0.250] |
| 1384 | 0.037 | [0.000, 0.176] |

The task565 Top-10 set has zero overlap with every other task, but zero overlap is not itself evidence of isolation: the 10,000-permutation null gives p=1.000 for each such observed zero, with only 448 candidate heads and very unstable per-query rankings. At the full-distribution level, several task pairs do differ from randomized head identity after BH correction, while other pairs do not. The strongest same-direction positive-gain overlap is between tasks565 and 1603 (0.566, adjusted p<0.001). These findings show structured relationships in some full profiles, not clean head partitions.

## Conclusion and architecture guidance

1. **Historical FV recovery is task-dependent.** Task565's historical vector improves reference NLL beyond all required controls, but it does not restore generated text. Task1603 lacks clear specificity over its unrelated historical vector and performs worse than its final FV.
2. **Formatting is not the main failure here.** The model already emits exact Yes/No strings with zero invalid-format responses; explicit instruction and correct ICL do not restore the two old generation tasks. Their outputs are label-concentrated and short, whereas the task references require open-text generation.
3. **Some head-profile differences exceed a random identity null, but Top-10 evidence is weak.** Full-profile correlation tests identify pair-specific structure. Top-10 zero overlap is common under the null and Top-10 rankings are unstable across queries, so it cannot establish task isolation.
4. **Architecture decision:** the evidence motivates only a small, controlled Conditional Routing pilot with the same all-head randomization and task-agnostic controls. It does not validate a routing benefit or justify broad NAS. Head Isolation is not supported by zero Top-10 overlap alone; retain it as a comparison arm only if the pilot independently demonstrates stable head groups.

## Reproduction and artifacts

Run from the project root in the existing local Windows environment. Model loading uses `local_files_only=True`; no model download occurs.

```powershell
.\conda-env\python.exe scripts\replay_phase5b_boundaries.py --preflight-only
.\conda-env\python.exe scripts\replay_phase5b_boundaries.py
.\conda-env\python.exe scripts\extract_phase5b_historical_fv.py --preflight-only
.\conda-env\python.exe scripts\extract_phase5b_historical_fv.py
.\conda-env\python.exe scripts\analyze_phase5b_heads.py
.\conda-env\python.exe scripts\run_phase5b_interventions.py --preflight-only
.\conda-env\python.exe scripts\run_phase5b_interventions.py
.\conda-env\python.exe scripts\write_phase5b_report.py
.\conda-env\python.exe scripts\audit_phase5b.py
```

Key artifacts: `configs/phase5b_fv_recovery.yaml`, `results/phase5b/boundary_replay_manifest.json`, `results/phase5b/extraction/`, `results/phase5b/head_similarity/`, `results/phase5b/interventions/metrics.json`, `results/phase5b/interventions/tuning_grid.json`, per-condition locked predictions/NLL under `results/phase5b/interventions/evaluation/`, and `results/phase5b/audit.json`.

The split discipline is build dev[0:10], tune dev[10:25], locked evaluation dev[25:50], and demonstrations from train only. Test was not loaded; formal follow-up training and NAS were not run.
