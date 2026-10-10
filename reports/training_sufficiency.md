# Generation-Task Training Sufficiency

## Protocol

Each generation task was trained independently from the local Qwen3-0.6B starting checkpoint for one continuous 250-step run. Metrics were sampled at the 50-, 125-, and 250-step prefixes, with one persistent AdamW optimizer per task run. This preserves comparable run prefixes instead of resetting optimizer state at each measurement point. Training used the formal settings: full parameters, microbatch 1, gradient accumulation 4, learning rate `1e-5`, 512-token sequence cap, and 64-token target cap. Each validation point used all 50 dev examples with greedy generation, up to 64 new tokens. The test split was not loaded.

Generated-token length counts EOS when emitted.

## Validation Curve

| Task | Steps | Cumulative mean train loss ↓ | Dev token NLL ↓ | ROUGE-L ↑ | Token-F1 ↑ | Mean generated tokens |
|---|---:|---:|---:|---:|---:|---:|
| task565 | 50 | 2.440 | 2.238 | **0.250** | **0.250** | 7.60 |
| task565 | 125 | 2.299 | **2.171** | 0.237 | 0.240 | 7.96 |
| task565 | 250 | 1.740 | 2.334 | 0.242 | 0.244 | 8.28 |
| task1603 | 50 | 1.377 | 1.206 | 0.241 | 0.247 | 9.96 |
| task1603 | 125 | 1.074 | 0.914 | 0.272 | 0.274 | 8.76 |
| task1603 | 250 | **0.774** | **0.869** | **0.418** | **0.419** | 9.96 |

Both tasks learn measurable output behavior. Over the complete runs, train loss fell from 4.659 to 1.232 for task565 and from 3.208 to 0.679 for task1603. The 250-step runs processed 9,106 and 14,752 supervised target tokens, respectively, across 1,000 sampled examples per task.

## Budget Recommendation

- **task565: 50 optimizer steps** when ROUGE-L is the primary objective. Its best measured ROUGE-L is at step 50; increasing to 125 or 250 did not improve generation quality. Step 125 gives the lowest measured NLL, so retain dev NLL as a secondary early-stop signal.
- **task1603: 250 optimizer steps** under the current setup. ROUGE-L rises at every checkpoint and reaches 0.418 at 250, while NLL also improves. The curve has not clearly plateaued by 250, so use 250 as the next controlled budget rather than claiming it is a global optimum.

The evidence does not suggest a prompt/data pipeline failure that would block architecture work: loss and dev generation scores improve substantially, particularly for task1603. The architecture smoke test still uses 50 steps per task to match the current baseline protocol; it is an execution check, not a comparison at the recommended task-specific budgets.

## VRAM and Checkpoints

| Task | Training peak allocated | Training peak reserved | Dev-evaluation peak allocated |
|---|---:|---:|---:|
| task565 | 10.16 GiB | 10.74 GiB | 9.29 GiB |
| task1603 | 10.28 GiB | 10.37 GiB | 9.34 GiB |

The two 250-step checkpoints are under `D:/Task-Aware-TransNAS/results/architecture_pilot/training_sufficiency/`. The per-step loss traces, validation metrics, predictions, token counts, and sample-order hashes are in `sufficiency_metrics.json` and each task's prediction folders.

## Scope

The two tasks start independently from Qwen3-0.6B; this sufficiency run does not include a continual transition between them. Validation uses dev only and no model or hyperparameter was selected with test data.
