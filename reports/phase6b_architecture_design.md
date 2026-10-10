# Phase 6B Conditional Residual Architecture Design

## Research question

Can one fixed-capacity residual computation, selected from the current input, reduce interference among four continual-learning tasks without storing or forwarding a historical model?

The Phase 6A evidence argues against copying old activations indiscriminately: historical residual-stream patches helped their source generation task, while cross-task transfers often hurt. In the Phase 6B compatibility audit, activation patching and parameter rollback each showed negative off-diagonal token-NLL transfer in 5 of 12 directed task pairs. The best source component also varied by task. These are historical-model-assisted measurements; they identify information available in the older checkpoints and do not imply that the final model itself still contains that information. See [phase6b_compatibility.md](phase6b_compatibility.md) for the matrix and its limits.

## Shared implementation

Each residual branch is installed on a decoder block’s output and added to the hidden state. The Qwen3-0.6B trunk remains the same in all arms. Every adapter branch is a two-layer nonlinear bottleneck with SiLU. Its down projection starts at zero, so adding the branch initially leaves the pretrained model’s function unchanged.

| Arm | Residual computation | Conditioning | Location |
| --- | --- | --- | --- |
| B0 | Original Transformer | None | None |
| B1-static | One always-active width-256 bottleneck per selected block | Static | Blocks 25 and 27 |
| B2-input | Four shared width-64 bottleneck experts per selected block; top-1 dispatch for each token | Current token hidden state; no Task ID | Blocks 25 and 27 |
| B3-oracle | Same four experts and capacity as B2 | Known task index selects one expert | Blocks 25 and 27 |
| B2-random-location | Same input router and capacity as B2 | Current token hidden state | Seed-selected blocks 21 and 22 |

Layer indices are zero-based. The random-location pool is `[20, 21, 22, 23, 24, 26]`: it samples two later decoder blocks while excluding the Phase 6A-selected blocks 25 and 27. Seed `20261010` selects `[21, 22]`. The sampler and unit test now encode this rule directly, so the stored control pair is reproducible.

## B2 routing mechanism

For each token hidden state `h`, B2 computes router logits `W h + b`, then selects one of four experts. Training uses hard Gumbel selection with a straight-through probability gradient; evaluation uses deterministic top-1 selection. Only the selected token/expert pairs execute their bottleneck branch. The residual result is added back to that block’s hidden state.

This is input-dependent dispatch: two token states can select different branches. The router receives no task label. A fixed shared set of experts is used throughout learning, so persistent parameter count does not grow with the number of tasks. B3 intentionally breaks the no-Task-ID rule as an oracle reference.

The controls isolate several mechanisms:

- **No-op initialization:** compare logits before and after installing the zero-output branch.
- **Random route:** choose experts randomly for each token after B2 training.
- **Static gate:** use the router bias to choose one constant expert per layer.
- **Residual off:** return zero from the added branch after B2 training.
- **Random location:** retain the B2 routing rule and capacity while moving it to the reproducibly sampled blocks.

The implementation uses neither LoRA nor replay. Training starts from the same local base-model weights, does not load historical Phase 6A checkpoints, and saves one final state per non-baseline run without intermediate stage snapshots.

## Parameter and compute budget

With hidden size `H=1024`, two selected blocks `L=2`, four paths `K=4`, and bottleneck width `r=64`:

- B1: `2 × L × H × K × r = 1,048,576` added parameters.
- B2/B3/random-location: `2 × L × H × K × r + L × (H × K + K) = 1,056,776` added parameters.
- B2’s router adds 8,200 parameters over B1, a 0.78% difference relative to B1’s adapter budget. B2 adds 0.1773% of the 596,049,920-parameter trunk.
- The measured-run FLOPs estimate is 0.0432% above the base 512-token prefill for B2; it excludes dispatch, activation, and cache costs. B1’s dense width-256 branch estimate is 0.1677%.

The design keeps capacity approximately matched while making the input route a small fraction of the trunk. Peak allocation was about 10.9 GiB on the 24 GiB RTX 4090. Eight final adapter states occupy about 19.1 GB in decimal bytes; no intermediate checkpoints were retained.

## Pilot and mechanism evidence

The seed-10 pilot uses the same task-specific step budgets, AdamW settings, sample order, and supervised target-token budgets as the paired A0 baseline. S1 is 565 → 1603 → 1501 → 1384; S2 is 565 → 1501 → 1603 → 1384. Each task is evaluated on 50 dev examples. Test data is not loaded.

All adapter arms passed the exact initialization-output check and saved-state reload check. Sampled training traces contain nonzero branch gradients and nonzero branch outputs after the zero-output start. The input router therefore executed and updated, but this alone does not show that input-dependent selection improved continual learning.

The result does not establish the proposed mechanism:

- B2 beat parameter-matched B1 on 4 of 8 task/order primary-score comparisons. It lost on SMCalFlow in both orders; the DSTC3 change reversed direction between orders.
- On these 50-example dev evaluations, the random-route control matched B2’s primary score in all eight task/order cells. Static-gate and residual-off controls also matched in most cells; on S2 SMCalFlow, static/off scored above input routing.
- Mean dominant-expert share was 0.777 and normalized route entropy was 0.373. The average total-variation distance for the same task across task orders was 0.303, versus 0.047 between different tasks within an order. These descriptive route statistics shift more with order/training than with task identity in this pilot.
- The oracle arm did not consistently outperform B2 or B1. The random-location arm was better than B2 on SMCalFlow in both orders, but it is only one fixed sampled pair and one seed.

The routing controls are single final evaluations, not repeated random-route trials. These metrics are descriptive for a small pilot; they do not support a causal claim that conditional routing removes representation interference.

## Decision on NAS

Do not start NAS or extend to seeds 20 and 30 from this pilot. The seed-10 gate was not met: B2 lacks a stable advantage over B1 and random/static/off route controls, and the order sensitivity is substantial. Current evidence is insufficient to make Task-Conditional Residual Pathways the architecture-search foundation of an ACL 2027 paper.

If a future replicated pilot first establishes a stable advantage, the defensible search variables would be layer location, bottleneck width, route granularity, and path-sharing ratio. None should enter a NAS search space yet. The next useful experiment is a replicated, larger-dev confirmation with predeclared route controls and a requirement that B2 improve both old-generation retention and new-classification performance without depending on task order.

## Implementation and verification

- Implementation: [phase6b_modules.py](../scripts/phase6b_modules.py), [run_phase6b_pilot.py](../scripts/run_phase6b_pilot.py), and [phase6b.yaml](../configs/phase6b.yaml).
- Mechanism tests cover exact no-op initialization, dynamic hidden-state routing, explicit oracle/control semantics, gradients, parameter matching, and reproducible random-layer selection.
- Consolidated measurements: [phase6b_pilot_results.md](phase6b_pilot_results.md), [pilot_summary.json](../results/phase6b/pilot_summary.json), and [pilot_task_metrics.csv](../results/phase6b/pilot_task_metrics.csv).
