# A2-lite Parameter-Matched Design

## Design target

A2-lite is a compact grouped SwiGLU expert branch intended to add approximately 1–2% parameters to Qwen3-0.6B while keeping the original model function at initialization. The new branch is attached to each decoder block's MLP output and is scaled by a per-layer scalar initialized to zero. The design uses four experts and top-2 expert selection per token.

Each block's added parameters are `4 * hidden_size` router weights and biases, plus `4 * (3 * hidden_size * expert_width)` expert weights, plus one residual scalar. With hidden size 1,024 and 28 layers, the totals are:

| Variant | Expert width | Added parameters | Added vs A0 | Dense-control total added parameters |
|---|---:|---:|---:|---:|
| A2-lite-16 | 16 | 5,619,852 | 0.943% | 5,619,852 |
| A2-lite-32 | 32 | 11,124,876 | 1.867% | 11,124,876 |

The width-32 version meets the 1–2% target. Width 16 is retained as a smaller ablation below 1%. Each has a dense input-conditioned mixture control with exactly the same parameter count; the control evaluates all four experts, while the sparse candidate evaluates only its selected two.

## Approximate compute at a 512-token prefill

The FLOP estimate counts two operations per multiply-accumulate for model and expert linear layers. It excludes normalization, nonlinearities, softmax, routing/scatter overhead, and kernel effects, so it is a design estimate rather than a measured latency result.

| Variant | Approx. FLOPs per prefill token | Approx. FLOPs per 512-token sequence | Per-token change vs A0 |
|---|---:|---:|---:|
| A0 | 1,250,689,024 | 640,352,780,288 | — |
| A2-lite-16 | 1,256,423,424 | 643,288,793,088 | +0.46% |
| A2-lite-16-control | 1,261,928,448 | 646,107,365,376 | +0.90% |
| A2-lite-32 | 1,261,928,448 | 646,107,365,376 | +0.90% |
| A2-lite-32-control | 1,272,938,496 | 651,744,509,952 | +1.78% |

The top-2 width-32 branch has the same approximate FLOPs as the top-2 width-16 dense control because each evaluates two experts at the same aggregate intermediate width (2×32 vs 4×16). The parameter counts differ because all four width-32 experts are stored in both matched variants.

## Function and gradient checks

The CPU module test confirms exact equality between A0 and both A2-lite-32 variants at zero initialization, exact candidate/control parameter equality, finite output, and a nonzero residual-scalar gradient. It also confirms that the added path can receive gradients after the scalar opens. No A2-lite formal training was run or is included in the Phase 4B experiment matrix.

## Suggested next experiment

If the Phase 4B controlled-attention comparison supports a broader architecture study, start A2-lite at width 32 and its exact-parameter dense control. Keep the same local base weights, stream orders, seeds, task budgets, dev matrix, and final-checkpoint policy. Retain width 16 only if the compute/parameter trade-off is itself a study question. Compare per-task routing load and expert utilization during training before claiming that sparse execution provides a useful compute benefit.
