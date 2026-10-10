# Phase 4B Architecture Audit

## Scope

This audit covers the three candidates in the controlled attention study: A0, A1, and A1-control. The models were rebuilt from the local Qwen3-0.6B checkpoint at `MODEL_DIR`; `local_files_only=True` was used, and the model file SHA-256 is `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. The test split was not loaded.

The detailed machine-readable record is `D:/Task-Aware-TransNAS/results/phase4b/audit/architecture_audit.json`.

## Architecture and capacity match

Qwen3-0.6B has 28 decoder layers, hidden size 1,024, and 16 attention heads of width 128. A1 inserts an input-conditioned cross-head interaction on the concatenated head values immediately before each attention `o_proj`. Shared per-head query and key projections use router width 8; a 16-by-16 softmax mixes head values. The mixed branch is multiplied by one learned scalar per layer and added to the original head values.

A1-control uses the same query/key parameter shapes and the same per-layer scalar, but only applies an independent sigmoid gate to each head. It does not pass values between heads. This capacity-matched control tests whether cross-head communication contributes beyond adding the router parameters and a residual path.

| Candidate | Total parameters | Trainable parameters | Added vs A0 |
|---|---:|---:|---:|
| A0 | 596,049,920 | 596,049,920 | 0 |
| A1 | 596,107,292 | 596,107,292 | 57,372 |
| A1-control | 596,107,292 | 596,107,292 | 57,372 |

All parameters, including the original Qwen3 weights, remain trainable. The new branch scalar is initialized to exactly zero. No LoRA or replay is used.

## Initialization, shape, and generation checks

Selected tensor fingerprints from the local source checkpoint match across A0, A1, and A1-control. With the added scalar at zero, maximum absolute logit differences from A0 were exactly 0 for each candidate at input lengths 1, 17, 75, 160, and 256 tokens. A right-padded batch with 160-token and 80-token examples (80 pad tokens) also matched exactly at the attended positions.

Greedy generation matched the base model for all three candidates. Incremental cached decoding and full-sequence decoding selected the same next token. Their maximum absolute logit difference was `1.34e-5` (relative L2 `1.02e-6`), consistent across the base and zero-initialized variants. The candidate's cached logits matched A0 exactly.

## Gate and branch gradient audit

An actual two-step full-parameter CUDA update was run for A1 and A1-control with gradient checkpointing. In both variants, all 28 per-layer scalars received finite, nonzero gradients on step one while every query/key gradient and added-branch RMS remained zero. After the first update opened the scalar gates, step two produced nonzero query/key gradients and nonzero added-branch activity in all 28 layers. This confirms that zero initialization preserves the initial function while allowing the gate to learn first and then expose the router path to gradients.

| Candidate | Step 1 layers with gate gradient | Step 1 layers with Q/K gradient | Step 2 layers with Q/K gradient and active branch |
|---|---:|---:|---:|
| A1 | 28/28 | 0/28 | 28/28 |
| A1-control | 28/28 | 0/28 | 28/28 |

The formal experiment will also record each layer's gate value, clipped gate-gradient magnitude, and branch RMS after every optimizer update. The branch RMS is measured from the last microbatch forward pass in that update.

## Audit conclusion

The variants share the same local base weights, have exact zero-initialized function preservation, and match added parameter counts. Variable-length input, padding, cached decoding, greedy generation, and the initial gate-gradient path passed. This establishes implementation readiness; it does not establish a training or continual-learning advantage. That comparison is the purpose of the 18 formal runs.
