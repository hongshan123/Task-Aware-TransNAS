# Architecture Design and Smoke Pilot

## Candidates

All candidates start from the same local Qwen3-0.6B weights and keep every parameter trainable. No LoRA, replay, or other continual-learning algorithm is used.

| Candidate | Added computation | Total / trainable parameters | Added parameters |
|---|---|---:|---:|
| A0 | Original Qwen3 Transformer | 596,049,920 / 596,049,920 | 0 |
| A1 | Per-token cross-head interaction before each attention output projection | 596,107,292 / 596,107,292 | 57,372 |
| A1-control | Per-head input-conditioned gate with no cross-head messages | 596,107,292 / 596,107,292 | 57,372 |
| A2 | Per-token top-2 routing over four added grouped SwiGLU experts per block | 860,405,900 / 860,405,900 | 264,355,980 |
| A2-control | Dense input-conditioned mixture over the same four experts | 860,405,900 / 860,405,900 | 264,355,980 |

**A1 calculation.** Qwen3 produces 16 attention-head outputs with 128 features per head before `o_proj`. Each layer reshapes that 2,048-wide input to `[batch, sequence, 16, 128]`. Shared query/key projections map each head to an 8-dimensional router space. A per-token 16×16 softmax mixes information across heads; a learned scalar multiplies the mixed-head residual before it is added to the original head outputs. The scalar starts at zero. A1-control has the same query/key weights and scalar, but uses each head's diagonal score as an independent sigmoid gate.

**A2 calculation.** The original dense MLP remains unchanged. Each decoder block also has a router from hidden size 1,024 to four experts and four SwiGLU experts with intermediate width 768 each. Per token, the sparse path selects the top two router scores, normalizes those two weights, and evaluates only the selected experts. The weighted expert output is added to the original MLP output through a scalar initialized to zero. A2-control uses the same router and expert parameters but evaluates all four experts and combines them with a dense softmax mixture.

The controls exactly match the candidate's new parameter count, isolating cross-head messages in A1 and sparse conditional execution in A2. Controls passed module-level parameter-count and forward tests, but were not trained in the smoke pilot.

## Function Preservation and Compute

The initialization audit used a 75-token dev prompt. A0, A1, and A2 each had maximum absolute logit difference **0.0** from the base model. After saving and reloading each checkpoint by rebuilding its architecture modules and loading the local safetensors, maximum logit difference was also **0.0**. Qwen3 omits the tied `lm_head.weight` key from saved safetensors; it remains tied to the restored embedding weight.

Approximate FLOPs below use the Qwen3 configuration and a 512-token prefill. The formula counts two FLOPs per multiply-add and the attention/MLP matrix multiplies. It omits normalization, activations, softmax and kernel overhead; measured runtime may differ.

| Candidate | Approx. FLOPs per prefill token | Approx. FLOPs per 512-token sequence | Relative to A0 |
|---|---:|---:|---:|
| A0 | 1.251 GFLOP | 640.35 GFLOP | — |
| A1 | 1.254 GFLOP | 642.29 GFLOP | +0.30% |
| A1-control | 1.253 GFLOP | 641.35 GFLOP | +0.16% |
| A2 | 1.515 GFLOP | 775.76 GFLOP | +21.14% |
| A2-control | 1.779 GFLOP | 911.05 GFLOP | +42.28% |

## Matched Smoke Protocol and Results

The same homogeneous order was used for all three candidates: task565 → task1603 → task1501 → task1384. Each task received 50 optimizer steps, microbatch 1, gradient accumulation 4, learning rate `1e-5`, and the same 512/64 sequence and target caps. The pilot evaluates the first 10 dev rows per task after each stage; this smaller matrix is for pipeline verification only. The test split was not loaded.

| Candidate | Task565 target tokens | Task1603 | Task1501 | Task1384 | Peak allocated VRAM | Peak reserved VRAM |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 1,805 | 2,900 | 600 | 600 | 10.42 GiB | 12.04 GiB |
| A1 | 1,805 | 2,900 | 600 | 600 | 10.42 GiB | 12.05 GiB |
| A2 | 1,805 | 2,900 | 600 | 600 | 14.99 GiB | 16.27 GiB |

All three runs consumed the same supervised target-token counts and the same per-task example order; SHA-256 hashes match across candidates. Every parameter was trainable. All checkpoints saved and reloaded successfully. A2 fits the 24 GiB RTX 4090 with about 8 GiB remaining at its measured peak and did not require OOM recovery.

The 10-row dev smoke metrics show zero ROUGE-L for both generation tasks after the classification tasks across all three candidates. This is consistent with the generation-collapse diagnosis, but the subset and smoke budget are too small to rank architectures. A1/A2 are ready for the next **matched candidate-versus-control** study; the controls must be trained there before attributing any change to structure.

## Outputs and Reproduction

- Full architecture metrics, parameter counts, token-budget hashes, FLOPs, initialization checks, and reload checks: `D:/Task-Aware-TransNAS/results/architecture_pilot/architecture_pilot_metrics.json`
- Per-candidate dev matrices and full-parameter checkpoints: `D:/Task-Aware-TransNAS/results/architecture_pilot/`
- Architecture code: [architecture_modules.py](../scripts/architecture_modules.py)
- Smoke runner: [run_architecture_pilot.py](../scripts/run_architecture_pilot.py)
- CPU unit tests: [test_architecture_modules.py](../tests/test_architecture_modules.py)

Run on Windows from the project root:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\conda-env\python.exe scripts\run_architecture_pilot.py
```
