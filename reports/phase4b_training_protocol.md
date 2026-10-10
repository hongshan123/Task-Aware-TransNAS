# Phase 4B Training Protocol

## Question and comparison

Phase 4B compares A1 cross-head interaction with both the unmodified A0 model and the parameter-matched A1-control. It tests whether learned cross-head communication improves task acquisition or continual retention under the frozen full-parameter fine-tuning recipe. It does not test NAS or select architectures using the test split.

All 18 runs start from the same local Qwen3-0.6B weights (`MODEL_DIR`, SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`). Local-only loading is enforced. All base and added parameters are trainable; no LoRA and no replay are used.

## Run matrix and sequences

Each order is crossed with seeds 10, 20, and 30 and all three architectures, giving 18 full sequential runs.

| Sequence | Task order |
|---|---|
| S1 | task565 → task1603 → task1501 → task1384 |
| S2 | task565 → task1501 → task1603 → task1384 |

The paired task seeds are derived from the run seed and task ID. Thus, for a given sequence and seed, all architectures receive the same per-task example order. The S1/S2 runs also use the same task-specific row sampling seed for a task, so the order change does not silently change that task's examples. Training IDs, dev IDs, and their hashes are stored per run.

## Primary task-calibrated budget

The primary protocol applies the same task-specific step budget to every architecture, seed, and order:

| Task | Optimizer steps |
|---|---:|
| task565 CIRCA answer generation | 50 |
| task1603 SMCalFlow sentence generation | 250 |
| task1501 DSTC3 answer generation | 50 |
| task1384 Deal-or-No-Deal classification | 50 |

These task-level budgets were calibrated on prior development-split training-sufficiency results before the Phase 4B architecture comparison. They are frozen across architectures and task orders; no architecture gets task-specific extra tuning or step selection. Each run therefore has 400 optimizer steps. Microbatch size is 1 with gradient accumulation 4, maximum sequence length 512, maximum target length 64, AdamW with `foreach=False`, learning rate `1e-5`, weight decay `0.01`, gradient clipping at 1.0, and gradient checkpointing. The optimizer is recreated at each task boundary, matching the existing formal recipe. Effective supervised target-token totals and per-step tokens are recorded.

## Equal-token reference protocol

For a future budget-sensitivity comparison, use a common budget of 2,900 supervised target tokens per task. This reference is the observed target-token count from the prior 50-step task1603 pilot and gives each task the same number of supervised target tokens. Consume examples in the deterministic task-seeded order and stop at the first optimizer-step boundary that reaches or exceeds 2,900 tokens; record the overshoot. Keep the 2,900-token rule identical across architectures, seeds, and orders. This equal-token reference is specified but is not part of the 18 primary runs.

## Evaluation and recorded outcomes

The task-stream manifest caps each task at 500 training rows and 50 dev rows. The complete 50-row dev partition for every task is evaluated at the baseline and after every task stage. At each evaluation stage the runner stores the task's configured primary metric (ROUGE-L F1 for generation; accuracy for classification), teacher-forced token NLL, generation behavior, and the prediction rows.

Generation diagnostics include generated token length, exact normalized `Yes` and `No` shares, combined exact Yes/No share, parse success, and EOS termination. Classification predictions also retain the parser output and labels. The metric matrix keeps accuracy and ROUGE-L in their own units.

Continual-learning summaries include raw and baseline-headroom-normalized acquisition gains, cross-stage normalized AAA, nonnegative normalized forgetting, and normalized backward transfer. The per-task stage matrix remains available so an aggregate cannot hide a task-specific collapse. For A1 and A1-control, every optimizer update records all 28 layer gate values, post-clipping gate-gradient magnitudes, and added-branch RMS. The runner also records per-task training/evaluation durations, CUDA peak allocated/reserved memory, GPU name, parameter counts, input/output file hashes, and a final checkpoint reload logit comparison.

Only the final checkpoint from each completed run is retained. Dev predictions and metrics from every stage are kept. The test split is not loaded.

## Storage and runtime preflight

The initial preflight used the actual FP32 parameter counts, rather than the smaller compressed source checkpoint size. It estimated 40.01 GiB for the 18 final checkpoints, plus 1 GiB for predictions/metrics, 3 GiB temporary-save allowance, and 8 GiB safety margin: 52.01 GiB required. D: had 118.27 GiB free when preflight ran. The configured output root is `D:/Task-Aware-TransNAS/results/phase4b`; previous Phase 4A artifacts are outside that directory and are not modified. The 2-hour estimate is a training-only lower bound; total wall time will be longer because every run evaluates four tasks at five stages and reload-checks its final checkpoint.

## Paired analysis plan

Pair candidates by sequence and seed. Report means, standard deviations, and paired 95% t intervals for A1−A0, A1-control−A0, and A1−A1-control. The overall paired set has six runs; show S1 and S2 separately with three seeds each and report the difference in candidate effects between orders. Treat this as an exploratory, low-n comparison and show individual paired values rather than relying on significance tests. The comparison report will answer whether A1 improves new-task acquisition, whether it preserves earlier tasks better than both controls, whether any apparent gain is order-specific, and whether per-layer gates/branches remain active during training.

## Execution

From the project root in PowerShell:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\conda-env\python.exe scripts\run_phase4b_comparison.py
```

The runner skips only a run with both a final metrics file and a final checkpoint, and writes a partial manifest after each completed run so the experiment can resume after interruption.
