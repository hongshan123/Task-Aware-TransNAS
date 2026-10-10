# Continual instruction-tuning pilot and formal baseline

The original run below is the bounded pilot. A separate formal four-task, three-order full-parameter experiment was subsequently completed; its protocol and results are summarized in the final section. Both use the authors' CITB `cl_dialogue_tasks` stream and published `order1`. The source repository is [hyintell/CITB](https://github.com/hyintell/CITB). The raw task JSON remains on D: and is not copied into this Git repository.

## Bounded pilot protocol

- Tasks follow their order in official `order1`: `task848_pubmedqa_classification`, `task611_mutual_multi_turn_dialogue`, `task1590_diplomacy_text_generation`, `task1500_dstc3_classification` (original stream positions 1, 2, 6, and 11). This subset covers the three CITB task categories.
- For each task, the first 50 source instances are held out for test, the next 50 for dev, and the remaining instances form the train pool. Up to 500 train instances are sampled without replacement using seed 10. The dev split is retained but not used for tuning.
- The upstream preparation code accepts a seed argument but does not apply it before its Python `random.shuffle`; this project applies seed 10 explicitly to make the training subset reproducible. The exact instance IDs, source JSON checksums, commit, and counts are in `reports/data_manifest.json`.
- Test instances are excluded from training and development. They are used only for the baseline and sequential accuracy matrix, with the cap of 50 examples per task.
- Each task gets 20 optimizer steps, micro-batch size 1, gradient accumulation 4, constant learning rate `2e-5`, fresh AdamW optimizer state, and the previous task's model weights. Parameters remain FP32; forward computation uses CUDA BF16 autocast and gradient checkpointing. No adapters, replay buffer, or NAS changes are part of this run.
- Evaluation uses greedy decoding and normalized exact match against any official reference output. Forgetting is the mean, over prior tasks, of the best post-task accuracy observed before the current task minus the current accuracy. The matrix and per-instance predictions are saved on D:.
- If CUDA runs out of memory, the current optimizer step is retried with a shorter maximum sequence length. Evaluation reduces input length and then generation length. Every fallback is recorded in metrics.
- `MODEL_DIR` is loaded from `.env`; both tokenizer and model use `local_files_only=True`. Offline environment flags are set before model loading. The model directory is read-only from this workflow.

## Windows PowerShell commands

```powershell
conda activate .\conda-env
python -m pip install -e . --no-deps
python .\scripts\prepare_citb.py
python .\scripts\run_continual_baseline.py --mode smoke
python .\scripts\run_continual_baseline.py --mode pilot
```

All source data, Hugging Face and Torch caches, checkpoints, result JSON, and logs are written under `D:\Task-Aware-TransNAS`. `results` and `checkpoints` in the project root are Windows directory junctions for convenient access. The pilot is intentionally bounded and should not be presented as a full CITB benchmark.

## Formal three-stream follow-up

The formal baseline replaced the pilot task subset with four eligible tasks balanced at 500 train / 50 dev / 50 test instances each. Two tasks are free-text generation from the CITB Dialogue Generation category; two are Yes/No classification from Dialogue State Tracking. Task pairs and homogeneous, heterogeneous, and mixed orders were selected using frozen prompt-representation and final-block gradient similarities from training records only. The fixed IDs, source hashes, categories, similarity evidence, split IDs, and stream definitions are in `data/task_stream_manifest.json`.

The complete task definition remains in every prompt. Long task1501 dialogue inputs are truncated from the beginning to retain the recent dialogue and question. A 3-by-3 full-parameter learning-rate/step grid was selected using the homogeneous stream and dev data only; the selected learning rate is `1e-5`, with 50 optimizer steps per task and gradient accumulation 4. The test set was opened only after this choice was frozen.

All three streams completed with all 596,049,920 model parameters trainable, 512-token sequence limits, BF16 autocast, FP32 weights, and no OOM recovery. Each stream saved four task-boundary checkpoints and a 5-stage-by-4-task test matrix. The aggregate report is `reports/baseline_full_analysis.md`; combined metrics and a 3,000-prediction ID audit are under `D:\Task-Aware-TransNAS\results\baseline_full`.

To reproduce the frozen formal run after activating the local environment:

```powershell
python .\scripts\prepare_citb.py --config .\configs\continual_baseline_full.yaml
python .\scripts\analyze_citb_task_relations.py
python .\scripts\run_formal_baseline.py --phase validate
python .\scripts\run_formal_baseline.py --phase tune --reset-tuning
python .\scripts\build_task_stream_manifest.py
python .\scripts\run_formal_baseline.py --phase run --sequence homogeneous
python .\scripts\run_formal_baseline.py --phase run --sequence heterogeneous
python .\scripts\run_formal_baseline.py --phase run --sequence mixed
python .\scripts\summarize_formal_runs.py
```

To repeat development-only tuning first, run `python .\scripts\run_formal_baseline.py --phase tune --reset-tuning`. That resets the saved hyperparameters and evaluates the nine configured candidates on dev before freezing a new choice; the test split is not read in that phase.
