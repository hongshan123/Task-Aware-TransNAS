# Generation and answer-format diagnostics

## Method

The same five checkpoints from the completed homogeneous stream were compared: the local base model and the checkpoints after tasks 1–4. Dev outputs were freshly generated with `local_files_only=True`; test outputs were read from the already saved formal run for the matching checkpoint. Each split has 50 examples per task. No training occurred, and dev/test comparisons here do not select new hyperparameters.

For output lengths, word counts use the project tokenizer's word splitter; token counts use the model tokenizer on the decoded output. Empty output is measured separately. For Yes/No tasks, format errors are parser failures. For free-text tasks, the diagnostic marks a one-token Yes/No answer, empty output, leaked chat/control markers, or echoed prompt labels as a format error. This is a format heuristic; ROUGE-L remains the quality score.

## Free-text generation by checkpoint

Each pair is **dev / test**. Format-error includes the short Yes/No label collapse that appears after the classification stage.

### task565 (Circa answer generation)

| Checkpoint | ROUGE-L | Mean words | Mean tokenizer tokens | Empty rate | Format-error rate | EOS termination |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.1973 / 0.1927 | 7.1800 / 7.7200 | 9.2600 / 9.6600 | 0.0% / 0.0% | 0.0% / 2.0% | 100.0% / 100.0% |
| After task 1 | 0.2499 / 0.2003 | 5.1200 / 5.4400 | 6.6000 / 6.9200 | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% |
| After task 2 | 0.0724 / 0.0954 | 6.3400 / 6.1400 | 8.6200 / 8.6000 | 0.0% / 0.0% | 4.0% / 0.0% | 100.0% / 100.0% |
| After task 3 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| After task 4 | 0.0000 / 0.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |

### task1603 (SMCalFlow sentence generation)

| Checkpoint | ROUGE-L | Mean words | Mean tokenizer tokens | Empty rate | Format-error rate | EOS termination |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.1579 / 0.1068 | 19.1000 / 18.6400 | 25.8000 / 25.1800 | 0.0% / 0.0% | 0.0% / 0.0% | 98.0% / 96.0% |
| After task 1 | 0.0957 / 0.0954 | 5.9800 / 6.1200 | 8.0400 / 8.1000 | 0.0% / 0.0% | 0.0% / 0.0% | 100.0% / 100.0% |
| After task 2 | 0.1313 / 0.2352 | 4.5400 / 5.5000 | 6.7400 / 8.5000 | 0.0% / 0.0% | 6.0% / 4.0% | 100.0% / 98.0% |
| After task 3 | 0.0188 / 0.0224 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| After task 4 | 0.0068 / 0.0400 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |

## Yes/No parsing by checkpoint

| Checkpoint | task1501 Accuracy dev/test | task1501 parse success dev/test | task1384 Accuracy dev/test | task1384 parse success dev/test |
|---|---:|---:|---:|---:|
| Base | 0.7600 / 0.6400 | 100.0% / 100.0% | 0.8800 / 0.8400 | 100.0% / 100.0% |
| After task 1 | 0.0000 / 0.0000 | 0.0% / 0.0% | 0.0000 / 0.0000 | 0.0% / 0.0% |
| After task 2 | 0.1200 / 0.1600 | 16.0% / 18.0% | 0.0000 / 0.0000 | 0.0% / 0.0% |
| After task 3 | 0.9000 / 0.7400 | 100.0% / 100.0% | 0.2800 / 0.2400 | 100.0% / 100.0% |
| After task 4 | 0.8600 / 0.8600 | 100.0% / 100.0% | 0.9000 / 0.8200 | 100.0% / 100.0% |

## Representative dev predictions

The first saved dev example for each generation task is shown at the base and final checkpoint.

| Task | Instance | Reference | Base output | Final output |
|---|---|---|---|---|
| task565 | `task565-f110ed857dc84e6eb8f82fb0f15a5b9d` | It's my fiance's birthday. | None. | Yes |
| task1603 | `task1603-6fb41960ddda4b4190d0b1140f114003` | Is this good? | The Agent's response would be: "Sure! Let me know when you're available to view the apartment at Lincoln Square. We'll be at 10 am tomorrow morning." | Yes |

## Finding

The initial task565 comparison that looked contradictory used different checkpoints. On the base checkpoint, task565 is 0.1973 on dev and 0.1927 on test (gap +0.0046). On the final tuned checkpoint, it is 0.0000 on both dev and test. Thus the roughly 0.19-to-zero change is checkpoint-dependent generation collapse, not a dev/test split gap for task565.

task1603 starts at 0.1579 dev / 0.1068 test and ends at 0.0068 dev / 0.0400 test. The base split gap is +0.0511 and the final gap is −0.0332; these are smaller than the change across checkpoints and do not explain the final near-zero performance.

The length and format diagnostics show the failure mode: after task 3 (task1501, a Yes/No classification task), both generation tasks produce one-token answers on every evaluated example, usually `Yes`. Their dev/test mean lengths drop to 1 word and 1 tokenizer token. These responses are not empty; EOS termination remains 100%. This points to response-mode collapse after full-parameter sequential SFT on the short-label task, rather than truncation, empty generations, or a failure to stop generation. This is an observed association in this run, not a causal estimate.

At the base checkpoint, both classification tasks parse every dev and test answer. After the first generation task, parser success falls to zero for both. After task 3 it returns to 100% for task1501, while task1384 remains at 0.28 dev / 0.24 test Accuracy with all outputs parseable; after task 4 both are again fully parseable. The class table separates answer formatting from class accuracy.

Machine-readable stage metrics: `D:/Task-Aware-TransNAS/results/baseline_full/generation_diagnostics/metrics.json`. The representative JSONL contains three records for every task, checkpoint, and split: `D:\Task-Aware-TransNAS\results\baseline_full\generation_diagnostics\representative_predictions.jsonl`. Full fresh dev predictions are stored under `D:/Task-Aware-TransNAS/results/baseline_full/generation_diagnostics/predictions/`; the original test predictions remain in each formal sequence directory.
