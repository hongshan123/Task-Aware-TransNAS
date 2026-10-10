# Generation Collapse Mechanism Analysis

## Protocol

The analysis uses the existing homogeneous-stream checkpoints: the original local Qwen3-0.6B, the checkpoint after the two generation tasks, and the checkpoints after each of the two classification tasks. It evaluates the 50-row development split for `task565_circa_answer_generation` and `task1603_smcalflow_sentence_generation`. The test split was not loaded.

Teacher-forced NLL is token-weighted over the first reference answer, using the project's existing chat-template encoder, 512-token sequence limit, and 64-token target cap. Greedy generation uses at most 64 new tokens. `Yes/No exact` is the share of generations whose normalized full output is exactly `Yes` or `No`. `P(Yes/No)` is the mean first-step probability mass on the tokenizer's `Yes`/`No` tokens, including the leading-space forms. `P(reference first)` is the mean probability of the first token in the canonical reference.

Mean generated-token counts include EOS when the model emits it. A mean count of 2.00 here corresponds to a one-token decoded `Yes`/`No` answer plus EOS.

## Results

| Checkpoint | Generation task | Token NLL ↓ | ROUGE-L ↑ | Token-F1 ↑ | Mean generated tokens | Yes/No exact | P(Yes/No) | P(reference first) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base | task565 | 4.171 | 0.197 | 0.207 | 10.26 | 0% | 0.437 | 0.040 |
| After both generation tasks | task565 | 2.720 | 0.072 | 0.074 | 9.62 | 4% | 0.207 | 0.243 |
| After first classification | task565 | 4.066 | 0.000 | 0.000 | 2.00 | 100% | 0.990 | 0.004 |
| After second classification | task565 | 5.088 | 0.000 | 0.000 | 2.00 | 100% | 1.000 | 0.000 |
| Base | task1603 | 3.351 | 0.158 | 0.173 | 26.78 | 0% | 0.028 | 0.071 |
| After both generation tasks | task1603 | 1.165 | 0.131 | 0.137 | 7.74 | 6% | 0.091 | 0.202 |
| After first classification | task1603 | 1.955 | 0.019 | 0.019 | 2.00 | 100% | 0.959 | 0.121 |
| After second classification | task1603 | 3.058 | 0.007 | 0.007 | 2.00 | 100% | 1.000 | 0.057 |

At the base checkpoint, the most common first token was `Yes` on 48% of task565 examples and `The` on 80% of task1603 examples. After generation-task training, the respective top tokens were `I` (86%) and `How` (62%). After the first classification task, `No` became the top token on 100% of both tasks. After the second classification task, `Yes` was top on 100% of task565 and 98% of task1603.

## Interpretation

The decoded failure has a strong output-behavior component: after classification training, every dev example on both generation tasks produces a one-token decoded `Yes` or `No` answer, followed by EOS. First-step probability mass on those tokens rises from 0.207/0.091 after generation training to 0.990/0.959 after the first classification task.

The evidence also shows deterioration in reference-conditioned likelihood, so the collapse is not only a decoding preference. For task565, NLL worsens from 2.720 after generation training to 5.088 after both classification tasks, while reference-first-token probability falls from 0.243 to approximately zero. For task1603, NLL worsens from 1.165 to 3.058 and reference-first-token probability falls from 0.202 to 0.057. Its final NLL remains slightly below the untuned base value of 3.351, but is much worse than its post-generation checkpoint.

**Conclusion:** classification training creates a dominant short-answer behavior and also damages the conditional likelihood of reference answers. The conditional damage is especially severe for task565. The NLL is a likelihood proxy for the canonical first reference, not a score over every alternative reference; the decoded metrics use the project's max-over-reference scoring rule.

## Artifacts

- Metrics: `D:/Task-Aware-TransNAS/results/architecture_pilot/generation_collapse/mechanism_metrics.json`
- Per-stage dev generations: `D:/Task-Aware-TransNAS/results/architecture_pilot/generation_collapse/<stage>/dev/`
- Runner: [run_phase4_diagnostics.py](../scripts/run_phase4_diagnostics.py)
