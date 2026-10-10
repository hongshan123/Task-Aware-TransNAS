# Phase 6B: Cross-Task Representation Compatibility

## Protocol and scope

This matrix uses final A0/S1/seed-10 as the target M and the historical Phase 6A boundaries as reference sources: the after-task1603 checkpoint for tasks565/1603 and after-task1501 for tasks1501/1384. `C_ij` is reported in two forms: target-native primary score difference (ROUGE-L for generation, accuracy for classification) and homogeneous token-NLL improvement (final baseline NLL minus patched NLL, positive means help). Every cell uses the same locked dev IDs 25–49. The four diagonals reuse the Phase 6A locked confirmation; Phase 6A cross-type edges are reused read-only; this Phase 6B supplement measured the remaining eight same-type off-diagonal conditions (200 per-example rows).

The historical source models provide the restoration information in each `Patch_i`. This score is not a claim that the final model natively retains that behavior. The subsequent conditional residual pilot starts from the final pretrained model and does not load or forward historical checkpoints. Only train/dev data were read; no optimizer steps, test split, or NAS were used for this compatibility analysis.

## Compatibility results

Activation patching produced negative off-diagonal NLL transfer in 5/12 directed task pairs; parameter rollback did so in 5/12. Signs are defined per target; lower NLL is a positive compatibility effect. Primary metric changes remain target-native and should not be averaged across ROUGE-L and accuracy.

| Source operation | Source task | Diagonal NLL recovery | Mean off-diagonal NLL benefit | Positive transfer targets |
|---|---|---:|---:|---|
| residual_stream | 565 CIRCA | +2.248 | -1.469 | 1603 SMCalFlow |
| residual_stream | 1603 SMCalFlow | +1.455 | -3.008 | 565 CIRCA |
| attention | 1501 DSTC3 | +0.098 | +0.207 | 565 CIRCA, 1603 SMCalFlow |
| mlp | 1384 Deal-or-No-Deal | +0.015 | +0.572 | 565 CIRCA, 1603 SMCalFlow, 1501 DSTC3 |

The selected activation source operations are task-specific: residual-stream block output for tasks565/1603, Attention output for task1501, and MLP output for task1384. No single operation is uniformly compatible across the full matrix. The diagonal values show how much historically sourced signal helps its own task; off-diagonal values show that signal's transfer cost/benefit to the other tasks. This supports sharing a small residual computation through conditional routing rather than sharing historical activations or weights directly.

## Similarity and gradient relation

The prompt-representation and gradient-cosine matrices come from the existing train-only task-relation analysis (frozen Qwen3-0.6B, 32 train prompts for representations and 8 for gradients of layer-27 Attention `o_proj` plus MLP `down_proj`). Negative gradient cosine marks first-order conflict. Over only 12 directed off-diagonal pairs, Spearman correlations between those similarities and NLL compatibility are: activation—representation 0.212, gradient cosine 0.170, negative-conflict magnitude -0.025; parameter rollback—representation 0.270, gradient cosine 0.227, negative-conflict magnitude -0.151. These are exploratory, non-independent estimates; they do not establish that similarity or gradient alignment causes transfer compatibility.

The selected task pairs are not explained by one scalar relation: task565 and task1603 have high representation similarity (0.898) but modest positive gradient cosine (0.254); task1501 and task1384 have high representation similarity (0.920) and strong gradient alignment (0.952). Cross-task restoration outcomes still depend on the selected component and target metric.

## Artifacts and limits

- [Primary compatibility matrix](../figures/phase6b/compatibility_primary_heatmap.png)
- [Token-NLL compatibility matrix](../figures/phase6b/compatibility_nll_heatmap.png)
- [Similarity/gradient relation](../figures/phase6b/similarity_gradient_compatibility.png)
- Complete values: [compatibility_matrices.csv](../results/phase6b/compatibility_matrices.csv), [compatibility_manifest.json](../results/phase6b/compatibility_manifest.json)

The patch operations are selected screen winners and the dev sample is small. Within-task diagonal source operations vary by task, so the matrix is a compatibility audit of those selected recoveries rather than a fully crossed experiment over all components at all layers. Similarity estimates are computed from training examples; the transfer matrix is on dev examples. The results justify testing input-conditioned residual paths as a new architecture, but do not by themselves justify NAS.
