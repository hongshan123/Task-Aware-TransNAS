# Phase 5 Figure QA

## Figure claim and evidence map

**Claim.** At the Phase 4B A0 endpoint, the old generation tasks show distinct causal head-effect profiles; task 565 has little final Function Vector cosine with other tasks, while task 1603 retains moderate cosine with both classification tasks. Across five homogeneous A0 stages, Function Vector drift is only descriptively related to held-out NLL. These figures motivate a constrained Conditional Routing pilot and do not establish that routing improves performance.

| Figure | Inferential role | Source data |
|---|---|---|
| Causal head-importance heatmap | Compare the sign and magnitude of sequence log-probability gain from patching each of 448 query heads across four tasks. A single symmetric scale is shared across panels. | results/phase5/figure_data/phase5_head_importance.csv |
| Function Vector cosine matrix | Compare task-vector directions at the Phase 4B A0 final checkpoint. | results/phase5/figure_data/phase5_fv_cosine_matrix.csv |
| Stage drift and NLL | Show Function Vector cosine to pretrained beside token NLL across the pretrained checkpoint and four post-task checkpoints. | results/phase5/figure_data/phase5_stage_drift_vs_nll.csv |

Task labels used in figures:

| Task ID | Display label |
|---|---|
| 565 | CIRCA |
| 1603 | SMCalFlow |
| 1501 | DSTC3 |
| 1384 | Deal/No Deal |

## Data and statistical notes

- Head effects and rankings use the build dev subset, indices 0–9, with task demonstrations drawn from train. Each task contributes 10 query prompts. The reported heatmap value is the mean change in reference-sequence log probability from activation patching; positive values favor the reference answer.
- The cosine matrix contains all 16 ordered task pairs at the Phase 4B A0 final checkpoint. Cosine is descriptive and symmetric; its diagonal is 1 by definition.
- The stage plot uses five checkpoints from the homogeneous A0 trajectory. Each task and checkpoint has 25 locked evaluation dev examples (indices 25–49). The NLL is the cached teacher-forced token NLL. Lines connect checkpoint stages to make the trajectory readable; no causal or inferential trend claim is made.
- The intervention-tuning subset, indices 10–24, is not used to form these plotted measurements. Test data were not read.
- The plots show observed summaries without confidence intervals. For the stage plot, the experimental units shown are five checkpoints per task; the NLL values aggregate the 25 evaluation examples at each checkpoint.
- These are data plots, not image plates. No source imagery, crop, local contrast adjustment, pseudo-coloring, stitching, or image reuse is involved.

## Layout and export

All plots were drawn, previewed, exported, and visually inspected with Python/Matplotlib. Text remains editable in SVG and PDF. The head heatmap and stage figure use 182.9 mm width; the cosine matrix uses 88.9 mm width. Each figure is exported as PNG preview, SVG, PDF, and 600 dpi TIFF. CSV tables preserve the values rendered in each plot.

The head heatmap uses one zero-centered diverging scale, whose limit is the maximum absolute observed head effect across all four tasks; no values are clipped. The cosine matrix uses the full cosine domain from −1 to 1. The stage figure uses one color per task across both panels. Figure-label task abbreviations are mapped above.

## Automated and visual QA

| Check | Result |
|---|---|
| Plot-source validator | 21 PASS, 0 WARN, 0 FAIL |
| Head-heatmap alignment | PASS; 4 comparisons, 1.5 pt tolerance |
| Stage-figure alignment | PASS; 1 comparison, 1.5 pt tolerance |
| Cosine-matrix alignment | Not applicable; single plot area |
| PDF text-size audit | All PDFs auditable; minimum text run 6 pt, no run below 5 pt |
| PDF collision audit | All three PDFs PASS; 0 FAIL and 0 WARN |
| Repository tests | 22 passed, 0 failed |
| Phase 5 structural audit | PASS; all result-cache checks passed |
| Final-size visual inspection | Labels, shared heatmap scale, annotations, axes, and legend are legible; no visible collisions |

Audit outputs and measured alignment reports are stored beside the figure exports in figures/phase5/. The render-time alignment utility is included at scripts/figure_qa/audit_panel_alignment.py.

## Reproduction

Regenerate the figures and source CSV files from cached results with:

    .\conda-env\python.exe scripts\plot_phase5_figures.py

The complete analysis command is:

    .\conda-env\python.exe scripts\run_phase5_fv.py
