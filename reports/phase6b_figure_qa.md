# Phase 6B Figure QA

## Scope and interpretation

The four figures summarize the seed-10 pilot on 50 locked development examples per task, across task orders S1 and S2. The test split was not loaded. These are pilot diagnostics, not confirmatory estimates. With one seed, no between-seed uncertainty interval is available; the figures do not show confidence intervals or error bands.

The results do not show a stable B2-input advantage over parameter-matched B1-static. The plots should be read alongside `phase6b_pilot_results.md` and `phase6b_architecture_design.md`; they do not establish a causal routing mechanism.

## Figure set

| Figure | Content | Export |
| --- | --- | --- |
| `pilot_final_primary_scores` | Primary task metrics for each architecture and task order, with generation and classification metrics kept task-specific. | PDF, editable SVG, 300 dpi PNG preview |
| `pilot_b2_vs_b1_delta` | Per-task B2-input minus B1-static primary-score differences for S1 and S2. | PDF, editable SVG, 300 dpi PNG preview |
| `pilot_b2_route_controls` | B2 route-control comparisons across tasks and task orders. | PDF, editable SVG, 300 dpi PNG preview |
| `pilot_b2_route_usage` | Token routing usage across the eight sequence-task cases and layer/expert routes. | PDF, SVG, 300 dpi PNG preview |

The multi-panel exports are 182.9 mm wide. PDF/SVG are the primary editable exports; PNG files are previews. The redundant 600 dpi TIFF exports were omitted to keep the figure bundle small. This is a draft/research bundle and has not been prepared to a journal-specific submission contract.

## Automated review

| Check | Result |
| --- | --- |
| Source preflight (`summarize_phase6b_pilot.py`) | 18 PASS, 0 FAIL, 3 WARN; `ready: true` |
| PDF text audit, 5 pt minimum | All four auditable; minimum text size is 8 pt (10 pt for route usage); zero runs below 5 pt |
| Rendered collision audit | All four stored audit reports say PASS; zero FAIL and zero WARN |
| Multi-panel alignment | All three alignment reports say PASS at 1.5 pt tolerance; four comparisons per figure |
| Visual inspection | All four PNG previews inspected at the exported layout size; labels, task/order grouping, and legends are legible without overlap |

The three source-preflight warnings are documented choices: no TIFF is included, the PNG previews are 300 dpi rather than the validator's 600 dpi default, and no uncertainty band is drawn. The last warning is expected for a one-seed pilot and must not be read as an uncertainty estimate. The source checker flags seed/mean terms in the plotting input, but there is no multi-seed aggregation behind these plotted results.

## Reproduction

Run `python scripts/summarize_phase6b_pilot.py` after making the local Phase 6B run metrics available under `results/phase6b/`. This rebuilds the summary tables, reports, and figures. The large model checkpoints and raw run outputs remain local and are excluded from Git.
