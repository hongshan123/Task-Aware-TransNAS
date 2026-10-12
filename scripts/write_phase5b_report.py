"""Render the auditable Phase 5B findings from saved machine-readable outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch
import transformers
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from .phase_paths import storage_path
except ImportError:  # Running this file directly from scripts/.
    from phase_paths import storage_path
ROOT = storage_path("results") / "phase5b"
REPORT = storage_path("reports") / "phase5b_fv_recovery.md"

SHORT = {
    "task565_circa_answer_generation": "565",
    "task1603_smcalflow_sentence_generation": "1603",
    "task1501_dstc3_answer_generation": "1501",
    "task1384_deal_or_no_dialog_classification": "1384",
}
CLASSIFICATION_TASKS = {
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification",
}


def _f(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _p(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    replay = _read_json(ROOT / "boundary_replay_manifest.json")
    fv = _read_json(ROOT / "extraction" / "historical_vector_descriptors.json")
    metrics = _read_json(ROOT / "interventions" / "metrics.json")
    head = _read_json(ROOT / "head_similarity" / "full_448_analysis.json")
    tune = _read_json(ROOT / "interventions" / "tuning_grid.json")
    preflight = _read_json(ROOT / "interventions" / "preflight.json")
    intervention_config = _read_json(ROOT / "interventions" / "run_config.json")
    experiment_config = yaml.safe_load((PROJECT_ROOT / "configs" / "phase5b_fv_recovery.yaml").read_text(encoding="utf-8"))

    by_task_condition = {(row["task_id"], row["condition"]): row for row in metrics["rows"]}
    fv_rows = {row["task_id"]: row for row in fv["summaries"]}
    lines = [
        "# Phase 5B: Function Vector Recovery Verification",
        "",
        "## Scope and provenance",
        "",
        "This analysis tests whether same-trajectory historical Function Vectors restore old task behavior after the A0/S1 task565 → task1603 → task1501 → task1384 stream. The historical task565 and task1603 checkpoints had not been retained by the original run, so the minimum required S1 sequence was replayed into the Phase 5B output directory. The replay manifest records exact agreement with the original final checkpoint across **596,049,920 parameters** (maximum absolute difference 0), and exact equality of sample order, supervised tokens, optimizer steps, and recorded losses at every task stage. Original Phase 4B/5 outputs were only read; all new experiment results are under `results/phase5b/`.",
        "",
        "The replay used the same task order, dataset ID hashes, task seeds, optimizer settings, and training rows. It did not load dev during training or load test; NAS was not run. The two historical vectors use the model immediately after task565 and immediately after task1603, respectively.",
        "",
        "## Method",
        "",
        "* Historical vectors: for each task, average the last-token, per-query-head `o_proj` contributions on the same 10 four-shot correct-ICL build prompts used by Phase 5. Rank all 448 `(layer, query_head)` candidates by sequence log-probability gain over the label-shuffled ICL counterfactual, then sum the top 10 mean head contributions. Each historical scan stores all 448 per-query effects and train support IDs.",
        "* Checkpoint controls: compare the historical vector with the same task's Phase 5 final collapsed vector, a seeded norm-matched random vector, the other generation task's historical vector, and subtraction of the final task1384 classifier vector.",
        "* Tuning: each available injection vector was tuned separately on dev indices 10–24 (15 examples/task), over layers `[5, 12, 19, 26]` and scales `[0.25, 0.5, 1, 2]`, minimizing token-weighted teacher-forced reference NLL. The selected setting was then locked.",
        "* Locked evaluation: all 25 examples/task at dev indices 25–49. Baseline, vector conditions, a format-instruction condition, and a correct-label four-shot ICL condition were evaluated. Four ICL demonstrations came only from train and all four labels were preserved.",
        "* Injection path: at the selected Qwen3 layer, the FV is added directly to `self_attn.o_proj` output, the same 1024-wide attention-output space used to construct the per-head contributions. Teacher forcing patches from the final prompt position through reference tokens; cached generation patches the final output position on prefill and on every single-token decode call. Greedy generation is capped at 64 tokens. Paired bootstrap intervals use 5,000 resamples of the 25 locked IDs.",
        "",
        "The intervention preflight verified Qwen3 geometry `[28 layers, 16 query heads, 8 KV heads, 128 head_dim, 1024 hidden]`, the selected `self_attn.o_proj` output hook and its 1024-wide tensor, a zero-vector no-op (NLL-sum difference 0), the expected teacher-forced patch count, and both prefill and cached decode hook events. The test split was not loaded.",
        "",
        "## Historical vectors",
        "",
        "| Historical boundary | Task | L2 norm | Cosine to same-task final FV | Cosine to same-task pretrained FV |",
        "|---|---|---:|---:|---:|",
    ]
    for task_id in ("task565_circa_answer_generation", "task1603_smcalflow_sentence_generation"):
        row = fv_rows[task_id]
        lines.append(
            f"| After task {SHORT[task_id]} | {task_id} | {_f(row['vector_l2_norm'])} | "
            f"{_f(row['cosine_to_same_task_final_collapsed_fv'])} | {_f(row['cosine_to_same_task_pretrained_fv'])} |"
        )
    lines.extend([
        "",
        "The vectors point in directions distinct from the final collapsed vectors (cosines near zero). A low cosine alone does not establish recovery; the controlled locked evaluation below is the behavioral check.",
        "",
        "## Locked development results",
        "",
        "ROUGE-L is shown for every task; it is the primary generation measure for the two old generation tasks. `NLL` is token-weighted reference NLL for the exact prompt condition. Generated length includes EOS when emitted.",
        "",
        "| Task | Condition | ROUGE-L | Parsed accuracy | Token NLL | Generated tokens | Exact Yes | Exact No | Exact Yes/No | Most-common label among parseable | Parseable | Invalid format |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(metrics["rows"], key=lambda item: (SHORT[item["task_id"]], item["condition"])):
        lines.append(
            f"| {SHORT[row['task_id']]} | {row['condition']} | {_f(row['rouge_l_f1'])} | "
            f"{_f(row['parsed_accuracy']) if row['task_id'] in CLASSIFICATION_TASKS else '—'} | "
            f"{_f(row['teacher_forced_token_nll'])} | {_f(row['mean_generated_token_count_including_eos'], 2)} | "
            f"{_p(row['exact_yes_output_rate'])} | {_p(row['exact_no_output_rate'])} | {_p(row['exact_yes_no_output_rate'])} | "
            f"{_p(row['most_common_label_concentration_among_parseable'])} | "
            f"{_p(row['parseable_yes_no_label_rate'])} | {_p(row['invalid_answer_format_rate'])} |"
        )

    lines.extend([
        "",
        "### What the output-format measures mean",
        "",
        "* **Exact Yes/No output rate** counts only outputs that equal `Yes` or `No` after NFKC normalization, trimming, and case folding, divided by all outputs.",
        "* **Most-common label concentration** is the larger parsed Yes/No count divided by parseable outputs only.",
        "* **Invalid answer-format rate** is the share for which the existing unambiguous Yes/No parser returns no label.",
        "",
        "The legacy Phase 5 field `yes_no_collapse_rate` is parser-based: it took the larger parsed label count divided by all outputs. It is neither a strict exact-string rate nor the new concentration denominator when some outputs are unparseable. Phase 5 artifacts were left unchanged. In this Phase 5B locked sample, all outputs parsed and were exactly Yes/No (100% exact and 0% invalid), but label concentration differs: task565 was 100% one label, task1603 was 92%, task1384 was 84%, and task1501 was 56%. Thus a 100% exact Yes/No rate alone does not mean all examples collapsed to the same label.",
        "",
        "### Direct historical-vector contrasts",
        "",
        "Paired deltas below are historical FV minus the named control. For NLL, negative values favor the historical FV. For ROUGE-L, positive values favor the historical FV.",
        "",
        "| Task | Contrast | Mean per-example NLL delta (95% CI) | ROUGE-L delta (95% CI) |",
        "|---|---|---:|---:|",
    ])
    contrasts = metrics["direct_historical_fv_vs_control_bootstrap"]
    contrast_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in contrasts:
        if row["metric"].startswith("mean_per_example_nll"):
            metric_key = "nll"
        elif row["metric"].startswith("rouge_l_f1"):
            metric_key = "rouge"
        else:
            continue
        contrast_groups.setdefault((row["task_id"], row["contrast"]), {})[metric_key] = row
    for (task_id, contrast), values in sorted(contrast_groups.items()):
        nll_row = values["nll"]
        rouge_row = values["rouge"]
        nll_cell = f"{_f(nll_row['mean_delta'])} [{_f(nll_row['ci95_low'])}, {_f(nll_row['ci95_high'])}]"
        rouge_cell = f"{_f(rouge_row['mean_delta'])} [{_f(rouge_row['ci95_low'])}, {_f(rouge_row['ci95_high'])}]"
        lines.append(f"| {SHORT[task_id]} | {contrast.replace('historical_task_boundary_fv_minus_', '')} | {nll_cell} | {rouge_cell} |")

    lines.extend([
        "",
        "For task565, the historical vector improves teacher-forced NLL over the final, random, and unrelated controls, while all 25 generated answers remain the same collapsed `Yes` response (ROUGE-L stays 0). This supports a task-specific effect on reference likelihood, but not recovery of generated text. For task1603, the historical vector improves paired NLL over random, but the interval against the unrelated historical vector reaches zero and it performs worse than the final collapsed vector. The historical injection, format hint, and true ICL preserve baseline outputs on all 25 IDs; the final vector changes one label but leaves ROUGE-L at 0.014 and length at two tokens. Evidence for task1603-specific recovery is weak.",
        "",
        "The explicit format instruction and true four-shot ICL did not change the old tasks' generated outputs or ROUGE-L. They did not restore the missing generation behavior. A few NLL shifts are measurable, but NLL changes alone do not recover the output function.",
        "",
        "## Full 448-head causal-effect similarity",
        "",
        "For each task, this compares the mean sequence-log-probability gain vector over all 448 query heads from 10 Phase 5 build prompts on the final collapsed checkpoint. Pearson/Spearman correlations and positive-gain weighted overlap are tested against 10,000 permutations that independently shuffle one task's head identities while preserving its score distribution. P values are BH-adjusted across the six task pairs within each statistic.",
        "",
        "| Pair | Pearson r (BH p) | Spearman ρ (BH p) | Positive-gain weighted overlap (BH p) | Top-10 overlap (permutation p) |",
        "|---|---:|---:|---:|---:|",
    ])
    with (ROOT / "head_similarity" / "pairwise_similarity.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            a, b = SHORT[row["task_a"]], SHORT[row["task_b"]]
            lines.append(
                f"| {a}–{b} | {_f(row['pearson'], 3)} ({_f(row['pearson_permutation_p_bh_6_pairs'], 4)}) | "
                f"{_f(row['spearman'], 3)} ({_f(row['spearman_permutation_p_bh_6_pairs'], 4)}) | "
                f"{_f(row['positive_gain_weighted_overlap'], 3)} ({_f(row['weighted_overlap_permutation_p_bh_6_pairs'], 4)}) | "
                f"{row['top_10_overlap']} ({_f(row['top_10_permutation_p'], 4)}) |"
            )
    lines.extend([
        "",
        "Top-10 query-head sets are also unstable across build prompts. Mean within-task pairwise Top-10 Jaccard over the 45 pairs of prompts is:",
        "",
        "| Task | Mean Top-10 Jaccard | Range |",
        "|---|---:|---:|",
    ])
    for row in head["within_task_topk_stability"]:
        if row["top_k"] == 10:
            lines.append(
                f"| {SHORT[row['task_id']]} | {_f(row['mean_jaccard'])} | "
                f"[{_f(row['minimum_jaccard'])}, {_f(row['maximum_jaccard'])}] |"
            )
    lines.extend([
        "",
        "The task565 Top-10 set has zero overlap with every other task, but zero overlap is not itself evidence of isolation: the 10,000-permutation null gives p=1.000 for each such observed zero, with only 448 candidate heads and very unstable per-query rankings. At the full-distribution level, several task pairs do differ from randomized head identity after BH correction, while other pairs do not. The strongest same-direction positive-gain overlap is between tasks565 and 1603 (0.566, adjusted p<0.001). These findings show structured relationships in some full profiles, not clean head partitions.",
        "",
        "## Conclusion and architecture guidance",
        "",
        "1. **Historical FV recovery is task-dependent.** Task565's historical vector improves reference NLL beyond all required controls, but it does not restore generated text. Task1603 lacks clear specificity over its unrelated historical vector and performs worse than its final FV.",
        "2. **Formatting is not the main failure here.** The model already emits exact Yes/No strings with zero invalid-format responses; explicit instruction and correct ICL do not restore the two old generation tasks. Their outputs are label-concentrated and short, whereas the task references require open-text generation.",
        "3. **Some head-profile differences exceed a random identity null, but Top-10 evidence is weak.** Full-profile correlation tests identify pair-specific structure. Top-10 zero overlap is common under the null and Top-10 rankings are unstable across queries, so it cannot establish task isolation.",
        "4. **Architecture decision:** the evidence motivates only a small, controlled Conditional Routing pilot with the same all-head randomization and task-agnostic controls. It does not validate a routing benefit or justify broad NAS. Head Isolation is not supported by zero Top-10 overlap alone; retain it as a comparison arm only if the pilot independently demonstrates stable head groups.",
        "",
        "## Reproduction and artifacts",
        "",
        "Run from the project root in the existing local Windows environment. Model loading uses `local_files_only=True`; no model download occurs.",
        "",
        "```powershell",
        r".\conda-env\python.exe scripts\replay_phase5b_boundaries.py --preflight-only",
        r".\conda-env\python.exe scripts\replay_phase5b_boundaries.py",
        r".\conda-env\python.exe scripts\extract_phase5b_historical_fv.py --preflight-only",
        r".\conda-env\python.exe scripts\extract_phase5b_historical_fv.py",
        r".\conda-env\python.exe scripts\analyze_phase5b_heads.py",
        r".\conda-env\python.exe scripts\run_phase5b_interventions.py --preflight-only",
        r".\conda-env\python.exe scripts\run_phase5b_interventions.py",
        r".\conda-env\python.exe scripts\write_phase5b_report.py",
        r".\conda-env\python.exe scripts\audit_phase5b.py",
        "```",
        "",
        "Key artifacts: `configs/phase5b_fv_recovery.yaml`, `results/phase5b/boundary_replay_manifest.json`, `results/phase5b/extraction/`, `results/phase5b/head_similarity/`, `results/phase5b/interventions/metrics.json`, `results/phase5b/interventions/tuning_grid.json`, per-condition locked predictions/NLL under `results/phase5b/interventions/evaluation/`, and `results/phase5b/audit.json`.",
        "",
        "The split discipline is build dev[0:10], tune dev[10:25], locked evaluation dev[25:50], and demonstrations from train only. Test was not loaded; formal follow-up training and NAS were not run.",
        "",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    source_paths = [
        PROJECT_ROOT / "configs" / "phase5b_fv_recovery.yaml",
        PROJECT_ROOT / "scripts" / "replay_phase5b_boundaries.py",
        PROJECT_ROOT / "scripts" / "extract_phase5b_historical_fv.py",
        PROJECT_ROOT / "scripts" / "analyze_phase5b_heads.py",
        PROJECT_ROOT / "scripts" / "run_phase5b_interventions.py",
        PROJECT_ROOT / "scripts" / "phase5b_common.py",
        PROJECT_ROOT / "scripts" / "audit_phase5b.py",
        PROJECT_ROOT / "scripts" / "write_phase5b_report.py",
    ]
    canonical_config = {
        "phase": "5B",
        "input_yaml": experiment_config,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
        "intervention_runtime_and_selected_parameters": intervention_config,
        "intervention_preflight": preflight,
        "replay_certification": {
            "manifest_path": "results/phase5b/boundary_replay_manifest.json",
            "manifest_sha256": _sha256(ROOT / "boundary_replay_manifest.json"),
            "status": replay.get("status"),
            "final_weight_comparison": replay.get("final_weight_comparison"),
            "comparison_with_original_training": replay.get("comparison_with_original_training"),
            "train_instance_id_sha256_by_task": replay.get("train_instance_id_sha256_by_task"),
            "test_split_loaded": replay.get("test_split_loaded"),
            "nas_started": replay.get("nas_started"),
        },
        "historical_vector_extraction": {
            "preflight": _read_json(ROOT / "extraction" / "preflight.json"),
            "descriptors": fv,
        },
        "head_similarity_config_and_results": {
            "permutation_repeats": head["permutation_repeats"],
            "random_seed": head["random_seed"],
            "permutation_null": head["permutation_null"],
            "top_k": head["top_k"],
            "full_result_path": "results/phase5b/head_similarity/full_448_analysis.json",
        },
        "metrics_definitions_and_results_path": {
            "definitions": metrics["metric_definitions"],
            "result_path": "results/phase5b/interventions/metrics.json",
            "evaluation_cache_manifest": metrics["evaluation_cache_manifest"],
        },
        "source_file_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths},
        "outputs": {
            "report": "reports/phase5b_fv_recovery.md",
            "audit": "results/phase5b/audit.json",
            "test_split_loaded": False,
            "nas_started": False,
        },
    }
    (ROOT / "run_config.json").write_text(json.dumps(canonical_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
