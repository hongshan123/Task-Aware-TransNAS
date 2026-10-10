# CITB Task Relation Analysis

This selection analysis used only order1 training records after the official 50-test / 50-dev holdouts. Each eligible candidate has exactly 500 training examples after the cap. No test scores or test generations were used.

Frozen Qwen3-0.6B representations: mean last-token prompt embeddings over 32 training records/task. Supervised gradient similarity: cosine of mean gradients over 8 training records/task for the final-block o_proj and down_proj weights. Each analysis prompt retained the full task definition and truncated only the input tail to fit 384 tokens.

## Selection constraints and measured pairing

The selected set is constrained to contain two tasks from each of two official CITB categories and two tasks of each reference format (classification or generation). Each similarity pair is homogeneous in both category and reference format; the two pairs differ on both. Within-pair and cross-pair ordering is then chosen from the measured combined similarity.

{
  "pair_a": [
    "task565_circa_answer_generation",
    "task1603_smcalflow_sentence_generation"
  ],
  "pair_b": [
    "task1501_dstc3_answer_generation",
    "task1384_deal_or_no_dialog_classification"
  ],
  "within_similarity_z": 2.1735218167304993,
  "minimum_pair_similarity_z": 1.1862355470657349,
  "cross_similarity_z": -0.04057122766971588,
  "separation_z": 2.214093044400215,
  "selection_score": 2.510651931166649,
  "category_pair_a": "Dialogue Generation",
  "category_pair_b": "Dialogue State Tracking",
  "output_type_pair_a": "generation",
  "output_type_pair_b": "classification"
}

## Three reproducible sequences

### Homogeneous

task565_circa_answer_generation -> task1603_smcalflow_sentence_generation -> task1501_dstc3_answer_generation -> task1384_deal_or_no_dialog_classification

Mean adjacent combined similarity z: 1.534

### Heterogeneous

task565_circa_answer_generation -> task1501_dstc3_answer_generation -> task1603_smcalflow_sentence_generation -> task1384_deal_or_no_dialog_classification

Mean adjacent combined similarity z: 0.080

### Mixed

task565_circa_answer_generation -> task1501_dstc3_answer_generation -> task1384_deal_or_no_dialog_classification -> task1603_smcalflow_sentence_generation

Mean adjacent combined similarity z: 1.049

## Highest measured pairwise similarities

| Task A | Task B | Frozen representation cosine | Gradient cosine | Combined z | Same official category |
|---|---|---:|---:|---:|---|
| task1501_dstc3_answer_generation | task1384_deal_or_no_dialog_classification | 0.9201 | 0.9518 | 3.161 | True |
| task294_storycommonsense_motiv_text_generation | task1384_deal_or_no_dialog_classification | 0.8012 | 0.9766 | 2.523 | False |
| task294_storycommonsense_motiv_text_generation | task1501_dstc3_answer_generation | 0.7878 | 0.9331 | 2.329 | False |
| task1500_dstc3_classification | task1501_dstc3_answer_generation | 0.8220 | 0.4475 | 1.248 | True |
| task565_circa_answer_generation | task1603_smcalflow_sentence_generation | 0.8981 | 0.2539 | 1.186 | True |
| task1600_smcalflow_sentence_generation | task1603_smcalflow_sentence_generation | 0.8763 | 0.2980 | 1.174 | True |
| task1600_smcalflow_sentence_generation | task1501_dstc3_answer_generation | 0.7905 | 0.4632 | 1.103 | False |
| task1500_dstc3_classification | task1384_deal_or_no_dialog_classification | 0.7994 | 0.4396 | 1.094 | True |
| task294_storycommonsense_motiv_text_generation | task1500_dstc3_classification | 0.7736 | 0.4298 | 0.915 | False |
| task294_storycommonsense_motiv_text_generation | task1600_smcalflow_sentence_generation | 0.7455 | 0.4634 | 0.838 | False |
| task576_curiosity_dialogs_answer_generation | task1600_smcalflow_sentence_generation | 0.8656 | 0.1876 | 0.819 | True |
| task1600_smcalflow_sentence_generation | task1384_deal_or_no_dialog_classification | 0.7378 | 0.4692 | 0.808 | False |
| task565_circa_answer_generation | task1729_personachat_generate_next | 0.8643 | 0.1827 | 0.798 | True |
| task1600_smcalflow_sentence_generation | task1729_personachat_generate_next | 0.8266 | 0.2580 | 0.775 | True |
| task1714_convai3_sentence_generation | task576_curiosity_dialogs_answer_generation | 0.8612 | 0.1731 | 0.755 | True |
| task1714_convai3_sentence_generation | task1600_smcalflow_sentence_generation | 0.8405 | 0.2090 | 0.727 | True |
| task611_mutual_multi_turn_dialogue | task1501_dstc3_answer_generation | 0.7093 | 0.4853 | 0.682 | False |
| task576_curiosity_dialogs_answer_generation | task1501_dstc3_answer_generation | 0.8252 | 0.2150 | 0.653 | False |
| task1729_personachat_generate_next | task1603_smcalflow_sentence_generation | 0.8715 | 0.1034 | 0.632 | True |
| task576_curiosity_dialogs_answer_generation | task1603_smcalflow_sentence_generation | 0.8642 | 0.0973 | 0.572 | True |

Full pair matrices, candidate counts/categories, sampled training IDs, gradient norms, and token truncation diagnostics are in D:\Task-Aware-TransNAS\data\task_selection\task_relation_analysis.json.
