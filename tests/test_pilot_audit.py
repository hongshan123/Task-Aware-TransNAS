import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pilot as audit


class PilotAuditTests(unittest.TestCase):
    def test_task611_parses_option_not_arbitrary_prose(self):
        self.assertEqual(audit.parse_classification("task611_mutual_multi_turn_dialogue", "(C) F: Let's go."), "C")
        self.assertEqual(audit.parse_classification("task611_mutual_multi_turn_dialogue", "Answer: b"), "B")
        self.assertIsNone(audit.parse_classification("task611_mutual_multi_turn_dialogue", "I think option C is plausible."))

    def test_classification_labels_normalize_punctuation(self):
        self.assertEqual(audit.parse_classification("task848_pubmedqa_classification", "Answer: 1."), "1")
        self.assertEqual(audit.parse_classification("task1500_dstc3_classification", "Don't Care."), "Don't Care")
        self.assertEqual(audit.parse_classification("task1500_dstc3_classification", "The price range is Moderate."), "Moderate")
        self.assertIsNone(audit.parse_classification("task1500_dstc3_classification", "Don't Know"))

    def test_generation_metrics_handle_overlap_and_multiple_references(self):
        scores = audit.generation_scores("We should work with Austria tomorrow.", ["We should work with Austria.", "A different response."])
        self.assertGreater(scores["rouge_l_f1"], 0.8)
        self.assertGreater(scores["token_f1"], 0.8)
        self.assertEqual(scores["exact_match"], 0.0)

    def test_cross_task_averages_use_baseline_headroom_normalization(self):
        stages = ["baseline", "after_task1", "after_task2"]
        tasks = ["class_task", "generation_task"]
        score = {
            "baseline": {
                "class_task": {"primary_metric": "accuracy", "primary_score": 0.5},
                "generation_task": {"primary_metric": "rouge_l_f1", "primary_score": 0.2},
            },
            "after_task1": {
                "class_task": {"primary_metric": "accuracy", "primary_score": 0.75},
                "generation_task": {"primary_metric": "rouge_l_f1", "primary_score": 0.1},
            },
            "after_task2": {
                "class_task": {"primary_metric": "accuracy", "primary_score": 0.6},
                "generation_task": {"primary_metric": "rouge_l_f1", "primary_score": 0.3},
            },
        }
        summary = audit.build_cl_summary(stages, tasks, score)
        self.assertAlmostEqual(summary["after_task1"]["aaa_normalized_seen_tasks"], 0.5)
        self.assertAlmostEqual(summary["after_task1"]["normalized_average_accuracy_all_tasks"], 0.1875)
        self.assertAlmostEqual(summary["after_task2"]["average_forgetting_normalized"], 0.3)
        self.assertIn("accuracy", summary["after_task1"]["average_accuracy_by_metric"])
        self.assertIn("rouge_l_f1", summary["after_task1"]["average_accuracy_by_metric"])


if __name__ == "__main__":
    unittest.main()
