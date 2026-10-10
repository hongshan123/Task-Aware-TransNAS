from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from task_aware_transnas.cl.metrics import continual_learning_metrics, normalized_gain


class ContinualMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.order = ["a_cls", "b_gen", "c_cls"]
        self.metric_by_task = {"a_cls": "accuracy", "b_gen": "rouge_l_f1", "c_cls": "accuracy"}
        self.scores = {
            "baseline": {"a_cls": 0.2, "b_gen": 0.2, "c_cls": 0.3},
            "after_a": {"a_cls": 0.8, "b_gen": 0.2, "c_cls": 0.3},
            "after_b": {"a_cls": 0.7, "b_gen": 0.8, "c_cls": 0.3},
            "after_c": {"a_cls": 0.65, "b_gen": 1.0, "c_cls": 0.9},
        }

    def test_normalized_gain_uses_task_baseline_headroom(self) -> None:
        self.assertAlmostEqual(normalized_gain(0.8, 0.2), 0.75)
        self.assertEqual(normalized_gain(0.2, 0.2), 0.0)
        with self.assertRaises(ValueError):
            normalized_gain(1.1, 0.2)

    def test_raw_score_matrix_is_preserved(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        self.assertEqual(result["raw_task_primary_score_matrix"], self.scores)
        self.assertAlmostEqual(result["stage_metrics"]["after_c"]["raw_task_primary_scores"]["b_gen"], 1.0)

    def test_nonnegative_forgetting_clips_each_task_before_averaging(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        final = result["stage_metrics"]["after_c"]
        self.assertAlmostEqual(final["nonnegative_forgetting_by_prior_task_normalized"]["a_cls"], 0.1875)
        self.assertEqual(final["nonnegative_forgetting_by_prior_task_normalized"]["b_gen"], 0.0)
        self.assertAlmostEqual(final["average_forgetting_nonnegative_normalized"], 0.09375)
        self.assertLess(final["average_forgetting_signed_normalized_legacy"], 0.0)
        self.assertGreater(final["average_forgetting_nonnegative_normalized"], 0.0)

    def test_improvement_on_one_task_never_cancels_another_tasks_forgetting(self) -> None:
        modified = {stage: dict(row) for stage, row in self.scores.items()}
        modified["after_c"]["b_gen"] = 0.95
        result = continual_learning_metrics(modified, self.order, self.metric_by_task)
        final = result["stage_metrics"]["after_c"]
        self.assertAlmostEqual(final["nonnegative_forgetting_by_prior_task_normalized"]["a_cls"], 0.1875)
        self.assertEqual(final["nonnegative_forgetting_by_prior_task_normalized"]["b_gen"], 0.0)
        self.assertAlmostEqual(final["average_forgetting_nonnegative_normalized"], 0.09375)

    def test_backward_transfer_compares_with_immediate_acquisition_stage(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        final = result["stage_metrics"]["after_c"]
        self.assertAlmostEqual(final["backward_transfer_by_task_normalized"]["a_cls"], -0.1875)
        self.assertAlmostEqual(final["backward_transfer_by_task_normalized"]["b_gen"], 0.25)
        self.assertAlmostEqual(final["backward_transfer_normalized"], 0.03125)

    def test_seen_raw_scores_keep_accuracy_and_rouge_separate(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        seen = result["stage_metrics"]["after_c"]["mean_seen_task_score_by_metric"]
        self.assertAlmostEqual(seen["accuracy"], 0.775)
        self.assertAlmostEqual(seen["rouge_l_f1"], 1.0)
        self.assertNotIn("primary_score", seen)

    def test_cross_stage_aaa_averages_stage_local_seen_gains(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        post_stage_aaa = [
            result["stage_metrics"][stage]["mean_seen_normalized_gain"]
            for stage in ("after_a", "after_b", "after_c")
        ]
        expected = sum(post_stage_aaa) / len(post_stage_aaa)
        self.assertAlmostEqual(result["cross_stage_aaa_normalized_final"], expected)
        self.assertAlmostEqual(result["stage_metrics"]["after_c"]["cross_stage_aaa_normalized"], expected)

    def test_baseline_has_no_seen_tasks_or_forgetting(self) -> None:
        result = continual_learning_metrics(self.scores, self.order, self.metric_by_task)
        baseline = result["stage_metrics"]["baseline"]
        self.assertEqual(baseline["tasks_seen"], [])
        self.assertEqual(baseline["average_forgetting_nonnegative_normalized"], 0.0)
        self.assertEqual(baseline["backward_transfer_normalized"], 0.0)

    def test_rejects_missing_task_score(self) -> None:
        malformed = {stage: dict(row) for stage, row in self.scores.items()}
        del malformed["after_b"]["c_cls"]
        with self.assertRaises(ValueError):
            continual_learning_metrics(malformed, self.order, self.metric_by_task)


if __name__ == "__main__":
    unittest.main()
