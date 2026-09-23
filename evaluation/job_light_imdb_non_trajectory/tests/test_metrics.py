from __future__ import annotations

import math
import unittest

from evaluation.job_light_imdb_non_trajectory.joblight_eval.metrics import (
    EPSILON,
    raw_q_error,
    smoothed_q_error,
    summarize_latency,
    summarize_predictions,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
    LatencyRecord,
    PredictionRecord,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.runner import (
    _summarize_inference_profiles,
)


class MetricsTest(unittest.TestCase):
    def test_raw_and_smoothed_q_error(self) -> None:
        self.assertEqual(raw_q_error(20, 10), 2.0)
        self.assertEqual(smoothed_q_error(0.2, 10), 10.0)
        self.assertEqual(raw_q_error(0, 0), 1.0)
        self.assertEqual(raw_q_error(0, 10), 10 / EPSILON)

    def test_percentiles_and_sub_one_counters(self) -> None:
        records = [
            PredictionRecord("w", 0, "ok", 10, 0.0),
            PredictionRecord("w", 1, "ok", 10, 0.005),
            PredictionRecord("w", 2, "ok", 10, 0.5),
            PredictionRecord("w", 3, "ok", 10, 10.0),
            PredictionRecord("w", 4, "unsupported", 10, None),
        ]
        summary = summarize_predictions(records)
        self.assertEqual(summary["query_count"], 5)
        self.assertEqual(summary["scored_query_count"], 4)
        self.assertEqual(summary["estimate_lt_1_count"], 3)
        self.assertEqual(summary["estimate_lt_0_1_count"], 2)
        self.assertEqual(summary["estimate_lt_0_01_count"], 2)
        self.assertEqual(summary["zero_estimate_count"], 1)
        self.assertTrue(math.isfinite(summary["raw_q_error"]["p99"]))
        self.assertEqual(summary["smoothed_q_error"]["max"], 10.0)
        # All four scored records have truth=10 (non-zero), so none are true-zero
        self.assertEqual(summary["true_zero_matching_count"], 0)
        self.assertIsNone(summary["smoothed_q_error_true_zero"]["p50"])
        # raw_q_error_true_positive should match raw_q_error since all truth > 0
        self.assertEqual(summary["raw_q_error_true_positive"]["max"], summary["raw_q_error"]["max"])

    def test_latency_and_throughput(self) -> None:
        records = [LatencyRecord("w", index, 0, 100.0) for index in range(10)]
        summary = summarize_latency(records)
        self.assertEqual(summary["mean_ms"], 100.0)
        self.assertAlmostEqual(summary["throughput_queries_per_second"], 10.0)

    def test_latency_profiles_keep_cpu_and_gpu_separate(self) -> None:
        records = (
            LatencyRecord("w", 0, 0, 10.0, device="cpu", device_name="CPU"),
            LatencyRecord("w", 0, 0, 2.0, device="cuda", device_name="Test GPU"),
        )
        profiles = _summarize_inference_profiles(records, configured_device="cuda")
        self.assertEqual(set(profiles), {"cpu", "cuda"})
        self.assertEqual(profiles["cpu"]["mean_ms"], 10.0)
        self.assertEqual(profiles["cuda"]["mean_ms"], 2.0)
        self.assertEqual(profiles["cuda"]["device_names"], ["Test GPU"])


if __name__ == "__main__":
    unittest.main()
