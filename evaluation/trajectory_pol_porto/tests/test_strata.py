from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from evaluation.eval_core.records import PredictionRecord
from evaluation.trajectory_pol_porto.traj_eval.strata import (
    TRAJECTORY_SUMMARY_EXTENSION,
    _bootstrap,
    fraction_bucket,
    trajectory_summary,
    truth_decade,
)


@dataclass(frozen=True)
class _Record:
    query_id: int
    true_cardinality: int
    category: str = "spatial.range"
    addon_count: int = 0
    features: dict = None


def _prediction(query_id, truth, estimate, status="ok", diagnostic=""):
    return PredictionRecord("w", query_id, status, truth, estimate, diagnostic=diagnostic)


class BucketTest(unittest.TestCase):
    def test_buckets(self) -> None:
        self.assertEqual(truth_decade(1), "[1e0,1e1)")
        self.assertEqual(truth_decade(999), "[1e2,1e3)")
        self.assertEqual(truth_decade(1000), "[1e3,1e4)")
        self.assertEqual(fraction_bucket(None), None)
        self.assertEqual(fraction_bucket(1e-9), "<1e-6")
        self.assertEqual(fraction_bucket(3e-4), "[1e-4,1e-3)")
        self.assertEqual(fraction_bucket(0.5), "[1e-1,1]")
        self.assertEqual(fraction_bucket(1.0), "[1e-1,1]")


class SummaryTest(unittest.TestCase):
    def _summary(self):
        records = [
            _Record(i, 10 * (i + 1), "spatial.range" if i < 30 else "temporal.range", i % 3,
                    {"window_area_fraction": 1e-3 if i < 30 else None, "interval_fraction": None})
            for i in range(40)
        ]
        predictions = []
        for record in records:
            if record.query_id == 0:
                predictions.append(_prediction(0, record.true_cardinality, None, "unsupported"))
            elif record.query_id == 1:
                predictions.append(_prediction(1, record.true_cardinality, None, "failed"))
            elif record.query_id in (2, 3):
                predictions.append(_prediction(record.query_id, record.true_cardinality,
                                               record.true_cardinality * 4.0, diagnostic="approx:x"))
            else:
                predictions.append(_prediction(record.query_id, record.true_cardinality,
                                               record.true_cardinality * 2.0))
        return records, trajectory_summary("w", records, predictions)

    def test_coverage_counts_exact_approximate_unsupported_and_failed(self) -> None:
        _, summary = self._summary()
        self.assertEqual(summary["coverage"], {
            "query_count": 40, "scored_count": 38, "ok_exact_count": 36, "ok_approx_count": 2,
            "unsupported_count": 1, "failed_count": 1, "coverage_fraction": 38 / 40,
        })
        exact = summary["strata"]["all:exact_only"]
        self.assertEqual(exact["raw_q_error"]["max"], 2.0)
        self.assertEqual(summary["strata"]["all"]["raw_q_error"]["max"], 4.0)
        self.assertAlmostEqual(summary["strata"]["all"]["log10_ratio"]["p50"], np.log10(2.0))
        self.assertEqual(summary["strata"]["all"]["overestimate_fraction"], 1.0)

    def test_strata_partition_by_category_addons_and_area(self) -> None:
        _, summary = self._summary()
        strata = summary["strata"]
        self.assertEqual(strata["category:spatial.range"]["query_count"], 30)
        self.assertEqual(strata["category:temporal.range"]["query_count"], 10)
        self.assertEqual(sum(strata[f"addons:{n}"]["query_count"] for n in range(3)), 40)
        self.assertEqual(strata["window_area_fraction:[1e-3,1e-2)"]["query_count"], 30)
        self.assertNotIn("interval_fraction:None", strata)

    def test_bootstrap_is_deterministic_and_brackets_the_point_estimate(self) -> None:
        values = np.random.default_rng(1).lognormal(size=500)
        a, b = _bootstrap(values, "seed"), _bootstrap(values, "seed")
        self.assertEqual(a, b)
        for q in (50, 90, 95, 99):
            low, high = a[f"p{q}"]
            self.assertLessEqual(low, np.percentile(values, q))
            self.assertGreaterEqual(high, np.percentile(values, q))
        self.assertEqual(_bootstrap(values[:5], "seed")["p50"], None)

    def test_aggregation_across_seeds(self) -> None:
        _, one = self._summary()
        _, two = self._summary()
        aggregated = TRAJECTORY_SUMMARY_EXTENSION.aggregate([one, two])
        metric = aggregated["strata"]["all"]["raw_q_error.p50"]
        self.assertEqual(metric, {"mean": 2.0, "std": 0.0})
        self.assertIn("raw_q_error_ci.p50.low", aggregated["strata"]["all"])
        rows = TRAJECTORY_SUMMARY_EXTENSION.rows({"method_id": "m"}, "w", aggregated)
        self.assertEqual(rows[0]["stratum"], "all")
        self.assertIn("| m |", TRAJECTORY_SUMMARY_EXTENSION.markdown("t", [({"method_id": "m"}, "w", aggregated)]))


if __name__ == "__main__":
    unittest.main()
