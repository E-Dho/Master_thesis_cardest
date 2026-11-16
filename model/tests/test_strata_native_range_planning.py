"""Rare-support planning without native ranges (IMDB ablation imdb_ablation_no_native_ranges)."""

from __future__ import annotations

import unittest

import numpy as np

from model.src.data.strata import _numeric_root_candidates, predicate_probability_map


def _candidates(native_range_probability: float):
    values = tuple(float(v) for v in range(40))
    counts = np.array([1000.0] * 30 + [3.0] * 10)  # a rare upper tail
    probabilities = predicate_probability_map({
        "wildcard_probability": 0.2, "equality_probability": 0.2,
        "lower_bound_probability": 0.3 if native_range_probability == 0.0 else 0.2,
        "upper_bound_probability": 0.3 if native_range_probability == 0.0 else 0.2,
        "native_range_probability": native_range_probability,
    })
    return list(_numeric_root_candidates(
        0, "title:production_year", values, counts, float(counts.sum()),
        n_total=20000 * 2048, predicate_probabilities=probabilities, threshold=100.0,
        source="test", semantic_type="ordered",
    ))


class NativeRangePlanningTest(unittest.TestCase):
    def test_no_range_support_is_planned_without_native_ranges(self) -> None:
        strata = _candidates(0.0)
        self.assertTrue(strata)
        self.assertTrue(all(s.expected_range_support is None for s in strata))
        self.assertNotIn("native_range", {s.support_bottleneck for s in strata})

    def test_native_ranges_keep_their_planning(self) -> None:
        # the evaluated configuration (native_range_probability 0.2) is unchanged
        strata = _candidates(0.2)
        self.assertTrue(all(s.expected_range_support is not None for s in strata))
        self.assertIn("native_range", {s.support_bottleneck for s in strata})


if __name__ == "__main__":
    unittest.main()
