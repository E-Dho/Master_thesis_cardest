from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters.foj_sampling import (
    FojSamplingAdapter,
    _allowed_domain_ids,
    estimate_from_foj,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import WorkloadConfig


class FojSamplingTest(unittest.TestCase):
    def test_exact_synthetic_fanout_correction(self) -> None:
        predicate = np.asarray([1, 1, 0, 1], dtype=float)
        indicators = np.asarray([1, 1, 1, 0], dtype=float)
        inverse = np.asarray([0.5, 0.5, 1.0, 1.0], dtype=float)
        estimate = estimate_from_foj(
            8.0,
            predicate,
            included_indicator_product=indicators,
            inverse_weight_product=inverse,
        )
        self.assertEqual(estimate, 2.0)

    def test_sampling_converges_without_bias(self) -> None:
        population = np.asarray([0.0, 0.5, 1.0, 1.0])
        expected = 100.0 * population.mean()
        estimates = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            sample = rng.choice(population, size=10_000, replace=True)
            estimates.append(estimate_from_foj(100.0, sample))
        self.assertAlmostEqual(float(np.mean(estimates)), expected, delta=0.15)

    def test_domain_predicates(self) -> None:
        domain = (1, 2, 3, "__OUTER_MISSING__")
        self.assertEqual(_allowed_domain_ids(domain, ">=", 2), {1, 2})
        self.assertEqual(_allowed_domain_ids(domain, "=", 3), {2})

    def test_required_projection_covers_all_workload_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workload_path = Path(temporary) / "queries.csv"
            workload_path.write_text(
                "movie_companies mc,title t#mc.movie_id=t.id#"
                "mc.company_id,=,17,t.production_year,>=,2015#1\n",
                encoding="utf-8",
            )
            adapter = object.__new__(FojSamplingAdapter)
            adapter.config = SimpleNamespace(
                workloads=(WorkloadConfig("job_light", workload_path),)
            )
            self.assertEqual(
                adapter._required_predicate_columns(),
                {
                    "movie_companies": {"company_id"},
                    "title": {"production_year"},
                },
            )


if __name__ == "__main__":
    unittest.main()
