from __future__ import annotations

import unittest

from evaluation.job_light_imdb_non_trajectory.joblight_eval.own_model_selection import (
    CandidateRejected,
    require_complete_candidate,
)


class OwnModelSelectionTest(unittest.TestCase):
    def test_accepts_complete_canonical_workloads(self) -> None:
        require_complete_candidate(
            "job_light", {"query_count": 70, "scored_query_count": 70, "coverage_fraction": 1.0}
        )
        require_complete_candidate(
            "job_light_ranges",
            {"query_count": 1000, "scored_query_count": 1000, "coverage_fraction": 1.0},
        )

    def test_rejects_partial_coverage(self) -> None:
        with self.assertRaisesRegex(CandidateRejected, "coverage"):
            require_complete_candidate(
                "job_light_ranges",
                {"query_count": 1000, "scored_query_count": 1000, "coverage_fraction": 0.999},
            )

    def test_rejects_wrong_canonical_query_count(self) -> None:
        with self.assertRaisesRegex(CandidateRejected, "expected 70"):
            require_complete_candidate(
                "job_light", {"query_count": 69, "scored_query_count": 69, "coverage_fraction": 1.0}
            )


if __name__ == "__main__":
    unittest.main()
