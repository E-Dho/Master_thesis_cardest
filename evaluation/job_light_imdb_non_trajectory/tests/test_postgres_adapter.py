from __future__ import annotations

import unittest

from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters.postgres import (
    extract_plan_rows,
)


class PostgresAdapterTest(unittest.TestCase):
    def test_extracts_top_level_join_rows(self) -> None:
        payload = [{"Plan": {"Node Type": "Hash Join", "Plan Rows": 321, "Plans": []}}]
        self.assertEqual(extract_plan_rows(payload), 321.0)

    def test_extracts_top_level_parallel_rows(self) -> None:
        payload = [{"Plan": {"Node Type": "Gather", "Plan Rows": 456, "Plans": []}}]
        self.assertEqual(extract_plan_rows(payload), 456.0)

    def test_aggregate_fixture_demonstrates_why_count_is_rewritten(self) -> None:
        payload = [
            {
                "Plan": {
                    "Node Type": "Aggregate",
                    "Plan Rows": 1,
                    "Plans": [{"Node Type": "Seq Scan", "Plan Rows": 999}],
                }
            }
        ]
        self.assertEqual(extract_plan_rows(payload), 1.0)


if __name__ == "__main__":
    unittest.main()
