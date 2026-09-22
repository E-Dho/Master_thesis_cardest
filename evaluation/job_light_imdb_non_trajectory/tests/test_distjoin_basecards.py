from __future__ import annotations

import unittest

from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import parse_csv_query
from evaluation.job_light_imdb_non_trajectory.scripts.distjoin_basecards import base_cardinality_sql


class DistJoinBaseCardinalityTest(unittest.TestCase):
    def test_removes_filters_but_preserves_join(self) -> None:
        query = parse_csv_query(
            "title t,movie_info mi#t.id=mi.movie_id#t.production_year,>,2000#7",
            "fixture", 0,
        )
        self.assertEqual(
            base_cardinality_sql(query),
            "SELECT COUNT(*) FROM title t, movie_info mi WHERE t.id=mi.movie_id",
        )


if __name__ == "__main__":
    unittest.main()
