from __future__ import annotations

import unittest

from evaluation.job_light_imdb_non_trajectory.joblight_eval.mscn_ranges import (
    DomainRankEncoder,
    assert_test_disjoint,
    generate_range_training_queries,
    label_queries,
    regenerate_table_bitmaps,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import parse_csv_query


class MscnRangesTest(unittest.TestCase):
    def test_rank_encoding_preserves_lexical_order_and_equality(self) -> None:
        encoder = DomainRankEncoder.from_values(["z", "aa", "m", "aa"])
        self.assertLess(encoder.encode("aa"), encoder.encode("m"))
        self.assertLess(encoder.encode("m"), encoder.encode("z"))
        self.assertEqual(encoder.encode("aa"), encoder.encode("aa"))

    def test_disjoint_workloads(self) -> None:
        first = parse_csv_query("title t##t.id,=,1#1", "train", 0)
        second = parse_csv_query("title t##t.id,=,2#1", "test", 0)
        assert_test_disjoint([first], [second])
        with self.assertRaises(ValueError):
            assert_test_disjoint([first], [first])

    def test_deterministic_generation_labeling_and_bitmaps(self) -> None:
        template = parse_csv_query(
            "title t##t.production_year,>=,2015,t.title,=,Alpha#1", "test", 0
        )
        domains = {
            "t.production_year": [2010, 2015, 2020, 2024],
            "t.title": ["Alpha", "Beta", "Zulu"],
        }
        first = generate_range_training_queries([template], domains, count=8, seed=7)
        second = generate_range_training_queries([template], domains, count=8, seed=7)
        self.assertEqual(first, second)
        assert_test_disjoint(first, [template])
        labeled = label_queries(first[:1], lambda sql: 17)
        self.assertEqual(labeled[0].true_cardinality, 17)
        query = parse_csv_query("title t##t.production_year,>=,2015#1", "w", 0)
        bitmaps = regenerate_table_bitmaps(
            [query],
            {"title": [{"production_year": 2010}, {"production_year": 2020}]},
        )
        self.assertEqual(bitmaps[(0, "title")].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
