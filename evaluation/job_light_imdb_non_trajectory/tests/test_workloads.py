from __future__ import annotations

import os
import unittest
from pathlib import Path

from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
    FilterPredicate,
    JoinPredicate,
    QueryRecord,
    TableRef,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
    load_workload,
    parse_csv_query,
    parse_sql_query,
    query_to_sql,
    rewrite_count_sql_for_explain,
)


class WorkloadTest(unittest.TestCase):
    def test_csv_and_sql_roundtrip_with_ranges_and_quotes(self) -> None:
        query = QueryRecord(
            workload="job_light_ranges",
            query_id=7,
            tables=(TableRef("title", "t"), TableRef("movie_keyword", "mk")),
            joins=(JoinPredicate("t.id", "=", "mk.movie_id"),),
            filters=(
                FilterPredicate("t.production_year", ">=", 2015),
                FilterPredicate("t.production_year", "<=", 2020),
                FilterPredicate("t.title", "=", "Director's cut and more"),
            ),
            true_cardinality=12,
            source_line="",
        )
        parsed = parse_sql_query(query_to_sql(query), query.workload, 7, 12)
        self.assertEqual(parsed.tables, query.tables)
        self.assertEqual(parsed.joins, query.joins)
        self.assertEqual(parsed.filters, query.filters)

    def test_canonical_csv_parser(self) -> None:
        query = parse_csv_query(
            "title t,movie_info mi#t.id=mi.movie_id#t.production_year,>=,2015,t.kind_id,=,1#42",
            "job_light",
            0,
        )
        self.assertEqual(query.true_cardinality, 42)
        self.assertEqual(len(query.filters), 2)

    def test_count_rewrite(self) -> None:
        sql = "SELECT COUNT(*) FROM title t WHERE t.id > 10;"
        self.assertEqual(
            rewrite_count_sql_for_explain(sql),
            "SELECT 1 FROM title t WHERE t.id > 10;",
        )
        with self.assertRaises(ValueError):
            rewrite_count_sql_for_explain("SELECT 1 FROM title;")

    def test_full_upstream_workloads_when_available(self) -> None:
        paths = {
            "job_light": os.environ.get("JOBLIGHT_QUERIES_CSV"),
            "job_light_ranges": os.environ.get("JOBLIGHT_RANGES_QUERIES_CSV"),
        }
        if not all(paths.values()):
            self.skipTest("set JOBLIGHT_QUERIES_CSV and JOBLIGHT_RANGES_QUERIES_CSV")
        expected = {"job_light": 70, "job_light_ranges": 1000}
        for workload, raw_path in paths.items():
            queries = load_workload(Path(raw_path), workload)
            self.assertEqual(len(queries), expected[workload])
            for query in queries:
                parsed = parse_sql_query(
                    query_to_sql(query), workload, query.query_id, query.true_cardinality
                )
                self.assertEqual(parsed.tables, query.tables)
                self.assertEqual(parsed.joins, query.joins)
                self.assertEqual(parsed.filters, query.filters)


if __name__ == "__main__":
    unittest.main()
