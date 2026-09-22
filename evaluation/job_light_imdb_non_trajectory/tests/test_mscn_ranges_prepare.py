from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
    FilterPredicate,
    JoinPredicate,
    QueryRecord,
    TableRef,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mscn_ranges_prepare.py"
SPEC = importlib.util.spec_from_file_location("mscn_ranges_prepare", SCRIPT)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


class MscnRangesPrepareTests(unittest.TestCase):
    def setUp(self):
        self.templates = [
            QueryRecord(
                "ranges", 0,
                (TableRef("title", "title"), TableRef("movie_keyword", "movie_keyword")),
                (JoinPredicate("title.id", "=", "movie_keyword.movie_id"),),
                (
                    FilterPredicate("title.production_year", ">=", 2015.0),
                    FilterPredicate("movie_keyword.keyword_id", "=", 8200),
                ),
                3,
                "",
            ),
            QueryRecord(
                "ranges", 1,
                (TableRef("title", "title"), TableRef("movie_keyword", "movie_keyword")),
                (JoinPredicate("title.id", "=", "movie_keyword.movie_id"),),
                (
                    FilterPredicate("title.production_year", "<=", 2000.0),
                    FilterPredicate("movie_keyword.keyword_id", "=", 7),
                ),
                4,
                "",
            ),
        ]
        self.domains = {
            "title.production_year": tuple(float(year) for year in range(1990, 2021)),
            "movie_keyword.keyword_id": (7, 8200),
        }

    def test_generated_bounds_only_loosen_nonempty_templates(self):
        generated = PREP.generate_loosened_range_queries(
            self.templates, self.domains, count=80, seed=7
        )
        self.assertEqual(len(generated), 80)
        for query in generated:
            year = query.filters[0]
            if query.filters[1].value == 8200:
                self.assertIn(year.operator, {">", ">="})
                self.assertLessEqual(year.value, 2015.0)
                if year.operator == ">":
                    self.assertLess(year.value, 2015.0)
            else:
                self.assertIn(year.operator, {"<", "<="})
                self.assertGreaterEqual(year.value, 2000.0)
                if year.operator == "<":
                    self.assertGreater(year.value, 2000.0)

    def test_ranked_bitmap_roundtrip_shape(self):
        samples = {
            "title": [{"production_year": float(1990 + index % 31)} for index in range(1000)],
            "movie_keyword": [{"keyword_id": 7 if index % 2 else 8200} for index in range(1000)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary) / "workload"
            PREP.materialize_ranked_workload(self.templates, self.domains, samples, prefix)
            self.assertEqual(prefix.with_suffix(".bitmaps").stat().st_size, 2 * (4 + 2 * 125))
            lines = prefix.with_suffix(".csv").read_text().splitlines()
            self.assertIn("title.production_year,>=,25", lines[0])

    def test_imdb_backslash_escaped_quotes_do_not_shift_columns(self):
        payload = 'title,production_year\n"The \\"Quoted\\" Title",2015\n'
        row = next(PREP._dict_rows(io.StringIO(payload)))
        self.assertEqual(row["title"], 'The "Quoted" Title')
        self.assertEqual(row["production_year"], "2015")

    def test_varchar_literals_that_look_numeric_remain_strings(self):
        query = QueryRecord(
            "ranges", 2, (TableRef("title", "title"),), (),
            (FilterPredicate("title.imdb_index", ">=", 1),), 2, "",
        )
        converted = PREP.coerce_query_types([query])[0]
        self.assertEqual(converted.filters[0].value, "1")
        self.assertIn("'1'", PREP.query_to_sql(converted))

    def test_grouped_rows_produce_exact_range_cardinality(self):
        query = self.templates[0]
        rows = [(2014.0, 5), (2015.0, 7), (2016.0, 11), (None, 101)]
        self.assertEqual(
            PREP.grouped_cardinality(query, ("title.production_year",), rows),
            18,
        )

    def test_grouped_label_sql_keeps_only_equalities_in_where(self):
        sql, columns = PREP._grouped_label_sql(self.templates[0])
        self.assertEqual(columns, ("title.production_year",))
        self.assertIn("movie_keyword.keyword_id=8200", sql)
        self.assertNotIn("production_year>=", sql)
        self.assertIn("GROUP BY title.production_year", sql)
        self.assertIn("agg_movie_keyword AS", sql)
        self.assertIn("COUNT(*)::bigint AS __count", sql)
        self.assertIn("SUM(movie_keyword.__count::numeric)", sql)

    def test_grouped_label_sql_preaggregates_each_child(self):
        query = QueryRecord(
            "ranges", 3,
            (
                TableRef("cast_info", "ci"),
                TableRef("movie_keyword", "mk"),
                TableRef("title", "t"),
            ),
            (
                JoinPredicate("t.id", "=", "ci.movie_id"),
                JoinPredicate("t.id", "=", "mk.movie_id"),
            ),
            (
                FilterPredicate("ci.role_id", "=", 1),
                FilterPredicate("ci.nr_order", ">=", 4),
                FilterPredicate("t.production_year", "<", 2000),
            ),
            7, "",
        )
        sql, columns = PREP._grouped_label_sql(query)
        self.assertEqual(columns, ("ci.nr_order", "t.production_year"))
        self.assertIn("agg_ci AS", sql)
        self.assertIn("agg_mk AS", sql)
        self.assertIn("ci.role_id=1", sql)
        self.assertIn("SUM(ci.__count::numeric * mk.__count::numeric)", sql)
        self.assertNotIn("FROM cast_info, movie_keyword, title", sql)


if __name__ == "__main__":
    unittest.main()
