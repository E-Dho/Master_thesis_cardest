from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
    parse_csv_query,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deepdb_bridge.py"
SPEC = importlib.util.spec_from_file_location("deepdb_bridge", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


class DeepDbBridgeTests(unittest.TestCase):
    def test_header_removal_preserves_escaped_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            target = Path(temporary) / "target.csv"
            source.write_bytes(b'id,text\n1,"a, b"\n2,"quoted \\"value\\""\n')
            BRIDGE.copy_without_header(source, target)
            self.assertEqual(target.read_bytes(), b'1,"a, b"\n2,"quoted \\"value\\""\n')

    def test_synthetic_fixture_is_headerless_and_has_expected_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            BRIDGE.write_fixture(root, rows=3)
            with (root / "title.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 12)
            self.assertEqual(rows[0][0], "1")

    def test_deepdb_sql_canonicalizes_child_join_tree(self):
        query = parse_csv_query(
            "movie_companies,movie_info_idx,title#"
            "title.id=movie_companies.movie_id,"
            "movie_companies.movie_id=movie_info_idx.movie_id#"
            "title.production_year,>=,2004#12",
            "job_light_ranges",
            0,
        )
        sql = BRIDGE.deepdb_compatible_query_to_sql(query)
        self.assertIn("movie_companies.movie_id=title.id", sql)
        self.assertIn("movie_info_idx.movie_id=title.id", sql)
        self.assertNotIn("movie_companies.movie_id=movie_info_idx.movie_id", sql)
        self.assertIn("title.production_year>=2004", sql)

    def test_deepdb_sql_preserves_aliases_and_filters(self):
        query = parse_csv_query(
            "movie_companies mc,title t,movie_keyword mk#"
            "t.id=mc.movie_id,t.id=mk.movie_id#mk.keyword_id,=,117#148552",
            "job_light",
            0,
        )
        sql = BRIDGE.deepdb_compatible_query_to_sql(query)
        self.assertIn("mc.movie_id=t.id", sql)
        self.assertIn("mk.movie_id=t.id", sql)
        self.assertIn("mk.keyword_id=117", sql)

    def test_deepdb_sql_rejects_disconnected_join_graph(self):
        query = parse_csv_query(
            "movie_companies,movie_info_idx,title#"
            "title.id=movie_companies.movie_id#title.kind_id,=,1#3",
            "invalid",
            0,
        )
        with self.assertRaisesRegex(ValueError, "connect every table"):
            BRIDGE.deepdb_compatible_query_to_sql(query)

    def test_native_schema_exclusions_are_explicitly_unsupported(self):
        query = parse_csv_query(
            "title t# #t.production_year,>=,2000,t.phonetic_code,=,S123#3",
            "ranges",
            0,
        )
        schema = SimpleNamespace(
            table_dictionary={
                "title": SimpleNamespace(
                    attributes=["production_year", "phonetic_code"],
                    irrelevant_attributes=["phonetic_code"],
                )
            }
        )
        self.assertEqual(
            BRIDGE.deepdb_unsupported_filter_columns(query, schema),
            ("t.phonetic_code",),
        )


if __name__ == "__main__":
    unittest.main()
