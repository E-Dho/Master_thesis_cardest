"""The production SQL paths, end to end, on DuckDB with its spatial extension.

The synthetic POL database is loaded into DuckDB as ``pol.agents``,
``pol.trips`` and ``pol.segments`` (with a real ``segment_geom``). Then

1. every label SQL of the sampler is executed and compared with the numpy
   backend,
2. a small benchmark is generated with the SQL backend and merged, and
3. the verification job (fetch queries, numpy recount, EXISTS reformulation,
   ST_Intersects comparison) runs on every merged query.

A thin session adapter translates the few PostgreSQL-specific spellings
(``ST_MakeEnvelope`` with an SRID, ``%s`` parameters, ``= ANY(%s)``, epoch
microseconds). The test is skipped if duckdb or its spatial extension is
unavailable.
"""

import re
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.backends import SqlBackend
from query_generation.benchmark_v3.build_pools import data_domains, diagnostics, domain_violations
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import generate_category
from query_generation.benchmark_v3.merge import merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.benchmark_v3.sampler import Sampler
from query_generation.benchmark_v3.verify import Fetcher, run_verification, stratified_sample, summarize, verify_record

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

CONFIG = Path(__file__).with_name("pol_v3_config.json")
ENVELOPE_SRID = re.compile(r"(ST_MakeEnvelope\([^,]+,[^,]+,[^,]+,[^,]+), \d+\)")
EPOCH_US = re.compile(r"\(EXTRACT\(EPOCH FROM ([a-z]\.\w+)\) \* 1000000\)::bigint")


class DuckSession:
    def __init__(self, con):
        self.con = con

    @staticmethod
    def translate(sql: str) -> str:
        sql = ENVELOPE_SRID.sub(r"\1)", sql)
        sql = EPOCH_US.sub(r"epoch_us(\1)", sql)
        sql = sql.replace("= ANY(%s)", "IN (SELECT UNNEST(?))").replace("%s", "?")
        sql = sql.replace("ST_MakePoint(", "ST_Point(")
        return sql

    def scalar(self, sql, params=None):
        row = self.con.execute(self.translate(sql), params or []).fetchone()
        return None if row is None else row[0]

    def rows(self, sql, params=None):
        return [tuple(r) for r in self.con.execute(self.translate(sql), params or []).fetchall()]


def load_duckdb(rows):
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("CREATE SCHEMA pol")
    con.execute('CREATE TABLE pol.agents (agent_id BIGINT, age DOUBLE, "educationLevel" VARCHAR, interest VARCHAR, joviality DOUBLE, family_size INTEGER)')
    con.executemany("INSERT INTO pol.agents VALUES (?, ?, ?, ?, ?, ?)", [
        (r["agent_id"], r["age"], r["educationLevel"], r["interest"], r["joviality"], r["family_size"]) for r in rows["agents"]
    ])
    con.execute("CREATE TABLE pol.trips (trip_id BIGINT, agent_id BIGINT, start_time TIMESTAMP, end_time TIMESTAMP, num_of_segments INTEGER)")
    con.executemany("INSERT INTO pol.trips VALUES (?, ?, ?, ?, ?)", [
        (r["trip_id"], r["agent_id"], r["start"], r["end"], r["num_of_segments"]) for r in rows["trips"]
    ])
    con.execute("CREATE TABLE pol.segments (trip_id BIGINT, segment_idx INTEGER, s_x DOUBLE, s_y DOUBLE, e_x DOUBLE, e_y DOUBLE, t_s TIMESTAMP, t_e TIMESTAMP)")
    con.executemany("INSERT INTO pol.segments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        (r["trip_id"], r["segment_idx"], r["sx"], r["sy"], r["ex"], r["ey"], r["ts"], r["te"]) for r in rows["segments"]
    ])
    con.execute("ALTER TABLE pol.segments ADD COLUMN segment_geom GEOMETRY")
    con.execute("UPDATE pol.segments SET segment_geom = ST_MakeLine(ST_Point(s_x, s_y), ST_Point(e_x, e_y))")
    return con


def _spatial_available():
    if duckdb is None:
        return False
    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
        return True
    except Exception:  # pragma: no cover - offline environments
        return False


@unittest.skipUnless(_spatial_available(), "duckdb spatial extension unavailable")
class DuckDbEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG, single_count=5, multi_count=6)
        cls.data, rows = synthetic.build(cls.settings, agents=30, trips_per_agent=5, seed=21)
        cls.pools = synthetic.pools_from(cls.settings, rows)
        cls.session = DuckSession(load_duckdb(rows))
        cls.sql = SqlBackend(cls.session)
        cls.numpy = NumpyBackend(cls.settings, cls.data)

    def test_label_sql_equals_numpy_for_every_category(self):
        wide = load_settings(CONFIG, single_count=25, multi_count=25)
        sampler = Sampler(wide, self.pools, base_seed=77)
        positive = 0
        for category in self.settings.categories():
            for index in range(25):
                record = sampler.candidate(category, index, 0)
                expected = self.numpy.count(record)
                self.assertEqual(self.sql.count(record), expected, record["slot_id"])
                positive += expected > 0
        self.assertGreater(positive, 300)

    def test_postgis_formulation_equals_numpy_for_spatial_queries(self):
        from query_generation.benchmark_v3.verify import postgis_sql

        wide = load_settings(CONFIG, single_count=40, multi_count=40)
        sampler = Sampler(wide, self.pools, base_seed=13)
        checked = positive = 0
        for category in wide.categories():
            if category.dimension not in ("spatial", "spatio_temporal"):
                continue
            for index in range(40):
                record = sampler.candidate(category, index, 0)
                expected = self.numpy.count(record)
                self.assertEqual(int(self.session.scalar(postgis_sql(wide, record))), expected, record["slot_id"])
                checked += 1
                positive += expected > 0
        self.assertEqual(checked, 8 * 40)
        self.assertGreater(positive, 200)

    def _merged_rows(self):
        if not hasattr(type(self), "_rows_cache"):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                for category in self.settings.categories():
                    generate_category(self.settings, self.pools, category, base_seed=5,
                                      count_fn=self.sql.count, output_dir=out,
                                      progress_interval=0, log=lambda m: None)
                type(self)._rows_cache = merge(self.settings, out)["rows"]
        return self._rows_cache

    def test_sample_is_interleaved_and_exact_per_category(self):
        rows = self._merged_rows()
        sample = stratified_sample(rows, per_category=3, seed=1)
        self.assertEqual(len(sample), 16 * 3)
        first_round = [row["category"]["key"] for row in sample[:16]]
        self.assertEqual(len(set(first_round)), 16, "every category appears once per round")
        self.assertEqual(sample, stratified_sample(list(reversed(rows)), per_category=3, seed=1))

    def test_resume_deadline_and_binding(self):
        rows = self._merged_rows()
        binding = {"benchmark_sha256": "x", "per_category": 2, "seed": 1}
        quiet = dict(log=lambda m: None, per_category=2, seed=1, max_rows=10**7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            results, finished = run_verification(self.settings, self.session, rows, results_path=path,
                                                 binding=binding, deadline=time.time() - 1, **quiet)
            self.assertEqual((results, finished), ([], False))
            full, finished = run_verification(self.settings, self.session, rows, results_path=path,
                                              binding=binding, **quiet)
            self.assertTrue(finished)
            self.assertEqual(len(full), 32)
            # drop the last durable result and resume: only that query is verified again
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            calls = []
            with mock.patch("query_generation.benchmark_v3.verify.verify_record",
                            side_effect=lambda *a: calls.append(a[3]["query_id"]) or verify_record(*a)):
                resumed, finished = run_verification(self.settings, self.session, rows, results_path=path,
                                                     binding=binding, **quiet)
            self.assertTrue(finished)
            self.assertEqual(calls, [full[-1]["query_id"]])
            self.assertEqual([r["query_id"] for r in resumed], [r["query_id"] for r in full])
            with self.assertRaises(SystemExit):
                run_verification(self.settings, self.session, rows, results_path=path,
                                 binding=dict(binding, seed=2), **quiet)

    def test_row_limit_skips_numpy_but_not_postgis(self):
        rows = [r for r in self._merged_rows() if r["category"]["dimension"] in ("spatial", "spatio_temporal")]
        fetcher = Fetcher(self.settings, self.session, max_rows=1)
        results = [verify_record(self.settings, fetcher, self.session, row) for row in rows]
        summary = summarize(results)
        self.assertEqual(summary["postgis_checked"], len(rows))
        self.assertEqual(summary["postgis_mismatches"], [])
        self.assertGreater(summary["numpy_skipped"], 0)
        self.assertTrue(all("postgis_seconds" in r for r in results))

    def test_data_domains_and_violations(self):
        domains = data_domains(self.settings, self.session)
        seg = self.data["segments"]
        geom = domains["segments.segment_geom"]
        self.assertEqual(geom["min_x"], min(seg["sx"].min(), seg["ex"].min()))
        self.assertEqual(geom["max_y"], max(seg["sy"].max(), seg["ey"].max()))
        self.assertEqual(domains["agents.age"]["max"], self.data["agents"]["age"].max())
        self.assertEqual(domains_violations := domain_violations(self.settings, domains), [])
        domains["agents.age"]["max"] = 999.0
        domains["trips.trip_time"]["min"] = "2019-01-01 00:00:00.000000"
        violations = domain_violations(self.settings, domains)
        self.assertEqual(len(violations), 2, violations)
        self.assertEqual(domains_violations, [])

    def test_diagnostics_never_raise(self):
        result = diagnostics(self.settings, self.session)
        self.assertEqual(set(result), {"pg_stats_columns_per_table", "segment_tgeom_start_offset_seconds"})

    def test_generate_merge_verify_on_sql(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for category in self.settings.categories():
                generate_category(self.settings, self.pools, category, base_seed=5,
                                  count_fn=self.sql.count, output_dir=out,
                                  progress_interval=0, log=lambda m: None)
            rows = merge(self.settings, out)["rows"]
        fetcher = Fetcher(self.settings, self.session, max_rows=10**7)
        results = [verify_record(self.settings, fetcher, self.session, row) for row in rows]
        summary = summarize(results)
        self.assertEqual(summary["numpy_mismatches"], [])
        self.assertEqual(summary["exists_mismatches"], [])
        self.assertEqual(summary["st_intersects_disagreements"], 0)
        self.assertGreaterEqual(summary["numpy_checked"], 12 * 5)
        self.assertEqual(summary["exists_checked"], 8 * 6)
        self.assertEqual(summary["postgis_mismatches"], [])
        self.assertEqual(summary["postgis_checked"], 2 * 5 + 2 * 6 + 2 * 5 + 2 * 6)


if __name__ == "__main__":
    unittest.main()
