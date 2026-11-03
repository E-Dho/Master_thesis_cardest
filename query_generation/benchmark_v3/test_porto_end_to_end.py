"""Porto v3 end to end on synthetic Porto-shaped data.

The data reproduces what the Porto probes found: 15-second segments,
stationary segments, a few GPS jumps far outside the window domain, call
types A/B/C with origin_stand NULL for A and C (and for some B). Checks:

- the label SQL (in DuckDB) equals NumpyBackend for every category, with
  queries on the nullable origin_stand among them (NULL never matches);
- every category generates and the merge gates pass with the Porto config;
- the verifier, which fetches rows from SQL (NULL arrives as None), agrees on
  queries with origin_stand (pilot verification 22728877 crashed there);
- the GPS outliers are reported under window_domain_outside, not as domain
  violations, unless they lie beyond the reach of one-sided windows.
"""

import random
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from query_generation.benchmark_v3.backends import SqlBackend
from query_generation.benchmark_v3.build_pools import data_domains, domain_violations, window_domain_outside
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.exact import to_us
from query_generation.benchmark_v3.generate import generate_category
from query_generation.benchmark_v3.merge import merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.benchmark_v3.sampler import Sampler
from query_generation.benchmark_v3.synthetic import pools_from
from query_generation.benchmark_v3.verify import Fetcher, summarize, verify_record

from .test_duckdb_end_to_end import DuckSession, _spatial_available, duckdb

CONFIG = Path(__file__).with_name("porto_v3_config.json")
CENTER = (-40435.5, 165509.6)  # median endpoint of the Porto data
START = datetime(2013, 7, 1, 0, 0, 53)


def porto_rows(trips: int = 160, seed: int = 5):
    rng = random.Random(seed)
    trip_rows, segment_rows = [], []
    for trip_id in range(1, trips + 1):
        call_type = rng.choices("ABC", weights=(22, 48, 30))[0]
        stand = rng.randint(1, 63) if call_type == "B" and rng.random() > 0.05 else None
        t = START + timedelta(seconds=rng.uniform(0, 364 * 86400 - 7200))
        t = t.replace(microsecond=0)
        x, y = CENTER[0] + rng.gauss(0, 3000), CENTER[1] + rng.gauss(0, 3000)
        n = rng.randint(1, 40)
        start = t
        for idx in range(n):
            shape = rng.random()
            if shape < 0.05:
                nx, ny = x, y  # stationary
            elif shape < 0.07:
                nx, ny = x + rng.choice((-1, 1)) * 500_000.0, y + 300_000.0  # GPS jump
            else:
                nx, ny = x + rng.uniform(-200, 200), y + rng.uniform(-200, 200)
            end = t + timedelta(seconds=15)
            segment_rows.append({"trip_id": trip_id, "segment_idx": idx, "sx": x, "sy": y, "ex": nx, "ey": ny,
                                 "ts": t, "te": end})
            if shape < 0.05 or shape >= 0.07:
                x, y = nx, ny  # a jump does not move the vehicle
            t = end
        trip_rows.append({"trip_id": trip_id, "call_type": call_type, "origin_stand": stand,
                          "num_of_segments": n, "start": start, "end": t})
    return {"trips": trip_rows, "segments": segment_rows}


def porto_data(settings, rows):
    trips, segments = rows["trips"], rows["segments"]
    data = {
        "trips": {
            "trip_id": np.asarray([r["trip_id"] for r in trips], dtype=np.int64),
            "call_type": np.asarray([r["call_type"] for r in trips]),
            "origin_stand": np.asarray([np.nan if r["origin_stand"] is None else float(r["origin_stand"])
                                        for r in trips]),
            "num_of_segments": np.asarray([r["num_of_segments"] for r in trips], dtype=np.int64),
            "trip_time.start": np.asarray([to_us(r["start"]) for r in trips], dtype=np.int64),
            "trip_time.end": np.asarray([to_us(r["end"]) for r in trips], dtype=np.int64),
        },
        "segments": {
            "trip_id": np.asarray([r["trip_id"] for r in segments], dtype=np.int64),
            "segment_idx": np.asarray([r["segment_idx"] for r in segments], dtype=np.int64),
            **{key: np.asarray([r[key] for r in segments], dtype=np.float64) for key in ("sx", "sy", "ex", "ey")},
            "ts": np.asarray([to_us(r["ts"]) for r in segments], dtype=np.int64),
            "te": np.asarray([to_us(r["te"]) for r in segments], dtype=np.int64),
        },
    }
    temporal = settings.segments.temporal_attribute
    data["segments"][f"{temporal}.start"] = data["segments"]["ts"]
    data["segments"][f"{temporal}.end"] = data["segments"]["te"]
    return data


def load_porto_duckdb(rows):
    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("CREATE SCHEMA porto")
    con.execute("CREATE TABLE porto.trips (trip_id BIGINT, call_type VARCHAR, origin_stand INTEGER, "
                "start_time TIMESTAMP, end_time TIMESTAMP, num_of_segments INTEGER)")
    con.executemany("INSERT INTO porto.trips VALUES (?, ?, ?, ?, ?, ?)", [
        (r["trip_id"], r["call_type"], r["origin_stand"], r["start"], r["end"], r["num_of_segments"])
        for r in rows["trips"]])
    con.execute("CREATE TABLE porto.segments (trip_id BIGINT, segment_idx INTEGER, s_x DOUBLE, s_y DOUBLE, "
                "e_x DOUBLE, e_y DOUBLE, t_s TIMESTAMP, t_e TIMESTAMP)")
    con.executemany("INSERT INTO porto.segments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        (r["trip_id"], r["segment_idx"], r["sx"], r["sy"], r["ex"], r["ey"], r["ts"], r["te"])
        for r in rows["segments"]])
    con.execute("ALTER TABLE porto.segments ADD COLUMN segment_geom GEOMETRY")
    con.execute("UPDATE porto.segments SET segment_geom = ST_MakeLine(ST_Point(s_x, s_y), ST_Point(e_x, e_y))")
    return con


class _Porto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG, single_count=4, multi_count=6)
        cls.rows = porto_rows()
        cls.data = porto_data(cls.settings, cls.rows)
        cls.pools = pools_from(cls.settings, cls.rows)
        cls.numpy = NumpyBackend(cls.settings, cls.data)


class PortoGenerationTest(_Porto):
    def test_every_category_generates_and_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            for category in self.settings.categories():
                summary = generate_category(self.settings, self.pools, category, base_seed=11,
                                            count_fn=self.numpy.count, output_dir=Path(tmp),
                                            progress_interval=0, log=lambda message: None)
                self.assertTrue(summary["finished"], category.key)
            rows = merge(self.settings, Path(tmp))["rows"]
        self.assertEqual(len(rows), 8 * 4 + 8 * 6)
        tables = {tuple(row["tables"]) for row in rows}
        self.assertTrue(tables <= {("segments",), ("trips",), ("segments", "trips"), ("trips", "segments")})
        attributes = {p["attribute"] for row in rows for p in row["predicates"]}
        self.assertTrue({"call_type", "segment_geom", "segment_time"} <= attributes, attributes)


@unittest.skipUnless(_spatial_available(), "duckdb spatial extension unavailable")
class PortoSqlEqualsNumpyTest(_Porto):
    def test_label_sql_equals_numpy_including_nullable_origin_stand(self):
        sql = SqlBackend(DuckSession(load_porto_duckdb(self.rows)))
        wide = load_settings(CONFIG, single_count=30, multi_count=30)
        sampler = Sampler(wide, self.pools, base_seed=29)
        positive = stand_queries = 0
        for category in wide.categories():
            for index in range(30):
                record = sampler.candidate(category, index, 0)
                expected = self.numpy.count(record)
                self.assertEqual(sql.count(record), expected, record["slot_id"])
                positive += expected > 0
                stand_queries += any(p["attribute"] == "origin_stand" for p in record["predicates"])
        self.assertGreater(positive, 150)
        self.assertGreater(stand_queries, 10)  # NULL handling exercised



@unittest.skipUnless(_spatial_available(), "duckdb spatial extension unavailable")
class PortoVerifyAndDomainsTest(_Porto):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.session = DuckSession(load_porto_duckdb(cls.rows))

    def test_verify_with_nullable_origin_stand(self):
        wide = load_settings(CONFIG, single_count=12, multi_count=12)
        with tempfile.TemporaryDirectory() as tmp:
            for category in wide.categories():
                generate_category(wide, self.pools, category, base_seed=31, count_fn=self.numpy.count,
                                  output_dir=Path(tmp), progress_interval=0, log=lambda message: None)
            rows = merge(wide, Path(tmp))["rows"]
        fetcher = Fetcher(wide, self.session, max_rows=10**7)
        results = [verify_record(wide, fetcher, self.session, row) for row in rows]
        summary = summarize(results)
        for key in ("numpy_mismatches", "exists_mismatches", "postgis_mismatches"):
            self.assertEqual(summary[key], [], key)
        self.assertEqual(summary["st_intersects_disagreements"], 0)
        by_id = {result["query_id"]: result for result in results}
        stand = [row for row in rows if row["category"]["dimension"] != "standard"
                 and any(p["attribute"] == "origin_stand" for p in row["predicates"])]
        checked = [by_id[row["query_id"]] for row in stand if "numpy_count" in by_id[row["query_id"]]]
        self.assertGreater(len(checked), 5)  # rows with NULL origin_stand were fetched and compared
        self.assertTrue(all(result["numpy_ok"] for result in checked))

    def test_fetched_null_origin_stand_is_nan(self):
        fetcher = Fetcher(self.settings, self.session, max_rows=10**7)
        trips = fetcher._select("trips", [], "")
        stand = trips["origin_stand"]
        self.assertEqual(stand.dtype, np.float64)
        self.assertEqual(int(np.isnan(stand).sum()), sum(r["origin_stand"] is None for r in self.rows["trips"]))
        self.assertEqual(trips["call_type"].dtype.kind, "U")  # nominal, no NULLs: unchanged

    def test_outliers_are_window_domain_outside_not_violations(self):
        domains = data_domains(self.settings, self.session)
        self.assertEqual(domain_violations(self.settings, domains), [])
        outside = window_domain_outside(self.settings, domains)
        self.assertTrue(outside)  # the synthetic GPS jumps leave the window box
        self.assertEqual({entry["attribute"] for entry in outside}, {"segments.segment_geom"})
        for entry in outside:
            beyond = entry["data"] < entry["window"] if entry["field"].startswith("min") else entry["data"] > entry["window"]
            self.assertTrue(beyond, entry)
        margin = self.settings.unbounded_spatial_margin
        far = dict(domains["segments.segment_geom"], max_y=entry_bound(self.settings, "max_y") + margin + 1.0)
        violations = domain_violations(self.settings, dict(domains, **{"segments.segment_geom": far}))
        self.assertEqual(len(violations), 1, violations)
        self.assertIn("beyond the reach", violations[0])
        late = dict(domains, **{"trips.num_of_segments": {"min": 1, "max": 99999}})
        self.assertEqual(len(domain_violations(self.settings, late)), 1)  # ordinary attributes unchanged


def entry_bound(settings, field):
    return float(settings.attr("segments", "segment_geom")["domain"][field])


if __name__ == "__main__":
    unittest.main()
