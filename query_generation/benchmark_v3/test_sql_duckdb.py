"""The rendered SQL computes the same answers as the numpy reference.

DuckDB stands in for PostgreSQL (same SQL subset: CASE, GREATEST/LEAST,
float8 arithmetic, timestamp literals, EXTRACT(EPOCH FROM interval)). Only the
PostGIS index prefilter ``segment_geom && envelope`` is removed, which the
exact test implies anyway. The test is skipped when duckdb is not installed.
"""

import random
import re
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from query_generation.benchmark_v3 import exact
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.query_model import predicate_sql, timestamp_text

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

CONFIG = Path(__file__).with_name("pol_v3_config.json")
PREFILTER = re.compile(r"s\.segment_geom && ST_MakeEnvelope\([^)]*\) AND ")


@unittest.skipIf(duckdb is None, "duckdb not installed")
class SqlMatchesNumpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG)
        rng = random.Random(3)
        base = datetime(2019, 9, 1)
        rows = []
        for i in range(4000):
            sx, sy = rng.uniform(740000, 744000), rng.uniform(3734000, 3738800)
            shape = rng.random()
            if shape < 0.1:
                ex, ey, dt = sx, sy, 0
            elif shape < 0.2:
                ex, ey, dt = sx, sy, rng.randint(1, 900_000_000)
            elif shape < 0.3:
                ex, ey, dt = sx + rng.uniform(-400, 400), sy, rng.randint(1, 900_000_000)
            else:
                ex, ey, dt = sx + rng.uniform(-400, 400), sy + rng.uniform(-400, 400), rng.randint(1, 900_000_000)
            ts = base + timedelta(microseconds=rng.randint(0, 3 * 86400 * 10**6))
            rows.append((i, sx, sy, ex, ey, ts, ts + timedelta(microseconds=dt)))
        cls.rows = rows
        cls.con = duckdb.connect()
        cls.con.execute("CREATE TABLE seg (id INTEGER, s_x DOUBLE, s_y DOUBLE, e_x DOUBLE, e_y DOUBLE, t_s TIMESTAMP, t_e TIMESTAMP)")
        cls.con.executemany("INSERT INTO seg VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        cols = list(zip(*rows))
        cls.arrays = {
            "sx": np.asarray(cols[1]), "sy": np.asarray(cols[2]),
            "ex": np.asarray(cols[3]), "ey": np.asarray(cols[4]),
            "ts": np.asarray([exact.to_us(v) for v in cols[5]], dtype=np.int64),
            "te": np.asarray([exact.to_us(v) for v in cols[6]], dtype=np.int64),
        }
        cls.rng = rng
        cls.base = base

    def _sql_ids(self, predicate):
        condition = PREFILTER.sub("", predicate_sql(self.settings, predicate))
        self.assertNotIn("ST_", condition)
        return {row[0] for row in self.con.execute(f"SELECT id FROM seg s WHERE {condition}").fetchall()}

    def _window(self):
        cx, cy = self.rng.uniform(740000, 744000), self.rng.uniform(3734000, 3738800)
        w, h = self.rng.expovariate(1 / 300), self.rng.expovariate(1 / 300)
        return {"min_x": cx - w, "min_y": cy - h, "max_x": cx + w, "max_y": cy + h}

    def _interval(self):
        center = self.base + timedelta(microseconds=self.rng.randint(0, 3 * 86400 * 10**6))
        width = timedelta(seconds=float(np.exp(self.rng.uniform(np.log(60), np.log(3 * 86400)))))
        lower, upper = center - width / 2, center + width / 2
        mode = self.rng.random()
        return (
            None if mode < 0.2 else timestamp_text(lower),
            None if 0.2 <= mode < 0.4 else timestamp_text(upper),
        )

    def test_spatial(self):
        a = self.arrays
        for _ in range(60):
            predicate = {"kind": "spatial", "table": "segments", "attribute": "segment_geom", **self._window()}
            window = (predicate["min_x"], predicate["min_y"], predicate["max_x"], predicate["max_y"])
            expected = set(np.flatnonzero(exact.spatial_mask(a["sx"], a["sy"], a["ex"], a["ey"], window)).tolist())
            self.assertEqual(self._sql_ids(predicate), expected)

    def test_spatio_temporal(self):
        a = self.arrays
        nonempty = 0
        for _ in range(120):
            lower, upper = self._interval()
            if lower is None and upper is None:
                continue
            predicate = {
                "kind": "spatio_temporal", "table": "segments", "attribute": "segment_geom",
                **self._window(), "lower": lower, "upper": upper,
            }
            window = (predicate["min_x"], predicate["min_y"], predicate["max_x"], predicate["max_y"])
            to = lambda v: None if v is None else exact.to_us(datetime.fromisoformat(v))
            mask = exact.spatio_temporal_mask(a["sx"], a["sy"], a["ex"], a["ey"], a["ts"], a["te"], window, to(lower), to(upper))
            expected = set(np.flatnonzero(mask).tolist())
            nonempty += bool(expected)
            self.assertEqual(self._sql_ids(predicate), expected)
        self.assertGreater(nonempty, 10, "the test should exercise matching windows")

    def test_temporal(self):
        a = self.arrays
        for _ in range(60):
            lower, upper = self._interval()
            if lower is None and upper is None:
                continue
            predicate = {"kind": "temporal", "table": "segments", "attribute": "segment_time", "lower": lower, "upper": upper}
            to = lambda v: None if v is None else exact.to_us(datetime.fromisoformat(v))
            expected = set(np.flatnonzero(exact.temporal_mask(a["ts"], a["te"], to(lower), to(upper))).tolist())
            self.assertEqual(self._sql_ids(predicate), expected)


if __name__ == "__main__":
    unittest.main()
