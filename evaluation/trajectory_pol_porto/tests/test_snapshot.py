"""Snapshot export, loading, consistency checks and the label-reproduction gate.

The PostgreSQL test needs a running server; set TRAJ_EVAL_TEST_PG to libpq
keywords, e.g. ``host=/var/tmp/claude_pg/sock port=55499 user=claude``. It
creates and drops its own database.
"""

from __future__ import annotations

import json
import os
import random
import struct
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.numpy_backend import NumpyBackend

from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap
from evaluation.trajectory_pol_porto.traj_eval.snapshot_gate import merge, run_shard

from ._fixture import MULTI, SINGLE, V3_CONFIG, benchmark_rows, write_benchmark

SETTINGS = load_settings(V3_CONFIG, single_count=SINGLE, multi_count=MULTI)
PG = os.environ.get("TRAJ_EVAL_TEST_PG")


# --- binary COPY decoding without a server ----------------------------------

class _FakeCopy:
    def __init__(self, payload: bytes, rng: random.Random):
        self.payload, self.rng = payload, rng

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        position = 0
        while position < len(self.payload):
            size = self.rng.randint(1, 97)
            yield memoryview(self.payload[position: position + size])
            position += size


class _FakeConnection:
    def __init__(self, payload: bytes, seed: int = 0):
        self.payload, self.rng = payload, random.Random(seed)

    @contextmanager
    def cursor(self):
        connection = self

        class Cursor:
            def copy(self, sql):
                return _FakeCopy(connection.payload, connection.rng)

        yield Cursor()


def _binary_copy(columns, null_at=None) -> bytes:
    out = bytearray(snap.BINARY_SIGNATURE + struct.pack(">ii", 0, 0))
    for row in range(len(columns[0][1])):
        out += struct.pack(">h", len(columns))
        for index, (kind, values) in enumerate(columns):
            if null_at == (row, index):
                out += struct.pack(">i", -1)
                continue
            out += struct.pack(">i", 8) + struct.pack(">q" if kind == "int64" else ">d", values[row])
    return bytes(out + struct.pack(">h", -1))


class BinaryCopyTest(unittest.TestCase):
    def test_decodes_rows_split_across_arbitrary_chunks(self) -> None:
        rng = np.random.default_rng(3)
        ints, floats = rng.integers(-2**62, 2**62, 1000), rng.normal(size=1000)
        specs = [snap.ColumnSpec("a", "", "int64"), snap.ColumnSpec("b", "", "float64")]
        for seed in range(5):
            with tempfile.TemporaryDirectory() as tmp:
                payload = _binary_copy([("int64", ints.tolist()), ("float64", floats.tolist())])
                snap._copy_binary(_FakeConnection(payload, seed), "x", specs, 1000, Path(tmp), "t")
                self.assertTrue(np.array_equal(np.load(Path(tmp) / "tables/t/a.npy"), ints))
                self.assertTrue(np.array_equal(np.load(Path(tmp) / "tables/t/b.npy"), floats))

    def test_null_switches_to_the_cursor_path_and_row_count_is_checked(self) -> None:
        specs = [snap.ColumnSpec("a", "", "int64")]
        with tempfile.TemporaryDirectory() as tmp:
            payload = _binary_copy([("int64", [1, 2, 3])], null_at=(1, 0))
            with self.assertRaises(snap._NeedsCursor):
                snap._copy_binary(_FakeConnection(payload), "x", specs, 3, Path(tmp), "t")
            payload = _binary_copy([("int64", [1, 2, 3])])
            with self.assertRaisesRegex(ValueError, "more rows than counted"):
                snap._copy_binary(_FakeConnection(payload), "x", specs, 2, Path(tmp), "t")
            with self.assertRaisesRegex(ValueError, "exported 3 rows, counted 4"):
                snap._copy_binary(_FakeConnection(payload), "x", specs, 4, Path(tmp), "t")


# --- specs ------------------------------------------------------------------

class SpecTest(unittest.TestCase):
    def test_specs_cover_what_the_numpy_backend_reads(self) -> None:
        specs = {table: {spec.name: spec for spec in columns} for table, columns in snap.table_specs(SETTINGS).items()}
        data, _ = synthetic.build(SETTINGS)
        for table, arrays in data.items():
            stored = set(specs[table]) | ({"segment_time.start", "segment_time.end"} if table == "segments" else set())
            self.assertEqual(set(arrays), stored, table)
        self.assertEqual(specs["segments"]["ts"].sql, "(EXTRACT(EPOCH FROM s.t_s) * 1000000)::bigint")
        self.assertIn("ORDER BY s.trip_id, s.segment_idx", snap.export_sql(SETTINGS, "segments", list(specs["segments"].values())))


# --- snapshot from synthetic arrays (no server) -------------------------------

def _write_snapshot_from_arrays(root: Path, data) -> Path:
    tables = {}
    for table, arrays in data.items():
        columns = {}
        for name, array in arrays.items():
            if table == "segments" and name.startswith("segment_time."):
                continue
            path = root / "tables" / table / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(array))
            columns[name] = {}
        tables[table] = {"columns": columns}
    snap.derive_trips(SETTINGS, root)
    manifest = {"snapshot_version": snap.SNAPSHOT_VERSION, "config_sha256": load_settings(V3_CONFIG).sha256,
                "tables": tables, "aliases": {"segments": {"segment_time.start": "ts", "segment_time.end": "te"}},
                "files": {}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class GateTest(unittest.TestCase):
    def test_sharded_gate_passes_resumes_and_catches_a_wrong_label(self) -> None:
        data, _ = synthetic.build(SETTINGS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _write_snapshot_from_arrays(root / "snap", data)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            for shard in range(3):
                self.assertEqual(run_shard(snapshot, V3_CONFIG, benchmark, root / "gate", shard, 3, log=lambda m: None), 0)
            # running a shard again recounts nothing and adds no duplicates
            run_shard(snapshot, V3_CONFIG, benchmark, root / "gate", 1, 3, log=lambda m: None)
            report = merge(benchmark, root / "gate", root / "report.json")
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["recounted"], 8 * SINGLE + 8 * MULTI)
            rows = [json.loads(line) for line in benchmark.read_text().splitlines()]
            rows[5]["cardinality"] += 1
            for key in ("join_cardinality", "entity_cardinality"):
                if rows[5][key] is not None:
                    rows[5][key] += 1
            tampered = write_benchmark(root / "tampered.jsonl", rows)
            for shard in range(3):
                run_shard(snapshot, V3_CONFIG, tampered, root / "gate2", shard, 3, log=lambda m: None)
            report = merge(tampered, root / "gate2", root / "report2.json")
            self.assertFalse(report["passed"])
            self.assertEqual([m["query_id"] for m in report["mismatches"]], [rows[5]["query_id"]])

    def test_missing_shard_fails_the_merge(self) -> None:
        data, _ = synthetic.build(SETTINGS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _write_snapshot_from_arrays(root / "snap", data)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            run_shard(snapshot, V3_CONFIG, benchmark, root / "gate", 0, 2, log=lambda m: None)
            report = merge(benchmark, root / "gate", root / "report.json")
            self.assertFalse(report["passed"])
            self.assertGreater(report["missing_count"], 0)


# --- the real export against PostgreSQL ---------------------------------------

SCHEMA = """
CREATE SCHEMA pol;
CREATE TABLE pol.agents (agent_id bigint PRIMARY KEY, age double precision NOT NULL,
  "educationLevel" text NOT NULL, interest text NOT NULL, joviality double precision NOT NULL,
  family_size integer NOT NULL);
CREATE TABLE pol.trips (trip_id bigint PRIMARY KEY, agent_id bigint NOT NULL REFERENCES pol.agents,
  start_time timestamp NOT NULL, end_time timestamp NOT NULL, num_of_segments integer NOT NULL);
CREATE TABLE pol.segments (trip_id bigint NOT NULL REFERENCES pol.trips, segment_idx integer NOT NULL,
  s_x double precision NOT NULL, s_y double precision NOT NULL, e_x double precision NOT NULL,
  e_y double precision NOT NULL, t_s timestamp NOT NULL, t_e timestamp NOT NULL,
  PRIMARY KEY (trip_id, segment_idx));
"""


@unittest.skipUnless(PG, "set TRAJ_EVAL_TEST_PG to run the PostgreSQL export test")
class PostgresExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import psycopg

        cls.psycopg = psycopg
        cls.dbname = f"traj_snapshot_test_{uuid.uuid4().hex[:8]}"
        with psycopg.connect(f"{PG} dbname=postgres", autocommit=True) as admin:
            admin.execute(f"CREATE DATABASE {cls.dbname}")
        cls.data, rows = synthetic.build(SETTINGS)
        with psycopg.connect(f"{PG} dbname={cls.dbname}") as connection:
            connection.execute(SCHEMA)
            with connection.cursor() as cursor:
                # insert in shuffled order: the export must sort by primary key
                for table, columns, items in (
                    ("pol.agents", ["agent_id", "age", "educationLevel", "interest", "joviality", "family_size"], rows["agents"]),
                    ("pol.trips", ["trip_id", "agent_id", "start_time", "end_time", "num_of_segments"],
                     [{**r, "start_time": r["start"], "end_time": r["end"]} for r in rows["trips"]]),
                    ("pol.segments", ["trip_id", "segment_idx", "s_x", "s_y", "e_x", "e_y", "t_s", "t_e"],
                     [{**r, "s_x": r["sx"], "s_y": r["sy"], "e_x": r["ex"], "e_y": r["ey"], "t_s": r["ts"], "t_e": r["te"]}
                      for r in rows["segments"]]),
                ):
                    shuffled = list(items)
                    random.Random(1).shuffle(shuffled)
                    quoted = ", ".join(f'"{c}"' for c in columns)
                    with cursor.copy(f"COPY {table} ({quoted}) FROM STDIN") as copy:
                        for item in shuffled:
                            copy.write_row([item[c] for c in columns])
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "snapshot"
        connect = lambda: psycopg.connect(f"{PG} dbname={cls.dbname} options='-c TimeZone=Europe/Berlin'")
        cls.manifest = snap.export_snapshot(SETTINGS, connect, cls.root, config_path=V3_CONFIG, log=lambda m: None)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        with cls.psycopg.connect(f"{PG} dbname=postgres", autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {cls.dbname}")

    def test_arrays_equal_the_source_data_exactly(self) -> None:
        loaded = snap.load_snapshot(self.root)
        for table, arrays in self.data.items():
            order = np.lexsort((arrays["segment_idx"], arrays["trip_id"])) if table == "segments" else \
                np.argsort(arrays["trip_id" if table == "trips" else "agent_id"])
            for name, values in arrays.items():
                expected = np.asarray(values)[order]
                self.assertTrue(np.array_equal(np.asarray(loaded[table][name]), expected), f"{table}.{name}")
        self.assertEqual(self.manifest["tables"]["segments"]["method"], "binary_copy")
        self.assertEqual(self.manifest["tables"]["agents"]["method"], "cursor")
        self.assertEqual(self.manifest["database"]["timezone"], "Europe/Berlin")
        self.assertTrue(self.manifest["checks"]["passed"])

    def test_every_label_is_reproduced_on_the_snapshot(self) -> None:
        backend = NumpyBackend(SETTINGS, snap.load_snapshot(self.root))
        for row in benchmark_rows():
            self.assertEqual(backend.count(row), row["cardinality"], row["query_id"])

    def test_trip_index_matches_brute_force(self) -> None:
        index = snap.load_trip_index(self.root)
        seg = snap.load_snapshot(self.root)["segments"]
        for i in range(0, index["trip_id"].size, 7):
            rows = slice(index["offset"][i], index["offset"][i + 1])
            self.assertTrue(np.all(seg["trip_id"][rows] == index["trip_id"][i]))
            self.assertEqual(index["xmin"][i], min(seg["sx"][rows].min(), seg["ex"][rows].min()))
            self.assertEqual(index["ymax"][i], max(seg["sy"][rows].max(), seg["ey"][rows].max()))
            self.assertEqual(index["tmax_us"][i], seg["te"][rows].max())

    def test_checksums_detect_changes(self) -> None:
        self.assertEqual(snap.verify_snapshot(self.root), [])
        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            copy = Path(tmp) / "copy"
            shutil.copytree(self.root, copy)
            array = np.load(copy / "tables/segments/sx.npy")
            array[0] += 1.0
            np.save(copy / "tables/segments/sx.npy", array)
            self.assertEqual(snap.verify_snapshot(copy), ["checksum mismatch tables/segments/sx.npy"])

    def test_existing_snapshot_is_not_overwritten_without_force(self) -> None:
        with self.assertRaises(FileExistsError):
            snap.export_snapshot(SETTINGS, lambda: None, self.root, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
