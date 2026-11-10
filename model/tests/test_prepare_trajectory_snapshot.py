"""Snapshot-based preparation (model/scripts/prepare_trajectory_snapshot.py).

POL: the snapshot converter must reproduce the staging-TSV preparation that
trained the POL model (prepare_pol_staging_data.py) exactly: columns, domains,
encoded rows, provenance and the compact trajectory index. Porto: NULL
origin_stand is the model's __SQL_NULL__ value, domains follow
complete_ordinary_domain, the sample source validates and a few training
steps run with the Porto config.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from model.scripts import prepare_trajectory_snapshot as prep
from model.src.config import load_simple_yaml
from model.src.data.complete_domain_preparation import SQL_NULL, complete_ordinary_domain

REPO = Path(__file__).resolve().parents[2]
POL_CONFIG = REPO / "model/configs/pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"
PORTO_CONFIG = REPO / "model/configs/porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"
EPOCH = datetime(2023, 3, 1, tzinfo=timezone.utc)


def _segments(rng, trip_ids):
    rows = {name: [] for name in ("trip_id", "segment_idx", "sx", "sy", "ex", "ey", "ts", "te")}
    starts, ends, counts = [], [], []
    for trip in trip_ids:
        k = int(rng.integers(1, 6))
        t = int(rng.integers(0, 10 ** 6)) * 1_000_000 + int(rng.integers(0, 1_000_000))
        x, y = float(np.round(rng.uniform(-500, 500), 3)), float(np.round(rng.uniform(0, 900), 3))
        starts.append(t)
        for index in range(k):
            dt = int(rng.integers(1, 120)) * 1_000_000 + int(rng.integers(0, 1_000_000))
            nx, ny = float(np.round(x + rng.uniform(-50, 50), 3)), float(np.round(y + rng.uniform(-50, 50), 3))
            if index == 1:  # a stationary segment (Porto has 1.4 %)
                nx, ny = x, y
            for name, value in (("trip_id", trip), ("segment_idx", index), ("sx", x), ("sy", y), ("ex", nx),
                                ("ey", ny), ("ts", t), ("te", t + dt)):
                rows[name].append(value)
            x, y, t = nx, ny, t + dt
        ends.append(t)
        counts.append(k)
    arrays = {name: np.asarray(values, dtype=np.int64 if name in ("trip_id", "segment_idx", "ts", "te") else np.float64)
              for name, values in rows.items()}
    return arrays, np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64), np.asarray(counts)


def _write_snapshot(root: Path, tables) -> Path:
    manifest = {"snapshot_version": "traj_snapshot_v1", "config_sha256": "test", "tables": {}}
    for table, columns in tables.items():
        manifest["tables"][table] = {"columns": {name: {} for name in columns}}
        (root / "tables" / table).mkdir(parents=True, exist_ok=True)
        for name, values in columns.items():
            np.save(root / "tables" / table / f"{name}.npy", values)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _text_time(microseconds: int) -> str:
    return datetime.fromtimestamp(microseconds / 1e6, tz=timezone.utc).replace(tzinfo=None).strftime(
        "%Y-%m-%d %H:%M:%S.%f") if microseconds % 1_000_000 else datetime.fromtimestamp(
        microseconds // 1_000_000, tz=timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def _config(source: Path, target: Path, prepared: Path, *, steps: bool = False) -> Path:
    text = source.read_text(encoding="utf-8")
    for old in ("/work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest/data/pol_prepared/pol_50m_mbr",
                "/work_beegfs/sunip956/master_thesis_trajectories/model_data/porto_v3_mbr"):
        text = text.replace(old, str(prepared))
    if steps:  # a few tiny training steps
        for old, new in (("hidden_sizes: [512, 512, 512, 512]", "hidden_sizes: [16]"),
                         ("batch_size: 2048", "batch_size: 16"), ("sample_batch_size: 2048", "sample_batch_size: 16"),
                         ("steps_per_epoch: 20000", "steps_per_epoch: 3"),
                         ("support_planning_steps: 20000", "support_planning_steps: 3"),
                         ("checkpoint_interval_steps: 500", "checkpoint_interval_steps: 3"),
                         ("validation_interval_steps: 250", "validation_interval_steps: 3"),
                         ("interval_steps: 250", "interval_steps: 3"), ("batch_size: 512", "batch_size: 16"),
                         ("progress_interval_steps: 25", "progress_interval_steps: 1"),
                         ("minimum_expected_context_support: 100", "minimum_expected_context_support: 1"),
                         ("device: auto", "device: cpu")):
            text = text.replace(old, new)
        text = text.replace("output_directory: model/runs/", f"output_directory: {target.parent}/runs/")
    target.write_text(text, encoding="utf-8")
    return target


class PolEquivalenceTest(unittest.TestCase):
    def test_snapshot_preparation_equals_the_staging_preparation(self) -> None:
        rng = np.random.default_rng(3)
        agent_ids = np.arange(1, 7)
        agents = {"agent_id": agent_ids, "age": np.round(rng.uniform(18, 80, 6), 1),
                  "educationLevel": np.array(["Bachelors", "HighSchool", "Graduate", "Bachelors", "Low", "Graduate"]),
                  "interest": np.array(["a", "b", "a", "c", "b", "a"]), "joviality": np.round(rng.uniform(0, 1, 6), 4),
                  "family_size": rng.integers(1, 5, 6)}
        trip_ids = np.arange(10, 25)
        segments, starts, ends, counts = _segments(rng, trip_ids)
        trips = {"trip_id": trip_ids, "agent_id": rng.choice(agent_ids[:-1], trip_ids.size),  # agent 6: no trips
                 "trip_time.start": starts, "trip_time.end": ends, "num_of_segments": counts}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _write_snapshot(root / "snap", {"agents": agents, "trips": trips, "segments": segments})
            staging = root / "staging"
            staging.mkdir()
            (staging / "agents.tsv").write_text("".join(
                f"{a}\t{age!r}\t{edu}\t{interest}\t{jov!r}\t{fam}\n" for a, age, edu, interest, jov, fam in zip(
                    agent_ids, agents["age"].tolist(), agents["educationLevel"], agents["interest"],
                    agents["joviality"].tolist(), agents["family_size"])))
            (staging / "trips.tsv").write_text("".join(
                f"{t}\t{a}\t{_text_time(int(s))}\t{_text_time(int(e))}\t{n}\n"
                for t, a, s, e, n in zip(trip_ids, trips["agent_id"], starts, ends, counts)))
            (staging / "segments.tsv").write_text("".join(
                f"{t}\t{i}\t{sx!r}\t{sy!r}\t{ex!r}\t{ey!r}\t{_text_time(int(ts))}\t{_text_time(int(te))}\n"
                for t, i, sx, sy, ex, ey, ts, te in zip(*(segments[k].tolist() for k in
                                                         ("trip_id", "segment_idx", "sx", "sy", "ex", "ey", "ts", "te")))))
            reference = root / "reference"
            config = _config(POL_CONFIG, root / "pol.yaml", reference)
            subprocess.run([sys.executable, "-m", "model.scripts.prepare_pol_staging_data", "--config", str(config),
                            "--staging-dir", str(staging)], cwd=REPO, check=True, capture_output=True)
            mine = root / "from_snapshot"
            prep.prepare("pol", config, snapshot, mine, log=lambda message: None)
            report = prep.compare(mine, reference, index=mine / "trajectory_segment_index",
                                  reference_index=reference / "trajectory_segment_index")
        self.assertTrue(report["passed"], json.dumps(report, indent=1)[:3000])
        self.assertEqual(report["differing_sample_rows"], 0)
        self.assertTrue(all(report["index_arrays_equal"].values()))

    def test_compare_finds_a_changed_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(4)
            segments, starts, ends, counts = _segments(rng, np.arange(1, 6))
            snapshot = _write_snapshot(root / "snap", {
                "trips": {"trip_id": np.arange(1, 6), "call_type": np.array(list("ABCAB")),
                          "origin_stand": np.array([np.nan, 3.0, np.nan, 7.0, 3.0]), "num_of_segments": counts,
                          "trip_time.start": starts, "trip_time.end": ends},
                "segments": segments})
            config = _config(PORTO_CONFIG, root / "porto.yaml", root / "a")
            prep.prepare("porto", config, snapshot, root / "a", log=lambda message: None)
            prep.prepare("porto", config, snapshot, root / "b", log=lambda message: None)
            self.assertTrue(prep.compare(root / "a", root / "b")["passed"])
            rows = np.load(root / "b" / "sample_rows.npy", mmap_mode="r+")
            rows[2, 0] = (rows[2, 0] + 1) % 3
            rows.flush()
            self.assertEqual(prep.compare(root / "a", root / "b")["differing_sample_rows"], 1)


class PortoPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(5)
        trip_ids = np.arange(100, 140)
        segments, starts, ends, counts = _segments(rng, trip_ids)
        call_type = rng.choice(np.array(["A", "B", "C"]), trip_ids.size)
        stand = np.where(call_type == "B", rng.integers(1, 64, trip_ids.size).astype(float), np.nan)
        stand[np.flatnonzero(call_type == "B")[:2]] = np.nan  # B trips with a missing stand
        cls.trips = {"trip_id": trip_ids, "call_type": call_type, "origin_stand": stand, "num_of_segments": counts,
                     "trip_time.start": starts, "trip_time.end": ends}
        cls.segments = segments
        snapshot = _write_snapshot(root / "snap", {"trips": cls.trips, "segments": segments})
        cls.prepared = root / "porto_prepared"
        cls.config = _config(PORTO_CONFIG, root / "porto.yaml", cls.prepared, steps=True)
        prep.prepare("porto", cls.config, snapshot, log=lambda message: None)
        cls.manifest = json.loads((cls.prepared / "manifest.json").read_text())
        from model.src.data.schema import ModelMetadata

        cls.metadata = ModelMetadata.from_json_dict(cls.manifest["metadata"])
        cls.root = root

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_columns_domains_and_null(self) -> None:
        names = [column.name for column in self.metadata.columns]
        self.assertEqual(names[:6], ["trips:call_type", "trips:origin_stand", "trips:start_time", "trips:end_time",
                                     "trips:num_of_segments", "trips:trip_geom"])
        self.assertEqual(names[-3:], ["I_trips", "I_segments", "F_trips_to_segments"])
        self.assertEqual((self.metadata.join_root, self.metadata.join_tables), ("trips", ("trips", "segments")))
        stand = self.metadata.columns[1]
        self.assertEqual(stand.domain[0], SQL_NULL)
        expected = complete_ordinary_domain([None if np.isnan(v) else int(v) for v in self.trips["origin_stand"]])
        self.assertEqual(stand.domain, expected)
        for index, values in ((0, self.trips["call_type"].tolist()),
                              (2, (self.trips["trip_time.start"] / 1e6).tolist())):
            self.assertEqual(self.metadata.columns[index].domain, complete_ordinary_domain(values))
        rows = np.load(self.prepared / "sample_rows.npy")
        trip_of_row = np.searchsorted(self.trips["trip_id"], self.segments["trip_id"])
        null_rows = np.isnan(self.trips["origin_stand"])[trip_of_row]
        self.assertTrue(np.all(rows[null_rows, 1] == 0))
        self.assertTrue(np.all(rows[~null_rows, 1] > 0))
        self.assertEqual(rows.shape, (self.segments["trip_id"].size, len(names)))
        # a seconds value equals the model's own timestamp parser on the text
        from model.src.data.trajectory_distinct import _timestamp_to_float

        first = int(self.segments["ts"][0])
        self.assertEqual(self.metadata.columns[names.index("segments:t_s")].domain[rows[0, names.index("segments:t_s")]],
                         _timestamp_to_float(_text_time(first)))

    def test_range_conditions_exclude_null(self) -> None:
        from model.src.evaluation.pol_query_adapter import pol_workload_record_to_context

        context = pol_workload_record_to_context(
            {"query_id": "q", "tables": ["trips"], "predicates": [
                {"table": "trips", "attribute": "origin_stand", "mode": "unbounded", "operator": "<", "value": 40}]},
            self.metadata, srid=3763)
        token = dict(context.ordinary_predicates)["trips:origin_stand"]
        self.assertFalse(token.satisfies(SQL_NULL))

    def test_sample_source_and_a_few_training_steps(self) -> None:
        from model.src.data.sample_sources import sample_source_from_config
        from model.src.training.resmade_trainer import train_resmade_sample_source

        config = load_simple_yaml(str(self.config))
        source = sample_source_from_config(config)  # validates the trajectory index against the metadata
        self.assertEqual([c.name for c in source.metadata.columns][0], "segments:segment_idx")  # segments first
        result = train_resmade_sample_source(source, config)
        self.assertTrue(Path(result.checkpoint_path).exists())
        self.assertTrue(np.isfinite(result.last_loss))


if __name__ == "__main__":
    unittest.main()
