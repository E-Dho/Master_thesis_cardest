"""Data-shift ablation: split selection, subset snapshot and in-distribution relabelling (traj_eval/data_shift.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_pol_porto.traj_eval import data_shift as ds
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap
from evaluation.trajectory_pol_porto.traj_eval import snapshot_gate
from evaluation.trajectory_pol_porto.traj_eval.workloads import select_rows

from . import _fixture, _porto_fixture


def _brute_join_cardinality(data, kept=None, parent=True) -> int:
    trip_ids = np.asarray(data["trips"]["trip_id"])
    keep = np.ones(trip_ids.size, dtype=bool) if kept is None else np.isin(trip_ids, kept)
    segments = np.asarray(data["segments"]["trip_id"])
    rows = 0
    for trip in trip_ids[keep]:
        rows += max(1, int(np.count_nonzero(segments == trip)))
    if parent:
        agents = np.asarray(data["agents"]["agent_id"])
        with_trips = set(np.asarray(data["trips"]["agent_id"])[keep].tolist())
        rows += sum(1 for agent in agents if agent not in with_trips)
    return rows


class DataShiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.data = _fixture.synthetic_data()
        cls.snapshot = _fixture.write_synthetic_snapshot(cls.root / "snapshot", cls.data)
        cls.benchmark = _fixture.write_benchmark(cls.root / "benchmark.jsonl")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _split(self, mode: str, name: str, **kwargs) -> Path:
        out = self.root / name
        if not out.exists():
            ds.write_split(out, ds.select(self.snapshot, _fixture.V3_CONFIG, mode, **kwargs))
        return out

    def test_random_split_is_reproducible_and_keeps_80_percent(self) -> None:
        a = ds.select(self.snapshot, _fixture.V3_CONFIG, "random", seed=3)
        b = ds.select(self.snapshot, _fixture.V3_CONFIG, "random", seed=3)
        c = ds.select(self.snapshot, _fixture.V3_CONFIG, "random", seed=4)
        np.testing.assert_array_equal(a["kept_trip_ids"], b["kept_trip_ids"])
        self.assertFalse(np.array_equal(a["kept_trip_ids"], c["kept_trip_ids"]))
        total = len(self.data["trips"]["trip_id"])
        self.assertEqual(a["record"]["trips_kept"], round(0.8 * total))
        self.assertTrue(np.all(np.diff(a["kept_trip_ids"]) > 0))

    def test_temporal_split_keeps_the_earliest_trips(self) -> None:
        selection = ds.select(self.snapshot, _fixture.V3_CONFIG, "temporal")
        record = selection["record"]
        self.assertEqual(record["start_column"], "trip_time.start")
        start = np.asarray(self.data["trips"]["trip_time.start"])
        kept = np.isin(np.asarray(self.data["trips"]["trip_id"]), selection["kept_trip_ids"])
        self.assertLessEqual(start[kept].max(), start[~kept].min())
        self.assertEqual(start[kept].max(), record["cutoff_start_us"])

    def test_join_cardinalities_match_a_brute_force_count(self) -> None:
        for mode in ds.MODES:
            selection = ds.select(self.snapshot, _fixture.V3_CONFIG, mode)
            record = selection["record"]
            self.assertEqual(record["full_join_cardinality_full"], _brute_join_cardinality(self.data))
            self.assertEqual(record["full_join_cardinality_subset"],
                             _brute_join_cardinality(self.data, selection["kept_trip_ids"]))
            self.assertEqual(record["parent"], ["agents", "agent_id"])
            np.testing.assert_array_equal(selection["all_trip_ids"], self.data["trips"]["trip_id"])
            np.testing.assert_array_equal(selection["all_trip_parent_ids"], self.data["trips"]["agent_id"])

    def test_subset_snapshot_holds_exactly_the_kept_data(self) -> None:
        split = self._split("random", "split_random")
        record, kept = ds.read_split(split)
        subset = self.root / "subset_random"
        if not subset.exists():
            ds.subset_snapshot(self.snapshot, _fixture.V3_CONFIG, split, subset)
        data = snap.load_snapshot(subset)
        trips = np.asarray(self.data["trips"]["trip_id"])
        mask = np.isin(trips, kept)
        np.testing.assert_array_equal(data["trips"]["trip_id"], trips[mask])
        np.testing.assert_array_equal(data["trips"]["trip_time.start"], np.asarray(self.data["trips"]["trip_time.start"])[mask])
        seg_mask = np.isin(np.asarray(self.data["segments"]["trip_id"]), kept)
        for name in ("trip_id", "sx", "ey", "ts"):
            np.testing.assert_array_equal(data["segments"][name], np.asarray(self.data["segments"][name])[seg_mask])
        np.testing.assert_array_equal(data["agents"]["agent_id"], self.data["agents"]["agent_id"])  # whole
        index = snap.load_trip_index(subset)
        np.testing.assert_array_equal(index["trip_id"], np.unique(data["segments"]["trip_id"]))
        manifest = snap.read_manifest(subset)
        self.assertEqual(manifest["subset"]["kept_trip_ids_sha256"], record["kept_trip_ids_sha256"])
        self.assertEqual(snap.verify_snapshot(subset), [])  # every file matches its checksum
        with self.assertRaises(FileExistsError):
            ds.subset_snapshot(self.snapshot, _fixture.V3_CONFIG, split, subset)

    def test_relabelled_benchmark_counts_on_the_subset(self) -> None:
        from query_generation.benchmark_v3.numpy_backend import NumpyBackend

        split = self._split("temporal", "split_temporal")
        subset = self.root / "subset_temporal"
        ds.subset_snapshot(self.snapshot, _fixture.V3_CONFIG, split, subset)
        results = self.root / "gate_temporal"
        for shard in range(2):
            snapshot_gate.run_shard(subset, _fixture.V3_CONFIG, self.benchmark, results, shard, 2, log=lambda m: None)
        out = self.root / "relabelled" / "benchmark.jsonl"
        summary = ds.relabel(self.benchmark, results, split, subset, out)
        rows = [json.loads(line) for line in self.benchmark.read_text().splitlines()]
        relabelled = [json.loads(line) for line in out.read_text().splitlines()]
        backend = NumpyBackend(_fixture.synthetic_settings(), snap.load_snapshot(subset))
        self.assertEqual(len(relabelled), len(rows))
        for old, new in zip(rows, relabelled):
            self.assertEqual(new["cardinality"], backend.count(old))
            self.assertLessEqual(new["cardinality"], old["cardinality"])  # a subset never counts more
            self.assertEqual({k: v for k, v in new.items() if k not in ("cardinality", "join_cardinality",
                                                                       "entity_cardinality", "data_shift_relabel")},
                             {k: v for k, v in old.items() if k not in ("cardinality", "join_cardinality",
                                                                       "entity_cardinality")})
            self.assertEqual(new["data_shift_relabel"]["full_data_cardinality"], old["cardinality"])
        self.assertGreater(summary["labels_changed"], 0)
        # the pipeline's loader accepts it, zero labels included
        self.assertEqual(len(select_rows(out, {"relation": "multi"})), 8 * _fixture.MULTI)
        # gate results of another snapshot, or an incomplete gate, are refused
        with self.assertRaisesRegex(ValueError, "not the subset snapshot"):
            ds.relabel(self.benchmark, results, split, self.snapshot, self.root / "x.jsonl")
        partial = self.root / "gate_partial"
        snapshot_gate.run_shard(subset, _fixture.V3_CONFIG, self.benchmark, partial, 0, 2, log=lambda m: None)
        with self.assertRaisesRegex(ValueError, "shard files"):
            ds.relabel(self.benchmark, partial, split, subset, self.root / "y.jsonl")

    def test_cli(self) -> None:
        out = self.root / "cli_split"
        self.assertEqual(ds.main(["select", "--snapshot", str(self.snapshot), "--config", str(_fixture.V3_CONFIG),
                                  "--mode", "random", "--seed", "1", "--output", str(out)]), 0)
        record, kept = ds.read_split(out)
        self.assertEqual((record["mode"], record["seed"], len(kept)), ("random", 1, record["trips_kept"]))
        np.testing.assert_array_equal(np.load(out / ds.ALL_TRIPS_FILE), self.data["trips"]["trip_id"])
        np.testing.assert_array_equal(np.load(out / ds.PARENTS_FILE), self.data["trips"]["agent_id"])
        with self.assertRaises(FileExistsError):
            ds.main(["select", "--snapshot", str(self.snapshot), "--config", str(_fixture.V3_CONFIG),
                     "--mode", "random", "--output", str(out)])


class SubmitDataShiftTest(unittest.TestCase):
    def _submit(self, env_extra: dict):
        import os
        import subprocess

        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "sbatch"
            log = Path(tmp) / "sbatch.log"
            counter = Path(tmp) / "id"
            counter.write_text("100")
            fake.write_text('#!/bin/bash\necho "$@" >> "$FAKE_SBATCH_LOG"\n'
                            'n=$(cat "$FAKE_COUNTER"); n=$((n + 1)); echo $n > "$FAKE_COUNTER"; echo $n\n')
            fake.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ("TRAJ_DATASET", "SPLIT", "MODEL_REPO", "GATE_SHARDS")}
            env.update({"PATH": f"{tmp}:{os.environ['PATH']}", "FAKE_SBATCH_LOG": str(log), "FAKE_COUNTER": str(counter),
                        "REPO_ROOT": str(repo), **env_extra})
            result = subprocess.run(["bash", str(repo / "evaluation/trajectory_pol_porto/slurm/submit_data_shift.sh")],
                                    env=env, capture_output=True, text=True)
            submitted = log.read_text().splitlines() if log.exists() else []
        return result, submitted

    def test_chain_and_dataset_paths(self) -> None:
        for dataset, prepared in (("pol_v3", "Master_thesis_cardest/data/pol_prepared/pol_50m_mbr"),
                                  ("porto_v3", "model_data/porto_v3_mbr")):
            result, lines = self._submit({"TRAJ_DATASET": dataset, "SPLIT": "temporal80", "MODEL_REPO": "/m"})
            self.assertEqual(result.returncode, 0, result.stderr)
            steps = [line.split("STEP=")[1].split(",")[0].split()[0] for line in lines]
            self.assertEqual(steps, ["select", "subset", "gate", "relabel", "prepare"])
            self.assertNotIn("--dependency", lines[0])
            self.assertIn("--dependency=afterok:101 ", lines[1])  # subset after select
            self.assertIn("--dependency=afterok:102 --array=0-15 ", lines[2])  # the gate after the subset
            self.assertIn("--dependency=afterok:103 ", lines[3])  # relabel after the whole gate array
            self.assertIn("--dependency=afterok:101 ", lines[4])  # prepare after select
            self.assertTrue(all(f"TRAJ_DATASET={dataset}" in line and "SPLIT=temporal80" in line for line in lines))
            self.assertIn(f"FULL_PREPARED=/work_beegfs/sunip956/master_thesis_trajectories/{prepared}", lines[4])
            self.assertIn("MODEL_REPO=/m", lines[4])
            self.assertIn("PREPARE_JOB=105", result.stdout)
        result, lines = self._submit({"SPLIT": "random80"})
        self.assertNotEqual(result.returncode, 0)  # MODEL_REPO missing
        self.assertEqual(lines, [])


class PortoDataShiftTest(unittest.TestCase):
    def test_porto_has_no_parent_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = _porto_fixture.write_porto_snapshot(root / "snapshot")
            data = _porto_fixture.porto_synthetic_data()
            for mode in ds.MODES:
                selection = ds.select(snapshot, _porto_fixture.PORTO_CONFIG, mode)
                record = selection["record"]
                self.assertIsNone(record["parent"])
                self.assertEqual(record["full_join_cardinality_full"], _brute_join_cardinality(data, parent=False))
                self.assertEqual(record["full_join_cardinality_subset"],
                                 _brute_join_cardinality(data, selection["kept_trip_ids"], parent=False))
            split = root / "split"
            ds.write_split(split, ds.select(snapshot, _porto_fixture.PORTO_CONFIG, "temporal"))
            ds.subset_snapshot(snapshot, _porto_fixture.PORTO_CONFIG, split, root / "subset")
            self.assertEqual(snap.load_snapshot(root / "subset")["trips"]["trip_id"].size,
                             ds.read_split(split)[0]["trips_kept"])


if __name__ == "__main__":
    unittest.main()
