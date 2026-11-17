"""Data-shift ablation, model side: the 80 % preparation as a subset of the full one, and fine-tuning.

On a synthetic POL staging set: the full preparation (the evaluated config, small
sizes), a temporal 80 % split of the trips, ``subset_prepared_full_join``, and an
independent fresh preparation of the filtered staging files. Every row of the
subset must decode to exactly the values of the fresh preparation (fanouts
included); training runs on the subset; fine-tuning starts from its checkpoint
on the full preparation; a checkpoint with other value domains is refused.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "model/configs/pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staging(directory: Path, *, agents: int = 30, trips_per_agent: int = 8) -> dict:
    rng = np.random.default_rng(0)
    directory.mkdir(parents=True)
    fmt = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")  # noqa: E731
    agent_lines, trip_lines, segment_lines, trips = [], [], [], []
    trip_id = 100
    for agent in range(1, agents + 1):
        agent_lines.append(f"{agent}\t{20 + agent % 40}.0\t{['Low', 'HighSchool', 'Graduate'][agent % 3]}\t"
                           f"{'ABCD'[agent % 4]}\t{(agent % 10) / 10}\t{1 + agent % 4}")
        for _ in range(trips_per_agent):
            trip_id += 1
            count = int(rng.integers(2, 7))
            start = 1577836800 + int(rng.integers(0, 86400 * 20))
            x, y, t = 500000 + rng.uniform(0, 5000), 4400000 + rng.uniform(0, 5000), start
            for index in range(count):
                nx, ny, nt = x + rng.normal(0, 200), y + rng.normal(0, 200), t + int(rng.integers(30, 600))
                segment_lines.append(f"{trip_id}\t{index}\t{x:.3f}\t{y:.3f}\t{nx:.3f}\t{ny:.3f}\t{fmt(t)}\t{fmt(nt)}")
                x, y, t = nx, ny, nt
            trip_lines.append(f"{trip_id}\t{agent}\t{fmt(start)}\t{fmt(t)}\t{count}")
            trips.append((trip_id, agent, start, count))
    (directory / "agents.tsv").write_text("\n".join(agent_lines) + "\n")
    (directory / "trips.tsv").write_text("\n".join(trip_lines) + "\n")
    (directory / "segments.tsv").write_text("\n".join(segment_lines) + "\n")
    return {"trips": np.asarray(trips, dtype=np.int64)}


def _config(path: Path, prepared: Path, run: Path, *, steps: int = 4, extra: str = "") -> Path:
    text = CONFIG.read_text()
    text = re.sub(r"(?m)^  prepared_directory: .*", f"  prepared_directory: {prepared}", text)
    for key in ("trajectory_ids_path", "segment_ids_path"):
        text = re.sub(rf"(?m)^  {key}: .*/([^/]+)$", lambda m: f"  {key}: {prepared}/{m.group(1)}", text)
    text = re.sub(r"(?m)^  trajectory_index_path: .*", f"  trajectory_index_path: {prepared}/trajectory_segment_index", text)
    for pattern, value in ((r"hidden_sizes: \[512, 512, 512, 512\]", "hidden_sizes: [16, 16]"),
                           (r"(?m)^  sample_batch_size: 2048", "  sample_batch_size: 32"),
                           (r"(?m)^  batch_size: 2048", "  batch_size: 32"),
                           (r"(?m)^  steps_per_epoch: 20000", f"  steps_per_epoch: {steps}"),
                           (r"(?m)^  checkpoint_interval_steps: 500", "  checkpoint_interval_steps: 2"),
                           (r"(?m)^  validation_interval_steps: 250", "  validation_interval_steps: 2"),
                           (r"(?m)^  interval_steps: 250", "  interval_steps: 2"),
                           (r"(?m)^  fresh_sampler_batches: 4", "  fresh_sampler_batches: 1"),
                           (r"(?m)^    support_planning_steps: 20000", f"    support_planning_steps: {steps}"),
                           (r"(?m)^  batch_size: 512", "  batch_size: 8"),
                           (r"(?m)^  device: auto", "  device: cpu")):
        text, n = re.subn(pattern, value, text)
        assert n >= 1, pattern
    text = re.sub(r"(?m)^  output_directory: .*", f"  output_directory: {run}", text)
    if extra:
        text = text.replace("training:\n", "training:\n" + extra, 1)
    path.write_text(text)
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, "-m", *args], cwd=REPO, capture_output=True, text=True,
                            env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    return result


def _decoded_rows(prepared: Path) -> dict:
    manifest = json.loads((prepared / "manifest.json").read_text())
    domains = [column["domain"] for column in manifest["metadata"]["columns"]]
    rows = np.load(prepared / "sample_rows.npy")
    segments = np.load(prepared / "sample_segment_ids.npy")
    decoded = {}
    for row, segment in zip(rows, segments):
        decoded[(int(segment[0]), int(segment[1]))] = tuple(domains[i][int(code)] for i, code in enumerate(row))
    return decoded


def _split(directory: Path, trips: np.ndarray, *, fraction: float = 0.8) -> np.ndarray:
    """A temporal split in the format of evaluation/.../data_shift.py select."""

    directory.mkdir(parents=True)
    trip_ids, parents, start, counts = trips[:, 0], trips[:, 1], trips[:, 2], trips[:, 3]
    order = np.lexsort((trip_ids, start))
    keep_count = int(round(fraction * trip_ids.size))
    kept_mask = np.zeros(trip_ids.size, dtype=bool)
    kept_mask[order[:keep_count]] = True
    kept = np.sort(trip_ids[kept_mask])
    np.save(directory / "kept_trip_ids.npy", kept)
    np.save(directory / "all_trip_ids.npy", trip_ids)
    np.save(directory / "all_trip_parent_ids.npy", parents)
    all_parents = np.unique(parents)
    record = {
        "mode": "temporal", "fraction": fraction, "entity_table": "trips", "parent": ["agents", "agent_id"],
        "trips_total": int(trip_ids.size), "trips_kept": int(kept.size),
        "parents_losing_every_trip": int(all_parents.size - np.unique(parents[kept_mask]).size),
        "full_join_cardinality_full": int(np.maximum(counts, 1).sum()),
        "full_join_cardinality_subset": int(np.maximum(counts[kept_mask], 1).sum()),
        "kept_trip_ids_sha256": _sha256(directory / "kept_trip_ids.npy"),
        "all_trip_ids_sha256": _sha256(directory / "all_trip_ids.npy"),
        "all_trip_parent_ids_sha256": _sha256(directory / "all_trip_parent_ids.npy"),
    }
    (directory / "split.json").write_text(json.dumps(record))
    return kept


class DataShiftSubsetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from model.scripts.subset_prepared_full_join import subset

        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        cls.staging = _staging(root / "staging")
        cls.full = root / "prepared_full"
        _run("model.scripts.prepare_pol_staging_data", "--config",
             str(_config(root / "full.yaml", cls.full, root / "runs/full")), "--staging-dir", str(root / "staging"))
        cls.kept = _split(root / "split", cls.staging["trips"])
        cls.subset = root / "prepared_subset"
        cls.summary = subset(cls.full, root / "split", cls.subset)
        # an independent fresh preparation of the filtered staging files
        filtered = root / "staging_filtered"
        filtered.mkdir()
        kept = set(cls.kept.tolist())
        (filtered / "agents.tsv").write_text((root / "staging/agents.tsv").read_text())
        for name in ("trips.tsv", "segments.tsv"):
            lines = (root / "staging" / name).read_text().splitlines()
            (filtered / name).write_text("\n".join(l for l in lines if int(l.split("\t")[0]) in kept) + "\n")
        cls.fresh = root / "prepared_fresh"
        _run("model.scripts.prepare_pol_staging_data", "--config",
             str(_config(root / "fresh.yaml", cls.fresh, root / "runs/fresh")), "--staging-dir", str(filtered))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_subset_rows_decode_to_a_fresh_preparation_of_the_kept_data(self) -> None:
        subset_rows, fresh_rows = _decoded_rows(self.subset), _decoded_rows(self.fresh)
        self.assertEqual(set(subset_rows), set(fresh_rows))
        self.assertEqual(subset_rows, fresh_rows)  # every column, F_agents_to_trips included
        self.assertGreater(self.summary["fanout_rows_changed"], 0)  # agents did lose trips
        manifest = json.loads((self.subset / "manifest.json").read_text())
        fresh = json.loads((self.fresh / "manifest.json").read_text())
        self.assertEqual(manifest["join_cardinality"], fresh["join_cardinality"])
        self.assertEqual(manifest["metadata"]["full_join_cardinality"], fresh["metadata"]["full_join_cardinality"])
        full = json.loads((self.full / "manifest.json").read_text())
        self.assertEqual([c["domain"] for c in manifest["metadata"]["columns"]],
                         [c["domain"] for c in full["metadata"]["columns"]])  # the full vocabulary
        self.assertEqual((self.summary["full_join_cardinality"], self.summary["subset_join_cardinality"]),
                         (full["join_cardinality"], fresh["join_cardinality"]))
        self.assertEqual(json.loads((self.subset / "subset.json").read_text())["subset_join_cardinality"],
                         fresh["join_cardinality"])

    def test_inconsistent_inputs_are_refused(self) -> None:
        from model.scripts.subset_prepared_full_join import subset

        bad = self.root / "split_bad"
        bad.mkdir()
        for name in ("kept_trip_ids.npy", "all_trip_ids.npy", "all_trip_parent_ids.npy"):
            (bad / name).write_bytes((self.root / "split" / name).read_bytes())
        record = json.loads((self.root / "split" / "split.json").read_text())
        (bad / "split.json").write_text(json.dumps({**record, "full_join_cardinality_full": 1}))
        with self.assertRaisesRegex(ValueError, "not the same data"):
            subset(self.full, bad, self.root / "out_bad")
        (bad / "split.json").write_text(json.dumps({**record, "parents_losing_every_trip": 2}))
        with self.assertRaisesRegex(ValueError, "lose every trip"):
            subset(self.full, bad, self.root / "out_bad2")
        with self.assertRaises(FileExistsError):
            subset(self.full, self.root / "split", self.subset)

    def test_training_on_the_subset_and_fine_tuning_on_the_full_data(self) -> None:
        import torch

        from model.src.config import load_simple_yaml
        from model.src.data.sample_sources import sample_source_from_config
        from model.src.training.resmade_trainer import _initialize_from_checkpoint, build_resmade_from_config

        stale = self.root / "runs/stale"
        _run("model.scripts.train_resmade", "--config", str(_config(self.root / "stale.yaml", self.subset, stale)))
        checkpoint = stale / "checkpoint_best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(payload["join_cardinality"], float(self.summary["subset_join_cardinality"]))
        # fine-tuning: the full preparation, starting from the stale model's weights
        tuned = self.root / "runs/tuned"
        config = _config(self.root / "tuned.yaml", self.full, tuned, extra=f"  init_checkpoint: {checkpoint}\n")
        _run("model.scripts.train_resmade", "--config", str(config))
        summary = json.loads((tuned / "training_summary.json").read_text())
        self.assertEqual(summary["init_checkpoint"]["path"], str(checkpoint))
        self.assertEqual(summary["init_checkpoint"]["join_cardinality"], float(self.summary["subset_join_cardinality"]))
        tuned_payload = torch.load(tuned / "checkpoint_best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(tuned_payload["join_cardinality"], float(self.summary["full_join_cardinality"]))
        # the weights are loaded exactly
        loaded = load_simple_yaml(str(config))
        source = sample_source_from_config(loaded)
        model = build_resmade_from_config(source.metadata, loaded)
        _initialize_from_checkpoint(model, str(checkpoint), source.metadata, "cpu")
        for name, tensor in payload["model_state_dict"].items():
            self.assertTrue(torch.equal(model.state_dict()[name], tensor), name)
        # a model with other value domains (the fresh preparation) cannot be fine-tuned here
        fresh_run = self.root / "runs/fresh_model"
        _run("model.scripts.train_resmade", "--config", str(_config(self.root / "fresh_train.yaml", self.fresh, fresh_run, steps=2)))
        with self.assertRaisesRegex(ValueError, "value domains"):
            _initialize_from_checkpoint(build_resmade_from_config(source.metadata, loaded),
                                        str(fresh_run / "checkpoint_best.pt"), source.metadata, "cpu")
        self.assertIsNone(_initialize_from_checkpoint(model, None, source.metadata, "cpu"))


class PortoSubsetTest(unittest.TestCase):
    """Porto (prepared from the snapshot; trips are the root, no parent fanout to recompute)."""

    def test_subset_equals_a_fresh_preparation_and_trains(self) -> None:
        from model.scripts import prepare_trajectory_snapshot as prep
        from model.scripts.subset_prepared_full_join import subset
        from model.src.config import load_simple_yaml
        from model.src.data.sample_sources import sample_source_from_config
        from model.src.training.resmade_trainer import train_resmade_sample_source
        from model.tests.test_prepare_trajectory_snapshot import PORTO_CONFIG, _config as snapshot_config
        from model.tests.test_prepare_trajectory_snapshot import _segments, _write_snapshot

        rng = np.random.default_rng(11)
        trip_ids = np.arange(100, 160)
        segments, starts, ends, counts = _segments(rng, trip_ids)
        call_type = rng.choice(np.array(["A", "B", "C"]), trip_ids.size)
        stand = np.where(call_type == "B", rng.integers(1, 64, trip_ids.size).astype(float), np.nan)
        trips = {"trip_id": trip_ids, "call_type": call_type, "origin_stand": stand, "num_of_segments": counts,
                 "trip_time.start": starts, "trip_time.end": ends}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full"
            prep.prepare("porto", snapshot_config(PORTO_CONFIG, root / "full.yaml", full),
                         _write_snapshot(root / "snap", {"trips": trips, "segments": segments}), log=lambda m: None)
            split = root / "split"
            split.mkdir()
            order = np.lexsort((trip_ids, starts))
            kept_mask = np.zeros(trip_ids.size, dtype=bool)
            kept_mask[order[:48]] = True
            np.save(split / "kept_trip_ids.npy", trip_ids[kept_mask])
            np.save(split / "all_trip_ids.npy", trip_ids)
            (split / "split.json").write_text(json.dumps({
                "mode": "temporal", "entity_table": "trips", "parent": None, "parents_losing_every_trip": 0,
                "full_join_cardinality_full": int(counts.sum()),
                "full_join_cardinality_subset": int(counts[kept_mask].sum()),
                "kept_trip_ids_sha256": _sha256(split / "kept_trip_ids.npy"),
                "all_trip_ids_sha256": _sha256(split / "all_trip_ids.npy")}))
            summary = subset(full, split, root / "subset")
            self.assertEqual(summary["fanout_rows_changed"], 0)
            keep_segment = np.isin(segments["trip_id"], trip_ids[kept_mask])
            fresh = root / "fresh"
            prep.prepare("porto", snapshot_config(PORTO_CONFIG, root / "fresh.yaml", fresh),
                         _write_snapshot(root / "snap_kept", {
                             "trips": {k: v[kept_mask] for k, v in trips.items()},
                             "segments": {k: v[keep_segment] for k, v in segments.items()}}), log=lambda m: None)
            self.assertEqual(_decoded_rows(root / "subset"), _decoded_rows(fresh))
            config = load_simple_yaml(str(snapshot_config(PORTO_CONFIG, root / "train.yaml", root / "subset", steps=True)))
            result = train_resmade_sample_source(sample_source_from_config(config), config)
            self.assertTrue(np.isfinite(result.last_loss))


if __name__ == "__main__":
    unittest.main()
