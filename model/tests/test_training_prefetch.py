"""training.prefetch_workers: steps prepared in worker processes give the same training.

Every random stream of a step is seeded by its step number, so preparing the
CPU part of future steps in forked workers must reproduce the serial run exactly:
metrics, training summary (rare-support counters included) and checkpoints. Two
fixtures: the synthetic source (plain steps) and a small POL-shaped trajectory
preparation with rare support and the dedup head (rare batches, dedup targets).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.src.config import load_simple_yaml, validate_config

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.training.resmade_trainer import train_resmade_sample_source

REPO = Path(__file__).resolve().parents[2]
TIMING_KEYS = ("timing", "seconds", "per_second", "prefetch", "worker_phase", "path", "checkpoint")


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if not any(s in k for s in TIMING_KEYS)}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _outputs(directory: Path):
    metrics = [_strip(json.loads(line)) for line in (directory / "training_metrics.jsonl").read_text().splitlines()]
    summary = _strip(json.loads((directory / "training_summary.json").read_text()))
    checkpoints = {}
    for path in sorted(directory.glob("*.pt")):
        payload = torch.load(path, weights_only=False)
        checkpoints[path.name] = (
            {name: tensor.clone() for name, tensor in payload["model_state_dict"].items()},
            {index: {key: tensor.clone() for key, tensor in state.items()}
             for index, state in payload["optimizer_state_dict"]["state"].items()},
            payload["step"],
        )
    return metrics, summary, checkpoints


def _assert_same(test: unittest.TestCase, serial: Path, parallel: Path) -> None:
    a_metrics, a_summary, a_checkpoints = _outputs(serial)
    b_metrics, b_summary, b_checkpoints = _outputs(parallel)
    test.assertEqual(a_metrics, b_metrics)
    test.assertEqual(a_summary, b_summary)
    test.assertEqual(a_checkpoints.keys(), b_checkpoints.keys())
    for name, (weights, optimizer, step) in a_checkpoints.items():
        other_weights, other_optimizer, other_step = b_checkpoints[name]
        test.assertEqual(step, other_step)
        test.assertTrue(all(torch.equal(weights[n], other_weights[n]) for n in weights), name)
        test.assertTrue(all(torch.equal(optimizer[i][k], other_optimizer[i][k])
                            for i in optimizer for k in optimizer[i]), name)
    report = json.loads((parallel / "training_summary.json").read_text())
    test.assertEqual(report["prefetch_workers"], 2)
    test.assertIn("predicate_context_generation", report["worker_phase_timing_summary"])
    test.assertIn("prefetch_wait", report["phase_timing_summary"])


def _write_pol_staging(staging: Path) -> None:
    """A few hundred POL-shaped agents/trips/segments, plus single-segment agents (rare strata)."""

    rng = np.random.default_rng(0)
    staging.mkdir(parents=True)
    agents, trips, segments = [], [], []
    trip_id = 0
    for agent in range(1, 141):
        agents.append(f"{agent}\t{rng.integers(18, 60)}.0\t{['Graduate', 'Bachelors', 'Low'][agent % 3]}\t"
                      f"{'ABC'[agent % 3]}\t{rng.random():.6f}\t{1 + agent % 3}")
        trip_count = 0 if agent % 5 == 0 else (1 if agent > 120 else int(rng.integers(1, 4)))
        for _ in range(trip_count):
            trip_id += 1
            length = 1 if agent > 120 else int(rng.integers(2, 8))
            start = 1_600_000_000 + int(rng.integers(0, 30_000_000))
            x, y, t = rng.uniform(740_000, 744_000), rng.uniform(3_734_000, 3_739_000), start
            for index in range(length):
                dx, dy, dt = rng.normal(0, 40), rng.normal(0, 40), int(rng.integers(5, 30))
                segments.append(f"{trip_id}\t{index}\t{x:.2f}\t{y:.2f}\t{x + dx:.2f}\t{y + dy:.2f}\t"
                                f"{_iso(t)}\t{_iso(t + dt)}")
                x, y, t = x + dx, y + dy, t + dt
            trips.append(f"{trip_id}\t{agent}\t{_iso(start)}\t{_iso(t)}\t{length}")
    (staging / "agents.tsv").write_text("\n".join(agents) + "\n")
    (staging / "trips.tsv").write_text("\n".join(trips) + "\n")
    (staging / "segments.tsv").write_text("\n".join(segments) + "\n")


def _iso(epoch_seconds: int) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(epoch_seconds, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000")


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class PrefetchEquivalenceTest(unittest.TestCase):
    def test_synthetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as serial, tempfile.TemporaryDirectory() as parallel:
            for directory, workers in ((serial, 0), (parallel, 2)):
                config = load_simple_yaml("model/configs/resmade_smoke.yaml")
                config["model"]["hidden_sizes"] = [16]
                config["model"]["embedding_size"] = 4
                config["training"].update(batch_size=8, steps_per_epoch=7, checkpoint_interval_steps=3,
                                          validation_interval_steps=2, prefetch_workers=workers)
                config["logging"]["output_directory"] = directory
                train_resmade_sample_source(SyntheticFullJoinSampleSource(), config)
            _assert_same(self, Path(serial), Path(parallel))

    def test_dropout_sees_the_same_random_stream(self) -> None:
        # the loader must not draw from the global torch generator that dropout uses
        with tempfile.TemporaryDirectory() as serial, tempfile.TemporaryDirectory() as parallel:
            for directory, workers in ((serial, 0), (parallel, 2)):
                config = load_simple_yaml("model/configs/resmade_smoke.yaml")
                config["model"]["hidden_sizes"] = [16, 16]
                config["model"]["embedding_size"] = 4
                config["model"]["residual_dropout"] = 0.3
                config["training"].update(batch_size=8, steps_per_epoch=5, checkpoint_interval_steps=0,
                                          validation_interval_steps=0, prefetch_workers=workers)
                config["logging"]["output_directory"] = directory
                train_resmade_sample_source(SyntheticFullJoinSampleSource(), config)
            _assert_same(self, Path(serial), Path(parallel))

    def test_pol_trajectory_source_with_rare_support_and_dedup_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_pol_staging(root / "staging")
            base = (REPO / "model/configs/pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml").read_text()
            old = "/work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest/data/pol_prepared/pol_50m_mbr"
            self.assertIn(old, base)
            base = base.replace(old, str(root / "prepared"))
            (root / "prepare.yaml").write_text(base)
            subprocess.run([sys.executable, "-m", "model.scripts.prepare_pol_staging_data", "--config",
                            str(root / "prepare.yaml"), "--staging-dir", str(root / "staging")],
                           cwd=REPO, check=True, capture_output=True, text=True)
            outputs = {}
            for name, workers in (("serial", 0), ("parallel", 2)):
                config = load_simple_yaml(str(root / "prepare.yaml"))
                validate_config(config)
                config["model"]["hidden_sizes"] = [32]
                config["model"]["embedding_size"] = 8
                config["training"].update(device="cpu", batch_size=64, checkpoint_interval_steps=4,
                                          validation_interval_steps=3, prefetch_workers=workers)
                config["validation"].update(interval_steps=3, batch_size=64, fresh_sampler_batches=2)
                config["rare_auxiliary"]["batch_size"] = 16
                # the fixture is tiny: demand more support so that discovery selects strata
                config["rare_support"]["discovery"]["minimum_expected_context_support"] = 10**9
                config["logging"]["output_directory"] = str(root / name)
                source = sample_source_from_config(config)  # rare-support planning sees the configured steps
                config["training"]["steps_per_epoch"] = 8
                train_resmade_sample_source(source, config)
                outputs[name] = root / name
            summary = json.loads((outputs["serial"] / "training_summary.json").read_text())
            self.assertTrue(summary["rare_support"]["enabled"])
            self.assertGreater(summary["rare_support"]["rare_rows_drawn"], 0)
            self.assertGreater(summary["trajectory_distinct"]["trajectory_targets_generated"], 0)
            _assert_same(self, outputs["serial"], outputs["parallel"])


class PrefetchSourceCheckTest(unittest.TestCase):
    def test_only_step_seeded_sources_are_accepted(self) -> None:
        from types import SimpleNamespace

        from model.src.data.full_join_sampler import (
            LiveNeuroCardFullJoinSampleSource,
            NeuroCardFullJoinSampleSource,
        )
        from model.src.training.resmade_trainer import _assert_prefetch_deterministic_source

        fixture = object.__new__(NeuroCardFullJoinSampleSource)
        fixture.sampling_mode = "fixture"
        _assert_prefetch_deterministic_source(fixture)
        live = object.__new__(LiveNeuroCardFullJoinSampleSource)
        live.sampling_mode = "live"  # buffered sampler: batches depend on earlier calls
        with self.assertRaisesRegex(ValueError, "prefetch_workers"):
            _assert_prefetch_deterministic_source(live)
        disguised = object.__new__(NeuroCardFullJoinSampleSource)
        disguised.sampling_mode = "live"
        with self.assertRaisesRegex(ValueError, "prefetch_workers"):
            _assert_prefetch_deterministic_source(disguised)
        with self.assertRaisesRegex(ValueError, "prefetch_workers"):
            _assert_prefetch_deterministic_source(SimpleNamespace(base_source=fixture))  # unknown wrapper


if __name__ == "__main__":
    unittest.main()
