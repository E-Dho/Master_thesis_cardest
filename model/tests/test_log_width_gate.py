"""predicate_generation.log_width_gate on a small prepared POL dataset.

Per context and dimension (time, space) a coin with probability p decides whether
the bounds on that dimension's columns are drawn at a log-uniform distance from the
row's own value (between min_width and the column's domain span) or uniformly over
the domain ranks as before. Checked here:

- p = 0 (and no section) gives exactly the contexts of the uniform generator;
- p = 1: every context is still satisfied by its row, and temporal intervals and
  spatial windows become much narrower (narrow ones appear at all);
- p = 0.5 mixes the two;
- the numeric view of a domain skips sentinels and refuses an unordered domain.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.predicates.generation import PredicateTrainingContextGenerator, _numeric_rank_values
from model.src.predicates.operators import PredicateOp

REPO = Path(__file__).resolve().parents[2]
BASE_CONFIG = "model/configs/pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"
PREPARED = "/work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest/data/pol_prepared/pol_50m_mbr"

GATE = {
    "probability": 1.0,
    "dimensions": {
        "space": {"min_width": 1.0, "columns": [
            "segments:s_x", "segments:s_y", "segments:e_x", "segments:e_y",
            "segments:seg_min_x", "segments:seg_max_x", "segments:seg_min_y", "segments:seg_max_y"]},
        "time": {"min_width": 1.0, "columns": [
            "trips:start_time", "trips:end_time", "segments:t_s", "segments:t_e"]},
    },
}


def _seconds(value) -> float:
    from model.src.data.trajectory_distinct import _timestamp_to_float

    return _timestamp_to_float(value)


class _PolFixture:
    directory: tempfile.TemporaryDirectory
    metadata = None
    rows = None
    config = None

    @classmethod
    def load(cls):
        if cls.metadata is not None:
            return cls
        from model.src.data.sample_sources import sample_source_from_config
        from model.tests.test_training_prefetch import _write_pol_staging

        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        _write_pol_staging(root / "staging")
        text = (REPO / BASE_CONFIG).read_text().replace(PREPARED, str(root / "prepared"))
        (root / "prepare.yaml").write_text(text)
        subprocess.run([sys.executable, "-m", "model.scripts.prepare_pol_staging_data", "--config",
                        str(root / "prepare.yaml"), "--staging-dir", str(root / "staging")],
                       cwd=REPO, check=True, capture_output=True, text=True)
        config = load_simple_yaml(str(root / "prepare.yaml"))
        validate_config(config)
        config["rare_support"]["enabled"] = False  # contexts only; rare strata are irrelevant here
        config["rare_auxiliary"]["enabled"] = False
        source = sample_source_from_config(config)
        cls.config = config
        cls.metadata = source.metadata
        cls.rows = np.concatenate([source.batches(256, seed=seed).encoded_values for seed in range(8)])
        return cls


def _generator(probability: float | None) -> PredicateTrainingContextGenerator:
    fixture = _PolFixture.load()
    config = dict(fixture.config["predicate_generation"])
    if probability is not None:
        config["log_width_gate"] = {**GATE, "probability": probability}
    return PredicateTrainingContextGenerator(config)


def _contexts(probability: float | None, seed: int = 3):
    fixture = _PolFixture.load()
    return _generator(probability).generate_batch(
        encoded_rows=fixture.rows, metadata=fixture.metadata, rng=np.random.default_rng(seed)
    )


def _widths(contexts) -> dict[str, list[float]]:
    widths = {"segment_interval": [], "trip_interval": [], "window_x": []}
    for context in contexts:
        query = context.trajectory_query
        for temporal in getattr(query, "temporal_predicates", ()) if query is not None else ():
            widths["segment_interval"].append(_seconds(temporal.upper) - _seconds(temporal.lower))
        for spatial in getattr(query, "spatial_predicates", ()) if query is not None else ():
            widths["window_x"].append(float(spatial.max_x) - float(spatial.min_x))
        start = context.ordinary_predicates.get("trips:start_time")
        end = context.ordinary_predicates.get("trips:end_time")
        if (start is not None and end is not None and start.op == PredicateOp.LESS_EQUAL
                and end.op == PredicateOp.GREATER_EQUAL):
            widths["trip_interval"].append(_seconds(start.value) - _seconds(end.value))
    return widths


class LogWidthGateTest(unittest.TestCase):
    def test_probability_zero_is_the_uniform_generator(self) -> None:
        absent, absent_targets, _ = _contexts(None)
        zero, zero_targets, _ = _contexts(0.0)
        self.assertEqual([c.tokens for c in absent], [c.tokens for c in zero])
        self.assertEqual([c.trajectory_query for c in absent], [c.trajectory_query for c in zero])
        self.assertTrue(np.array_equal(absent_targets, zero_targets))

    def test_gated_contexts_are_satisfied_by_their_rows(self) -> None:
        for probability in (0.5, 1.0):
            contexts, _, stats = _contexts(probability)
            self.assertEqual(stats.rejected_unsatisfied_contexts, 0, probability)
            self.assertEqual(stats.generated_contexts, len(_PolFixture.load().rows), probability)

    def test_gate_makes_intervals_and_windows_narrow(self) -> None:
        # The fixture's domains are sparse (a few thousand timestamps over a year), so
        # snapping a bound to the next domain value widens it far more than on the
        # real data; the checks are therefore relative.
        uniform = _widths(_contexts(0.0)[0])
        mixed = _widths(_contexts(0.5)[0])
        gated = _widths(_contexts(1.0)[0])
        for key, factor in (("segment_interval", 0.1), ("trip_interval", 0.1), ("window_x", 0.5)):
            self.assertGreater(len(uniform[key]), 20, key)
            self.assertGreater(len(gated[key]), 20, key)
            self.assertLess(np.median(gated[key]), factor * np.median(uniform[key]), key)
            self.assertLess(np.median(gated[key]), np.median(mixed[key]), key)
            self.assertLess(np.median(mixed[key]), np.median(uniform[key]), key)
        self.assertEqual(sum(width < 100.0 for width in uniform["window_x"]), 0)
        self.assertGreater(np.mean(np.asarray(gated["window_x"]) < 100.0), 0.1)
        self.assertEqual(sum(width < 3600.0 for width in uniform["segment_interval"]), 0)
        self.assertGreater(sum(width < 3600.0 for width in gated["segment_interval"]), 0)

    def test_endpoint_windows_without_mbr_columns(self) -> None:
        # the no-MBR ablation's path: windows on the segment endpoints
        fixture = _PolFixture.load()
        widths = {}
        for probability in (0.0, 1.0):
            config = dict(fixture.config["predicate_generation"])
            config["trajectory_spatial"] = {"enabled": False}
            config["log_width_gate"] = {**GATE, "probability": probability}
            contexts, _, stats = PredicateTrainingContextGenerator(config).generate_batch(
                encoded_rows=fixture.rows, metadata=fixture.metadata, rng=np.random.default_rng(5)
            )
            self.assertEqual(stats.rejected_unsatisfied_contexts, 0)
            widths[probability] = _widths(contexts)["window_x"]
            self.assertGreater(len(widths[probability]), 20)
        self.assertLess(np.median(widths[1.0]), 0.5 * np.median(widths[0.0]))

    def test_seeds_reproduce(self) -> None:
        first, _, _ = _contexts(0.5, seed=11)
        second, _, _ = _contexts(0.5, seed=11)
        self.assertEqual([c.tokens for c in first], [c.tokens for c in second])

    def test_invalid_configuration_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be in"):
            PredicateTrainingContextGenerator({"log_width_gate": {"probability": 1.5}})
        with self.assertRaisesRegex(ValueError, "needs dimensions"):
            PredicateTrainingContextGenerator({"log_width_gate": {"probability": 0.5}})
        with self.assertRaisesRegex(ValueError, "min_width must be positive"):
            PredicateTrainingContextGenerator({"log_width_gate": {"probability": 0.5, "dimensions": {
                "time": {"min_width": 0, "columns": ["trips:start_time"]}}}})

    def test_yaml_section_parses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.yaml"
            path.write_text(
                "predicate_generation:\n"
                "  log_width_gate:\n"
                "    probability: 0.5\n"
                "    dimensions:\n"
                "      time:\n"
                "        min_width: 1.0\n"
                "        columns:\n"
                "          - trips:start_time\n"
                "          - trips:end_time\n"
            )
            gate = load_simple_yaml(str(path))["predicate_generation"]["log_width_gate"]
        generator = PredicateTrainingContextGenerator({"log_width_gate": gate})
        self.assertEqual(generator.log_width_probability, 0.5)
        self.assertEqual(generator.log_width_column_dimension,
                         {"trips:start_time": "time", "trips:end_time": "time"})


class NumericRankValuesTest(unittest.TestCase):
    def test_timestamps_sentinels_and_order(self) -> None:
        values = ("2020-01-01T00:00:00.000", "2020-01-01 00:01:00", "2020-01-02T00:00:00.000", "__SQL_NULL__")
        numbers, first, stop = _numeric_rank_values(values)
        self.assertEqual((first, stop), (0, 3))
        self.assertEqual(list(np.diff(numbers[:3])), [60.0, 86340.0])
        self.assertEqual(_numeric_rank_values((1.0, 2.5, 7.0))[1:], (0, 3))
        self.assertIsNone(_numeric_rank_values((3.0, 1.0)))
        self.assertIsNone(_numeric_rank_values(("a", "b")))


if __name__ == "__main__":
    unittest.main()
