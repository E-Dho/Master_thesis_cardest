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


def _training_predicate_config(config: dict) -> dict:
    """The generator configuration training uses (adds the top-level trajectory_spatial)."""

    from model.src.training.resmade_trainer import _predicate_generation_config

    return _predicate_generation_config(config)


def _generator(probability: float | None) -> PredicateTrainingContextGenerator:
    fixture = _PolFixture.load()
    config = _training_predicate_config(fixture.config)
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
    def test_fixture_uses_the_training_mbr_path(self) -> None:
        generator = _generator(0.5)
        self.assertTrue(generator.trajectory_spatial_enabled)
        contexts, _, _ = _contexts(0.5)
        kinds = {type(spatial).__name__ for context in contexts if context.trajectory_query is not None
                 for spatial in getattr(context.trajectory_query, "spatial_predicates", ())}
        self.assertEqual(kinds, {"SegmentMbrSpatialPredicate"})

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
        # Segment-MBR windows: when both drawn bounds fall inside the row's own MBR the
        # window is the part between them, so a few narrow windows exist even uniformly.
        uniform_narrow = np.mean(np.asarray(uniform["window_x"]) < 100.0)
        gated_narrow = np.mean(np.asarray(gated["window_x"]) < 100.0)
        self.assertLess(uniform_narrow, 0.05)
        self.assertGreater(gated_narrow, 0.1)
        self.assertGreater(gated_narrow, 5 * uniform_narrow)
        self.assertEqual(sum(width < 3600.0 for width in uniform["segment_interval"]), 0)
        self.assertGreater(sum(width < 3600.0 for width in gated["segment_interval"]), 0)

    def test_endpoint_windows_without_mbr_columns(self) -> None:
        # the no-MBR ablation's path: windows on the segment endpoints
        fixture = _PolFixture.load()
        widths = {}
        for probability in (0.0, 1.0):
            config = _training_predicate_config(fixture.config)
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


def _write_dense_pol_staging(staging: Path) -> None:
    """Long segments (5-50 min) and trips packed into a few hours: timestamp domains
    dense relative to segment durations, unlike the sparse prefetch fixture."""

    from model.tests.test_training_prefetch import _iso

    rng = np.random.default_rng(1)
    staging.mkdir(parents=True)
    agents, trips, segments = [], [], []
    trip_id = 0
    for agent in range(1, 121):
        agents.append(f"{agent}\t{rng.integers(18, 60)}.0\t{['Graduate', 'Bachelors', 'Low'][agent % 3]}\t"
                      f"{'ABC'[agent % 3]}\t{rng.random():.6f}\t{1 + agent % 3}")
        for _ in range(int(rng.integers(1, 4))):
            trip_id += 1
            length = int(rng.integers(2, 6))
            start = 1_600_000_000 + int(rng.integers(0, 20_000))
            x, y, t = rng.uniform(740_000, 744_000), rng.uniform(3_734_000, 3_739_000), start
            for index in range(length):
                dx, dy, dt = rng.normal(0, 300), rng.normal(0, 300), int(rng.integers(300, 3000))
                segments.append(f"{trip_id}\t{index}\t{x:.2f}\t{y:.2f}\t{x + dx:.2f}\t{y + dy:.2f}\t"
                                f"{_iso(t)}\t{_iso(t + dt)}")
                x, y, t = x + dx, y + dy, t + dt
            trips.append(f"{trip_id}\t{agent}\t{_iso(start)}\t{_iso(t)}\t{length}")
    (staging / "agents.tsv").write_text("\n".join(agents) + "\n")
    (staging / "trips.tsv").write_text("\n".join(trips) + "\n")
    (staging / "segments.tsv").write_text("\n".join(segments) + "\n")


class DenseTimestampGateTest(unittest.TestCase):
    """Bounds drawn from a row's start and end can cross when the log distances are
    shorter than the row; for the segment temporal context (one interval query) the
    generator must then return the interval inside the row (lower <= upper). Ordinary
    start/end predicates are independent and keep their values."""

    @classmethod
    def setUpClass(cls) -> None:
        from model.src.data.sample_sources import sample_source_from_config

        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        _write_dense_pol_staging(root / "staging")
        text = (REPO / BASE_CONFIG).read_text().replace(PREPARED, str(root / "prepared"))
        (root / "prepare.yaml").write_text(text)
        subprocess.run([sys.executable, "-m", "model.scripts.prepare_pol_staging_data", "--config",
                        str(root / "prepare.yaml"), "--staging-dir", str(root / "staging")],
                       cwd=REPO, check=True, capture_output=True, text=True)
        config = load_simple_yaml(str(root / "prepare.yaml"))
        config["rare_support"]["enabled"] = False
        config["rare_auxiliary"]["enabled"] = False
        source = sample_source_from_config(config)
        cls.metadata = source.metadata
        cls.rows = np.concatenate([source.batches(512, seed=seed).encoded_values for seed in range(6)])
        cls.predicate_config = _training_predicate_config(config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def _generate(self, probability: float):
        config = {**self.predicate_config, "log_width_gate": {**GATE, "probability": probability}}
        return PredicateTrainingContextGenerator(config).generate_batch(
            encoded_rows=self.rows, metadata=self.metadata, rng=np.random.default_rng(2)
        )

    def _value(self, row, name: str) -> float:
        index = [column.name for column in self.metadata.columns].index(name)
        return _seconds(self.metadata.columns[index].domain[int(row[index])])

    def test_no_inverted_intervals_and_narrow_ones_inside_rows(self) -> None:
        for probability in (0.5, 1.0):
            with self.subTest(probability=probability):
                self._check_no_inverted_intervals(probability)

    def _check_no_inverted_intervals(self, probability: float) -> None:
        contexts, rows, stats = self._generate(probability)
        self.assertEqual(stats.rejected_unsatisfied_contexts, 0)
        segment_intervals = inside_segment = trip_pairs = inside_trip = spanning_trip = 0
        for context, row in zip(contexts, rows):
            query = context.trajectory_query
            for temporal in getattr(query, "temporal_predicates", ()) if query is not None else ():
                lower, upper = _seconds(temporal.lower), _seconds(temporal.upper)
                self.assertLessEqual(lower, upper)
                segment_intervals += 1
                inside_segment += lower > self._value(row, "segments:t_s") and upper < self._value(row, "segments:t_e")
            for start_name, end_name in (("trips:start_time", "trips:end_time"), ("segments:t_s", "segments:t_e")):
                start = context.ordinary_predicates.get(start_name)
                end = context.ordinary_predicates.get(end_name)
                # Ordinary start/end bounds are independent Duet predicates: L > U is a
                # valid conjunction (rows spanning the gap) and is left as drawn.
                if (start is not None and end is not None and start.op == PredicateOp.LESS_EQUAL
                        and end.op == PredicateOp.GREATER_EQUAL and start_name == "trips:start_time"):
                    trip_pairs += 1
                    lower, upper = _seconds(end.value), _seconds(start.value)
                    inside_trip += (lower <= upper and lower > self._value(row, start_name)
                                    and upper < self._value(row, end_name))
                    spanning_trip += lower > upper
        self.assertGreater(segment_intervals, 100)
        self.assertGreater(trip_pairs, 20)
        # narrow intervals inside the row's own span exist (the case that was inverted)
        self.assertGreater(inside_segment, 0.03 * segment_intervals)
        self.assertGreater(inside_trip, 0)
        if probability == 1.0:
            self.assertGreater(spanning_trip, 0)  # independent predicates are not rewritten

    def test_uniform_generator_unchanged_on_dense_domains(self) -> None:
        absent = PredicateTrainingContextGenerator(self.predicate_config).generate_batch(
            encoded_rows=self.rows, metadata=self.metadata, rng=np.random.default_rng(2)
        )
        zero = self._generate(0.0)
        self.assertEqual([c.tokens for c in absent[0]], [c.tokens for c in zero[0]])


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
