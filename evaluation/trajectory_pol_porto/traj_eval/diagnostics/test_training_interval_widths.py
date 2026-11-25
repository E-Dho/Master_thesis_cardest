from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model.src.data.trajectory_distinct import SegmentTemporalPredicate, TrajectoryQuerySemantics
from model.src.predicates.generation import GeneratedTrainingContext
from model.src.predicates.operators import PredicateOp, PredicateToken

from evaluation.trajectory_pol_porto.traj_eval.diagnostics.training_interval_widths import (
    analyze_benchmark,
    extract_context_intervals,
    interval_report_key,
    one_sided_report_key,
    width_bin,
)


class TrainingIntervalWidthsTest(unittest.TestCase):
    def test_benchmark_reads_spatio_temporal_segment_interval(self) -> None:
        record = {
            "category": {"dimension": "spatio_temporal"},
            "predicates": [{
                "role": "core",
                "kind": "spatio_temporal",
                "attribute": "segment_geom",
                "lower": "2020-01-01 00:00:00",
                "upper": "2020-01-01 00:30:00",
                "temporal_width_seconds": 1800.0,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            report = analyze_benchmark(path)
        stats = report["intervals"]["segment_time/spatio_temporal"]
        self.assertEqual(stats["two_sided_context_count"], 1)
        self.assertEqual(stats["bins"]["1_min_to_1_h"]["count"], 1)

    def test_maps_extractor_labels_to_report_groups(self) -> None:
        self.assertEqual(
            interval_report_key("segment_time", "trajectory_temporal"),
            "segment_trajectory_temporal",
        )
        self.assertEqual(
            interval_report_key("segment_time", "ordinary"),
            "segment_ordinary",
        )
        self.assertEqual(interval_report_key("trip_time", "ordinary"), "trip_ordinary")
        self.assertEqual(one_sided_report_key("segment_time_ordinary"), "segment_ordinary")
        self.assertEqual(one_sided_report_key("trip_time_ordinary"), "trip_ordinary")

    def test_extracts_temporal_and_ordinary_pairs_with_ranges_and_inside(self) -> None:
        temporal = SegmentTemporalPredicate(
            "segments:t_s", "segments:t_e", lower=20.0, upper=10.0,
        )
        context = GeneratedTrainingContext(
            tokens=(),
            included_tables=frozenset({"segments", "trips"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={
                # These are the encoded scalar representation of ``temporal``
                # and must not be double-counted as ordinary predicates.
                "segments:t_s": PredicateToken(PredicateOp.LESS_THAN, 10.0),
                "segments:t_e": PredicateToken(PredicateOp.GREATER_EQUAL, 20.0),
                "trips:start_time": PredicateToken.range(100.0, 160.0),
                "trips:end_time": PredicateToken.range(120.0, 200.0),
            },
            trajectory_query=TrajectoryQuerySemantics(temporal_predicates=(temporal,)),
        )

        observations, one_sided, equalities = extract_context_intervals(context)
        self.assertEqual(one_sided, {})
        self.assertEqual(equalities, {})
        self.assertEqual(
            {(item.kind, item.source, item.width_seconds) for item in observations},
            {("segment_time", "trajectory_temporal", -10.0), ("trip_time", "ordinary", 40.0)},
        )
        self.assertEqual(width_bin(-10.0), "inside")
        self.assertEqual(width_bin(40.0), "lt_1_min")

    def test_counts_one_sided_and_equality_time_predicates(self) -> None:
        context = GeneratedTrainingContext(
            tokens=(),
            included_tables=frozenset({"segments", "trips"}),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={
                "segments:t_s": PredicateToken(PredicateOp.LESS_EQUAL, 60.0),
                "trips:start_time": PredicateToken.equal(100.0),
                "trips:end_time": PredicateToken.equal(110.0),
            },
        )
        observations, one_sided, equalities = extract_context_intervals(context)
        self.assertEqual(observations, [])
        self.assertEqual(one_sided, {"segment_time_ordinary": 1})
        self.assertEqual(equalities, {"trips:start_time": 1, "trips:end_time": 1})


if __name__ == "__main__":
    unittest.main()
