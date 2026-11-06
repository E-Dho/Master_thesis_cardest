"""Segment-MBR overcount statistic (traj_eval/mbr_overcount.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_pol_porto.tests._fixture import (
    V3_CONFIG, benchmark_rows, synthetic_data, synthetic_settings, write_benchmark, write_synthetic_snapshot,
)
from evaluation.trajectory_pol_porto.traj_eval import mbr_overcount as mo


class SegmentClassesTest(unittest.TestCase):
    def test_hand_checked_segments(self) -> None:
        # window [0, 10] x [0, 10]
        segments = {
            "sx": np.array([2.0, -5.0, -5.0, 20.0, 5.0, -5.0, 10.0]),
            "sy": np.array([2.0, 5.0, 8.0, 20.0, 5.0, -5.0, 12.0]),
            "ex": np.array([3.0, 15.0, 8.0, 30.0, 25.0, 15.0, 12.0]),
            "ey": np.array([3.0, 5.0, 21.0, 30.0, 25.0, 15.0, 10.0]),
        }
        # 0 inside, 1 crossing, 2 corner (y = x + 13 misses), 3 far away,
        # 4 one endpoint inside, 5 diagonal crossing, 6 corner touching nothing but box corner (10, 10)? -> box
        #   x [10, 12], y [10, 12] overlaps W only in the point (10, 10); the line from (10, 12) to (12, 10)
        #   passes x + y = 22, so it misses W: a corner segment.
        predicate = {"kind": "spatial", "min_x": 0.0, "min_y": 0.0, "max_x": 10.0, "max_y": 10.0}
        classes = mo.segment_classes(segments, predicate)
        self.assertEqual(classes["index"].tolist(), [0, 1, 2, 4, 5, 6])
        self.assertEqual(classes["line"].tolist(), [True, True, False, True, True, False])
        both_outside = ~classes["start_inside"] & ~classes["end_inside"]
        self.assertEqual(both_outside.tolist(), [False, True, True, False, True, True])

    def test_spatio_temporal_decoupling_is_separated(self) -> None:
        # a segment crossing W during seconds 0..100 of its 1000-s span; the interval covers 500..600
        segments = {
            "sx": np.array([-5.0]), "sy": np.array([5.0]), "ex": np.array([95.0]), "ey": np.array([5.0]),
            "ts": np.array([0], dtype=np.int64), "te": np.array([1_000_000_000], dtype=np.int64),
        }
        predicate = {"kind": "spatio_temporal", "min_x": 0.0, "min_y": 0.0, "max_x": 10.0, "max_y": 10.0,
                     "lower": "1970-01-01 00:08:20", "upper": "1970-01-01 00:10:00"}
        classes = mo.segment_classes(segments, predicate)
        self.assertEqual(classes["index"].tolist(), [0])
        self.assertTrue(classes["line"][0])
        self.assertFalse(classes["exact"][0])


class OvercountRunTest(unittest.TestCase):
    def _run(self, root: Path, benchmark: Path, shards: int = 2):
        snapshot = write_synthetic_snapshot(root / "snap")
        for shard in range(shards):
            self.assertEqual(mo.run_shard(snapshot, V3_CONFIG, benchmark, root / "out", shard, shards,
                                          log=lambda m: None), 0)
        return snapshot, mo.merge(benchmark, root / "out", root / "report", snapshot)

    def test_counts_are_nested_and_reproduce_the_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            _, report = self._run(root, benchmark)
            self.assertTrue(report["passed"], report["mismatches"])
            spatial = [row for row in benchmark_rows() if mo.spatial_predicate(row) is not None]
            self.assertEqual(report["queries_evaluated"], len(spatial))
            self.assertEqual(report["queries_excluded_without_spatial_predicate"], len(benchmark_rows()) - len(spatial))
            items = [json.loads(line) for path in sorted((root / "out").glob("shard_*.jsonl"))
                     for line in path.read_text().splitlines()]
            self.assertEqual(len(items), len(spatial))
            for item in items:
                self.assertLessEqual(item["exact_count"], item["decoupled_count"])
                self.assertLessEqual(item["decoupled_count"], item["mbr_count"])
                self.assertEqual(item["anomaly_endpoint_inside_line_misses"], 0)
                self.assertLessEqual(item["segments_corner"], item["segments_both_endpoints_outside"])
                self.assertEqual(item["segments_mbr"], item["segments_exact"] + item["segments_corner"]
                                 + item["segments_decoupled_only"])
                if item["dimension"] == "spatial":
                    self.assertEqual(item["decoupled_count"], item["exact_count"])
                if item["relation"] == "single":
                    self.assertEqual(item["segments_exact"], item["label"])
                    self.assertEqual(item["segments_mbr"], item["mbr_count"])
            for suffix in (".json", ".md", "_per_query.csv", "_overcounted.csv"):
                self.assertTrue(Path(f"{root / 'report'}{suffix}").exists())
            # the overcounted queries are listed by id, in the JSON and in their own CSV
            overcounted = sorted(item["query_id"] for item in items if item["mbr_count"] > item["exact_count"])
            self.assertGreater(len(overcounted), 0)  # the synthetic fixture has some
            self.assertEqual(sorted(report["overcounted_query_ids"]), overcounted)
            import csv
            with Path(f"{root / 'report'}_overcounted.csv").open(encoding="utf-8") as handle:
                listed = list(csv.DictReader(handle))
            self.assertEqual(sorted(row["query_id"] for row in listed), overcounted)
            ratios = [float(row["ratio_mbr_to_exact"]) for row in listed]
            self.assertEqual(ratios, sorted(ratios, reverse=True))
            for row in listed:
                self.assertEqual(int(row["overcount"]), int(row["overcount_corner"]) + int(row["overcount_decoupling"]))
                self.assertEqual(int(row["overcount"]), int(row["mbr_count"]) - int(row["exact_count"]))

    def test_mbr_count_matches_a_brute_force_box_count(self) -> None:
        data = synthetic_data()
        seg = data["segments"]
        backend = mo._backend_class()(synthetic_settings(), data)
        checked = 0
        for row in benchmark_rows():
            predicate = mo.spatial_predicate(row)
            if predicate is None or row["category"]["relation"] != "single" or len(row["predicates"]) != 1:
                continue
            box = ((np.maximum(seg["sx"], seg["ex"]) >= predicate["min_x"])
                   & (np.minimum(seg["sx"], seg["ex"]) <= predicate["max_x"])
                   & (np.maximum(seg["sy"], seg["ey"]) >= predicate["min_y"])
                   & (np.minimum(seg["sy"], seg["ey"]) <= predicate["max_y"]))
            if predicate["kind"] == "spatio_temporal":
                from query_generation.benchmark_v3 import exact
                from query_generation.benchmark_v3.numpy_backend import _bound_us
                box &= exact.temporal_mask(seg["ts"], seg["te"], _bound_us(predicate.get("lower")),
                                           _bound_us(predicate.get("upper")))
            self.assertEqual(mo.evaluate_query(backend, row)["mbr_count"], int(box.sum()))
            checked += 1
        self.assertGreater(checked, 0)

    def test_a_wrong_label_fails_the_merge(self) -> None:
        rows = [dict(row) for row in benchmark_rows()]
        for row in rows:
            if mo.spatial_predicate(row) is not None:
                row["cardinality"] += 1
                break
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl", rows)
            _, report = self._run(root, benchmark)
            self.assertFalse(report["passed"])
            self.assertEqual(report["mismatch_count"], 1)
            self.assertIn("reference_count", report["mismatches"][0])


if __name__ == "__main__":
    unittest.main()
