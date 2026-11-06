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
from evaluation.trajectory_pol_porto.traj_eval.snapshot_gate import BindingError, _sha256

QUIET = lambda message: None


def checksummed_snapshot(root: Path) -> Path:
    """The fixture snapshot with real per-array checksums in its manifest, as the export writes them."""

    snapshot = write_synthetic_snapshot(root)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    manifest["files"] = {str(path.relative_to(snapshot)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                         for path in sorted(snapshot.rglob("*.npy"))}
    (snapshot / "manifest.json").write_text(json.dumps(manifest))
    return snapshot


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
        # allow_dirty: the test checkout may hold uncommitted edits; the refusal is tested separately
        snapshot = checksummed_snapshot(root / "snap")
        self.assertTrue(mo.verify(snapshot, root / "out", log=QUIET)["passed"])
        for shard in range(shards):
            self.assertEqual(mo.run_shard(snapshot, V3_CONFIG, benchmark, root / "out", shard, shards,
                                          log=QUIET, allow_dirty=True), 0)
        return snapshot, mo.merge(benchmark, root / "out", root / "report", snapshot)

    def test_counts_are_nested_and_reproduce_the_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            _, report = self._run(root, benchmark)
            self.assertTrue(report["passed"], (report["mismatches"], report["invariant_violations"]))
            self.assertEqual(report["invariant_violation_count"], 0)
            self.assertEqual(report["binding"]["source_sha256"], mo.source_sha256())
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
                self.assertLessEqual(item["segments_corner"] + item["segments_crossing"],
                                     item["segments_both_endpoints_outside"])
                self.assertEqual(item["segments_mbr"], item["segments_exact"] + item["segments_corner"]
                                 + item["segments_decoupled_only"])
                self.assertEqual(mo.invariant_violations(item), [])
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


    def test_a_violated_invariant_fails_the_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            snapshot, report = self._run(root, benchmark)
            self.assertTrue(report["passed"])
            shard = sorted((root / "out").glob("shard_*.jsonl"))[0]
            lines = [json.loads(line) for line in shard.read_text().splitlines()]
            lines[0]["mbr_count"] = lines[0]["exact_count"] - 1  # breaks exact <= decoupled <= mbr
            lines[1]["segments_corner"] += 1                      # breaks the segment partition
            shard.write_text("".join(json.dumps(line) + "\n" for line in lines))
            report = mo.merge(benchmark, root / "out", root / "report", snapshot)
            self.assertFalse(report["passed"])
            self.assertEqual(report["mismatch_count"], 0)
            flagged = {v["query_id"]: v["problems"] for v in report["invariant_violations"]}
            self.assertIn("nesting exact <= decoupled <= mbr", flagged[lines[0]["query_id"]])
            self.assertTrue(any("partition" in p for p in flagged[lines[1]["query_id"]]))

    def test_merge_rechecks_the_label_instead_of_trusting_the_match_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            snapshot, report = self._run(root, benchmark)
            self.assertTrue(report["passed"])
            shard = sorted((root / "out").glob("shard_*.jsonl"))[0]
            lines = [json.loads(line) for line in shard.read_text().splitlines()]
            # a stored label that disagrees with the benchmark, with match left True
            lines[0]["label"] += 1
            # exact_count and label agree with each other but not with the benchmark
            lines[1]["label"] += 1
            lines[1]["exact_count"] += 1
            lines[1]["decoupled_count"] += 1
            lines[1]["mbr_count"] += 1
            shard.write_text("".join(json.dumps(line) + "\n" for line in lines))
            report = mo.merge(benchmark, root / "out", root / "report", snapshot)
            self.assertFalse(report["passed"])
            self.assertEqual({m["query_id"] for m in report["mismatches"]}, {lines[0]["query_id"], lines[1]["query_id"]})

    def test_invariants_catch_an_endpoint_anomaly(self) -> None:
        item = {"exact_count": 3, "decoupled_count": 3, "mbr_count": 4, "dimension": "spatial", "relation": "multi",
                "anomaly_endpoint_inside_line_misses": 1, "segments_mbr": 5, "segments_exact": 3,
                "segments_corner": 1, "segments_decoupled_only": 0, "segments_crossing": 0,
                "segments_both_endpoints_outside": 1}
        self.assertEqual(mo.invariant_violations(item), ["line with an endpoint inside the window misses it"])


class ProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.benchmark = write_benchmark(self.root / "benchmark.jsonl")
        self.snapshot = checksummed_snapshot(self.root / "snap")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, out: Path, shard: int = 0, shards: int = 2, **kwargs) -> int:
        kwargs.setdefault("allow_dirty", True)
        return mo.run_shard(self.snapshot, V3_CONFIG, self.benchmark, out, shard, shards, log=QUIET, **kwargs)

    def test_run_needs_a_passed_verification(self) -> None:
        with self.assertRaises(mo.ProvenanceError):
            self._run(self.root / "out")

    def test_verification_catches_a_modified_array(self) -> None:
        path = self.snapshot / "tables" / "segments" / "sx.npy"
        array = np.load(path)
        array[0] += 1.0  # a corner-only change could keep every label; the checksum still differs
        np.save(path, array)
        record = mo.verify(self.snapshot, self.root / "out", log=QUIET)
        self.assertFalse(record["passed"])
        with self.assertRaises(mo.ProvenanceError):
            self._run(self.root / "out")

    def test_an_array_changed_after_verification_is_refused(self) -> None:
        self.assertTrue(mo.verify(self.snapshot, self.root / "out", log=QUIET)["passed"])
        self.assertEqual(self._run(self.root / "out", 0), 0)
        path = self.snapshot / "tables" / "segments" / "ex.npy"
        np.save(path, np.concatenate([np.load(path), [0.0]]))  # other size
        with self.assertRaises(mo.ProvenanceError):
            self._run(self.root / "out", 1)
        with self.assertRaises(mo.ProvenanceError):
            mo.merge(self.benchmark, self.root / "out", self.root / "report", self.snapshot)

    def test_an_array_not_covered_by_the_manifest_fails_verification(self) -> None:
        manifest = json.loads((self.snapshot / "manifest.json").read_text())
        del manifest["files"]["tables/segments/sy.npy"]
        (self.snapshot / "manifest.json").write_text(json.dumps(manifest))
        record = mo.verify(self.snapshot, self.root / "out", log=QUIET)
        self.assertFalse(record["passed"])
        self.assertTrue(any("sy.npy" in problem for problem in record["problems"]))

    def test_a_missing_array_is_recorded_as_a_failed_verification(self) -> None:
        (self.snapshot / "tables" / "segments" / "ey.npy").unlink()
        record = mo.verify(self.snapshot, self.root / "out", log=QUIET)
        self.assertFalse(record["passed"])
        self.assertTrue(any("ey.npy" in problem for problem in record["problems"]))
        stored = json.loads((self.root / "out" / mo.VERIFICATION_FILE).read_text())
        self.assertFalse(stored["passed"])
        with self.assertRaises(mo.ProvenanceError):
            self._run(self.root / "out")

    def test_an_array_deleted_after_verification_is_refused(self) -> None:
        self.assertTrue(mo.verify(self.snapshot, self.root / "out", log=QUIET)["passed"])
        (self.snapshot / "tables" / "segments" / "ey.npy").unlink()
        with self.assertRaises(mo.ProvenanceError):
            mo.require_verified(self.snapshot, self.root / "out")

    def test_job_scripts_read_the_same_version_as_the_module(self) -> None:
        import re
        import subprocess

        script = Path(mo.__file__).resolve().parents[1] / "slurm" / "submit_mbr_overcount.sh"
        pattern = re.search(r"sed -n '([^']+)'", script.read_text()).group(1)
        version = subprocess.run(["sed", "-n", pattern, mo.__file__], capture_output=True, text=True,
                                 check=True).stdout.strip()
        self.assertEqual(version, str(mo.STATISTIC_VERSION))
        self.assertEqual(mo.main(["version"]), 0)

    def test_uncommitted_source_changes_are_refused(self) -> None:
        from unittest import mock

        self.assertTrue(mo.verify(self.snapshot, self.root / "out", log=QUIET)["passed"])
        for state in ({"code_revision": "abc", "source_files_modified": True},
                      {"code_revision": None, "source_files_modified": None}):
            with mock.patch.object(mo, "code_state", return_value=state):
                with self.assertRaises(mo.ProvenanceError):
                    self._run(self.root / "out", allow_dirty=False)
        with mock.patch.object(mo, "code_state", return_value={"code_revision": "abc", "source_files_modified": False}):
            self.assertEqual(self._run(self.root / "out", allow_dirty=False), 0)

    def test_shards_of_different_source_states_cannot_be_combined(self) -> None:
        from unittest import mock

        out = self.root / "out"
        self.assertTrue(mo.verify(self.snapshot, out, log=QUIET)["passed"])
        self.assertEqual(self._run(out, 0), 0)
        with mock.patch.object(mo, "source_sha256", return_value="0" * 64):  # another dirty patch, same HEAD
            with self.assertRaises(BindingError):
                self._run(out, 1)

    def test_source_hash_follows_the_file_contents(self) -> None:
        from unittest import mock

        real = mo._sha256
        first = mo.source_sha256()
        with mock.patch.object(mo, "_sha256", side_effect=lambda path: "x" if path.name == "exact.py" else real(path)):
            self.assertNotEqual(mo.source_sha256(), first)


if __name__ == "__main__":
    unittest.main()
