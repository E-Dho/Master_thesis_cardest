"""P7: true matching-segment counts (traj_eval/segment_truth.py) and the error split."""

from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_pol_porto.tests._fixture import (
    V3_CONFIG, benchmark_rows, synthetic_data, synthetic_settings, write_benchmark,
)
from evaluation.trajectory_pol_porto.tests.test_mbr_overcount import checksummed_snapshot
from evaluation.trajectory_pol_porto.traj_eval import mbr_overcount as mo
from evaluation.trajectory_pol_porto.traj_eval import own_model_error_split as split
from evaluation.trajectory_pol_porto.traj_eval import segment_truth as st

QUIET = lambda message: None  # noqa: E731


class SegmentTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.benchmark = write_benchmark(root / "benchmark.jsonl")
        cls.snapshot = checksummed_snapshot(root / "snap")
        out = root / "out"
        assert mo.verify(cls.snapshot, out, log=QUIET)["passed"]
        for shard in range(3):  # allow_dirty: the test checkout may hold uncommitted edits
            assert st.run_shard(cls.snapshot, V3_CONFIG, cls.benchmark, out, shard, 3, log=QUIET,
                                allow_dirty=True) == 0
        cls.report = st.merge(cls.benchmark, out, root / "truth", cls.snapshot)
        with (root / "truth_per_query.csv").open(encoding="utf-8", newline="") as handle:
            cls.items = {row["query_id"]: row for row in csv.DictReader(handle)}
        cls.root = root

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_merge_passes_and_covers_every_multi_query_with_segments(self) -> None:
        self.assertTrue(self.report["passed"], {k: self.report[k] for k in ("missing", "label_mismatches",
                                                                              "invariant_violations")})
        expected = [row for row in benchmark_rows() if st.selected(row)]
        self.assertEqual(self.report["queries_evaluated"], len(expected))
        self.assertGreater(self.report["queries_with_model_overcount"], 0)

    def test_counts_against_independent_computations(self) -> None:
        backend = mo._backend_class()(synthetic_settings(), synthetic_data())
        checked_rows = checked_mbr = 0
        for row in benchmark_rows():
            if not st.selected(row):
                continue
            item = self.items[row["query_id"]]
            self.assertEqual(int(item["distinct_exact"]), row["cardinality"])
            if set(p["table"] for p in row["predicates"]) <= {"segments"}:
                # no trip or agent predicate: the matching segments are a single-table count
                single = copy.deepcopy(row)
                single["tables"], single["category"] = ["segments"], {**row["category"], "relation": "single"}
                self.assertEqual(int(item["matching_segments_exact"]), backend.count(single), row["query_id"])
                checked_rows += 1
            predicate = mo.spatial_predicate(row)
            if predicate is not None:
                self.assertEqual(int(item["distinct_model"]), mo.evaluate_query(backend, row)["mbr_count"])
                checked_mbr += 1
            else:
                self.assertEqual(item["matching_segments_model"], item["matching_segments_exact"])
        self.assertGreater(checked_rows, 5)
        self.assertGreater(checked_mbr, 10)

    def test_merge_refuses_a_changed_label(self) -> None:
        rows = [dict(row) for row in benchmark_rows()]
        target = next(row for row in rows if st.selected(row))
        target["cardinality"] += 1
        changed = write_benchmark(self.root / "changed.jsonl", rows)
        with self.assertRaises(Exception):  # bound to the original benchmark
            st.merge(changed, self.root / "out", self.root / "changed_truth", self.snapshot)


class ErrorSplitTest(unittest.TestCase):
    def test_terms_add_up_and_isolate_each_factor(self) -> None:
        truth = {"query_id": "q", "matching_segments_model": "100", "distinct_model": "20", "distinct_exact": "10"}
        component = {"route": "distinct", "status": "ok", "zero_support": "False", "estimate_kind": "dedup",
                     "estimated_cardinality": "40.0", "matching_segment_estimate": "200.0", "traj_dedup_factor": "0.2"}
        terms = split.split_query(component, truth)
        self.assertAlmostEqual(terms["segment"], math.log10(2))  # M_hat / M
        self.assertAlmostEqual(terms["dedup"], 0.0)  # a_hat == a = 0.2
        self.assertAlmostEqual(terms["semantics"], math.log10(2))  # D / D_exact
        self.assertAlmostEqual(terms["total"], math.log10(40 / 10))
        self.assertIsNone(split.split_query({**component, "zero_support": "True"}, truth))
        self.assertIsNone(split.split_query({**component, "route": "join"}, truth))

    def test_the_ablation_applies_factor_one(self) -> None:
        # own_model_no_dedup reports M_hat; the stored head output must not be credited to it
        truth = {"query_id": "q", "matching_segments_model": "100", "distinct_model": "20", "distinct_exact": "10"}
        component = {"route": "distinct", "status": "ok", "zero_support": "False",
                     "estimate_kind": "matching_segments", "estimated_cardinality": "200.0",
                     "matching_segment_estimate": "200.0", "traj_dedup_factor": "0.2"}
        terms = split.split_query(component, truth)
        self.assertAlmostEqual(terms["dedup"], math.log10(1 / 0.2))  # a_hat = 1 against a = 0.2
        self.assertAlmostEqual(terms["total"], math.log10(200 / 10))
        with self.assertRaises(ValueError):  # reported estimate inconsistent with its kind
            split.split_query({**component, "estimated_cardinality": "40.0"}, truth)
        with self.assertRaises(ValueError):  # files without the kind are refused
            split.split_query({k: v for k, v in component.items() if k != "estimate_kind"}, truth)

    def test_runs_are_joined_by_workload_position(self) -> None:
        rows = list(benchmark_rows())
        multi = [row for row in rows if row["category"]["relation"] == "multi"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = write_benchmark(root / "benchmark.jsonl")
            truth_path = root / "truth.csv"
            with truth_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=st.CSV_FIELDS)
                writer.writeheader()
                for row in multi:
                    if st.selected(row):
                        writer.writerow({"query_id": row["query_id"], "dimension": row["category"]["dimension"],
                                         "matching_segments_model": 4 * row["cardinality"],
                                         "distinct_model": row["cardinality"], "distinct_exact": row["cardinality"]})
            run = root / "run"
            (run / "exchange").mkdir(parents=True)
            with (run / "exchange" / "pol_v3_multi_predictions_components.csv").open("w", encoding="utf-8",
                                                                                      newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("query_id", "route", "status", "zero_support",
                                                            "estimate_kind", "estimated_cardinality",
                                                            "matching_segment_estimate", "traj_dedup_factor"))
                writer.writeheader()
                for position, row in enumerate(multi):
                    writer.writerow({"query_id": position, "route": "distinct" if st.selected(row) else "join",
                                     "status": "ok", "zero_support": "False", "estimate_kind": "dedup",
                                     "estimated_cardinality": 2 * row["cardinality"],
                                     "matching_segment_estimate": 8 * row["cardinality"], "traj_dedup_factor": 0.25})
            report = split.split_runs(benchmark, truth_path, [run])
            # an ablation run cannot be pooled with it
            ablation = root / "ablation"
            (ablation / "exchange").mkdir(parents=True)
            text = (run / "exchange" / "pol_v3_multi_predictions_components.csv").read_text()
            rows_out = list(csv.DictReader(text.splitlines()))
            for item in rows_out:
                item["estimate_kind"] = "matching_segments"
                item["estimated_cardinality"] = item["matching_segment_estimate"]
            with (ablation / "exchange" / "pol_v3_multi_predictions_components.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
                writer.writeheader()
                writer.writerows(rows_out)
            ablation_report = split.split_runs(benchmark, truth_path, [ablation])
            with self.assertRaises(ValueError):
                split.split_runs(benchmark, truth_path, [run, ablation])
        self.assertEqual(ablation_report["estimate_kind"], "matching_segments")
        self.assertAlmostEqual(ablation_report["pooled"]["all"]["dedup"]["median_abs_log10"], math.log10(4))
        pooled = report["pooled"]["all"]
        self.assertEqual(pooled["queries"], sum(1 for row in multi if st.selected(row)))
        self.assertAlmostEqual(pooled["segment"]["median_abs_log10"], math.log10(2))
        self.assertAlmostEqual(pooled["dedup"]["median_abs_log10"], 0.0)
        self.assertAlmostEqual(pooled["semantics"]["median_abs_log10"], 0.0)
        self.assertEqual(pooled["dominant_share"]["segment"], 1.0)


if __name__ == "__main__":
    unittest.main()
