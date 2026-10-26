from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.trajectory_pol_porto.traj_eval.workloads import (
    TRAJ_V3_JSONL_FORMAT,
    Domains,
    WorkloadError,
    load_traj_workload,
    normalize_select,
    select_rows,
    workload_statistics,
    write_query_index,
)

from ._fixture import MULTI, SINGLE, V3_CONFIG, benchmark_rows, write_benchmark


class SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_benchmark(Path(self.tmp.name) / "benchmark.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_relations_split_the_file_and_ids_are_positions_in_file_order(self) -> None:
        multi = load_traj_workload(self.path, "pol_multi", {"relation": "multi"})
        single = load_traj_workload(self.path, "pol_single", {"relation": "single"})
        self.assertEqual(len(multi), 8 * MULTI)
        self.assertEqual(len(single), 8 * SINGLE)
        self.assertEqual([r.query_id for r in multi], list(range(len(multi))))
        source_order = [row["query_id"] for row in benchmark_rows() if row["category"]["relation"] == "multi"]
        self.assertEqual([r.source_query_id for r in multi], source_order)
        self.assertEqual({r.label_kind for r in multi}, {"distinct_trips"})
        self.assertEqual({r.label_kind for r in single}, {"row_count"})

    def test_select_values_may_be_lists_and_are_order_independent(self) -> None:
        a = normalize_select({"dimension": ["spatial", "temporal"], "relation": "multi"})
        b = normalize_select({"relation": "multi", "dimension": ("spatial", "temporal")})
        self.assertEqual(a, b)
        rows = select_rows(self.path, a)
        self.assertEqual(len(rows), 4 * MULTI)
        with self.assertRaisesRegex(WorkloadError, "unknown select keys"):
            normalize_select({"table": "trips"})

    def test_truth_comes_from_the_label_of_the_relation(self) -> None:
        for record, row in zip(load_traj_workload(self.path, "w"), benchmark_rows()):
            field = "entity_cardinality" if row["category"]["relation"] == "multi" else "join_cardinality"
            self.assertEqual(record.true_cardinality, row[field])


class ContractTest(unittest.TestCase):
    def _load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark(Path(tmp) / "b.jsonl", rows)
            return load_traj_workload(path, "w")

    def test_rejects_other_semantics_mismatched_labels_and_duplicates(self) -> None:
        rows = [copy.deepcopy(row) for row in benchmark_rows()]
        bad = copy.deepcopy(rows)
        bad[0]["semantics_version"] = "pol_segment_coupled_v2"
        with self.assertRaisesRegex(WorkloadError, "semantics_version"):
            self._load(bad)
        bad = copy.deepcopy(rows)
        multi = next(i for i, r in enumerate(bad) if r["category"]["relation"] == "multi")
        bad[multi]["entity_cardinality"] += 1
        with self.assertRaisesRegex(WorkloadError, "differs from entity_cardinality"):
            self._load(bad)
        bad = copy.deepcopy(rows)
        bad[1]["predicate_hash"] = bad[0]["predicate_hash"]
        with self.assertRaisesRegex(WorkloadError, "duplicate predicate_hash"):
            self._load(bad)
        bad = copy.deepcopy(rows)
        bad[multi]["label_kind"] = "row_count"
        bad[multi]["join_cardinality"] = bad[multi]["cardinality"]
        with self.assertRaisesRegex(WorkloadError, "multi query with label row_count"):
            self._load(bad)


class FeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = write_benchmark(Path(cls.tmp.name) / "benchmark.jsonl")
        cls.domains = Domains.from_benchmark_config(V3_CONFIG)
        cls.records = load_traj_workload(cls.path, "w", None, cls.domains)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_domain_relative_features_are_fractions_within_the_domain(self) -> None:
        spatial = [r for r in self.records if r.dimension in ("spatial", "spatio_temporal")]
        temporal = [r for r in self.records if r.dimension in ("temporal", "spatio_temporal")]
        self.assertTrue(spatial and temporal)
        for record in spatial:
            self.assertGreater(record.features["window_area_fraction"], 0.0)
            self.assertLessEqual(record.features["window_area_fraction"], 1.0)
        for record in temporal:
            self.assertGreaterEqual(record.features["interval_fraction"], 0.0)
            self.assertLessEqual(record.features["interval_fraction"], 1.0)
        for record in self.records:
            if record.dimension == "standard":
                self.assertIsNone(record.features["window_area_fraction"])
                self.assertIsNone(record.features["interval_fraction"])

    def test_unbounded_windows_are_clipped_to_the_domain(self) -> None:
        unbounded = [r for r in self.records if r.dimension == "spatial" and r.interval == "unbounded"]
        self.assertTrue(unbounded)
        for record in unbounded:
            self.assertLessEqual(record.features["window_width_x"],
                                 self.domains.spatial[2] - self.domains.spatial[0] + 1e-6)

    def test_bounded_interval_length_matches_its_bounds(self) -> None:
        from datetime import datetime

        for record in self.records:
            p = next((p for p in record.predicates if p["kind"] == "temporal"), None)
            if p is None or p.get("lower") is None or p.get("upper") is None:
                continue
            expected = (datetime.fromisoformat(p["upper"]) - datetime.fromisoformat(p["lower"])).total_seconds()
            # clipped to the domain, so never longer than the bounds say
            self.assertLessEqual(record.features["interval_seconds"], expected + 1e-6)

    def test_interval_length_ignores_the_machine_time_zone(self) -> None:
        import os
        import time

        row = copy.deepcopy(next(r for r in benchmark_rows()
                                 if r["category"]["key"] == "temporal.range.multi"))
        predicate = next(p for p in row["predicates"] if p["kind"] == "temporal")
        # across the 2020-03-29 DST change in Europe/Berlin
        predicate["lower"], predicate["upper"] = "2020-03-28 12:00:00", "2020-03-30 12:00:00"
        previous = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "Europe/Berlin"
            time.tzset()
            from evaluation.trajectory_pol_porto.traj_eval.workloads import query_features

            with_domains = query_features(row, self.domains)
            without = query_features(row, None)
        finally:
            if previous is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous
            time.tzset()
        self.assertEqual(with_domains["interval_seconds"], 48 * 3600.0)
        self.assertEqual(without["interval_seconds"], 48 * 3600.0)

    def test_statistics_and_query_index(self) -> None:
        stats = workload_statistics(self.records)
        self.assertEqual(stats["query_count"], 8 * SINGLE + 8 * MULTI)
        self.assertEqual(stats["label_kind_counts"], {"distinct_trips": 8 * MULTI, "row_count": 8 * SINGLE})
        self.assertEqual(stats["zero_true_cardinality_count"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "query_index.csv"
            write_query_index(self.records, index)
            rows = list(csv.DictReader(index.open()))
        self.assertEqual(len(rows), len(self.records))
        self.assertEqual(rows[3]["predicate_hash"], self.records[3].predicate_hash)


class FormatTest(unittest.TestCase):
    def test_parse_validate_and_count_selected_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark(Path(tmp) / "b.jsonl")
            resolve = lambda value: Path(tmp) / str(value) if not Path(str(value)).is_absolute() else Path(value)
            value = {"format": "traj_v3_jsonl", "queries_jsonl": "b.jsonl",
                     "select": {"relation": "multi"}, "expected_query_count": 8 * MULTI,
                     "benchmark_config": str(V3_CONFIG)}
            workload = TRAJ_V3_JSONL_FORMAT.parse_config("pol_multi", value, resolve)
            TRAJ_V3_JSONL_FORMAT.validate(workload)
            self.assertEqual(workload.queries_file, path)
            meta = TRAJ_V3_JSONL_FORMAT.metadata(workload, TRAJ_V3_JSONL_FORMAT.load(workload))
            self.assertEqual(meta["select"], {"relation": "multi"})
            self.assertEqual(meta["semantics_versions"], ["pol_exact_st_v3"])
            self.assertEqual(json.loads(TRAJ_V3_JSONL_FORMAT.placeholders(workload)["workload_select_json"]),
                             {"relation": "multi"})
            wrong = TRAJ_V3_JSONL_FORMAT.parse_config("pol_multi", {**value, "expected_query_count": 8000}, resolve)
            with self.assertRaisesRegex(ValueError, "expected 8000 queries, observed 64"):
                TRAJ_V3_JSONL_FORMAT.validate(wrong)
            with self.assertRaisesRegex(ValueError, "queries_jsonl is required"):
                TRAJ_V3_JSONL_FORMAT.parse_config("x", {"format": "traj_v3_jsonl"}, resolve)


if __name__ == "__main__":
    unittest.main()
