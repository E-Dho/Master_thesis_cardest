"""P6 TraHist and TraHist-Temporal on the synthetic POL fixture.

Needs the pinned upstream checkout (``YJ960/TraHist`` at 870e8ce): set
``TRAJ_EVAL_TRAHIST_SOURCE`` to it. Gates:

- the temporal statistics with the full time range equal upstream's ``Sp`` /
  ``ST`` exactly, so TraHist-Temporal reproduces TraHist on every spatial-only
  query (spec gate 6);
- the time-restricted per-bucket statistics equal a brute-force recount from
  the points (spec gate 7 at the histogram level);
- coverage rules (single-table, attribute add-ons, base without time).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.runner import run_seed
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snapshot_module
from evaluation.trajectory_pol_porto.traj_eval.adapters import trahist
from evaluation.trajectory_pol_porto.traj_eval.workloads import load_traj_workload

from ._fixture import synthetic_data, synthetic_settings, write_benchmark, write_synthetic_snapshot
from .test_p3_adapters import _config

SOURCE = os.environ.get("TRAJ_EVAL_TRAHIST_SOURCE")
REVISION = "870e8ce6db0e6363eeca80dce193623034c858ab"
UNIT = 100.0  # about 48 x 52 cells over the synthetic extent


@unittest.skipUnless(SOURCE, "set TRAJ_EVAL_TRAHIST_SOURCE to the TraHist checkout (870e8ce)")
class TraHistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")
        cls.upstream = trahist.load_upstream(Path(SOURCE), REVISION)
        data = snapshot_module.load_snapshot(cls.root / "snapshot")
        trips = snapshot_module.load_trip_index(cls.root / "snapshot")
        cls.points = trahist.trajectory_points(data, trips, "segments")
        common = dict(unit_size_m=UNIT, alpha=0.5, beta=1.0, kappa=3, connectivity="axis", agg_mode="mean",
                      ratio_eps=1.0, time_buckets=12)
        cls.base_arrays, cls.base_meta = trahist.build_arrays(cls.upstream, cls.points, temporal=False, **common)
        cls.temp_arrays, cls.temp_meta = trahist.build_arrays(cls.upstream, cls.points, temporal=True, **common)
        settings = synthetic_settings()
        cls.base = trahist.TraHistModel(settings, cls.base_arrays, cls.base_meta, cls.upstream)
        cls.temporal = trahist.TraHistModel(settings, cls.temp_arrays, cls.temp_meta, cls.upstream)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _records(self, relation="multi"):
        return [r for r in load_traj_workload(self.root / "benchmark.jsonl", "w") if r.relation == relation]

    def test_points_are_the_polyline_vertices(self) -> None:
        data = synthetic_data()
        segments, trips = data["segments"], data["trips"]
        self.assertEqual(self.points["x"].size, segments["sx"].size + trips["trip_id"].size)
        self.assertEqual(int(self.points["n_trajectories"]), trips["trip_id"].size)
        self.assertGreater(self.base_meta["buckets"], 1)

    def test_full_range_reproduces_upstream_statistics_exactly(self) -> None:
        Sp, ST, ratio = self.temporal.time_restricted(0, self.temporal.time.count - 1)
        np.testing.assert_array_equal(Sp, self.base_arrays["Sp"])
        np.testing.assert_array_equal(ST, self.base_arrays["ST"])
        np.testing.assert_array_equal(ratio, self.base_arrays["ratio_per_bucket"])
        np.testing.assert_array_equal(self.temp_arrays["label_map"], self.base_arrays["label_map"])

    def test_spatial_queries_equal_the_upstream_estimator(self) -> None:
        spatial = [r for r in self._records() if self.base.estimate(r)[0] == "ok"]
        self.assertGreater(len(spatial), 5)
        for record in spatial:
            self.assertEqual(self.temporal.estimate(record), self.base.estimate(record), record.query_id)
        # random windows inside and across the extent
        rng = np.random.default_rng(1)
        grid = self.base.grid
        for _ in range(200):
            low = grid.mins + rng.random(2) * (grid.maxs - grid.mins)
            size = np.exp(rng.uniform(np.log(10.0), np.log(8000.0), 2))
            window = (*low, *(low + size))
            box = self.base._query_box(window)
            labels = self.temporal._sampled_labels(box)
            expected = self.upstream.get_estimation_traj(
                __import__("pandas").Series({"x_low": box[0][0], "y_low": box[0][1],
                                             "x_high": box[1][0], "y_high": box[1][1]}),
                grid, self.base.label_map, self.base.Sp, self.base.ratio)
            got = self.temporal._combine(labels, *self.temporal._restricted_for(
                np.unique(labels) - 1, 0, self.temporal.time.count - 1)[::2])
            self.assertEqual(got, expected)

    def test_time_restricted_statistics_match_a_brute_force_recount(self) -> None:
        model, points = self.temporal, self.points
        cells = trahist.cell_index(points["x"], points["y"], model.grid.mins, model.grid.unit_size,
                                   model.grid.num_cells)
        owner = model.label_map.ravel() - 1
        bucket_of_point = model.time.assign(points["time_us"])
        cells_per_bucket = np.bincount(owner, minlength=model.Sp.size)
        rng = np.random.default_rng(2)
        for _ in range(25):
            a, b = sorted(rng.integers(0, model.time.count, 2))
            Sp, ST, _ = model.time_restricted(int(a), int(b))
            inside = (bucket_of_point >= a) & (bucket_of_point <= b)
            want_points = np.bincount(owner[cells[inside]], minlength=Sp.size) / cells_per_bucket
            pairs = np.unique(np.column_stack((cells[inside], points["trajectory"][inside])), axis=0)
            want_distinct = np.bincount(owner[pairs[:, 0]], minlength=Sp.size) / cells_per_bucket if pairs.size else 0
            np.testing.assert_allclose(Sp, want_points, rtol=0, atol=1e-12)
            np.testing.assert_allclose(ST, want_distinct, rtol=0, atol=1e-12)

    def test_coverage_rules(self) -> None:
        multi = self._records()
        for record in self._records("single"):
            self.assertEqual(self.temporal.estimate(record)[0], "unsupported")
        statuses = {}
        for record in multi:
            kinds = {(p["table"], p["kind"]) for p in record.predicates}
            base, temporal = self.base.estimate(record), self.temporal.estimate(record)
            statuses.setdefault(record.category, set()).add(temporal[0])
            space_time_only = kinds <= {("segments", "spatial"), ("segments", "spatio_temporal"),
                                        ("segments", "temporal"), ("trips", "temporal")}
            self.assertEqual(temporal[0] == "ok", space_time_only, (record.query_id, kinds))
            has_time = bool(kinds & {("segments", "spatio_temporal"), ("segments", "temporal"), ("trips", "temporal")})
            self.assertEqual(base[0] == "ok", space_time_only and not has_time, record.query_id)
            if temporal[0] == "ok":
                self.assertGreaterEqual(temporal[1], 0.0)
                self.assertEqual(temporal[2] == trahist.TRIP_TIME_DIAGNOSTIC, ("trips", "temporal") in kinds)
        self.assertTrue(all(s == {"unsupported"} for c, s in statuses.items() if c.startswith("standard")))

    def test_interval_outside_the_data_is_zero(self) -> None:
        record = next(r for r in self._records() if r.category.startswith("spatio_temporal.range")
                      and self.temporal.estimate(r)[0] == "ok")
        import dataclasses

        shifted = dict(record.predicates[0])
        shifted.update(lower="2030-01-01 00:00:00", upper="2030-01-02 00:00:00")
        far = dataclasses.replace(record, predicates=(shifted,))
        self.assertEqual(self.temporal.estimate(far), ("ok", 0.0, ""))

    def test_pipeline_runs_both_variants(self) -> None:
        for variant, temporal in (("base", "false"), ("temporal", "true")):
            path = _config(self.root, "trahist", "traj_trahist", variant, trahist_source=SOURCE,
                           trahist_revision=REVISION, unit_size_m=UNIT, temporal=temporal, time_buckets=12)
            config = load_experiment_config(path)
            run = run_seed(config, 0, run_id=variant)
            summary = __import__("json").loads((run / "summary.json").read_text())
            coverage = summary["workloads"]["pol_v3_multi"]["extension"]["coverage"]
            self.assertGreater(coverage["scored_count"], 0, variant)
            single = summary["workloads"]["pol_v3_single"]["extension"]["coverage"]
            self.assertEqual(single["scored_count"], 0)

    def test_tuning_sweep_on_a_validation_workload(self) -> None:
        import json

        from evaluation.trajectory_pol_porto.traj_eval import tune

        path = _config(self.root, "trahist", "traj_trahist", "sweep", trahist_source=SOURCE,
                       trahist_revision=REVISION, unit_size_m=UNIT, temporal="true", time_buckets=12)
        output = self.root / "tuning"
        rows = tune.sweep(path, self.root / "benchmark.jsonl", tune.parse_grid(["unit_size_m=50,100", "time_buckets=6"]),
                          None, output, log=lambda message: None)
        self.assertEqual([json.loads(row["parameters"])["unit_size_m"] for row in rows], [50, 100])
        for row in rows:
            self.assertIn("_tuning", row["run_directory"])  # never mixed with the test runs
            self.assertGreater(row["serialized_bytes"], 0)
            self.assertGreater(row["theoretical_bytes"], 0)
            self.assertIsNotNone(row["q_p50"])
        self.assertTrue((output / "tuning.csv").exists())
        resolved = json.loads((Path(rows[0]["run_directory"]) / "resolved_config.json").read_text())
        self.assertEqual(list(resolved["raw"]["workloads"]) if "raw" in resolved else list(resolved["workloads"]),
                         ["validation_multi"])



class ShippedTraHistConfigTest(unittest.TestCase):
    def test_configs_pin_upstream_and_differ_only_in_the_extension(self) -> None:
        from evaluation.eval_core.config import load_simple_yaml

        configs = Path(__file__).resolve().parents[1] / "configs" / "pol_v3"
        base = load_simple_yaml(configs / "trahist.yaml")
        temporal = load_simple_yaml(configs / "trahist_temporal.yaml")
        for raw in (base, temporal):
            self.assertEqual(raw["extends"], "_base.yaml")
            self.assertEqual(raw["adapter"]["type"], "traj_trahist")
            self.assertEqual(raw["adapter"]["trahist_revision"], REVISION)
            self.assertEqual(raw["experiment"]["seeds"], [0])  # deterministic
            self.assertEqual(raw["experiment"]["protocol"], "adapted")
        self.assertIs(base["adapter"]["temporal"], False)
        self.assertIs(temporal["adapter"]["temporal"], True)
        shared = {k: v for k, v in base["adapter"].items() if k != "temporal"}
        self.assertEqual({k: temporal["adapter"][k] for k in shared}, shared)
        self.assertEqual(temporal["adapter"]["time_buckets"], 100)  # spec start; tuned on validation


if __name__ == "__main__":
    unittest.main()
