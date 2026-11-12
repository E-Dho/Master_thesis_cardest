"""P6 CASE+ and CASE+-Temporal.

- grid traversal = dense sampling of every segment;
- a planarized trajectory counts exactly once in every query region with a
  real face count, and Example 9/10 of the paper (a trajectory outside the
  region whose virtual counts cover it) gives CASE 2 and CASE+ 1;
- every trajectory that is not a tree of cells and crossed edges (a cycle,
  a jump between segments) is planarized too (review of f20c7d7);
- CASE+-Temporal with the full time range equals the base exactly (gate 6),
  and over buckets a..b it equals CASE+ built on every trajectory cut to
  a..b (a cut trajectory is planarized on its own);
- the sparse storage (2026-11-12) answers exactly what the dense histograms
  answer: a dense reference rebuilt from the stored counts and rectangles, with
  the dense query formulas, on POL and Porto data;
- coverage rules and a pipeline run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from itertools import product
from pathlib import Path

import numpy as np

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.runner import run_seed
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snapshot_module
from evaluation.trajectory_pol_porto.traj_eval.adapters import case_plus
from evaluation.trajectory_pol_porto.traj_eval.adapters.case_plus import CaseModel, Grid, build_case, traverse
from evaluation.trajectory_pol_porto.traj_eval.workloads import load_traj_workload

from ._fixture import synthetic_settings, write_benchmark, write_synthetic_snapshot
from .test_p3_adapters import _config

UNIT = 200.0


def _trajectories(paths, times=None):
    """Segments and trip offsets from lists of (x, y) vertices (unit-time segments by default)."""
    sx, sy, ex, ey, ts, te, offsets = [], [], [], [], [], [], [0]
    clock = 0
    for index, path in enumerate(paths):
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            sx.append(x0); sy.append(y0); ex.append(x1); ey.append(y1)
            start = clock if times is None else times[index]
            ts.append(start); te.append(start + 1)
            clock += 1
        offsets.append(len(sx))
    segments = {k: np.asarray(v, dtype=np.float64) for k, v in (("sx", sx), ("sy", sy), ("ex", ex), ("ey", ey))}
    segments.update(ts=np.asarray(ts, dtype=np.int64), te=np.asarray(te, dtype=np.int64))
    return segments, {"offset": np.asarray(offsets, dtype=np.int64)}


def dense(arrays):
    """The dense per-key histograms the sparse storage stands for: real/<kind> and virtual/<kind>."""
    ny, nx = (int(v) for v in np.asarray(arrays["grid/shape"]))
    K = np.asarray(arrays["keys"]).reshape(-1, 4).shape[0]
    shapes = {"cell": (ny, nx), "vedge": (ny, max(nx - 1, 1)), "hedge": (max(ny - 1, 1), nx),
              "vertex": (max(ny - 1, 1), max(nx - 1, 1))}
    out = {}
    for prefix in ("real", "preal"):
        for kind in case_plus.HISTOGRAMS:
            h, w = shapes[kind]
            array = np.zeros((K, h, w), dtype=np.int64)
            element = np.asarray(arrays[f"{prefix}/{kind}/element"]).astype(np.int64)
            np.add.at(array, (np.asarray(arrays[f"{prefix}/{kind}/key"]).astype(np.int64), element // w, element % w),
                      np.asarray(arrays[f"{prefix}/{kind}/count"]).astype(np.int64))
            out[f"{prefix}/{kind}"] = array
    bounds = np.asarray(arrays["rect/bounds"]).astype(np.int64).reshape(-1, 5)
    for kind in case_plus.VIRTUAL:
        array = np.zeros((K,) + shapes[kind], dtype=np.int64)
        for (key, r0, r1, c0, c1), count in zip(bounds, np.asarray(arrays["rect/count"]).astype(np.int64)):
            r1 = r1 - (kind in ("hedge", "vertex"))
            c1 = c1 - (kind in ("vedge", "vertex"))
            if r1 >= r0 and c1 >= c0:
                array[key, r0:r1 + 1, c0:c1 + 1] += count
        if kind in case_plus.HISTOGRAMS:
            array -= out[f"preal/{kind}"]
        assert array.min() >= 0, kind
        out[f"virtual/{kind}"] = array
    return out


def dense_count(arrays, model, rect, a, b, plus):
    """The dense CASE / CASE+ query (the formulas of the dense implementation before 2026-11-12)."""
    hist = dense(arrays)
    keys = model.selected(a, b)
    r0, r1, c0, c1 = rect

    def total(name, r0, r1, c0, c1):
        return 0 if r1 < r0 or c1 < c0 else int(hist[name][keys, r0:r1 + 1, c0:c1 + 1].sum())

    def minimum(name, r0, r1, c0, c1):
        if r1 < r0 or c1 < c0:
            return None
        return 0 if keys.size == 0 else int(hist[name][keys, r0:r1 + 1, c0:c1 + 1].sum(axis=0).min())

    faces = total("real/cell", r0, r1, c0, c1)
    if faces == 0:
        return 0.0
    edges = total("real/vedge", r0, r1, c0, c1 - 1) + total("real/hedge", r0, r1 - 1, c0, c1)
    vfaces = total("virtual/cell", r0, r1, c0, c1)
    vedges = total("virtual/vedge", r0, r1, c0, c1 - 1) + total("virtual/hedge", r0, r1 - 1, c0, c1)
    line = r1 == r0 or c1 == c0
    n = (faces + vfaces) - (edges + vedges) + (0 if line else total("virtual/vertex", r0, r1 - 1, c0, c1 - 1))
    if plus:
        minima = [minimum("virtual/cell", r0, r1, c0, c1)]
        edge = [m for m in (minimum("virtual/vedge", r0, r1, c0, c1 - 1), minimum("virtual/hedge", r0, r1 - 1, c0, c1))
                if m is not None]
        if edge:
            minima.append(min(edge))
        if not line:
            minima.append(minimum("virtual/vertex", r0, r1 - 1, c0, c1 - 1))
        n -= min(m for m in minima if m is not None)
    return float(max(n, 0))


def _model(segments, trips, unit=1.0, plus=True, temporal=False, buckets=1, extent=None):
    arrays, meta = build_case(segments, trips, unit_size_m=unit, temporal=temporal, time_buckets=buckets,
                              extent=extent)
    meta["plus"] = plus
    return CaseModel(synthetic_settings(), arrays, meta), arrays, meta


class TraversalTest(unittest.TestCase):
    def test_cells_equal_dense_sampling(self) -> None:
        rng = np.random.default_rng(4)
        grid = Grid(np.array([0.0, 0.0]), 1.0, (12, 15))
        sx, sy = rng.uniform(0, 15, 300), rng.uniform(0, 12, 300)
        ex, ey = np.clip(sx + rng.normal(0, 4, 300), 0, 14.999), np.clip(sy + rng.normal(0, 4, 300), 0, 11.999)
        result = traverse(grid, sx, sy, ex, ey)
        for seg in range(300):
            t = np.linspace(0.0, 1.0, 200_001)
            cols = grid.column(sx[seg] + t * (ex[seg] - sx[seg]))
            rows = grid.row(sy[seg] + t * (ey[seg] - sy[seg]))
            sampled = set((rows * grid.nx + cols).tolist())
            got = set(result["cell"][result["cell_seg"] == seg].tolist())
            self.assertEqual(got, sampled, seg)
            moves = np.flatnonzero((np.diff(rows) != 0) | (np.diff(cols) != 0))
            vertical = sum(int(cols[m + 1] != cols[m]) for m in moves)
            horizontal = sum(int(rows[m + 1] != rows[m]) for m in moves)
            self.assertEqual(int(np.sum(result["vedge_seg"] == seg)), vertical, seg)
            self.assertEqual(int(np.sum(result["hedge_seg"] == seg)), horizontal, seg)


class CaseHistogramTest(unittest.TestCase):
    def test_planarized_trajectory_counts_once_in_every_region(self) -> None:
        # a U-shape: leaves and re-enters rows/columns, so Stage 2 planarizes it
        path = [(0.5, 0.5), (0.5, 4.5), (4.5, 4.5), (4.5, 0.5)]
        model, arrays, meta = _model(*_trajectories([path]), extent=(0, 0, 5, 5))
        self.assertEqual(meta["planarized_trajectories"], 1)
        real = dense(arrays)["real/cell"][0]
        for r0, r1, c0, c1 in product(range(5), range(5), range(5), range(5)):
            if r1 < r0 or c1 < c0:
                continue
            expected = 1.0 if real[r0:r1 + 1, c0:c1 + 1].sum() > 0 else 0.0
            self.assertEqual(model.count((r0, r1, c0, c1), 0, 0), expected, (r0, r1, c0, c1))

    def test_cycles_and_jumps_are_planarized(self) -> None:
        # around a 2x2 block: cells and crossed edges form a cycle (F - E = 0 without planarization)
        loop = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.6)]
        model, _, meta = _model(*_trajectories([loop]), extent=(0, 0, 3, 3))
        self.assertEqual(meta["planarized_trajectories"], 1)
        self.assertEqual(meta["planarized_as_non_tree"], 1)
        for r0, r1, c0, c1 in product(range(3), range(3), range(3), range(3)):
            if r1 >= r0 and c1 >= c0 and r0 <= 1 and c0 <= 1:
                self.assertEqual(model.count((r0, r1, c0, c1), 0, 0), 1.0, (r0, r1, c0, c1))
        # two segments that do not join: the trajectory jumps from one cell to a distant one
        segments, trips = _trajectories([[(0.5, 0.5), (0.6, 0.6)]])
        jump = {k: np.concatenate((v, v)) for k, v in segments.items()}
        jump["sx"][1], jump["ex"][1], jump["sy"][1], jump["ey"][1] = 2.5, 2.6, 0.5, 0.6
        model, _, meta = _model(jump, {"offset": np.array([0, 2])}, extent=(0, 0, 3, 3))
        self.assertEqual(meta["planarized_trajectories"], 1)
        self.assertEqual(model.count((0, 0, 0, 2), 0, 0), 1.0)

    def test_straight_trajectory_is_not_planarized(self) -> None:
        model, arrays, meta = _model(*_trajectories([[(0.5, 0.5), (3.5, 0.5)]]), extent=(0, 0, 5, 5))
        self.assertEqual(meta["planarized_trajectories"], 0)
        self.assertEqual(int(dense(arrays)["virtual/cell"].sum()), 0)
        self.assertEqual(arrays["rect/count"].size, 0)
        self.assertEqual(model.count((0, 0, 0, 3), 0, 0), 1.0)
        self.assertEqual(model.count((0, 0, 1, 2), 0, 0), 1.0)

    def test_paper_example_virtual_object_outside_the_region(self) -> None:
        # T2 runs around the region rows/cols 2..4 (its MBR covers it, it never enters);
        # T1 lies inside it. CASE counts T2's virtual object, CASE+ removes it.
        t2 = [(1.5, 1.5), (5.5, 1.5), (5.5, 5.5), (1.5, 5.5)]
        t1 = [(3.2, 3.2), (3.8, 3.6)]
        segments, trips = _trajectories([t2, t1])
        region = (2, 4, 2, 4)
        case, _, meta = _model(segments, trips, plus=False, extent=(0, 0, 7, 7))
        plus, _, _ = _model(segments, trips, plus=True, extent=(0, 0, 7, 7))
        self.assertEqual(meta["planarized_trajectories"], 1)
        self.assertEqual(case.count(region, 0, 0), 2.0)
        self.assertEqual(plus.count(region, 0, 0), 1.0)
        # no real face count in the region: 0 (Algorithm 2, lines 1-3)
        empty = _model(*_trajectories([t2]), plus=False, extent=(0, 0, 7, 7))[0]
        self.assertEqual(empty.count(region, 0, 0), 0.0)


class CaseTemporalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")
        data = snapshot_module.load_snapshot(cls.root / "snapshot")
        cls.segments = data["segments"]
        cls.trips = snapshot_module.load_trip_index(cls.root / "snapshot")
        cls.base, cls.base_arrays, cls.base_meta = _model(cls.segments, cls.trips, unit=UNIT)
        cls.temporal, cls.temp_arrays, cls.temp_meta = _model(cls.segments, cls.trips, unit=UNIT, temporal=True,
                                                              buckets=12)
        cls.records = load_traj_workload(cls.root / "benchmark.jsonl", "w")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_full_range_equals_the_base(self) -> None:
        self.assertGreater(self.base_meta["planarized_trajectories"], 0)
        full = (0, self.temporal.time.count - 1)
        rng = np.random.default_rng(8)
        grid = self.base.grid
        for _ in range(300):
            r0, r1 = sorted(rng.integers(0, grid.ny, 2))
            c0, c1 = sorted(rng.integers(0, grid.nx, 2))
            rect = (int(r0), int(r1), int(c0), int(c1))
            self.assertEqual(self.temporal.count(rect, *full), self.base.count(rect, 0, 0), rect)
        spatial = [r for r in self.records if self.base.estimate(r)[0] == "ok"]
        self.assertGreater(len(spatial), 5)
        for record in spatial:
            self.assertEqual(self.temporal.estimate(record), self.base.estimate(record), record.query_id)

    def test_restricted_counts_are_case_on_the_restricted_trajectories(self) -> None:
        """Gate: CASE+-Temporal over buckets a..b == CASE+ built on every trajectory cut to a..b."""
        segments, offsets = self.segments, np.asarray(self.trips["offset"])
        sx, sy, ex, ey = (np.asarray(segments[k]) for k in ("sx", "sy", "ex", "ey"))
        extent = (float(min(sx.min(), ex.min())), float(min(sy.min(), ey.min())),
                  float(max(sx.max(), ex.max())), float(max(sy.max(), ey.max())))
        first = self.temporal.time.assign(np.asarray(segments["ts"]))
        last = self.temporal.time.assign(np.asarray(segments["te"]))
        rng = np.random.default_rng(9)
        grid = self.temporal.grid
        checked = 0
        for _ in range(12):
            a, b = (int(v) for v in sorted(rng.integers(0, self.temporal.time.count, 2)))
            keep = (first <= b) & (last >= a)
            if not keep.any():
                continue
            seg_traj = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))[keep]
            cut_offsets = np.concatenate(([0], np.cumsum(np.bincount(seg_traj)[np.unique(seg_traj)])))
            cut = {k: np.asarray(segments[k])[keep] for k in ("sx", "sy", "ex", "ey", "ts", "te")}
            for plus in (False, True):
                reference = _model(cut, {"offset": cut_offsets}, unit=UNIT, plus=plus, extent=extent)[0]
                model = CaseModel(synthetic_settings(), self.temp_arrays, {**self.temp_meta, "plus": plus})
                self.assertEqual((reference.grid.ny, reference.grid.nx), (grid.ny, grid.nx))
                for _ in range(15):
                    r0, r1 = (int(v) for v in sorted(rng.integers(0, grid.ny, 2)))
                    c0, c1 = (int(v) for v in sorted(rng.integers(0, grid.nx, 2)))
                    rect = (r0, r1, c0, c1)
                    self.assertEqual(model.count(rect, a, b), reference.count(rect, 0, 0), (rect, a, b, plus))
                    checked += 1
        self.assertGreater(checked, 100)

    def test_cut_trajectory_keeps_its_planarization(self) -> None:
        # review case: a U-shape (time 0-30), then a stationary tail at time 3000; four short
        # trajectories elsewhere at time 2000-2010 put the bucket boundary at 2005, so the U lies in
        # bucket 0 and the tail in bucket 1. A bottom-row query over bucket 0 must count the U once.
        u = [(0.5, 0.5), (0.5, 3.5), (3.5, 3.5), (3.5, 0.5), (3.5, 0.5)]
        paths = [u] + [[(10.5, 10.5), (10.6, 10.6)]] * 4
        segments, trips = _trajectories(paths)
        segments["ts"] = np.array([0, 10, 20, 3000, 2000, 2000, 2000, 2000], dtype=np.int64)
        segments["te"] = segments["ts"] + 10
        model, _, meta = _model(segments, trips, temporal=True, buckets=2, extent=(0, 0, 12, 12))
        self.assertEqual(model.time.boundaries.tolist()[1], 2005)
        self.assertEqual(meta["longest_span_buckets"], 2)
        bottom = (0, 0, 0, 3)
        self.assertEqual(model.count(bottom, 0, 0), 1.0)  # the cut U is planarized on its own
        self.assertEqual(model.count(bottom, 0, 1), 1.0)
        self.assertEqual(model.count(bottom, 1, 1), 1.0)  # only the tail, in cell (0, 3)
        self.assertEqual(model.count((0, 3, 0, 3), 0, 0), 1.0)

    def test_coverage_rules(self) -> None:
        for record in self.records:
            kinds = {(p["table"], p["kind"]) for p in record.predicates}
            base, temporal = self.base.estimate(record), self.temporal.estimate(record)
            if record.relation == "single":
                self.assertEqual(temporal[0], "unsupported")
                continue
            space_time_only = kinds <= {("segments", "spatial"), ("segments", "spatio_temporal"),
                                        ("segments", "temporal"), ("trips", "temporal")}
            has_time = bool(kinds & {("segments", "spatio_temporal"), ("segments", "temporal"), ("trips", "temporal")})
            self.assertEqual(temporal[0] == "ok", space_time_only, record.query_id)
            self.assertEqual(base[0] == "ok", space_time_only and not has_time, record.query_id)
            if temporal[0] == "ok":
                self.assertGreaterEqual(temporal[1], 0.0)
                self.assertEqual(temporal[2] == case_plus.TRIP_TIME_DIAGNOSTIC, ("trips", "temporal") in kinds)

    def test_pipeline_runs_both_variants(self) -> None:
        for variant, temporal in (("base", "false"), ("temporal", "true")):
            path = _config(self.root, "case_plus", "traj_case_plus", variant, unit_size_m=UNIT, temporal=temporal,
                           time_buckets=12, plus="true")
            run = run_seed(load_experiment_config(path), 0, run_id=variant)
            summary = json.loads((run / "summary.json").read_text())
            self.assertGreater(summary["workloads"]["pol_v3_multi"]["extension"]["coverage"]["scored_count"], 0)
            self.assertEqual(summary["workloads"]["pol_v3_single"]["extension"]["coverage"]["scored_count"], 0)
            meta = json.loads((run / "model" / "model_meta.json").read_text())
            self.assertGreater(meta["theoretical_bytes"], 0)


class SparseStorageTest(unittest.TestCase):
    """The sparse histograms answer exactly what the dense ones answer (and are far smaller)."""

    def _check(self, segments, trips, settings, unit, temporal, buckets, extent, rng, queries):
        arrays, meta = build_case(segments, trips, unit_size_m=unit, temporal=temporal, time_buckets=buckets,
                                  extent=extent)
        self.assertEqual(meta["storage"].split()[0], "sparse")
        ny, nx = (int(v) for v in arrays["grid/shape"])
        cells = np.asarray(arrays["real/cell/element"]).astype(np.int64)
        rows, cols = cells // nx, cells % nx
        checked = nonzero = 0
        for plus in (False, True):
            model = CaseModel(settings, arrays, {**meta, "plus": plus})
            T = model.time.count
            for index in range(queries):
                if index % 5 == 0:  # the whole grid, the largest one-sided windows
                    rect = (0, ny - 1, 0, nx - 1)
                else:  # around the data
                    r0, r1 = sorted(int(v) for v in rng.integers(rows.min() - 2, rows.max() + 3, 2))
                    c0, c1 = sorted(int(v) for v in rng.integers(cols.min() - 2, cols.max() + 3, 2))
                    rect = (max(r0, 0), min(r1, ny - 1), max(c0, 0), min(c1, nx - 1))
                a, b = sorted(int(v) for v in rng.integers(0, T, 2))
                value = model.count(rect, a, b)
                self.assertEqual(value, dense_count(arrays, model, rect, a, b, plus), (rect, a, b, plus))
                checked += 1
                nonzero += value > 0
        self.assertGreater(nonzero, checked // 4)
        stored = sum(np.asarray(v).nbytes for v in arrays.values())
        dense_bytes = meta["keys"] * 7 * ny * nx * 4  # the int32 dense histograms
        return stored, dense_bytes

    def test_pol_base_and_temporal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_synthetic_snapshot(Path(tmp) / "snapshot")
            data, trips = snapshot_module.load_snapshot(root), snapshot_module.load_trip_index(root)
        rng = np.random.default_rng(21)
        for unit, temporal, buckets in ((UNIT, False, 1), (UNIT, True, 12), (60.0, True, 7)):
            stored, dense_bytes = self._check(data["segments"], trips, synthetic_settings(), unit, temporal, buckets,
                                              None, rng, 60)
            if temporal:
                self.assertLess(stored * 10, dense_bytes)

    def test_porto_window_domain(self) -> None:
        from evaluation.trajectory_pol_porto.traj_eval.adapters.base import grid_extent

        from ._porto_fixture import porto_settings, write_porto_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            root = write_porto_snapshot(Path(tmp) / "snapshot")
            data, trips = snapshot_module.load_snapshot(root), snapshot_module.load_trip_index(root)
        settings = porto_settings()
        extent = grid_extent(settings, "window_domain")
        rng = np.random.default_rng(22)
        # GPS jumps planarize trajectories whose rectangles reach the border cells
        stored, dense_bytes = self._check(data["segments"], trips, settings, 2000.0, True, 5, extent, rng, 60)
        self.assertLess(stored * 20, dense_bytes)

    def test_size_guard_on_found_elements(self) -> None:
        from unittest import mock

        from evaluation.trajectory_pol_porto.traj_eval.temporal import SettingTooLarge

        segments, trips = _trajectories([[(0.5, 0.5), (3.5, 0.5), (3.5, 3.5)]])
        with mock.patch.object(case_plus, "MAX_ELEMENTS", 3), self.assertRaises(SettingTooLarge):
            build_case(segments, trips, unit_size_m=1.0, temporal=False, time_buckets=1, extent=(0, 0, 5, 5))


class ShippedCaseConfigTest(unittest.TestCase):
    def test_configs(self) -> None:
        from evaluation.eval_core.config import load_simple_yaml

        configs = Path(__file__).resolve().parents[1] / "configs" / "pol_v3"
        base = load_simple_yaml(configs / "case_plus.yaml")
        temporal = load_simple_yaml(configs / "case_plus_temporal.yaml")
        for raw in (base, temporal):
            self.assertEqual(raw["adapter"]["type"], "traj_case_plus")
            self.assertIs(raw["adapter"]["plus"], True)
            self.assertEqual(raw["experiment"]["seeds"], [0])
        self.assertIs(base["adapter"]["temporal"], False)
        self.assertIs(temporal["adapter"]["temporal"], True)
        best = load_simple_yaml(configs / "case_plus_temporal_best.yaml")
        # tuned on validation (select_tuned, 2026-11-06); budget = the own model's inference-only size
        self.assertEqual(base["adapter"]["unit_size_m"], 25)
        self.assertEqual((temporal["adapter"]["unit_size_m"], temporal["adapter"]["time_buckets"]), (200, 3200))
        self.assertEqual((best["adapter"]["unit_size_m"], best["adapter"]["time_buckets"]), (100, 3200))
        for raw in (base, temporal, best):
            self.assertEqual(raw["adapter"]["budget_bytes"], 872606460)
        # the best pick is the budget config with another grid and its own variant id
        tuned = ("unit_size_m", "time_buckets")
        self.assertEqual({k: v for k, v in best["adapter"].items() if k not in tuned},
                         {k: v for k, v in temporal["adapter"].items() if k not in tuned})
        self.assertEqual(best["experiment"]["variant_id"], "temporal_best")
        self.assertEqual({k: v for k, v in best["experiment"].items() if k not in ("variant_id", "display_name")},
                         {k: v for k, v in temporal["experiment"].items() if k not in ("variant_id", "display_name")})


if __name__ == "__main__":
    unittest.main()
