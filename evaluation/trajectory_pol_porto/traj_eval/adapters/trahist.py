"""TraHist and TraHist-Temporal (P6) on the v3 multi (distinct-trajectory) workload.

**Base: TraHist** (Jo et al., PACMMOD 2026 / arXiv 2610.22310), run from the
pinned upstream checkout (``YJ960/TraHist`` at ``adapter.trahist_revision``;
the repository carries no licence, so it is imported from the checkout, not
copied). Its construction is used unchanged (``algos/trahist/core.py``):

1. CountGrid: trajectory points on a regular grid of cell size
   ``adapter.unit_size_m``; per cell the number of points ``Cp`` and of
   distinct trajectories ``CT``;
2. LabelMap: density-adaptive partition of the cells into buckets
   (``alpha``, ``beta``, ``kappa``, ``connectivity``);
3. per bucket the mean of ``Cp`` and ``CT`` over its cells (``Sp``, ``ST``)
   and the ratio ``ST / max(Sp, eps)``;
4. query: sample the query rectangle at cell resolution, map the samples
   through the LabelMap, ``estimate = (sum_k n_k Sp_k) * (sum_k n_k r_k / n)``.

The trajectory points are the polyline vertices of every trip: the start of
each segment and the end of its last segment, with their times (float64; the
upstream runner rounds to float32 to save memory). TraHist is spatial only:
queries with a time predicate are ``unsupported`` for the base variant.

**TraHist-Temporal** (extension by us, spec "Temporal extension for CASE+ and
TraHist", ``traj_eval/temporal.py``): the same grid and LabelMap; every bucket
additionally keeps, per equi-depth time bucket, its point count and the
``F``/``L`` statistics of the distinct-trajectory counts of its cells. For a
query interval the buckets ``a..b`` it intersects are selected and

    Sp_k([a, b]) = (points of bucket k in a..b) / cells_k
    ST_k([a, b]) = (sum over cells of k of D_cell([a, b])) / cells_k

replace ``Sp_k`` / ``ST_k`` in TraHist's own estimator. ``ST_k`` is a mean of
*per-cell* distinct counts, exactly as upstream (which averages ``CT`` over
the cells), so the bucket's ``F``/``L`` are the sums of its cells' ``F``/``L``
(not the union over the bucket). With the full time range this reproduces
the base estimate exactly (tested against upstream). A temporal-only query
uses the full spatial extent.

Adaptations, all documented in the README:

- the query rectangle is clipped to the grid's extent before upstream's
  sampling (upstream clips out-of-grid samples onto the border cells and
  would count them repeatedly; v3's one-sided windows reach 1,000 km past
  the data);
- a time predicate on the trip table (``trips.trip_time``) is answered with
  the point times and carries ``approx:trip_time_as_point_time``.

Unsupported: single-table queries (row counts), attribute predicates (decided
2026-10-26: CASE+/TraHist cover space and time only), segment-level attribute
predicates, more than one window or interval. Deterministic: one seed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..temporal import BucketCounts, Occurrences, TemporalDistinct, TimeBuckets
from .base import Estimate, InProcessTrajectoryAdapter
from .space_time import parse_space_time

TRIP_TIME_DIAGNOSTIC = "approx:trip_time_as_point_time"
UNIT_BYTES = 8  # upstream's theoretical memory unit (S_UNIT_BYTES)
_UPSTREAM: Dict[str, Any] = {}


def load_upstream(source: Path, revision: Optional[str]) -> Any:
    """``algos/trahist/core.py`` of the pinned checkout (revision checked)."""
    source = Path(source).resolve()
    key = str(source)
    if key in _UPSTREAM:
        return _UPSTREAM[key]
    if revision:
        try:
            head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"cannot read the TraHist checkout revision at {source}: {error}") from error
        if not head.startswith(revision):
            raise ValueError(f"TraHist checkout {source} is at {head}, expected {revision}")
    path = source / "algos" / "trahist" / "core.py"
    spec = importlib.util.spec_from_file_location("trahist_upstream_core", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    _UPSTREAM[key] = module
    return module


def trajectory_points(data: Mapping[str, Mapping[str, np.ndarray]], trips: Mapping[str, np.ndarray],
                      segment_table: str) -> Dict[str, np.ndarray]:
    """Polyline vertices: every segment start plus each trip's last segment end.

    ``trajectory`` is the dense trip index (snapshot order); ``time_us`` the
    vertex time; ``midpoint_us`` the segment time midpoints (for the
    equi-depth time buckets).
    """
    segments = data[segment_table]
    offsets = np.asarray(trips["offset"], dtype=np.int64)
    counts = np.diff(offsets)
    last = offsets[1:] - 1
    n_trips = counts.size
    x = np.concatenate((np.asarray(segments["sx"], dtype=np.float64), np.asarray(segments["ex"], dtype=np.float64)[last]))
    y = np.concatenate((np.asarray(segments["sy"], dtype=np.float64), np.asarray(segments["ey"], dtype=np.float64)[last]))
    ts, te = np.asarray(segments["ts"], dtype=np.int64), np.asarray(segments["te"], dtype=np.int64)
    time_us = np.concatenate((ts, te[last]))
    trajectory = np.concatenate((np.repeat(np.arange(n_trips, dtype=np.int64), counts), np.arange(n_trips, dtype=np.int64)))
    return {"x": x, "y": y, "time_us": time_us, "trajectory": trajectory,
            "midpoint_us": ts + (te - ts) // 2, "t_min_us": np.int64(ts.min()), "t_max_us": np.int64(te.max()),
            "n_trajectories": np.int64(n_trips)}


def cell_index(x: np.ndarray, y: np.ndarray, mins: np.ndarray, unit: np.ndarray, num_cells: np.ndarray) -> np.ndarray:
    """Flat (C-order) cell of every point, exactly as upstream's CountGrid assigns it."""
    xy = np.column_stack((x, y))
    idx = ((xy - mins) / unit).astype(np.int64)
    idx = np.clip(idx, 0, num_cells - 1)
    return idx[:, 0] * int(num_cells[1]) + idx[:, 1]


class TraHistModel:
    def __init__(self, settings: Any, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any], upstream: Any):
        self.settings = settings
        self.segments = settings.segments
        self.entity = settings.base["entity"]["table"]
        self.upstream = upstream
        self.temporal = bool(meta["temporal"])
        mins = np.asarray(arrays["grid/mins"], dtype=np.float64)
        maxs = np.asarray(arrays["grid/maxs"], dtype=np.float64)
        unit = np.asarray(arrays["grid/unit_size"], dtype=np.float64)
        num_cells = np.asarray(arrays["grid/num_cells"], dtype=np.int64)
        # what upstream's estimator reads from its CountGrid
        self.grid = SimpleNamespace(mins=mins, maxs=maxs, unit_size=unit, num_cells=num_cells)
        self.label_map = np.asarray(arrays["label_map"], dtype=np.int32)
        self.Sp = np.asarray(arrays["Sp"], dtype=np.float64)
        self.ratio = np.asarray(arrays["ratio_per_bucket"], dtype=np.float64)
        self.ratio_eps = float(meta["ratio_eps"])
        if self.temporal:
            self.time = TimeBuckets(np.asarray(arrays["time/boundaries"], dtype=np.int64))
            self.points = BucketCounts.from_arrays(arrays, "P")
            self.distinct = TemporalDistinct.from_arrays(arrays, "T")
            self.cells = np.asarray(arrays["cells_per_bucket"], dtype=np.float64)

    # -- query translation --------------------------------------------------

    def _parse(self, record: Any) -> Tuple[Optional[Estimate], Optional[Tuple[float, ...]], Any, str]:
        """(early result, window, interval, diagnostic); interval: None = no time predicate."""
        query, reason = parse_space_time(self.settings, record, "TraHist")
        if query is None:
            return ("unsupported", None, reason), None, None, ""
        if query.interval is not None and not self.temporal:
            return ("unsupported", None, "time predicate (TraHist is spatial only; see TraHist-Temporal)"), None, None, ""
        return None, query.window, query.interval, TRIP_TIME_DIAGNOSTIC if query.trip_time else ""

    def _query_box(self, window: Optional[Tuple[float, ...]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """The window clipped to the grid's extent (full extent without a window); None: no data there.

        A window thinner than one sample after clipping keeps one sample at its
        clipped start (upstream's sampling yields at least one point for any
        non-empty range).
        """
        grid = self.grid
        if window is None:
            low, high = grid.mins.copy(), grid.maxs.copy()
        else:
            low, high = np.array(window[:2], dtype=np.float64), np.array(window[2:], dtype=np.float64)
            if np.any(high < grid.mins) or np.any(low > grid.maxs):
                return None
            low, high = np.maximum(low, grid.mins), np.minimum(high, grid.maxs)
        high = np.where(high <= low, low + 0.5 * grid.unit_size, high)
        return low, high

    def _sampled_labels(self, box: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Labels of upstream's cell-resolution samples of ``box`` (as in ``get_estimation_traj``)."""
        grid = self.grid
        samples = self.upstream._get_sampled_points(box[0], box[1], grid.unit_size)
        idxs = ((samples - grid.mins) / grid.unit_size).astype(int)
        idxs = np.clip(idxs, 0, grid.num_cells - 1)
        labels = self.label_map[tuple(idxs.T)]
        return labels[labels > 0]

    @staticmethod
    def _combine(labels: np.ndarray, Sp: np.ndarray, ratio: np.ndarray) -> float:
        """Upstream's ``get_estimation_traj`` formula on given per-bucket Sp / ratio."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = labels.shape[0]
        k_idx = unique_labels - 1
        est_points = float(np.sum(counts * Sp[k_idx]))
        avg_ratio = float(np.sum(counts * ratio[k_idx]) / total_samples) if total_samples > 0 else 0.0
        est = est_points * avg_ratio
        return 0.0 if est < 1e-4 else float(est)

    def time_restricted(self, a: int, b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``Sp``, ``ST`` and the ratio of every bucket over time buckets ``a..b``."""
        stats = np.arange(self.Sp.size)
        Sp = self.points.range_sum(stats, a, b) / self.cells
        ST = self.distinct.distinct(stats, a, b) / self.cells
        return Sp, ST, ST / np.maximum(Sp, self.ratio_eps)

    def estimate(self, record: Any) -> Estimate:
        early, window, interval, diagnostic = self._parse(record)
        if early is not None:
            return early
        box = self._query_box(window)
        if box is None:
            return "ok", 0.0, diagnostic  # the window misses the data extent
        if not self.temporal:  # the base estimator: upstream's code on the clipped window
            import pandas as pd

            query = pd.Series({"x_low": box[0][0], "y_low": box[0][1], "x_high": box[1][0], "y_high": box[1][1]})
            return "ok", float(self.upstream.get_estimation_traj(
                query, self.grid, self.label_map, self.Sp, self.ratio)), diagnostic
        labels = self._sampled_labels(box)
        if labels.size == 0:
            return "ok", 0.0, diagnostic
        buckets = (0, self.time.count - 1) if interval is None else self.time.interval(*interval)
        if buckets is None:
            return "ok", 0.0, diagnostic  # the interval misses the data's time range
        Sp, _, ratio = self._restricted_for(np.unique(labels) - 1, *buckets)
        return "ok", self._combine(labels, Sp, ratio), diagnostic

    def _restricted_for(self, k_idx: np.ndarray, a: int, b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Time-restricted Sp, ST, ratio (full-length arrays, filled for ``k_idx`` only)."""
        Sp = np.zeros_like(self.Sp)
        ST = np.zeros_like(self.Sp)
        Sp[k_idx] = self.points.range_sum(k_idx, a, b) / self.cells[k_idx]
        ST[k_idx] = self.distinct.distinct(k_idx, a, b) / self.cells[k_idx]
        ratio = ST / np.maximum(Sp, self.ratio_eps)
        return Sp, ST, ratio


def build_arrays(upstream: Any, points: Mapping[str, np.ndarray], *, unit_size_m: float, alpha: float, beta: float,
                 kappa: int, connectivity: str, agg_mode: str, ratio_eps: float, temporal: bool,
                 time_buckets: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Upstream TraHist on the points, plus the temporal statistics if requested."""
    import pandas as pd

    frame = pd.DataFrame({"x": points["x"], "y": points["y"], "traj_id": points["trajectory"]})
    model = upstream.build_trahist(frame, unit_size=(float(unit_size_m), float(unit_size_m)), alpha=float(alpha),
                                   beta=float(beta), kappa=int(kappa), connectivity=str(connectivity),
                                   agg_mode=str(agg_mode), ratio_eps=float(ratio_eps))
    del frame
    label_map = np.asarray(model.label_map, dtype=np.int32)
    if np.any(label_map <= 0):
        raise ValueError("TraHist left unlabelled cells")
    K = int(model.lm.n_segments)
    arrays: Dict[str, np.ndarray] = {
        "grid/mins": np.asarray(model.cg.mins, dtype=np.float64),
        "grid/maxs": np.asarray(model.cg.maxs, dtype=np.float64),
        "grid/unit_size": np.asarray(model.cg.unit_size, dtype=np.float64),
        "grid/num_cells": np.asarray(model.cg.num_cells, dtype=np.int64),
        "label_map": label_map,
        "Sp": np.asarray(model.Sp, dtype=np.float64),
        "ST": np.asarray(model.ST, dtype=np.float64),
        "ratio_per_bucket": np.asarray(model.ratio_per_bucket, dtype=np.float64),
    }
    n_cells = int(label_map.size)
    meta: Dict[str, Any] = {
        "method": "trahist_temporal" if temporal else "trahist",
        "temporal": bool(temporal), "unit_size_m": float(unit_size_m), "alpha": float(alpha), "beta": float(beta),
        "kappa": int(kappa), "connectivity": str(connectivity), "agg_mode": str(agg_mode),
        "ratio_eps": float(ratio_eps), "grid_cells": n_cells, "grid_shape": [int(v) for v in model.cg.num_cells],
        "buckets": K, "points": int(points["x"].size), "trajectories": int(points["n_trajectories"]),
        "upstream_theoretical_bytes": int((n_cells + 2 * K) * UNIT_BYTES),
    }
    if temporal:
        buckets = TimeBuckets.equi_depth(points["midpoint_us"], int(time_buckets),
                                         int(points["t_min_us"]), int(points["t_max_us"]))
        cells = cell_index(points["x"], points["y"], arrays["grid/mins"], arrays["grid/unit_size"],
                           arrays["grid/num_cells"])
        time_bucket = buckets.assign(points["time_us"])
        owner = label_map.ravel().astype(np.int64) - 1
        occurrences = Occurrences.from_events(cells, points["trajectory"], time_bucket, n_cells,
                                              int(points["n_trajectories"]), buckets.count)
        distinct = TemporalDistinct.build(occurrences, owner_of_unit=owner, n_stats=K)
        del occurrences
        point_counts = BucketCounts.from_pairs(owner[cells], time_bucket, K, buckets.count)
        arrays.update({"time/boundaries": buckets.boundaries, "cells_per_bucket": np.bincount(owner, minlength=K),
                       **point_counts.arrays("P"), **distinct.arrays("T")})
        temporal_units = (buckets.boundaries.size + 2 * point_counts.value.size
                          + 2 * distinct.occupancy.value.size + 3 * distinct.links.count.size)
        meta.update({"time_buckets_requested": int(time_buckets), "time_buckets": buckets.count,
                     "time_bucket_depth": "segments by time midpoint (equi-depth)",
                     "links": distinct.links.total, "link_entries": int(distinct.links.count.size),
                     "temporal_theoretical_bytes": int(temporal_units * UNIT_BYTES)})
        meta["theoretical_bytes"] = meta["upstream_theoretical_bytes"] + meta["temporal_theoretical_bytes"]
    else:
        meta["theoretical_bytes"] = meta["upstream_theoretical_bytes"]
    return arrays, meta


class TraHistAdapter(InProcessTrajectoryAdapter):
    latency_scope = "predicate_translation_and_histogram_estimate"
    timing_role = "in_process_trahist"

    @property
    def upstream(self) -> Any:
        source = self.config.adapter.get("trahist_source")
        if not source:
            raise ValueError("adapter.trahist_source (the pinned TraHist checkout) is required")
        return load_upstream(self._resolve(source), self.config.adapter.get("trahist_revision"))

    def prepare(self):
        metrics = super().prepare()
        import pandas
        import scipy

        upstream = self.upstream  # fails early on a missing checkout or a wrong revision
        metrics.detail.update({
            "trahist_source": str(self._resolve(self.config.adapter["trahist_source"])),
            "trahist_revision": self.config.adapter.get("trahist_revision"),
            "trahist_module": str(getattr(upstream, "__file__", "")),
            "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
        })
        return metrics

    def _flag(self, key: str, default: bool = False) -> bool:
        value = self.config.adapter.get(key, default)
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")

    def _build_model(self, data, trips) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        adapter = self.config.adapter
        points = trajectory_points(data, trips, self.settings.segments.table)
        arrays, meta = build_arrays(
            self.upstream, points,
            unit_size_m=float(adapter.get("unit_size_m", 100.0)), alpha=float(adapter.get("alpha", 0.5)),
            beta=float(adapter.get("beta", 1.0)), kappa=int(adapter.get("kappa", 3)),
            connectivity=str(adapter.get("connectivity", "axis")), agg_mode=str(adapter.get("agg_mode", "mean")),
            ratio_eps=float(adapter.get("ratio_eps", 1.0)), temporal=self._flag("temporal"),
            time_buckets=int(adapter.get("time_buckets", 100)))
        meta["trahist_revision"] = adapter.get("trahist_revision")
        return arrays, meta

    def _load_model(self, arrays, meta) -> TraHistModel:
        return TraHistModel(self.settings, arrays, meta, self.upstream)

    def _estimate(self, model: TraHistModel, record: Any) -> Estimate:
        return model.estimate(record)
