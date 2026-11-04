"""CASE+ and CASE+-Temporal (P6) on the v3 multi (distinct-trajectory) workload.

**Base: CASE / CASE+** (Fanaeepour, Kulik, Tanin, Rubinstein, "The CASE
histogram", GeoInformatica 19(4), 2015; no public code, re-implemented from
Algorithms 1-3 and Fig. 12). On a regular grid (cell size ``unit_size_m``
over the data extent):

- *Construction* (Algorithm 1), per trajectory: the cells its segments
  intersect and the cell edges they cross, each counted once per trajectory
  (a segment through a grid vertex is perturbed to cross the vertical line
  first). If the trajectory has the *distinct-counting problem* (Stage 2) it
  is planarized to the minimum bounding rectangle of its cells: its visited
  cells and crossed edges are real counts ``F``/``E``; the MBR's unvisited
  cells, its uncrossed interior edges and all its interior vertices are
  virtual counts ``F'``/``E'``/``V'`` (Fig. 12), so the MBR is one convex
  planar body. Otherwise only ``F``/``E`` are counted.
- *Query* (Algorithm 2) on the cells the query rectangle intersects (Remark
  2): ``N = 0`` without real face counts; otherwise ``N = (F + F') - (E + E')
  + V'`` over the region's cells, its interior edges and interior vertices
  (no vertices for a one-row or one-column region).
- *CASE+* (Algorithm 3): subtract ``N' = min(min F', min E', min V')`` over
  the region (``min F'`` and ``min E'`` for one row or column; ``min F'`` for
  one cell): the virtual objects that fully cover the region. ``T = N - N'``.

Stage 2 reading (documented): the paper's rule flags a trajectory when two of
its cells share a row or column, are "not neighbours" and their "connection
edge" is not crossed. We read it as: two visited cells of one row (column)
that are not joined by a straight run of visited cells whose shared edges the
trajectory crossed. For adjacent cells this is exactly the edge condition;
for cells further apart it is the condition under which a one-row (one-column)
region containing both sees them disconnected.

**CASE+-Temporal** (extension by us, spec "Temporal extension for CASE+ and
TraHist", ``traj_eval/temporal.py``): every real count gets per-time-bucket
occupancy (a segment occupies the equi-depth buckets its time span
intersects) and the ``F``/``L`` statistics, so the time-restricted real
counts are exact distinct counts per cell/edge over bucket-aligned intervals.
The virtual counts of a planarized trajectory belong to its convex body as a
whole and are kept per trajectory time span ``[i, j]`` (first and last
bucket): they apply when the query buckets ``a..b`` contain the whole span
(``a <= i`` and ``j <= b``, the same dominance test as a link). Then all of
the trajectory's real counts are present as well, and the restricted counts
are exactly CASE's planar body; when the interval cuts a trajectory, only
its real counts inside the interval remain (Euler counting without
planarization). With the full time range the counts equal the base exactly
(gate). The query and CASE+ formulas are unchanged.

Adaptations: the query rectangle selects every cell it intersects and is
clipped to the grid; ``trips.trip_time`` predicates use the segment times
(``approx:trip_time_as_segment_time``). Unsupported: single-table queries,
attribute and segment-attribute predicates; the base variant also time
predicates. Deterministic, one seed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from ..temporal import TimeBuckets
from .base import Estimate, InProcessTrajectoryAdapter
from .space_time import parse_space_time

TRIP_TIME_DIAGNOSTIC = "approx:trip_time_as_segment_time"
UNIT_BYTES = 8
KEY_LIMIT = 1 << 62


# ---------------------------------------------------------------------------
# Grid and segment traversal
# ---------------------------------------------------------------------------


class Grid:
    """``ny x nx`` square cells of side ``unit`` from ``origin`` (x, y); row = y index, column = x index."""

    def __init__(self, origin: np.ndarray, unit: float, shape: Tuple[int, int]):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.unit = float(unit)
        self.ny, self.nx = int(shape[0]), int(shape[1])

    @classmethod
    def covering(cls, xmin: float, ymin: float, xmax: float, ymax: float, unit: float) -> "Grid":
        nx = max(1, int(math.ceil((xmax - xmin) / unit)))
        ny = max(1, int(math.ceil((ymax - ymin) / unit)))
        return cls(np.array([xmin, ymin]), unit, (ny, nx))

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        x0, y0 = self.origin
        return x0, y0, x0 + self.nx * self.unit, y0 + self.ny * self.unit

    def column(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((np.asarray(x, dtype=np.float64) - self.origin[0]) / self.unit), 0, self.nx - 1).astype(np.int64)

    def row(self, y: np.ndarray) -> np.ndarray:
        return np.clip(np.floor((np.asarray(y, dtype=np.float64) - self.origin[1]) / self.unit), 0, self.ny - 1).astype(np.int64)

    def cell_range(self, window: Optional[Tuple[float, ...]]) -> Optional[Tuple[int, int, int, int]]:
        """Rows r0..r1 and columns c0..c1 of the cells a closed window intersects; None if it misses the grid."""
        if window is None:
            return 0, self.ny - 1, 0, self.nx - 1
        x0, y0, x1, y1 = self.extent
        if window[2] < x0 or window[0] > x1 or window[3] < y0 or window[1] > y1:
            return None
        c0, c1 = (int(v) for v in self.column(np.array([window[0], window[2]])))
        r0, r1 = (int(v) for v in self.row(np.array([window[1], window[3]])))
        return r0, r1, c0, c1


def traverse(grid: Grid, sx: np.ndarray, sy: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> Dict[str, np.ndarray]:
    """Cells each segment intersects and the cell edges it crosses (exact grid traversal).

    Returns ``cell_seg``/``cell`` (flat ``row * nx + col``; the start cell and
    every cell entered) and ``vedge_seg``/``vedge`` (vertical edges, flat
    ``row * (nx-1) + min col``) and ``hedge_seg``/``hedge`` (horizontal,
    ``min row * nx + col``). Crossings at the same parameter (a grid vertex)
    are ordered x first: the perturbation of Algorithm 1.
    """
    sx, sy, ex, ey = (np.asarray(v, dtype=np.float64) for v in (sx, sy, ex, ey))
    n = sx.size
    c0, c1 = grid.column(sx), grid.column(ex)
    r0, r1 = grid.row(sy), grid.row(ey)
    seg_all = np.arange(n, dtype=np.int64)

    def crossings(start, end, a0, a1, origin):
        """Grid lines crossed between cell index ``start`` and ``end``: owner, parameter t, step."""
        count = np.abs(end - start)
        step = np.sign(end - start)
        owner = np.repeat(seg_all, count)
        offset = np.arange(int(count.sum()), dtype=np.int64) - np.repeat(np.cumsum(count) - count, count)
        lines = np.where(step[owner] > 0, start[owner] + 1 + offset, start[owner] - offset)
        delta = a1[owner] - a0[owner]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (origin + lines * grid.unit - a0[owner]) / delta
        return owner, np.where(np.isfinite(t), t, 0.0), step[owner]

    xo, xt, xs = crossings(c0, c1, sx, ex, grid.origin[0])
    yo, yt, ys = crossings(r0, r1, sy, ey, grid.origin[1])
    owner = np.concatenate((xo, yo))
    t = np.concatenate((xt, yt))
    axis = np.concatenate((np.zeros(xo.size, dtype=np.int8), np.ones(yo.size, dtype=np.int8)))
    step = np.concatenate((xs, ys))
    order = np.lexsort((axis, t, owner))  # by segment, then parameter, x before y on ties
    owner, axis, step = owner[order], axis[order], step[order]
    dx = np.where(axis == 0, step, 0)
    dy = np.where(axis == 1, step, 0)
    first = np.cumsum(np.bincount(owner, minlength=n)) - np.bincount(owner, minlength=n)

    def within(delta):  # inclusive cumulative steps inside each segment
        total = np.cumsum(delta)
        exclusive = total - delta
        return total - exclusive[first[owner]] if delta.size else delta

    col_after = c0[owner] + within(dx)
    row_after = r0[owner] + within(dy)
    col_before, row_before = col_after - dx, row_after - dy
    is_x = axis == 0
    nx = grid.nx
    return {
        "cell_seg": np.concatenate((seg_all, owner)),
        "cell": np.concatenate((r0 * nx + c0, row_after * nx + col_after)),
        "vedge_seg": owner[is_x],
        "vedge": row_before[is_x] * max(nx - 1, 1) + np.minimum(col_before, col_after)[is_x],
        "hedge_seg": owner[~is_x],
        "hedge": np.minimum(row_before, row_after)[~is_x] * nx + col_before[~is_x],
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _unique_keys(*parts: Tuple[np.ndarray, int]) -> np.ndarray:
    """Unique combined keys of integer columns ``(values, size)`` (mixed radix)."""
    size = 1
    for _, radix in parts:
        size *= max(int(radix), 1)
    if size >= KEY_LIMIT:
        raise OverflowError("key space too large; use fewer cells or time buckets")
    key = np.zeros(parts[0][0].size, dtype=np.int64)
    for values, radix in parts:
        key = key * int(radix) + np.asarray(values, dtype=np.int64)
    return np.unique(key)


def _split(keys: np.ndarray, *radices: int) -> Tuple[np.ndarray, ...]:
    out = []
    for radix in reversed(radices):
        out.append(keys % radix)
        keys = keys // radix
    out.append(keys)
    return tuple(reversed(out))


def detect_problem(grid: Grid, n_traj: int, traj_cell: np.ndarray, traj_vedge: np.ndarray,
                   traj_hedge: np.ndarray) -> np.ndarray:
    """Stage 2 per trajectory (see the module docstring for the reading of the rule).

    ``traj_cell`` etc. are sorted unique keys ``traj * n_elements + element``.
    """
    ny, nx = grid.ny, grid.nx
    n_cells, n_v, n_h = ny * nx, ny * max(nx - 1, 1), max(ny - 1, 1) * nx
    problem = np.zeros(n_traj, dtype=bool)
    traj, cell = traj_cell // n_cells, traj_cell % n_cells
    row, col = cell // nx, cell % nx
    # rows: keys are sorted by (traj, row, col)
    same = (traj[1:] == traj[:-1]) & (row[1:] == row[:-1])
    adjacent = col[1:] - col[:-1] == 1
    vkey = traj[:-1] * n_v + row[:-1] * max(nx - 1, 1) + col[:-1]
    crossed = _member(vkey, traj_vedge) & adjacent
    problem[traj[1:][same & ~crossed]] = True
    # columns: sort by (traj, col, row)
    order = np.lexsort((row, col, traj))
    t2, r2, c2 = traj[order], row[order], col[order]
    same = (t2[1:] == t2[:-1]) & (c2[1:] == c2[:-1])
    adjacent = r2[1:] - r2[:-1] == 1
    hkey = t2[:-1] * n_h + r2[:-1] * nx + c2[:-1]
    crossed = _member(hkey, traj_hedge) & adjacent
    problem[t2[1:][same & ~crossed]] = True
    return problem


def _member(keys: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    if sorted_keys.size == 0:
        return np.zeros(keys.size, dtype=bool)
    position = np.clip(np.searchsorted(sorted_keys, keys), 0, sorted_keys.size - 1)
    return sorted_keys[position] == keys


def _rect_add(shape: Tuple[int, int, int], pair: np.ndarray, r0, r1, c0, c1) -> np.ndarray:
    """Per pair, the number of rectangles [r0..r1] x [c0..c1] covering each element (empty rectangles skipped)."""
    P, ny, nx = shape
    keep = (r1 >= r0) & (c1 >= c0)
    pair, r0, r1, c0, c1 = pair[keep], r0[keep], r1[keep], c0[keep], c1[keep]
    diff = np.zeros((P, ny + 1, nx + 1), dtype=np.int64)
    np.add.at(diff, (pair, r0, c0), 1)
    np.add.at(diff, (pair, r0, c1 + 1), -1)
    np.add.at(diff, (pair, r1 + 1, c0), -1)
    np.add.at(diff, (pair, r1 + 1, c1 + 1), 1)
    return np.cumsum(np.cumsum(diff, axis=1), axis=2)[:, :ny, :nx]


def build_case(segments: Mapping[str, np.ndarray], trips: Mapping[str, np.ndarray], *, unit_size_m: float,
               temporal: bool, time_buckets: int, chunk_segments: int = 4_000_000,
               extent: Optional[Tuple[float, float, float, float]] = None
               ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """All CASE histograms (real per time bucket, links, virtual per trajectory span).

    The grid covers ``extent`` (xmin, ymin, xmax, ymax), by default the data
    extent; points outside it fall into the border cells.
    """
    sx, sy, ex, ey = (np.asarray(segments[k], dtype=np.float64) for k in ("sx", "sy", "ex", "ey"))
    ts, te = np.asarray(segments["ts"], dtype=np.int64), np.asarray(segments["te"], dtype=np.int64)
    offsets = np.asarray(trips["offset"], dtype=np.int64)
    n_traj = int(offsets.size - 1)
    seg_traj = np.repeat(np.arange(n_traj, dtype=np.int64), np.diff(offsets))
    if extent is None:
        extent = (float(min(sx.min(), ex.min())), float(min(sy.min(), ey.min())),
                  float(max(sx.max(), ex.max())), float(max(sy.max(), ey.max())))
    grid = Grid.covering(*(float(v) for v in extent), float(unit_size_m))
    ny, nx = grid.ny, grid.nx
    sizes = {"cell": ny * nx, "vedge": ny * max(nx - 1, 1), "hedge": max(ny - 1, 1) * nx}
    if temporal:
        buckets = TimeBuckets.equi_depth(ts + (te - ts) // 2, int(time_buckets), int(ts.min()), int(te.max()))
    else:
        buckets = TimeBuckets(np.array([int(ts.min()), int(te.max()) + 1], dtype=np.int64))
    B = buckets.count
    seg_first, seg_last = buckets.assign(ts).astype(np.int64), buckets.assign(te).astype(np.int64)

    # real occurrences (trajectory, element, bucket), deduplicated, chunk by chunk
    occurrences: Dict[str, list] = {kind: [] for kind in sizes}
    for start in range(0, sx.size, chunk_segments):
        stop = min(start + chunk_segments, sx.size)
        cells = traverse(grid, sx[start:stop], sy[start:stop], ex[start:stop], ey[start:stop])
        for kind in sizes:
            seg = cells[f"{kind}_seg"] + start
            element = cells[kind]
            first, last = seg_first[seg], seg_last[seg]
            span = last - first + 1  # a segment occupies every bucket its time span intersects
            if np.any(span > 1):
                repeat = np.repeat(np.arange(seg.size), span)
                bucket = first[repeat] + (np.arange(repeat.size) - np.repeat(np.cumsum(span) - span, span))
                seg, element = seg[repeat], element[repeat]
            else:
                bucket = first
            occurrences[kind].append(_unique_keys((seg_traj[seg], n_traj), (element, sizes[kind]), (bucket, B)))
    real: Dict[str, Tuple[np.ndarray, ...]] = {}
    for kind, parts in occurrences.items():
        keys = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
        real[kind] = _split(keys, sizes[kind], B)  # traj, element, bucket (sorted)
    del occurrences

    # whole-trajectory element sets, problem detection, MBRs, spans
    whole = {kind: np.unique(t * sizes[kind] + e) for kind, (t, e, _) in real.items()}
    problem = detect_problem(grid, n_traj, whole["cell"], whole["vedge"], whole["hedge"])
    traj_of_cell, cell = whole["cell"] // sizes["cell"], whole["cell"] % sizes["cell"]
    row, col = cell // nx, cell % nx
    big = np.iinfo(np.int64).max
    r_min = np.full(n_traj, big); r_max = np.full(n_traj, -1); c_min = np.full(n_traj, big); c_max = np.full(n_traj, -1)
    np.minimum.at(r_min, traj_of_cell, row); np.maximum.at(r_max, traj_of_cell, row)
    np.minimum.at(c_min, traj_of_cell, col); np.maximum.at(c_max, traj_of_cell, col)
    span_i = np.full(n_traj, B, dtype=np.int64); span_j = np.full(n_traj, -1, dtype=np.int64)
    np.minimum.at(span_i, seg_traj, seg_first); np.maximum.at(span_j, seg_traj, seg_last)

    arrays: Dict[str, np.ndarray] = {
        "grid/origin": grid.origin, "grid/unit": np.array([grid.unit]), "grid/shape": np.array([ny, nx]),
        "time/boundaries": buckets.boundaries,
    }
    # real counts per bucket (dense) and links
    shapes = {"cell": (ny, nx), "vedge": (ny, max(nx - 1, 1)), "hedge": (max(ny - 1, 1), nx)}
    link_total = 0
    for kind, (t, element, bucket) in real.items():
        counts = np.bincount(bucket * sizes[kind] + element, minlength=B * sizes[kind])
        arrays[f"real/{kind}"] = counts.reshape((B,) + shapes[kind]).astype(np.int32)
        same = (t[1:] == t[:-1]) & (element[1:] == element[:-1])
        keys, multiplicity = np.unique((element[:-1][same] * B + bucket[:-1][same]) * B + bucket[1:][same],
                                       return_counts=True)
        e, i, j = _split(keys, B, B)
        arrays.update({f"links/{kind}/row": (e // shapes[kind][1]).astype(np.int32),
                       f"links/{kind}/col": (e % shapes[kind][1]).astype(np.int32),
                       f"links/{kind}/i": i.astype(np.int32), f"links/{kind}/j": j.astype(np.int32),
                       f"links/{kind}/count": multiplicity.astype(np.int32)})
        link_total += int(multiplicity.sum())

    # virtual counts of planarized trajectories, per distinct trajectory span (i, j)
    planar = np.flatnonzero(problem)
    span_keys, pair_of = np.unique(span_i[planar] * B + span_j[planar], return_inverse=True)
    P = span_keys.size
    arrays["virtual/pairs"] = np.column_stack((span_keys // B, span_keys % B)).astype(np.int32).reshape(P, 2)
    rp0, rp1, cp0, cp1 = r_min[planar], r_max[planar], c_min[planar], c_max[planar]
    if P * ny * nx * 4 > 4_000_000_000:
        raise MemoryError(f"{P} trajectory spans x {ny * nx} cells is too large; use a coarser grid or fewer buckets")
    cover = {
        "cell": _rect_add((P, ny, nx), pair_of, rp0, rp1, cp0, cp1),
        "vedge": _rect_add((P,) + shapes["vedge"], pair_of, rp0, rp1, cp0, cp1 - 1),
        "hedge": _rect_add((P,) + shapes["hedge"], pair_of, rp0, rp1 - 1, cp0, cp1),
        "vertex": _rect_add((P, max(ny - 1, 1), max(nx - 1, 1)), pair_of, rp0, rp1 - 1, cp0, cp1 - 1),
    }
    pair_of_traj = np.full(n_traj, -1, dtype=np.int64)
    pair_of_traj[planar] = pair_of
    for kind in sizes:  # real elements of planarized trajectories are not virtual
        t = whole[kind] // sizes[kind]
        e = whole[kind] % sizes[kind]
        keep = pair_of_traj[t] >= 0
        flat = cover[kind].reshape(P, -1) if P else cover[kind].reshape(0, sizes[kind])
        np.subtract.at(flat, (pair_of_traj[t[keep]], e[keep]), 1)
        cover[kind] = flat.reshape(cover[kind].shape)
        if np.any(cover[kind] < 0):
            raise AssertionError(f"negative virtual {kind} counts")
    if nx == 1:
        cover["vedge"][:] = 0
        cover["vertex"][:] = 0
    if ny == 1:
        cover["hedge"][:] = 0
        cover["vertex"][:] = 0
    for kind, values in cover.items():
        arrays[f"virtual/{kind}"] = values.astype(np.int32)

    def stored(array: np.ndarray) -> int:  # dense or (index, value) pairs, whichever is smaller
        return int(min(array.size, 2 * np.count_nonzero(array)))

    units = sum(stored(arrays[f"real/{k}"]) for k in sizes) + sum(stored(v) for v in cover.values())
    units += 5 * int(sum(arrays[f"links/{k}/count"].size for k in sizes)) + 2 * P + buckets.boundaries.size
    meta = {
        "method": "case_plus_temporal" if temporal else "case_plus", "temporal": bool(temporal),
        "unit_size_m": float(unit_size_m), "grid_shape": [ny, nx], "grid_cells": ny * nx,
        "trajectories": n_traj, "planarized_trajectories": int(planar.size),
        "time_buckets_requested": int(time_buckets) if temporal else 1, "time_buckets": B,
        "time_bucket_depth": "segments by time midpoint (equi-depth)" if temporal else None,
        "trajectory_spans": P, "links": link_total,
        "stage2_rule": "two visited cells of one row/column not joined by a straight run of visited cells with crossed edges",
        "theoretical_bytes": int(units * UNIT_BYTES),
        "theoretical_bytes_rule": "8 bytes per entry; each histogram dense or as (index, value) pairs, whichever is smaller; 5 per link entry",
    }
    return arrays, meta


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _prefix(array: np.ndarray) -> np.ndarray:
    """Inclusive prefix sums over the last two axes, padded with a zero row/column."""
    out = np.zeros(array.shape[:-2] + (array.shape[-2] + 1, array.shape[-1] + 1), dtype=np.int64)
    out[..., 1:, 1:] = np.cumsum(np.cumsum(array.astype(np.int64), axis=-2), axis=-1)
    return out


def _rect_sum(prefix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    if r1 < r0 or c1 < c0:
        return np.zeros(prefix.shape[:-2], dtype=np.int64)
    return (prefix[..., r1 + 1, c1 + 1] - prefix[..., r0, c1 + 1] - prefix[..., r1 + 1, c0] + prefix[..., r0, c0])


class CaseModel:
    def __init__(self, settings: Any, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]):
        self.settings = settings
        self.temporal = bool(meta["temporal"])
        self.plus = bool(meta.get("plus", True))
        shape = np.asarray(arrays["grid/shape"], dtype=np.int64)
        self.grid = Grid(np.asarray(arrays["grid/origin"]), float(np.asarray(arrays["grid/unit"])[0]),
                         (int(shape[0]), int(shape[1])))
        self.time = TimeBuckets(np.asarray(arrays["time/boundaries"], dtype=np.int64))
        self.kinds = ("cell", "vedge", "hedge")
        # real: cumulative over buckets, then 2D prefix -> any bucket range and rectangle in O(1)
        self.real = {}
        for kind in self.kinds:
            values = np.asarray(arrays[f"real/{kind}"], dtype=np.int64)
            cumulative = np.concatenate((np.zeros((1,) + values.shape[1:], dtype=np.int64), np.cumsum(values, axis=0)))
            self.real[kind] = _prefix(cumulative)
        self.links = {kind: {part: np.asarray(arrays[f"links/{kind}/{part}"], dtype=np.int64)
                             for part in ("row", "col", "i", "j", "count")} for kind in self.kinds}
        self.pairs = np.asarray(arrays["virtual/pairs"], dtype=np.int64).reshape(-1, 2)
        self.virtual = {kind: np.asarray(arrays[f"virtual/{kind}"], dtype=np.int64)
                        for kind in ("cell", "vedge", "hedge", "vertex")}
        self.virtual_prefix = {kind: _prefix(values) for kind, values in self.virtual.items()}

    # -- restricted counts --------------------------------------------------

    def _real_sum(self, kind: str, a: int, b: int, r0: int, r1: int, c0: int, c1: int) -> int:
        if r1 < r0 or c1 < c0:
            return 0
        prefix = self.real[kind]
        total = int(_rect_sum(prefix[b + 1], r0, r1, c0, c1) - _rect_sum(prefix[a], r0, r1, c0, c1))
        links = self.links[kind]
        if links["count"].size:
            keep = ((links["i"] >= a) & (links["j"] <= b) & (links["row"] >= r0) & (links["row"] <= r1)
                    & (links["col"] >= c0) & (links["col"] <= c1))
            total -= int(links["count"][keep].sum())
        return total

    def _pairs_in(self, a: int, b: int) -> np.ndarray:
        if self.pairs.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero((self.pairs[:, 0] >= a) & (self.pairs[:, 1] <= b))

    def _virtual_sum(self, kind: str, pairs: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
        if pairs.size == 0 or r1 < r0 or c1 < c0:
            return 0
        return int(_rect_sum(self.virtual_prefix[kind][pairs], r0, r1, c0, c1).sum())

    def _virtual_min(self, kind: str, pairs: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> Optional[int]:
        if r1 < r0 or c1 < c0:
            return None  # no such elements in the region
        if pairs.size == 0:
            return 0
        return int(self.virtual[kind][pairs, r0:r1 + 1, c0:c1 + 1].sum(axis=0).min())

    def count(self, rect: Tuple[int, int, int, int], a: int, b: int) -> float:
        r0, r1, c0, c1 = rect
        faces = self._real_sum("cell", a, b, r0, r1, c0, c1)
        if faces == 0:
            return 0.0
        pairs = self._pairs_in(a, b)
        edges = self._real_sum("vedge", a, b, r0, r1, c0, c1 - 1) + self._real_sum("hedge", a, b, r0, r1 - 1, c0, c1)
        vfaces = self._virtual_sum("cell", pairs, r0, r1, c0, c1)
        vedges = (self._virtual_sum("vedge", pairs, r0, r1, c0, c1 - 1)
                  + self._virtual_sum("hedge", pairs, r0, r1 - 1, c0, c1))
        line = r1 == r0 or c1 == c0
        vvertices = 0 if line else self._virtual_sum("vertex", pairs, r0, r1 - 1, c0, c1 - 1)
        n = (faces + vfaces) - (edges + vedges) + vvertices
        if self.plus:
            minima = [self._virtual_min("cell", pairs, r0, r1, c0, c1)]
            edge_minima = [m for m in (self._virtual_min("vedge", pairs, r0, r1, c0, c1 - 1),
                                       self._virtual_min("hedge", pairs, r0, r1 - 1, c0, c1)) if m is not None]
            if edge_minima:
                minima.append(min(edge_minima))
            if not line:
                minima.append(self._virtual_min("vertex", pairs, r0, r1 - 1, c0, c1 - 1))
            n -= min(m for m in minima if m is not None)
        return float(max(n, 0))

    def estimate(self, record: Any) -> Estimate:
        query, reason = parse_space_time(self.settings, record, "CASE+")
        if query is None:
            return "unsupported", None, reason
        if query.interval is not None and not self.temporal:
            return "unsupported", None, "time predicate (CASE+ is spatial only; see CASE+-Temporal)"
        diagnostic = TRIP_TIME_DIAGNOSTIC if query.trip_time else ""
        rect = self.grid.cell_range(query.window)
        if rect is None:
            return "ok", 0.0, diagnostic
        buckets = (0, self.time.count - 1) if query.interval is None else self.time.interval(*query.interval)
        if buckets is None:
            return "ok", 0.0, diagnostic
        return "ok", self.count(rect, *buckets), diagnostic


class CasePlusAdapter(InProcessTrajectoryAdapter):
    latency_scope = "predicate_translation_and_histogram_estimate"
    timing_role = "in_process_case_plus"

    def _flag(self, key: str, default: bool) -> bool:
        value = self.config.adapter.get(key, default)
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")

    def _build_model(self, data, trips) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        adapter = self.config.adapter
        arrays, meta = build_case(data[self.settings.segments.table], trips,
                                  unit_size_m=float(adapter.get("unit_size_m", 100.0)),
                                  temporal=self._flag("temporal", False),
                                  time_buckets=int(adapter.get("time_buckets", 100)))
        meta["plus"] = self._flag("plus", True)
        return arrays, meta

    def _load_model(self, arrays, meta) -> CaseModel:
        return CaseModel(self.settings, arrays, meta)

    def _estimate(self, model: CaseModel, record: Any) -> Estimate:
        return model.estimate(record)
