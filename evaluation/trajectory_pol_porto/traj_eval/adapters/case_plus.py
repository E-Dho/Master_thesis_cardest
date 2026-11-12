"""CASE+ and CASE+-Temporal (P6) on the v3 multi (distinct-trajectory) workload.

**Base: CASE / CASE+** (Fanaeepour, Kulik, Tanin, Rubinstein, "The CASE
histogram", GeoInformatica 19(4), 2015; no public code, re-implemented from
Algorithms 1-3 and Fig. 12). On a regular grid (cell size ``unit_size_m``
over the data extent, or with ``adapter.grid_extent: window_domain`` over the
configured window domain, Porto; segment ends outside it fall into the border
cells):

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
region containing both sees them disconnected. In addition (review of
f20c7d7), every trajectory whose cells and crossed edges do not form a tree
is planarized: a cycle (e.g. around a block) would give ``F - E = 0`` and a
jump between non-adjacent segments ``F - E = 2``; Euler counting of an
unplanarized trajectory is only exact for a tree.

**CASE+-Temporal** (extension by us, spec "Temporal extension for CASE+ and
TraHist", note 5): time-restricted CASE+ is CASE+ applied to every trajectory
restricted to the query interval. A trajectory whose segments span the
equi-depth time buckets ``i..j`` (a segment occupies every bucket its time
span intersects) has ``s(s+1)/2`` possible restrictions ``lo..hi`` (``s =
j - i + 1``); each is processed by the full CASE construction (Stage 2,
tree check, planarization) at build time, and the counts are summed per key
``(i, j, lo, hi)``. A query over buckets ``a..b`` takes, for every span it
intersects, the key with ``lo = max(a, i)`` and ``hi = min(b, j)``, so the
counts are exactly those of CASE built on the cut trajectories (tested), and
a cut trajectory keeps its own planarization. Trips are much shorter than a
bucket, so almost every trajectory has one or three restrictions. With the
full time range the counts equal the base exactly (gate).

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

from ..temporal import SettingTooLarge, TimeBuckets
from .base import Estimate, InProcessTrajectoryAdapter, grid_extent
from .space_time import parse_space_time

TRIP_TIME_DIAGNOSTIC = "approx:trip_time_as_segment_time"
UNIT_BYTES = 8
KEY_LIMIT = 1 << 62
#: (piece, element) pairs found by the traversal: refused above this (each kind is an int64
#: array of this length during the build)
MAX_ELEMENTS = 600_000_000
#: entries gathered at once by a rectangle query (bounds the query's temporary memory)
QUERY_CHUNK = 8_000_000
HISTOGRAMS = ("cell", "vedge", "hedge")
VIRTUAL = ("cell", "vedge", "hedge", "vertex")


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
        raise SettingTooLarge("key space too large; use fewer cells or time buckets")
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


def _piece_table(span_i: np.ndarray, span_j: np.ndarray, max_pieces: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every bucket range ``lo..hi`` inside each trajectory's span: (owner, lo, hi, offsets per trajectory)."""
    length = span_j - span_i + 1
    counts = length * (length + 1) // 2
    total = int(counts.sum())
    if total > max_pieces:
        raise SettingTooLarge(f"{total} restricted trajectories (longest span {int(length.max())} buckets); "
                          "use fewer time buckets")
    owner = np.repeat(np.arange(span_i.size, dtype=np.int64), counts)
    lo = np.empty(total, dtype=np.int64)
    hi = np.empty(total, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    for size in np.unique(length):
        pairs = np.array([(a, b) for a in range(int(size)) for b in range(a, int(size))], dtype=np.int64)
        trajectories = np.flatnonzero(length == size)
        index = (offsets[trajectories][:, None] + np.arange(pairs.shape[0])[None, :]).ravel()
        lo[index] = (span_i[trajectories][:, None] + pairs[None, :, 0]).ravel()
        hi[index] = (span_i[trajectories][:, None] + pairs[None, :, 1]).ravel()
    return owner, lo, hi, offsets


def _index_dtype(maximum: int) -> np.dtype:
    """The smallest unsigned integer type that holds 0..maximum."""
    for dtype in (np.uint8, np.uint16, np.uint32):
        if maximum <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    return np.dtype(np.int64)


def _sparse(name: str, key: np.ndarray, element: np.ndarray, n_keys: int, size: int) -> Dict[str, np.ndarray]:
    """Counts per (key, element), non-zero only, sorted by element then key."""
    if key.size == 0:
        combined, count = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        combined, count = np.unique(np.asarray(element, dtype=np.int64) * n_keys + np.asarray(key, dtype=np.int64),
                                    return_counts=True)
    return {f"{name}/element": (combined // n_keys).astype(_index_dtype(max(size - 1, 0))),
            f"{name}/key": (combined % n_keys).astype(_index_dtype(max(n_keys - 1, 0))),
            f"{name}/count": count.astype(_index_dtype(int(count.max(initial=0))))}


def build_case(segments: Mapping[str, np.ndarray], trips: Mapping[str, np.ndarray], *, unit_size_m: float,
               temporal: bool, time_buckets: int, chunk_segments: int = 4_000_000,
               extent: Optional[Tuple[float, float, float, float]] = None,
               max_pieces: int = 200_000_000) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """All CASE histograms, per (trajectory span, restriction) key.

    A trajectory spanning time buckets ``i..j`` is processed by CASE once for
    every bucket range ``lo..hi`` inside its span (its segments whose time span
    intersects ``lo..hi``): detection, planarization, real and virtual counts.
    The counts are summed per key ``(i, j, lo, hi)``. A query over buckets
    ``a..b`` takes, for every span ``i..j`` it intersects, the key with
    ``lo = max(a, i)`` and ``hi = min(b, j)``: exactly CASE applied to every
    trajectory restricted to the query interval. The base variant has one
    bucket, one key and one restriction per trajectory (the whole trajectory).

    The grid covers ``extent`` (xmin, ymin, xmax, ymax), by default the data
    extent; points outside it fall into the border cells.
    """
    sx, sy, ex, ey = (np.asarray(segments[k], dtype=np.float64) for k in ("sx", "sy", "ex", "ey"))
    ts, te = np.asarray(segments["ts"], dtype=np.int64), np.asarray(segments["te"], dtype=np.int64)
    offsets = np.asarray(trips["offset"], dtype=np.int64)
    n_traj = int(offsets.size - 1)
    n_seg = sx.size
    seg_traj = np.repeat(np.arange(n_traj, dtype=np.int64), np.diff(offsets))
    if extent is None:
        extent = (float(min(sx.min(), ex.min())), float(min(sy.min(), ey.min())),
                  float(max(sx.max(), ex.max())), float(max(sy.max(), ey.max())))
    grid = Grid.covering(*(float(v) for v in extent), float(unit_size_m))
    ny, nx = grid.ny, grid.nx
    shapes = {"cell": (ny, nx), "vedge": (ny, max(nx - 1, 1)), "hedge": (max(ny - 1, 1), nx)}
    sizes = {kind: shape[0] * shape[1] for kind, shape in shapes.items()}
    if temporal:
        buckets = TimeBuckets.equi_depth(ts + (te - ts) // 2, int(time_buckets), int(ts.min()), int(te.max()))
    else:
        buckets = TimeBuckets(np.array([int(ts.min()), int(te.max()) + 1], dtype=np.int64))
    B = buckets.count
    seg_first, seg_last = buckets.assign(ts).astype(np.int64), buckets.assign(te).astype(np.int64)
    span_i = np.full(n_traj, B, dtype=np.int64)
    span_j = np.full(n_traj, -1, dtype=np.int64)
    np.minimum.at(span_i, seg_traj, seg_first)
    np.maximum.at(span_j, seg_traj, seg_last)

    # restricted trajectories ("pieces") and the segments of each
    owner, lo, hi, piece_offsets = _piece_table(span_i, span_j, max_pieces)
    n_pieces = owner.size
    per_traj = np.diff(piece_offsets)
    candidates = per_traj[seg_traj]
    cand_seg = np.repeat(np.arange(n_seg, dtype=np.int64), candidates)
    cand_piece = (piece_offsets[seg_traj][cand_seg]
                  + np.arange(cand_seg.size, dtype=np.int64) - np.repeat(np.cumsum(candidates) - candidates, candidates))
    keep = (lo[cand_piece] <= seg_last[cand_seg]) & (hi[cand_piece] >= seg_first[cand_seg])
    member_seg, member_piece = cand_seg[keep], cand_piece[keep]
    del cand_seg, cand_piece, keep
    pieces_of_seg = np.bincount(member_seg, minlength=n_seg)
    seg_member_offset = np.concatenate(([0], np.cumsum(pieces_of_seg))).astype(np.int64)

    # elements of every piece (deduplicated), chunk by chunk over segments
    found: Dict[str, list] = {kind: [] for kind in sizes}
    found_count = {kind: 0 for kind in sizes}
    for start in range(0, n_seg, chunk_segments):
        stop = min(start + chunk_segments, n_seg)
        visited = traverse(grid, sx[start:stop], sy[start:stop], ex[start:stop], ey[start:stop])
        for kind in sizes:
            seg = visited[f"{kind}_seg"] + start
            element = visited[kind]
            repeat = pieces_of_seg[seg]
            index = np.repeat(np.arange(seg.size), repeat)
            local = np.arange(index.size, dtype=np.int64) - np.repeat(np.cumsum(repeat) - repeat, repeat)
            piece = member_piece[seg_member_offset[seg[index]] + local]
            found[kind].append(_unique_keys((piece, n_pieces), (element[index], sizes[kind])))
            found_count[kind] += found[kind][-1].size
            if found_count[kind] > MAX_ELEMENTS:
                raise SettingTooLarge(f"more than {MAX_ELEMENTS} (restricted trajectory, {kind}) pairs at "
                                      f"{grid.ny * grid.nx} cells; use a coarser grid or fewer buckets")
    elements = {kind: (np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64))
                for kind, parts in found.items()}
    del found

    # Stage 2 per piece, plus the Euler preconditions of an unplanarized trajectory:
    # its cells and crossed edges must form a tree (connected, E = F - 1)
    problem = detect_problem(grid, n_pieces, elements["cell"], elements["vedge"], elements["hedge"])
    faces = np.bincount(elements["cell"] // sizes["cell"], minlength=n_pieces)
    edges = (np.bincount(elements["vedge"] // sizes["vedge"], minlength=n_pieces)
             + np.bincount(elements["hedge"] // sizes["hedge"], minlength=n_pieces))
    end_cell = grid.row(ey) * nx + grid.column(ex)
    start_cell = grid.row(sy) * nx + grid.column(sx)
    broken = np.zeros(n_seg, dtype=np.int64)  # junction k -> k+1 inside one trajectory jumps cells
    if n_seg > 1:
        broken[:-1] = (seg_traj[1:] == seg_traj[:-1]) & (end_cell[:-1] != start_cell[1:])
    broken_prefix = np.concatenate(([0], np.cumsum(broken)))
    seg_lo = np.full(n_pieces, n_seg, dtype=np.int64)
    seg_hi = np.full(n_pieces, -1, dtype=np.int64)
    np.minimum.at(seg_lo, member_piece, member_seg)
    np.maximum.at(seg_hi, member_piece, member_seg)
    jumps = broken_prefix[np.maximum(seg_hi, seg_lo)] - broken_prefix[seg_lo]
    tree = (edges == faces - 1) & (jumps == 0)
    stage2 = problem.copy()
    problem |= ~tree

    # keys (trajectory span, restriction)
    key_values, key_of_piece = np.unique(((span_i[owner] * B + span_j[owner]) * B + lo) * B + hi,
                                         return_inverse=True)
    K = key_values.size
    keys = np.column_stack(_split(key_values, B, B, B)).astype(np.int32).reshape(K, 4)

    arrays: Dict[str, np.ndarray] = {
        "grid/origin": grid.origin, "grid/unit": np.array([grid.unit]), "grid/shape": np.array([ny, nx]),
        "time/boundaries": buckets.boundaries, "keys": keys,
    }
    # Sparse storage (2026-11-12): only non-zero counts. Real counts per (key, element); the
    # virtual counts of a planarized piece are its cell MBR (as a rectangle per kind) minus its
    # real elements, so they are stored as the rectangles (with multiplicity) plus the real
    # counts of the planarized pieces. Queries are exactly those of the dense histograms.
    planar = np.flatnonzero(problem & (faces > 0))
    is_planar = np.zeros(n_pieces, dtype=bool)
    is_planar[planar] = True
    entries = 0
    for kind in sizes:
        piece, element = elements[kind] // sizes[kind], elements[kind] % sizes[kind]
        for prefix, keep in (("real", None), ("preal", is_planar[piece])):
            p, e = (piece, element) if keep is None else (piece[keep], element[keep])
            arrays.update(_sparse(f"{prefix}/{kind}", key_of_piece[p], e, K, sizes[kind]))
            entries += arrays[f"{prefix}/{kind}/count"].size
    piece_of_cell = elements["cell"] // sizes["cell"]
    cell = elements["cell"] % sizes["cell"]
    big = np.iinfo(np.int64).max
    r0 = np.full(n_pieces, big); r1 = np.full(n_pieces, -1); c0 = np.full(n_pieces, big); c1 = np.full(n_pieces, -1)
    np.minimum.at(r0, piece_of_cell, cell // nx); np.maximum.at(r1, piece_of_cell, cell // nx)
    np.minimum.at(c0, piece_of_cell, cell % nx); np.maximum.at(c1, piece_of_cell, cell % nx)
    rect_rows = np.column_stack((key_of_piece[planar], r0[planar], r1[planar], c0[planar], c1[planar]))
    rect_rows, rect_count = (np.unique(rect_rows, axis=0, return_counts=True) if planar.size
                             else (np.zeros((0, 5), dtype=np.int64), np.zeros(0, dtype=np.int64)))
    arrays["rect/bounds"] = rect_rows.astype(_index_dtype(max(int(rect_rows.max(initial=0)), K)))
    arrays["rect/count"] = rect_count.astype(_index_dtype(int(rect_count.max(initial=0))))
    del piece_of_cell, cell

    units = 3 * entries + 6 * rect_count.size + 4 * K + buckets.boundaries.size
    whole = (lo == span_i[owner]) & (hi == span_j[owner])
    meta = {
        "method": "case_plus_temporal" if temporal else "case_plus", "temporal": bool(temporal),
        "unit_size_m": float(unit_size_m), "grid_shape": [ny, nx], "grid_cells": ny * nx,
        "trajectories": n_traj, "restricted_trajectories": int(n_pieces),
        "planarized_trajectories": int(np.count_nonzero(problem[whole] & (faces[whole] > 0))),
        "planarized_by_stage2": int(np.count_nonzero(stage2[whole])),
        "planarized_as_non_tree": int(np.count_nonzero(~tree[whole] & ~stage2[whole])),
        "time_buckets_requested": int(time_buckets) if temporal else 1, "time_buckets": B,
        "time_bucket_depth": "segments by time midpoint (equi-depth)" if temporal else None,
        "longest_span_buckets": int((span_j - span_i).max() + 1), "keys": int(K),
        "stage2_rule": ("two visited cells of one row/column not joined by a straight run of visited cells with "
                        "crossed edges; additionally every trajectory whose cells and crossed edges are not a tree "
                        "(a cycle, or a jump between segments) is planarized"),
        "theoretical_bytes": int(units * UNIT_BYTES),
        "theoretical_bytes_rule": ("8 bytes per number: (element, key, count) per non-zero real count and per "
                                   "non-zero real count of a planarized piece, (key, 4 bounds, multiplicity) per "
                                   "virtual rectangle, 4 per key, the bucket boundaries"),
        "storage": "sparse (non-zero counts; virtual counts as rectangles minus planarized real counts)",
        "nonzero_entries": int(entries), "virtual_rectangles": int(rect_count.size),
    }
    return arrays, meta


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class SparseCounts:
    """One histogram's non-zero counts per (key, element), sorted by element (row-major) then key."""

    def __init__(self, arrays: Mapping[str, np.ndarray], name: str, shape: Tuple[int, int], n_keys: int):
        # elements and keys stay in their compact stored types; counts are summed, so int64
        self.element = np.asarray(arrays[f"{name}/element"])
        self.key = np.asarray(arrays[f"{name}/key"])
        self.count = np.asarray(arrays[f"{name}/count"]).astype(np.int64)
        self.rows, self.width = int(shape[0]), int(shape[1])
        # whole-grid totals per key (a region covering the grid needs no scan)
        self.key_total = np.bincount(self.key, weights=self.count, minlength=n_keys).astype(np.int64)

    def _whole(self, r0: int, r1: int, c0: int, c1: int) -> bool:
        return r0 <= 0 and c0 <= 0 and r1 >= self.rows - 1 and c1 >= self.width - 1

    def _blocks(self, r0: int, r1: int, c0: int, c1: int):
        """Entry indices inside rows r0..r1, columns c0..c1, in blocks of at most QUERY_CHUNK entries."""
        rows = np.arange(r0, r1 + 1, dtype=np.int64)
        dtype = self.element.dtype  # same type as the stored elements: no conversion of the whole array
        lo = np.searchsorted(self.element, (rows * self.width + c0).astype(dtype), side="left")
        hi = np.searchsorted(self.element, (rows * self.width + c1).astype(dtype), side="right")
        if c0 == 0 and c1 == self.width - 1:  # whole rows: one contiguous range
            lo, hi = lo[:1], hi[-1:]
        sizes = hi - lo
        start = 0
        while start < sizes.size:
            stop, total = start, 0
            while stop < sizes.size and (stop == start or total + sizes[stop] <= QUERY_CHUNK):
                total += int(sizes[stop]); stop += 1
            if total:
                n = sizes[start:stop]
                first = np.repeat(lo[start:stop] - (np.cumsum(n) - n), n)
                yield first + np.arange(total, dtype=np.int64)
            start = stop

    def total(self, selected: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
        """Sum of the counts of the selected keys (boolean mask over keys) inside the rectangle."""
        if r1 < r0 or c1 < c0:
            return 0
        if self._whole(r0, r1, c0, c1):
            return int(self.key_total[selected].sum())
        out = 0
        for index in self._blocks(r0, r1, c0, c1):
            keep = selected[self.key[index]]
            out += int(self.count[index][keep].sum())
        return out

    def per_element(self, selected: np.ndarray, r0: int, r1: int, c0: int, c1: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """(element, summed count of the selected keys) inside the rectangle, elements unique."""
        elements, counts = [], []
        for index in self._blocks(r0, r1, c0, c1):
            index = index[selected[self.key[index]]]
            elements.append(self.element[index]); counts.append(self.count[index])
        if not elements:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        element, count = np.concatenate(elements), np.concatenate(counts)
        if element.size == 0:
            return element, count
        # entries are sorted by element (blocks in row order, rows ascending): group runs
        starts = np.flatnonzero(np.concatenate(([True], element[1:] != element[:-1])))
        return element[starts], np.add.reduceat(count, starts)


class CaseModel:
    """CASE+ / CASE+-Temporal queries on the sparse histograms (exactly the dense formulas).

    For a query over buckets ``a..b`` the selected keys are one per trajectory span; real
    sums add the selected keys' counts inside the rectangle; a virtual sum is the selected
    rectangles' overlap with it minus the planarized pieces' real counts inside it; a virtual
    minimum (CASE+) builds the per-element virtual counts over the query rectangle only (it is
    0 without building anything when the selected rectangles cannot cover every element).
    """

    def __init__(self, settings: Any, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]):
        self.settings = settings
        self.temporal = bool(meta["temporal"])
        self.plus = bool(meta.get("plus", True))
        shape = np.asarray(arrays["grid/shape"], dtype=np.int64)
        self.grid = Grid(np.asarray(arrays["grid/origin"]), float(np.asarray(arrays["grid/unit"])[0]),
                         (int(shape[0]), int(shape[1])))
        self.time = TimeBuckets(np.asarray(arrays["time/boundaries"], dtype=np.int64))
        self.keys = np.asarray(arrays["keys"], dtype=np.int64).reshape(-1, 4)
        ny, nx = self.grid.ny, self.grid.nx
        shapes = {"cell": (ny, nx), "vedge": (ny, max(nx - 1, 1)), "hedge": (max(ny - 1, 1), nx)}
        K = self.keys.shape[0]
        self.real = {kind: SparseCounts(arrays, f"real/{kind}", shapes[kind], K) for kind in HISTOGRAMS}
        self.preal = {kind: SparseCounts(arrays, f"preal/{kind}", shapes[kind], K) for kind in HISTOGRAMS}
        bounds = np.asarray(arrays["rect/bounds"]).astype(np.int64).reshape(-1, 5)
        self.rect_key = bounds[:, 0]
        self.rect_count = np.asarray(arrays["rect/count"]).astype(np.int64)
        r0, r1, c0, c1 = bounds[:, 1], bounds[:, 2], bounds[:, 3], bounds[:, 4]
        # each piece's virtual elements per kind: its cells, interior edges and interior vertices
        self.rects = {"cell": (r0, r1, c0, c1), "vedge": (r0, r1, c0, c1 - 1),
                      "hedge": (r0, r1 - 1, c0, c1), "vertex": (r0, r1 - 1, c0, c1 - 1)}

    def selected(self, a: int, b: int) -> np.ndarray:
        """Per trajectory span intersecting ``a..b``, the key of its restriction to ``a..b``."""
        i, j, lo, hi = self.keys.T
        return np.flatnonzero((i <= b) & (j >= a) & (lo == np.maximum(a, i)) & (hi == np.minimum(b, j)))

    def _mask(self, keys: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.keys.shape[0], dtype=bool)
        mask[keys] = True
        return mask

    def _clipped(self, kind: str, mask: np.ndarray, r0: int, r1: int, c0: int, c1: int):
        """The selected rectangles of ``kind`` clipped to the region (non-empty only) and their multiplicity."""
        pr0, pr1, pc0, pc1 = self.rects[kind]
        keep = mask[self.rect_key]
        a0, a1 = np.maximum(pr0[keep], r0), np.minimum(pr1[keep], r1)
        b0, b1 = np.maximum(pc0[keep], c0), np.minimum(pc1[keep], c1)
        count = self.rect_count[keep]
        nonempty = (a1 >= a0) & (b1 >= b0)
        return a0[nonempty], a1[nonempty], b0[nonempty], b1[nonempty], count[nonempty]

    def _sum(self, name: str, mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
        if r1 < r0 or c1 < c0:
            return 0
        prefix, kind = name.split("/")
        if prefix == "real":
            return self.real[kind].total(mask, r0, r1, c0, c1)
        a0, a1, b0, b1, count = self._clipped(kind, mask, r0, r1, c0, c1)
        covered = int((count * (a1 - a0 + 1) * (b1 - b0 + 1)).sum())
        return covered - (self.preal[kind].total(mask, r0, r1, c0, c1) if kind in self.preal else 0)

    def _min(self, name: str, mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> Optional[int]:
        """Minimum over the region's elements of the selected keys' summed virtual counts.

        The rectangles' coverage is piecewise constant on the blocks their (clipped) bounds cut
        the region into, so it is built on those blocks, not per element. A block keeps its
        coverage as a candidate if some element of it has no planarized real count; every element
        with one is a candidate with coverage minus its summed count. Exact, as the dense minimum.
        """
        if r1 < r0 or c1 < c0:
            return None  # no such elements in the region
        kind = name.split("/")[1]
        a0, a1, b0, b1, count = self._clipped(kind, mask, r0, r1, c0, c1)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if int((count * (a1 - a0 + 1) * (b1 - b0 + 1)).sum()) < h * w:
            return 0  # some element is covered by no selected rectangle: its virtual count is 0
        rows = np.unique(np.concatenate(([r0, r1 + 1], a0, a1 + 1)))
        cols = np.unique(np.concatenate(([c0, c1 + 1], b0, b1 + 1)))
        nr, nc = rows.size - 1, cols.size - 1
        diff = np.zeros((nr + 1, nc + 1), dtype=np.int64)
        i0, i1 = np.searchsorted(rows, a0), np.searchsorted(rows, a1 + 1)
        j0, j1 = np.searchsorted(cols, b0), np.searchsorted(cols, b1 + 1)
        np.add.at(diff, (i0, j0), count)
        np.add.at(diff, (i0, j1), -count)
        np.add.at(diff, (i1, j0), -count)
        np.add.at(diff, (i1, j1), count)
        cover = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:nr, :nc]
        if kind not in self.preal or int(cover.min()) == 0:
            return int(cover.min())  # an uncovered block has no planarized real counts either
        element, summed = self.preal[kind].per_element(mask, r0, r1, c0, c1)
        if element.size == 0:
            return int(cover.min())
        width = self.preal[kind].width
        element = element.astype(np.int64)
        bi = np.searchsorted(rows, element // width, side="right") - 1
        bj = np.searchsorted(cols, element % width, side="right") - 1
        occupied = np.bincount(bi * nc + bj, minlength=nr * nc).reshape(nr, nc)
        free = occupied < np.diff(rows)[:, None] * np.diff(cols)[None, :]
        values = cover[bi, bj] - summed
        best = int(values.min())
        if free.any():
            best = min(best, int(cover[free].min()))
        return best

    def count(self, rect: Tuple[int, int, int, int], a: int, b: int) -> float:
        r0, r1, c0, c1 = rect
        keys = self._mask(self.selected(a, b))
        faces = self._sum("real/cell", keys, r0, r1, c0, c1)
        if faces == 0:
            return 0.0
        edges = self._sum("real/vedge", keys, r0, r1, c0, c1 - 1) + self._sum("real/hedge", keys, r0, r1 - 1, c0, c1)
        vfaces = self._sum("virtual/cell", keys, r0, r1, c0, c1)
        vedges = (self._sum("virtual/vedge", keys, r0, r1, c0, c1 - 1)
                  + self._sum("virtual/hedge", keys, r0, r1 - 1, c0, c1))
        line = r1 == r0 or c1 == c0
        vvertices = 0 if line else self._sum("virtual/vertex", keys, r0, r1 - 1, c0, c1 - 1)
        n = (faces + vfaces) - (edges + vedges) + vvertices
        if self.plus:
            minima = [self._min("virtual/cell", keys, r0, r1, c0, c1)]
            edge_minima = [m for m in (self._min("virtual/vedge", keys, r0, r1, c0, c1 - 1),
                                       self._min("virtual/hedge", keys, r0, r1 - 1, c0, c1)) if m is not None]
            if edge_minima:
                minima.append(min(edge_minima))
            if not line:
                minima.append(self._min("virtual/vertex", keys, r0, r1 - 1, c0, c1 - 1))
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
        extent = grid_extent(self.settings, adapter.get("grid_extent"))
        arrays, meta = build_case(data[self.settings.segments.table], trips,
                                  unit_size_m=float(adapter.get("unit_size_m", 100.0)),
                                  temporal=self._flag("temporal", False),
                                  time_buckets=int(adapter.get("time_buckets", 100)), extent=extent)
        meta["plus"] = self._flag("plus", True)
        if extent is not None:
            meta.update({"grid_extent": "window_domain", "grid_extent_box": list(extent)})
        return arrays, meta

    def _load_model(self, arrays, meta) -> CaseModel:
        return CaseModel(self.settings, arrays, meta)

    def _estimate(self, model: CaseModel, record: Any) -> Estimate:
        return model.estimate(record)
