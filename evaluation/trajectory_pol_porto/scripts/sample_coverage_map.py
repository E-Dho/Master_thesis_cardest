"""Where the trajectory-sampling baseline's samples lie: density rasters per sample size.

Draws exactly the trip samples of ``traj_sampling`` (``adapters/sampling.draw_sample``:
the same seeded simple random sample of whole trips, same random stream) for each
sample size, and rasterizes their segments into one grid over the spatial domain,
next to the full data. Read-only on the snapshot; writes one compressed ``.npz``
with the count rasters and a JSON with coverage figures. Rendering is separate
(``render_sample_coverage.py``), so this needs only numpy.

Each segment is rasterized as ``--points-per-segment`` evenly spaced points from
start to end (a segment is counted once per cell it touches at those points), so
long segments are not reduced to one dot.

Coverage, per sample size, relative to the full data on the same grid:

- ``cells_hit``: share of the cells with any segment that the sample touches;
- ``segment_mass_in_hit_cells``: share of all rasterized segment points that lie in
  touched cells (how much of the traffic sits where the sample has data);
- ``cells_hit_by_full_count``: the same share split by the full data's count per
  cell (1, 2-9, 10-99, ...), which shows that missed cells are the lightly used ones.

    python evaluation/trajectory_pol_porto/scripts/sample_coverage_map.py \\
        --snapshot-root $B/traj_snapshots/pol_v3 \\
        --benchmark-config query_generation/benchmark_v3/pol_v3_config.json \\
        --sizes 1000,10000,100000,1000000 --seed 0 --cell-m 10 --output pol_coverage.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.trajectory_pol_porto.traj_eval import snapshot as snapshot_module  # noqa: E402
from evaluation.trajectory_pol_porto.traj_eval.adapters.sampling import STREAM_SALT  # noqa: E402

COUNT_BUCKETS = (1, 10, 100, 1000, 10000)


def sample_positions(total: int, n: int, seed: int) -> np.ndarray:
    """The trip positions ``draw_sample`` takes (same generator, same call)."""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_SALT]))
    return np.sort(rng.choice(total, size=n, replace=False))


def domain_extent(settings, choice: str, segments: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    """The grid extent: the configured spatial domain of the segment geometry, or the data extent."""

    if choice == "data":
        return (float(min(np.min(segments["sx"]), np.min(segments["ex"]))),
                float(min(np.min(segments["sy"]), np.min(segments["ey"]))),
                float(max(np.max(segments["sx"]), np.max(segments["ex"]))),
                float(max(np.max(segments["sy"]), np.max(segments["ey"]))))
    table, attribute = settings.segments.table, settings.segments.spatial_attribute
    spec = next(a for a in settings.table(table)["attributes"] if a["name"] == attribute)
    domain = spec["domain"]
    return tuple(float(domain[key]) for key in ("min_x", "min_y", "max_x", "max_y"))  # type: ignore[return-value]


def rasterize(segments: Dict[str, np.ndarray], trip_of_segment: np.ndarray, masks: Sequence[np.ndarray],
              extent: Tuple[float, float, float, float], cell: float, points: int,
              chunk: int = 4_000_000) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Counts per cell for all segments and for each trip mask (one grid, row 0 = min y).

    A segment adds 1 to every distinct cell it touches at ``points`` evenly spaced
    positions; points outside the extent are dropped.
    """

    min_x, min_y, max_x, max_y = extent
    width = int(np.ceil((max_x - min_x) / cell))
    height = int(np.ceil((max_y - min_y) / cell))
    full = np.zeros(width * height, dtype=np.int64)
    sampled = [np.zeros(width * height, dtype=np.int64) for _ in masks]
    fractions = np.linspace(0.0, 1.0, points) if points > 1 else np.array([0.5])
    total = int(segments["sx"].size)
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        sx = np.asarray(segments["sx"][start:stop], dtype=np.float64)
        sy = np.asarray(segments["sy"][start:stop], dtype=np.float64)
        dx = np.asarray(segments["ex"][start:stop], dtype=np.float64) - sx
        dy = np.asarray(segments["ey"][start:stop], dtype=np.float64) - sy
        cells = np.empty((stop - start, fractions.size), dtype=np.int64)
        for k, f in enumerate(fractions):
            col = np.floor((sx + f * dx - min_x) / cell)
            row = np.floor((sy + f * dy - min_y) / cell)
            inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
            cells[:, k] = np.where(inside, row * width + col, -1).astype(np.int64)
        cells.sort(axis=1)
        distinct = np.ones_like(cells, dtype=bool)
        distinct[:, 1:] = cells[:, 1:] != cells[:, :-1]
        distinct &= cells >= 0
        rows_of_point = np.broadcast_to(np.arange(stop - start)[:, None], cells.shape)[distinct]
        flat = cells[distinct]
        full += np.bincount(flat, minlength=full.size)
        trips = trip_of_segment[start:stop][rows_of_point]
        for mask, counts in zip(masks, sampled):
            counts += np.bincount(flat[mask[trips]], minlength=counts.size)
    shape = (height, width)
    return full.reshape(shape), [counts.reshape(shape) for counts in sampled]


def coverage(full: np.ndarray, sampled: np.ndarray) -> Dict[str, object]:
    occupied = full > 0
    hit = occupied & (sampled > 0)
    by_count = {}
    edges = list(COUNT_BUCKETS) + [np.inf]
    for low, high in zip(edges[:-1], edges[1:]):
        in_bucket = occupied & (full >= low) & (full < high)
        label = f"{low}+" if high == np.inf else f"{low}-{int(high) - 1}"
        cells = int(in_bucket.sum())
        by_count[label] = {"cells": cells, "hit": float((hit & in_bucket).sum() / cells) if cells else None}
    return {
        "occupied_cells": int(occupied.sum()),
        "cells_hit": float(hit.sum() / occupied.sum()),
        "segment_mass_in_hit_cells": float(full[hit].sum() / full.sum()),
        "cells_hit_by_full_count": by_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--sizes", default="1000,10000,100000,1000000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell-m", type=float, required=True)
    parser.add_argument("--extent", choices=("window_domain", "data"), default="window_domain")
    parser.add_argument("--points-per-segment", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite")

    from query_generation.benchmark_v3.config import load_settings

    started = time.perf_counter()
    settings = load_settings(args.benchmark_config)
    data = snapshot_module.load_snapshot(args.snapshot_root, expected_config_sha256=settings.sha256)
    trips = snapshot_module.load_trip_index(args.snapshot_root)
    entity = settings.base["entity"]
    if not np.array_equal(np.asarray(data[entity["table"]][entity["key"]]), np.asarray(trips["trip_id"])):
        raise SystemExit("entity table and derived trips differ (as draw_sample checks)")
    segments = data[settings.segments.table]
    total = int(trips["trip_id"].size)
    counts = np.asarray(trips["segment_count"], dtype=np.int64)
    if int(counts.sum()) != int(segments["sx"].size):
        raise SystemExit("trip segment counts do not add up to the segment table")
    trip_of_segment = np.repeat(np.arange(total, dtype=np.int64), counts)
    sizes = [int(s) for s in args.sizes.split(",") if s]
    if any(not 1 <= s <= total for s in sizes):
        raise SystemExit(f"sizes must lie in [1, {total}]")
    masks = []
    for n in sizes:
        mask = np.zeros(total, dtype=bool)
        mask[sample_positions(total, n, args.seed)] = True
        masks.append(mask)
    extent = domain_extent(settings, args.extent, segments)
    full, sampled = rasterize(segments, trip_of_segment, masks, extent, args.cell_m, args.points_per_segment)
    summary = {
        "snapshot_root": str(args.snapshot_root),
        "benchmark_config": str(args.benchmark_config),
        "population_trips": total,
        "population_segments": int(segments["sx"].size),
        "seed": args.seed,
        "extent": args.extent,
        "extent_xy": extent,
        "cell_m": args.cell_m,
        "grid_shape": list(full.shape),
        "points_per_segment": args.points_per_segment,
        "samples": [
            {"sample_trips": n, "sample_fraction": n / total,
             "sampled_segments": int(counts[mask].sum()),
             **coverage(full, grid)}
            for n, mask, grid in zip(sizes, masks, sampled)
        ],
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, full=full.astype(np.uint32),
                        **{f"n{n}": grid.astype(np.uint32) for n, grid in zip(sizes, sampled)},
                        sizes=np.asarray(sizes), extent=np.asarray(extent))
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
