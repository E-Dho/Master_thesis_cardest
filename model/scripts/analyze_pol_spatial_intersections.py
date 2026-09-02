from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from model.src.data.trajectory_distinct import (
    CompactTrajectorySegmentIndex,
    segment_rectangle_intersects_mask,
)


@dataclass(frozen=True)
class SpatialRectangleQuery:
    query_id: str
    predicate_index: int
    table: str
    attribute: str
    mode: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    category_key: str | None = None


@dataclass(frozen=True)
class SpatialIntersectionStats:
    query_id: str
    predicate_index: int
    table: str
    attribute: str
    mode: str
    category_key: str | None
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    window_width: float
    window_height: float
    window_area: float
    M_Q: int
    B_Q: int
    O_Q: int
    C_Q: int
    B_fraction: float | None
    O_fraction: float | None
    C_fraction: float | None
    current_endpoint_model_recall: float | None
    mean_matching_segment_length: float | None
    min_matching_segment_length: float | None
    p50_matching_segment_length: float | None
    p90_matching_segment_length: float | None
    p95_matching_segment_length: float | None
    p99_matching_segment_length: float | None
    max_matching_segment_length: float | None
    window_width_over_mean_matching_segment_length: float | None
    window_height_over_mean_matching_segment_length: float | None
    sqrt_window_area_over_mean_matching_segment_length: float | None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze how often endpoint-contained spatial conditioning misses "
            "physical POL ST_Intersects segment/rectangle matches."
        )
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help="One or more POL workload JSONL files or directories containing queries.jsonl.",
    )
    parser.add_argument(
        "--segments-index",
        default=None,
        help="Prepared compact trajectory_segment_index directory.",
    )
    parser.add_argument(
        "--segments-tsv",
        default=None,
        help=(
            "Headerless POL staging segments.tsv. Intended for small/debug inputs; "
            "use --segments-index for the 50M/60M workload."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--bucket-edges",
        default="0,0.5,1,2,5,10,inf",
        help="Comma-separated bucket edges for sqrt(area)/mean-segment-length.",
    )
    args = parser.parse_args()
    if bool(args.segments_index) == bool(args.segments_tsv):
        raise SystemExit("set exactly one of --segments-index or --segments-tsv")
    if int(args.chunk_size) <= 0:
        raise SystemExit("--chunk-size must be positive")

    queries = list(_load_spatial_queries([Path(value) for value in args.queries]))
    if args.max_queries is not None:
        queries = queries[: int(args.max_queries)]
    if not queries:
        raise SystemExit("no spatial predicates found in query workload")

    segments = (
        _segments_from_index(Path(args.segments_index))
        if args.segments_index
        else _segments_from_tsv(Path(args.segments_tsv))
    )
    results: list[SpatialIntersectionStats] = []
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as sink:
        for query in queries:
            stats = analyze_rectangle_query(
                query,
                sx=segments["s_x"],
                sy=segments["s_y"],
                ex=segments["e_x"],
                ey=segments["e_y"],
                chunk_size=int(args.chunk_size),
            )
            _validate_stats(stats)
            results.append(stats)
            sink.write(json.dumps(asdict(stats), sort_keys=True) + "\n")

    if args.output_csv is not None:
        _write_csv(Path(args.output_csv), results)
    summary = summarize_results(results, bucket_edges=_parse_bucket_edges(args.bucket_edges))
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"spatial_queries_analyzed={len(results)}")
    print(f"output_jsonl={output_jsonl}")
    if args.output_csv is not None:
        print(f"output_csv={args.output_csv}")
    print(f"summary={args.summary}")
    global_fractions = summary["cardinality_weighted_global_fractions"]
    print(
        "global_fractions="
        f"B:{global_fractions['B_fraction']},"
        f"O:{global_fractions['O_fraction']},"
        f"C:{global_fractions['C_fraction']}"
    )


def analyze_rectangle_query(
    query: SpatialRectangleQuery,
    *,
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    chunk_size: int = 1_000_000,
) -> SpatialIntersectionStats:
    min_x, max_x = sorted((float(query.min_x), float(query.max_x)))
    min_y, max_y = sorted((float(query.min_y), float(query.max_y)))
    b_count = 0
    o_count = 0
    c_count = 0
    length_sum = 0.0
    length_min = math.inf
    length_max = 0.0
    matching_length_chunks: list[np.ndarray] = []
    total = len(sx)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        sx_chunk = np.asarray(sx[start:stop], dtype=float)
        sy_chunk = np.asarray(sy[start:stop], dtype=float)
        ex_chunk = np.asarray(ex[start:stop], dtype=float)
        ey_chunk = np.asarray(ey[start:stop], dtype=float)
        inside_start = (
            (min_x <= sx_chunk)
            & (sx_chunk <= max_x)
            & (min_y <= sy_chunk)
            & (sy_chunk <= max_y)
        )
        inside_end = (
            (min_x <= ex_chunk)
            & (ex_chunk <= max_x)
            & (min_y <= ey_chunk)
            & (ey_chunk <= max_y)
        )
        intersects = segment_rectangle_intersects_mask(
            sx_chunk,
            sy_chunk,
            ex_chunk,
            ey_chunk,
            min_x,
            min_y,
            max_x,
            max_y,
        )
        both = intersects & inside_start & inside_end
        one = intersects & (inside_start ^ inside_end)
        crossing = intersects & ~(inside_start | inside_end)
        b_count += int(np.count_nonzero(both))
        o_count += int(np.count_nonzero(one))
        c_count += int(np.count_nonzero(crossing))
        if np.any(intersects):
            lengths = np.hypot(
                ex_chunk[intersects] - sx_chunk[intersects],
                ey_chunk[intersects] - sy_chunk[intersects],
            )
            length_sum += float(np.sum(lengths))
            length_min = min(length_min, float(np.min(lengths)))
            length_max = max(length_max, float(np.max(lengths)))
            matching_length_chunks.append(lengths.astype(float, copy=False))
    matching = b_count + o_count + c_count
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    mean_length = None if matching == 0 else length_sum / matching
    length_percentiles = _length_percentiles(matching_length_chunks)
    return SpatialIntersectionStats(
        query_id=query.query_id,
        predicate_index=query.predicate_index,
        table=query.table,
        attribute=query.attribute,
        mode=query.mode,
        category_key=query.category_key,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        window_width=width,
        window_height=height,
        window_area=area,
        M_Q=matching,
        B_Q=b_count,
        O_Q=o_count,
        C_Q=c_count,
        B_fraction=_fraction(b_count, matching),
        O_fraction=_fraction(o_count, matching),
        C_fraction=_fraction(c_count, matching),
        current_endpoint_model_recall=_fraction(b_count, matching),
        mean_matching_segment_length=mean_length,
        min_matching_segment_length=None if matching == 0 else length_min,
        p50_matching_segment_length=length_percentiles["p50"],
        p90_matching_segment_length=length_percentiles["p90"],
        p95_matching_segment_length=length_percentiles["p95"],
        p99_matching_segment_length=length_percentiles["p99"],
        max_matching_segment_length=None if matching == 0 else length_max,
        window_width_over_mean_matching_segment_length=_safe_ratio(width, mean_length),
        window_height_over_mean_matching_segment_length=_safe_ratio(height, mean_length),
        sqrt_window_area_over_mean_matching_segment_length=_safe_ratio(
            math.sqrt(area), mean_length
        ),
    )


def summarize_results(
    results: Sequence[SpatialIntersectionStats],
    *,
    bucket_edges: Sequence[float],
) -> dict[str, Any]:
    total_m = sum(result.M_Q for result in results)
    total_b = sum(result.B_Q for result in results)
    total_o = sum(result.O_Q for result in results)
    total_c = sum(result.C_Q for result in results)
    return {
        "spatial_queries_analyzed": len(results),
        "total_matching_segments": int(total_m),
        "total_B_Q": int(total_b),
        "total_O_Q": int(total_o),
        "total_C_Q": int(total_c),
        "ratio_percentiles": {
            "B_Q_over_M_Q": _percentile_summary([r.B_fraction for r in results]),
            "O_Q_over_M_Q": _percentile_summary([r.O_fraction for r in results]),
            "C_Q_over_M_Q": _percentile_summary([r.C_fraction for r in results]),
            "current_endpoint_model_recall": _percentile_summary(
                [r.current_endpoint_model_recall for r in results]
            ),
        },
        "matching_segment_length_percentiles_by_query": {
            "mean": _percentile_summary([r.mean_matching_segment_length for r in results]),
            "p50": _percentile_summary([r.p50_matching_segment_length for r in results]),
            "p90": _percentile_summary([r.p90_matching_segment_length for r in results]),
            "p95": _percentile_summary([r.p95_matching_segment_length for r in results]),
            "p99": _percentile_summary([r.p99_matching_segment_length for r in results]),
        },
        "normalized_window_size_percentiles": {
            "width_over_mean_matching_segment_length": _percentile_summary(
                [r.window_width_over_mean_matching_segment_length for r in results]
            ),
            "height_over_mean_matching_segment_length": _percentile_summary(
                [r.window_height_over_mean_matching_segment_length for r in results]
            ),
            "sqrt_area_over_mean_matching_segment_length": _percentile_summary(
                [r.sqrt_window_area_over_mean_matching_segment_length for r in results]
            ),
        },
        "cardinality_weighted_global_fractions": {
            "B_fraction": _fraction(total_b, total_m),
            "O_fraction": _fraction(total_o, total_m),
            "C_fraction": _fraction(total_c, total_m),
        },
        "normalized_window_size_buckets": _bucket_summary(results, bucket_edges),
        "sanity_checks": {
            "all_partition_counts_match": all(
                result.B_Q + result.O_Q + result.C_Q == result.M_Q for result in results
            ),
            "all_counts_in_range": all(
                0 <= result.B_Q <= result.M_Q
                and 0 <= result.O_Q <= result.M_Q
                and 0 <= result.C_Q <= result.M_Q
                for result in results
            ),
        },
    }


def _load_spatial_queries(paths: Sequence[Path]) -> Iterator[SpatialRectangleQuery]:
    for path in paths:
        files = sorted(path.glob("**/queries.jsonl")) if path.is_dir() else [path]
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    category = record.get("category") or {}
                    category_key = category.get("key")
                    query_id = str(record.get("query_id") or f"{file_path}:{line_number}")
                    for predicate_index, predicate in enumerate(record.get("predicates", ())):
                        if predicate.get("mode") not in {"spatial_intersects", "spatial_unbounded"}:
                            continue
                        yield SpatialRectangleQuery(
                            query_id=query_id,
                            predicate_index=int(predicate_index),
                            table=str(predicate.get("table", "")),
                            attribute=str(predicate.get("attribute", "")),
                            mode=str(predicate.get("mode", "")),
                            min_x=float(predicate["min_x"]),
                            min_y=float(predicate["min_y"]),
                            max_x=float(predicate["max_x"]),
                            max_y=float(predicate["max_y"]),
                            category_key=None if category_key is None else str(category_key),
                        )


def _segments_from_index(path: Path) -> dict[str, np.ndarray]:
    index = CompactTrajectorySegmentIndex.from_directory(path)
    return {
        "s_x": index.s_x,
        "s_y": index.s_y,
        "e_x": index.e_x,
        "e_y": index.e_y,
    }


def _segments_from_tsv(path: Path) -> dict[str, np.ndarray]:
    s_x: list[float] = []
    s_y: list[float] = []
    e_x: list[float] = []
    e_y: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            row = line.split("\t")
            if len(row) < 6:
                raise SystemExit(
                    f"bad segments.tsv row {line_number}: expected at least 6 columns"
                )
            s_x.append(float(row[2]))
            s_y.append(float(row[3]))
            e_x.append(float(row[4]))
            e_y.append(float(row[5]))
    if not s_x:
        raise SystemExit(f"no segment rows found in {path}")
    return {
        "s_x": np.asarray(s_x, dtype=float),
        "s_y": np.asarray(s_y, dtype=float),
        "e_x": np.asarray(e_x, dtype=float),
        "e_y": np.asarray(e_y, dtype=float),
    }


def _validate_stats(stats: SpatialIntersectionStats) -> None:
    if stats.B_Q + stats.O_Q + stats.C_Q != stats.M_Q:
        raise ValueError(f"partition sanity check failed for {stats.query_id}")
    for name in ("B_Q", "O_Q", "C_Q"):
        value = int(getattr(stats, name))
        if value < 0 or value > stats.M_Q:
            raise ValueError(f"count sanity check failed for {stats.query_id}: {name}={value}")


def _write_csv(path: Path, results: Sequence[SpatialIntersectionStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("", encoding="utf-8")
        return
    fields = list(asdict(results[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _bucket_summary(
    results: Sequence[SpatialIntersectionStats],
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    buckets = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        members = [
            result
            for result in results
            if result.sqrt_window_area_over_mean_matching_segment_length is not None
            and lower <= result.sqrt_window_area_over_mean_matching_segment_length < upper
        ]
        total_m = sum(result.M_Q for result in members)
        buckets.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "query_count": len(members),
                "total_matching_segments": int(total_m),
                "weighted_O_fraction": _fraction(sum(r.O_Q for r in members), total_m),
                "weighted_C_fraction": _fraction(sum(r.C_Q for r in members), total_m),
                "mean_O_fraction": _mean([r.O_fraction for r in members]),
                "mean_C_fraction": _mean([r.C_fraction for r in members]),
                "p50_O_fraction": _percentile([r.O_fraction for r in members], 50),
                "p50_C_fraction": _percentile([r.C_fraction for r in members], 50),
            }
        )
    return buckets


def _parse_bucket_edges(value: str) -> tuple[float, ...]:
    edges = []
    for part in value.split(","):
        text = part.strip().lower()
        edges.append(math.inf if text in {"inf", "+inf", "infinity"} else float(text))
    if len(edges) < 2 or any(left >= right for left, right in zip(edges[:-1], edges[1:])):
        raise SystemExit("--bucket-edges must be a strictly increasing comma-separated list")
    return tuple(edges)


def _percentile_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "count": len(filtered),
        "p50": _percentile(filtered, 50),
        "p90": _percentile(filtered, 90),
        "p95": _percentile(filtered, 95),
        "p99": _percentile(filtered, 99),
        "max": None if not filtered else float(max(filtered)),
    }


def _percentile(values: Iterable[float | None], percentile: float) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not filtered:
        return None
    return float(np.percentile(np.asarray(filtered, dtype=float), percentile))


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not filtered:
        return None
    return float(np.mean(np.asarray(filtered, dtype=float)))


def _length_percentiles(chunks: Sequence[np.ndarray]) -> dict[str, float | None]:
    if not chunks:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    values = np.concatenate(chunks)
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _safe_ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
