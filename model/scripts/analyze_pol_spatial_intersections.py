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
    category_dimension: str | None
    category_interval: str | None
    category_relation: str | None
    center_source: str | None
    range_source: str | None
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
    category_dimension: str | None
    category_interval: str | None
    category_relation: str | None
    center_source: str | None
    range_source: str | None
    category_key: str | None
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    window_width: float
    window_height: float
    window_area: float
    M_Q: int
    M_MBR_Q: int
    B_Q: int
    O_Q: int
    C_Q: int
    MBR_false_positives: int
    B_fraction: float | None
    O_fraction: float | None
    C_fraction: float | None
    current_endpoint_model_recall: float | None
    MBR_precision: float | None
    MBR_overestimation: float | None
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
    parser.add_argument(
        "--bounded-range-breakdown-csv",
        default=None,
        help=(
            "Optional CSV for bounded segment-range groups by "
            "interval x width source x center source."
        ),
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--include-other-spatial-predicates",
        action="store_true",
        help=(
            "Also analyze spatial predicates outside segments:segment_geom. "
            "By default only segment-geometry predicates are analyzed."
        ),
    )
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

    queries = list(
        _load_spatial_queries(
            [Path(value) for value in args.queries],
            include_other_spatial_predicates=bool(args.include_other_spatial_predicates),
        )
    )
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
    if args.bounded_range_breakdown_csv is not None:
        _write_bounded_range_breakdown_csv(
            Path(args.bounded_range_breakdown_csv), results
        )
    summary = summarize_results(
        results,
        bucket_edges=_parse_bucket_edges(args.bucket_edges),
        include_other_spatial_predicates=bool(args.include_other_spatial_predicates),
    )
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"spatial_queries_analyzed={len(results)}")
    print(f"output_jsonl={output_jsonl}")
    if args.output_csv is not None:
        print(f"output_csv={args.output_csv}")
    if args.bounded_range_breakdown_csv is not None:
        print(f"bounded_range_breakdown_csv={args.bounded_range_breakdown_csv}")
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
    mbr_count = 0
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
        seg_min_x = np.minimum(sx_chunk, ex_chunk)
        seg_max_x = np.maximum(sx_chunk, ex_chunk)
        seg_min_y = np.minimum(sy_chunk, ey_chunk)
        seg_max_y = np.maximum(sy_chunk, ey_chunk)
        mbr_overlaps = (
            (seg_min_x <= max_x)
            & (seg_max_x >= min_x)
            & (seg_min_y <= max_y)
            & (seg_max_y >= min_y)
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
        mbr_count += int(np.count_nonzero(mbr_overlaps))
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
    mbr_false_positives = mbr_count - matching
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
        category_dimension=query.category_dimension,
        category_interval=query.category_interval,
        category_relation=query.category_relation,
        center_source=query.center_source,
        range_source=query.range_source,
        category_key=query.category_key,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        window_width=width,
        window_height=height,
        window_area=area,
        M_Q=matching,
        M_MBR_Q=mbr_count,
        B_Q=b_count,
        O_Q=o_count,
        C_Q=c_count,
        MBR_false_positives=mbr_false_positives,
        B_fraction=_fraction(b_count, matching),
        O_fraction=_fraction(o_count, matching),
        C_fraction=_fraction(c_count, matching),
        current_endpoint_model_recall=_fraction(b_count, matching),
        MBR_precision=_fraction(matching, mbr_count),
        MBR_overestimation=_fraction(mbr_count, matching),
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
    include_other_spatial_predicates: bool = False,
) -> dict[str, Any]:
    total_m = sum(result.M_Q for result in results)
    total_b = sum(result.B_Q for result in results)
    total_o = sum(result.O_Q for result in results)
    total_c = sum(result.C_Q for result in results)
    total_mbr = sum(result.M_MBR_Q for result in results)
    total_mbr_false_positives = sum(result.MBR_false_positives for result in results)
    return {
        "spatial_queries_analyzed": len(results),
        "predicate_filter": {
            "default_segments_segment_geom_only": not include_other_spatial_predicates,
            "include_other_spatial_predicates": include_other_spatial_predicates,
        },
        "total_matching_segments": int(total_m),
        "total_mbr_candidate_segments": int(total_mbr),
        "total_mbr_false_positives": int(total_mbr_false_positives),
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
            "MBR_precision": _percentile_summary([r.MBR_precision for r in results]),
            "MBR_overestimation": _percentile_summary(
                [r.MBR_overestimation for r in results]
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
            "description": (
                "Cardinality-weighted fractions are computed as sum_Q count_Q / "
                "sum_Q M_Q and are therefore different from query-level percentile "
                "statistics over count_Q / M_Q."
            ),
            "B_fraction": _fraction(total_b, total_m),
            "O_fraction": _fraction(total_o, total_m),
            "C_fraction": _fraction(total_c, total_m),
            "MBR_precision": _fraction(total_m, total_mbr),
            "MBR_overestimation": _fraction(total_mbr, total_m),
            "MBR_false_positive_fraction": _fraction(total_mbr_false_positives, total_mbr),
        },
        "normalized_window_size_buckets": _bucket_summary(results, bucket_edges),
        "normalized_window_size_buckets_by_interval": {
            label: _bucket_summary(members, bucket_edges)
            for label, members in _partition(
                results, lambda r: _label(r.category_interval)
            ).items()
        },
        "normalized_window_size_buckets_for_range_queries": _bucket_summary(
            [r for r in results if r.category_interval == "range"],
            bucket_edges,
        ),
        "stratified_statistics": {
            "interval": _grouped_summaries(results, lambda r: _label(r.category_interval)),
            "center_source": _grouped_summaries(results, lambda r: _label(r.center_source)),
            "range_source": _grouped_summaries(results, lambda r: _label(r.range_source)),
            "category_dimension": _grouped_summaries(
                results, lambda r: _label(r.category_dimension)
            ),
            "relation": _grouped_summaries(results, lambda r: _label(r.category_relation)),
            "spatial_predicate_mode": _grouped_summaries(results, lambda r: _label(r.mode)),
            "interval_x_width_source_x_center_source": _grouped_summaries(
                results,
                lambda r: _interaction_label(
                    r.category_interval,
                    r.range_source,
                    r.center_source,
                ),
            ),
            "bounded_range_interval_x_width_source_x_center_source": _grouped_summaries(
                [
                    result
                    for result in results
                    if result.mode == "spatial_intersects"
                    and result.category_interval == "range"
                ],
                lambda r: _interaction_label(
                    r.category_interval,
                    r.range_source,
                    r.center_source,
                ),
            ),
        },
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
            "all_mbr_counts_cover_true_matches": all(
                result.M_Q <= result.M_MBR_Q for result in results
            ),
            "all_mbr_false_positives_nonnegative": all(
                result.MBR_false_positives >= 0 for result in results
            ),
        },
    }


def _load_spatial_queries(
    paths: Sequence[Path],
    *,
    include_other_spatial_predicates: bool = False,
) -> Iterator[SpatialRectangleQuery]:
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
                        table = str(predicate.get("table", ""))
                        attribute = str(predicate.get("attribute", ""))
                        if (
                            not include_other_spatial_predicates
                            and (table, attribute) != ("segments", "segment_geom")
                        ):
                            continue
                        yield SpatialRectangleQuery(
                            query_id=query_id,
                            predicate_index=int(predicate_index),
                            table=table,
                            attribute=attribute,
                            mode=str(predicate.get("mode", "")),
                            category_dimension=_optional_str(category.get("dimension")),
                            category_interval=_optional_str(category.get("interval")),
                            category_relation=_optional_str(category.get("relation")),
                            center_source=_optional_str(predicate.get("center_source")),
                            range_source=_optional_str(predicate.get("range_source")),
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
    if stats.M_Q > stats.M_MBR_Q:
        raise ValueError(
            f"MBR sanity check failed for {stats.query_id}: "
            f"M_Q={stats.M_Q} > M_MBR_Q={stats.M_MBR_Q}"
        )
    if stats.MBR_false_positives < 0:
        raise ValueError(
            f"MBR false-positive sanity check failed for {stats.query_id}: "
            f"{stats.MBR_false_positives}"
        )
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


def _write_bounded_range_breakdown_csv(
    path: Path,
    results: Sequence[SpatialIntersectionStats],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "interval",
        "width_source",
        "center_source",
        "query_count",
        "sum_M_Q",
        "sum_M_MBR_Q",
        "sum_B_Q",
        "sum_O_Q",
        "sum_C_Q",
        "sum_MBR_false_positives",
        "B_weighted",
        "O_weighted",
        "C_weighted",
        "MBR_precision_weighted",
        "MBR_overestimation_weighted",
        "MBR_false_positive_fraction_weighted",
        "median_M_Q",
        "n_C_fraction_gt_0_5",
        "median_M_Q_C_fraction_gt_0_5",
        "n_C_fraction_gt_0_9",
        "median_M_Q_C_fraction_gt_0_9",
        "B_p50",
        "B_p90",
        "B_p95",
        "B_p99",
        "B_max",
        "O_p50",
        "O_p90",
        "O_p95",
        "O_p99",
        "O_max",
        "C_p50",
        "C_p90",
        "C_p95",
        "C_p99",
        "C_max",
        "MBR_precision_p50",
        "MBR_precision_p90",
        "MBR_precision_p95",
        "MBR_precision_p99",
        "MBR_precision_max",
        "MBR_overestimation_p50",
        "MBR_overestimation_p90",
        "MBR_overestimation_p95",
        "MBR_overestimation_p99",
        "MBR_overestimation_max",
    ]
    rows = bounded_range_breakdown_rows(results)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bounded_range_breakdown_rows(
    results: Sequence[SpatialIntersectionStats],
) -> list[dict[str, Any]]:
    bounded = [
        result
        for result in results
        if result.table == "segments"
        and result.attribute == "segment_geom"
        and result.mode == "spatial_intersects"
        and result.category_interval == "range"
    ]
    rows = []
    groups = _partition(
        bounded,
        lambda r: _interaction_label(
            r.category_interval,
            r.range_source,
            r.center_source,
        ),
    )
    for label, members in sorted(groups.items()):
        interval, width_source, center_source = label.split(" x ", maxsplit=2)
        total_m = sum(result.M_Q for result in members)
        total_mbr = sum(result.M_MBR_Q for result in members)
        total_b = sum(result.B_Q for result in members)
        total_o = sum(result.O_Q for result in members)
        total_c = sum(result.C_Q for result in members)
        total_mbr_false_positives = sum(result.MBR_false_positives for result in members)
        c_gt_0_5 = [
            result.M_Q
            for result in members
            if result.C_fraction is not None and result.C_fraction > 0.5
        ]
        c_gt_0_9 = [
            result.M_Q
            for result in members
            if result.C_fraction is not None and result.C_fraction > 0.9
        ]
        rows.append(
            {
                "interval": interval,
                "width_source": width_source,
                "center_source": center_source,
                "query_count": len(members),
                "sum_M_Q": int(total_m),
                "sum_M_MBR_Q": int(total_mbr),
                "sum_B_Q": int(total_b),
                "sum_O_Q": int(total_o),
                "sum_C_Q": int(total_c),
                "sum_MBR_false_positives": int(total_mbr_false_positives),
                "B_weighted": _fraction(total_b, total_m),
                "O_weighted": _fraction(total_o, total_m),
                "C_weighted": _fraction(total_c, total_m),
                "MBR_precision_weighted": _fraction(total_m, total_mbr),
                "MBR_overestimation_weighted": _fraction(total_mbr, total_m),
                "MBR_false_positive_fraction_weighted": _fraction(
                    total_mbr_false_positives, total_mbr
                ),
                "median_M_Q": _percentile([r.M_Q for r in members], 50),
                "n_C_fraction_gt_0_5": len(c_gt_0_5),
                "median_M_Q_C_fraction_gt_0_5": _percentile(c_gt_0_5, 50),
                "n_C_fraction_gt_0_9": len(c_gt_0_9),
                "median_M_Q_C_fraction_gt_0_9": _percentile(c_gt_0_9, 50),
                "B_p50": _percentile([r.B_fraction for r in members], 50),
                "B_p90": _percentile([r.B_fraction for r in members], 90),
                "B_p95": _percentile([r.B_fraction for r in members], 95),
                "B_p99": _percentile([r.B_fraction for r in members], 99),
                "B_max": _max_or_none([r.B_fraction for r in members]),
                "O_p50": _percentile([r.O_fraction for r in members], 50),
                "O_p90": _percentile([r.O_fraction for r in members], 90),
                "O_p95": _percentile([r.O_fraction for r in members], 95),
                "O_p99": _percentile([r.O_fraction for r in members], 99),
                "O_max": _max_or_none([r.O_fraction for r in members]),
                "C_p50": _percentile([r.C_fraction for r in members], 50),
                "C_p90": _percentile([r.C_fraction for r in members], 90),
                "C_p95": _percentile([r.C_fraction for r in members], 95),
                "C_p99": _percentile([r.C_fraction for r in members], 99),
                "C_max": _max_or_none([r.C_fraction for r in members]),
                "MBR_precision_p50": _percentile(
                    [r.MBR_precision for r in members], 50
                ),
                "MBR_precision_p90": _percentile(
                    [r.MBR_precision for r in members], 90
                ),
                "MBR_precision_p95": _percentile(
                    [r.MBR_precision for r in members], 95
                ),
                "MBR_precision_p99": _percentile(
                    [r.MBR_precision for r in members], 99
                ),
                "MBR_precision_max": _max_or_none(
                    [r.MBR_precision for r in members]
                ),
                "MBR_overestimation_p50": _percentile(
                    [r.MBR_overestimation for r in members], 50
                ),
                "MBR_overestimation_p90": _percentile(
                    [r.MBR_overestimation for r in members], 90
                ),
                "MBR_overestimation_p95": _percentile(
                    [r.MBR_overestimation for r in members], 95
                ),
                "MBR_overestimation_p99": _percentile(
                    [r.MBR_overestimation for r in members], 99
                ),
                "MBR_overestimation_max": _max_or_none(
                    [r.MBR_overestimation for r in members]
                ),
            }
        )
    return rows


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
        total_mbr = sum(result.M_MBR_Q for result in members)
        total_mbr_false_positives = sum(result.MBR_false_positives for result in members)
        buckets.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "query_count": len(members),
                "total_matching_segments": int(total_m),
                "total_mbr_candidate_segments": int(total_mbr),
                "weighted_MBR_precision": _fraction(total_m, total_mbr),
                "weighted_MBR_overestimation": _fraction(total_mbr, total_m),
                "weighted_MBR_false_positive_fraction": _fraction(
                    total_mbr_false_positives, total_mbr
                ),
                "weighted_O_fraction": _fraction(sum(r.O_Q for r in members), total_m),
                "weighted_C_fraction": _fraction(sum(r.C_Q for r in members), total_m),
                "mean_O_fraction": _mean([r.O_fraction for r in members]),
                "mean_C_fraction": _mean([r.C_fraction for r in members]),
                "p50_O_fraction": _percentile([r.O_fraction for r in members], 50),
                "p50_C_fraction": _percentile([r.C_fraction for r in members], 50),
                "MBR_precision": _percentile_summary(
                    [r.MBR_precision for r in members]
                ),
                "MBR_overestimation": _percentile_summary(
                    [r.MBR_overestimation for r in members]
                ),
            }
        )
    return buckets


def _grouped_summaries(
    results: Sequence[SpatialIntersectionStats],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    return {
        label: _group_summary(members)
        for label, members in sorted(_partition(results, key_fn).items())
    }


def _group_summary(results: Sequence[SpatialIntersectionStats]) -> dict[str, Any]:
    total_m = sum(result.M_Q for result in results)
    total_mbr = sum(result.M_MBR_Q for result in results)
    total_b = sum(result.B_Q for result in results)
    total_o = sum(result.O_Q for result in results)
    total_c = sum(result.C_Q for result in results)
    total_mbr_false_positives = sum(result.MBR_false_positives for result in results)
    return {
        "query_count": len(results),
        "sum_M_Q": int(total_m),
        "sum_M_MBR_Q": int(total_mbr),
        "sum_B_Q": int(total_b),
        "sum_O_Q": int(total_o),
        "sum_C_Q": int(total_c),
        "sum_MBR_false_positives": int(total_mbr_false_positives),
        "cardinality_weighted_fractions": {
            "B_fraction": _fraction(total_b, total_m),
            "O_fraction": _fraction(total_o, total_m),
            "C_fraction": _fraction(total_c, total_m),
            "MBR_precision": _fraction(total_m, total_mbr),
            "MBR_overestimation": _fraction(total_mbr, total_m),
            "MBR_false_positive_fraction": _fraction(total_mbr_false_positives, total_mbr),
        },
        "ratio_percentiles": {
            "B_Q_over_M_Q": _percentile_summary([r.B_fraction for r in results]),
            "O_Q_over_M_Q": _percentile_summary([r.O_fraction for r in results]),
            "C_Q_over_M_Q": _percentile_summary([r.C_fraction for r in results]),
            "MBR_precision": _percentile_summary([r.MBR_precision for r in results]),
            "MBR_overestimation": _percentile_summary(
                [r.MBR_overestimation for r in results]
            ),
        },
    }


def _partition(
    results: Sequence[SpatialIntersectionStats],
    key_fn: Any,
) -> dict[str, list[SpatialIntersectionStats]]:
    groups: dict[str, list[SpatialIntersectionStats]] = {}
    for result in results:
        groups.setdefault(str(key_fn(result)), []).append(result)
    return groups


def _label(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    return str(value)


def _interaction_label(*values: Any) -> str:
    return " x ".join(_label(value) for value in values)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


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


def _max_or_none(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not filtered:
        return None
    return float(max(filtered))


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
