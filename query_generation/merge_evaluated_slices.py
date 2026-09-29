#!/usr/bin/env python3
"""Validate and merge independently evaluated slices of a frozen workload."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_segment_coupled_semantics(row: dict[str, Any]) -> None:
    if row["category"]["dimension"] != "spatio_temporal":
        return
    dimensions = {predicate["dimension"] for predicate in row["predicates"]}
    if not {"spatial", "temporal"} <= dimensions:
        raise SystemExit(f"{row['query_id']} lacks a spatial or temporal predicate")
    for predicate in row["predicates"]:
        if predicate["dimension"] in {"spatial", "temporal"} and predicate["table"] != "segments":
            raise SystemExit(f"{row['query_id']} has non-segment {predicate['dimension']} predicate")


def merge_rows(original: list[dict[str, Any]], slice_paths: list[Path]) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for path in slice_paths:
        for row in load_jsonl(path):
            index = int(row.get("source_row_index", -1))
            if index in by_index:
                raise SystemExit(f"duplicate source_row_index {index}")
            if index < 0 or index >= len(original):
                raise SystemExit(f"out-of-range source_row_index {index}")
            expected = original[index]
            if row.get("query_id") != expected.get("query_id"):
                raise SystemExit(f"query_id mismatch at source_row_index {index}")
            if row.get("join_cardinality") is None:
                raise SystemExit(f"missing join cardinality for {row['query_id']}")
            if row["category"]["relation"] == "multi":
                if row.get("entity_cardinality") is None:
                    raise SystemExit(f"missing entity cardinality for {row['query_id']}")
                if int(row["entity_cardinality"]) > int(row["join_cardinality"]):
                    raise SystemExit(f"entity cardinality exceeds join cardinality for {row['query_id']}")
            validate_segment_coupled_semantics(row)
            by_index[index] = row
    missing = [index for index in range(len(original)) if index not in by_index]
    if missing:
        raise SystemExit(f"missing evaluated rows ({len(missing)}): {missing[:10]}")
    return [by_index[index] for index in range(len(original))]


def summary(rows: list[dict[str, Any]], original: Path, slice_paths: list[Path]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]["key"]].append(row)
    category_stats = {}
    for category, category_rows in sorted(by_category.items()):
        values = [float(row["join_cardinality"]) for row in category_rows]
        seconds = [float(row.get("join_evaluation_seconds") or 0) + float(row.get("entity_evaluation_seconds") or 0) for row in category_rows]
        category_stats[category] = {
            "rows": len(category_rows),
            "join_cardinality": {"min": min(values), "median": median(values), "max": max(values), "p90": percentile(values, 0.90), "p95": percentile(values, 0.95)},
            "evaluation_seconds": {"total": sum(seconds), "average_per_query": sum(seconds) / len(seconds), "p90_per_query": percentile(seconds, 0.90), "p95_per_query": percentile(seconds, 0.95)},
        }
    return {
        "rows": len(rows),
        "unique_query_ids": len({row["query_id"] for row in rows}),
        "category_counts": dict(sorted(Counter(row["category"]["key"] for row in rows).items())),
        "distinct_trajectory_truth_rows": sum(row.get("entity_cardinality") is not None for row in rows),
        "source_workload": str(original),
        "evaluated_slices": [str(path) for path in slice_paths],
        "category_statistics": category_stats,
    }


def validate_workload_shape(rows: list[dict[str, Any]], queries_per_category: int | None, expected_distinct_rows: int | None) -> None:
    if queries_per_category is not None:
        counts = Counter(row["category"]["key"] for row in rows)
        if len(counts) != 16 or any(count != queries_per_category for count in counts.values()):
            raise SystemExit(f"expected 16 categories with {queries_per_category} rows each, got {dict(sorted(counts.items()))}")
    if expected_distinct_rows is not None:
        observed = sum(row.get("entity_cardinality") is not None for row in rows)
        if observed != expected_distinct_rows:
            raise SystemExit(f"expected {expected_distinct_rows} distinct-trajectory truths, got {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge fully evaluated workload slices by source_row_index.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--slice", action="append", required=True, help="Evaluated JSONL slice; repeat for every slice.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--queries-per-category", type=int)
    parser.add_argument("--expected-distinct-rows", type=int)
    args = parser.parse_args()
    original = Path(args.original)
    slices = [Path(path) for path in args.slice]
    rows = merge_rows(list(load_jsonl(original)), slices)
    validate_workload_shape(rows, args.queries_per_category, args.expected_distinct_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    Path(args.summary).write_text(json.dumps(summary(rows, original, slices), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
