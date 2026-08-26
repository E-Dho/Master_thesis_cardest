#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from query_generator import DIMENSIONS, INTERVALS, RELATIONS


def category_order() -> List[str]:
    return [
        f"{dimension}.{interval}.{relation}"
        for dimension in DIMENSIONS
        for interval in INTERVALS
        for relation in RELATIONS
    ]


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("sources must use CATEGORY=PATH")
    category, path = value.split("=", 1)
    if category not in category_order():
        raise argparse.ArgumentTypeError(f"unknown category: {category}")
    return category, Path(path)


def selected_rows(category: str, paths: List[Path], queries_per_category: int) -> List[Dict[str, Any]]:
    rows_by_index: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            if row["category"]["key"] != category:
                continue
            category_index = int(row["category"]["category_index"])
            if category_index in rows_by_index:
                continue
            rows_by_index[category_index] = row

    if len(rows_by_index) < queries_per_category:
        raise SystemExit(
            f"{category} has {len(rows_by_index)} rows across sources, expected at least {queries_per_category}"
        )
    rows = [rows_by_index[index] for index in sorted(rows_by_index)[:queries_per_category]]
    expected_indexes = list(range(queries_per_category))
    indexes = [row["category"]["category_index"] for row in rows]
    if indexes != expected_indexes:
        raise SystemExit(f"{category} has non-contiguous indexes: first indexes are {indexes[:10]}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_summary(path: Path, rows: List[Dict[str, Any]], sources: Dict[str, List[Path]]) -> None:
    category_counts = Counter(row["category"]["key"] for row in rows)
    multi = [row for row in rows if row["category"]["relation"] == "multi"]
    bad_multi = [
        row["query_id"]
        for row in multi
        if row["entity_cardinality"] is None
        or row["join_cardinality"] is None
        or row["entity_cardinality"] > row["join_cardinality"]
    ]
    summary = {
        "rows": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "multi_rows": len(multi),
        "bad_multi_entity_gt_join": bad_multi,
        "min_join_cardinality": min(row["join_cardinality"] for row in rows),
        "max_join_cardinality": max(row["join_cardinality"] for row in rows),
        "sources": {category: [str(path) for path in paths] for category, paths in sorted(sources.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if bad_multi:
        raise SystemExit("entity_cardinality exceeded join_cardinality")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge category/chunk query-generation JSONL outputs.")
    parser.add_argument("--output", required=True, help="Merged JSONL output path.")
    parser.add_argument("--summary", required=True, help="Merged summary JSON path.")
    parser.add_argument("--queries-per-category", type=int, default=500)
    parser.add_argument("--source", action="append", type=parse_source, required=True, help="CATEGORY=PATH source mapping.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.queries_per_category <= 0:
        raise SystemExit("--queries-per-category must be positive")

    sources: Dict[str, List[Path]] = {}
    for category, path in args.source:
        sources.setdefault(category, []).append(path)
    expected = category_order()
    missing = [category for category in expected if category not in sources]
    if missing:
        raise SystemExit(f"missing sources for categories: {', '.join(missing)}")

    merged: List[Dict[str, Any]] = []
    for category in expected:
        for row in selected_rows(category, sources[category], args.queries_per_category):
            output_row = dict(row)
            output_row["source_query_id"] = row["query_id"]
            output_row["source_jsonl"] = [str(path) for path in sources[category]]
            output_row["query_id"] = f"q{len(merged) + 1:08d}"
            merged.append(output_row)

    write_jsonl(Path(args.output), merged)
    write_summary(Path(args.summary), merged, sources)
    print(f"merged {len(merged)} queries to {args.output}")


if __name__ == "__main__":
    main()
