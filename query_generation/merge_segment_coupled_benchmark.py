#!/usr/bin/env python3
"""Replace corrected POL rows in the original 8,000-query workload."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .rewrite_segment_coupled_queries import needs_segment_correction
except ImportError:  # Direct execution from query_generation/ on the cluster.
    from rewrite_segment_coupled_queries import needs_segment_correction


CORRECTION_NAME = "segment_coupled_spatio_temporal_v1"


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def load_corrected(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    corrected: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            query_id = str(row["query_id"])
            if query_id in corrected:
                raise SystemExit(f"duplicate corrected query_id: {query_id}")
            correction = row.get("semantic_correction", {})
            if correction.get("name") != CORRECTION_NAME:
                raise SystemExit(f"{query_id} is not a {CORRECTION_NAME} row")
            if row.get("join_cardinality") is None or row.get("entity_cardinality") is None:
                raise SystemExit(f"{query_id} has missing evaluated cardinalities")
            if int(row["entity_cardinality"]) > int(row["join_cardinality"]):
                raise SystemExit(f"{query_id} has entity cardinality greater than join cardinality")
            corrected[query_id] = row
    return corrected


def merge_rows(original: List[Dict[str, Any]], corrected: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    original_ids = [str(row["query_id"]) for row in original]
    if len(set(original_ids)) != len(original_ids):
        raise SystemExit("original workload has duplicate query IDs")
    expected_ids = {str(row["query_id"]) for row in original if needs_segment_correction(row)}
    corrected_ids = set(corrected)
    missing = sorted(expected_ids - corrected_ids)
    unexpected = sorted(corrected_ids - expected_ids)
    if missing or unexpected:
        message = []
        if missing:
            message.append(f"missing corrected IDs ({len(missing)}): {', '.join(missing[:10])}")
        if unexpected:
            message.append(f"unexpected corrected IDs ({len(unexpected)}): {', '.join(unexpected[:10])}")
        raise SystemExit("; ".join(message))
    return [corrected.get(str(row["query_id"]), row) for row in original]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge exact segment-coupled rows into a POL workload.")
    parser.add_argument("--original", required=True)
    parser.add_argument("--corrected", action="append", required=True, help="Evaluated corrected JSONL shard; repeat for each shard.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    original_path = Path(args.original)
    corrected_paths = [Path(value) for value in args.corrected]
    original = list(load_jsonl(original_path))
    corrected = load_corrected(corrected_paths)
    merged = merge_rows(original, corrected)
    write_jsonl(Path(args.output), merged)
    summary = {
        "rows": len(merged),
        "replaced_segment_coupled_rows": len(corrected),
        "unchanged_rows": len(merged) - len(corrected),
        "semantic_correction": CORRECTION_NAME,
        "original": str(original_path),
        "corrected_shards": [str(path) for path in corrected_paths],
        "category_counts": dict(sorted(Counter(row["category"]["key"] for row in merged).items())),
        "distinct_trajectory_truth_rows": sum(row.get("entity_cardinality") is not None for row in merged),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
