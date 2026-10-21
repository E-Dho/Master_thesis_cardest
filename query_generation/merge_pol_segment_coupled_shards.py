#!/usr/bin/env python3
"""Validate and merge resumable POL segment-coupled generation shards."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .generate_pol_segment_coupled_benchmark import (
        build_summary,
        validate_record,
        validate_workload_shape,
    )
    from .query_generator import all_valid_categories, config_hash, load_config
except ImportError:
    from generate_pol_segment_coupled_benchmark import (
        build_summary,
        validate_record,
        validate_workload_shape,
    )
    from query_generator import all_valid_categories, config_hash, load_config


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from exc


def merge_shards(
    shard_dir: Path,
    *,
    config: dict[str, Any],
    queries_per_category: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    expected_hash = config_hash(config)
    merged: list[dict[str, Any]] = []
    source_files: list[str] = []
    for category in all_valid_categories(config):
        paths = sorted(shard_dir.glob(f"{category.key}.chunk-*.jsonl"))
        if not paths:
            raise ValueError(f"no shards found for {category.key} in {shard_dir}")
        by_index: dict[int, dict[str, Any]] = {}
        for path in paths:
            source_files.append(str(path))
            for row in load_jsonl(path):
                if row.get("config_hash") != expected_hash:
                    raise ValueError(f"{path} contains a row from a different config")
                if row.get("category", {}).get("key") != category.key:
                    raise ValueError(f"{path} contains a row from another category")
                validate_record(row)
                category_index = int(row["category"]["category_index"])
                if category_index in by_index:
                    raise ValueError(
                        f"duplicate {category.key} category_index {category_index}"
                    )
                by_index[category_index] = row
        expected_indexes = list(range(queries_per_category))
        if sorted(by_index) != expected_indexes:
            missing = sorted(set(expected_indexes) - set(by_index))
            extra = sorted(set(by_index) - set(expected_indexes))
            raise ValueError(
                f"{category.key} shard coverage mismatch; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        for category_index in expected_indexes:
            row = dict(by_index[category_index])
            row["source_query_id"] = row["query_id"]
            row["query_id"] = f"q{len(merged) + 1:08d}"
            merged.append(row)

    validate_workload_shape(
        merged, all_valid_categories(config), queries_per_category
    )
    return merged, source_files


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_merged_summary(
    rows: list[dict[str, Any]], config_path: str, sources: list[str]
) -> dict[str, Any]:
    summary = build_summary(rows, config_path)
    summary.update(
        {
            "shard_count": len(sources),
            "shards": sources,
            "join_evaluation_seconds": sum(
                float(row.get("join_evaluation_seconds") or 0.0) for row in rows
            ),
            "entity_evaluation_seconds": sum(
                float(row.get("entity_evaluation_seconds") or 0.0) for row in rows
            ),
            "category_counts": dict(
                sorted(Counter(row["category"]["key"] for row in rows).items())
            ),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--queries-per-category", type=int, required=True)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rows, sources = merge_shards(
        Path(args.shard_dir),
        config=config,
        queries_per_category=args.queries_per_category,
    )
    write_jsonl_atomic(Path(args.output), rows)
    summary = build_merged_summary(rows, args.config, sources)
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"merged {len(rows)} queries from {len(sources)} durable shards")


if __name__ == "__main__":
    main()
