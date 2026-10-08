#!/usr/bin/env python3
"""Replace only true-zero POL workload rows with positive rejection samples.

The input benchmark remains untouched.  Its positive rows are copied as-is;
each row whose exact database join cardinality is zero is replaced by a fresh
draw from the same category-local generator distribution, accepted only when
its final (segment-coupled) SQL has positive cardinality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

try:
    from .merge_evaluated_slices import validate_segment_coupled_semantics
    from .query_generator import (
        Category,
        LiveCenterCache,
        QueryExecutor,
        QueryGenerator,
        load_config,
    )
    from .rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row
except ImportError:  # Direct execution from query_generation/ on the cluster.
    from merge_evaluated_slices import validate_segment_coupled_semantics
    from query_generator import (
        Category,
        LiveCenterCache,
        QueryExecutor,
        QueryGenerator,
        load_config,
    )
    from rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row


REPLACEMENT_NAME = "positive_rejection_sampling_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sql_hash(row: dict[str, Any]) -> str:
    normalized = " ".join(str(row["sql"]).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def replacement_seed(base_seed: int, query_id: str, attempt: int) -> int:
    payload = f"{base_seed}:{query_id}:{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def is_true_zero(row: dict[str, Any]) -> bool:
    value = row.get("join_cardinality")
    if value is None:
        raise ValueError(f"{row.get('query_id')} has no join cardinality")
    return int(value) == 0


def source_category(row: dict[str, Any]) -> Category:
    correction = row.get("semantic_correction")
    category = correction.get("original", {}).get("category") if correction else None
    category = category or row["category"]
    return Category.parse(str(category["key"]))


def query_ordinal(query_id: str, fallback: int) -> int:
    if query_id.startswith("q") and query_id[1:].isdigit():
        return int(query_id[1:])
    return fallback + 1


def normalize_final_semantics(candidate: dict[str, Any], source_workload: str) -> dict[str, Any]:
    if needs_segment_correction(candidate):
        return rewrite_row(candidate, source_workload)
    return candidate


def evaluate_candidate(candidate: dict[str, Any], executor: QueryExecutor) -> None:
    started = time.perf_counter()
    candidate["join_cardinality"] = int(executor.scalar(candidate["sql"]))
    candidate["join_evaluation_seconds"] = time.perf_counter() - started
    if candidate["category"]["relation"] != "multi":
        candidate["entity_cardinality"] = None
        candidate["entity_evaluation_seconds"] = None
        return
    entity_sql = candidate.get("entity_sql")
    if not entity_sql:
        raise ValueError(f"{candidate['query_id']} is multi-relation but has no entity SQL")
    started = time.perf_counter()
    candidate["entity_cardinality"] = int(executor.scalar(entity_sql))
    candidate["entity_evaluation_seconds"] = time.perf_counter() - started
    if candidate["entity_cardinality"] > candidate["join_cardinality"]:
        raise ValueError(f"{candidate['query_id']} has entity cardinality above join cardinality")


def replacement_record(
    target: dict[str, Any],
    *,
    config: dict[str, Any],
    executor: QueryExecutor,
    live_centers: LiveCenterCache,
    base_seed: int,
    max_attempts: int,
    source_workload: str,
    used_sql_hashes: set[str],
    source_index: int,
) -> tuple[dict[str, Any], int]:
    target_key = str(target["category"]["key"])
    category = source_category(target)
    target_query_id = str(target["query_id"])
    for attempt in range(1, max_attempts + 1):
        generator = QueryGenerator(
            config,
            seed=replacement_seed(base_seed, target_query_id, attempt),
            executor=executor,
            evaluate_cardinalities=False,
            live_centers=live_centers,
        )
        candidate = generator.generate_one(
            category,
            query_ordinal(target_query_id, source_index),
            int(target["category"].get("category_index", source_index)),
        )
        candidate = normalize_final_semantics(candidate, source_workload)
        if candidate["category"]["key"] != target_key:
            continue
        evaluate_candidate(candidate, executor)
        candidate_hash = canonical_sql_hash(candidate)
        if candidate["join_cardinality"] <= 0 or candidate_hash in used_sql_hashes:
            continue
        candidate["query_id"] = target_query_id
        candidate["category"]["category_index"] = target["category"].get("category_index")
        candidate["nonzero_replacement"] = {
            "name": REPLACEMENT_NAME,
            "source_row_index": source_index,
            "replacement_seed": replacement_seed(base_seed, target_query_id, attempt),
            "attempt": attempt,
            "replaced_query_id": target_query_id,
            "replaced_join_cardinality": int(target["join_cardinality"]),
            "replaced_entity_cardinality": target.get("entity_cardinality"),
            "replaced_sql_sha256": canonical_sql_hash(target),
            "source_category": category.key,
            "final_category": target_key,
        }
        return candidate, attempt
    raise RuntimeError(
        f"could not find a positive replacement for {target_query_id} after {max_attempts} attempts"
    )


def validate_output(rows: Sequence[dict[str, Any]], source_rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) != len(source_rows):
        raise ValueError("replacement output changed the workload row count")
    ids = [str(row["query_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("replacement output contains duplicate query IDs")
    if Counter(row["category"]["key"] for row in rows) != Counter(
        row["category"]["key"] for row in source_rows
    ):
        raise ValueError("replacement output changed final category counts")
    for row in rows:
        if int(row["join_cardinality"]) <= 0:
            raise ValueError(f"{row['query_id']} is not strictly positive")
        if row["category"]["relation"] == "multi":
            entity = row.get("entity_cardinality")
            if entity is None or int(entity) <= 0:
                raise ValueError(f"{row['query_id']} lacks a positive distinct-trip truth")
            if int(entity) > int(row["join_cardinality"]):
                raise ValueError(f"{row['query_id']} has entity cardinality above join cardinality")
        validate_segment_coupled_semantics(row)


def replace_true_zeros(
    source_rows: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    executor: QueryExecutor,
    base_seed: int,
    max_attempts: int,
    source_workload: str,
    sample_cache_size: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    output = [copy.deepcopy(row) for row in source_rows]
    used_sql_hashes = {
        canonical_sql_hash(row) for row in source_rows if not is_true_zero(row)
    }
    live_centers = LiveCenterCache(config, executor, sample_cache_size)
    attempts: list[int] = []
    for index, target in enumerate(source_rows):
        if not is_true_zero(target):
            continue
        replacement, replacement_attempts = replacement_record(
            target,
            config=config,
            executor=executor,
            live_centers=live_centers,
            base_seed=base_seed,
            max_attempts=max_attempts,
            source_workload=source_workload,
            used_sql_hashes=used_sql_hashes,
            source_index=index,
        )
        used_sql_hashes.add(canonical_sql_hash(replacement))
        output[index] = replacement
        attempts.append(replacement_attempts)
    validate_output(output, source_rows)
    return output, attempts


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    source_path: Path,
    source_rows: Sequence[dict[str, Any]],
    seed: int,
    attempts: Sequence[int],
) -> dict[str, Any]:
    replacements = [row for row in rows if row.get("nonzero_replacement")]
    return {
        "name": REPLACEMENT_NAME,
        "source_workload": str(source_path),
        "source_workload_sha256": file_sha256(source_path),
        "seed": seed,
        "rows": len(rows),
        "unique_query_ids": len({row["query_id"] for row in rows}),
        "retained_positive_rows": len(source_rows) - len(replacements),
        "replaced_true_zero_rows": len(replacements),
        "remaining_true_zero_rows": sum(is_true_zero(row) for row in rows),
        "category_counts": dict(sorted(Counter(row["category"]["key"] for row in rows).items())),
        "distinct_trajectory_truth_rows": sum(
            row.get("entity_cardinality") is not None for row in rows
        ),
        "replacement_attempts": {
            "total": sum(attempts),
            "mean": mean(attempts) if attempts else 0.0,
            "max": max(attempts, default=0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace true-zero POL workload rows with positive category-local rejection samples."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--sample-cache-size", type=int, default=2048)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user")
    args = parser.parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be positive")
    source_path = Path(args.input)
    source_rows = load_jsonl(source_path)
    if not source_rows:
        raise SystemExit("input workload is empty")
    config = load_config(Path(args.config))
    with QueryExecutor(
        host=args.host, port=args.port, dbname=args.dbname, user=args.user
    ) as executor:
        rows, attempts = replace_true_zeros(
            source_rows,
            config=config,
            executor=executor,
            base_seed=args.seed,
            max_attempts=args.max_attempts,
            source_workload=str(source_path),
            sample_cache_size=args.sample_cache_size,
        )
    write_jsonl(Path(args.output), rows)
    summary = build_summary(
        rows,
        source_path=source_path,
        source_rows=source_rows,
        seed=args.seed,
        attempts=attempts,
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
