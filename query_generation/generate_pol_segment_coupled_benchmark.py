#!/usr/bin/env python3
"""Generate an evaluated, bucket-balanced POL segment-coupled workload.

The generic generator creates ordinary SQL records.  POL spatio-temporal rows
occasionally need same-segment normalization before their truth is measured.
This entry point applies that normalization before execution, preserving the
original workload bucket and avoiding the legacy post-hoc category relabeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .query_generator import (
        Category,
        LiveCenterCache,
        QueryExecutor,
        QueryGenerator,
        all_valid_categories,
        config_hash,
        load_config,
        parse_categories,
    )
    from .rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row
except ImportError:  # Direct execution from query_generation/ on the cluster.
    from query_generator import (
        Category,
        LiveCenterCache,
        QueryExecutor,
        QueryGenerator,
        all_valid_categories,
        config_hash,
        load_config,
        parse_categories,
    )
    from rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row


GENERATION_STATE_VERSION = 1


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # BeeGFS may reject directory fsync. The file itself is still synced.
        return


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        fsync_directory(path.parent)


def load_jsonl_recover_trailing_write(path: Path) -> list[dict[str, Any]]:
    """Load durable rows, discarding only an incomplete final JSON line."""

    if not path.exists():
        return []
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    valid_bytes = 0
    lines = payload.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped:
            valid_bytes += len(raw_line)
            continue
        try:
            rows.append(json.loads(stripped.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(lines) - 1:
                raise ValueError(f"invalid JSONL row {index + 1} in {path}") from exc
            with path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            return rows
        valid_bytes += len(raw_line)

    if payload and not payload.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def shard_seed(base_seed: int, category_key: str, start_index: int) -> int:
    payload = f"{base_seed}:{category_key}:{start_index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def state_binding(
    config: dict[str, Any],
    *,
    category: Category,
    queries_per_category: int,
    start_index: int,
    query_count: int,
    base_seed: int,
    sample_cache_size: int,
) -> dict[str, Any]:
    return {
        "config_hash": config_hash(config),
        "category": category.key,
        "queries_per_category": int(queries_per_category),
        "start_index": int(start_index),
        "query_count": int(query_count),
        "base_seed": int(base_seed),
        "shard_seed": shard_seed(base_seed, category.key, start_index),
        "sample_cache_size": int(sample_cache_size),
    }


def load_generation_state(
    path: Path, live_centers: LiveCenterCache, binding: dict[str, Any]
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", -1)) != GENERATION_STATE_VERSION:
        raise SystemExit(f"{path} has unsupported generation state version")
    stored = payload.get("binding") or {}
    mismatched = sorted(key for key, value in binding.items() if stored.get(key) != value)
    if mismatched:
        raise SystemExit(
            f"{path} belongs to a different {', '.join(mismatched)}; "
            "use a different shard path or remove both shard files"
        )
    live_centers.restore(payload.get("center_pools") or {})


def generation_state_payload(
    live_centers: LiveCenterCache, binding: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": GENERATION_STATE_VERSION,
        "binding": binding,
        "center_pools": live_centers.snapshot(),
        "center_pool_fingerprints": live_centers.fingerprint(),
    }


def evaluate_record(record: dict[str, Any], executor: QueryExecutor) -> None:
    started = time.perf_counter()
    record["join_cardinality"] = int(executor.scalar(record["sql"]))
    record["join_evaluation_seconds"] = time.perf_counter() - started
    record["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if record["category"]["relation"] != "multi":
        record["entity_cardinality"] = None
        record["entity_evaluation_seconds"] = None
        return
    entity_sql = record.get("entity_sql")
    if not entity_sql:
        raise ValueError(f"{record['query_id']} is a multi bucket without entity SQL")
    started = time.perf_counter()
    record["entity_cardinality"] = int(executor.scalar(entity_sql))
    record["entity_evaluation_seconds"] = time.perf_counter() - started
    if record["entity_cardinality"] > record["join_cardinality"]:
        raise ValueError(f"{record['query_id']} has entity cardinality above join cardinality")


def validate_record(record: dict[str, Any]) -> None:
    for predicate in record["predicates"]:
        if predicate.get("table") == "trips" and predicate.get("attribute") == "trip_geom":
            raise ValueError(f"{record['query_id']} contains prohibited trips.trip_geom")
    correction = record.get("semantic_correction") or {}
    original_category = (correction.get("original") or {}).get("category")
    if original_category is not None and original_category != record["category"]:
        raise ValueError(f"{record['query_id']} changed category during segment correction")
    if record["category"]["relation"] == "multi":
        if record.get("entity_cardinality") is None:
            raise ValueError(f"{record['query_id']} is missing distinct-trip truth")
    elif record.get("entity_cardinality") is not None:
        raise ValueError(f"{record['query_id']} has unexpected distinct-trip truth")


def generate_records(
    config: dict[str, Any],
    *,
    categories: Sequence[Category],
    queries_per_category: int,
    seed: int,
    executor: QueryExecutor,
    source_label: str,
) -> list[dict[str, Any]]:
    generator = QueryGenerator(
        config,
        seed=seed,
        executor=executor,
        evaluate_cardinalities=False,
    )
    records: list[dict[str, Any]] = []
    ordinal = 0
    for category in categories:
        for category_index in range(queries_per_category):
            ordinal += 1
            record = generator.generate_one(category, ordinal, category_index)
            if needs_segment_correction(record):
                record = rewrite_row(record, source_label)
            evaluate_record(record, executor)
            validate_record(record)
            records.append(record)
    validate_workload_shape(records, categories, queries_per_category)
    return records


def generate_shard_incrementally(
    config: dict[str, Any],
    *,
    category: Category,
    queries_per_category: int,
    start_index: int,
    query_count: int,
    base_seed: int,
    sample_cache_size: int,
    executor: QueryExecutor,
    source_label: str,
    output_path: Path,
    state_path: Path,
    progress_interval: int = 10,
) -> tuple[list[dict[str, Any]], int]:
    """Generate one category shard and durably checkpoint every evaluated row."""

    if start_index < 0 or query_count <= 0:
        raise ValueError("start_index must be non-negative and query_count must be positive")
    if start_index + query_count > queries_per_category:
        raise ValueError("shard extends beyond queries_per_category")

    binding = state_binding(
        config,
        category=category,
        queries_per_category=queries_per_category,
        start_index=start_index,
        query_count=query_count,
        base_seed=base_seed,
        sample_cache_size=sample_cache_size,
    )
    if output_path.exists() and output_path.stat().st_size and not state_path.exists():
        raise SystemExit(
            f"{output_path} exists without {state_path}; refusing an unverifiable resume"
        )

    live_centers: LiveCenterCache

    def persist_state() -> None:
        write_json_atomic(
            state_path, generation_state_payload(live_centers, binding)
        )

    live_centers = LiveCenterCache(
        config,
        executor,
        sample_cache_size,
        on_pool_cached=persist_state,
    )
    if state_path.exists():
        load_generation_state(state_path, live_centers, binding)
    else:
        persist_state()

    generator = QueryGenerator(
        config,
        seed=int(binding["shard_seed"]),
        executor=executor,
        evaluate_cardinalities=False,
        live_centers=live_centers,
    )
    completed = load_jsonl_recover_trailing_write(output_path)
    if len(completed) > query_count:
        raise SystemExit(f"{output_path} contains more rows than this shard permits")

    global_categories = all_valid_categories(config)
    category_positions = {value.key: index for index, value in enumerate(global_categories)}
    if category.key not in category_positions:
        raise ValueError(f"category {category.key} is not enabled by the config")
    category_position = category_positions[category.key]

    # Recreate cheap generator state for completed rows. Persisted center pools
    # make the replay deterministic without re-running cardinality queries.
    for offset, saved in enumerate(completed):
        category_index = start_index + offset
        query_ordinal = category_position * queries_per_category + category_index + 1
        generated = generator.generate_one(category, query_ordinal, category_index)
        if needs_segment_correction(generated):
            generated = rewrite_row(generated, source_label)
        expected = {
            key: generated.get(key)
            for key in ("query_id", "category", "tables", "predicates", "sql", "entity_sql")
        }
        observed = {key: saved.get(key) for key in expected}
        if observed != expected:
            raise SystemExit(
                f"{output_path} row {offset + 1} does not match deterministic replay"
            )
        validate_record(saved)

    resumed = len(completed)
    end_index = start_index + query_count
    for category_index in range(start_index + resumed, end_index):
        query_ordinal = category_position * queries_per_category + category_index + 1
        record = generator.generate_one(category, query_ordinal, category_index)
        if needs_segment_correction(record):
            record = rewrite_row(record, source_label)
        evaluate_record(record, executor)
        validate_record(record)
        append_jsonl(output_path, record)
        completed.append(record)
        generated_count = len(completed)
        if progress_interval > 0 and (
            generated_count % progress_interval == 0 or generated_count == query_count
        ):
            print(
                f"{category.key} shard {start_index}:{end_index}: "
                f"durably wrote {generated_count}/{query_count}",
                flush=True,
            )

    validate_workload_shape(completed, [category], query_count)
    return completed, resumed


def validate_workload_shape(
    records: Iterable[dict[str, Any]],
    categories: Sequence[Category],
    queries_per_category: int,
) -> None:
    rows = list(records)
    expected = Counter(category.key for category in categories)
    expected = Counter({key: count * queries_per_category for key, count in expected.items()})
    observed = Counter(str(record["category"]["key"]) for record in rows)
    if observed != expected:
        raise ValueError(f"bucket mismatch: expected {dict(expected)}, got {dict(observed)}")
    expected_multi = sum(
        queries_per_category for category in categories if category.relation == "multi"
    )
    observed_multi = sum(record.get("entity_cardinality") is not None for record in rows)
    if observed_multi != expected_multi:
        raise ValueError(
            f"expected {expected_multi} distinct-trip truths, got {observed_multi}"
        )


def build_summary(records: Sequence[dict[str, Any]], config_path: str) -> dict[str, Any]:
    counts = Counter(str(record["category"]["key"]) for record in records)
    return {
        "rows": len(records),
        "category_counts": dict(sorted(counts.items())),
        "ordinary_count_rows": sum(record["category"]["relation"] == "single" for record in records),
        "distinct_trip_rows": sum(record["category"]["relation"] == "multi" for record in records),
        "segment_coupled_rows": sum("semantic_correction" in record for record in records),
        "prohibited_trip_geom_rows": 0,
        "config": config_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--queries-per-category", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--sample-cache-size", type=int, default=2048)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--state")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--user")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    categories = parse_categories(config, args.categories)
    if len(categories) != 1:
        raise SystemExit("incremental generation accepts exactly one --categories value")
    if args.queries_per_category <= 0:
        raise SystemExit("--queries-per-category must be positive")
    query_count = args.query_count
    if query_count is None:
        query_count = args.queries_per_category - args.start_index
    output_path = Path(args.output)
    state_path = Path(args.state) if args.state else Path(str(output_path) + ".state.json")
    with QueryExecutor(args.host, args.port, args.dbname, args.user) as executor:
        records, resumed = generate_shard_incrementally(
            config,
            category=categories[0],
            queries_per_category=args.queries_per_category,
            start_index=args.start_index,
            query_count=query_count,
            base_seed=args.seed,
            sample_cache_size=args.sample_cache_size,
            executor=executor,
            source_label=str(config_path),
            output_path=output_path,
            state_path=state_path,
            progress_interval=args.progress_interval,
        )
    summary = build_summary(records, str(config_path))
    summary.update(
        {
            "category": categories[0].key,
            "queries_per_category": args.queries_per_category,
            "start_index": args.start_index,
            "query_count": query_count,
            "resumed_rows": resumed,
            "output": str(output_path),
            "state": str(state_path),
        }
    )
    write_json_atomic(Path(args.summary), summary)


if __name__ == "__main__":
    main()
