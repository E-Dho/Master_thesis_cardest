#!/usr/bin/env python3
"""Replace only true-zero POL workload rows with positive rejection samples.

The input benchmark remains untouched.  Its positive rows are copied as-is;
each row whose exact database join cardinality is zero is replaced by a fresh
draw from the same category-local generator distribution, accepted only when
its final (segment-coupled) SQL has positive cardinality.

Replacements are appended to a progress sidecar as they are accepted and the
live-center pools are persisted next to the output, so an interrupted run
resumes instead of re-evaluating every candidate against the database.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
        config_hash,
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
        config_hash,
        load_config,
    )
    from rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row


REPLACEMENT_NAME = "positive_rejection_sampling_v1"
CENTERS_CACHE_VERSION = 1
PROGRESS_HEADER_KEY = "progress_header"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row and force it to disk, so an interrupted run loses nothing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if created:
        fsync_directory(path.parent)


def fsync_directory(path: Path) -> None:
    """Make a file creation or rename in ``path`` durable."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    fsync_directory(path.parent)


def centers_binding(
    *,
    config: dict[str, Any],
    source_workload_sha256: str,
    sample_cache_size: int,
    seed: int,
) -> dict[str, Any]:
    """What a persisted center-pool snapshot is only valid for."""

    return {
        "config_hash": config_hash(config),
        "source_workload_sha256": source_workload_sha256,
        "sample_cache_size": int(sample_cache_size),
        "seed": int(seed),
    }


def load_centers_cache(
    path: Path,
    live_centers: LiveCenterCache,
    binding: dict[str, Any],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pools" not in payload:
        raise SystemExit(
            f"{path} is not a center-pool snapshot written by this script; delete it to refetch"
        )
    if int(payload.get("version", -1)) != CENTERS_CACHE_VERSION:
        raise SystemExit(f"{path} has center-pool snapshot version {payload.get('version')!r}")
    stored = payload.get("binding") or {}
    mismatched = sorted(key for key, value in binding.items() if stored.get(key) != value)
    if mismatched:
        raise SystemExit(
            f"{path} was written for a different {', '.join(mismatched)}; delete it to refetch"
        )
    live_centers.restore(payload["pools"])


def centers_cache_payload(
    live_centers: LiveCenterCache, binding: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": CENTERS_CACHE_VERSION,
        "binding": binding,
        "pools": live_centers.snapshot(),
    }


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


def resolved_category_index(target: dict[str, Any], source_index: int) -> int:
    value = target["category"].get("category_index")
    return source_index if value is None else int(value)


def query_ordinal(query_id: str, fallback: int) -> int:
    if query_id.startswith("q") and query_id[1:].isdigit():
        return int(query_id[1:])
    return fallback + 1


def table_subset_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter("+".join(sorted(row["tables"])) for row in rows)
    return dict(sorted(counts.items()))


def normalize_final_semantics(candidate: dict[str, Any], source_workload: str) -> dict[str, Any]:
    if needs_segment_correction(candidate):
        return rewrite_row(candidate, source_workload)
    return candidate


def evaluate_candidate(candidate: dict[str, Any], executor: QueryExecutor) -> None:
    started = time.perf_counter()
    candidate["join_cardinality"] = int(executor.scalar(candidate["sql"]))
    candidate["join_evaluation_seconds"] = time.perf_counter() - started
    candidate["evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
    category_index = resolved_category_index(target, source_index)
    for attempt in range(1, max_attempts + 1):
        seed = replacement_seed(base_seed, target_query_id, attempt)
        generator = QueryGenerator(
            config,
            seed=seed,
            executor=executor,
            evaluate_cardinalities=False,
            live_centers=live_centers,
        )
        candidate = generator.generate_one(
            category,
            query_ordinal(target_query_id, source_index),
            category_index,
        )
        candidate = normalize_final_semantics(candidate, source_workload)
        if candidate["category"]["key"] != target_key:
            continue
        evaluate_candidate(candidate, executor)
        candidate_hash = canonical_sql_hash(candidate)
        if candidate["join_cardinality"] <= 0 or candidate_hash in used_sql_hashes:
            continue
        candidate["query_id"] = target_query_id
        candidate["category"]["category_index"] = category_index
        candidate["source_row_index"] = source_index
        correction = candidate.get("semantic_correction")
        if correction is not None:
            # The pre-correction SQL of a replacement is never executed, so the
            # cardinalities carried in ``original`` are None by construction.
            correction["original_cardinalities_evaluated"] = False
        candidate["nonzero_replacement"] = {
            "name": REPLACEMENT_NAME,
            "source_row_index": source_index,
            "replacement_seed": seed,
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


def ensure_progress_header(path: Path, binding: dict[str, Any]) -> None:
    """Start a progress file with the run it belongs to, before the first row."""

    if not path.exists():
        append_jsonl(path, {PROGRESS_HEADER_KEY: binding})


def load_progress(
    path: Path | None,
    source_rows: Sequence[dict[str, Any]],
    base_seed: int,
    binding: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reload replacements accepted by an earlier run of this same job.

    Every row is checked against the source workload it claims to replace, so a
    progress file left over from a different input, a different generator
    config, a different seed or a different row ordering is rejected instead of
    silently reused.  Replacements must also be distinct from one another, since
    their SQL hashes seed the duplicate guard for the rows still to be sampled.
    """

    if path is None or not path.exists():
        return {}
    records = load_jsonl(path)
    if not records or PROGRESS_HEADER_KEY not in records[0]:
        raise SystemExit(
            f"{path} has no progress header; it was not written by this script, "
            "or predates run binding -- delete it to start over"
        )
    stored = records[0][PROGRESS_HEADER_KEY] or {}
    mismatched = sorted(key for key, value in binding.items() if stored.get(key) != value)
    if mismatched:
        raise SystemExit(
            f"{path} was written for a different {', '.join(mismatched)}; delete it to start over"
        )
    expected_config_hash = binding["config_hash"]
    by_id = {str(row["query_id"]): (index, row) for index, row in enumerate(source_rows)}
    resumed: dict[str, dict[str, Any]] = {}
    seen_sql: dict[str, str] = {}
    for row in records[1:]:
        query_id = str(row.get("query_id"))
        meta = row.get("nonzero_replacement") or {}
        if query_id not in by_id:
            raise SystemExit(f"progress file holds unknown query_id {query_id}")
        if query_id in resumed:
            raise SystemExit(f"progress file holds duplicate replacement for {query_id}")
        index, target = by_id[query_id]
        if int(meta.get("source_row_index", -1)) != index:
            raise SystemExit(f"progress file row index mismatch for {query_id}")
        if not is_true_zero(target):
            raise SystemExit(f"progress file replaces non-zero source row {query_id}")
        if meta.get("replaced_sql_sha256") != canonical_sql_hash(target):
            raise SystemExit(f"progress file does not match the source workload for {query_id}")
        expected_seed = replacement_seed(base_seed, query_id, int(meta.get("attempt", 0)))
        if int(meta.get("replacement_seed", -1)) != expected_seed:
            raise SystemExit(f"progress file was written with a different --seed ({query_id})")
        if int(row["join_cardinality"]) <= 0:
            raise SystemExit(f"progress file holds a non-positive replacement for {query_id}")
        if expected_config_hash is not None and row.get("config_hash") != expected_config_hash:
            raise SystemExit(
                f"progress file was written with a different generator config ({query_id})"
            )
        row_hash = canonical_sql_hash(row)
        if row_hash in seen_sql:
            raise SystemExit(
                f"progress file holds duplicate replacement SQL for {query_id} "
                f"and {seen_sql[row_hash]}"
            )
        seen_sql[row_hash] = query_id
        resumed[query_id] = row
    return resumed


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
    source_workload_sha256: str = "",
    progress_path: Path | None = None,
    centers_cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    output = [copy.deepcopy(row) for row in source_rows]
    used_sql_hashes = {
        canonical_sql_hash(row) for row in source_rows if not is_true_zero(row)
    }
    binding = centers_binding(
        config=config,
        source_workload_sha256=source_workload_sha256,
        sample_cache_size=sample_cache_size,
        seed=base_seed,
    )
    cached_pool_keys: list[str] = []
    live_centers: LiveCenterCache

    def persist_centers() -> None:
        nonlocal cached_pool_keys
        if centers_cache_path is None or live_centers.pool_keys() == cached_pool_keys:
            return
        write_json_atomic(centers_cache_path, centers_cache_payload(live_centers, binding))
        cached_pool_keys = live_centers.pool_keys()

    # The cache invokes this callback immediately after storing every newly
    # fetched pool, before generation continues with predicate construction.
    live_centers = LiveCenterCache(
        config, executor, sample_cache_size, on_pool_cached=persist_centers
    )
    if centers_cache_path is not None and centers_cache_path.exists():
        load_centers_cache(centers_cache_path, live_centers, binding)
    cached_pool_keys = live_centers.pool_keys()

    resumed = load_progress(progress_path, source_rows, base_seed, binding)
    if progress_path is not None:
        ensure_progress_header(progress_path, binding)
    # Every resumed replacement occupies a slot in the final workload, so its
    # SQL has to guard the rows still to be sampled from the first candidate
    # onwards -- not only once the loop reaches that row's own index.
    used_sql_hashes.update(canonical_sql_hash(row) for row in resumed.values())
    attempts: list[int] = []
    replaced_ids: list[str] = []
    for index, target in enumerate(source_rows):
        if not is_true_zero(target):
            continue
        query_id = str(target["query_id"])
        if query_id in resumed:
            replacement = resumed[query_id]
            replacement_attempts = int(replacement["nonzero_replacement"]["attempt"])
        else:
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
            # The cache callback persists every newly fetched pool before
            # generation continues, so the pools behind this row are on disk
            # before the row itself is.
            if progress_path is not None:
                append_jsonl(progress_path, replacement)
        used_sql_hashes.add(canonical_sql_hash(replacement))
        output[index] = replacement
        attempts.append(replacement_attempts)
        replaced_ids.append(query_id)
    persist_centers()
    provenance = {
        "replaced_query_ids": replaced_ids,
        "resumed_replacements": sum(1 for query_id in replaced_ids if query_id in resumed),
        "center_pools": live_centers.fingerprint(),
    }
    return output, attempts, provenance


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    source_path: Path,
    source_rows: Sequence[dict[str, Any]],
    seed: int,
    attempts: Sequence[int],
    provenance: dict[str, Any],
    source_sha256: str | None = None,
) -> dict[str, Any]:
    replaced = len(provenance["replaced_query_ids"])
    return {
        "name": REPLACEMENT_NAME,
        "source_workload": str(source_path),
        "source_workload_sha256": source_sha256 or file_sha256(source_path),
        "seed": seed,
        "rows": len(rows),
        "unique_query_ids": len({row["query_id"] for row in rows}),
        "retained_positive_rows": len(source_rows) - replaced,
        "replaced_true_zero_rows": replaced,
        "resumed_replacements": provenance["resumed_replacements"],
        "remaining_true_zero_rows": sum(is_true_zero(row) for row in rows),
        "category_counts": dict(sorted(Counter(row["category"]["key"] for row in rows).items())),
        "table_subset_counts": {
            "source": table_subset_counts(source_rows),
            "output": table_subset_counts(rows),
        },
        "distinct_trajectory_truth_rows": sum(
            row.get("entity_cardinality") is not None for row in rows
        ),
        "center_pools": provenance["center_pools"],
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
    parser.add_argument(
        "--progress",
        help="JSONL sidecar of accepted replacements; defaults to <output>.partial.jsonl. "
        "An existing file is validated against the source workload and resumed.",
    )
    parser.add_argument(
        "--centers-cache",
        help="JSON snapshot of the live-center pools; defaults to <output>.centers.json. "
        "The pools are drawn with an unseeded ORDER BY random(), so this file is what "
        "makes a run reproducible.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable both sidecars.")
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
    source_sha256 = file_sha256(source_path)
    output_path = Path(args.output)
    if args.no_progress:
        progress_path = None
        centers_cache_path = None
    else:
        progress_path = Path(
            args.progress or output_path.with_name(output_path.name + ".partial.jsonl")
        )
        centers_cache_path = Path(
            args.centers_cache or output_path.with_name(output_path.name + ".centers.json")
        )
    with QueryExecutor(
        host=args.host, port=args.port, dbname=args.dbname, user=args.user
    ) as executor:
        rows, attempts, provenance = replace_true_zeros(
            source_rows,
            config=config,
            executor=executor,
            base_seed=args.seed,
            max_attempts=args.max_attempts,
            source_workload=str(source_path),
            sample_cache_size=args.sample_cache_size,
            source_workload_sha256=source_sha256,
            progress_path=progress_path,
            centers_cache_path=centers_cache_path,
        )
    # Write before validating: a validation failure must not discard hours of
    # database evaluation.
    write_jsonl(output_path, rows)
    summary = build_summary(
        rows,
        source_path=source_path,
        source_rows=source_rows,
        seed=args.seed,
        attempts=attempts,
        provenance=provenance,
        source_sha256=source_sha256,
    )
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "center_pools"}, sort_keys=True))
    validate_output(rows, source_rows)


if __name__ == "__main__":
    main()
