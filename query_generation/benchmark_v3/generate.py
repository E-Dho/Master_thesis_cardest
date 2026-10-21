#!/usr/bin/env python3
"""Generate and label one v3 category, resumably (SEMANTICS.md, section 5).

For every slot ``index`` of the category, candidates ``attempt = 0, 1, ...``
are drawn deterministically. A candidate is labelled exactly with one SQL
statement; it is rejected if its predicate hash was already accepted in this
category or if its label is 0, and accepted otherwise. Accepted rows and
rejections are appended to JSONL files and fsynced one by one, so an
interrupted task resumes without re-running any finished label query.

Exit codes: 0 finished, 75 stopped at the deadline (resubmit to resume).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from query_generation.generate_pol_segment_coupled_benchmark import (
    append_jsonl,
    load_jsonl_recover_trailing_write,
    write_json_atomic,
)
from query_generation.query_generator import Category

from . import SEMANTICS_VERSION
from .config import Settings, load_settings
from .pools import Pools
from .sampler import Sampler

STATE_VERSION = 1
EXIT_DEADLINE = 75


class SlotExhausted(RuntimeError):
    pass


def run_paths(output_dir: Path, category: Category) -> Dict[str, Path]:
    prefix = output_dir / category.key
    return {
        "accepted": Path(f"{prefix}.accepted.jsonl"),
        "rejected": Path(f"{prefix}.rejected.jsonl"),
        "state": Path(f"{prefix}.state.json"),
        "summary": Path(f"{prefix}.summary.json"),
    }


def binding(settings: Settings, pools: Pools, category: Category, base_seed: int) -> Dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "semantics_version": SEMANTICS_VERSION,
        "config_sha256": settings.sha256,
        "pools_sha256": pools.sha256,
        "category": category.key,
        "count": settings.count_for(category),
        "base_seed": int(base_seed),
    }


def check_or_write_state(path: Path, expected: Dict[str, Any], *, has_rows: bool) -> None:
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8")).get("binding", {})
        mismatched = sorted(key for key in expected if stored.get(key) != expected[key])
        if mismatched:
            raise SystemExit(
                f"{path} belongs to a different run ({', '.join(mismatched)} differ); "
                "use a new output directory"
            )
        return
    if has_rows:
        raise SystemExit(f"{path} is missing next to existing rows; refusing an unverifiable resume")
    write_json_atomic(path, {"binding": expected})


def generate_category(
    settings: Settings,
    pools: Pools,
    category: Category,
    *,
    base_seed: int,
    count_fn: Callable[[Dict[str, Any]], int],
    output_dir: Path,
    deadline: Optional[float] = None,
    progress_interval: int = 10,
    is_timeout: Callable[[BaseException], bool] = lambda error: False,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> Dict[str, Any]:
    paths = run_paths(output_dir, category)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = load_jsonl_recover_trailing_write(paths["accepted"])
    rejected = load_jsonl_recover_trailing_write(paths["rejected"])
    check_or_write_state(
        paths["state"],
        binding(settings, pools, category, base_seed),
        has_rows=bool(accepted or rejected),
    )

    sampler = Sampler(settings, pools, base_seed)
    target = settings.count_for(category)
    hashes = set()
    for position, row in enumerate(accepted):
        # Deterministic replay: the stored row must be exactly what the
        # sampler produces for its slot, otherwise config or pools changed.
        index, attempt = row["slot"]["index"], row["slot"]["attempt"]
        if index != position:
            raise SystemExit(f"{paths['accepted']}: row {position} has slot index {index}")
        replay = sampler.candidate(category, index, attempt)
        if replay["predicate_hash"] != row["predicate_hash"] or replay["sql"] != row["sql"]:
            raise SystemExit(f"{paths['accepted']}: slot {row['slot_id']} does not replay")
        hashes.add(row["predicate_hash"])
    if len(accepted) > target:
        raise SystemExit(f"{paths['accepted']} holds more rows than the grid allows")

    next_attempt: Dict[int, int] = {}
    for row in rejected:
        index = row["slot"]["index"]
        next_attempt[index] = max(next_attempt.get(index, 0), row["slot"]["attempt"] + 1)

    resumed = len(accepted)
    for index in range(len(accepted), target):
        attempt = next_attempt.get(index, 0)
        while True:
            if deadline is not None and time.time() >= deadline:
                summary = summarize(settings, category, accepted, rejected, base_seed, resumed, finished=False)
                write_json_atomic(paths["summary"], summary)
                log(f"{category.key}: deadline reached at {len(accepted)}/{target}; resubmit to resume")
                return summary
            if attempt >= settings.max_attempts:
                raise SlotExhausted(
                    f"{category.key} slot {index}: no accepted candidate in {settings.max_attempts} attempts"
                )
            record = sampler.candidate(category, index, attempt)
            if record["predicate_hash"] in hashes:
                rejection = rejection_row(record, "duplicate", None, 0.0)
            else:
                started = time.perf_counter()
                try:
                    cardinality = int(count_fn(record))
                except Exception as error:  # noqa: BLE001 - only timeouts are tolerated
                    if not is_timeout(error):
                        raise
                    rejection = rejection_row(record, "timeout", None, time.perf_counter() - started)
                else:
                    seconds = time.perf_counter() - started
                    if cardinality > 0:
                        finish_record(record, cardinality, seconds)
                        append_jsonl(paths["accepted"], record)
                        accepted.append(record)
                        hashes.add(record["predicate_hash"])
                        break
                    rejection = rejection_row(record, "zero", 0, seconds)
            append_jsonl(paths["rejected"], rejection)
            rejected.append(rejection)
            attempt += 1
        done = len(accepted)
        if progress_interval > 0 and (done % progress_interval == 0 or done == target):
            log(f"{category.key}: {done}/{target} accepted, {len(rejected)} rejected")

    summary = summarize(settings, category, accepted, rejected, base_seed, resumed, finished=True)
    write_json_atomic(paths["summary"], summary)
    return summary


def finish_record(record: Dict[str, Any], cardinality: int, seconds: float) -> None:
    record["cardinality"] = cardinality
    record["label_seconds"] = seconds
    record["labeled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Field names used by the existing model adapters.
    if record["category"]["relation"] == "multi":
        record["entity_cardinality"] = cardinality
        record["join_cardinality"] = None
    else:
        record["join_cardinality"] = cardinality
        record["entity_cardinality"] = None


def rejection_row(record: Dict[str, Any], reason: str, cardinality: Optional[int], seconds: float) -> Dict[str, Any]:
    return {
        "slot_id": record["slot_id"],
        "slot": record["slot"],
        "category": record["category"],
        "reason": reason,
        "cardinality": cardinality,
        "label_seconds": seconds,
        "predicate_hash": record["predicate_hash"],
        "tables": record["tables"],
        "predicates": record["predicates"],
        "sql": record["sql"],
        "config_sha256": record["config_sha256"],
        "pools_sha256": record["pools_sha256"],
    }


def summarize(settings, category, accepted, rejected, base_seed, resumed, *, finished) -> Dict[str, Any]:
    label_seconds = [row["label_seconds"] for row in accepted] + [
        row["label_seconds"] for row in rejected if row["label_seconds"]
    ]
    attempts = Counter(row["slot"]["attempt"] for row in accepted)
    return {
        "category": category.key,
        "finished": finished,
        "target": settings.count_for(category),
        "accepted": len(accepted),
        "rejected": dict(Counter(row["reason"] for row in rejected)),
        "accepted_attempt_histogram": {str(key): value for key, value in sorted(attempts.items())},
        "resumed_from": resumed,
        "label_seconds_total": sum(label_seconds),
        "label_seconds_accepted": sum(row["label_seconds"] for row in accepted),
        "label_seconds_max": max(label_seconds, default=0.0),
        "base_seed": int(base_seed),
        "config_sha256": settings.sha256,
        "semantics_version": SEMANTICS_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--single-count", type=int, help="override the grid (pilot runs)")
    parser.add_argument("--multi-count", type=int, help="override the grid (pilot runs)")
    parser.add_argument("--statement-timeout-ms", type=int, default=0, help="0 disables the timeout")
    parser.add_argument("--deadline-seconds", type=float, help="stop cleanly after this many seconds")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .backends import SqlBackend, SqlSession

    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config), single_count=args.single_count, multi_count=args.multi_count)
    pools = Pools.load(Path(args.pools))
    category = Category.parse(args.category)
    deadline = time.time() + args.deadline_seconds if args.deadline_seconds else None
    with SqlSession(args.host, args.port, args.dbname, args.user, args.statement_timeout_ms) as session:
        backend = SqlBackend(session)
        summary = generate_category(
            settings,
            pools,
            category,
            base_seed=args.base_seed,
            count_fn=backend.count,
            output_dir=Path(args.output_dir),
            deadline=deadline,
            progress_interval=args.progress_interval,
            is_timeout=SqlSession.is_timeout,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["finished"] else EXIT_DEADLINE


if __name__ == "__main__":
    sys.exit(main())
