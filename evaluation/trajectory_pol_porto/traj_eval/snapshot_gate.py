"""Label-reproduction gate: every benchmark label recounted on the snapshot.

``query_generation.benchmark_v3.numpy_backend.NumpyBackend`` evaluates each
query of a ``benchmark.jsonl`` on the snapshot arrays, and the count must equal
the stored label exactly. Passing this gate shows that the snapshot, the
structured predicates and the label semantics agree, before any baseline is
built from the snapshot.

Work is split into shards (``query position % shards == shard``); each shard
appends one JSON line per query and fsyncs, so a killed or time-limited job
resumes where it stopped (exit 75 at ``--deadline-seconds``). ``merge`` checks
that every query was recounted exactly once and writes the report; it exits 1
on any mismatch or gap.

    python -m evaluation.trajectory_pol_porto.traj_eval.snapshot_gate run \\
        --snapshot DIR --config pol_v3_config.json --benchmark benchmark.jsonl \\
        --results-dir DIR/gate/test --shard 0 --shards 16
    python -m evaluation.trajectory_pol_porto.traj_eval.snapshot_gate merge \\
        --benchmark benchmark.jsonl --results-dir DIR/gate/test --report DIR/gate/test_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

EXIT_DEADLINE = 75


def _rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _done(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    done = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break  # a torn last line from a killed job; it is recomputed
            done[item["query_id"]] = item
    return done


def _shard_path(results_dir: Path, shard: int, shards: int) -> Path:
    return results_dir / f"shard_{shard:03d}_of_{shards:03d}.jsonl"


def run_shard(snapshot: Path, config: Path, benchmark: Path, results_dir: Path, shard: int, shards: int,
              deadline_seconds: Optional[float] = None, log=print) -> int:
    from query_generation.benchmark_v3.config import load_settings
    from query_generation.benchmark_v3.numpy_backend import NumpyBackend

    from .snapshot import load_snapshot

    if not 0 <= shard < shards:
        raise ValueError("--shard must be in [0, --shards)")
    started = time.time()
    settings = load_settings(config)
    data = load_snapshot(snapshot, expected_config_sha256=settings.sha256)
    backend = NumpyBackend(settings, data)
    rows = [row for position, row in enumerate(_rows(benchmark)) if position % shards == shard]
    results_dir.mkdir(parents=True, exist_ok=True)
    path = _shard_path(results_dir, shard, shards)
    done = _done(path)
    # rewrite the file without a possibly torn last line before appending
    with path.open("w", encoding="utf-8") as handle:
        for item in done.values():
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    log(f"shard {shard}/{shards}: {len(rows)} queries, {len(done)} already done")
    mismatches = sum(not item["match"] for item in done.values())
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["query_id"] in done:
                continue
            if deadline_seconds is not None and time.time() - started > deadline_seconds:
                log(f"deadline reached; resubmit to continue ({path})")
                return EXIT_DEADLINE
            query_started = time.perf_counter()
            counted = int(backend.count(row))
            item = {
                "query_id": row["query_id"],
                "category": row["category"]["key"],
                "label_kind": row["label_kind"],
                "expected": int(row["cardinality"]),
                "counted": counted,
                "match": counted == int(row["cardinality"]),
                "seconds": round(time.perf_counter() - query_started, 4),
            }
            mismatches += not item["match"]
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not item["match"]:
                log(f"MISMATCH {item}")
    log(f"shard {shard}/{shards} finished, {mismatches} mismatch(es)")
    return 0


def merge(benchmark: Path, results_dir: Path, report: Path, snapshot: Optional[Path] = None) -> Dict[str, Any]:
    rows = _rows(benchmark)
    expected = {row["query_id"]: row for row in rows}
    found: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    files = sorted(results_dir.glob("shard_*_of_*.jsonl"))
    shard_counts = {path.name.split("_of_")[1] for path in files}
    if len(shard_counts) > 1:
        raise ValueError(f"results of different shard counts in {results_dir}: {sorted(shard_counts)}")
    for path in files:
        for query_id, item in _done(path).items():
            if query_id in found:
                duplicates.append(query_id)
            found[query_id] = item
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    mismatches = [item for item in found.values() if not item["match"]]
    label_changed = [qid for qid, item in found.items()
                     if qid in expected and item["expected"] != expected[qid]["cardinality"]]
    seconds = sorted(item["seconds"] for item in found.values())
    by_category = Counter(item["category"] for item in found.values())
    result = {
        "passed": not (missing or unexpected or mismatches or duplicates or label_changed),
        "benchmark": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
        "snapshot": str(snapshot) if snapshot else None,
        "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json") if snapshot else None,
        "queries": len(expected),
        "recounted": len(found),
        "mismatch_count": len(mismatches),
        "mismatches": sorted(mismatches, key=lambda item: item["query_id"])[:200],
        "missing_count": len(missing),
        "missing": missing[:200],
        "unexpected": unexpected[:200],
        "duplicates": duplicates[:200],
        "stored_label_changed": label_changed[:200],
        "recounted_by_category": dict(sorted(by_category.items())),
        "seconds_total": round(sum(seconds), 2),
        "seconds_p50": seconds[len(seconds) // 2] if seconds else None,
        "seconds_max": seconds[-1] if seconds else None,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recount benchmark labels on a snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--snapshot", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--benchmark", required=True, type=Path)
    run.add_argument("--results-dir", required=True, type=Path)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--deadline-seconds", type=float)
    combine = sub.add_parser("merge")
    combine.add_argument("--benchmark", required=True, type=Path)
    combine.add_argument("--results-dir", required=True, type=Path)
    combine.add_argument("--report", required=True, type=Path)
    combine.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    log = lambda message: print(message, flush=True)
    if args.command == "run":
        return run_shard(args.snapshot, args.config, args.benchmark, args.results_dir, args.shard,
                         args.shards, args.deadline_seconds, log)
    result = merge(args.benchmark, args.results_dir, args.report, args.snapshot)
    log(json.dumps({key: result[key] for key in ("passed", "queries", "recounted", "mismatch_count",
                                                  "missing_count", "seconds_total")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
