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

A results directory is bound to its inputs: the first shard writes
``binding.json`` (SHA-256 of the snapshot manifest and of the benchmark, the
v3 config hash, the shard count), created atomically so parallel shards cannot
race. Every later shard, every resume and ``merge`` refuse a directory bound to
other inputs, and every result line carries its query's ``predicate_hash``, so
a result can never be credited to another query. Use a new results directory
for new inputs.

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
GATE_VERSION = 1
BINDING_FILE = "binding.json"


class BindingError(ValueError):
    """A results directory holds results of other inputs."""


def gate_binding(snapshot: Path, benchmark: Path, config_sha256: str, shards: int) -> Dict[str, Any]:
    return {
        "gate_version": GATE_VERSION,
        "snapshot_manifest_sha256": _sha256(Path(snapshot) / "manifest.json"),
        "benchmark_sha256": _sha256(Path(benchmark)),
        "config_sha256": config_sha256,
        "shards": int(shards),
    }


def bind_results_dir(results_dir: Path, binding: Dict[str, Any]) -> None:
    """Create ``binding.json`` atomically, or check that it matches."""

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / BINDING_FILE
    if not path.exists():
        if any(results_dir.glob("shard_*_of_*.jsonl")):
            raise BindingError(f"{results_dir} holds shard results without {BINDING_FILE}; use a new results directory")
        temporary = results_dir / f".{BINDING_FILE}.{os.getpid()}"
        temporary.write_text(json.dumps(binding, indent=2, sort_keys=True), encoding="utf-8")
        try:
            os.link(temporary, path)  # atomic; fails if another shard was first
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    stored = json.loads(path.read_text(encoding="utf-8"))
    differing = sorted(key for key in set(stored) | set(binding) if stored.get(key) != binding.get(key))
    if differing:
        raise BindingError(
            f"{results_dir} was produced for other inputs (differs in {differing}); "
            "use a new results directory"
        )


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
    bind_results_dir(results_dir, gate_binding(snapshot, benchmark, settings.sha256, shards))
    path = _shard_path(results_dir, shard, shards)
    done = _done(path)
    hashes = {row["query_id"]: row["predicate_hash"] for row in rows}
    foreign = [qid for qid, item in done.items() if hashes.get(qid) != item.get("predicate_hash")]
    if foreign:
        raise BindingError(f"{path} holds results for other queries (e.g. {foreign[:3]})")
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
                "predicate_hash": row["predicate_hash"],
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


def merge(benchmark: Path, results_dir: Path, report: Path, snapshot: Path) -> Dict[str, Any]:
    binding_path = results_dir / BINDING_FILE
    if not binding_path.exists():
        raise BindingError(f"{results_dir} has no {BINDING_FILE}")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    for key, value in (("benchmark_sha256", _sha256(benchmark)),
                       ("snapshot_manifest_sha256", _sha256(Path(snapshot) / "manifest.json"))):
        if binding.get(key) != value:
            raise BindingError(f"{results_dir} was produced for another {key.split('_')[0]} ({key} differs)")
    rows = _rows(benchmark)
    expected = {row["query_id"]: row for row in rows}
    found: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    files = sorted(results_dir.glob("shard_*_of_*.jsonl"))
    shard_counts = {int(path.stem.split("_of_")[1]) for path in files}
    if shard_counts - {int(binding["shards"])}:
        raise BindingError(f"shard files {sorted(shard_counts)} do not match the bound shard count {binding['shards']}")
    for path in files:
        for query_id, item in _done(path).items():
            if query_id in found:
                duplicates.append(query_id)
            found[query_id] = item
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    mismatches = [item for item in found.values() if not item["match"]]
    foreign = sorted(qid for qid, item in found.items()
                     if qid in expected and item.get("predicate_hash") != expected[qid]["predicate_hash"])
    label_changed = [qid for qid, item in found.items()
                     if qid in expected and item["expected"] != expected[qid]["cardinality"]]
    seconds = sorted(item["seconds"] for item in found.values())
    by_category = Counter(item["category"] for item in found.values())
    result = {
        "passed": not (missing or unexpected or mismatches or duplicates or label_changed or foreign),
        "binding": binding,
        "predicate_hash_mismatches": foreign[:200],
        "benchmark": str(benchmark),
        "benchmark_sha256": _sha256(benchmark),
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": binding["snapshot_manifest_sha256"],
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
    combine.add_argument("--snapshot", required=True, type=Path)
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
