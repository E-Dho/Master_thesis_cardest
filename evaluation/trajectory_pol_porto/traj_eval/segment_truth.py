"""True matching-segment counts of multi queries (P7 error split).

v3 multi labels count distinct trajectories only. The own model estimates
``D_hat = M_hat * a_hat`` (matching segments times ``traj_dedup_factor``), so
splitting its error needs the true number of matching segments. For every
multi query that joins ``segments`` this computes on the snapshot, with the
``NumpyBackend`` of the label gate:

- ``matching_segments_exact`` / ``distinct_exact``: segment rows (resp.
  trajectories) satisfying every predicate under the v3 semantics;
  ``distinct_exact`` must equal the label (checked);
- ``matching_segments_model`` / ``distinct_model``: the same under the
  semantics the model is asked to answer: the spatial window as segment MBR
  overlap and, for spatio-temporal queries, space and time tested separately
  (``mbr_overcount``'s box candidates). Equal to the exact values for queries
  without a spatial predicate.

``traj_eval/own_model_error_split.py`` combines these with the bridge's
components. Sharded, resumable and provenance-bound like ``mbr_overcount``
(same snapshot verification step, committed source files).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import mbr_overcount
from .snapshot_gate import EXIT_DEADLINE, BindingError, _done, _rows, _sha256, _shard_path, bind_results_dir

STATISTIC = "matching_segment_truth"
STATISTIC_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MODULES = (
    "evaluation.trajectory_pol_porto.traj_eval.segment_truth",
    *mbr_overcount.SOURCE_MODULES,
)
CSV_FIELDS = ("query_id", "predicate_hash", "category", "dimension", "addon_predicates", "label",
              "matching_segments_exact", "distinct_exact", "matching_segments_model", "distinct_model", "match",
              "seconds")
SEGMENT_TABLE = "segments"


def selected(row: Dict[str, Any]) -> bool:
    return row["category"]["relation"] == "multi" and SEGMENT_TABLE in row["tables"]


def evaluate_query(backend, record: Dict[str, Any]) -> Dict[str, Any]:
    segments = backend.data[SEGMENT_TABLE]
    entity = backend.settings.entity_table
    join_col, _ = backend.join_keys[(SEGMENT_TABLE, entity)]
    trip_ids = np.asarray(segments[join_col])
    entities = backend.qualifying_entities(record, skip_table=SEGMENT_TABLE)
    related = np.isin(trip_ids, entities)
    exact = backend.table_mask(SEGMENT_TABLE, record["predicates"]) & related
    predicate = mbr_overcount.spatial_predicate(record)
    if predicate is None:
        model = exact
    else:
        classes = mbr_overcount.segment_classes(segments, predicate)
        model = np.zeros(trip_ids.size, dtype=bool)
        model[classes["index"]] = True
        for other in record["predicates"]:
            if other is not predicate and other["table"] == SEGMENT_TABLE:
                model &= np.asarray(backend.predicate_mask(other))
        model &= related
    distinct_exact = int(np.unique(trip_ids[exact]).size)
    label = int(record["cardinality"])
    return {
        "query_id": record["query_id"], "predicate_hash": record["predicate_hash"],
        "category": record["category"]["key"], "dimension": record["category"]["dimension"],
        "addon_predicates": int(record.get("addon_predicates") or 0), "label": label,
        "matching_segments_exact": int(np.count_nonzero(exact)), "distinct_exact": distinct_exact,
        "matching_segments_model": int(np.count_nonzero(model)),
        "distinct_model": int(np.unique(trip_ids[model]).size),
        "match": distinct_exact == label,
    }


def invariant_violations(item: Dict[str, Any]) -> List[str]:
    problems = []
    if not item["distinct_exact"] <= item["matching_segments_exact"]:
        problems.append("distinct_exact > matching_segments_exact")
    if not item["distinct_model"] <= item["matching_segments_model"]:
        problems.append("distinct_model > matching_segments_model")
    if not (item["matching_segments_exact"] <= item["matching_segments_model"]
            and item["distinct_exact"] <= item["distinct_model"]):
        problems.append("model semantics count below the exact count")
    if item["dimension"] not in ("spatial", "spatio_temporal") and (
            item["matching_segments_model"] != item["matching_segments_exact"]):
        problems.append("non-spatial query with model != exact")
    return problems


def source_files() -> List[Path]:
    return sorted({Path(importlib.import_module(name).__file__).resolve() for name in SOURCE_MODULES})


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0")
        digest.update(_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def code_state() -> Dict[str, Any]:
    paths = [str(path.relative_to(REPO_ROOT)) for path in source_files()]
    try:
        revision = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", *paths],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"code_revision": None, "source_files_modified": None}
    return {"code_revision": revision, "source_files_modified": bool(status)}


def binding(snapshot: Path, benchmark: Path, config_sha256: str, shards: int, *,
            allow_dirty: bool = False) -> Dict[str, Any]:
    state = code_state()
    if not allow_dirty and (state["code_revision"] is None or state["source_files_modified"]):
        raise mbr_overcount.ProvenanceError(
            f"the source files are not a committed git state ({state}); commit them, or pass --allow-dirty")
    return {"statistic": STATISTIC, "statistic_version": STATISTIC_VERSION, **state,
            "source_sha256": source_sha256(), "snapshot_manifest_sha256": _sha256(Path(snapshot) / "manifest.json"),
            "benchmark_sha256": _sha256(Path(benchmark)), "config_sha256": config_sha256, "shards": int(shards)}


def run_shard(snapshot: Path, config: Path, benchmark: Path, results_dir: Path, shard: int, shards: int,
              deadline_seconds: Optional[float] = None, log=print, *, allow_dirty: bool = False) -> int:
    from query_generation.benchmark_v3.config import load_settings

    from .snapshot import load_snapshot

    if not 0 <= shard < shards:
        raise ValueError("--shard must be in [0, --shards)")
    started = time.time()
    mbr_overcount.require_verified(snapshot, results_dir)
    settings = load_settings(config)
    bind_results_dir(results_dir, binding(snapshot, benchmark, settings.sha256, shards, allow_dirty=allow_dirty))
    data = load_snapshot(snapshot, expected_config_sha256=settings.sha256)
    backend = mbr_overcount._backend_class()(settings, data)
    rows = [row for position, row in enumerate(r for r in _rows(benchmark) if selected(r)) if position % shards == shard]
    path = _shard_path(results_dir, shard, shards)
    done = _done(path)
    hashes = {row["query_id"]: row["predicate_hash"] for row in rows}
    foreign = [qid for qid, item in done.items() if hashes.get(qid) != item.get("predicate_hash")]
    if foreign:
        raise BindingError(f"{path} holds results for other queries (e.g. {foreign[:3]})")
    with path.open("w", encoding="utf-8") as handle:  # drop a possibly torn last line
        for item in done.values():
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    log(f"shard {shard}/{shards}: {len(rows)} queries, {len(done)} already done")
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["query_id"] in done:
                continue
            if deadline_seconds is not None and time.time() - started > deadline_seconds:
                log(f"deadline reached; resubmit to continue ({path})")
                return EXIT_DEADLINE
            query_started = time.perf_counter()
            item = evaluate_query(backend, row)
            item["seconds"] = round(time.perf_counter() - query_started, 4)
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if not item["match"]:
                log(f"MISMATCH {item}")
    log(f"shard {shard}/{shards} finished")
    return 0


def merge(benchmark: Path, results_dir: Path, output_prefix: Path, snapshot: Path) -> Dict[str, Any]:
    binding_path = results_dir / "binding.json"
    if not binding_path.exists():
        raise BindingError(f"{results_dir} has no binding.json")
    bound = json.loads(binding_path.read_text(encoding="utf-8"))
    if bound.get("statistic") != STATISTIC or bound.get("statistic_version") != STATISTIC_VERSION:
        raise BindingError(f"{results_dir} holds results of another statistic or version")
    mbr_overcount.require_verified(snapshot, results_dir)
    for key, value in (("benchmark_sha256", _sha256(benchmark)),
                       ("snapshot_manifest_sha256", _sha256(Path(snapshot) / "manifest.json"))):
        if bound.get(key) != value:
            raise BindingError(f"{results_dir} was produced for other inputs ({key} differs)")
    expected = {row["query_id"]: row for row in _rows(benchmark) if selected(row)}
    found: Dict[str, Dict[str, Any]] = {}
    duplicates = []
    for path in sorted(results_dir.glob("shard_*_of_*.jsonl")):
        if int(path.stem.split("_of_")[1]) != int(bound["shards"]):
            raise BindingError(f"{path.name} does not match the bound shard count {bound['shards']}")
        for query_id, item in _done(path).items():
            if query_id in found:
                duplicates.append(query_id)
            found[query_id] = item
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    foreign = sorted(q for q, item in found.items()
                     if q in expected and item["predicate_hash"] != expected[q]["predicate_hash"])
    # the label check is redone from the benchmark, never taken from the stored flag
    mismatches = sorted(q for q, item in found.items()
                        if q in expected and int(item["distinct_exact"]) != int(expected[q]["cardinality"]))
    items = [found[q] for q in sorted(found) if q in expected]
    violations = [{"query_id": item["query_id"], "problems": problems}
                  for item in items for problems in [invariant_violations(item)] if problems]
    report = {
        "passed": not (missing or unexpected or duplicates or foreign or mismatches or violations),
        "statistic": STATISTIC, "binding": bound,
        "merge_code_state": {**code_state(), "source_sha256": source_sha256()},
        "benchmark": str(benchmark), "snapshot": str(snapshot), "queries_evaluated": len(items),
        "queries_expected": len(expected), "missing": missing[:200], "unexpected": unexpected[:200],
        "duplicates": duplicates[:200], "predicate_hash_mismatches": foreign[:200],
        "label_mismatches": mismatches[:200], "invariant_violations": violations[:200],
        "queries_with_model_overcount": sum(1 for item in items
                                            if item["matching_segments_model"] > item["matching_segments_exact"]),
        "seconds_total": round(sum(item.get("seconds", 0.0) for item in items), 2),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{output_prefix}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with Path(f"{output_prefix}_per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="True matching-segment counts of multi queries (P7 error split)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    check = sub.add_parser("verify", help="check the snapshot arrays against the manifest checksums once")
    check.add_argument("--snapshot", required=True, type=Path)
    check.add_argument("--results-dir", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--snapshot", required=True, type=Path)
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--benchmark", required=True, type=Path)
    run.add_argument("--results-dir", required=True, type=Path)
    run.add_argument("--shard", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--deadline-seconds", type=float)
    run.add_argument("--allow-dirty", action="store_true")
    combine = sub.add_parser("merge")
    combine.add_argument("--snapshot", required=True, type=Path)
    combine.add_argument("--benchmark", required=True, type=Path)
    combine.add_argument("--results-dir", required=True, type=Path)
    combine.add_argument("--output-prefix", required=True, type=Path,
                         help="writes <prefix>.json and <prefix>_per_query.csv")
    args = parser.parse_args(argv)
    log = lambda message: print(message, flush=True)  # noqa: E731
    if args.command == "version":
        print(STATISTIC_VERSION)
        return 0
    if args.command == "verify":
        return 0 if mbr_overcount.verify(args.snapshot, args.results_dir, log=log)["passed"] else 1
    if args.command == "run":
        return run_shard(args.snapshot, args.config, args.benchmark, args.results_dir, args.shard, args.shards,
                         args.deadline_seconds, log=log, allow_dirty=args.allow_dirty)
    report = merge(args.benchmark, args.results_dir, args.output_prefix, args.snapshot)
    log(json.dumps({key: report[key] for key in ("passed", "queries_evaluated", "queries_expected",
                                                 "queries_with_model_overcount")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
