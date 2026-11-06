"""How much a segment's bounding box overcounts the spatial part of v3 queries.

A method that represents each segment by its minimum bounding rectangle (MBR)
counts a segment as soon as the box overlaps the query window ``W``. The exact
v3 predicate (SEMANTICS.md 2.1) counts it only if the segment's *line* meets
``W``. A segment whose start and end point both lie outside ``W`` falls into
one of two classes:

* **crossing**: the line passes through ``W`` -- the exact predicate and the
  MBR both count it (correctly);
* **corner**: the line misses ``W`` although its box overlaps ``W`` (the line
  passes by a corner of ``W``) -- only the MBR counts it. These are the
  false positives behind the MBR overcount.

For spatio-temporal queries a box method also tests space and time
separately (box overlaps ``W`` and ``[t_s, t_e]`` overlaps ``[L, U]``), so
there is a second source of overcount, **decoupling**: the line meets ``W``
and the segment's time span meets ``[L, U]``, but not at the same time. The
three nested answers are

    exact (label)  <=  decoupled (line, time separately)  <=  MBR (box, time separately)

and ``MBR - exact`` splits into ``MBR - decoupled`` (corner effect) plus
``decoupled - exact`` (decoupling). For spatial queries decoupled = exact.

Each answer is computed with ``NumpyBackend.count`` -- the code path that
reproduced every benchmark label in the P2 gate -- with only the
spatial/spatio-temporal predicate's segment mask replaced. All other
predicates, joins and the distinct-trip count are unchanged, so for multi
queries the numbers are distinct trajectories, for single queries segment
rows. The exact count must equal the stored label; a query where it does not
is reported as a mismatch and fails the merge.

Segment-level counts are taken over the segments that satisfy every other
predicate of the query (other segment predicates and, for multi queries, a
trip that satisfies the trip/agent predicates) and, for spatio-temporal
queries, whose time span overlaps ``[L, U]``.

Queries without a spatial or spatio-temporal predicate (``standard.*``,
``temporal.*``) cannot be affected and are only counted as excluded.

Provenance. ``verify`` re-checks every snapshot array against the SHA-256 in
the manifest once and writes ``snapshot_verification.json`` (with the size and
mtime of every array); ``run`` and ``merge`` refuse to start unless that record
exists for the same manifest and every array still has the recorded size and
mtime. Every results directory is bound (``binding.json``) to the snapshot
manifest, benchmark, config, shard count, statistic version, the git revision
and a SHA-256 over the source files of the computation; ``run`` refuses source
files with uncommitted changes (``--allow-dirty`` only for local experiments;
the source hash still tells two dirty states apart).

Invariants. ``merge`` fails on any label mismatch and on any query violating
``exact <= decoupled <= mbr``, the segment partition
``mbr = exact + corner + decoupled_only + endpoint_anomaly``, a nonzero
endpoint anomaly (a line with an endpoint inside ``W`` that misses ``W``) or,
for single queries, segment counts that disagree with the row counts.

Work is sharded and resumable exactly like the label gate (``snapshot_gate``):
``run`` appends one fsynced JSON line per query, ``merge`` checks completeness
and writes the report (JSON, with the ids of all overcounted queries), a
Markdown table, a CSV of every evaluated query and a CSV of only the
overcounted queries (``mbr_count > exact_count``), most overcounted first.

    python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount verify \\
        --snapshot DIR --results-dir OUT
    python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount run \\
        --snapshot DIR --config pol_v3_config.json --benchmark benchmark.jsonl \\
        --results-dir OUT --shard 0 --shards 16
    python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount merge \\
        --snapshot DIR --benchmark benchmark.jsonl --results-dir OUT --report-prefix OUT/../report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .snapshot_gate import EXIT_DEADLINE, BindingError, _done, _rows, _sha256, _shard_path, bind_results_dir

STATISTIC = "segment_mbr_overcount"
STATISTIC_VERSION = 2  # 2: corner requires both endpoints outside; provenance and invariant gates
SPATIAL_KINDS = ("spatial", "spatio_temporal")
REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFICATION_FILE = "snapshot_verification.json"
# every module whose code decides a number in the report
SOURCE_MODULES = (
    "evaluation.trajectory_pol_porto.traj_eval.mbr_overcount",
    "evaluation.trajectory_pol_porto.traj_eval.snapshot_gate",
    "evaluation.trajectory_pol_porto.traj_eval.snapshot",
    "query_generation.benchmark_v3.numpy_backend",
    "query_generation.benchmark_v3.exact",
    "query_generation.benchmark_v3.config",
    "query_generation.query_generator",
)


class ProvenanceError(RuntimeError):
    """The inputs or the code cannot be tied to a unique, verified state."""


# ---------------------------------------------------------------------------
# Per-query computation


def spatial_predicate(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The query's single spatial or spatio-temporal predicate, or None."""

    found = [p for p in record["predicates"] if p["kind"] in SPATIAL_KINDS]
    if len(found) > 1:
        raise ValueError(f"{record['query_id']}: more than one spatial predicate")
    return found[0] if found else None


def _inside(x: np.ndarray, y: np.ndarray, window: Tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = window
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def segment_classes(segments: Dict[str, np.ndarray], predicate: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Box candidates of one predicate and the exact tests on them.

    Returns ``index`` (segment positions whose MBR overlaps ``W`` and, for a
    spatio-temporal predicate, whose time span overlaps ``[L, U]``) and, aligned
    with it, boolean arrays ``line`` (the line meets ``W``), ``exact`` (the v3
    predicate), ``start_inside`` and ``end_inside``.
    """

    from query_generation.benchmark_v3 import exact
    from query_generation.benchmark_v3.numpy_backend import _bound_us

    window = tuple(float(predicate[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
    x0, y0, x1, y1 = window
    sx, sy, ex, ey = (segments[name] for name in ("sx", "sy", "ex", "ey"))
    # MBR overlap without materialising min/max arrays: max(s, e) >= a0 <=> s >= a0 or e >= a0.
    box = (sx >= x0) | (ex >= x0)
    box &= (sx <= x1) | (ex <= x1)
    box &= (sy >= y0) | (ey >= y0)
    box &= (sy <= y1) | (ey <= y1)
    lower_us = upper_us = None
    if predicate["kind"] == "spatio_temporal":
        lower_us, upper_us = _bound_us(predicate.get("lower")), _bound_us(predicate.get("upper"))
        box &= exact.temporal_mask(segments["ts"], segments["te"], lower_us, upper_us)
    index = np.flatnonzero(box)
    sub = {name: np.asarray(segments[name][index]) for name in ("sx", "sy", "ex", "ey")}
    line = exact.spatial_mask(sub["sx"], sub["sy"], sub["ex"], sub["ey"], window)
    if predicate["kind"] == "spatio_temporal":
        ts, te = np.asarray(segments["ts"][index]), np.asarray(segments["te"][index])
        coupled = exact.spatio_temporal_mask(sub["sx"], sub["sy"], sub["ex"], sub["ey"], ts, te,
                                             window, lower_us, upper_us)
    else:
        coupled = line
    return {
        "index": index,
        "line": line,
        "exact": coupled,
        "start_inside": _inside(sub["sx"], sub["sy"], window),
        "end_inside": _inside(sub["ex"], sub["ey"], window),
    }


def _backend_class():
    from query_generation.benchmark_v3.numpy_backend import NumpyBackend

    class OverrideBackend(NumpyBackend):
        """NumpyBackend whose mask for one predicate object can be replaced."""

        _override: Optional[Tuple[Dict[str, Any], np.ndarray]] = None

        def predicate_mask(self, predicate):
            if self._override is not None and predicate is self._override[0]:
                return self._override[1]
            return super().predicate_mask(predicate)

        def count_with(self, record, predicate, mask) -> int:
            self._override = (predicate, mask)
            try:
                return int(self.count(record))
            finally:
                self._override = None

        def qualifying_entities(self, record, skip_table: str) -> Optional[np.ndarray]:
            """Entity keys satisfying every table's predicates except ``skip_table`` (multi only)."""

            if record["category"]["relation"] == "single":
                return None
            entity = self.settings.entity_table
            entity_key = self.settings.base["entity"]["key"]
            mask = self.table_mask(entity, record["predicates"])
            for other in record["tables"]:
                if other in (entity, skip_table):
                    continue
                other_col, entity_col = self.join_keys[(other, entity)]
                matching = np.asarray(self.data[other][other_col])[self.table_mask(other, record["predicates"])]
                mask &= np.isin(np.asarray(self.data[entity][entity_col]), matching)
            return np.asarray(self.data[entity][entity_key])[mask]

    return OverrideBackend


def evaluate_query(backend, record: Dict[str, Any]) -> Dict[str, Any]:
    predicate = spatial_predicate(record)
    if predicate is None:
        raise ValueError(f"{record['query_id']} has no spatial predicate")
    table = predicate["table"]
    segments = backend.data[table]
    rows = len(segments["sx"])
    classes = segment_classes(segments, predicate)
    index = classes["index"]

    def full(selection: np.ndarray) -> np.ndarray:
        mask = np.zeros(rows, dtype=bool)
        mask[index[selection]] = True
        return mask

    everything = np.ones(index.size, dtype=bool)
    exact_count = backend.count_with(record, predicate, full(classes["exact"]))
    mbr_count = backend.count_with(record, predicate, full(everything))
    if predicate["kind"] == "spatio_temporal":
        decoupled_count = backend.count_with(record, predicate, full(classes["line"]))
    else:
        decoupled_count = exact_count

    # segments that satisfy every other predicate of the query
    relevant = np.ones(index.size, dtype=bool)
    for other in record["predicates"]:
        if other is not predicate and other["table"] == table:
            relevant &= np.asarray(backend.predicate_mask(other))[index]
    entities = backend.qualifying_entities(record, skip_table=table)
    if entities is not None:
        join_col, _ = backend.join_keys[(table, backend.settings.entity_table)]
        relevant &= np.isin(np.asarray(segments[join_col][index]), entities)

    line, coupled = classes["line"], classes["exact"]
    both_outside = ~classes["start_inside"] & ~classes["end_inside"]
    count = lambda mask: int(np.count_nonzero(relevant & mask))
    label = int(record["cardinality"])
    result = {
        "query_id": record["query_id"],
        "predicate_hash": record["predicate_hash"],
        "category": record["category"]["key"],
        "dimension": record["category"]["dimension"],
        "interval": record["category"]["interval"],
        "relation": record["category"]["relation"],
        "label_kind": record["label_kind"],
        "addon_predicates": int(record.get("addon_predicates") or 0),
        "spatial_mode": predicate.get("spatial_mode", predicate.get("mode")),
        "window_area_m2": (float(predicate["max_x"]) - float(predicate["min_x"]))
        * (float(predicate["max_y"]) - float(predicate["min_y"])),
        "label": label,
        "exact_count": exact_count,
        "decoupled_count": decoupled_count,
        "mbr_count": mbr_count,
        "match": exact_count == label,
        "overcounted": mbr_count > exact_count,
        "overcount": mbr_count - exact_count,
        "overcount_corner": mbr_count - decoupled_count,
        "overcount_decoupling": decoupled_count - exact_count,
        "ratio_mbr_to_exact": mbr_count / exact_count if exact_count else None,
        "segments_mbr": count(everything),
        "segments_exact": count(coupled),
        "segments_corner": count(both_outside & ~line),
        "segments_decoupled_only": count(line & ~coupled),
        "segments_both_endpoints_outside": count(both_outside),
        "segments_crossing": count(both_outside & line),
        # sanity: a line with an endpoint inside W always meets W (should be 0)
        "anomaly_endpoint_inside_line_misses": count(~both_outside & ~line),
    }
    if not result["match"]:
        result["reference_count"] = int(backend.count(record))  # unmodified label path, for diagnosis
    return result


def invariant_violations(item: Dict[str, Any]) -> List[str]:
    """Checks every stored result must pass; any violation fails the merge."""

    problems = []
    if not item["exact_count"] <= item["decoupled_count"] <= item["mbr_count"]:
        problems.append("nesting exact <= decoupled <= mbr")
    if item["dimension"] == "spatial" and item["decoupled_count"] != item["exact_count"]:
        problems.append("spatial query with decoupled != exact")
    if item["anomaly_endpoint_inside_line_misses"]:
        problems.append("line with an endpoint inside the window misses it")
    if item["segments_mbr"] != (item["segments_exact"] + item["segments_corner"] + item["segments_decoupled_only"]
                                + item["anomaly_endpoint_inside_line_misses"]):
        problems.append("segment partition mbr = exact + corner + decoupled_only + anomaly")
    if not item["segments_crossing"] + item["segments_corner"] <= item["segments_both_endpoints_outside"]:
        problems.append("crossing + corner exceed the segments with both endpoints outside")
    if item["relation"] == "single" and (item["segments_exact"] != item["exact_count"]
                                         or item["segments_mbr"] != item["mbr_count"]):
        problems.append("single query whose segment counts differ from its row counts")
    if item["dimension"] == "spatial" and item["segments_decoupled_only"]:
        problems.append("spatial query with decoupling-only segments")
    return problems


# ---------------------------------------------------------------------------
# Sharded, resumable run


def source_files() -> List[Path]:
    return [Path(importlib.import_module(name).__file__).resolve() for name in SOURCE_MODULES]


def source_sha256() -> str:
    """SHA-256 over the relative paths and contents of :data:`SOURCE_MODULES`."""

    digest = hashlib.sha256()
    for path in sorted(source_files()):
        digest.update(str(path.relative_to(REPO_ROOT)).encode() + b"\0")
        digest.update(_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def code_state() -> Dict[str, Any]:
    """Git revision and whether any source file differs from it (None: not a git checkout)."""

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
        raise ProvenanceError(
            "the source files are not a committed git state "
            f"({state}); commit them, or pass --allow-dirty for a local experiment")
    return {
        "statistic": STATISTIC,
        "statistic_version": STATISTIC_VERSION,
        **state,
        "source_sha256": source_sha256(),
        "snapshot_manifest_sha256": _sha256(Path(snapshot) / "manifest.json"),
        "benchmark_sha256": _sha256(Path(benchmark)),
        "config_sha256": config_sha256,
        "shards": int(shards),
    }


def _loaded_arrays(snapshot: Path) -> List[str]:
    """Manifest paths of every array ``load_snapshot`` reads."""

    from .snapshot import read_manifest

    manifest = read_manifest(snapshot)
    return sorted(f"tables/{table_id}/{column}.npy"
                  for table_id, table in manifest["tables"].items() for column in table["columns"])


def _stats(snapshot: Path, relative: Iterable[str]) -> Dict[str, List[int]]:
    stats = {}
    for name in relative:
        info = (Path(snapshot) / name).stat()
        stats[name] = [info.st_size, info.st_mtime_ns]
    return stats


def verify(snapshot: Path, results_dir: Path, log=print) -> Dict[str, Any]:
    """Check every snapshot array against its manifest checksum; record the result."""

    from .snapshot import read_manifest, verify_snapshot

    snapshot = Path(snapshot)
    started = time.time()
    problems = verify_snapshot(snapshot)
    listed = set(read_manifest(snapshot)["files"])
    loaded = _loaded_arrays(snapshot)
    problems += [f"{name} is read but has no checksum in the manifest" for name in loaded if name not in listed]
    record = {
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json"),
        "passed": not problems,
        "problems": problems[:200],
        "files_checked": len(listed),
        "array_stats": _stats(snapshot, sorted(listed)),
        "seconds": round(time.time() - started, 2),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / VERIFICATION_FILE).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    log(f"snapshot verification: passed={record['passed']} files={len(listed)} problems={len(problems)}")
    return record


def require_verified(snapshot: Path, results_dir: Path) -> Dict[str, Any]:
    """Refuse unless ``verify`` passed for this manifest and no array changed since (size, mtime)."""

    path = results_dir / VERIFICATION_FILE
    if not path.exists():
        raise ProvenanceError(f"{path} is missing; run the verify step first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("passed"):
        raise ProvenanceError(f"snapshot verification failed: {record.get('problems')}")
    if record.get("snapshot_manifest_sha256") != _sha256(Path(snapshot) / "manifest.json"):
        raise ProvenanceError("snapshot verification was done for another manifest")
    loaded = _loaded_arrays(snapshot)
    missing = [name for name in loaded if name not in record["array_stats"]]
    changed = [name for name, stat in _stats(snapshot, loaded).items() if record["array_stats"].get(name) != stat]
    if missing or changed:
        raise ProvenanceError(f"snapshot arrays changed since verification: {(missing + changed)[:5]}")
    return record


def selected_rows(benchmark: Path) -> List[Dict[str, Any]]:
    return [row for row in _rows(benchmark) if spatial_predicate(row) is not None]


def run_shard(snapshot: Path, config: Path, benchmark: Path, results_dir: Path, shard: int, shards: int,
              deadline_seconds: Optional[float] = None, log=print, *, allow_dirty: bool = False) -> int:
    from query_generation.benchmark_v3.config import load_settings

    from .snapshot import load_snapshot

    if not 0 <= shard < shards:
        raise ValueError("--shard must be in [0, --shards)")
    started = time.time()
    require_verified(snapshot, results_dir)
    settings = load_settings(config)
    bind_results_dir(results_dir, binding(snapshot, benchmark, settings.sha256, shards, allow_dirty=allow_dirty))
    data = load_snapshot(snapshot, expected_config_sha256=settings.sha256)
    backend = _backend_class()(settings, data)
    rows = [row for position, row in enumerate(selected_rows(benchmark)) if position % shards == shard]
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
            for problem in invariant_violations(item):
                log(f"INVARIANT {item['query_id']}: {problem}")
    log(f"shard {shard}/{shards} finished")
    return 0


# ---------------------------------------------------------------------------
# Merge and report


def _quantiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {key: None for key in ("p50", "p90", "p99", "max", "mean")}
    array = np.asarray(values, dtype=float)
    return {
        "p50": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(items)
    n = len(items)
    share = lambda k: (k / n) if n else None
    over = [i for i in items if i["mbr_count"] > i["exact_count"]]
    corner_queries = [i for i in items if i["mbr_count"] > i["decoupled_count"]]
    decoupling_queries = [i for i in items if i["decoupled_count"] > i["exact_count"]]
    with_corner_segments = [i for i in items if i["segments_corner"] > 0]
    with_crossing_segments = [i for i in items if i["segments_crossing"] > 0]
    sums = {key: int(sum(i[key] for i in items)) for key in (
        "label", "exact_count", "decoupled_count", "mbr_count", "segments_mbr", "segments_exact",
        "segments_corner", "segments_decoupled_only", "segments_both_endpoints_outside", "segments_crossing",
        "anomaly_endpoint_inside_line_misses")}
    return {
        "queries": n,
        "queries_overcounted": len(over),
        "queries_overcounted_share": share(len(over)),
        "queries_overcounted_by_corner_segments": len(corner_queries),
        "queries_overcounted_by_corner_segments_share": share(len(corner_queries)),
        "queries_overcounted_by_decoupling": len(decoupling_queries),
        "queries_with_corner_segments": len(with_corner_segments),
        "queries_with_corner_segments_share": share(len(with_corner_segments)),
        "queries_with_crossing_segments": len(with_crossing_segments),
        "sums": sums,
        "segments_corner_share_of_mbr": (sums["segments_corner"] / sums["segments_mbr"]) if sums["segments_mbr"] else None,
        "overcount_total": sums["mbr_count"] - sums["exact_count"],
        "overcount_corner_total": sums["mbr_count"] - sums["decoupled_count"],
        "overcount_decoupling_total": sums["decoupled_count"] - sums["exact_count"],
        "ratio_mbr_to_exact": _quantiles([i["mbr_count"] / i["exact_count"] for i in items]),
        "ratio_mbr_to_decoupled": _quantiles([i["mbr_count"] / i["decoupled_count"] for i in items]),
        "relative_overcount": _quantiles([(i["mbr_count"] - i["exact_count"]) / i["exact_count"] for i in items]),
    }


CSV_FIELDS = ["query_id", "category", "label_kind", "addon_predicates", "spatial_mode", "window_area_m2", "label",
              "exact_count", "decoupled_count", "mbr_count", "overcounted", "overcount", "overcount_corner",
              "overcount_decoupling", "ratio_mbr_to_exact", "segments_mbr", "segments_exact", "segments_corner",
              "segments_decoupled_only", "segments_both_endpoints_outside", "segments_crossing",
              "anomaly_endpoint_inside_line_misses", "match", "predicate_hash"]


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Segment-MBR overcount on the spatial part of the benchmark",
        "",
        f"Benchmark `{report['benchmark']}` (sha256 `{report['benchmark_sha256'][:16]}…`), "
        f"snapshot manifest sha256 `{report['snapshot_manifest_sha256'][:16]}…`, "
        f"code `{report['binding']['code_revision']}` (source sha256 `{report['binding']['source_sha256'][:16]}…`), "
        f"statistic v{report['binding']['statistic_version']}; snapshot arrays verified against the manifest "
        f"({report['snapshot_verification']['files_checked']} files).",
        f"{report['queries_total']} queries in the benchmark, {report['queries_evaluated']} with a spatial or "
        f"spatio-temporal predicate (the others cannot be affected). Exact count = stored label for "
        f"{report['queries_evaluated'] - report['mismatch_count']} of {report['queries_evaluated']}; "
        f"invariant violations: {report['invariant_violation_count']}.",
        "",
        "Counts are distinct trajectories for multi queries and segment rows for single queries. "
        "*Corner* segments: start and end outside the window, line misses it, box overlaps it. "
        "*Crossing* segments: start and end outside, line passes through (counted correctly by both).",
        "",
        "| Group | Queries | Overcounted (MBR > exact) | … by corner segments | … by decoupling | "
        "With ≥1 corner segment | Corner segments / MBR segments | Crossing segments | MBR/exact p50 | p90 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in report["groups"].items():
        s = group
        lines.append(
            f"| {name} | {s['queries']} | {s['queries_overcounted']} ({100 * s['queries_overcounted_share']:.1f} %) | "
            f"{s['queries_overcounted_by_corner_segments']} | {s['queries_overcounted_by_decoupling']} | "
            f"{s['queries_with_corner_segments']} | {s['sums']['segments_corner']:,} / {s['sums']['segments_mbr']:,} | "
            f"{s['sums']['segments_crossing']:,} | {_fmt(s['ratio_mbr_to_exact']['p50'])} | "
            f"{_fmt(s['ratio_mbr_to_exact']['p90'])} | {_fmt(s['ratio_mbr_to_exact']['p99'])} | "
            f"{_fmt(s['ratio_mbr_to_exact']['max'])} |")
    return "\n".join(lines) + "\n"


def merge(benchmark: Path, results_dir: Path, report_prefix: Path, snapshot: Path) -> Dict[str, Any]:
    binding_path = results_dir / "binding.json"
    if not binding_path.exists():
        raise BindingError(f"{results_dir} has no binding.json")
    bound = json.loads(binding_path.read_text(encoding="utf-8"))
    if bound.get("statistic") != STATISTIC or bound.get("statistic_version") != STATISTIC_VERSION:
        raise BindingError(f"{results_dir} holds results of another statistic or version")
    verification = require_verified(snapshot, results_dir)
    for key, value in (("benchmark_sha256", _sha256(benchmark)),
                       ("snapshot_manifest_sha256", _sha256(Path(snapshot) / "manifest.json"))):
        if bound.get(key) != value:
            raise BindingError(f"{results_dir} was produced for other inputs ({key} differs)")
    all_rows = _rows(benchmark)
    expected = {row["query_id"]: row for row in all_rows if spatial_predicate(row) is not None}
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
    foreign = sorted(q for q, item in found.items() if q in expected and item["predicate_hash"] != expected[q]["predicate_hash"])
    mismatches = sorted((item for item in found.values() if not item["match"]), key=lambda item: item["query_id"])
    items = [found[q] for q in sorted(found) if q in expected]
    violations = [{"query_id": item["query_id"], "problems": problems}
                  for item in items for problems in [invariant_violations(item)] if problems]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[f"all {item['relation']}"].append(item)
        groups[f"{item['dimension']} {item['relation']}"].append(item)
        groups[item["category"]].append(item)
        if item["relation"] == "multi":
            groups[f"{item['dimension']} multi, {item['addon_predicates']} add-on(s)"].append(item)
    order = sorted(groups, key=lambda name: (not name.startswith("all"), name.count("."), name))
    report = {
        "passed": not (missing or unexpected or duplicates or foreign or mismatches or violations),
        "statistic": STATISTIC,
        "binding": bound,
        "merge_code_state": {**code_state(), "source_sha256": source_sha256()},
        "snapshot_verification": {key: verification[key] for key in (
            "passed", "files_checked", "snapshot_manifest_sha256", "verified_at_utc", "seconds")},
        "benchmark": str(benchmark),
        "benchmark_sha256": bound["benchmark_sha256"],
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": bound["snapshot_manifest_sha256"],
        "queries_total": len(all_rows),
        "queries_evaluated": len(expected),
        "queries_excluded_without_spatial_predicate": len(all_rows) - len(expected),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
        "missing_count": len(missing),
        "missing": missing[:200],
        "unexpected": unexpected[:200],
        "duplicates": duplicates[:200],
        "predicate_hash_mismatches": foreign[:200],
        "invariant_violation_count": len(violations),
        "invariant_violations": violations[:200],
        "groups": {name: summarize(groups[name]) for name in order},
        "overcounted_query_ids": [item["query_id"] for item in items if item["mbr_count"] > item["exact_count"]],
        "seconds_total": round(sum(item.get("seconds", 0.0) for item in items), 2),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "exact_count": "label semantics (pol_exact_st_v3), recomputed on the snapshot; must equal the label",
            "decoupled_count": "segment line meets the window and segment time span meets the interval, tested separately",
            "mbr_count": "segment MBR overlaps the window (and time span meets the interval for spatio-temporal queries)",
            "segments_corner": "both endpoints outside the window, line misses it, MBR overlaps it (MBR false positive)",
            "anomaly_endpoint_inside_line_misses": "an endpoint inside the window but the line misses it; must be 0",
            "segments_crossing": "both endpoints outside the window, line passes through it (true positive)",
            "segment_scope": "segments satisfying every other predicate of the query (and the interval, if spatio-temporal)",
        },
    }
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{report_prefix}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    Path(f"{report_prefix}.md").write_text(markdown(report), encoding="utf-8")
    with Path(f"{report_prefix}_per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    overcounted = sorted((item for item in items if item["mbr_count"] > item["exact_count"]),
                         key=lambda item: (-item["mbr_count"] / item["exact_count"], item["query_id"]))
    with Path(f"{report_prefix}_overcounted.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(overcounted)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Segment-MBR overcount of the benchmark's spatial predicates")
    sub = parser.add_subparsers(dest="command", required=True)
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
    run.add_argument("--allow-dirty", action="store_true",
                     help="accept uncommitted source changes (local experiments only; recorded in the binding)")
    combine = sub.add_parser("merge")
    combine.add_argument("--snapshot", required=True, type=Path)
    combine.add_argument("--benchmark", required=True, type=Path)
    combine.add_argument("--results-dir", required=True, type=Path)
    combine.add_argument("--report-prefix", required=True, type=Path,
                         help="writes <prefix>.json, <prefix>.md, <prefix>_per_query.csv and <prefix>_overcounted.csv")
    args = parser.parse_args(argv)
    log = lambda message: print(message, flush=True)
    if args.command == "verify":
        return 0 if verify(args.snapshot, args.results_dir, log)["passed"] else 1
    if args.command == "run":
        return run_shard(args.snapshot, args.config, args.benchmark, args.results_dir, args.shard, args.shards,
                         args.deadline_seconds, log, allow_dirty=args.allow_dirty)
    report = merge(args.benchmark, args.results_dir, args.report_prefix, args.snapshot)
    log(json.dumps({key: report[key] for key in ("passed", "queries_total", "queries_evaluated", "mismatch_count",
                                                  "missing_count", "invariant_violation_count")}))
    log(markdown(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
