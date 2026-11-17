"""Data-shift ablation (decided 2026-11-17): the own model trained on 80 % of the trips.

The literature's update experiments (DeepDB, Naru, DDUp, "Are we ready", STATS-CEB)
train on part of the data and evaluate against the data as it is at query time.
Here the model is trained on 80 % of the trips and evaluated on the unchanged
benchmark, whose labels are counts on 100 % of the data: a *stale* model asked
about a database that has grown by the remaining 20 %. Two splits, as in DeepDB:

- ``random``: 80 % of the trips, drawn with a fixed seed (the new data come from
  the same distribution);
- ``temporal``: the earliest 80 % of the trips by trip start time (the new data
  are the latest trips, an append-only workload).

A trip is kept or dropped together with all its segments; every other table
(POL's agents) is kept whole. This module

- ``select``: chooses the kept trips from the evaluation snapshot and records the
  split (counts, temporal cutoff, both full-outer-join cardinalities);
- ``subset-snapshot``: writes the snapshot of the kept data (same format; the
  gate and every snapshot reader work on it unchanged);
- ``relabel``: turns the gate's counts of the benchmark on that snapshot
  (``snapshot_gate run`` with the subset snapshot) into a benchmark with the
  in-distribution labels: the same queries, labelled on the 80 % data (the
  control that separates "the model learned less" from "the data changed").

The model side (the 80 % preparation and fine-tuning) lives on branch
``model/data-shift``; it reads ``kept_trip_ids.npy`` written here.

    python -m evaluation.trajectory_pol_porto.traj_eval.data_shift select \\
        --snapshot SNAP --config pol_v3_config.json --mode temporal --output DIR
    python -m evaluation.trajectory_pol_porto.traj_eval.data_shift subset-snapshot \\
        --snapshot SNAP --config pol_v3_config.json --split DIR --output DIR/snapshot
    python -m evaluation.trajectory_pol_porto.traj_eval.data_shift relabel \\
        --benchmark benchmark.jsonl --results-dir GATE_DIR --split DIR --snapshot DIR/snapshot \\
        --output DIR/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SPLIT_VERSION = 1
FRACTION = 0.8
MODES = ("random", "temporal")
SPLIT_FILE = "split.json"
KEPT_FILE = "kept_trip_ids.npy"
ALL_TRIPS_FILE = "all_trip_ids.npy"
PARENTS_FILE = "all_trip_parent_ids.npy"  # aligned with ALL_TRIPS_FILE (POL: agent_id)


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------


def entity_parent(settings: Any) -> Optional[Tuple[str, str]]:
    """``(parent table, key column in the entity table)`` of the join edge into the entity table, if any."""

    entity = settings.entity_table
    aliases = {table_id: table["alias"] for table_id, table in settings.base["tables"].items()}
    for join in settings.base["joins"]:
        if join["right"] != entity:
            continue
        for side in join["condition"].split("="):
            alias, column = side.strip().split(".")
            if alias == aliases[entity]:
                return join["left"], column
        raise ValueError(f"cannot read the entity key from join condition {join['condition']!r}")
    return None


def entity_start_column(settings: Any) -> str:
    """The snapshot column with the entity's start time (the temporal split key)."""

    table = settings.base["tables"][settings.entity_table]
    names = [a["name"] for a in table["attributes"] if a.get("type") == "temporal_interval"]
    if len(names) != 1:
        raise ValueError(f"expected one temporal interval on {settings.entity_table}, found {names}")
    return f"{names[0]}.start"


def full_outer_join_cardinality(segment_count_per_trip: np.ndarray, trip_parent_index: Optional[np.ndarray],
                                parent_count: int) -> int:
    """Rows of the root-anchored full outer join parent -> trips -> segments.

    A trip contributes ``max(1, segments)`` rows, a parent without trips one row
    (NeuroCard's full outer join, as the model's preparation builds it).
    """

    per_trip = np.maximum(np.asarray(segment_count_per_trip, dtype=np.int64), 1)
    if trip_parent_index is None:
        return int(per_trip.sum())
    trips_per_parent = np.bincount(trip_parent_index, minlength=parent_count)
    return int(per_trip.sum() + np.count_nonzero(trips_per_parent == 0))


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def _trip_tables(snapshot: Path, settings: Any):
    from .snapshot import load_snapshot, load_trip_index

    data = load_snapshot(snapshot, expected_config_sha256=settings.sha256)
    entity = settings.entity_table
    key = settings.base["entity"]["key"]
    trips = data[entity]
    trip_ids = np.asarray(trips[key], dtype=np.int64)
    if np.any(np.diff(trip_ids) <= 0):
        raise ValueError(f"{entity}.{key} is not strictly increasing in the snapshot")
    index = load_trip_index(snapshot)
    segment_counts = np.zeros(trip_ids.size, dtype=np.int64)
    position = np.searchsorted(trip_ids, np.asarray(index["trip_id"], dtype=np.int64))
    if np.any(position >= trip_ids.size) or np.any(trip_ids[np.minimum(position, trip_ids.size - 1)] != index["trip_id"]):
        raise ValueError("segments reference trips that are not in the entity table")
    segment_counts[position] = np.asarray(index["segment_count"], dtype=np.int64)
    parent = entity_parent(settings)
    parent_index = parent_count = None
    if parent is not None:
        parent_table, parent_key = parent
        parent_ids = np.asarray(data[parent_table][settings.base["tables"][parent_table]["primary_key"]], dtype=np.int64)
        if np.any(np.diff(parent_ids) <= 0):
            raise ValueError(f"{parent_table} primary key is not strictly increasing in the snapshot")
        values = np.asarray(trips[parent_key], dtype=np.int64)
        parent_index = np.searchsorted(parent_ids, values)
        if np.any(parent_index >= parent_ids.size) or np.any(parent_ids[np.minimum(parent_index, parent_ids.size - 1)] != values):
            raise ValueError(f"{entity}.{parent_key} references missing {parent_table} rows")
        parent_count = int(parent_ids.size)
    return data, trip_ids, segment_counts, parent_index, parent_count


def select(snapshot: Path, config: Path, mode: str, *, fraction: float = FRACTION, seed: int = 0) -> Dict[str, Any]:
    """The kept trips (sorted ids) and the split record."""

    from query_generation.benchmark_v3.config import load_settings

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    settings = load_settings(config)
    data, trip_ids, segment_counts, parent_index, parent_count = _trip_tables(snapshot, settings)
    total = int(trip_ids.size)
    keep_count = int(round(fraction * total))
    record: Dict[str, Any] = {"mode": mode, "fraction": fraction}
    if mode == "random":
        order = np.random.default_rng(seed).permutation(total)
        kept = np.zeros(total, dtype=bool)
        kept[order[:keep_count]] = True
        record["seed"] = seed
    else:
        start_column = entity_start_column(settings)
        start = np.asarray(data[settings.entity_table][start_column], dtype=np.int64)
        order = np.lexsort((trip_ids, start))  # earliest start first, ties by trip id
        kept = np.zeros(total, dtype=bool)
        kept[order[:keep_count]] = True
        cutoff = int(start[order[keep_count - 1]])
        record.update(start_column=start_column, cutoff_start_us=cutoff,
                      cutoff_start=_timestamp(cutoff),
                      first_dropped_start=_timestamp(int(start[order[keep_count]])) if keep_count < total else None)
    kept_parent_count = None
    lost_parents = 0
    if parent_index is not None:
        before = np.bincount(parent_index, minlength=parent_count)
        after = np.bincount(parent_index[kept], minlength=parent_count)
        lost_parents = int(np.count_nonzero((before > 0) & (after == 0)))
        kept_parent_count = int(np.count_nonzero(after > 0))
    record.update({
        "split_version": SPLIT_VERSION,
        "entity_table": settings.entity_table,
        "parent": list(entity_parent(settings)) if entity_parent(settings) else None,
        "trips_total": total, "trips_kept": int(kept.sum()),
        "segments_total": int(segment_counts.sum()), "segments_kept": int(segment_counts[kept].sum()),
        "parents_with_trips_kept": kept_parent_count,
        "parents_losing_every_trip": lost_parents,
        "full_join_cardinality_full": full_outer_join_cardinality(segment_counts, parent_index, parent_count or 0),
        "full_join_cardinality_subset": full_outer_join_cardinality(
            segment_counts[kept], None if parent_index is None else parent_index[kept], parent_count or 0),
        "config_sha256": settings.sha256,
        "snapshot": str(snapshot), "snapshot_manifest_sha256": _sha256(Path(snapshot) / "manifest.json"),
    })
    parent_ids = None
    if parent_index is not None:
        parent_table = record["parent"][0]
        key = settings.base["tables"][parent_table]["primary_key"]
        parent_ids = np.asarray(data[parent_table][key], dtype=np.int64)[parent_index]
    return {"kept_trip_ids": trip_ids[kept], "record": record, "all_trip_ids": trip_ids,
            "all_trip_parent_ids": parent_ids}


def write_split(output: Path, selection: Dict[str, Any]) -> Dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new directory")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.save(partial / KEPT_FILE, np.asarray(selection["kept_trip_ids"], dtype=np.int64))
    record = dict(selection["record"])
    record["kept_trip_ids_sha256"] = _sha256(partial / KEPT_FILE)
    # for the model-side preparation (branch model/data-shift): every trip, and its parent row
    np.save(partial / ALL_TRIPS_FILE, np.asarray(selection["all_trip_ids"], dtype=np.int64))
    record["all_trip_ids_sha256"] = _sha256(partial / ALL_TRIPS_FILE)
    if selection.get("all_trip_parent_ids") is not None:
        np.save(partial / PARENTS_FILE, np.asarray(selection["all_trip_parent_ids"], dtype=np.int64))
        record["all_trip_parent_ids_sha256"] = _sha256(partial / PARENTS_FILE)
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    (partial / SPLIT_FILE).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return record


def read_split(directory: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    directory = Path(directory)
    record = json.loads((directory / SPLIT_FILE).read_text(encoding="utf-8"))
    kept = np.load(directory / KEPT_FILE)
    if _sha256(directory / KEPT_FILE) != record["kept_trip_ids_sha256"]:
        raise ValueError(f"{directory / KEPT_FILE} does not match its split record")
    return record, kept


# ---------------------------------------------------------------------------
# subset snapshot
# ---------------------------------------------------------------------------


def subset_snapshot(snapshot: Path, config: Path, split: Path, output: Path) -> Dict[str, Any]:
    """The snapshot of the kept trips (and all their segments; other tables whole)."""

    from query_generation.benchmark_v3.config import load_settings

    from . import snapshot as snap

    settings = load_settings(config)
    record, kept = read_split(split)
    if record["config_sha256"] != settings.sha256:
        raise ValueError("the split was made under another v3 config")
    if record["snapshot_manifest_sha256"] != _sha256(Path(snapshot) / "manifest.json"):
        raise ValueError("the split was made on another snapshot")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new directory")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    manifest = snap.read_manifest(snapshot)
    entity = settings.entity_table
    key = settings.base["entity"]["key"]
    segments = settings.segments
    segment_trip_key = segments.trip_key.split(".")[-1]
    tables = {}
    for table_id, table in manifest["tables"].items():
        directory = Path(snapshot) / "tables" / table_id
        if table_id == entity:
            mask = np.isin(np.load(directory / f"{key}.npy", mmap_mode="r"), kept)
        elif table_id == segments.table:
            mask = np.isin(np.load(directory / f"{segment_trip_key}.npy", mmap_mode="r"), kept)
        else:
            mask = None
        rows = None
        for name in table["columns"]:
            source = snap._column_path(Path(snapshot), table_id, name)
            target = partial / source.relative_to(Path(snapshot))
            target.parent.mkdir(parents=True, exist_ok=True)
            if mask is None:
                shutil.copyfile(source, target)
                rows = int(np.load(source, mmap_mode="r").shape[0])
            else:
                array = np.load(source, mmap_mode="r")
                np.save(target, np.asarray(array[mask]))
                rows = int(mask.sum())
        tables[table_id] = {**table, "rows": rows if rows is not None else table.get("rows")}
    if tables[entity]["rows"] != record["trips_kept"]:
        raise ValueError(f"kept {tables[entity]['rows']} trips, the split records {record['trips_kept']}")
    if tables[segments.table]["rows"] != record["segments_kept"]:
        raise ValueError(f"kept {tables[segments.table]['rows']} segments, the split records {record['segments_kept']}")
    derived = snap.derive_trips(settings, partial)
    files = {}
    for path in sorted(partial.rglob("*.npy")):
        array = np.load(path, mmap_mode="r")
        files[str(path.relative_to(partial))] = {"sha256": _sha256(path), "bytes": path.stat().st_size,
                                                 "dtype": str(array.dtype), "shape": list(array.shape)}
    subset_manifest = {**{k: v for k, v in manifest.items() if k not in ("tables", "files", "derived", "checks")},
                       "tables": tables, "derived": derived, "files": files,
                       "subset": {**record, "source_snapshot": str(snapshot),
                                  "source_manifest_sha256": _sha256(Path(snapshot) / "manifest.json"),
                                  "split_directory": str(split)},
                       "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (partial / "manifest.json").write_text(json.dumps(subset_manifest, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return subset_manifest


# ---------------------------------------------------------------------------
# relabel (in-distribution control)
# ---------------------------------------------------------------------------


def relabel(benchmark: Path, results_dir: Path, split: Path, snapshot: Path, output: Path) -> Dict[str, Any]:
    """The benchmark with every label replaced by the gate's count on the subset snapshot.

    The gate results must be bound (``binding.json``) to this benchmark and to
    ``snapshot``, and ``snapshot`` must be the subset snapshot of ``split``.
    """

    from .snapshot_gate import BINDING_FILE, _done
    from .workloads import LABEL_FIELD

    record, _ = read_split(split)
    subset = json.loads((Path(snapshot) / "manifest.json").read_text(encoding="utf-8")).get("subset") or {}
    if subset.get("kept_trip_ids_sha256") != record["kept_trip_ids_sha256"]:
        raise ValueError(f"{snapshot} is not the subset snapshot of split {split}")
    binding = json.loads((Path(results_dir) / BINDING_FILE).read_text(encoding="utf-8"))
    if binding.get("benchmark_sha256") != _sha256(benchmark):
        raise ValueError("the gate results belong to another benchmark")
    if binding.get("snapshot_manifest_sha256") != _sha256(Path(snapshot) / "manifest.json"):
        raise ValueError("the gate results were counted on another snapshot")
    rows = [json.loads(line) for line in Path(benchmark).read_text(encoding="utf-8").splitlines() if line.strip()]
    counted: Dict[str, Dict[str, Any]] = {}
    shards = sorted(Path(results_dir).glob("shard_*.jsonl"))
    if len(shards) != int(binding["shards"]):
        raise ValueError(f"{len(shards)} shard files, the gate ran {binding['shards']} shards")
    for path in shards:
        for query_id, item in _done(path).items():
            if query_id in counted:
                raise ValueError(f"query {query_id} counted twice")
            counted[query_id] = item
    missing = [row["query_id"] for row in rows if row["query_id"] not in counted]
    if missing:
        raise ValueError(f"{len(missing)} queries not counted (e.g. {missing[:3]}); run every gate shard first")
    if len(counted) != len(rows):
        raise ValueError("the gate results hold queries that are not in the benchmark")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    relabelled, zero, changed = [], 0, 0
    for row in rows:
        item = counted[row["query_id"]]
        if item["predicate_hash"] != row["predicate_hash"]:
            raise ValueError(f"query {row['query_id']}: gate result for other predicates")
        if int(item["expected"]) != int(row["cardinality"]):
            raise ValueError(f"query {row['query_id']}: gate ran against other labels")
        value = int(item["counted"])
        zero += value == 0
        changed += value != int(row["cardinality"])
        new = dict(row)
        new["cardinality"] = value
        new[LABEL_FIELD[row["label_kind"]]] = value
        new["data_shift_relabel"] = {"split": record["mode"], "full_data_cardinality": int(row["cardinality"])}
        relabelled.append(new)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in relabelled), encoding="utf-8")
    temporary.rename(output)
    summary = {"queries": len(rows), "labels_changed": changed, "zero_labels": zero,
               "split": record["mode"], "benchmark": str(benchmark), "output": str(output),
               "output_sha256": _sha256(output)}
    (output.with_name(output.stem + "_relabel.json")).write_text(json.dumps(summary, indent=2, sort_keys=True),
                                                                encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# helpers and CLI
# ---------------------------------------------------------------------------


def _timestamp(microseconds: int) -> str:
    return (datetime(1970, 1, 1) + timedelta(microseconds=int(microseconds))).isoformat(sep=" ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--fraction", type=float, default=FRACTION)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("subset-snapshot")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("relabel")
    p.add_argument("--benchmark", required=True, type=Path)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--snapshot", required=True, type=Path, help="the subset snapshot the gate counted on")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "select":
        record = write_split(args.output, select(args.snapshot, args.config, args.mode,
                                                 fraction=args.fraction, seed=args.seed))
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "subset-snapshot":
        manifest = subset_snapshot(args.snapshot, args.config, args.split, args.output)
        print(json.dumps({"tables": {k: v.get("rows") for k, v in manifest["tables"].items()},
                          "subset": manifest["subset"]["mode"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps(relabel(args.benchmark, args.results_dir, args.split, args.snapshot, args.output),
                         indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
