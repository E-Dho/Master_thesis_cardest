#!/usr/bin/env python3
"""Data-shift ablation (decided 2026-11-17): the full preparation restricted to 80 % of the trips.

The own model is trained on 80 % of the trips and evaluated on the unchanged
benchmark (labels on 100 % of the data): a stale model. To make fine-tuning on
the full data possible afterwards (the update strategy of Naru and DDUp), the
80 % model must share the full model's vocabulary, so its preparation is not
rebuilt from the smaller data but taken from the full preparation:

- the rows of the prepared full outer join whose trip is kept (a trip keeps all
  its segments; rows without a trip, i.e. a parent without trips, stay);
- the same column metadata and value domains, hence the same value codes. The
  domains therefore also hold values that occur only in the dropped 20 %, which
  the model never sees in training (as in Naru's update experiment, whose
  columns span the whole table);
- the fanout of the edge into the trips table (POL: ``F_agents_to_trips``, the
  agent's number of trips) recomputed for the kept trips; the other columns of
  a kept row do not change (indicators, a trip's own segment count);
- ``join_cardinality`` / ``full_join_cardinality`` = the kept row count, which
  must equal the split's full-outer-join count of the subset (computed on the
  evaluation snapshot), and the full preparation's must equal the split's full
  count, so prepared data and snapshot are the same data;
- the trajectory index: the full index's arrays (every kept trip has all its
  segments), with a manifest carrying the subset's schema hash.

The split directory comes from ``evaluation.trajectory_pol_porto.traj_eval.
data_shift select`` (branch ``feature/data-shift``): ``split.json``,
``kept_trip_ids.npy``, ``all_trip_ids.npy`` and, with a parent table,
``all_trip_parent_ids.npy``. It also writes ``subset.json`` (both join
cardinalities), which the evaluation bridge reads for the count-refreshed
variant.

    python -m model.scripts.subset_prepared_full_join \\
        --prepared .../pol_50m_mbr --split .../splits/pol_v3_temporal80 --output .../pol_50m_mbr_temporal80
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SUBSET_VERSION = 1
CHUNK_ROWS = 4_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(split: Path) -> Dict[str, Any]:
    split = Path(split)
    record = json.loads((split / "split.json").read_text(encoding="utf-8"))
    files = {"kept": ("kept_trip_ids.npy", "kept_trip_ids_sha256"),
             "all": ("all_trip_ids.npy", "all_trip_ids_sha256"),
             "parents": ("all_trip_parent_ids.npy", "all_trip_parent_ids_sha256")}
    arrays: Dict[str, Optional[np.ndarray]] = {}
    for name, (file, key) in files.items():
        if key not in record:
            if name == "parents" and record.get("parent") is None:
                arrays[name] = None
                continue
            raise ValueError(f"split record lacks {key}")
        if _sha256(split / file) != record[key]:
            raise ValueError(f"{split / file} does not match its split record")
        arrays[name] = np.load(split / file)
    return {"record": record, **arrays}


def subset(prepared: Path, split: Path, output: Path, *, trajectory_index: Optional[Path] = None) -> Dict[str, Any]:
    from model.src.data.schema import ColumnKind, ModelMetadata
    from model.src.data.trajectory_distinct import trajectory_index_compatibility_hash

    prepared, output = Path(prepared), Path(output)
    trajectory_index = Path(trajectory_index) if trajectory_index else prepared / "trajectory_segment_index"
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new directory")
    loaded = read_split(split)
    record = loaded["record"]
    kept_trips, all_trips, parents = loaded["kept"], loaded["all"], loaded["parents"]
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    metadata = ModelMetadata.from_json_dict(manifest["metadata"])
    rows = np.load(prepared / "sample_rows.npy", mmap_mode="r")
    trajectory_ids = np.load(prepared / "sample_trajectory_ids.npy", mmap_mode="r", allow_pickle=False)
    segment_ids = np.load(prepared / "sample_segment_ids.npy", mmap_mode="r", allow_pickle=False)
    full_rows = int(rows.shape[0])
    if not (int(manifest["join_cardinality"]) == full_rows == trajectory_ids.shape[0] == segment_ids.shape[0]):
        raise ValueError("the preparation does not hold every row of the full outer join "
                         f"(join_cardinality {manifest['join_cardinality']}, {full_rows} sample rows)")
    if full_rows != int(record["full_join_cardinality_full"]):
        raise ValueError(f"the preparation has {full_rows} join rows, the snapshot split counts "
                         f"{record['full_join_cardinality_full']}: not the same data")
    if record["entity_table"] != "trips":
        raise ValueError("only trips are split")

    # rows to keep: a kept trip, or no trip at all (a parent without trips)
    is_trip = np.isin(trajectory_ids, all_trips)
    keep = np.isin(trajectory_ids, kept_trips) | ~is_trip
    padded_rows = int(np.count_nonzero(~is_trip))
    kept_rows = int(keep.sum())
    if kept_rows != int(record["full_join_cardinality_subset"]):
        raise ValueError(f"{kept_rows} rows kept, the split's subset join has {record['full_join_cardinality_subset']}")
    if int(record.get("parents_losing_every_trip") or 0):
        raise ValueError(f"{record['parents_losing_every_trip']} parent rows lose every trip; their padded "
                         "full-outer-join row cannot be encoded from the prepared rows")

    # the fanout of the edge into trips (the parent's number of trips) for the kept trips
    fanout_updates = []
    if record.get("parent") is not None:
        parent_table = record["parent"][0]
        source = f"{parent_table}->trips"
        columns = [i for i, column in enumerate(metadata.columns)
                   if column.kind == ColumnKind.FANOUT and column.fanout_source == source]
        if len(columns) != 1:
            raise ValueError(f"expected one fanout column {source}, found {len(columns)}")
        column = columns[0]
        domain = np.asarray(metadata.columns[column].domain, dtype=np.int64)  # fanout values (counts)
        if domain.min() < 0:
            raise ValueError("negative fanout value in the domain")
        code_of = np.full(int(domain.max()) + 1, -1, dtype=np.int64)
        code_of[domain] = np.arange(domain.size)
        trip_order = np.argsort(all_trips)
        sorted_trips = all_trips[trip_order]
        trip_parent = parents[trip_order]
        kept_parent = parents[np.isin(all_trips, kept_trips)]
        unique_parents, kept_counts = np.unique(kept_parent, return_counts=True)
        fanout_updates.append((column, domain, code_of, sorted_trips, trip_parent, unique_parents, kept_counts))

    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    out_rows = np.lib.format.open_memmap(partial / "sample_rows.npy", mode="w+", dtype=rows.dtype,
                                         shape=(kept_rows, rows.shape[1]))
    out_trajectory = np.lib.format.open_memmap(partial / "sample_trajectory_ids.npy", mode="w+",
                                               dtype=trajectory_ids.dtype, shape=(kept_rows,))
    out_segments = np.lib.format.open_memmap(partial / "sample_segment_ids.npy", mode="w+",
                                             dtype=segment_ids.dtype, shape=(kept_rows,) + segment_ids.shape[1:])
    position = 0
    changed_fanout_rows = 0
    for start in range(0, full_rows, CHUNK_ROWS):
        stop = min(full_rows, start + CHUNK_ROWS)
        mask = keep[start:stop]
        count = int(mask.sum())
        block = np.array(rows[start:stop][mask])
        block_trips = np.asarray(trajectory_ids[start:stop][mask])
        for column, domain, code_of, sorted_trips, trip_parent, unique_parents, kept_counts in fanout_updates:
            has_trip = np.isin(block_trips, sorted_trips)
            where = np.searchsorted(sorted_trips, block_trips[has_trip])
            parent = trip_parent[where]
            counts = kept_counts[np.searchsorted(unique_parents, parent)]
            old = domain[block[has_trip, column]]
            if np.any(counts > old):
                raise ValueError("a parent has more kept trips than trips")
            new_codes = code_of[counts]
            if np.any(new_codes < 0):
                missing = np.unique(counts[new_codes < 0])[:5].tolist()
                raise ValueError(f"fanout values {missing} are not in the {metadata.columns[column].name} domain")
            new_codes = new_codes.astype(block.dtype)
            changed_fanout_rows += int(np.count_nonzero(new_codes != block[has_trip, column]))
            block[np.flatnonzero(has_trip), column] = new_codes
        out_rows[position:position + count] = block
        out_trajectory[position:position + count] = block_trips
        out_segments[position:position + count] = np.asarray(segment_ids[start:stop][mask])
        position += count
    for array in (out_rows, out_trajectory, out_segments):
        array.flush()
    del out_rows, out_trajectory, out_segments

    subset_metadata = dataclasses.replace(metadata, full_join_cardinality=float(kept_rows), schema_hash=None)
    subset_metadata_json = subset_metadata.to_json_dict()
    subset_record = {
        "subset_version": SUBSET_VERSION,
        "split": {key: record[key] for key in sorted(record)},
        "split_directory": str(split),
        "source_prepared_directory": str(prepared),
        "source_manifest_sha256": _sha256(prepared / "manifest.json"),
        "source_schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "full_join_cardinality": full_rows,
        "subset_join_cardinality": kept_rows,
        "padded_rows_kept": padded_rows,
        "fanout_rows_changed": changed_fanout_rows,
        "domains": "unchanged: the full preparation's (values of the dropped trips included)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    new_manifest = dict(manifest)
    new_manifest.update(join_cardinality=kept_rows, sample_rows=kept_rows, metadata=subset_metadata_json,
                        data_shift_subset=subset_record)
    (partial / "manifest.json").write_text(json.dumps(new_manifest, sort_keys=True), encoding="utf-8")
    stats_path = prepared / "preparation_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats.update(sample_row_count=kept_rows, schema_hash=subset_metadata_json["schema_hash"],
                     trajectory_index_path=str(output / "trajectory_segment_index"), data_shift_subset=subset_record)
        (partial / "preparation_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    # the trajectory index: the full index's arrays, a manifest for the subset metadata
    index_manifest = json.loads((trajectory_index / "manifest.json").read_text(encoding="utf-8"))
    index_dir = partial / "trajectory_segment_index"
    index_dir.mkdir()
    arrays = dict(index_manifest.get("arrays", {}))
    for path in sorted(trajectory_index.glob("*.npy")):
        (index_dir / path.name).symlink_to(path.resolve())
    index_manifest["metadata"] = subset_metadata_json
    index_manifest["schema_hash"] = subset_metadata_json["schema_hash"]
    index_manifest["compatibility_hash"] = trajectory_index_compatibility_hash(
        subset_metadata,
        predicate_columns=tuple(index_manifest.get("predicate_columns", ())),
        trajectory_key=str(index_manifest["trajectory_key"]),
        entity_table=str(index_manifest.get("entity_table", "trips")),
        segment_table=str(index_manifest.get("segment_table", "segments")),
        segment_key=str(index_manifest.get("segment_key", "trip_id,segment_idx")),
        trajectory_static_columns=tuple(index_manifest.get("trajectory_static_columns", ())),
        segment_varying_columns=tuple(index_manifest.get("segment_varying_columns", ())),
        srid=index_manifest.get("srid"),
        format_version=int(index_manifest["format_version"]),
    )
    index_manifest["arrays"] = arrays
    index_manifest["data_shift_subset"] = {"source_index": str(trajectory_index.resolve()),
                                           "arrays": "symbolic links to the full index (every kept trip keeps all its segments)"}
    (index_dir / "manifest.json").write_text(json.dumps(index_manifest, sort_keys=True), encoding="utf-8")

    summary = {key: subset_record[key] for key in ("full_join_cardinality", "subset_join_cardinality",
                                                   "padded_rows_kept", "fanout_rows_changed")}
    summary.update(split_mode=record["mode"], kept_trip_ids_sha256=record["kept_trip_ids_sha256"],
                   schema_hash=subset_metadata_json["schema_hash"], prepared_directory=str(output))
    (partial / "subset.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared", required=True, type=Path, help="the full preparation")
    parser.add_argument("--split", required=True, type=Path, help="data_shift select output directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trajectory-index", type=Path, help="default: <prepared>/trajectory_segment_index")
    args = parser.parse_args(argv)
    print(json.dumps(subset(args.prepared, args.split, args.output, trajectory_index=args.trajectory_index),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
