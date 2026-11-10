"""Prepare the own model's training data from a trajectory evaluation snapshot.

The POL model was prepared from MobilityDB staging TSVs
(``prepare_pol_staging_data.py``). Porto has no staging TSVs; its canonical
data is the v3 evaluation snapshot (``traj_snapshots/porto_v3``: the arrays the
benchmark labels were recomputed from by the snapshot gate). This script
builds the same prepared layout from a snapshot, with the same conventions as
``prepare_pol_staging_data.py``:

- ``manifest.json`` (complete domains: every distinct value plus the
  ``__OUTER_MISSING__`` padding token, ``complete_ordinary_domain`` order),
  ``sample_rows.npy`` (every segment row of the full join, segment order),
  ``sample_trajectory_ids.npy``, ``sample_segment_ids.npy``,
  ``preparation_stats.json``;
- the compact trajectory segment index (``trajectory_segment_index/``) with
  the manifest fields of ``prepare_pol_trajectory_distinct.py``;
- columns: each table's data columns (spec below), then ``I_<table>``
  indicators, then ``F_<parent>_to_<child>`` fanouts; segment MBR columns
  when ``trajectory_spatial.representation`` is ``segment_mbr``; the constant
  ``trips:trip_geom`` placeholder of the POL schema; timestamps as epoch
  seconds of the naive stored timestamp (``_timestamp_to_float``).

SQL NULL (Porto ``origin_stand``: 52 % of trips; call types A and C have no
stand) is kept as the model's own ``__SQL_NULL__`` domain value (decided
2026-11-09: NULL modelled as a value, not dropped). It sorts before every
value, so range and one-sided conditions never include it, as in SQL.

``compare`` checks the converter against the existing POL preparation: the
same columns, domains and encoded rows, row for row.

    python -m model.scripts.prepare_trajectory_snapshot prepare --dataset porto \\
        --config model/configs/porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml \\
        --snapshot /work_beegfs/.../traj_snapshots/porto_v3
    python -m model.scripts.prepare_trajectory_snapshot compare \\
        --prepared /tmp/pol_from_snapshot --reference data/pol_prepared/pol_50m_mbr
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.complete_domain_preparation import (
    COMPLETE_METADATA_SOURCE,
    SQL_NULL,
    EncodedSampleRows,
    preparation_stats,
)
from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.trajectory_distinct import (
    SPATIAL_INTERSECTS_SEMANTICS_VERSION,
    TEMPORAL_OVERLAP_SEMANTICS_VERSION,
    TRAJECTORY_INDEX_FORMAT_VERSION,
    TRAJECTORY_TARGET_SEMANTICS_VERSION,
    CompactTrajectorySegmentIndex,
    trajectory_index_compatibility_hash,
)
from model.src.model.factorization import FactorizationConfig

SEGMENT_COLUMNS = (  # model name, snapshot column, kind
    ("segment_idx", "segment_idx", "int"),
    ("t_s", "ts", "time"),
    ("t_e", "te", "time"),
    ("s_x", "sx", "float"),
    ("s_y", "sy", "float"),
    ("e_x", "ex", "float"),
    ("e_y", "ey", "float"),
)
MBR_COLUMNS = ("seg_min_x", "seg_max_x", "seg_min_y", "seg_max_y")
TRIP_TIME_COLUMNS = (
    ("start_time", "trip_time.start", "time"),
    ("end_time", "trip_time.end", "time"),
    ("num_of_segments", "num_of_segments", "int"),
    ("trip_geom", None, "placeholder"),
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tables: Tuple[str, ...]  # join order, root first
    columns: Dict[str, Tuple[Tuple[str, Optional[str], str], ...]]  # non-segment tables
    parents: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # child -> (parent, foreign key)


SPECS = {
    # the column order and kinds of prepare_pol_staging_data.py
    "pol": DatasetSpec(
        name="pol",
        tables=("agents", "trips", "segments"),
        columns={
            "agents": (("age", "age", "float"), ("educationLevel", "educationLevel", "str"),
                       ("interest", "interest", "str"), ("joviality", "joviality", "float"),
                       ("family_size", "family_size", "int")),
            "trips": TRIP_TIME_COLUMNS,
        },
        parents={"trips": ("agents", "agent_id")},
    ),
    # Porto: no agents; the trip attributes of porto_v3_config.json, then the POL trip columns
    "porto": DatasetSpec(
        name="porto",
        tables=("trips", "segments"),
        columns={"trips": (("call_type", "call_type", "str"), ("origin_stand", "origin_stand", "int_nullable"),
                           *TRIP_TIME_COLUMNS)},
    ),
}


def _seconds(microseconds: np.ndarray) -> np.ndarray:
    """Epoch seconds of naive microsecond timestamps; equals ``_timestamp_to_float`` of the text."""
    values = np.asarray(microseconds, dtype=np.int64)
    if values.size and np.abs(values).max() >= 2 ** 53:
        raise ValueError("timestamps beyond exact float64 microseconds")
    return values.astype(np.float64) / 1e6


def column_values(table: Dict[str, np.ndarray], source: Optional[str], kind: str, rows: int) -> np.ndarray:
    if kind == "placeholder":
        return np.zeros(rows, dtype=np.int64)
    values = np.asarray(table[source])
    if kind == "time":
        return _seconds(values)
    if kind == "int":
        if values.dtype.kind == "f":
            if np.isnan(values).any() or not np.all(values == np.round(values)):
                raise ValueError(f"{source}: non-integer values in an integer column")
            return values.astype(np.int64)
        return values.astype(np.int64)
    if kind == "int_nullable":
        return values.astype(np.float64)  # NaN = SQL NULL
    if kind == "float":
        values = values.astype(np.float64)
        if np.isnan(values).any():
            raise ValueError(f"{source}: NULL in a non-nullable float column")
        return values
    if kind == "str":
        return values.astype(str)
    raise ValueError(f"unknown column kind {kind!r}")


def domain_and_codes(values: np.ndarray, kind: str) -> Tuple[Tuple[Any, ...], np.ndarray]:
    """The complete domain (as ``complete_ordinary_domain``) and the encoded ids of ``values``."""

    if kind == "placeholder":
        return (OUTER_MISSING,), np.zeros(values.size, dtype=np.int64)
    null = np.zeros(values.size, dtype=bool)
    present = values
    if kind == "int_nullable":
        null = np.isnan(values)
        present = values[~null]
        if not np.all(present == np.round(present)):
            raise ValueError("non-integer values in a nullable integer column")
        present = present.astype(np.int64)
    if present.dtype.kind == "f" and np.any((present == 0) & np.signbit(present)):
        # complete_ordinary_domain keeps -0.0 and 0.0 apart (distinct JSON keys); np.unique would merge them
        raise ValueError("negative zero in a float column; not supported by this converter")
    unique = np.unique(present)
    offset = 1 if null.any() else 0
    codes = np.zeros(values.size, dtype=np.int64)
    codes[~null] = np.searchsorted(unique, present) + offset
    domain = ((SQL_NULL,) if null.any() else ()) + tuple(unique.tolist()) + (OUTER_MISSING,)
    return domain, codes


def _dense_positive_domain(counts: np.ndarray) -> Tuple[int, ...]:
    maximum = int(counts.max()) if counts.size else 1
    return tuple(range(1, max(maximum, 1) + 1))


def _load_snapshot(root: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = {table: {name: np.load(root / "tables" / table / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                    for name in spec["columns"]}
            for table, spec in manifest["tables"].items()}
    return manifest, data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_index(keys: np.ndarray, lookup: np.ndarray, what: str) -> np.ndarray:
    """Positions of ``lookup`` in the strictly increasing ``keys``."""
    keys = np.asarray(keys)
    if keys.size > 1 and not np.all(keys[1:] > keys[:-1]):
        raise ValueError(f"{what}: keys are not strictly increasing")
    position = np.searchsorted(keys, lookup)
    if np.any(position >= keys.size) or np.any(keys[np.minimum(position, keys.size - 1)] != lookup):
        raise ValueError(f"{what}: references a missing key")
    return position


def build(spec: DatasetSpec, data: Dict[str, Dict[str, np.ndarray]], *, use_segment_mbr: bool
          ) -> Tuple[ModelMetadata, Dict[str, Any]]:
    """Metadata and the per-segment encoded columns (arrays aligned with the segment rows)."""

    segments = data["segments"]
    n = int(np.asarray(segments["trip_id"]).size)
    seg_trip = np.asarray(segments["trip_id"], dtype=np.int64)
    seg_idx = np.asarray(segments["segment_idx"], dtype=np.int64)
    if n > 1:
        same = seg_trip[1:] == seg_trip[:-1]
        if np.any(seg_trip[1:] < seg_trip[:-1]) or np.any(same & (seg_idx[1:] <= seg_idx[:-1])):
            raise ValueError("segments are not strictly ordered by (trip_id, segment_idx)")
    trips = data["trips"]
    trip_ids = np.asarray(trips["trip_id"], dtype=np.int64)
    trip_row = _row_index(trip_ids, seg_trip, "segments.trip_id -> trips")
    segments_per_trip = np.bincount(trip_row, minlength=trip_ids.size)
    if np.any(segments_per_trip == 0):
        raise ValueError("trips without segments: the full join would hold rows the segment order cannot express")
    rows_of = {"trips": trip_row, "segments": np.arange(n)}
    fanouts = []  # (name, table, source, counts per segment row)
    for child, (parent, key) in spec.parents.items():
        parent_ids = np.asarray(data[parent][f"{parent[:-1]}_id"], dtype=np.int64)
        child_parent = _row_index(parent_ids, np.asarray(data[child][key], dtype=np.int64), f"{child}.{key}")
        rows_of[parent] = child_parent[rows_of[child]]
        children = np.bincount(child_parent, minlength=parent_ids.size)
        fanouts.append((f"F_{parent}_to_{child}", parent, f"{parent}->{child}", children, rows_of[parent]))
    fanouts.append(("F_trips_to_segments", "trips", "trips->segments", segments_per_trip, trip_row))

    columns: List[ColumnMetadata] = []
    encoded: List[np.ndarray] = []
    for table in spec.tables:
        if table == "segments":
            specs = list(SEGMENT_COLUMNS)
            values_of = {name: column_values(segments, source, kind, n) for name, source, kind in specs}
            if use_segment_mbr:
                values_of["seg_min_x"] = np.minimum(values_of["s_x"], values_of["e_x"])
                values_of["seg_max_x"] = np.maximum(values_of["s_x"], values_of["e_x"])
                values_of["seg_min_y"] = np.minimum(values_of["s_y"], values_of["e_y"])
                values_of["seg_max_y"] = np.maximum(values_of["s_y"], values_of["e_y"])
                specs += [(name, None, "float") for name in MBR_COLUMNS]
            for name, _, kind in specs:
                domain, codes = domain_and_codes(values_of[name], kind)
                columns.append(ColumnMetadata(f"segments:{name}", ColumnKind.DATA, domain, table="segments"))
                encoded.append(codes)
            continue
        rows = int(np.asarray(data[table][f"{table[:-1]}_id"]).size)
        for name, source, kind in spec.columns[table]:
            values = column_values(data[table], source, kind, rows)
            domain, codes = domain_and_codes(values, kind)
            columns.append(ColumnMetadata(f"{table}:{name}", ColumnKind.DATA, domain, table=table))
            encoded.append(codes[rows_of[table]])
    for table in spec.tables:
        columns.append(ColumnMetadata(f"I_{table}", ColumnKind.INDICATOR, (0, 1), table=table))
        encoded.append(np.ones(n, dtype=np.int64))
    for name, table, source, counts, rows in fanouts:
        columns.append(ColumnMetadata(name, ColumnKind.FANOUT, _dense_positive_domain(counts), table=table,
                                      fanout_source=source))
        encoded.append(counts[rows].astype(np.int64) - 1)
    metadata = ModelMetadata(
        columns=tuple(columns), full_join_cardinality=float(n), column_order="data_indicators_fanouts",
        upstream_attribution={spec.name.upper(): "trajectory evaluation snapshot (traj_snapshot_v1)",
                              "NeuroCard": "full-outer-join indicators and fanout correction semantics"},
        join_root=spec.tables[0], join_tables=spec.tables,
        join_edges=tuple((parent, child) for child, (parent, _) in spec.parents.items()) + (("trips", "segments"),),
    )
    return metadata, {"encoded": encoded, "segment_trip": seg_trip, "segment_idx": seg_idx,
                      "segments_per_trip": segments_per_trip, "trip_ids": trip_ids, "segment_values": {
                          name: column_values(segments, source, kind, n) for name, source, kind in SEGMENT_COLUMNS}}


def write_sample_rows(path: Path, encoded: Sequence[np.ndarray], chunk_rows: int = 2_000_000) -> np.ndarray:
    n = encoded[0].size
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(n, len(encoded)))
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        for column, codes in enumerate(encoded):
            out[start:stop, column] = codes[start:stop]
    out.flush()
    return out


def write_index(directory: Path, *, metadata: ModelMetadata, parts: Dict[str, Any], trajectory_config: Dict[str, Any],
                source: Dict[str, Any], config_path: Optional[Path], force: bool) -> Dict[str, Any]:
    """The compact trajectory segment index, as ``prepare_pol_trajectory_distinct`` writes it."""

    started = time.perf_counter()
    keys = dict(
        trajectory_key=str(trajectory_config.get("trajectory_key", "trip_id")),
        entity_table=str(trajectory_config.get("entity_table", "trips")),
        segment_table=str(trajectory_config.get("segment_table", "segments")),
        segment_key=str(trajectory_config.get("segment_key", "trip_id,segment_idx")),
        trajectory_static_columns=tuple(str(v) for v in trajectory_config.get("trajectory_static_columns", ())),
        segment_varying_columns=tuple(str(v) for v in trajectory_config.get("segment_varying_columns", ())),
        srid=None if trajectory_config.get("srid") is None else int(trajectory_config["srid"]),
    )
    compatibility_hash = trajectory_index_compatibility_hash(
        metadata, predicate_columns=(), format_version=TRAJECTORY_INDEX_FORMAT_VERSION, **keys)
    if directory.exists() and not force:
        raise SystemExit(f"{directory} exists; pass --force-rebuild-index to replace it")
    tmp = directory.with_name(f".{directory.name}.tmp.{int(time.time())}.{os.getpid()}")
    tmp.mkdir(parents=True)
    counts = parts["segments_per_trip"]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    values = parts["segment_values"]
    arrays = {"trajectory_ids": parts["trip_ids"].astype(np.int64), "offsets": offsets,
              "segment_idx": parts["segment_idx"].astype(np.int32)}
    for name in ("t_s", "t_e", "s_x", "s_y", "e_x", "e_y"):
        arrays[name] = values[name].astype(np.float64)
    arrays["seg_min_x"] = np.minimum(values["s_x"], values["e_x"])
    arrays["seg_max_x"] = np.maximum(values["s_x"], values["e_x"])
    arrays["seg_min_y"] = np.minimum(values["s_y"], values["e_y"])
    arrays["seg_max_y"] = np.maximum(values["s_y"], values["e_y"])
    if not np.array_equal(arrays["segment_idx"], parts["segment_idx"]):
        raise ValueError("segment_idx does not fit int32")
    for name, array in arrays.items():
        np.save(tmp / f"{name}.npy", array)
    n_traj, n_seg = int(counts.size), int(offsets[-1])
    index_bytes = int(n_traj * 8 + (n_traj + 1) * 8 + n_seg * 4 + n_seg * 10 * 8)
    manifest = {
        "format_version": TRAJECTORY_INDEX_FORMAT_VERSION,
        "index_type": "compact_pol_segment_mmap",
        "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
        "temporal_semantics_version": TEMPORAL_OVERLAP_SEMANTICS_VERSION,
        "spatial_semantics_version": SPATIAL_INTERSECTS_SEMANTICS_VERSION,
        "metadata": metadata.to_json_dict(),
        "compatibility_hash": compatibility_hash,
        **{key: (list(value) if isinstance(value, tuple) else value) for key, value in keys.items()},
        "predicate_columns": [],
        "trajectory_count": n_traj,
        "segment_count": n_seg,
        "min_segments_per_trajectory": int(counts.min()),
        "mean_segments_per_trajectory": float(counts.mean()),
        "p95_segments_per_trajectory": int(np.percentile(counts, 95, method="higher")),
        "max_segments_per_trajectory": int(counts.max()),
        "index_bytes": index_bytes,
        "bytes_per_segment": float(index_bytes / max(1, n_seg)),
        "preparation_seconds": float(time.perf_counter() - started),
        "ordered_input_required": True,
        "source_segments_path": source["snapshot"],
        "source_segments_size_bytes": None,
        "source_fingerprint": source,
        "config_path": None if config_path is None else str(config_path),
        "schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "mbr_arrays_persisted": True,
        "arrays": {name: f"{name}.npy" for name in arrays},
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(directory)
    CompactTrajectorySegmentIndex.from_directory(directory)  # the loader's own validation
    return manifest


PREPARED_FILES = ("manifest.json", "sample_rows.npy", "sample_trajectory_ids.npy", "sample_segment_ids.npy",
                  "preparation_stats.json")


def prepare(dataset: str, config_path: Path, snapshot: Path, prepared: Optional[Path] = None, *,
            index_path: Optional[Path] = None, force: bool = False, log=print) -> Dict[str, Any]:
    """Write the prepared directory and the trajectory index; existing outputs are refused unless ``force``."""
    started = time.perf_counter()
    config = load_simple_yaml(str(config_path))
    validate_config(config)
    spatial = config.get("trajectory_spatial", {}) or {}
    use_mbr = bool(spatial.get("enabled", False))
    if use_mbr and spatial.get("representation") != "segment_mbr":
        raise ValueError("trajectory_spatial.representation must be segment_mbr when enabled")
    if prepared is not None and index_path is None:  # never write next to the configured data by accident
        index_path = Path(prepared) / "trajectory_segment_index"
    prepared = Path(prepared or config["dataset"]["prepared_directory"])
    index_dir = Path(index_path or config["dataset"].get("trajectory_index_path")
                     or prepared / "trajectory_segment_index")
    existing = [str(prepared / name) for name in PREPARED_FILES if (prepared / name).exists()]
    if index_dir.exists():
        existing.append(str(index_dir))
    if existing and not force:  # checked before anything is written (review of a24d561)
        raise SystemExit(f"refusing to overwrite existing prepared data: {existing}; pass --force to replace it")
    snapshot_manifest, data = _load_snapshot(snapshot)
    source = {"snapshot": str(snapshot.resolve()), "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json"),
              "snapshot_version": snapshot_manifest.get("snapshot_version"),
              "snapshot_config_sha256": snapshot_manifest.get("config_sha256")}
    metadata, parts = build(SPECS[dataset], data, use_segment_mbr=use_mbr)
    log(f"metadata: {len(metadata.columns)} columns, {int(metadata.full_join_cardinality)} rows")
    prepared.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 2,
        "dataset_name": str(config["dataset"].get("name", f"{dataset}_trajectory_full_join")),
        "dataset_type": "pol_trajectory_full_join",
        "join_cardinality": int(metadata.full_join_cardinality),
        "metadata": metadata.to_json_dict(),
        "source": "trajectory_snapshot",
        "metadata_source": COMPLETE_METADATA_SOURCE,
        "sample_source": "trajectory_snapshot",
        "sample_rows": int(metadata.full_join_cardinality),
        "domains_complete": True,
        "snapshot": source,
        "trajectory_spatial": dict(spatial),
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    rows = write_sample_rows(prepared / "sample_rows.npy", parts["encoded"])
    np.save(prepared / "sample_trajectory_ids.npy", parts["segment_trip"])
    np.save(prepared / "sample_segment_ids.npy", np.stack([parts["segment_trip"], parts["segment_idx"]], axis=1))
    index_manifest = write_index(index_dir, metadata=metadata, parts=parts,
                                 trajectory_config=config.get("trajectory_distinct", {}) or {}, source=source,
                                 config_path=config_path, force=force)
    stats = preparation_stats(metadata=metadata, encoded_sample=EncodedSampleRows(encoded_rows=np.asarray(rows)),
                              factorization_config=FactorizationConfig.from_dict(config.get("factorization", {})))
    stats.update({"snapshot": source, "trajectory_index_path": str(index_dir),
                  "trajectory_index_segment_count": index_manifest["segment_count"],
                  "preparation_seconds": float(time.perf_counter() - started),
                  "materialized_all_segment_rows": True})
    (prepared / "preparation_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True, default=str),
                                                     encoding="utf-8")
    summary = {"prepared_directory": str(prepared), "schema_hash": metadata.stable_schema_hash(),
               "join_cardinality": int(metadata.full_join_cardinality), "columns": len(metadata.columns),
               "trajectory_index_path": str(index_dir), "seconds": time.perf_counter() - started}
    log(json.dumps(summary, sort_keys=True))
    return summary


def compare(prepared: Path, reference: Path, chunk_rows: int = 2_000_000, *, index: Optional[Path] = None,
            reference_index: Optional[Path] = None) -> Dict[str, Any]:
    """Columns, domains and encoded rows of two preparations (e.g. snapshot vs staging TSVs), and their indexes."""

    def load(directory: Path) -> Dict[str, Any]:
        return json.loads((directory / "manifest.json").read_text())

    manifest_a, manifest_b = load(prepared), load(reference)
    mine, theirs = (ModelMetadata.from_json_dict(m["metadata"]) for m in (manifest_a, manifest_b))
    cardinalities = [[float(mine.full_join_cardinality), float(manifest_a.get("join_cardinality"))],
                     [float(theirs.full_join_cardinality), float(manifest_b.get("join_cardinality"))]]
    report: Dict[str, Any] = {"columns_equal": [c.name for c in mine.columns] == [c.name for c in theirs.columns],
                              "join_cardinality": cardinalities,
                              "join_cardinality_equal": len({value for pair in cardinalities for value in pair}) == 1,
                              "columns": {}}
    for a, b in zip(mine.columns, theirs.columns):
        same_values = len(a.domain) == len(b.domain) and all(
            (x == y) if isinstance(x, str) or isinstance(y, str) else float(x) == float(y)
            for x, y in zip(a.domain, b.domain))
        report["columns"][a.name] = {"kind_equal": a.kind == b.kind, "table_equal": a.table == b.table,
                                     "domain_size": [len(a.domain), len(b.domain)], "domain_equal": same_values}
    rows_a = np.load(prepared / "sample_rows.npy", mmap_mode="r")
    rows_b = np.load(reference / "sample_rows.npy", mmap_mode="r")
    report["sample_rows_shape"] = [list(rows_a.shape), list(rows_b.shape)]
    differing_rows = 0
    if rows_a.shape == rows_b.shape:
        for start in range(0, rows_a.shape[0], chunk_rows):
            stop = min(rows_a.shape[0], start + chunk_rows)
            differing_rows += int(np.any(rows_a[start:stop] != rows_b[start:stop], axis=1).sum())
    report["differing_sample_rows"] = differing_rows if rows_a.shape == rows_b.shape else None
    for name in ("sample_trajectory_ids.npy", "sample_segment_ids.npy"):
        report[f"{name}_equal"] = bool(np.array_equal(np.load(prepared / name, mmap_mode="r"),
                                                      np.load(reference / name, mmap_mode="r")))
    index_ok = True
    if index is not None and reference_index is not None:
        arrays = {}
        for name in json.loads((reference_index / "manifest.json").read_text())["arrays"]:
            a = np.load(index / f"{name}.npy", mmap_mode="r")
            b = np.load(reference_index / f"{name}.npy", mmap_mode="r")
            arrays[name] = bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
        report["index_arrays_equal"] = arrays
        index_ok = all(arrays.values())
    report["passed"] = bool(index_ok and report["columns_equal"] and report["join_cardinality_equal"]
                            and report["differing_sample_rows"] == 0
                            and all(all(v for k, v in c.items() if k != "domain_size")
                                    for c in report["columns"].values())
                            and report["sample_trajectory_ids.npy_equal"] and report["sample_segment_ids.npy_equal"])
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("prepare")
    run.add_argument("--dataset", required=True, choices=sorted(SPECS))
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--snapshot", required=True, type=Path)
    run.add_argument("--prepared-directory", type=Path, help="default: dataset.prepared_directory")
    run.add_argument("--trajectory-index-path", type=Path,
                     help="default: <prepared-directory>/trajectory_segment_index if --prepared-directory is "
                          "given, else dataset.trajectory_index_path")
    run.add_argument("--force", action="store_true", help="replace existing prepared files and trajectory index")
    check = sub.add_parser("compare")
    check.add_argument("--prepared", required=True, type=Path)
    check.add_argument("--reference", required=True, type=Path)
    check.add_argument("--index", type=Path, help="trajectory index of --prepared")
    check.add_argument("--reference-index", type=Path, help="trajectory index of --reference")
    check.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.dataset, args.config, args.snapshot, args.prepared_directory,
                index_path=args.trajectory_index_path, force=args.force)
        return 0
    report = compare(args.prepared, args.reference, index=args.index, reference_index=args.reference_index)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
