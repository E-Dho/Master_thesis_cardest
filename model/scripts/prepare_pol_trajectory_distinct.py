from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.schema import ModelMetadata
from model.src.data.trajectory_distinct import (
    CompactTrajectorySegmentIndex,
    SPATIAL_INTERSECTS_SEMANTICS_VERSION,
    TEMPORAL_OVERLAP_SEMANTICS_VERSION,
    TRAJECTORY_INDEX_FORMAT_VERSION,
    TRAJECTORY_TARGET_SEMANTICS_VERSION,
    _timestamp_to_float,
    trajectory_index_compatibility_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare POL trajectory-distinct provenance and compact segment index."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--segments-tsv",
        required=True,
        help="POL staging segments.tsv with trip_id, segment_idx, s_x, s_y, e_x, e_y, t_s, t_e.",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="compact index directory; defaults to dataset.trajectory_index_path.",
    )
    parser.add_argument(
        "--sample-provenance-tsv",
        default=None,
        help=(
            "optional TSV aligned to sample_rows.npy with either header columns "
            "trip_id/segment_idx or headerless first two columns."
        ),
    )
    parser.add_argument(
        "--sample-rows-path",
        default=None,
        help="optional sample_rows.npy path for provenance length validation.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Replace an existing compact index. Without this, compatible indexes are reused and stale indexes fail.",
    )
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    dataset = config["dataset"]
    trajectory_config = config.get("trajectory_distinct", {})
    prepared_directory = Path(dataset["prepared_directory"])
    manifest_path = prepared_directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing prepared POL manifest: {manifest_path}")
    metadata = ModelMetadata.from_json_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]
    )

    output_directory = Path(
        args.output_directory
        or dataset.get("trajectory_index_path")
        or (prepared_directory / "trajectory_segment_index")
    )
    if output_directory.suffix == ".npz":
        output_directory = output_directory.with_suffix("")
    compact_manifest = _write_ordered_segments_index(
        Path(args.segments_tsv),
        output_directory=output_directory,
        metadata=metadata,
        trajectory_key=str(trajectory_config.get("trajectory_key", "trip_id")),
        entity_table=str(trajectory_config.get("entity_table", "trips")),
        segment_table=str(trajectory_config.get("segment_table", "segments")),
        segment_key=str(trajectory_config.get("segment_key", "trip_id,segment_idx")),
        trajectory_static_columns=tuple(
            str(value) for value in trajectory_config.get("trajectory_static_columns", ())
        ),
        segment_varying_columns=tuple(
            str(value) for value in trajectory_config.get("segment_varying_columns", ())
        ),
        srid=(
            None
            if trajectory_config.get("srid") is None
            else int(trajectory_config.get("srid"))
        ),
        source_segments_path=Path(args.segments_tsv),
        config_path=Path(args.config),
        force_rebuild=bool(args.force_rebuild),
    )
    index = CompactTrajectorySegmentIndex.from_directory(output_directory)

    if args.sample_provenance_tsv is not None:
        trajectory_ids, segment_ids = _read_sample_provenance_tsv(
            Path(args.sample_provenance_tsv)
        )
        sample_rows_path = Path(
            args.sample_rows_path or (prepared_directory / "sample_rows.npy")
        )
        if sample_rows_path.exists():
            sample_rows = np.load(sample_rows_path, mmap_mode="r")
            if len(trajectory_ids) != len(sample_rows):
                raise SystemExit(
                    "sample provenance row count does not match sample_rows.npy: "
                    f"{len(trajectory_ids)} != {len(sample_rows)}"
        )
        np.save(prepared_directory / "sample_trajectory_ids.npy", np.asarray(trajectory_ids))
        np.save(prepared_directory / "sample_segment_ids.npy", np.asarray(segment_ids, dtype=np.int64))

    storage = index.storage_summary()
    print(f"trajectory_index_path={output_directory}")
    print(f"trajectory_index_format_version={compact_manifest['format_version']}")
    print(f"reused_existing_index={bool(compact_manifest.get('reused_existing_index', False))}")
    print(f"trajectory_count={storage['trajectory_count']}")
    print(f"segment_count={storage['segment_count']}")
    print(f"index_bytes={storage['index_bytes']}")
    print(f"bytes_per_segment={storage['bytes_per_segment']:.3f}")
    print(f"estimated_bytes_50m_segments={storage['estimated_bytes_50m_segments']}")
    print(f"compatibility_hash={compact_manifest['compatibility_hash']}")
    print(f"min_segments_per_trajectory={compact_manifest['min_segments_per_trajectory']}")
    print(f"mean_segments_per_trajectory={compact_manifest['mean_segments_per_trajectory']:.3f}")
    print(f"p95_segments_per_trajectory={compact_manifest['p95_segments_per_trajectory']}")
    print(f"max_segments_per_trajectory={compact_manifest['max_segments_per_trajectory']}")


def _write_ordered_segments_index(
    path: Path,
    *,
    output_directory: Path,
    metadata: ModelMetadata,
    trajectory_key: str,
    entity_table: str,
    segment_table: str,
    segment_key: str,
    trajectory_static_columns: tuple[str, ...],
    segment_varying_columns: tuple[str, ...],
    srid: int | None,
    source_segments_path: Path | None = None,
    config_path: Path | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing POL segments TSV: {path}")
    started = time.perf_counter()
    compatibility_hash = trajectory_index_compatibility_hash(
        metadata,
        predicate_columns=(),
        trajectory_key=trajectory_key,
        entity_table=entity_table,
        segment_table=segment_table,
        segment_key=segment_key,
        trajectory_static_columns=trajectory_static_columns,
        segment_varying_columns=segment_varying_columns,
        srid=srid,
        format_version=TRAJECTORY_INDEX_FORMAT_VERSION,
    )
    if output_directory.exists():
        if not force_rebuild:
            try:
                existing = CompactTrajectorySegmentIndex.from_directory(output_directory)
                existing.validate_runtime_compatibility(
                    existing.runtime_config,
                    metadata,
                )
                if existing.compatibility_hash == compatibility_hash:
                    manifest = json.loads(
                        (output_directory / "manifest.json").read_text(encoding="utf-8")
                    )
                    manifest["reused_existing_index"] = True
                    return manifest
                raise SystemExit(
                    "existing trajectory index compatibility hash does not match "
                    "the requested metadata/config. Rerun with --force-rebuild to "
                    "replace it after confirming no training job is using the directory."
                )
            except Exception as exc:
                raise SystemExit(
                    "existing trajectory index is missing, stale, or incompatible: "
                    f"{output_directory}. Rerun this preparation command with "
                    "--force-rebuild to replace it after confirming no training job is "
                    f"using the directory. Original error: {exc}"
                ) from exc
    counts = _scan_ordered_segments(path)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    tmp_directory = output_directory.with_name(
        f".{output_directory.name}.tmp.{int(time.time())}.{os.getpid()}"
    )
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory)
    tmp_directory.mkdir(parents=True, exist_ok=False)
    arrays = {
        "trajectory_ids": "trajectory_ids.npy",
        "offsets": "offsets.npy",
        "segment_idx": "segment_idx.npy",
        "t_s": "t_s.npy",
        "t_e": "t_e.npy",
        "s_x": "s_x.npy",
        "s_y": "s_y.npy",
        "e_x": "e_x.npy",
        "e_y": "e_y.npy",
        "seg_min_x": "seg_min_x.npy",
        "seg_max_x": "seg_max_x.npy",
        "seg_min_y": "seg_min_y.npy",
        "seg_max_y": "seg_max_y.npy",
    }
    trajectory_ids = np.lib.format.open_memmap(
        tmp_directory / arrays["trajectory_ids"],
        mode="w+",
        dtype=np.int64,
        shape=(counts["trajectory_count"],),
    )
    offsets = np.lib.format.open_memmap(
        tmp_directory / arrays["offsets"],
        mode="w+",
        dtype=np.int64,
        shape=(counts["trajectory_count"] + 1,),
    )
    segment_idx = np.lib.format.open_memmap(
        tmp_directory / arrays["segment_idx"],
        mode="w+",
        dtype=np.int32,
        shape=(counts["segment_count"],),
    )
    numeric_arrays = {
        "t_s": np.lib.format.open_memmap(tmp_directory / arrays["t_s"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "t_e": np.lib.format.open_memmap(tmp_directory / arrays["t_e"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "s_x": np.lib.format.open_memmap(tmp_directory / arrays["s_x"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "s_y": np.lib.format.open_memmap(tmp_directory / arrays["s_y"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "e_x": np.lib.format.open_memmap(tmp_directory / arrays["e_x"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "e_y": np.lib.format.open_memmap(tmp_directory / arrays["e_y"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "seg_min_x": np.lib.format.open_memmap(tmp_directory / arrays["seg_min_x"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "seg_max_x": np.lib.format.open_memmap(tmp_directory / arrays["seg_max_x"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "seg_min_y": np.lib.format.open_memmap(tmp_directory / arrays["seg_min_y"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
        "seg_max_y": np.lib.format.open_memmap(tmp_directory / arrays["seg_max_y"], mode="w+", dtype=np.float64, shape=(counts["segment_count"],)),
    }
    row_index = 0
    trajectory_index = -1
    previous_trip_id: int | None = None
    offsets[0] = 0
    for row in _iter_segment_rows(path):
        trip_id = row["trip_id"]
        if trip_id != previous_trip_id:
            trajectory_index += 1
            trajectory_ids[trajectory_index] = trip_id
            offsets[trajectory_index] = row_index
            previous_trip_id = trip_id
        segment_idx[row_index] = row["segment_idx"]
        numeric_arrays["t_s"][row_index] = row["t_s"]
        numeric_arrays["t_e"][row_index] = row["t_e"]
        numeric_arrays["s_x"][row_index] = row["s_x"]
        numeric_arrays["s_y"][row_index] = row["s_y"]
        numeric_arrays["e_x"][row_index] = row["e_x"]
        numeric_arrays["e_y"][row_index] = row["e_y"]
        numeric_arrays["seg_min_x"][row_index] = min(row["s_x"], row["e_x"])
        numeric_arrays["seg_max_x"][row_index] = max(row["s_x"], row["e_x"])
        numeric_arrays["seg_min_y"][row_index] = min(row["s_y"], row["e_y"])
        numeric_arrays["seg_max_y"][row_index] = max(row["s_y"], row["e_y"])
        row_index += 1
    offsets[counts["trajectory_count"]] = counts["segment_count"]
    for array in (trajectory_ids, offsets, segment_idx, *numeric_arrays.values()):
        array.flush()
    index_bytes = int(
        counts["trajectory_count"] * np.dtype(np.int64).itemsize
        + (counts["trajectory_count"] + 1) * np.dtype(np.int64).itemsize
        + counts["segment_count"] * np.dtype(np.int32).itemsize
        + counts["segment_count"] * 10 * np.dtype(np.float64).itemsize
    )
    manifest = {
        "format_version": TRAJECTORY_INDEX_FORMAT_VERSION,
        "index_type": "compact_pol_segment_mmap",
        "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
        "temporal_semantics_version": TEMPORAL_OVERLAP_SEMANTICS_VERSION,
        "spatial_semantics_version": SPATIAL_INTERSECTS_SEMANTICS_VERSION,
        "metadata": metadata.to_json_dict(),
        "compatibility_hash": compatibility_hash,
        "trajectory_key": trajectory_key,
        "entity_table": entity_table,
        "segment_table": segment_table,
        "segment_key": segment_key,
        "predicate_columns": [],
        "trajectory_static_columns": list(trajectory_static_columns),
        "segment_varying_columns": list(segment_varying_columns),
        "srid": srid,
        "trajectory_count": int(counts["trajectory_count"]),
        "segment_count": int(counts["segment_count"]),
        "min_segments_per_trajectory": int(counts["min_segments_per_trajectory"]),
        "mean_segments_per_trajectory": float(counts["mean_segments_per_trajectory"]),
        "p95_segments_per_trajectory": int(counts["p95_segments_per_trajectory"]),
        "max_segments_per_trajectory": int(counts["max_segments_per_trajectory"]),
        "index_bytes": index_bytes,
        "bytes_per_segment": float(index_bytes / max(1, int(counts["segment_count"]))),
        "preparation_seconds": float(time.perf_counter() - started),
        "ordered_input_required": True,
        "source_segments_path": str(source_segments_path or path),
        "source_segments_size_bytes": int(path.stat().st_size),
        "config_path": None if config_path is None else str(config_path),
        "schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "mbr_arrays_persisted": True,
        "arrays": arrays,
    }
    (tmp_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if output_directory.exists():
        shutil.rmtree(output_directory)
    tmp_directory.rename(output_directory)
    return manifest


def _scan_ordered_segments(path: Path) -> dict[str, float | int]:
    segment_count = 0
    trajectory_count = 0
    previous_key: tuple[int, int] | None = None
    previous_trip_id: int | None = None
    current_group_size = 0
    group_reservoir: list[int] = []
    reservoir_limit = 100_000
    group_sum = 0
    group_min: int | None = None
    group_max = 0
    for row in _iter_segment_rows(path):
        key = (row["trip_id"], row["segment_idx"])
        if previous_key is not None and key <= previous_key:
            raise SystemExit(
                "segments TSV must be strictly ordered by (trip_id, segment_idx); "
                f"observed {key} after {previous_key}"
            )
        previous_key = key
        if row["trip_id"] != previous_trip_id:
            if current_group_size:
                group_sum += current_group_size
                group_min = current_group_size if group_min is None else min(group_min, current_group_size)
                group_max = max(group_max, current_group_size)
                if len(group_reservoir) < reservoir_limit:
                    group_reservoir.append(current_group_size)
            current_group_size = 0
            trajectory_count += 1
            previous_trip_id = row["trip_id"]
        current_group_size += 1
        segment_count += 1
    if current_group_size:
        group_sum += current_group_size
        group_min = current_group_size if group_min is None else min(group_min, current_group_size)
        group_max = max(group_max, current_group_size)
        if len(group_reservoir) < reservoir_limit:
            group_reservoir.append(current_group_size)
    if segment_count <= 0:
        raise SystemExit(f"no segment rows read from {path}")
    group_array = np.asarray(group_reservoir, dtype=np.int64)
    return {
        "segment_count": int(segment_count),
        "trajectory_count": int(trajectory_count),
        "min_segments_per_trajectory": int(group_min or 0),
        "mean_segments_per_trajectory": float(group_sum / max(trajectory_count, 1)),
        "p95_segments_per_trajectory": int(np.percentile(group_array, 95)),
        "max_segments_per_trajectory": int(group_max),
    }


def _iter_segment_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row:
                continue
            if line_number == 1 and row[0] == "trip_id":
                continue
            if len(row) < 8:
                raise SystemExit(f"bad segments.tsv row {line_number}: expected 8 columns")
            trip_id = int(row[0])
            segment_idx = int(row[1])
            if trip_id < 0 or segment_idx < 0:
                raise SystemExit(f"bad negative ids in segments.tsv row {line_number}")
            s_x = float(row[2])
            s_y = float(row[3])
            e_x = float(row[4])
            e_y = float(row[5])
            t_s = _timestamp_to_float(row[6])
            t_e = _timestamp_to_float(row[7])
            if not all(math.isfinite(value) for value in (s_x, s_y, e_x, e_y, t_s, t_e)):
                raise SystemExit(f"non-finite segment value in segments.tsv row {line_number}")
            if t_e < t_s:
                raise SystemExit(f"segment t_e < t_s in segments.tsv row {line_number}")
            yield {
                "trip_id": trip_id,
                "segment_idx": segment_idx,
                "s_x": s_x,
                "s_y": s_y,
                "e_x": e_x,
                "e_y": e_y,
                "t_s": t_s,
                "t_e": t_e,
            }


def _read_sample_provenance_tsv(path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    if not path.exists():
        raise SystemExit(f"missing sample provenance TSV: {path}")
    trajectory_ids: list[int] = []
    segment_ids: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header: list[str] | None = None
        for line_number, row in enumerate(reader, 1):
            if not row:
                continue
            if line_number == 1 and not _looks_int(row[0]):
                header = [value.strip() for value in row]
                continue
            if header is None:
                trip_id = int(row[0])
                segment_idx = int(row[1])
            else:
                by_name = {name: row[index] for index, name in enumerate(header)}
                trip_id = int(by_name["trip_id"])
                segment_idx = int(by_name["segment_idx"])
            trajectory_ids.append(trip_id)
            segment_ids.append((trip_id, segment_idx))
    return trajectory_ids, segment_ids


def _looks_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
