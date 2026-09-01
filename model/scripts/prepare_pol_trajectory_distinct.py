from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.schema import ModelMetadata
from model.src.data.trajectory_distinct import (
    CompactTrajectorySegmentIndex,
    write_compact_trajectory_index,
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
    segments = _read_segments_tsv(Path(args.segments_tsv))
    compact_manifest = write_compact_trajectory_index(
        output_directory,
        metadata=metadata,
        trip_ids=segments["trip_id"],
        segment_idx=segments["segment_idx"],
        t_s=segments["t_s"],
        t_e=segments["t_e"],
        s_x=segments["s_x"],
        s_y=segments["s_y"],
        e_x=segments["e_x"],
        e_y=segments["e_y"],
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
        np.save(prepared_directory / "sample_segment_ids.npy", np.asarray(segment_ids, dtype=object))

    storage = index.storage_summary()
    print(f"trajectory_index_path={output_directory}")
    print(f"trajectory_index_format_version={compact_manifest['format_version']}")
    print(f"trajectory_count={storage['trajectory_count']}")
    print(f"segment_count={storage['segment_count']}")
    print(f"index_bytes={storage['index_bytes']}")
    print(f"bytes_per_segment={storage['bytes_per_segment']:.3f}")
    print(f"estimated_bytes_50m_segments={storage['estimated_bytes_50m_segments']}")
    print(f"compatibility_hash={compact_manifest['compatibility_hash']}")


def _read_segments_tsv(path: Path) -> dict[str, list[Any]]:
    if not path.exists():
        raise SystemExit(f"missing POL segments TSV: {path}")
    columns = {
        "trip_id": [],
        "segment_idx": [],
        "s_x": [],
        "s_y": [],
        "e_x": [],
        "e_y": [],
        "t_s": [],
        "t_e": [],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row:
                continue
            if line_number == 1 and row[0] == "trip_id":
                continue
            if len(row) < 8:
                raise SystemExit(f"bad segments.tsv row {line_number}: expected 8 columns")
            columns["trip_id"].append(int(row[0]))
            columns["segment_idx"].append(int(row[1]))
            columns["s_x"].append(float(row[2]))
            columns["s_y"].append(float(row[3]))
            columns["e_x"].append(float(row[4]))
            columns["e_y"].append(float(row[5]))
            columns["t_s"].append(row[6])
            columns["t_e"].append(row[7])
    if not columns["trip_id"]:
        raise SystemExit(f"no segment rows read from {path}")
    return columns


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
