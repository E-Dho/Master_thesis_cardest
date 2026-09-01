from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.schema import ModelMetadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a POL full-join fixture and aligned trajectory provenance."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--full-join-tsv",
        required=True,
        help=(
            "TSV containing trip_id, segment_idx, and every metadata column name. "
            "Rows are sampled once and provenance is written from the same row."
        ),
    )
    parser.add_argument("--sample-rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    prepared_directory = Path(config["dataset"]["prepared_directory"])
    manifest_path = prepared_directory / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing prepared POL manifest: {manifest_path}")
    metadata = ModelMetadata.from_json_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))["metadata"]
    )
    rows, trajectory_ids, segment_ids = _reservoir_sample_fixture(
        Path(args.full_join_tsv),
        metadata,
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
    )
    prepared_directory.mkdir(parents=True, exist_ok=True)
    np.save(prepared_directory / "sample_rows.npy", rows)
    np.save(prepared_directory / "sample_trajectory_ids.npy", trajectory_ids)
    np.save(prepared_directory / "sample_segment_ids.npy", segment_ids)
    print(f"sample_rows={rows.shape[0]}")
    print(f"sample_trajectory_ids={prepared_directory / 'sample_trajectory_ids.npy'}")
    print(f"sample_segment_ids={prepared_directory / 'sample_segment_ids.npy'}")


def _reservoir_sample_fixture(
    path: Path,
    metadata: ModelMetadata,
    *,
    sample_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_rows <= 0:
        raise SystemExit("--sample-rows must be positive")
    if not path.exists():
        raise SystemExit(f"missing full-join TSV: {path}")
    rng = np.random.default_rng(seed)
    reservoir: list[tuple[np.ndarray, int, tuple[int, int]]] = []
    seen = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"full-join TSV has no header: {path}")
        missing = [column.name for column in metadata.columns if column.name not in reader.fieldnames]
        for required in ("trip_id", "segment_idx"):
            if required not in reader.fieldnames:
                missing.append(required)
        if missing:
            raise SystemExit(f"full-join TSV is missing required columns: {missing}")
        for row in reader:
            encoded = _encode_fixture_row(row, metadata)
            trip_id = int(row["trip_id"])
            segment_idx = int(row["segment_idx"])
            item = (encoded, trip_id, (trip_id, segment_idx))
            seen += 1
            if len(reservoir) < sample_rows:
                reservoir.append(item)
            else:
                replacement = int(rng.integers(0, seen))
                if replacement < sample_rows:
                    reservoir[replacement] = item
    if not reservoir:
        raise SystemExit(f"no rows read from full-join TSV: {path}")
    encoded_rows = np.stack([item[0] for item in reservoir], axis=0).astype(np.int64)
    trajectory_ids = np.asarray([item[1] for item in reservoir], dtype=np.int64)
    segment_ids = np.asarray([item[2] for item in reservoir], dtype=np.int64)
    return encoded_rows, trajectory_ids, segment_ids


def _encode_fixture_row(row: dict[str, str], metadata: ModelMetadata) -> np.ndarray:
    encoded = np.zeros(len(metadata.columns), dtype=np.int64)
    for column_index, column in enumerate(metadata.columns):
        encoded[column_index] = column.encode_value(
            _coerce_domain_value(row[column.name], column.domain)
        )
    return encoded


def _coerce_domain_value(text: str, domain: tuple[Any, ...]) -> Any:
    for candidate in domain:
        if str(candidate) == text:
            return candidate
    try:
        numeric = float(text)
    except ValueError:
        return text
    for candidate in domain:
        try:
            if float(candidate) == numeric:
                return candidate
        except (TypeError, ValueError):
            continue
    return text


if __name__ == "__main__":
    main()
