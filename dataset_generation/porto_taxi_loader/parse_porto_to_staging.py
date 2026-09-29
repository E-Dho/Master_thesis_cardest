#!/usr/bin/env python3
"""Create deterministic, strict Porto taxi staging TSVs from train.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_URL = "https://www.kaggle.com/datasets/crailtap/taxi-trajectory"
SAMPLE_SECONDS = 15


class InvalidTrace(ValueError):
    """A row cannot participate in the strict complete-trace population."""


def parse_polyline(value: str) -> list[tuple[float, float]]:
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidTrace("invalid_polyline_json") from exc
    if not isinstance(raw, list) or len(raw) < 2:
        raise InvalidTrace("fewer_than_two_points")
    points: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise InvalidTrace("invalid_point_shape")
        try:
            longitude, latitude = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise InvalidTrace("non_numeric_coordinate") from exc
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise InvalidTrace("non_finite_coordinate")
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise InvalidTrace("coordinate_out_of_bounds")
        points.append((longitude, latitude))
    return points


def parse_complete_row(row: dict[str, str]) -> tuple[str, int, int, list[tuple[float, float]]]:
    if str(row.get("MISSING_DATA", "")).strip().lower() != "false":
        raise InvalidTrace("missing_data")
    source_trip_id = str(row.get("TRIP_ID", "")).strip()
    if not source_trip_id:
        raise InvalidTrace("missing_trip_id")
    try:
        taxi_id = int(str(row["TAXI_ID"]).strip())
        timestamp = int(str(row["TIMESTAMP"]).strip())
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidTrace("invalid_trip_metadata") from exc
    return source_trip_id, taxi_id, timestamp, parse_polyline(str(row.get("POLYLINE", "")))


def stable_hash(seed: str, source_trip_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{source_trip_id}".encode("utf-8")).hexdigest()


def timestamp_text(unix_seconds: int, point_index: int) -> str:
    moment = datetime.fromtimestamp(unix_seconds, tz=timezone.utc) + timedelta(seconds=point_index * SAMPLE_SECONDS)
    return moment.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")


def nullable_integer(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return r"\N"
    return str(int(text))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_candidate_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE candidates (hash TEXT PRIMARY KEY, source_trip_id TEXT UNIQUE NOT NULL, row_number INTEGER UNIQUE NOT NULL, segment_count INTEGER NOT NULL, selected_rank INTEGER)"
    )
    return connection


def scan_candidates(csv_path: Path, connection: sqlite3.Connection, seed: str) -> Counter[str]:
    stats: Counter[str] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise SystemExit(f"CSV is missing required columns: {sorted(required)}")
        batch: list[tuple[str, str, int, int]] = []
        for row_number, row in enumerate(reader, start=2):
            stats["source_rows"] += 1
            try:
                source_trip_id, _, _, points = parse_complete_row(row)
            except InvalidTrace as exc:
                stats[str(exc)] += 1
                continue
            batch.append((stable_hash(seed, source_trip_id), source_trip_id, row_number, len(points) - 1))
            if len(batch) >= 10_000:
                try:
                    connection.executemany("INSERT INTO candidates(hash, source_trip_id, row_number, segment_count) VALUES (?, ?, ?, ?)", batch)
                except sqlite3.IntegrityError as exc:
                    raise SystemExit("TRIP_ID values must be unique in train.csv") from exc
                connection.commit()
                batch.clear()
        if batch:
            try:
                connection.executemany("INSERT INTO candidates(hash, source_trip_id, row_number, segment_count) VALUES (?, ?, ?, ?)", batch)
            except sqlite3.IntegrityError as exc:
                raise SystemExit("TRIP_ID values must be unique in train.csv") from exc
            connection.commit()
    stats["eligible_trips"] = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    stats["eligible_segments"] = connection.execute("SELECT COALESCE(SUM(segment_count), 0) FROM candidates").fetchone()[0]
    return stats


def select_candidates(connection: sqlite3.Connection, target_segments: int) -> tuple[int, int]:
    selected_segments = 0
    selected_trips = 0
    updates: list[tuple[int, str]] = []
    for source_hash, segment_count in connection.execute("SELECT hash, segment_count FROM candidates ORDER BY hash"):
        if selected_segments + segment_count > target_segments:
            break
        selected_trips += 1
        selected_segments += segment_count
        updates.append((selected_trips, source_hash))
    connection.executemany("UPDATE candidates SET selected_rank = ? WHERE hash = ?", updates)
    connection.commit()
    return selected_trips, selected_segments


def selected_rows(connection: sqlite3.Connection) -> dict[int, int]:
    return {
        int(row_number): int(selected_rank)
        for row_number, selected_rank in connection.execute(
            "SELECT row_number, selected_rank FROM candidates WHERE selected_rank IS NOT NULL"
        )
    }


def write_staging(csv_path: Path, connection: sqlite3.Connection, staging_dir: Path) -> dict[str, int]:
    selected = selected_rows(connection)
    taxis: set[int] = set()
    trips_written = 0
    segments_written = 0
    staging_dir.mkdir(parents=True, exist_ok=True)
    trips_path = staging_dir / "trips.tsv"
    segments_path = staging_dir / "segments_wgs84.tsv"
    with trips_path.open("w", encoding="utf-8", newline="") as trips_handle, segments_path.open("w", encoding="utf-8", newline="") as segments_handle:
        trips_writer = csv.writer(trips_handle, delimiter="\t", lineterminator="\n")
        segments_writer = csv.writer(segments_handle, delimiter="\t", lineterminator="\n")
        with csv_path.open("r", encoding="utf-8", newline="") as source_handle:
            for row_number, row in enumerate(csv.DictReader(source_handle), start=2):
                trip_id = selected.get(row_number)
                if trip_id is None:
                    continue
                source_trip_id, taxi_id, unix_start, points = parse_complete_row(row)
                num_segments = len(points) - 1
                start_time = timestamp_text(unix_start, 0)
                end_time = timestamp_text(unix_start, num_segments)
                trips_writer.writerow([
                    trip_id,
                    source_trip_id,
                    taxi_id,
                    row.get("CALL_TYPE", "").strip(),
                    nullable_integer(row.get("ORIGIN_CALL")),
                    nullable_integer(row.get("ORIGIN_STAND")),
                    start_time,
                    end_time,
                    row.get("DAYTYPE", "").strip(),
                    num_segments,
                ])
                taxis.add(taxi_id)
                trips_written += 1
                for segment_idx, ((s_lon, s_lat), (e_lon, e_lat)) in enumerate(zip(points, points[1:])):
                    segments_writer.writerow([
                        trip_id,
                        segment_idx,
                        s_lon,
                        s_lat,
                        e_lon,
                        e_lat,
                        timestamp_text(unix_start, segment_idx),
                        timestamp_text(unix_start, segment_idx + 1),
                    ])
                    segments_written += 1
    with (staging_dir / "taxis.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for taxi_id in sorted(taxis):
            writer.writerow([taxi_id])
    with (staging_dir / "selected_trips.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in connection.execute(
            "SELECT selected_rank, source_trip_id, hash, segment_count FROM candidates WHERE selected_rank IS NOT NULL ORDER BY selected_rank"
        ):
            writer.writerow(row)
    return {"taxis": len(taxis), "trips": trips_written, "segments": segments_written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict, deterministic Porto taxi MobilityDB staging TSVs.")
    parser.add_argument("--input", required=True, help="Original Kaggle train.csv")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--target-segments", type=int, default=50_000_000)
    parser.add_argument("--selection-seed", default="porto_taxi_50m_v1")
    parser.add_argument("--keep-selection-db", action="store_true")
    args = parser.parse_args()
    if args.target_segments <= 0:
        raise SystemExit("--target-segments must be positive")

    csv_path = Path(args.input)
    if not csv_path.is_file():
        raise SystemExit(f"missing input CSV: {csv_path}")
    staging_dir = Path(args.staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    selection_db = staging_dir / "selection.sqlite"
    if selection_db.exists():
        selection_db.unlink()
    connection = create_candidate_db(selection_db)
    try:
        stats = scan_candidates(csv_path, connection, args.selection_seed)
        selected_trips, selected_segments = select_candidates(connection, args.target_segments)
        written = write_staging(csv_path, connection, staging_dir)
    finally:
        connection.close()

    if written["trips"] != selected_trips or written["segments"] != selected_segments:
        raise SystemExit("staging output does not match deterministic selection")
    metadata: dict[str, Any] = {
        "source_url": SOURCE_URL,
        "input_csv": str(csv_path),
        "input_sha256": sha256_file(csv_path),
        "selection_seed": args.selection_seed,
        "target_segments": args.target_segments,
        "selected_trips": selected_trips,
        "selected_segments": selected_segments,
        "sample_seconds": SAMPLE_SECONDS,
        "input_srid": 4326,
        "output_srid": 3763,
        "trace_policy": "MISSING_DATA=false and at least two finite WGS84 points",
        **dict(sorted(stats.items())),
        **written,
    }
    (staging_dir / "load_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_selection_db:
        selection_db.unlink(missing_ok=True)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
