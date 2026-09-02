from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.complete_domain_preparation import (
    COMPLETE_METADATA_SOURCE,
    EncodedSampleRows,
    complete_ordinary_domain,
    preparation_stats,
)
from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.trajectory_distinct import _timestamp_to_float
from model.src.model.factorization import FactorizationConfig
from model.scripts.prepare_pol_trajectory_distinct import _write_ordered_segments_index


AGENT_COLUMNS = (
    "agent_id",
    "age",
    "educationLevel",
    "interest",
    "joviality",
    "family_size",
)
TRIP_COLUMNS = (
    "trip_id",
    "agent_id",
    "start_time",
    "end_time",
    "num_of_segments",
)
SEGMENT_COLUMNS = (
    "trip_id",
    "segment_idx",
    "s_x",
    "s_y",
    "e_x",
    "e_y",
    "t_s",
    "t_e",
)
MODEL_SEGMENT_COLUMNS = (
    "segment_idx",
    "t_s",
    "t_e",
    "s_x",
    "s_y",
    "e_x",
    "e_y",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare POL model fixtures directly from MobilityDB staging TSVs. "
            "The staging files are expected to be headerless in the loader's fixed order."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Rows to materialize; 0 means all segment rows.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prepared-directory",
        default=None,
        help="Override dataset.prepared_directory from the config.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_simple_yaml(args.config)
    validate_config(config)
    staging = Path(args.staging_dir)
    prepared = Path(args.prepared_directory or config["dataset"]["prepared_directory"])
    agents_path = staging / "agents.tsv"
    trips_path = staging / "trips.tsv"
    segments_path = staging / "segments.tsv"
    for path in (agents_path, trips_path, segments_path):
        if not path.exists():
            raise SystemExit(f"missing POL staging TSV: {path}")

    agents, agent_domains = _read_agents(agents_path)
    trips, trip_domains, trips_per_agent = _read_trips(trips_path)
    segment_domains, segments_per_trip, segment_count = _scan_segments(segments_path)
    metadata = _build_metadata(
        agent_domains=agent_domains,
        trip_domains=trip_domains,
        segment_domains=segment_domains,
        trips_per_agent=trips_per_agent,
        segments_per_trip=segments_per_trip,
        full_join_cardinality=segment_count,
    )

    prepared.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 2,
        "dataset_name": str(config["dataset"].get("name", "pol_trajectory_full_join")),
        "dataset_type": "pol_trajectory_full_join",
        "join_cardinality": int(segment_count),
        "metadata": metadata.to_json_dict(),
        "source": "pol_mobilitydb_staging_tsv",
        "metadata_source": COMPLETE_METADATA_SOURCE,
        "sample_source": "pol_mobilitydb_staging_tsv",
        "sample_rows": int(segment_count if int(args.sample_rows) <= 0 else int(args.sample_rows)),
        "domains_complete": True,
        "staging_directory": str(staging),
    }
    (prepared / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    encoded_rows, trajectory_ids, segment_ids = _write_sample_rows(
        segments_path,
        metadata=metadata,
        agents=agents,
        trips=trips,
        trips_per_agent=trips_per_agent,
        segments_per_trip=segments_per_trip,
        output_directory=prepared,
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
    )
    np.save(prepared / "sample_trajectory_ids.npy", trajectory_ids)
    np.save(prepared / "sample_segment_ids.npy", segment_ids)

    trajectory_config = config.get("trajectory_distinct", {})
    index_dir = Path(config["dataset"].get("trajectory_index_path", prepared / "trajectory_segment_index"))
    if not index_dir.is_absolute():
        index_dir = prepared / index_dir.name if index_dir.parent == Path("data/pol_prepared/pol_50m") else index_dir
    if str(index_dir).startswith("data/"):
        index_dir = prepared / "trajectory_segment_index"
    compact_manifest = _write_ordered_segments_index(
        segments_path,
        output_directory=index_dir,
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
    )

    stats = preparation_stats(
        metadata=metadata,
        encoded_sample=EncodedSampleRows(encoded_rows=np.asarray(encoded_rows)),
        factorization_config=FactorizationConfig.from_dict(config.get("factorization", {})),
    )
    stats.update(
        {
            "staging_directory": str(staging),
            "trajectory_index_path": str(index_dir),
            "trajectory_index_segment_count": compact_manifest["segment_count"],
            "preparation_seconds": float(time.perf_counter() - started),
            "materialized_all_segment_rows": bool(int(args.sample_rows) <= 0),
        }
    )
    (prepared / "preparation_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"prepared_directory={prepared}")
    print(f"schema_hash={metadata.stable_schema_hash()}")
    print(f"join_cardinality={segment_count}")
    print(f"sample_rows={len(encoded_rows)}")
    print(f"trajectory_index_path={index_dir}")
    print(f"preparation_seconds={time.perf_counter() - started:.3f}")


def _read_agents(path: Path) -> tuple[dict[int, tuple[Any, ...]], dict[str, set[Any]]]:
    agents: dict[int, tuple[Any, ...]] = {}
    domains = {name: set() for name in AGENT_COLUMNS[1:]}
    for row in _iter_tsv(path):
        if len(row) < len(AGENT_COLUMNS):
            raise SystemExit(f"bad agents.tsv row: expected {len(AGENT_COLUMNS)} columns")
        agent_id = int(row[0])
        values = (
            float(row[1]),
            row[2],
            row[3],
            float(row[4]),
            int(row[5]),
        )
        agents[agent_id] = values
        for name, value in zip(AGENT_COLUMNS[1:], values):
            domains[name].add(value)
    return agents, domains


def _read_trips(
    path: Path,
) -> tuple[dict[int, tuple[int, Any, Any, int, Any]], dict[str, set[Any]], Counter[int]]:
    trips: dict[int, tuple[int, Any, Any, int, Any]] = {}
    domains = {name: set() for name in ("start_time", "end_time", "num_of_segments", "trip_geom")}
    trips_per_agent: Counter[int] = Counter()
    for row in _iter_tsv(path):
        if len(row) < len(TRIP_COLUMNS):
            raise SystemExit(f"bad trips.tsv row: expected {len(TRIP_COLUMNS)} columns")
        trip_id = int(row[0])
        agent_id = int(row[1])
        start_time = _timestamp_to_float(row[2])
        end_time = _timestamp_to_float(row[3])
        num_segments = int(row[4])
        trip_geom = OUTER_MISSING
        trips[trip_id] = (agent_id, start_time, end_time, num_segments, trip_geom)
        trips_per_agent[agent_id] += 1
        domains["start_time"].add(start_time)
        domains["end_time"].add(end_time)
        domains["num_of_segments"].add(num_segments)
        domains["trip_geom"].add(trip_geom)
    return trips, domains, trips_per_agent


def _scan_segments(
    path: Path,
) -> tuple[dict[str, set[Any]], Counter[int], int]:
    domains = {name: set() for name in SEGMENT_COLUMNS[1:]}
    segments_per_trip: Counter[int] = Counter()
    count = 0
    for segment in _iter_segment_values(path):
        trip_id = int(segment["trip_id"])
        segments_per_trip[trip_id] += 1
        count += 1
        for name in SEGMENT_COLUMNS[1:]:
            domains[name].add(segment[name])
    return domains, segments_per_trip, count


def _build_metadata(
    *,
    agent_domains: dict[str, set[Any]],
    trip_domains: dict[str, set[Any]],
    segment_domains: dict[str, set[Any]],
    trips_per_agent: Counter[int],
    segments_per_trip: Counter[int],
    full_join_cardinality: int,
) -> ModelMetadata:
    columns: list[ColumnMetadata] = []
    for name in AGENT_COLUMNS[1:]:
        columns.append(
            ColumnMetadata(
                f"agents:{name}",
                ColumnKind.DATA,
                complete_ordinary_domain(agent_domains[name]),
                table="agents",
            )
        )
    for name in ("start_time", "end_time", "num_of_segments", "trip_geom"):
        columns.append(
            ColumnMetadata(
                f"trips:{name}",
                ColumnKind.DATA,
                complete_ordinary_domain(trip_domains[name]),
                table="trips",
            )
        )
    for name in MODEL_SEGMENT_COLUMNS:
        columns.append(
            ColumnMetadata(
                f"segments:{name}",
                ColumnKind.DATA,
                complete_ordinary_domain(segment_domains[name]),
                table="segments",
            )
        )
    columns.extend(
        (
            ColumnMetadata("I_agents", ColumnKind.INDICATOR, (0, 1), table="agents"),
            ColumnMetadata("I_trips", ColumnKind.INDICATOR, (0, 1), table="trips"),
            ColumnMetadata("I_segments", ColumnKind.INDICATOR, (0, 1), table="segments"),
            ColumnMetadata(
                "F_agents_to_trips",
                ColumnKind.FANOUT,
                _dense_positive_domain(trips_per_agent.values()),
                table="agents",
                fanout_source="agents->trips",
            ),
            ColumnMetadata(
                "F_trips_to_segments",
                ColumnKind.FANOUT,
                _dense_positive_domain(segments_per_trip.values()),
                table="trips",
                fanout_source="trips->segments",
            ),
        )
    )
    return ModelMetadata(
        columns=tuple(columns),
        full_join_cardinality=float(full_join_cardinality),
        column_order="data_indicators_fanouts",
        upstream_attribution={
            "POL": "MobilityDB staging TSVs with trajectory segment provenance",
            "NeuroCard": "full-outer-join indicators and fanout correction semantics",
        },
        join_root="agents",
        join_tables=("agents", "trips", "segments"),
        join_edges=(("agents", "trips"), ("trips", "segments")),
    )


def _write_sample_rows(
    segments_path: Path,
    *,
    metadata: ModelMetadata,
    agents: dict[int, tuple[Any, ...]],
    trips: dict[int, tuple[int, Any, Any, int, Any]],
    trips_per_agent: Counter[int],
    segments_per_trip: Counter[int],
    output_directory: Path,
    sample_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_segments = sum(segments_per_trip.values())
    if sample_rows <= 0 or sample_rows >= total_segments:
        selected_indices = None
        output_rows = total_segments
    else:
        rng = np.random.default_rng(seed)
        selected_indices = set(
            int(index)
            for index in rng.choice(total_segments, size=sample_rows, replace=False)
        )
        output_rows = sample_rows
    encoded = np.lib.format.open_memmap(
        output_directory / "sample_rows.npy",
        mode="w+",
        dtype=np.int64,
        shape=(output_rows, len(metadata.columns)),
    )
    trajectory_ids = np.empty(output_rows, dtype=np.int64)
    segment_ids = np.empty((output_rows, 2), dtype=np.int64)
    domain_maps = [
        {value: index for index, value in enumerate(column.domain)}
        for column in metadata.columns
    ]
    out_index = 0
    for segment_index, segment in enumerate(_iter_segment_values(segments_path)):
        if selected_indices is not None and segment_index not in selected_indices:
            continue
        trip_id = int(segment["trip_id"])
        trip = trips.get(trip_id)
        if trip is None:
            raise SystemExit(f"segments.tsv references unknown trip_id {trip_id}")
        agent_id, start_time, end_time, num_segments, trip_geom = trip
        agent = agents.get(agent_id)
        if agent is None:
            raise SystemExit(f"trips.tsv references unknown agent_id {agent_id}")
        values = (
            *agent,
            start_time,
            end_time,
            num_segments,
            trip_geom,
            segment["segment_idx"],
            segment["t_s"],
            segment["t_e"],
            segment["s_x"],
            segment["s_y"],
            segment["e_x"],
            segment["e_y"],
            1,
            1,
            1,
            int(trips_per_agent[agent_id]),
            int(segments_per_trip[trip_id]),
        )
        if len(values) != len(metadata.columns):
            raise AssertionError("POL encoded value width does not match metadata")
        for column_index, value in enumerate(values):
            encoded[out_index, column_index] = domain_maps[column_index][value]
        trajectory_ids[out_index] = trip_id
        segment_ids[out_index, 0] = trip_id
        segment_ids[out_index, 1] = int(segment["segment_idx"])
        out_index += 1
    if out_index != output_rows:
        raise SystemExit(f"sample row count mismatch: wrote {out_index}, expected {output_rows}")
    encoded.flush()
    return np.asarray(encoded), trajectory_ids, segment_ids


def _iter_segment_values(path: Path) -> Iterable[dict[str, Any]]:
    for row in _iter_tsv(path):
        if len(row) < len(SEGMENT_COLUMNS):
            raise SystemExit(f"bad segments.tsv row: expected {len(SEGMENT_COLUMNS)} columns")
        yield {
            "trip_id": int(row[0]),
            "segment_idx": int(row[1]),
            "s_x": float(row[2]),
            "s_y": float(row[3]),
            "e_x": float(row[4]),
            "e_y": float(row[5]),
            "t_s": _timestamp_to_float(row[6]),
            "t_e": _timestamp_to_float(row[7]),
        }


def _iter_tsv(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def _dense_positive_domain(values: Iterable[int]) -> tuple[int, ...]:
    maximum = max((int(value) for value in values), default=1)
    if maximum <= 0:
        maximum = 1
    return tuple(range(1, maximum + 1))


if __name__ == "__main__":
    main()
