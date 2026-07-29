#!/usr/bin/env python3
import argparse
import csv
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ENTRY_RE = re.compile(r"\(([^()]*)\)")


def parse_time(value):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported timestamp: {value}")


def parse_entry(entry):
    vals = entry.split(",")
    if len(vals) != 6:
        raise ValueError(f"Expected 6 tuple fields, found {len(vals)} in {entry}")
    return {
        "sx": vals[0],
        "sy": vals[1],
        "ex": vals[2],
        "ey": vals[3],
        "ts": vals[4],
        "te": vals[5],
        "sx_f": float(vals[0]),
        "sy_f": float(vals[1]),
        "ex_f": float(vals[2]),
        "ey_f": float(vals[3]),
        "ts_dt": parse_time(vals[4]),
        "te_dt": parse_time(vals[5]),
    }


def iter_segment_entries(segment_list):
    return ENTRY_RE.findall(segment_list)


def trajectory_source_sort_key(path):
    match = re.search(r"Trajectory-(\d+)\.tsv\.zip$", path.name)
    part = int(match.group(1)) if match else 10**9
    return (str(path.parent), part, path.name)


def find_trajectory_sources(logs):
    sources = []
    sources.extend(path for path in logs.rglob("Trajectory*.tsv") if path.is_file())
    sources.extend(path for path in logs.rglob("Trajectory*.tsv.zip") if path.is_file())
    sources = sorted(set(sources), key=trajectory_source_sort_key)
    if not sources:
        raise RuntimeError(f"No trajectory TSV sources found below {logs}")
    return sources


def iter_text_lines(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if not name.endswith("/") and Path(name).name.startswith("Trajectory") and name.endswith(".tsv")
            )
            if not members:
                raise RuntimeError(f"No Trajectory*.tsv member found in {path}")
            for member in members:
                with archive.open(member) as handle:
                    for raw in handle:
                        yield raw.decode("utf-8", errors="replace")
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def read_agents_from_characteristics(path):
    agents = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline().rstrip("\n").rstrip("\t").split("\t")
        for line in handle:
            fields = line.rstrip("\n").rstrip("\t").split("\t")
            if not fields or len(fields) < len(header):
                continue
            row = dict(zip(header, fields))
            agent_id = int(row["agentId"])
            agents[agent_id] = {
                "agent_id": agent_id,
                "educationLevel": row["educationLevel"],
                "interest": row["interest"],
                "joviality": row["joviality"],
                "family_size": row["family:numberOfPeople"],
                "age": row.get("age") or row.get("initialAge"),
            }
    return agents


def attach_initial_ages(agents, path):
    earliest_by_agent = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 4:
                continue
            agent_id = int(row[1])
            if agent_id not in agents:
                continue
            timestamp = parse_time(row[2])
            previous = earliest_by_agent.get(agent_id)
            if previous is None or timestamp < previous[0]:
                earliest_by_agent[agent_id] = (timestamp, row[3])
    missing = []
    for agent_id, values in agents.items():
        if agent_id not in earliest_by_agent:
            missing.append(agent_id)
        else:
            values["age"] = earliest_by_agent[agent_id][1]
    if missing:
        raise RuntimeError(f"Missing initial age for {len(missing)} agents, first missing={missing[:5]}")


def ensure_agent_ages(agents, logs):
    missing = [agent_id for agent_id, row in agents.items() if row["age"] in (None, "")]
    if not missing:
        return "AgentCharacteristicsTable.tsv"

    financial_attributes = logs / "FinancialAttributesJournal.csv"
    if not financial_attributes.exists():
        raise RuntimeError(
            "Agent age is missing from AgentCharacteristicsTable.tsv and "
            f"{financial_attributes} does not exist"
        )
    attach_initial_ages(agents, financial_attributes)
    return "FinancialAttributesJournal.csv"


def read_trajectory_rows(logs):
    rows_by_agent = defaultdict(list)
    tuple_count = 0
    line_count = 0
    sources = find_trajectory_sources(logs)
    for source in sources:
        for source_line_no, line in enumerate(iter_text_lines(source), 1):
            line_count += 1
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"Bad TSV column count in {source}:{source_line_no}: {len(parts)}"
                )
            step, agent_id, event_time, segment_list = parts
            entries = iter_segment_entries(segment_list)
            if not entries:
                continue
            first = parse_entry(entries[0])
            last = parse_entry(entries[-1])
            if first["ts_dt"] > last["te_dt"]:
                raise ValueError(f"Trajectory row {source}:{source_line_no} has inverted row time span")
            rows_by_agent[int(agent_id)].append({
                "line_no": line_count,
                "source": str(source),
                "source_line_no": source_line_no,
                "step": int(step),
                "agent_id": int(agent_id),
                "event_time": event_time,
                "event_dt": parse_time(event_time),
                "segment_list": segment_list,
                "first": first,
                "last": last,
                "num_segments": len(entries),
            })
            tuple_count += len(entries)
    return rows_by_agent, tuple_count, line_count, sources


def offset_dataset_agents(dataset, dataset_index, agent_id_offset):
    logs = dataset / "logs"
    agents = read_agents_from_characteristics(logs / "AgentCharacteristicsTable.tsv")
    age_source = ensure_agent_ages(agents, logs)
    rows_by_agent, tuple_count, line_count, trajectory_sources = read_trajectory_rows(logs)

    offset_agents = {}
    offset_rows_by_agent = defaultdict(list)
    for local_agent_id, row in agents.items():
        global_agent_id = agent_id_offset + local_agent_id
        offset_row = dict(row)
        offset_row["agent_id"] = global_agent_id
        offset_agents[global_agent_id] = offset_row

    for local_agent_id, rows in rows_by_agent.items():
        global_agent_id = agent_id_offset + local_agent_id
        for row in rows:
            offset_row = dict(row)
            offset_row["agent_id"] = global_agent_id
            offset_rows_by_agent[global_agent_id].append(offset_row)

    dataset_meta = {
        "dataset_index": dataset_index,
        "dataset": str(dataset),
        "agent_id_offset": agent_id_offset,
        "local_agents": len(agents),
        "local_max_agent_id": max(agents) if agents else -1,
        "age_source": age_source,
        "trajectory_tuple_count": tuple_count,
        "trajectory_line_count": line_count,
        "trajectory_sources": ",".join(str(path) for path in trajectory_sources),
    }
    return offset_agents, offset_rows_by_agent, dataset_meta


def assign_trips(rows_by_agent, xy_tolerance):
    trip_summaries = {}
    row_assignments = defaultdict(list)
    per_agent_counts = {}
    total_segments = 0

    for agent_id in sorted(rows_by_agent):
        rows = sorted(rows_by_agent[agent_id], key=lambda r: (r["first"]["ts_dt"], r["event_dt"], r["line_no"]))
        trip_index = -1
        prev_end = None
        for row in rows:
            first = row["first"]
            starts_new = False
            if prev_end is None:
                starts_new = True
            else:
                time_gap = first["ts_dt"] != prev_end["te_dt"]
                xy_gap = math.hypot(first["sx_f"] - prev_end["ex_f"], first["sy_f"] - prev_end["ey_f"])
                starts_new = time_gap or xy_gap > xy_tolerance

            if starts_new:
                trip_index += 1
                trip_summaries[(agent_id, trip_index)] = {
                    "agent_id": agent_id,
                    "per_agent_trip_index": trip_index,
                    "start_time": first["ts"],
                    "end_time": row["last"]["te"],
                    "num_segments": 0,
                }

            summary = trip_summaries[(agent_id, trip_index)]
            summary["end_time"] = row["last"]["te"]
            summary["num_segments"] += row["num_segments"]
            row_assignments[agent_id].append((row, trip_index))
            prev_end = row["last"]
            total_segments += row["num_segments"]

        per_agent_counts[agent_id] = trip_index + 1

    m = max(per_agent_counts.values()) if per_agent_counts else 0
    exponent = math.ceil(math.log10(m + 1)) if m > 0 else 0
    base = 10 ** exponent
    return trip_summaries, row_assignments, per_agent_counts, total_segments, m, base


def write_outputs(agents, trip_summaries, row_assignments, staging_dir, metadata):
    staging_dir.mkdir(parents=True, exist_ok=True)
    agents_path = staging_dir / "agents.tsv"
    trips_path = staging_dir / "trips.tsv"
    segments_path = staging_dir / "segments.tsv"
    metadata_path = staging_dir / "load_metadata.txt"
    base = metadata["trip_id_base"]

    with agents_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for agent_id in sorted(agents):
            row = agents[agent_id]
            writer.writerow([
                row["agent_id"],
                row["age"],
                row["educationLevel"],
                row["interest"],
                row["joviality"],
                row["family_size"],
            ])

    with trips_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for key in sorted(trip_summaries):
            summary = trip_summaries[key]
            trip_id = summary["agent_id"] * base + summary["per_agent_trip_index"]
            writer.writerow([
                trip_id,
                summary["agent_id"],
                summary["start_time"],
                summary["end_time"],
                summary["num_segments"],
            ])

    segment_rows = 0
    with segments_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        segment_idx_by_trip = defaultdict(int)
        for agent_id in sorted(row_assignments):
            for row, trip_index in row_assignments[agent_id]:
                trip_id = agent_id * base + trip_index
                for entry in iter_segment_entries(row["segment_list"]):
                    seg = parse_entry(entry)
                    if seg["ts_dt"] > seg["te_dt"]:
                        raise ValueError(f"Segment time inversion in trajectory line {row['line_no']}")
                    segment_idx = segment_idx_by_trip[trip_id]
                    segment_idx_by_trip[trip_id] += 1
                    writer.writerow([
                        trip_id,
                        segment_idx,
                        seg["sx"],
                        seg["sy"],
                        seg["ex"],
                        seg["ey"],
                        seg["ts"],
                        seg["te"],
                    ])
                    segment_rows += 1

    metadata["written_segment_rows"] = segment_rows
    with metadata_path.open("w", encoding="utf-8") as handle:
        for key in sorted(metadata):
            handle.write(f"{key}={metadata[key]}\n")

    return agents_path, trips_path, segments_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Parse enriched POL logs to MobilityDB staging TSVs.")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--dataset-list")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--agent-id-stride", type=int)
    parser.add_argument("--xy-tolerance", type=float, default=0.01)
    parser.add_argument("--srid", type=int, default=26916)
    args = parser.parse_args()

    datasets = [Path(value) for value in args.dataset]
    if args.dataset_list:
        with Path(args.dataset_list).open("r", encoding="utf-8") as handle:
            datasets.extend(Path(line.strip()) for line in handle if line.strip() and not line.startswith("#"))
    if not datasets:
        parser.error("Provide at least one --dataset or --dataset-list entry")

    local_max_agent_id = -1
    local_agent_counts = []
    for dataset in datasets:
        agents_for_stride = read_agents_from_characteristics(dataset / "logs" / "AgentCharacteristicsTable.tsv")
        if agents_for_stride:
            local_max_agent_id = max(local_max_agent_id, max(agents_for_stride))
        local_agent_counts.append(len(agents_for_stride))
    agent_id_stride = args.agent_id_stride or (local_max_agent_id + 1)
    if agent_id_stride <= local_max_agent_id:
        raise RuntimeError(
            f"agent-id-stride={agent_id_stride} must be greater than max local agent id={local_max_agent_id}"
        )

    agents = {}
    rows_by_agent = defaultdict(list)
    dataset_metadata = []
    tuple_count = 0
    trajectory_line_count = 0
    for dataset_index, dataset in enumerate(datasets):
        offset = dataset_index * agent_id_stride
        offset_agents, offset_rows, meta = offset_dataset_agents(dataset, dataset_index, offset)
        overlap = set(agents).intersection(offset_agents)
        if overlap:
            raise RuntimeError(f"Global agent id collision, first collisions={sorted(overlap)[:5]}")
        agents.update(offset_agents)
        for agent_id, rows in offset_rows.items():
            rows_by_agent[agent_id].extend(rows)
        dataset_metadata.append(meta)
        tuple_count += meta["trajectory_tuple_count"]
        trajectory_line_count += meta["trajectory_line_count"]

    trip_summaries, row_assignments, per_agent_counts, total_segments, m, base = assign_trips(
        rows_by_agent, args.xy_tolerance
    )

    metadata = {
        "dataset": str(datasets[0]) if len(datasets) == 1 else "multiple",
        "dataset_count": len(datasets),
        "datasets": "|".join(str(dataset) for dataset in datasets),
        "agent_id_stride": agent_id_stride,
        "agent_id_formula": "global_agent_id=dataset_index*agent_id_stride+local_agent_id",
        "dataset_agent_id_offsets": "|".join(
            f"{meta['dataset_index']}:{meta['agent_id_offset']}" for meta in dataset_metadata
        ),
        "dataset_age_sources": "|".join(
            f"{meta['dataset_index']}:{meta['age_source']}" for meta in dataset_metadata
        ),
        "trajectory_sources": "|".join(
            f"{meta['dataset_index']}:{meta['trajectory_sources']}" for meta in dataset_metadata
        ),
        "srid": args.srid,
        "xy_tolerance": args.xy_tolerance,
        "agents": len(agents),
        "trajectory_line_count": trajectory_line_count,
        "trajectory_tuple_count": tuple_count,
        "assigned_segment_count": total_segments,
        "trips": len(trip_summaries),
        "max_per_agent_trip_count_m": m,
        "trip_id_base": base,
        "trip_id_formula": "trip_id=agent_id*B+per_agent_trip_index;B=10^ceil(log10(m+1));per_agent_trip_index_starts_at_0",
        "min_per_agent_trip_count": min(per_agent_counts.values()) if per_agent_counts else 0,
        "max_agent_id": max(agents) if agents else -1,
    }

    paths = write_outputs(agents, trip_summaries, row_assignments, Path(args.staging_dir), metadata)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
