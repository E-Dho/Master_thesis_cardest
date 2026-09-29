#!/usr/bin/env python3
"""Build the runtime Porto query-generator config from loaded database domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _timestamp(value: Any) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_config(domains: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "porto_taxi_50m_segment_coupled_v1",
        "schema": "porto",
        "srid": 3763,
        "entity": {"table": "trips", "key": "trip_id", "expression": "t.trip_id"},
        "tables": {
            "taxis": {
                "name": "porto.taxis", "alias": "x", "primary_key": "taxi_id", "flags": ["standard"],
                "attributes": [{"name": "taxi_id", "type": "integer", "dimension": "standard", "expression": "x.taxi_id", "domain": domains["taxi_id"]}],
            },
            "trips": {
                "name": "porto.trips", "alias": "t", "primary_key": "trip_id", "flags": ["standard"],
                "attributes": [
                    {"name": "call_type", "type": "nominal", "dimension": "standard", "expression": "t.call_type", "values": domains["call_type"]},
                    {"name": "origin_call", "type": "integer", "dimension": "standard", "expression": "t.origin_call", "domain": domains["origin_call"]},
                    {"name": "origin_stand", "type": "integer", "dimension": "standard", "expression": "t.origin_stand", "domain": domains["origin_stand"]},
                    {"name": "daytype", "type": "nominal", "dimension": "standard", "expression": "t.daytype", "values": domains["daytype"]},
                    {"name": "num_of_segments", "type": "integer", "dimension": "standard", "expression": "t.num_of_segments", "domain": domains["num_of_segments"]},
                ],
            },
            "segments": {
                "name": "porto.segments", "alias": "s", "primary_key": "trip_id, segment_idx", "flags": ["standard", "temporal", "spatial"],
                "attributes": [
                    {"name": "segment_idx", "type": "integer", "dimension": "standard", "expression": "s.segment_idx", "domain": domains["segment_idx"]},
                    {"name": "segment_time", "type": "temporal_interval", "dimension": "temporal", "start_expression": "s.t_s", "end_expression": "s.t_e", "temporal_geometry_expression": "s.segment_tgeom", "domain": domains["segment_time"]},
                    {"name": "segment_geom", "type": "geometry", "dimension": "spatial", "expression": "s.segment_geom", "srid": 3763, "domain": domains["segment_geom"]},
                ],
            },
        },
        "joins": [
            {"left": "taxis", "right": "trips", "condition": "x.taxi_id = t.taxi_id"},
            {"left": "trips", "right": "segments", "condition": "t.trip_id = s.trip_id"},
        ],
    }


def _scalar(cursor: Any, sql: str) -> Any:
    cursor.execute(sql)
    return cursor.fetchone()[0]


def _range(cursor: Any, table: str, column: str) -> dict[str, int]:
    cursor.execute(f"SELECT MIN({column}), MAX({column}) FROM {table} WHERE {column} IS NOT NULL")
    lower, upper = cursor.fetchone()
    if lower is None or upper is None or lower >= upper:
        raise SystemExit(f"invalid domain for {table}.{column}: {lower!r}, {upper!r}")
    return {"min": int(lower), "max": int(upper)}


def read_domains(host: str | None, port: int | None, dbname: str | None, user: str | None) -> dict[str, Any]:
    try:
        import psycopg  # type: ignore
    except ImportError as exc:
        raise SystemExit("psycopg is required to inspect Porto database domains") from exc
    kwargs = {key: value for key, value in {"host": host, "port": port, "dbname": dbname, "user": user}.items() if value is not None}
    with psycopg.connect(**kwargs) as connection, connection.cursor() as cursor:
        domains: dict[str, Any] = {
            "taxi_id": _range(cursor, "porto.taxis", "taxi_id"),
            "origin_call": _range(cursor, "porto.trips", "origin_call"),
            "origin_stand": _range(cursor, "porto.trips", "origin_stand"),
            "num_of_segments": _range(cursor, "porto.trips", "num_of_segments"),
            "segment_idx": _range(cursor, "porto.segments", "segment_idx"),
        }
        for key, column in (("call_type", "call_type"), ("daytype", "daytype")):
            cursor.execute(f"SELECT DISTINCT {column} FROM porto.trips ORDER BY {column}")
            domains[key] = [row[0] for row in cursor.fetchall()]
        cursor.execute("SELECT MIN(t_s), MAX(t_e), MIN(s_x), MIN(s_y), MAX(s_x), MAX(s_y) FROM porto.segments")
        start, end, min_x, min_y, max_x, max_y = cursor.fetchone()
        if start is None or end is None or min_x >= max_x or min_y >= max_y:
            raise SystemExit("invalid Porto segment temporal or spatial domain")
        domains["segment_time"] = {"min": _timestamp(start), "max": _timestamp(end)}
        domains["segment_geom"] = {"min_x": float(min_x), "min_y": float(min_y), "max_x": float(max_x), "max_y": float(max_y)}
    return domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a domain-complete Porto taxi query config.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    args = parser.parse_args()
    config = build_config(read_domains(args.host, args.port, args.dbname, args.user))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
