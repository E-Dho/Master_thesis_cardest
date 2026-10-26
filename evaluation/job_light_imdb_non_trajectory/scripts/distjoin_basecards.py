#!/usr/bin/env python3
"""Materialize exact unfiltered join cardinalities required by DistJoin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import load_workload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", action="append", required=True)
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import psycopg2

    representatives = {}
    for workload_index, path in enumerate(args.workload):
        for query in load_workload(path, f"workload_{workload_index}"):
            representatives.setdefault(",".join(sorted(table.name for table in query.tables)), query)
    existing = (
        json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    )
    with psycopg2.connect(args.postgres_dsn) as connection, connection.cursor() as cursor:
        for key, query in sorted(representatives.items()):
            if key in existing:
                continue
            sql = base_cardinality_sql(query)
            cursor.execute(sql)
            existing[key] = int(cursor.fetchone()[0])
            _write(args.output, existing)
            print(f"{key}={existing[key]}", flush=True)
    _write(args.output, existing)
    return 0


def base_cardinality_sql(query) -> str:
    tables = ", ".join(
        table.name if table.alias == table.name else f"{table.name} {table.alias}"
        for table in query.tables
    )
    joins = " AND ".join(f"{join.left}{join.operator}{join.right}" for join in query.joins)
    return f"SELECT COUNT(*) FROM {tables}" + (f" WHERE {joins}" if joins else "")


def _write(path: Path, payload: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
