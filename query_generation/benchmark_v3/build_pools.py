#!/usr/bin/env python3
"""Build the seeded center pools of a v3 run (once, before generation)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from query_generation.query_generator import parse_timestamp

from .config import Settings, load_settings
from .pools import Pools, encode_value, pool_columns, subsample


def data_domains(settings: Settings, session) -> Dict[str, Dict[str, Any]]:
    """Measured min/max of every ordered attribute, keyed like ``domain_overrides``."""

    seg = settings.segments
    domains: Dict[str, Dict[str, Any]] = {}
    for table_id, table in sorted(settings.base["tables"].items()):
        parts: List[str] = []
        keys: List[tuple] = []
        for attr in table["attributes"]:
            name = f"{table_id}.{attr['name']}"
            if attr["type"] in {"numeric", "integer"}:
                parts += [f"min({attr['expression']})", f"max({attr['expression']})"]
                keys += [(name, "min"), (name, "max")]
            elif attr["type"] == "temporal_interval":
                parts += [f"min({attr['start_expression']})", f"max({attr['end_expression']})"]
                keys += [(name, "min"), (name, "max")]
            elif attr["type"] == "geometry" and table_id == seg.table:
                c = seg.columns
                parts += [
                    f"least(min({c['sx']}), min({c['ex']}))", f"least(min({c['sy']}), min({c['ey']}))",
                    f"greatest(max({c['sx']}), max({c['ex']}))", f"greatest(max({c['sy']}), max({c['ey']}))",
                ]
                keys += [(name, "min_x"), (name, "min_y"), (name, "max_x"), (name, "max_y")]
        if not parts:
            continue
        values = session.rows(f"SELECT {', '.join(parts)} FROM {table['name']} {table['alias']}")[0]
        for (name, field), value in zip(keys, values):
            domains.setdefault(name, {})[field] = encode_value(value)
    return domains


def domain_violations(settings: Settings, domains: Dict[str, Dict[str, Any]]) -> List[str]:
    """Data outside the configured domain cannot be reached by range predicates."""

    problems = []
    for name, measured in sorted(domains.items()):
        table_id, _, attribute = name.partition(".")
        configured = settings.attr(table_id, attribute)["domain"]
        for field, value in measured.items():
            bound = configured[field]
            if field in ("min", "max") and settings.attr(table_id, attribute)["type"] == "temporal_interval":
                value, bound = parse_timestamp(value), parse_timestamp(bound)
            else:
                value, bound = float(value), float(bound)
            outside = value < bound if field.startswith("min") else value > bound
            if outside:
                problems.append(f"{name}.{field}: data {value} outside configured {bound}")
    return problems


def diagnostics(settings: Settings, session) -> Dict[str, Any]:
    """Informational checks; failures are recorded, never raised."""

    result: Dict[str, Any] = {}
    seg = settings.segments
    seg_table = settings.table(seg.table)

    def attempt(key, sql):
        try:
            result[key] = [[encode_value(v) for v in row] for row in session.rows(sql)]
        except Exception as error:  # noqa: BLE001 - diagnostics only
            result[key] = f"unavailable: {type(error).__name__}: {error}"

    attempt(
        "pg_stats_columns_per_table",
        "SELECT tablename, count(*) FROM pg_stats WHERE schemaname = "
        f"'{settings.base['schema']}' GROUP BY tablename ORDER BY tablename",
    )
    # segment_tgeom is not used for labels; this measures how far its start
    # time deviates from t_s (rounding vs. time-zone shift), on a 1 % sample.
    attempt(
        "segment_tgeom_start_offset_seconds",
        "SELECT count(*) FILTER (WHERE d <> 0), count(*), min(d), max(d), "
        "count(*) FILTER (WHERE abs(d) >= 1) FROM ("
        f"SELECT EXTRACT(EPOCH FROM (startTimestamp(segment_tgeom) - ({seg.columns['ts']} AT TIME ZONE 'UTC')))::float8 AS d "
        f"FROM {seg_table['name']} {seg_table['alias']} TABLESAMPLE SYSTEM (1) REPEATABLE (1)) q",
    )
    return result


def database_fingerprint(settings: Settings, session) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {}
    for table_id, table in sorted(settings.base["tables"].items()):
        fingerprint[f"{table_id}.rows"] = int(session.scalar(f"SELECT count(*) FROM {table['name']}"))
    fingerprint["data_domains"] = data_domains(settings, session)
    fingerprint["domain_violations"] = domain_violations(settings, fingerprint["data_domains"])
    fingerprint["diagnostics"] = diagnostics(settings, session)
    return fingerprint


def build_pools(settings: Settings, session, base_seed: int) -> Pools:
    repeatable = int(base_seed) % 2147483647
    tables: Dict[str, Any] = {}
    for table_id in sorted(settings.base["tables"]):
        columns = pool_columns(settings, table_id)
        if not columns:
            continue
        table = settings.table(table_id)
        reltuples = float(
            session.scalar("SELECT reltuples::float8 FROM pg_class WHERE oid = %s::regclass", [table["name"]])
        )
        if reltuples <= 0:
            reltuples = float(session.scalar(f"SELECT count(*) FROM {table['name']}"))
        percent = min(100.0, 100.0 * settings.pool_oversample * settings.pool_size / max(reltuples, 1.0))
        sql = (
            f"SELECT {', '.join(expression for _, expression in columns)} "
            f"FROM {table['name']} {table['alias']} "
            f"TABLESAMPLE BERNOULLI ({percent!r}) REPEATABLE ({repeatable})"
        )
        rows = session.rows(sql)
        seed = int.from_bytes(hashlib.blake2b(f"pools:{base_seed}:{table_id}".encode(), digest_size=8).digest(), "big")
        tables[table_id] = {
            "columns": [name for name, _ in columns],
            "rows": subsample(rows, settings.pool_size, seed),
            "sampled_rows": len(rows),
            "sql": sql,
        }
    base_identity = hashlib.sha256(
        json.dumps({"base": settings.base, "pools": settings.v3["pools"]}, sort_keys=True).encode()
    ).hexdigest()
    binding = {
        "base_seed": int(base_seed),
        "pool_size": settings.pool_size,
        "base_config_sha256": base_identity,
        "database": database_fingerprint(settings, session),
    }
    return Pools(tables, binding)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .backends import SqlSession

    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists():
        print(f"{output} exists; pools are built once per run and never overwritten", file=sys.stderr)
        return 1
    settings = load_settings(Path(args.config))
    with SqlSession(args.host, args.port, args.dbname, args.user) as session:
        pools = build_pools(settings, session, args.base_seed)
    pools.save(output)
    violations = pools.binding["database"]["domain_violations"]
    print(json.dumps({
        "domain_violations": violations,
        "pools": str(output),
        "sha256": pools.sha256,
        "rows": {table: len(payload["rows"]) for table, payload in pools.tables.items()},
        "binding": pools.binding,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
