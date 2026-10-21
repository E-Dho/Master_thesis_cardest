#!/usr/bin/env python3
"""Build the seeded center pools of a v3 run (once, before generation)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings, load_settings
from .pools import Pools, pool_columns, subsample


def database_fingerprint(settings: Settings, session) -> Dict[str, Any]:
    seg = settings.segments
    fingerprint: Dict[str, Any] = {}
    for table_id, table in sorted(settings.base["tables"].items()):
        fingerprint[f"{table_id}.rows"] = int(session.scalar(f"SELECT count(*) FROM {table['name']}"))
    seg_table = settings.table(seg.table)
    lo, hi = session.rows(
        f"SELECT min({seg.columns['ts']}), max({seg.columns['te']}) FROM {seg_table['name']} {seg_table['alias']}"
    )[0]
    fingerprint["segments.t_s.min"] = str(lo)
    fingerprint["segments.t_e.max"] = str(hi)
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
    print(json.dumps({
        "pools": str(output),
        "sha256": pools.sha256,
        "rows": {table: len(payload["rows"]) for table, payload in pools.tables.items()},
        "binding": pools.binding,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
