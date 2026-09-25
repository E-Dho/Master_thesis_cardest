from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pg_cluster_lock  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the isolated PostgreSQL JOB-light database")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("initialize", "analyze"):
        command = subparsers.add_parser(name)
        command.add_argument("--pg-bin", required=True, type=Path)
        command.add_argument("--pgdata", required=True, type=Path)
        command.add_argument("--socket-dir", required=True, type=Path)
        command.add_argument("--port", required=True, type=int)
        command.add_argument("--database", required=True)
        command.add_argument("--manifest", required=True, type=Path)
    initialize = subparsers.choices["initialize"]
    initialize.add_argument("--csv-directory", required=True, type=Path)
    initialize.add_argument("--schema", required=True, type=Path)
    initialize.add_argument("--indexes", required=True, type=Path)
    initialize.add_argument("--row-limit", type=int)
    args = parser.parse_args()

    # The data directory is shared between nodes: hold the cluster-visible lock
    # while the server runs and stop the server this command started.
    args.pgdata.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pg_cluster_lock.held(args.pgdata, label=f"postgres_admin {args.command}"):
            started = _ensure_server(args)
            try:
                if args.command == "initialize":
                    _initialize(args)
                else:
                    _analyze(args)
            finally:
                if started:
                    _run(args.pg_bin / "pg_ctl", "-D", args.pgdata, "-m", "fast", "-w", "stop")
    except pg_cluster_lock.LockError as exc:
        print(f"cannot lock {args.pgdata}: {exc}", file=sys.stderr)
        return exc.exit_code
    return 0


def _ensure_server(args) -> bool:
    """Start the server if it is not running; True if this call started it."""
    args.pgdata.parent.mkdir(parents=True, exist_ok=True)
    args.socket_dir.mkdir(parents=True, exist_ok=True)
    if not (args.pgdata / "PG_VERSION").exists():
        _run(args.pg_bin / "initdb", "-D", args.pgdata, "--auth=trust", "--encoding=UTF8")
    status = subprocess.run(
        [str(args.pg_bin / "pg_ctl"), "-D", str(args.pgdata), "status"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if status.returncode:
        _run(
            args.pg_bin / "pg_ctl",
            "-D",
            args.pgdata,
            "-o",
            f"-k {args.socket_dir} -p {args.port} -c listen_addresses=''",
            "-w",
            "start",
        )
        return True
    return False


def _initialize(args) -> None:
    if args.manifest.exists():
        print(f"reuse initialized database from {args.manifest}")
        return
    psycopg, sql = _psycopg()
    admin_dsn = f"dbname=postgres host={args.socket_dir} port={args.port}"
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (args.database,)
        ).fetchone()
        if exists is None:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(args.database)))
    dsn = f"dbname={args.database} host={args.socket_dir} port={args.port}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        table_count = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
        ).fetchone()[0]
        if table_count == 0:
            _run(
                args.pg_bin / "psql",
                "-h",
                args.socket_dir,
                "-p",
                args.port,
                "-d",
                args.database,
                "-v",
                "ON_ERROR_STOP=1",
                "-f",
                args.schema,
            )
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
            )
        ]
        for table in tables:
            csv_path = args.csv_directory / f"{table}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(csv_path)
            row_count = connection.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table))
            ).fetchone()[0]
            if row_count:
                continue
            statement = sql.SQL(
                "COPY {} FROM STDIN WITH "
                "(FORMAT CSV, HEADER TRUE, NULL '', ESCAPE E'\\\\')"
            ).format(sql.Identifier(table))
            with connection.cursor().copy(statement) as copy:
                if args.row_limit is None:
                    with csv_path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            copy.write(block)
                else:
                    if args.row_limit <= 0:
                        raise ValueError("--row-limit must be positive")
                    with csv_path.open(newline="", encoding="utf-8") as source:
                        reader = csv.reader(source, escapechar="\\")
                        next(reader)
                        for _ in zip(range(args.row_limit), reader):
                            pass
                        physical_line_count = reader.line_num
                    with csv_path.open(newline="", encoding="utf-8") as source:
                        payload = "".join(islice(source, physical_line_count))
                    copy.write(payload.encode("utf-8"))
        _run(
            args.pg_bin / "psql",
            "-h",
            args.socket_dir,
            "-p",
            args.port,
            "-d",
            args.database,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            args.indexes,
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "database": args.database,
                "csv_directory": str(args.csv_directory.resolve()),
                "schema": str(args.schema.resolve()),
                "indexes": str(args.indexes.resolve()),
                "tables": tables,
                "row_limit": args.row_limit,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _analyze(args) -> None:
    psycopg, _ = _psycopg()
    dsn = f"dbname={args.database} host={args.socket_dir} port={args.port}"
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("SET default_statistics_target = 100")
        connection.execute("ANALYZE")
        extended = connection.execute("SELECT COUNT(*) FROM pg_statistic_ext").fetchone()[0]
        if extended != 0:
            raise ValueError("PostgreSQL baseline must not contain extended statistics")
        statistics_bytes = connection.execute(
            "SELECT pg_total_relation_size('pg_catalog.pg_statistic')"
        ).fetchone()[0]
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "default_statistics_target": 100,
            "extended_statistics_count": 0,
            "statistics_storage_bytes": int(statistics_bytes),
        }
    )
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run(executable: Path, *arguments) -> None:
    subprocess.run([str(executable), *(str(value) for value in arguments)], check=True)


def _psycopg():
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError("postgres_admin requires psycopg>=3") from exc
    return psycopg, sql


if __name__ == "__main__":
    raise SystemExit(main())
