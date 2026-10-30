"""Shared data snapshot: the benchmark database exported once into numpy arrays.

Every baseline that builds from data (sampling, Exact-MBR, CASE+, TraHist,
DeepDB-MBR, ...) reads this snapshot instead of the database, so all of them
see byte-identical data, and the export cost is paid once outside every
method's build timer.

Layout of a snapshot directory::

    manifest.json
    tables/<table_id>/<column>.npy      one array per exported column
    derived/trips/<column>.npy          per-trip index and bounding boxes

``tables`` holds exactly the arrays ``query_generation.benchmark_v3.
numpy_backend.NumpyBackend`` consumes: per table the join keys and every
scalar, nominal and temporal attribute of the v3 config; temporal attributes
as ``<name>.start`` / ``<name>.end`` in int64 microseconds since 1970-01-01 of
the stored ``timestamp without time zone`` value (no time-zone conversion,
v3 SEMANTICS.md section 1); the segments table additionally ``sx, sy, ex, ey``
(float64) and ``ts, te`` (int64 microseconds). The segment temporal attribute
(``segment_time.start/.end``) is the same data as ``ts/te`` and is aliased
when loading, not stored twice. Rows are ordered by each table's primary key.

``derived/trips`` (segments must be sorted by trip, which the export
guarantees): ``trip_id``, ``offset`` (CSR: the segments of trip ``i`` are rows
``offset[i]:offset[i+1]``), ``segment_count``, the spatial bounding box
``xmin, ymin, xmax, ymax`` over both endpoints, and ``tmin_us, tmax_us``.

NULLs: a nullable integer or numeric column is stored as float64 with NaN; a
nullable nominal column stores :data:`NULL_TEXT`. Both never satisfy a v3
predicate, exactly like SQL NULL. The manifest records NULL counts.

The export first counts NULLs per column in SQL. Tables without text columns
or NULLs (the 50M-row segments table) are streamed with ``COPY ... (FORMAT
binary)`` and decoded directly into memory-mapped arrays; all others go
through a server-side cursor. Should a NULL still reach the binary decoder, it
is recognised at the first irregular row and the table falls back to the
cursor.

CLI::

    python -m evaluation.trajectory_pol_porto.traj_eval.snapshot export \\
        --config query_generation/benchmark_v3/pol_v3_config.json \\
        --output /work_beegfs/.../traj_snapshots/pol_v3 --host SOCKET --port 55432 --dbname pol_mobilitydb
    python -m evaluation.trajectory_pol_porto.traj_eval.snapshot verify --snapshot DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import struct
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

SNAPSHOT_VERSION = "traj_snapshot_v1"
NULL_TEXT = "\u0000NULL"
BINARY_SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
FETCH_BATCH = 200_000


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    sql: str
    #: "int64", "float64" or "text"
    kind: str


def _epoch_us(expression: str) -> str:
    # EXTRACT(EPOCH FROM timestamp without time zone) is the nominal number of
    # seconds since 1970-01-01, independent of the server's time zone.
    return f"(EXTRACT(EPOCH FROM {expression}) * 1000000)::bigint"


def table_specs(settings: Any) -> Dict[str, List[ColumnSpec]]:
    """Exported columns per table, derived from the v3 settings alone."""

    from query_generation.benchmark_v3.numpy_backend import JOIN_CONDITION

    alias_to_table = {table["alias"]: table_id for table_id, table in settings.base["tables"].items()}
    join_columns: Dict[str, set] = defaultdict(set)
    for join in settings.base["joins"]:
        left_alias, left_col, right_alias, right_col = JOIN_CONDITION.match(join["condition"]).groups()
        join_columns[alias_to_table[left_alias]].add(left_col)
        join_columns[alias_to_table[right_alias]].add(right_col)
    segments = settings.segments
    specs: Dict[str, List[ColumnSpec]] = {}
    for table_id, table in settings.base["tables"].items():
        alias = table["alias"]
        columns: Dict[str, ColumnSpec] = {}
        for column in sorted(join_columns[table_id]):
            columns[column] = ColumnSpec(column, f"{alias}.{column}::bigint", "int64")
        entity = settings.base["entity"]
        if table_id == entity["table"]:
            columns[entity["key"]] = ColumnSpec(entity["key"], f"{entity['expression']}::bigint", "int64")
        for attribute in table["attributes"]:
            kind = attribute["type"]
            name = attribute["name"]
            if kind == "integer":
                columns[name] = ColumnSpec(name, f"({attribute['expression']})::bigint", "int64")
            elif kind == "numeric":
                columns[name] = ColumnSpec(name, f"({attribute['expression']})::float8", "float64")
            elif kind == "nominal":
                columns[name] = ColumnSpec(name, f"({attribute['expression']})::text", "text")
            elif kind == "temporal_interval":
                if table_id == segments.table:
                    continue  # stored as ts/te, aliased on load
                columns[f"{name}.start"] = ColumnSpec(f"{name}.start", _epoch_us(attribute["start_expression"]), "int64")
                columns[f"{name}.end"] = ColumnSpec(f"{name}.end", _epoch_us(attribute["end_expression"]), "int64")
            elif kind == "geometry":
                continue  # segments: exported as raw endpoints below
            else:
                raise ValueError(f"unsupported attribute type {kind!r} for {table_id}.{name}")
        if table_id == segments.table:
            for key in ("sx", "sy", "ex", "ey"):
                columns[key] = ColumnSpec(key, f"({segments.columns[key]})::float8", "float64")
            for key in ("ts", "te"):
                columns[key] = ColumnSpec(key, _epoch_us(segments.columns[key]), "int64")
        specs[table_id] = list(columns.values())
    return specs


def order_by(settings: Any, table_id: str) -> str:
    table = settings.base["tables"][table_id]
    keys = [key.strip() for key in str(table["primary_key"]).split(",") if key.strip()]
    return ", ".join(f"{table['alias']}.{key}" for key in keys)


def export_sql(settings: Any, table_id: str, specs: List[ColumnSpec]) -> str:
    table = settings.base["tables"][table_id]
    return (
        f"SELECT {', '.join(spec.sql for spec in specs)} FROM {table['name']} {table['alias']} "
        f"ORDER BY {order_by(settings, table_id)}"
    )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class _NeedsCursor(Exception):
    """A binary COPY met a NULL or an unexpected field width."""


def _column_path(root: Path, table_id: str, column: str) -> Path:
    return root / "tables" / table_id / f"{column}.npy"


BLOCK_BYTES = 32 << 20


def _copy_binary(connection, sql: str, specs: List[ColumnSpec], rows: int, root: Path, table_id: str) -> None:
    """Stream ``COPY (sql) TO STDOUT (FORMAT binary)`` into memory-mapped arrays.

    psycopg yields one chunk per row, so chunks are collected into blocks of
    about :data:`BLOCK_BYTES` and each block is decoded with one
    ``np.frombuffer`` (fixed row layout: every column is an 8-byte value).
    """

    fields = len(specs)
    row_dtype = np.dtype(
        [("fields", ">i2")]
        + [item for i, spec in enumerate(specs)
           for item in ((f"l{i}", ">i4"), (f"v{i}", ">i8" if spec.kind == "int64" else ">f8"))]
    )
    targets = []
    for spec in specs:
        path = _column_path(root, table_id, spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        targets.append(np.lib.format.open_memmap(
            path, mode="w+", dtype=np.int64 if spec.kind == "int64" else np.float64, shape=(rows,)))
    state = {"written": 0, "header": False, "carry": b""}

    def consume(data: bytes) -> None:
        if not state["header"]:
            if len(data) < 19:
                state["carry"] = data
                return
            if data[:11] != BINARY_SIGNATURE:
                raise ValueError("unexpected binary COPY signature")
            extension = struct.unpack(">i", data[15:19])[0]
            if len(data) < 19 + extension:
                state["carry"] = data
                return
            data = data[19 + extension:]
            state["header"] = True
        complete = len(data) // row_dtype.itemsize
        if complete:
            block = np.frombuffer(data, dtype=row_dtype, count=complete)
            # A NULL (length -1, no value bytes) or a value of another width
            # shortens its row, so every later row of the block is misaligned
            # and its field count is garbage. Only the first irregular row is
            # read at a correct offset: classify that one. (The 2-byte trailer
            # is shorter than a row and never part of a decoded block.)
            regular = block["fields"] == fields
            for i in range(fields):
                regular &= block[f"l{i}"] == 8
            if not regular.all():
                first = int(np.argmin(regular))
                row = block[first]
                lengths = [int(row[f"l{i}"]) for i in range(fields)]
                if int(row["fields"]) == fields and any(length != 8 for length in lengths):
                    column = specs[next(i for i, length in enumerate(lengths) if length != 8)].name
                    kind = "a NULL" if -1 in lengths else "a value that is not 8 bytes wide"
                    raise _NeedsCursor(f"{table_id}.{column} has {kind} (row {state['written'] + first})")
                raise ValueError(f"{table_id}: unexpected field count in binary COPY "
                                 f"(row {state['written'] + first})")
            written = state["written"]
            if written + complete > rows:
                raise ValueError(f"{table_id}: more rows than counted ({rows})")
            for i, target in enumerate(targets):
                target[written: written + complete] = block[f"v{i}"]
            state["written"] = written + complete
        state["carry"] = data[complete * row_dtype.itemsize:]

    pending: List[bytes] = []
    pending_bytes = 0
    with connection.cursor() as cursor:
        with cursor.copy(f"COPY ({sql}) TO STDOUT (FORMAT binary)") as copy:
            for chunk in copy:
                pending.append(bytes(chunk))
                pending_bytes += len(chunk)
                if pending_bytes >= BLOCK_BYTES:
                    consume(state["carry"] + b"".join(pending))
                    pending, pending_bytes = [], 0
    consume(state["carry"] + b"".join(pending))
    if state["carry"] != b"\xff\xff":
        _classify_tail(state["carry"], specs, table_id, state["written"])
    if state["written"] != rows:
        raise ValueError(f"{table_id}: exported {state['written']} rows, counted {rows}")
    for target in targets:
        target.flush()
    del targets


def _classify_tail(tail: bytes, specs: List[ColumnSpec], table_id: str, written: int) -> None:
    """Explain bytes left before the trailer; always raises.

    A row shortened by a NULL in the last rows is never long enough to be
    decoded as a fixed-width row, so it stays here: parse it field by field.
    """
    position, row = 0, written
    while position + 2 <= len(tail):
        (count,) = struct.unpack_from(">h", tail, position)
        position += 2
        if count == -1:
            break
        if count != len(specs):
            break
        for index in range(count):
            if position + 4 > len(tail):
                break
            (length,) = struct.unpack_from(">i", tail, position)
            position += 4
            if length != 8:
                kind = "a NULL" if length == -1 else "a value that is not 8 bytes wide"
                raise _NeedsCursor(f"{table_id}.{specs[index].name} has {kind} (row {row})")
            position += 8
        row += 1
    raise ValueError(f"{table_id}: expected the COPY trailer, got {tail[:16]!r}")


def _fetch_with_cursor(connection, sql: str, specs: List[ColumnSpec], rows: int, root: Path,
                       table_id: str) -> Dict[str, int]:
    """Server-side cursor export for tables with text columns or NULLs."""

    values: List[List[Any]] = [[] for _ in specs]
    with connection.cursor(name=f"snapshot_{table_id}") as cursor:
        cursor.itersize = FETCH_BATCH
        cursor.execute(sql)
        while True:
            batch = cursor.fetchmany(FETCH_BATCH)
            if not batch:
                break
            for row in batch:
                for i, value in enumerate(row):
                    values[i].append(value)
    if len(values[0]) != rows:
        raise ValueError(f"{table_id}: exported {len(values[0])} rows, counted {rows}")
    nulls: Dict[str, int] = {}
    for spec, column in zip(specs, values):
        null_count = sum(value is None for value in column)
        nulls[spec.name] = null_count
        if spec.kind == "text":
            array = np.asarray([NULL_TEXT if value is None else str(value) for value in column], dtype=str)
        elif null_count:
            array = np.asarray([np.nan if value is None else float(value) for value in column], dtype=np.float64)
            if spec.kind == "int64" and any(
                value is not None and float(value) != int(value) for value in column
            ):
                raise ValueError(f"{table_id}.{spec.name}: non-integer values")
        else:
            array = np.asarray(column, dtype=np.int64 if spec.kind == "int64" else np.float64)
        path = _column_path(root, table_id, spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array, allow_pickle=False)
    return nulls


def _null_counts(connection, settings: Any, table_id: str, specs: List[ColumnSpec]) -> Dict[str, int]:
    """NULLs per exported column, counted by the server before choosing the export path."""

    table = settings.base["tables"][table_id]
    counts = ", ".join(f"count(*) FILTER (WHERE ({spec.sql}) IS NULL)" for spec in specs)
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {counts} FROM {table['name']} {table['alias']}")
        row = cursor.fetchone()
    return {spec.name: int(value) for spec, value in zip(specs, row)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(connection, sql: str) -> Any:
    with connection.cursor() as cursor:
        cursor.execute(sql)
        row = cursor.fetchone()
        return None if row is None else row[0]


def _database_identity(connection) -> Dict[str, Any]:
    identity = {"server_version": _scalar(connection, "SHOW server_version"),
                "database": _scalar(connection, "SELECT current_database()"),
                "timezone": _scalar(connection, "SHOW TimeZone")}
    with connection.cursor() as cursor:
        cursor.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
        identity["extensions"] = {name: version for name, version in cursor.fetchall()}
    return identity


def derive_trips(settings: Any, root: Path) -> Dict[str, Any]:
    """Per-trip CSR index and bounding boxes from the (trip-sorted) segments."""

    segments = settings.segments
    seg_dir = root / "tables" / segments.table
    trip_key = segments.trip_key.split(".")[-1]
    trip_ids = np.load(seg_dir / f"{trip_key}.npy", mmap_mode="r")
    if trip_ids.size == 0:
        raise ValueError("no segments exported")
    if np.any(np.diff(trip_ids) < 0):
        raise ValueError("segments are not sorted by trip")
    starts = np.flatnonzero(np.r_[True, np.diff(trip_ids) != 0])
    offsets = np.r_[starts, trip_ids.size].astype(np.int64)
    out = root / "derived" / "trips"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "trip_id.npy", np.asarray(trip_ids[starts], dtype=np.int64))
    np.save(out / "offset.npy", offsets)
    np.save(out / "segment_count.npy", np.diff(offsets))
    for name, first, second, reducer in (
        ("xmin", "sx", "ex", np.minimum), ("ymin", "sy", "ey", np.minimum),
        ("xmax", "sx", "ex", np.maximum), ("ymax", "sy", "ey", np.maximum),
    ):
        both = reducer(np.load(seg_dir / f"{first}.npy", mmap_mode="r"), np.load(seg_dir / f"{second}.npy", mmap_mode="r"))
        np.save(out / f"{name}.npy", reducer.reduceat(both, starts))
        del both
    np.save(out / "tmin_us.npy", np.minimum.reduceat(np.load(seg_dir / "ts.npy", mmap_mode="r"), starts))
    np.save(out / "tmax_us.npy", np.maximum.reduceat(np.load(seg_dir / "te.npy", mmap_mode="r"), starts))
    return {"trips_with_segments": int(starts.size)}


def consistency_checks(settings: Any, root: Path) -> Dict[str, Any]:
    """Facts the baselines rely on; any failure stops the export."""

    segments = settings.segments
    entity = settings.base["entity"]
    seg_dir = root / "tables" / segments.table
    trips_dir = root / "derived" / "trips"
    checks: Dict[str, Any] = {}
    ts, te = np.load(seg_dir / "ts.npy", mmap_mode="r"), np.load(seg_dir / "te.npy", mmap_mode="r")
    checks["segments_with_ts_after_te"] = int(np.count_nonzero(ts > te))
    trip_ids = np.load(seg_dir / f"{segments.trip_key.split('.')[-1]}.npy", mmap_mode="r")
    index = [k.strip() for k in str(settings.base["tables"][segments.table]["primary_key"]).split(",")]
    if len(index) == 2:
        second = np.load(seg_dir / f"{index[1]}.npy", mmap_mode="r")
        same_trip = np.diff(trip_ids) == 0
        checks["segments_not_strictly_ordered_within_trip"] = int(np.count_nonzero(same_trip & (np.diff(second) <= 0)))
    entity_ids = np.load(root / "tables" / entity["table"] / f"{entity['key']}.npy")
    derived_ids = np.load(trips_dir / "trip_id.npy")
    checks["entity_rows"] = int(entity_ids.size)
    checks["entity_rows_without_segments"] = int(np.count_nonzero(~np.isin(entity_ids, derived_ids)))
    checks["segment_trips_missing_in_entity_table"] = int(np.count_nonzero(~np.isin(derived_ids, entity_ids)))
    count_column = root / "tables" / entity["table"] / "num_of_segments.npy"
    if count_column.exists():
        declared = dict(zip(entity_ids.tolist(), np.load(count_column).tolist()))
        counts = np.load(trips_dir / "segment_count.npy")
        checks["num_of_segments_mismatches"] = int(sum(
            declared.get(trip, -1) != count for trip, count in zip(derived_ids.tolist(), counts.tolist())
        ))
    failed = {key: value for key, value in checks.items()
              if key not in ("entity_rows",) and value}
    checks["passed"] = not failed
    if failed:
        raise ValueError(f"snapshot consistency checks failed: {failed}")
    return checks


def export_snapshot(
    settings: Any,
    connect: Callable[[], Any],
    output: Path,
    *,
    config_path: Optional[Path] = None,
    log: Callable[[str], None] = print,
    force: bool = False,
) -> Dict[str, Any]:
    output = Path(output)
    if (output / "manifest.json").exists() and not force:
        raise FileExistsError(f"{output} already holds a snapshot; pass force to replace it")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    started = time.time()
    specs = table_specs(settings)
    tables: Dict[str, Any] = {}
    with connect() as connection:
        connection.autocommit = False  # one snapshot transaction for all tables
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        database = _database_identity(connection)
        for table_id, columns in specs.items():
            table = settings.base["tables"][table_id]
            sql = export_sql(settings, table_id, columns)
            rows = int(_scalar(connection, f"SELECT COUNT(*) FROM {table['name']}"))
            table_started = time.time()
            nulls = _null_counts(connection, settings, table_id, columns)
            method = "binary_copy"
            if any(spec.kind == "text" for spec in columns) or any(nulls.values()):
                method = "cursor"
                nulls = _fetch_with_cursor(connection, sql, columns, rows, partial, table_id)
            else:
                try:
                    _copy_binary(connection, sql, columns, rows, partial, table_id)
                except _NeedsCursor as reason:
                    log(f"{table_id}: {reason}; exporting with a cursor instead")
                    connection.rollback()
                    with connection.cursor() as cursor:
                        cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                    method = "cursor"
                    nulls = _fetch_with_cursor(connection, sql, columns, rows, partial, table_id)
            tables[table_id] = {
                "rows": rows,
                "sql": sql,
                "method": method,
                "seconds": round(time.time() - table_started, 3),
                "columns": {spec.name: {"kind": spec.kind, "sql": spec.sql, "null_count": nulls.get(spec.name, 0)}
                            for spec in columns},
            }
            log(f"{table_id}: {rows} rows via {method} in {tables[table_id]['seconds']} s")
        connection.rollback()
    derived = derive_trips(settings, partial)
    checks = consistency_checks(settings, partial)
    files = {}
    for path in sorted(partial.rglob("*.npy")):
        array = np.load(path, mmap_mode="r")
        files[str(path.relative_to(partial))] = {
            "sha256": _sha256(path), "bytes": path.stat().st_size,
            "dtype": str(array.dtype), "shape": list(array.shape),
        }
    segments = settings.segments
    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "semantics_version": settings.semantics_version,
        "config_sha256": settings.sha256,
        "config_path": str(config_path) if config_path else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_seconds": round(time.time() - started, 3),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "database": database,
        "tables": tables,
        "aliases": {segments.table: {f"{segments.temporal_attribute}.start": "ts",
                                     f"{segments.temporal_attribute}.end": "te"}},
        "derived": derived,
        "checks": checks,
        "files": files,
        "null_text": NULL_TEXT,
        "time_unit": "int64 microseconds since 1970-01-01 of the stored timestamp without time zone",
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)
    log(f"snapshot written to {output}")
    return manifest


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def read_manifest(root: Path) -> Dict[str, Any]:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))


def verify_snapshot(root: Path) -> List[str]:
    """Checksum every file against the manifest; returns the problems found."""

    root = Path(root)
    manifest = read_manifest(root)
    problems = []
    for relative, meta in manifest["files"].items():
        path = root / relative
        if not path.exists():
            problems.append(f"missing {relative}")
        elif _sha256(path) != meta["sha256"]:
            problems.append(f"checksum mismatch {relative}")
    return problems


def load_snapshot(root: Path, *, mmap_mode: Optional[str] = "r",
                  expected_config_sha256: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """The ``NumpyBackend`` data dict of a snapshot (arrays memory-mapped by default)."""

    root = Path(root)
    manifest = read_manifest(root)
    if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported snapshot version {manifest.get('snapshot_version')!r}")
    if expected_config_sha256 is not None and manifest["config_sha256"] != expected_config_sha256:
        raise ValueError("snapshot was exported under a different v3 config")
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for table_id, table in manifest["tables"].items():
        data[table_id] = {
            name: np.load(_column_path(root, table_id, name), mmap_mode=mmap_mode, allow_pickle=False)
            for name in table["columns"]
        }
    for table_id, aliases in manifest.get("aliases", {}).items():
        for alias, source in aliases.items():
            data[table_id][alias] = data[table_id][source]
    return data


def load_trip_index(root: Path, *, mmap_mode: Optional[str] = "r") -> Dict[str, np.ndarray]:
    directory = Path(root) / "derived" / "trips"
    return {path.stem: np.load(path, mmap_mode=mmap_mode) for path in sorted(directory.glob("*.npy"))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _connect_factory(args: argparse.Namespace) -> Callable[[], Any]:
    import psycopg

    kwargs = {"host": args.host, "port": args.port, "dbname": args.dbname}
    if args.user:
        kwargs["user"] = args.user
    return lambda: psycopg.connect(**kwargs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export or verify a trajectory data snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--config", required=True, type=Path, help="v3 benchmark config JSON")
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--host", required=True)
    export.add_argument("--port", type=int, required=True)
    export.add_argument("--dbname", required=True)
    export.add_argument("--user")
    export.add_argument("--force", action="store_true", help="replace an existing snapshot")
    check = sub.add_parser("verify")
    check.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        from query_generation.benchmark_v3.config import load_settings

        settings = load_settings(args.config)
        export_snapshot(settings, _connect_factory(args), args.output, config_path=args.config.resolve(),
                        log=lambda message: print(message, flush=True), force=args.force)
        return 0
    problems = verify_snapshot(args.snapshot)
    for problem in problems:
        print(problem)
    print("snapshot ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
