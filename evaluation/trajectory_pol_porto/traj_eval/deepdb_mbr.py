"""DeepDB-MBR: one row per trajectory (its bounding box plus trip and agent
attributes), learned by DeepDB, queried with box-overlap range predicates.

Two parts:

``export`` (pipeline environment, needs numpy and the P2 snapshot)
    writes ``trips_mbr.csv`` (no header) and ``layout.json`` into a shared
    directory. One row per trip, in snapshot order: ``trip_id``, every
    attribute of every non-segment table aligned to its trip (agents via the
    v3 join), temporal attributes as ``<column>_start`` / ``<column>_end``,
    and the box ``b_xmin, b_xmax, b_ymin, b_ymax, b_tmin, b_tmax``. Times are
    epoch seconds of the stored naive timestamps (``us / 1e6``, the same
    microseconds as ``NumpyBackend``).

:func:`translate` (standard library only, so the Python 3.8 DeepDB
environment imports it)
    turns one v3 query row into the conditions and SQL DeepDB parses. It is
    the question Exact-MBR answers exactly: trip and agent predicates on their
    columns, a segment window as closed box overlap
    (``b_xmin <= max_x AND b_xmax >= min_x AND ...``) and a segment interval
    as ``b_tmin <= upper AND b_tmax >= lower`` (missing bounds open), space
    and time tested separately. Answers that rely on the box carry
    ``approx:trajectory_box_overlap``. Unsupported, as for Exact-MBR:
    single-table queries and standard predicates on segment rows
    (``segment_idx``).

Column names are lower case and none is a prefix of another followed by an
operator, because DeepDB matches a condition to its column by substring.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TABLE = "trips_mbr"
ALIAS = "t"
CSV_NAME = "trips_mbr.csv"
LAYOUT_NAME = "layout.json"
LAYOUT_VERSION = 1
BOX_COLUMNS = ("b_xmin", "b_xmax", "b_ymin", "b_ymax", "b_tmin", "b_tmax")
APPROX_DIAGNOSTIC = "approx:trajectory_box_overlap"
SEGMENT_SPACE_TIME = ("spatial", "spatio_temporal", "temporal")
_EPOCH = datetime(1970, 1, 1)
_MICROSECOND = timedelta(microseconds=1)

Condition = Tuple[str, str, Any]  # (column, operator, value)


def epoch_seconds(text: str) -> float:
    """Seconds of a naive v3 timestamp, computed as ``microseconds / 1e6``."""
    value = datetime.fromisoformat(str(text))
    if value.tzinfo is not None:
        raise ValueError("v3 timestamps are naive")
    return ((value - _EPOCH) // _MICROSECOND) / 1e6


def _column_name(table_id: str, attribute: str, taken: Sequence[str]) -> str:
    base = f"{table_id[:1]}_{attribute}".lower()
    name = "".join(character if character.isalnum() or character == "_" else "_" for character in base)
    if name in taken or name in BOX_COLUMNS or name == "trip_id":
        raise ValueError(f"column name collision for {table_id}.{attribute}: {name}")
    return name


def build_layout(settings: Any) -> Dict[str, Any]:
    """Column layout of the trip table from the v3 settings (not from data)."""
    segments = settings.segments.table
    entity = settings.base["entity"]
    columns: List[Dict[str, Any]] = [{"name": "trip_id", "kind": "key"}]
    attributes: Dict[str, Dict[str, Any]] = {}
    taken: List[str] = []
    order = [entity["table"]] + [t for t in settings.base["tables"] if t not in (entity["table"], segments)]
    for table_id in order:
        for attribute in settings.table(table_id)["attributes"]:
            name, kind = attribute["name"], attribute["type"]
            column = _column_name(table_id, name, taken)
            if kind == "temporal_interval":
                entries = [(f"{column}_start", "start"), (f"{column}_end", "end")]
                attributes[f"{table_id}.{name}"] = {"kind": "temporal", "start": entries[0][0], "end": entries[1][0]}
                for entry, part in entries:
                    columns.append({"name": entry, "kind": "numeric", "table": table_id,
                                    "attribute": name, "part": part})
                    taken.append(entry)
            elif kind in ("numeric", "integer", "nominal"):
                attributes[f"{table_id}.{name}"] = {"kind": "nominal" if kind == "nominal" else "numeric",
                                                    "column": column}
                columns.append({"name": column, "kind": "nominal" if kind == "nominal" else "numeric",
                                "table": table_id, "attribute": name})
                taken.append(column)
            # geometry attributes do not exist outside the segment table
    for column in BOX_COLUMNS:
        columns.append({"name": column, "kind": "numeric", "table": segments, "attribute": "box"})
    names = [column["name"] for column in columns]
    for name in names:
        for other in names:
            if name != other and other.startswith(name) and other[len(name)] in "<=> ":
                raise ValueError(f"{name} is a DeepDB-ambiguous prefix of {other}")
    return {
        "layout_version": LAYOUT_VERSION,
        "table": TABLE,
        "columns": columns,
        "attributes": attributes,
        "segment_table": segments,
        "segment_temporal_attribute": settings.segments.temporal_attribute,
        "entity_table": entity["table"],
    }


def translate(row: Mapping[str, Any], layout: Mapping[str, Any]
              ) -> Tuple[Optional[List[Condition]], str]:
    """(conditions, diagnostic) or (None, unsupported reason) for one v3 query row."""
    relation = row["category"]["relation"] if "category" in row else row["relation"]
    if relation != "multi":
        return None, "single-table query (DeepDB-MBR models one row per trip)"
    segments = layout["segment_table"]
    conditions: List[Condition] = []
    surrogate = False
    for predicate in row["predicates"]:
        table, kind, attribute = predicate["table"], predicate["kind"], predicate["attribute"]
        if table == segments:
            if kind not in SEGMENT_SPACE_TIME or (
                    kind == "temporal" and attribute != layout["segment_temporal_attribute"]):
                return None, f"segment-level attribute predicate on {attribute}"
            surrogate = True
            if kind in ("spatial", "spatio_temporal"):
                conditions += [("b_xmin", "<=", float(predicate["max_x"])),
                               ("b_xmax", ">=", float(predicate["min_x"])),
                               ("b_ymin", "<=", float(predicate["max_y"])),
                               ("b_ymax", ">=", float(predicate["min_y"]))]
            if kind in ("temporal", "spatio_temporal"):
                if predicate.get("upper") is not None:
                    conditions.append(("b_tmin", "<=", epoch_seconds(predicate["upper"])))
                if predicate.get("lower") is not None:
                    conditions.append(("b_tmax", ">=", epoch_seconds(predicate["lower"])))
            continue
        entry = layout["attributes"].get(f"{table}.{attribute}")
        if entry is None:
            return None, f"attribute {table}.{attribute} is not in the trip table"
        if kind == "nominal_eq":
            value = str(predicate["value"])
            if "'" in value or '"' in value:
                return None, f"nominal value with a quote: {value!r}"
            conditions.append((entry["column"], "=", value))
        elif kind == "scalar_range":
            conditions += [(entry["column"], ">=", float(predicate["lower"])),
                           (entry["column"], "<=", float(predicate["upper"]))]
        elif kind == "scalar_unbounded":
            conditions.append((entry["column"], str(predicate["operator"]), float(predicate["value"])))
        elif kind == "temporal":
            if predicate.get("upper") is not None:
                conditions.append((entry["start"], "<=", epoch_seconds(predicate["upper"])))
            if predicate.get("lower") is not None:
                conditions.append((entry["end"], ">=", epoch_seconds(predicate["lower"])))
        else:
            return None, f"predicate kind {kind} on {table}.{attribute}"
    return conditions, APPROX_DIAGNOSTIC if surrogate else ""


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite bound {value}")
    # the shortest round-trip decimal, written without an exponent (sqlparse),
    # so DeepDB's float() gets back exactly this value
    return format(Decimal(repr(value)), "f")


def to_sql(conditions: Sequence[Condition]) -> str:
    """DeepDB-parsable ``SELECT COUNT(*)``; spaces around operators (negative literals)."""
    sql = f"SELECT COUNT(*) FROM {TABLE} {ALIAS}"
    if conditions:
        sql += " WHERE " + " AND ".join(f"{ALIAS}.{column} {operator} {_literal(value)}"
                                        for column, operator, value in conditions)
    return sql


def read_layout(directory: Path) -> Dict[str, Any]:
    layout = json.loads((Path(directory) / LAYOUT_NAME).read_text(encoding="utf-8"))
    if layout.get("layout_version") != LAYOUT_VERSION:
        raise ValueError(f"unsupported DeepDB-MBR layout version {layout.get('layout_version')}")
    return layout


# -- export (pipeline environment) -------------------------------------------

def trip_table(settings: Any, data: Mapping[str, Mapping[str, Any]], trips: Mapping[str, Any],
               layout: Mapping[str, Any]) -> Dict[str, Any]:
    """Column name -> numpy array, one entry per trip in snapshot order."""
    import numpy as np

    from .adapters.exact_mbr import align_trip_tables

    aligned = align_trip_tables(settings, data, trips)
    out: Dict[str, Any] = {"trip_id": np.asarray(trips["trip_id"], dtype=np.int64)}
    for column in layout["columns"]:
        name = column["name"]
        if column["kind"] == "key":
            continue
        if column.get("attribute") == "box":
            source = {"b_xmin": "xmin", "b_xmax": "xmax", "b_ymin": "ymin", "b_ymax": "ymax"}.get(name)
            if source:
                out[name] = np.asarray(trips[source], dtype=np.float64)
            else:
                out[name] = np.asarray(trips["tmin_us" if name == "b_tmin" else "tmax_us"], dtype=np.int64) / 1e6
            continue
        values = aligned[column["table"]]
        if "part" in column:
            out[name] = np.asarray(values[f"{column['attribute']}.{column['part']}"], dtype=np.int64) / 1e6
        else:
            out[name] = np.asarray(values[column["attribute"]])
    return out


def export(settings: Any, snapshot: Path, output: Path, *, config_path: Optional[Path] = None,
           log=print) -> Dict[str, Any]:
    """Write ``trips_mbr.csv`` + ``layout.json`` + ``export_manifest.json`` into ``output``."""
    import csv
    import hashlib

    from . import snapshot as snapshot_module

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    layout = build_layout(settings)
    data = snapshot_module.load_snapshot(snapshot, expected_config_sha256=settings.sha256)
    trips = snapshot_module.load_trip_index(snapshot)
    table = trip_table(settings, data, trips, layout)
    names = [column["name"] for column in layout["columns"]]
    rows = len(table["trip_id"])
    path = output / CSV_NAME
    temporary = path.with_suffix(".partial")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        block = 200_000
        for start in range(0, rows, block):
            stop = min(rows, start + block)
            columns = [table[name][start:stop].tolist() for name in names]
            lines = []
            for values in zip(*columns):
                lines.append([repr(value) if isinstance(value, float) else value for value in values])
            writer.writerows(lines)
        handle.flush()
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    temporary.replace(path)
    (output / LAYOUT_NAME).write_text(json.dumps(layout, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "rows": rows,
        "columns": names,
        "csv": str(path),
        "csv_sha256": digest.hexdigest(),
        "snapshot": str(snapshot),
        "snapshot_manifest_sha256": hashlib.sha256((Path(snapshot) / "manifest.json").read_bytes()).hexdigest(),
        "config_sha256": settings.sha256,
        "config": None if config_path is None else str(config_path),
        "nominal_values": {column["name"]: sorted({str(v) for v in table[column["name"]].tolist()})
                           for column in layout["columns"] if column["kind"] == "nominal"},
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"exported {rows} trips, {len(names)} columns -> {path}")
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from query_generation.benchmark_v3.config import load_settings

    parser = argparse.ArgumentParser(description="Export the DeepDB-MBR trip table from a P2 snapshot")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="v3 config JSON of the snapshot")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = export(load_settings(args.config), args.snapshot, args.output, config_path=args.config,
                      log=lambda message: print(message, flush=True))
    print(json.dumps({key: manifest[key] for key in ("rows", "csv_sha256", "snapshot_manifest_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
