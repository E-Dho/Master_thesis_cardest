"""Seeded, materialized center pools (SEMANTICS.md, section 5).

A pool is a fixed random sample of rows per table, built once per run by
``build_pools.py`` and shared by every generation task. Sampling centers from
the file instead of from ``ORDER BY random()`` makes every query reproducible
from (config, pools, seed) alone.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from query_generation.query_generator import parse_timestamp

from .config import Settings

POOLS_VERSION = 1


def pool_columns(settings: Settings, table_id: str) -> List[Tuple[str, str]]:
    """(column name, SQL expression) pairs stored in the pool of one table."""

    columns: List[Tuple[str, str]] = []
    seg = settings.segments
    if table_id == seg.table:
        columns.extend((key, seg.columns[key]) for key in ("sx", "sy", "ex", "ey", "ts", "te"))
    for attr in settings.table(table_id)["attributes"]:
        if attr["type"] in {"numeric", "integer"}:
            columns.append((attr["name"], attr["expression"]))
        elif attr["type"] == "temporal_interval" and table_id != seg.table:
            columns.append((f"{attr['name']}.start", attr["start_expression"]))
            columns.append((f"{attr['name']}.end", attr["end_expression"]))
    return columns


def encode_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="microseconds")
    if hasattr(value, "is_finite"):  # decimal.Decimal
        return float(value)
    return value


def subsample(rows: Sequence[Sequence[Any]], size: int, seed: int) -> List[List[Any]]:
    """Order-independent seeded subsample: sort canonically, then sample."""

    canonical = sorted(
        ([encode_value(value) for value in row] for row in rows),
        key=lambda row: json.dumps(row, default=str),
    )
    if len(canonical) <= size:
        return canonical
    return random.Random(seed).sample(canonical, size)


def payload_sha256(tables: Dict[str, Any]) -> str:
    text = json.dumps(tables, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Pools:
    def __init__(self, tables: Dict[str, Dict[str, Any]], binding: Optional[Dict[str, Any]] = None):
        self.tables = tables
        self.binding = binding or {}
        self.sha256 = payload_sha256(tables)
        self._cache: Dict[Tuple[str, str], List[Any]] = {}

    @classmethod
    def empty(cls) -> "Pools":
        return cls({})

    @classmethod
    def load(cls, path: Path) -> "Pools":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != POOLS_VERSION:
            raise ValueError(f"{path}: unsupported pools version")
        pools = cls(payload["tables"], payload.get("binding"))
        if payload.get("sha256") and payload["sha256"] != pools.sha256:
            raise ValueError(f"{path}: pool content does not match its stored sha256")
        return pools

    def save(self, path: Path) -> None:
        payload = {
            "version": POOLS_VERSION,
            "binding": self.binding,
            "sha256": self.sha256,
            "tables": self.tables,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def _column(self, table_id: str, name: str) -> List[Any]:
        key = (table_id, name)
        if key not in self._cache:
            table = self.tables.get(table_id)
            if not table or name not in table["columns"]:
                self._cache[key] = []
            else:
                index = table["columns"].index(name)
                self._cache[key] = [row[index] for row in table["rows"] if row[index] is not None]
        return self._cache[key]

    def scalar_values(self, table_id: str, attribute: str) -> List[float]:
        return self._column(table_id, attribute)

    def intervals(self, table_id: str, attribute: str, segment_table: str) -> List[Tuple[datetime, datetime]]:
        key = (table_id, f"{attribute}::interval")
        if key not in self._cache:
            if table_id == segment_table:
                starts, ends = self._column(table_id, "ts"), self._column(table_id, "te")
            else:
                starts = self._column(table_id, f"{attribute}.start")
                ends = self._column(table_id, f"{attribute}.end")
            self._cache[key] = [
                (parse_timestamp(start), parse_timestamp(end)) for start, end in zip(starts, ends)
            ]
        return self._cache[key]

    def segments(self, table_id: str) -> List[Tuple[float, float, float, float, datetime, datetime]]:
        key = (table_id, "::segments")
        if key not in self._cache:
            table = self.tables.get(table_id)
            rows: List[Any] = []
            if table:
                idx = [table["columns"].index(name) for name in ("sx", "sy", "ex", "ey", "ts", "te")]
                for row in table["rows"]:
                    sx, sy, ex, ey, ts, te = (row[i] for i in idx)
                    rows.append((float(sx), float(sy), float(ex), float(ey), parse_timestamp(ts), parse_timestamp(te)))
            self._cache[key] = rows
        return self._cache[key]
