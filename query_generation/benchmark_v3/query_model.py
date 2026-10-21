"""Structured v3 predicates and their SQL rendering (SEMANTICS.md, sections 2-3).

A query record stores every predicate as structured data (``kind`` plus its
bounds). The SQL is derived from that data here; the numpy backend evaluates
the same data with ``exact.py``. Nothing is ever parsed back out of SQL.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from query_generation.query_generator import ConfigError, parse_timestamp, sql_literal

from .config import Settings

STRUCTURAL_KEYS = (
    "kind",
    "table",
    "attribute",
    "value",
    "operator",
    "lower",
    "upper",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
)
SCALAR_OPERATORS = ("<", "<=", ">", ">=")


# -- literals ------------------------------------------------------------------

def timestamp_text(value: datetime) -> str:
    """Canonical text form with full microsecond precision."""

    if value.tzinfo is not None:
        raise ValueError("v3 timestamps are naive")
    return value.isoformat(sep=" ", timespec="microseconds")


def ts_sql(text: str) -> str:
    return f"timestamp '{timestamp_text(parse_timestamp(text))}'"


def float_sql(value: float) -> str:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"invalid float literal {value}")
    return f"{value!r}::float8"


def scalar_sql(value: Any, attr_type: str) -> str:
    if attr_type == "integer":
        return str(int(value))
    if attr_type == "numeric":
        return float_sql(value)
    return sql_literal(value)


# -- spatial / spatio-temporal SQL ---------------------------------------------------

def _axis_sql(s: str, e: str, a0: float, a1: float) -> Tuple[str, str]:
    lo_bound, hi_bound = float_sql(a0), float_sql(a1)
    inside = f"{s} >= {lo_bound} AND {s} <= {hi_bound}"
    first = f"({lo_bound} - {s}) / ({e} - {s})"
    second = f"({hi_bound} - {s}) / ({e} - {s})"
    lo = (
        f"CASE WHEN {e} = {s} THEN (CASE WHEN {inside} THEN 0.0::float8 ELSE 2.0::float8 END) "
        f"ELSE LEAST({first}, {second}) END"
    )
    hi = (
        f"CASE WHEN {e} = {s} THEN (CASE WHEN {inside} THEN 1.0::float8 ELSE -1.0::float8 END) "
        f"ELSE GREATEST({first}, {second}) END"
    )
    return lo, hi


def clip_sql(settings: Settings, window: Tuple[float, float, float, float]) -> Tuple[str, str]:
    """SQL for u0 and u1 (exact.clip_parameters)."""

    cols = settings.segments.columns
    x0, y0, x1, y1 = window
    lo_x, hi_x = _axis_sql(cols["sx"], cols["ex"], x0, x1)
    lo_y, hi_y = _axis_sql(cols["sy"], cols["ey"], y0, y1)
    u0 = f"GREATEST(0.0::float8, {lo_x}, {lo_y})"
    u1 = f"LEAST(1.0::float8, {hi_x}, {hi_y})"
    return u0, u1


def envelope_sql(settings: Settings, window: Tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = window
    return (
        f"ST_MakeEnvelope({float(x0)!r}, {float(y0)!r}, {float(x1)!r}, {float(y1)!r}, "
        f"{settings.srid})"
    )


def _time_parameter_sql(settings: Settings, bound: str) -> str:
    cols = settings.segments.columns
    return (
        f"(EXTRACT(EPOCH FROM ({ts_sql(bound)} - {cols['ts']}))::float8 / "
        f"EXTRACT(EPOCH FROM ({cols['te']} - {cols['ts']}))::float8)"
    )


def window_of(predicate: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(predicate["min_x"]),
        float(predicate["min_y"]),
        float(predicate["max_x"]),
        float(predicate["max_y"]),
    )


def check_bounds(predicate: Dict[str, Any]) -> None:
    """Windows and intervals must be non-empty; the clipping SQL assumes it."""

    if "min_x" in predicate:
        x0, y0, x1, y1 = window_of(predicate)
        if x0 > x1 or y0 > y1:
            raise ConfigError(f"empty spatial window {predicate}")
    lower, upper = predicate.get("lower"), predicate.get("upper")
    if predicate["kind"] in {"temporal", "spatio_temporal"} and lower is not None and upper is not None:
        if parse_timestamp(lower) > parse_timestamp(upper):
            raise ConfigError(f"empty temporal interval {lower} > {upper}")
    if predicate["kind"] == "scalar_range" and float(predicate["lower"]) > float(predicate["upper"]):
        raise ConfigError(f"empty range {predicate['lower']} > {predicate['upper']}")


def spatial_sql(settings: Settings, predicate: Dict[str, Any]) -> str:
    window = window_of(predicate)
    u0, u1 = clip_sql(settings, window)
    geometry = settings.segments.geometry_expression
    return f"{geometry} && {envelope_sql(settings, window)} AND {u0} <= {u1}"


def spatio_temporal_sql(settings: Settings, predicate: Dict[str, Any]) -> str:
    cols = settings.segments.columns
    window = window_of(predicate)
    lower, upper = predicate.get("lower"), predicate.get("upper")
    u0, u1 = clip_sql(settings, window)
    geometry = settings.segments.geometry_expression
    parts = [f"{geometry} && {envelope_sql(settings, window)}"]
    if upper is not None:
        parts.append(f"{cols['ts']} <= {ts_sql(upper)}")
    if lower is not None:
        parts.append(f"{cols['te']} >= {ts_sql(lower)}")
    parts.append(f"{u0} <= {u1}")
    timed = []
    if upper is not None:
        timed.append(f"{u0} <= {_time_parameter_sql(settings, upper)}")
    if lower is not None:
        timed.append(f"{u1} >= {_time_parameter_sql(settings, lower)}")
    if timed:
        parts.append(f"({cols['te']} = {cols['ts']} OR ({' AND '.join(timed)}))")
    return " AND ".join(parts)


# -- predicate SQL -----------------------------------------------------------------------

def predicate_sql(settings: Settings, predicate: Dict[str, Any]) -> str:
    check_bounds(predicate)
    kind = predicate["kind"]
    table, attribute = predicate["table"], predicate["attribute"]
    attr = settings.attr(table, attribute)
    if kind == "nominal_eq":
        return f"{attr['expression']} = {sql_literal(predicate['value'])}"
    if kind == "scalar_range":
        expr = attr["expression"]
        return (
            f"{expr} >= {scalar_sql(predicate['lower'], attr['type'])} AND "
            f"{expr} <= {scalar_sql(predicate['upper'], attr['type'])}"
        )
    if kind == "scalar_unbounded":
        if predicate["operator"] not in SCALAR_OPERATORS:
            raise ConfigError(f"invalid operator {predicate['operator']!r}")
        return f"{attr['expression']} {predicate['operator']} {scalar_sql(predicate['value'], attr['type'])}"
    if kind == "temporal":
        parts = []
        if predicate.get("upper") is not None:
            parts.append(f"{attr['start_expression']} <= {ts_sql(predicate['upper'])}")
        if predicate.get("lower") is not None:
            parts.append(f"{attr['end_expression']} >= {ts_sql(predicate['lower'])}")
        if not parts:
            raise ConfigError("temporal predicate needs at least one bound")
        return " AND ".join(parts)
    if kind == "spatial":
        _require_segments(settings, predicate)
        return spatial_sql(settings, predicate)
    if kind == "spatio_temporal":
        _require_segments(settings, predicate)
        if predicate.get("lower") is None and predicate.get("upper") is None:
            raise ConfigError("spatio-temporal predicate needs at least one time bound")
        return spatio_temporal_sql(settings, predicate)
    raise ConfigError(f"unknown predicate kind {kind!r}")


def _require_segments(settings: Settings, predicate: Dict[str, Any]) -> None:
    if predicate["table"] != settings.segments.table:
        raise ConfigError(f"{predicate['kind']} predicates are only allowed on {settings.segments.table}")


# -- queries --------------------------------------------------------------------------------

def from_clause(settings: Settings, table_ids: Sequence[str]) -> Tuple[str, List[Dict[str, str]]]:
    tables = list(table_ids)
    first = tables[0]
    first_table = settings.table(first)
    sql = f"FROM {first_table['name']} {first_table['alias']}"
    included = {first}
    remaining = set(tables[1:])
    joins_used: List[Dict[str, str]] = []
    while remaining:
        for join in settings.base["joins"]:
            left, right = join["left"], join["right"]
            new = right if left in included and right in remaining else (
                left if right in included and left in remaining else None
            )
            if new is None:
                continue
            table = settings.table(new)
            sql += f"\nJOIN {table['name']} {table['alias']} ON {join['condition']}"
            joins_used.append(dict(join))
            included.add(new)
            remaining.remove(new)
            break
        else:
            raise ConfigError(f"cannot connect tables {tables}")
    return sql, joins_used


def where_clause(settings: Settings, predicates: Iterable[Dict[str, Any]]) -> str:
    return " AND ".join(f"({predicate_sql(settings, predicate)})" for predicate in predicates)


def label_sql(settings: Settings, relation: str, table_ids: Sequence[str], predicates) -> str:
    source, _ = from_clause(settings, table_ids)
    where = where_clause(settings, predicates)
    if relation == "single":
        select = "SELECT COUNT(*) AS cardinality"
    elif relation == "multi":
        select = f"SELECT COUNT(DISTINCT {settings.entity_expression}) AS cardinality"
    else:
        raise ConfigError(f"unknown relation {relation!r}")
    return f"{select}\n{source}\nWHERE {where};"


def label_kind(relation: str) -> str:
    return {"single": "row_count", "multi": "distinct_trips"}[relation]


def structural_view(predicate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: predicate[key] for key in STRUCTURAL_KEYS if key in predicate}


def predicate_hash(category_key: str, table_ids: Sequence[str], predicates) -> str:
    """Canonical identity of a query: category, tables and predicate parameters."""

    payload = {
        "category": category_key,
        "tables": sorted(table_ids),
        "predicates": sorted(
            (structural_view(predicate) for predicate in predicates),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        ),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
