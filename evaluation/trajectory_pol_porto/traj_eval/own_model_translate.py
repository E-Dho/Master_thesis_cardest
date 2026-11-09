"""v3 query rows -> the own model's structured query records (P7).

The own model reads POL workload records through
``model.src.evaluation.pol_query_adapter.pol_workload_record_to_context``,
whose predicate ``mode`` vocabulary predates v3. :func:`translate` maps one v3
row (``SEMANTICS.md``) onto it without changing the model code:

- ``nominal_eq`` -> ``nominal_eq``; ``scalar_range`` -> ``range``;
  ``scalar_unbounded`` -> ``unbounded`` (same closed bounds and operators).
- ``temporal`` -> one ``temporal_unbounded`` part per present bound:
  ``start <= U`` and ``end >= L``. That is v3's closed overlap
  ``[start, end] intersects [L, U]`` exactly (decided 2026-11-09); the
  adapter's own ``temporal_overlap`` mode is half-open (``start < U``) and is
  therefore not used.
- ``spatial`` -> ``spatial_intersects`` on the window (a one-sided v3 window
  is already stored as a rectangle beyond the data domain). The model answers
  it with the segment bounding box: diagnostic ``approx:segment_mbr``.
- ``spatio_temporal`` -> the window and the interval as separate segment
  conditions: ``approx:segment_mbr`` and ``approx:decoupled_space_time``
  (v3 requires one position of the segment to satisfy both).

Routes (which model output answers the label):

- ``distinct``: multi queries joining ``segments``: matching segments times
  ``traj_dedup_factor`` (``COUNT(DISTINCT trip_id)``);
- ``join``: multi queries without ``segments`` (trips and agents only): the
  join row count, which equals the distinct trips because every other table
  is on the one side of its join with trips;
- ``rows``: single-table queries: the table's row count.

``clamp`` (Porto, decided 2026-11-09): a nullable integer column whose NULLs
the model sees as one extra value outside the data domain (``origin_stand``:
NULL -> 0, domain 1..63). Every condition on it is intersected with the
domain, so the NULL value never matches, as in SQL. Only the standard library
is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

APPROX_MBR = "approx:segment_mbr"
APPROX_DECOUPLED = "approx:decoupled_space_time"
ROUTES = ("distinct", "join", "rows")
SEGMENT_TABLE = "segments"
ENTITY_TABLE = "trips"


class TranslationError(ValueError):
    """A v3 row the translator does not know how to express."""


@dataclass(frozen=True)
class Translation:
    record: Dict[str, Any]
    route: str
    notes: Tuple[str, ...] = field(default=())

    @property
    def diagnostic(self) -> str:
        return ";".join(self.notes)


def parse_clamp(items: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """``table:attribute=lo:hi`` items -> ``{"table:attribute": (lo, hi)}``."""

    clamp: Dict[str, Tuple[int, int]] = {}
    for item in items:
        column, _, bounds = item.partition("=")
        lo, _, hi = bounds.partition(":")
        if ":" not in column or not lo or not hi:
            raise ValueError(f"--clamp expects table:attribute=lo:hi, got {item!r}")
        clamp[column] = (int(lo), int(hi))
    return clamp


def _route(row: Mapping[str, Any]) -> str:
    relation = row["category"]["relation"]
    tables = list(row["tables"])
    if relation == "single":
        if len(tables) != 1:
            raise TranslationError(f"{row['query_id']}: a single query uses one table, not {tables}")
        return "rows"
    if relation != "multi":
        raise TranslationError(f"{row['query_id']}: unknown relation {relation!r}")
    if ENTITY_TABLE not in tables:
        raise TranslationError(f"{row['query_id']}: a multi query joins {ENTITY_TABLE}")
    return "distinct" if SEGMENT_TABLE in tables else "join"


def _clamped_bounds(column: str, lower: Optional[int], upper: Optional[int],
                    domain: Tuple[int, int], query_id: Any) -> Dict[str, Any]:
    lo, hi = domain
    lower = lo if lower is None else max(lo, int(lower))
    upper = hi if upper is None else min(hi, int(upper))
    if lower > upper:
        # v3 labels are positive, so such a query cannot occur; refuse rather than guess
        raise TranslationError(f"{query_id}: condition on {column} excludes its whole domain {domain}")
    return {"mode": "range", "lower": lower, "upper": upper}


def _scalar(predicate: Mapping[str, Any], clamp: Mapping[str, Tuple[int, int]], query_id: Any
            ) -> Dict[str, Any]:
    kind = predicate["kind"]
    column = f"{predicate['table']}:{predicate['attribute']}"
    domain = clamp.get(column)
    if kind == "nominal_eq":
        if domain is not None and not domain[0] <= int(predicate["value"]) <= domain[1]:
            raise TranslationError(f"{query_id}: {column} = {predicate['value']} is outside {domain}")
        return {"mode": "nominal_eq", "value": predicate["value"]}
    if kind == "scalar_range":
        if domain is not None:
            return _clamped_bounds(column, predicate["lower"], predicate["upper"], domain, query_id)
        return {"mode": "range", "lower": predicate["lower"], "upper": predicate["upper"]}
    if kind == "scalar_unbounded":
        operator, value = predicate["operator"], predicate["value"]
        if operator not in ("<", "<=", ">", ">="):
            raise TranslationError(f"{query_id}: operator {operator!r}")
        if domain is not None:
            if predicate.get("type") != "integer" or float(value) != int(value):
                raise TranslationError(f"{query_id}: clamped column {column} needs integer conditions")
            value = int(value)
            lower, upper = {"<": (None, value - 1), "<=": (None, value),
                            ">": (value + 1, None), ">=": (value, None)}[operator]
            return _clamped_bounds(column, lower, upper, domain, query_id)
        return {"mode": "unbounded", "operator": operator, "value": value}
    raise TranslationError(f"{query_id}: not a scalar predicate kind {kind!r}")


def _temporal_parts(table: str, attribute: str, lower: Optional[str], upper: Optional[str], query_id: Any
                    ) -> List[Dict[str, Any]]:
    if lower is None and upper is None:
        raise TranslationError(f"{query_id}: temporal condition without bounds")
    parts = []
    if upper is not None:  # start <= U
        parts.append({"table": table, "attribute": attribute, "mode": "temporal_unbounded",
                      "operator": "<=", "value": str(upper)})
    if lower is not None:  # end >= L
        parts.append({"table": table, "attribute": attribute, "mode": "temporal_unbounded",
                      "operator": ">=", "value": str(lower)})
    return parts


def _window(predicate: Mapping[str, Any], srid: Optional[int]) -> Dict[str, Any]:
    window = {key: float(predicate[key]) for key in ("min_x", "min_y", "max_x", "max_y")}
    if not (window["min_x"] <= window["max_x"] and window["min_y"] <= window["max_y"]):
        raise TranslationError("empty spatial window")
    out = {"table": predicate["table"], "attribute": predicate["attribute"], "mode": "spatial_intersects",
           **window}
    if srid is not None:
        out["srid"] = int(srid)
    return out


def translate(row: Mapping[str, Any], *, clamp: Optional[Mapping[str, Tuple[int, int]]] = None,
              srid: Optional[int] = None, temporal_attribute: str = "segment_time") -> Translation:
    """One v3 row -> the model record, its route and approximation notes."""

    clamp = dict(clamp or {})
    query_id = row.get("query_id")
    predicates: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_columns = set()
    for predicate in row["predicates"]:
        kind = predicate["kind"]
        table, attribute = predicate["table"], predicate["attribute"]
        if kind in ("nominal_eq", "scalar_range", "scalar_unbounded"):
            column = f"{table}:{attribute}"
            if column in seen_columns:  # the model takes one condition per column
                raise TranslationError(f"{query_id}: two conditions on {column}")
            seen_columns.add(column)
            predicates.append({"table": table, "attribute": attribute, **_scalar(predicate, clamp, query_id)})
        elif kind == "temporal":
            predicates.extend(_temporal_parts(table, attribute, predicate.get("lower"), predicate.get("upper"),
                                              query_id))
        elif kind == "spatial":
            predicates.append(_window(predicate, srid))
            notes.append(APPROX_MBR)
        elif kind == "spatio_temporal":
            if table != SEGMENT_TABLE:
                raise TranslationError(f"{query_id}: spatio-temporal predicate on {table}")
            predicates.append(_window(predicate, srid))
            predicates.extend(_temporal_parts(table, temporal_attribute, predicate.get("lower"),
                                              predicate.get("upper"), query_id))
            notes.extend([APPROX_MBR, APPROX_DECOUPLED])
        else:
            raise TranslationError(f"{query_id}: unknown predicate kind {kind!r}")
    record = {"query_id": query_id, "tables": sorted(row["tables"]), "predicates": predicates}
    return Translation(record=record, route=_route(row), notes=tuple(dict.fromkeys(notes)))
