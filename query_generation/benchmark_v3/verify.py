#!/usr/bin/env python3
"""Independent recount of a stratified sample of the merged v3 benchmark.

For every sampled spatial, temporal or spatio-temporal query the job fetches
the candidate rows with prefilters that are implied by the predicates (so they
cannot drop a true match), evaluates the predicates with the numpy reference
(``exact.py``), and demands exact agreement with the stored label. Multi
queries are also recounted with a second SQL formulation (``EXISTS`` instead
of ``COUNT(DISTINCT ...)``). Diagnostics per query:

Every sampled spatial or spatio-temporal query is additionally recounted in
the database with an independent formulation (``postgis_sql``): PostGIS
``ST_Intersects`` for spatial predicates, and for spatio-temporal predicates
the segment is first cut to the query interval by linear interpolation and the
resulting sub-segment is tested with ``ST_Intersects``. This covers the wide
queries whose candidate rows are too many for the numpy recount.

Diagnostics per query:

- ``decoupled``: the count under the v2 semantics (geometry and time span
  tested separately) for spatio-temporal queries;
- ``st_intersects_disagreements``: fetched segments where PostGIS
  ``ST_Intersects`` and the clipping test disagree (expected: 0).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from query_generation.generate_pol_segment_coupled_benchmark import write_json_atomic

from . import exact
from .numpy_backend import JOIN_CONDITION, NumpyBackend
from .config import Settings, load_settings
from .query_model import envelope_sql, from_clause, predicate_sql, ts_sql, window_of

SPATIAL_KINDS = {"spatial", "spatio_temporal"}


def _epoch_us(expression: str) -> str:
    return f"(EXTRACT(EPOCH FROM {expression}) * 1000000)::bigint"


def _join_columns(settings: Settings) -> Dict[str, set]:
    alias_to_table = {table["alias"]: table_id for table_id, table in settings.base["tables"].items()}
    columns: Dict[str, set] = defaultdict(set)
    for join in settings.base["joins"]:
        left_alias, left_col, right_alias, right_col = JOIN_CONDITION.match(join["condition"]).groups()
        columns[alias_to_table[left_alias]].add(left_col)
        columns[alias_to_table[right_alias]].add(right_col)
    return columns


def _attribute_columns(settings: Settings, table_id: str) -> List[Tuple[str, str]]:
    """(name, SQL) for every scalar, nominal and non-segment temporal attribute."""

    seg = settings.segments
    columns = []
    for attr in settings.table(table_id)["attributes"]:
        if attr["type"] in {"numeric", "integer", "nominal"}:
            columns.append((attr["name"], attr["expression"]))
        elif attr["type"] == "temporal_interval":
            if table_id == seg.table:
                continue
            columns.append((f"{attr['name']}.start", _epoch_us(attr["start_expression"])))
            columns.append((f"{attr['name']}.end", _epoch_us(attr["end_expression"])))
    return columns


def _to_arrays(names: List[str], rows: List[Tuple[Any, ...]]) -> Dict[str, np.ndarray]:
    columns = list(zip(*rows)) if rows else [[] for _ in names]
    arrays = {}
    for name, values in zip(names, columns):
        array = np.asarray(values)
        if array.dtype == object and all(isinstance(v, (int, float)) for v in values):
            array = array.astype(float)
        arrays[name] = array
    return arrays


class Fetcher:
    def __init__(self, settings: Settings, session, max_rows: int):
        self.settings = settings
        self.session = session
        self.max_rows = max_rows
        self.join_columns = _join_columns(settings)
        self._agents: Optional[Dict[str, np.ndarray]] = None

    def _select(self, table_id: str, extra: List[Tuple[str, str]], where: str, params=None):
        table = self.settings.table(table_id)
        alias = table["alias"]
        columns = [(col, f"{alias}.{col}") for col in sorted(self.join_columns[table_id])]
        columns += _attribute_columns(self.settings, table_id) + extra
        sql = (
            f"SELECT {', '.join(expr for _, expr in columns)} FROM {table['name']} {alias}"
            + (f" WHERE {where}" if where else "")
        )
        return _to_arrays([name for name, _ in columns], self.session.rows(sql, params))

    def _count(self, table_id: str, where: str) -> int:
        table = self.settings.table(table_id)
        return int(self.session.scalar(f"SELECT count(*) FROM {table['name']} {table['alias']} WHERE {where}"))

    def fetch(self, record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Dict[str, np.ndarray]]], List[str], str]:
        """Returns (data, tables to evaluate, note); data is None when skipped."""

        settings, seg = self.settings, self.settings.segments
        tables = list(record["tables"])
        predicates = record["predicates"]
        seg_predicates = [p for p in predicates if p["table"] == seg.table]
        relation = record["category"]["relation"]
        data: Dict[str, Dict[str, np.ndarray]] = {}
        evaluated = list(tables)

        if seg.table in tables and not seg_predicates and relation == "multi":
            evaluated.remove(seg.table)  # every trip has segments (checked in main)
        elif seg.table in tables:
            prefilter = []
            for p in seg_predicates:
                if p["kind"] in SPATIAL_KINDS:
                    prefilter.append(f"{seg.geometry_expression} && {envelope_sql(settings, (p['min_x'], p['min_y'], p['max_x'], p['max_y']))}")
                if p["kind"] == "spatio_temporal" or (p["kind"] == "temporal" and p["attribute"] == seg.temporal_attribute):
                    if p.get("upper") is not None:
                        prefilter.append(f"{seg.columns['ts']} <= {ts_sql(p['upper'])}")
                    if p.get("lower") is not None:
                        prefilter.append(f"{seg.columns['te']} >= {ts_sql(p['lower'])}")
            if not prefilter:
                return None, evaluated, "no implied segment prefilter"
            where = " AND ".join(prefilter)
            candidates = self._count(seg.table, where)
            if candidates > self.max_rows:
                return None, evaluated, f"{candidates} candidate segments exceed --max-rows"
            extra = [(key, seg.columns[key]) for key in ("sx", "sy", "ex", "ey")]
            extra += [("ts", _epoch_us(seg.columns["ts"])), ("te", _epoch_us(seg.columns["te"]))]
            extra += [(f"{seg.temporal_attribute}.start", _epoch_us(seg.columns["ts"])),
                      (f"{seg.temporal_attribute}.end", _epoch_us(seg.columns["te"]))]
            for number, p in enumerate(p for p in seg_predicates if p["kind"] in SPATIAL_KINDS):
                window = (p["min_x"], p["min_y"], p["max_x"], p["max_y"])
                extra.append((f"__st_intersects_{number}", f"ST_Intersects({seg.geometry_expression}, {envelope_sql(settings, window)})"))
            data[seg.table] = self._select(seg.table, extra, where)

        entity = settings.entity_table
        if entity in tables:
            key = settings.base["entity"]["key"]
            if seg.table in data:
                ids = np.unique(data[seg.table][self._segment_trip_column()]).tolist()
                data[entity] = self._select(entity, [], f"{settings.table(entity)['alias']}.{key} = ANY(%s)", [ids])
            else:
                temporal = [p for p in predicates if p["table"] == entity and p["kind"] == "temporal"]
                if not temporal:
                    return None, evaluated, "no implied prefilter for the entity table"
                attr = settings.attr(entity, temporal[0]["attribute"])
                prefilter = []
                if temporal[0].get("upper") is not None:
                    prefilter.append(f"{attr['start_expression']} <= {ts_sql(temporal[0]['upper'])}")
                if temporal[0].get("lower") is not None:
                    prefilter.append(f"{attr['end_expression']} >= {ts_sql(temporal[0]['lower'])}")
                where = " AND ".join(prefilter)
                candidates = self._count(entity, where)
                if candidates > self.max_rows:
                    return None, evaluated, f"{candidates} candidate trips exceed --max-rows"
                data[entity] = self._select(entity, [], where)

        for table_id in tables:
            if table_id in data or table_id not in evaluated:
                continue
            if table_id == entity or table_id == seg.table:
                continue
            if self._agents is None:
                self._agents = self._select(table_id, [], "")
            data[table_id] = self._agents
        return data, evaluated, "ok"

    def _segment_trip_column(self) -> str:
        seg_table = self.settings.segments.table
        entity = self.settings.entity_table
        for join in self.settings.base["joins"]:
            left_alias, left_col, right_alias, right_col = JOIN_CONDITION.match(join["condition"]).groups()
            aliases = {left_alias: left_col, right_alias: right_col}
            seg_alias = self.settings.table(seg_table)["alias"]
            ent_alias = self.settings.table(entity)["alias"]
            if set(aliases) == {seg_alias, ent_alias}:
                return aliases[seg_alias]
        raise ValueError("segments do not join the entity table directly")


def _plain_envelope(window) -> str:
    """Envelope without SRID; compared only with lines built from raw coordinates."""

    x0, y0, x1, y1 = window
    return f"ST_MakeEnvelope({float(x0)!r}, {float(y0)!r}, {float(x1)!r}, {float(y1)!r})"


def postgis_predicate_sql(settings: Settings, predicate: Dict[str, Any]) -> str:
    """Independent SQL for spatial and spatio-temporal predicates (no clipping formula)."""

    seg = settings.segments
    cols = seg.columns
    window = window_of(predicate) if "min_x" in predicate else None
    if predicate["kind"] == "spatial":
        return f"ST_Intersects({seg.geometry_expression}, {envelope_sql(settings, window)})"
    if predicate["kind"] != "spatio_temporal":
        return predicate_sql(settings, predicate)
    lower, upper = predicate.get("lower"), predicate.get("upper")
    duration = f"EXTRACT(EPOCH FROM ({cols['te']} - {cols['ts']}))::float8"

    def fraction(bound: str, clamp: str, default: str) -> str:
        if bound is None:
            return default
        ratio = f"EXTRACT(EPOCH FROM ({ts_sql(bound)} - {cols['ts']}))::float8 / {duration}"
        return f"(CASE WHEN {cols['te']} = {cols['ts']} THEN 0.0::float8 ELSE {clamp}({default}, {ratio}) END)"

    ua = fraction(lower, "GREATEST", "0.0::float8")
    ub = fraction(upper, "LEAST", "1.0::float8")

    def point(u: str) -> str:
        return (
            f"ST_MakePoint({cols['sx']} + {u} * ({cols['ex']} - {cols['sx']}), "
            f"{cols['sy']} + {u} * ({cols['ey']} - {cols['sy']}))"
        )

    parts = [f"{seg.geometry_expression} && {envelope_sql(settings, window)}"]
    if upper is not None:
        parts.append(f"{cols['ts']} <= {ts_sql(upper)}")
    if lower is not None:
        parts.append(f"{cols['te']} >= {ts_sql(lower)}")
    parts.append(f"ST_Intersects(ST_MakeLine({point(ua)}, {point(ub)}), {_plain_envelope(window)})")
    return " AND ".join(parts)


def postgis_sql(settings: Settings, record: Dict[str, Any]) -> str:
    source, _ = from_clause(settings, record["tables"])
    where = " AND ".join(f"({postgis_predicate_sql(settings, p)})" for p in record["predicates"])
    if record["category"]["relation"] == "single":
        select = "SELECT COUNT(*)"
    else:
        select = f"SELECT COUNT(DISTINCT {settings.entity_expression})"
    return f"{select}\n{source}\nWHERE {where};"


def exists_sql(settings: Settings, record: Dict[str, Any]) -> str:
    """Second formulation of the distinct-trip label: semi-join instead of DISTINCT."""

    seg = settings.segments
    outer = [t for t in record["tables"] if t != seg.table]
    source, _ = from_clause(settings, outer)
    outer_predicates = [p for p in record["predicates"] if p["table"] != seg.table]
    conditions = [f"({predicate_sql(settings, p)})" for p in outer_predicates]
    if seg.table in record["tables"]:
        seg_table = settings.table(seg.table)
        join = next(
            j["condition"] for j in settings.base["joins"]
            if {j["left"], j["right"]} == {seg.table, settings.entity_table}
        )
        inner = [join] + [f"({predicate_sql(settings, p)})" for p in record["predicates"] if p["table"] == seg.table]
        conditions.append(
            f"EXISTS (SELECT 1 FROM {seg_table['name']} {seg_table['alias']} WHERE {' AND '.join(inner)})"
        )
    where = " AND ".join(conditions) if conditions else "TRUE"
    return f"SELECT COUNT(*)\n{source}\nWHERE {where};"


def verify_record(settings: Settings, fetcher: Optional[Fetcher], session, record: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "query_id": record["query_id"],
        "category": record["category"]["key"],
        "label": record["cardinality"],
    }
    started = time.perf_counter()
    if record["category"]["relation"] == "multi":
        result["exists_count"] = int(session.scalar(exists_sql(settings, record)))
        result["exists_ok"] = result["exists_count"] == record["cardinality"]
    if any(p["kind"] in SPATIAL_KINDS for p in record["predicates"]):
        result["postgis_count"] = int(session.scalar(postgis_sql(settings, record)))
        result["postgis_ok"] = result["postgis_count"] == record["cardinality"]
    if record["category"]["dimension"] != "standard" and fetcher is not None:
        data, evaluated, note = fetcher.fetch(record)
        result["numpy_note"] = note
        if data is not None:
            local = dict(record, tables=evaluated)
            backend = NumpyBackend(settings, data)
            result["numpy_count"] = backend.count(local)
            result["numpy_ok"] = result["numpy_count"] == record["cardinality"]
            result["fetched_rows"] = {t: len(next(iter(cols.values()))) if cols else 0 for t, cols in data.items()}
            seg = settings.segments
            if seg.table in data:
                spatial = [p for p in record["predicates"] if p["kind"] in SPATIAL_KINDS]
                disagreements = 0
                for number, p in enumerate(spatial):
                    table = data[seg.table]
                    mask = exact.spatial_mask(table["sx"], table["sy"], table["ex"], table["ey"],
                                              (p["min_x"], p["min_y"], p["max_x"], p["max_y"]))
                    disagreements += int(np.sum(mask != table[f"__st_intersects_{number}"].astype(bool)))
                result["st_intersects_disagreements"] = disagreements
            if any(p["kind"] == "spatio_temporal" for p in record["predicates"]):
                decoupled = copy.deepcopy(local)
                for p in decoupled["predicates"]:
                    if p["kind"] == "spatio_temporal":
                        p["kind"] = "spatio_temporal_decoupled"
                result["decoupled_count"] = backend.count(decoupled)
    result["seconds"] = time.perf_counter() - started
    return result


def stratified_sample(rows: List[Dict[str, Any]], per_category: int, seed: int):
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]["key"]].append(row)
    sample = []
    for key in sorted(by_category):
        items = list(by_category[key])
        random.Random(f"verify:{seed}:{key}").shuffle(items)
        sample.append((key, items))
    # Shuffled candidates per category; the caller stops after per_category
    # verified rows (or 4 x per_category tries, when many exceed --max-rows).
    return sample


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    numpy_checked = [r for r in results if "numpy_ok" in r]
    exists_checked = [r for r in results if "exists_ok" in r]
    postgis_checked = [r for r in results if "postgis_ok" in r]
    st = [r for r in results if "decoupled_count" in r]
    ratios = sorted(r["decoupled_count"] / r["label"] for r in st)
    return {
        "verified_queries": len(results),
        "numpy_checked": len(numpy_checked),
        "numpy_mismatches": [r["query_id"] for r in numpy_checked if not r["numpy_ok"]],
        "exists_checked": len(exists_checked),
        "exists_mismatches": [r["query_id"] for r in exists_checked if not r["exists_ok"]],
        "postgis_checked": len(postgis_checked),
        "postgis_mismatches": [r["query_id"] for r in postgis_checked if not r["postgis_ok"]],
        "numpy_skipped": sum(1 for r in results if r.get("numpy_note") not in (None, "ok")),
        "st_intersects_disagreements": sum(r.get("st_intersects_disagreements", 0) for r in results),
        "decoupled_overcount": {
            "queries": len(st),
            "queries_overcounted": sum(1 for r in st if r["decoupled_count"] > r["label"]),
            "ratio_median": ratios[len(ratios) // 2] if ratios else None,
            "ratio_max": ratios[-1] if ratios else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--single-count", type=int)
    parser.add_argument("--multi-count", type=int)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .backends import SqlSession

    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config), single_count=args.single_count, multi_count=args.multi_count)
    with open(args.benchmark, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    results: List[Dict[str, Any]] = []
    with SqlSession(args.host, args.port, args.dbname, args.user) as session:
        entity = settings.table(settings.entity_table)
        seg = settings.table(settings.segments.table)
        orphan_trips = int(session.scalar(
            f"SELECT count(*) FROM {entity['name']} {entity['alias']} WHERE NOT EXISTS "
            f"(SELECT 1 FROM {seg['name']} {seg['alias']} WHERE {settings.segments.trip_key} = "
            f"{entity['alias']}.{settings.base['entity']['key']})"
        ))
        if orphan_trips:
            print(f"{orphan_trips} trips have no segments; numpy recount assumptions do not hold", file=sys.stderr)
            return 2
        fetcher = Fetcher(settings, session, args.max_rows)
        for key, candidates in stratified_sample(rows, args.per_category, args.seed):
            checked = 0
            for tries, record in enumerate(candidates):
                if checked >= args.per_category or tries >= 4 * args.per_category:
                    break
                result = verify_record(settings, fetcher, session, record)
                results.append(result)
                if result.get("numpy_note") in (None, "ok"):
                    checked += 1
            print(f"{key}: verified {checked}", flush=True)
    report = {"summary": summarize(results), "results": results}
    write_json_atomic(Path(args.report), report)
    print(json.dumps(report["summary"], indent=2))
    summary = report["summary"]
    failed = summary["numpy_mismatches"] or summary["exists_mismatches"] or summary["postgis_mismatches"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
