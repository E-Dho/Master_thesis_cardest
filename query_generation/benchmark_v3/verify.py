#!/usr/bin/env python3
"""Independent recount of a stratified sample of the merged v3 benchmark.

For every sampled spatial, temporal or spatio-temporal query the job fetches
the candidate rows with prefilters that are implied by the predicates (so they
cannot drop a true match), evaluates the predicates with the numpy reference
(``exact.py``), and demands exact agreement with the stored label. Multi
queries are also recounted with a second SQL formulation (``EXISTS`` instead
of ``COUNT(DISTINCT ...)``). The sample is ``--per-category`` seeded queries per category, interleaved
across categories. Every result is appended to a JSONL file as soon as it is
computed, so an interrupted job resumes without repeating work; with
``--deadline-seconds`` the job stops cleanly (exit 75). The numpy recount is
only done when the implied prefilter yields at most ``--max-rows`` rows
(default 250,000); wider queries are covered by the in-database recounts.

Diagnostics per query:

Every sampled spatial or spatio-temporal query is additionally recounted in
the database with an independent formulation (``postgis_sql``): PostGIS
``ST_Intersects`` for spatial predicates, and for spatio-temporal predicates
the segment is first cut to the query interval by linear interpolation and the
resulting sub-segment is tested with ``ST_Intersects``. This covers the wide
queries whose candidate rows are too many for the numpy recount.

The sample is ``--per-category`` seeded queries per category, interleaved
across categories. Every result is appended to a JSONL file as soon as it is
computed, so an interrupted job resumes without repeating work; with
``--deadline-seconds`` the job stops cleanly (exit 75). The numpy recount is
only done when the implied prefilter yields at most ``--max-rows`` rows
(default 250,000); wider queries are covered by the in-database recounts.

Diagnostics per query:

- ``decoupled``: the count under the v2 semantics (geometry and time span
  tested separately) for spatio-temporal queries;
- ``st_intersects_disagreements``: fetched segments where PostGIS
  ``ST_Intersects`` and the clipping test disagree (expected: 0).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from query_generation.generate_pol_segment_coupled_benchmark import (
    append_jsonl,
    load_jsonl_recover_trailing_write,
    write_json_atomic,
)

from . import exact
from .numpy_backend import JOIN_CONDITION, NumpyBackend
from .config import Settings, load_settings
from .query_model import envelope_sql, from_clause, predicate_sql, ts_sql, window_of

SPATIAL_KINDS = {"spatial", "spatio_temporal"}
EXIT_DEADLINE = 75


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

    def _select(self, table_id: str, extra: List[Tuple[str, str]], where: str, params=None, limit: Optional[int] = None):
        table = self.settings.table(table_id)
        alias = table["alias"]
        columns = [(col, f"{alias}.{col}") for col in sorted(self.join_columns[table_id])]
        columns += _attribute_columns(self.settings, table_id) + extra
        sql = (
            f"SELECT {', '.join(expr for _, expr in columns)} FROM {table['name']} {alias}"
            + (f" WHERE {where}" if where else "")
            + (f" LIMIT {int(limit)}" if limit is not None else "")
        )
        return _to_arrays([name for name, _ in columns], self.session.rows(sql, params))

    @staticmethod
    def _size(columns: Dict[str, np.ndarray]) -> int:
        return len(next(iter(columns.values()))) if columns else 0

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
            extra = [(key, seg.columns[key]) for key in ("sx", "sy", "ex", "ey")]
            extra += [("ts", _epoch_us(seg.columns["ts"])), ("te", _epoch_us(seg.columns["te"]))]
            extra += [(f"{seg.temporal_attribute}.start", _epoch_us(seg.columns["ts"])),
                      (f"{seg.temporal_attribute}.end", _epoch_us(seg.columns["te"]))]
            for number, p in enumerate(p for p in seg_predicates if p["kind"] in SPATIAL_KINDS):
                window = (p["min_x"], p["min_y"], p["max_x"], p["max_y"])
                extra.append((f"__st_intersects_{number}", f"ST_Intersects({seg.geometry_expression}, {envelope_sql(settings, window)})"))
            # LIMIT max_rows + 1 tells "too many" apart without a separate COUNT scan.
            data[seg.table] = self._select(seg.table, extra, where, limit=self.max_rows + 1)
            if self._size(data[seg.table]) > self.max_rows:
                return None, evaluated, f"more than {self.max_rows} candidate segments"

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
                data[entity] = self._select(entity, [], where, limit=self.max_rows + 1)
                if self._size(data[entity]) > self.max_rows:
                    return None, evaluated, f"more than {self.max_rows} candidate trips"

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
    """Count with the independent predicate formulation.

    For multi queries the matching trips are computed first from the segments
    alone, in a MATERIALIZED CTE, and only then joined with the other tables.
    The planner cannot estimate predicates on constructed geometries; in the
    full run (job 22384196) a plain join with selective agent add-ons ran for
    about 20 hours. Fixing the evaluation order keeps the plan independent of
    those estimates. Only the plan changes, not the count.
    """

    seg = settings.segments
    predicates = record["predicates"]
    if record["category"]["relation"] == "single":
        source, _ = from_clause(settings, record["tables"])
        where = " AND ".join(f"({postgis_predicate_sql(settings, p)})" for p in predicates)
        return f"SELECT COUNT(*)\n{source}\nWHERE {where};"
    seg_predicates = [p for p in predicates if p["table"] == seg.table]
    if seg.table not in record["tables"] or not seg_predicates:
        source, _ = from_clause(settings, record["tables"])
        where = " AND ".join(f"({postgis_predicate_sql(settings, p)})" for p in predicates) or "TRUE"
        return f"SELECT COUNT(DISTINCT {settings.entity_expression})\n{source}\nWHERE {where};"
    seg_table = settings.table(seg.table)
    inner = " AND ".join(f"({postgis_predicate_sql(settings, p)})" for p in seg_predicates)
    outer_tables = [t for t in record["tables"] if t != seg.table]
    source, _ = from_clause(settings, outer_tables)
    outer = [f"({postgis_predicate_sql(settings, p)})" for p in predicates if p["table"] != seg.table]
    outer.insert(0, f"{settings.entity_expression} IN (SELECT trip_id FROM matching_trips)")
    return (
        f"WITH matching_trips AS MATERIALIZED (\n"
        f"  SELECT DISTINCT {seg.trip_key} AS trip_id FROM {seg_table['name']} {seg_table['alias']} WHERE {inner}\n)\n"
        f"SELECT COUNT(DISTINCT {settings.entity_expression})\n{source}\nWHERE {' AND '.join(outer)};"
    )


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


def _timed_check(result: Dict[str, Any], name: str, run, is_timeout) -> Optional[Any]:
    """Run one check; a statement timeout marks it unverified instead of failing the job."""

    step = time.perf_counter()
    try:
        return run()
    except Exception as error:  # noqa: BLE001 - only timeouts are tolerated
        if not is_timeout(error):
            raise
        result[f"{name}_timeout"] = True
        return None
    finally:
        result[f"{name}_seconds"] = time.perf_counter() - step


def _never_timeout(error: BaseException) -> bool:
    return False


def verify_record(
    settings: Settings,
    fetcher: Optional[Fetcher],
    session,
    record: Dict[str, Any],
    is_timeout=_never_timeout,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "query_id": record["query_id"],
        "category": record["category"]["key"],
        "label": record["cardinality"],
    }
    started = time.perf_counter()
    if record["category"]["relation"] == "multi":
        count = _timed_check(result, "exists", lambda: int(session.scalar(exists_sql(settings, record))), is_timeout)
        if count is not None:
            result["exists_count"] = count
            result["exists_ok"] = count == record["cardinality"]
    if any(p["kind"] in SPATIAL_KINDS for p in record["predicates"]):
        count = _timed_check(result, "postgis", lambda: int(session.scalar(postgis_sql(settings, record))), is_timeout)
        if count is not None:
            result["postgis_count"] = count
            result["postgis_ok"] = count == record["cardinality"]
    if record["category"]["dimension"] != "standard" and fetcher is not None:
        step = time.perf_counter()
        fetched = _timed_check(result, "fetch", lambda: fetcher.fetch(record), is_timeout)
        data, evaluated, note = fetched if fetched is not None else (None, record["tables"], "fetch timed out")
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
        result["numpy_seconds"] = time.perf_counter() - step
    result["seconds"] = time.perf_counter() - started
    return result


def stratified_sample(rows: List[Dict[str, Any]], per_category: int, seed: int) -> List[Dict[str, Any]]:
    """``per_category`` seeded rows per category, interleaved across categories.

    Interleaving means an interrupted run still covers every category evenly.
    """

    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]["key"]].append(row)
    picked = []
    for key in sorted(by_category):
        items = sorted(by_category[key], key=lambda row: row["query_id"])
        random.Random(f"verify:{seed}:{key}").shuffle(items)
        picked.append(items[:per_category])
    order = []
    for position in range(max((len(items) for items in picked), default=0)):
        order.extend(items[position] for items in picked if position < len(items))
    return order


def run_verification(
    settings: Settings,
    session,
    rows: List[Dict[str, Any]],
    *,
    results_path: Path,
    binding: Dict[str, Any],
    per_category: int,
    seed: int,
    max_rows: int,
    deadline: Optional[float] = None,
    log=lambda message: print(message, flush=True),
    is_timeout=_never_timeout,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Verify the sample, appending every result durably; resumes from ``results_path``."""

    stored = load_jsonl_recover_trailing_write(results_path)
    if stored:
        header = stored[0].get("binding")
        if header != binding:
            raise SystemExit(f"{results_path} belongs to a different verification run; use a new --results path")
        results = stored[1:]
    else:
        append_jsonl(results_path, {"binding": binding})
        results = []
    done = {result["query_id"] for result in results}
    sample = stratified_sample(rows, per_category, seed)
    fetcher = Fetcher(settings, session, max_rows)
    for number, record in enumerate(sample, start=1):
        if record["query_id"] in done:
            continue
        if deadline is not None and time.time() >= deadline:
            log(f"deadline reached after {len(results)}/{len(sample)} queries; resubmit to resume")
            return results, False
        result = verify_record(settings, fetcher, session, record, is_timeout)
        append_jsonl(results_path, result)
        results.append(result)
        failed = [name for name in ("numpy_ok", "exists_ok", "postgis_ok") if result.get(name) is False]
        timed_out = [name for name in ("exists", "postgis", "fetch") if result.get(f"{name}_timeout")]
        status = "MISMATCH " + ",".join(failed) if failed else "ok"
        if timed_out:
            status += " (timeout: " + ",".join(timed_out) + ")"
        log(f"{len(results)}/{len(sample)} {record['query_id']} {record['category']['key']}: {status} ({result['seconds']:.0f} s)")
    return results, True


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
        "timeouts": {
            name: [r["query_id"] for r in results if r.get(f"{name}_timeout")]
            for name in ("exists", "postgis", "fetch")
        },
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
    parser.add_argument("--results", help="durable per-query results (default: <report>.results.jsonl)")
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=250_000,
                        help="numpy recount only when the implied prefilter yields at most this many rows")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, help="stop cleanly after this many seconds (exit 75)")
    parser.add_argument("--check-timeout-seconds", type=int, default=1800,
                        help="statement timeout per check; a timed-out check is reported as unverified")
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
    benchmark = Path(args.benchmark)
    with benchmark.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    report_path = Path(args.report)
    results_path = Path(args.results) if args.results else Path(str(report_path) + ".results.jsonl")
    binding = {
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "per_category": args.per_category,
        "seed": args.seed,
    }
    deadline = time.time() + args.deadline_seconds if args.deadline_seconds else None
    with SqlSession(args.host, args.port, args.dbname, args.user, args.check_timeout_seconds * 1000) as session:
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
        results, finished = run_verification(
            settings, session, rows,
            results_path=results_path, binding=binding,
            per_category=args.per_category, seed=args.seed, max_rows=args.max_rows,
            deadline=deadline,
            is_timeout=SqlSession.is_timeout,
        )
    summary = summarize(results)
    summary["finished"] = finished
    write_json_atomic(report_path, {"binding": binding, "summary": summary, "results": results})
    print(json.dumps(summary, indent=2))
    if summary["numpy_mismatches"] or summary["exists_mismatches"] or summary["postgis_mismatches"]:
        return 1
    return 0 if finished else EXIT_DEADLINE


if __name__ == "__main__":
    sys.exit(main())
