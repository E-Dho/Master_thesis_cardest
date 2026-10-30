"""MobilityDB / PostGIS planner estimate: ``EXPLAIN (FORMAT JSON)`` row counts.

The estimate of a query is the planner's ``Plan Rows`` of the top node, read
with ``EXPLAIN`` (never executed). Single queries are planned as
``SELECT 1 FROM ... WHERE ...`` (the row estimate, as in the IMDB PostgreSQL
adapter); multi queries as ``SELECT t.trip_id FROM ... WHERE ... GROUP BY
t.trip_id``, so the top node is the planner's estimate of the number of
distinct trips. The planner's row estimate of the input of the distinct step
is recorded too (``planner_details.csv``: the first node below the aggregate
that is not a Gather, Sort or partial aggregate; rows below a Gather are per
process in EXPLAIN and are scaled to the total with PostgreSQL's parallel
divisor). It shows how much of the error comes from the distinct step.

Variants (``adapter.variant``):

``v3_sql``
    The label SQL of the benchmark itself (v3 ``from_clause`` and
    ``predicate_sql``): what the planner makes of exactly the query that
    produced the label, including the Liang-Barsky ``CASE`` terms.
``segment_stbox``
    Segment windows and intervals become ``s.segment_tgeom && <stbox>`` so
    that MobilityDB's own selectivity estimation applies; everything else is
    the v3 SQL. Box overlap instead of the exact test, so these estimates carry
    ``approx:segment_stbox`` (the pipeline's convention for surrogate answers).
``trip_stbox``
    Distinct-trip queries only: segment windows and intervals become
    ``t.trip_tgeom && <stbox>`` on the trip row, and the segments join is
    dropped. One box per trajectory, like Exact-MBR; ``approx:trip_stbox``.
    Attribute predicates on segment rows and single queries are unsupported.

The ``stbox`` literal depends on the MobilityDB version, so it is a template
(``adapter.stbox``: ``xt``, ``x`` and ``t`` formats with ``{xmin} {ymin}
{xmax} {ymax} {tmin} {tmax} {srid}``). A template is the literal's text only,
e.g. ``SRID={srid};STBOX X(({xmin},{ymin}),({xmax},{ymax}))``; the adapter
writes ``stbox '<text>'`` itself. Templates may not contain quotes: the config
parser strips leading and trailing quote characters from scalars, which once
removed the closing quote of a quoted template. Missing interval bounds use the data
domain of the segment time attribute. The spatial literals need the SRID of
the data (MobilityDB raises "Operation on mixed SRID" otherwise), so
``prepare`` of an stbox variant executes one overlap per template it uses on a
single row (``SELECT <tgeompoint> && <literal> ... LIMIT 1``) and fails early
on a syntax or SRID error.

Statistics: the adapter never runs ``ANALYZE`` unless ``adapter.analyze`` is
true. ``prepare`` records server, PostGIS and MobilityDB versions, settings,
``last_analyze`` per table and a fingerprint of the planner statistics
(``pg_stats``) of the benchmark tables; ``build`` records the statistics'
storage size as the model size. A fingerprint that differs between runs
means different statistics.

Connection: ``adapter.dsn`` (libpq keywords; ``${VAR}`` expands from the
environment, e.g. ``host=${MOBILITYDB_SOCKET_DIR} port=${MOBILITYDB_PORT}
dbname=${MOBILITYDB_DB}``). ``adapter.session_timezone`` (optional) is set on
every connection, because ``stbox`` and ``tgeompoint`` times are
``timestamptz``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.eval_core.adapters.base import AdapterEvaluation, StageMetrics
from evaluation.eval_core.config import WorkloadConfig
from evaluation.eval_core.workload_formats import load_queries

from .base import MODEL_META, Estimate, InProcessTrajectoryAdapter, _peak_rss_bytes

VARIANTS = ("v3_sql", "segment_stbox", "trip_stbox")
SEGMENT_SPACE_TIME = ("spatial", "spatio_temporal", "temporal")
RECORDED_SETTINGS = (
    "server_version", "default_statistics_target", "TimeZone", "work_mem", "effective_cache_size",
    "random_page_cost", "seq_page_cost", "cpu_tuple_cost", "jit", "max_parallel_workers_per_gather",
    "enable_hashagg", "enable_seqscan", "enable_indexscan", "enable_bitmapscan",
    "parallel_leader_participation", "max_parallel_workers",
)
PASS_THROUGH = ("Gather", "Gather Merge", "Sort", "Incremental Sort", "Materialize")
STATEMENT_TIMEOUT_MS = 60_000


def parallel_divisor(workers: int, leader_participation: bool = True) -> float:
    """PostgreSQL's ``get_parallel_divisor``: per-process rows x divisor = total rows."""

    divisor = float(workers)
    if leader_participation:
        leader = 1.0 - 0.3 * workers
        if leader > 0:
            divisor += leader
    return divisor


def plan_rows(payload: Any, leader_participation: bool = True
              ) -> Tuple[float, str, Optional[float], Optional[str], Optional[int]]:
    """(top Plan Rows, top node type, total input rows of the distinct step, its node, workers).

    The input is the first node below a top Aggregate/Group/Unique that is not a
    Gather, Sort or partial aggregate; below a Gather its per-process rows are
    scaled to the total.
    """

    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        payload = payload[0]
    plan = payload["Plan"]
    input_rows = input_type = workers = None
    if plan.get("Node Type") in ("Aggregate", "Group", "Unique") and plan.get("Plans"):
        node = plan["Plans"][0]
        while node.get("Plans") and (node["Node Type"] in PASS_THROUGH or (
                node["Node Type"] == "Aggregate" and node.get("Partial Mode") == "Partial")):
            if node["Node Type"] in ("Gather", "Gather Merge"):
                workers = int(node.get("Workers Planned") or 0)
            node = node["Plans"][0]
        input_rows, input_type = float(node["Plan Rows"]), str(node["Node Type"])
        if workers:
            input_rows *= parallel_divisor(workers, leader_participation)
    return float(plan["Plan Rows"]), str(plan["Node Type"]), input_rows, input_type, workers


class PlannerSql:
    """Renders the planner SQL of a v3 query for one variant."""

    def __init__(self, settings: Any, variant: str, stbox: Dict[str, str]):
        if variant not in VARIANTS:
            raise ValueError(f"adapter.variant must be one of {VARIANTS}")
        for name, template in stbox.items():
            if "'" in str(template) or '"' in str(template) or not str(template).strip().upper().startswith(
                    ("SRID=", "STBOX")):
                raise ValueError(
                    f"adapter.stbox.{name} must be the literal text only, e.g. "
                    f"'SRID={{srid}};STBOX X(...)' without quotes or a type name; got {template!r}")
        self.settings = settings
        self.variant = variant
        self.stbox = stbox
        self.segments = settings.segments
        domain = settings.attr(self.segments.table, self.segments.temporal_attribute)["domain"]
        self.time_domain = (str(domain["min"]), str(domain["max"]))

    def _temporal_geometry(self, table_id: str) -> str:
        """The tgeompoint column of a table, as the v3 config names it."""
        for attribute in self.settings.table(table_id)["attributes"]:
            if attribute.get("temporal_geometry_expression"):
                return str(attribute["temporal_geometry_expression"])
        raise ValueError(f"the v3 config names no temporal geometry for {table_id}")

    def _box(self, predicate: Dict[str, Any]) -> str:
        from query_generation.benchmark_v3.query_model import timestamp_text
        from query_generation.query_generator import parse_timestamp

        kind = predicate["kind"]
        values: Dict[str, Any] = {"srid": self.settings.srid}
        if kind in ("spatial", "spatio_temporal"):
            values.update(xmin=repr(float(predicate["min_x"])), ymin=repr(float(predicate["min_y"])),
                          xmax=repr(float(predicate["max_x"])), ymax=repr(float(predicate["max_y"])))
        if kind in ("temporal", "spatio_temporal"):
            lower = predicate.get("lower") or self.time_domain[0]
            upper = predicate.get("upper") or self.time_domain[1]
            values.update(tmin=timestamp_text(parse_timestamp(lower)),
                          tmax=timestamp_text(parse_timestamp(upper)))
        template = {"spatial": "x", "temporal": "t", "spatio_temporal": "xt"}[kind]
        if template not in self.stbox:
            raise ValueError(f"adapter.stbox.{template} is not configured")
        return f"stbox '{self.stbox[template].format(**values)}'"

    def literal_checks(self) -> List[Tuple[str, str]]:
        """(template, SQL) pairs that execute each stbox template this variant uses
        once against its tgeompoint column (one row), on domain-sized boxes."""

        from query_generation.benchmark_v3.query_model import from_clause

        if self.variant == "v3_sql":
            return []
        table = self.settings.entity_table if self.variant == "trip_stbox" else self.segments.table
        geometry = self._temporal_geometry(table)
        source, _ = from_clause(self.settings, [table])
        domain = self.settings.attr(self.segments.table, self.segments.spatial_attribute)["domain"]
        space = {key: domain[key] for key in ("min_x", "min_y", "max_x", "max_y")}
        probes = {"x": {"kind": "spatial", **space}, "t": {"kind": "temporal"},
                  "xt": {"kind": "spatio_temporal", **space}}
        return [(name, f"SELECT {geometry} && {self._box(probe)}\n{source}\nLIMIT 1")
                for name, probe in probes.items()]

    def render(self, record: Any) -> Tuple[Optional[str], str]:
        """(SQL or None when unsupported, unsupported reason or approx diagnostic)."""

        from query_generation.benchmark_v3.query_model import from_clause, predicate_sql

        tables = list(record.tables)
        predicates = list(record.predicates)
        conditions: List[str] = []
        diagnostic = ""
        if self.variant == "trip_stbox":
            if record.relation != "multi":
                return None, "single-table query (trip_stbox answers distinct-trip queries only)"
            geometry = self._temporal_geometry(self.settings.entity_table)
            for predicate in predicates:
                if predicate["table"] != self.segments.table:
                    conditions.append(predicate_sql(self.settings, predicate))
                elif predicate["kind"] in SEGMENT_SPACE_TIME and (
                        predicate["kind"] != "temporal" or predicate["attribute"] == self.segments.temporal_attribute):
                    conditions.append(f"{geometry} && {self._box(predicate)}")
                    diagnostic = "approx:trip_stbox"
                else:
                    return None, f"segment-level attribute predicate on {predicate['attribute']}"
            tables = [table for table in tables if table != self.segments.table]
        else:
            segment_geometry = self._temporal_geometry(self.segments.table)
            for predicate in predicates:
                is_box = (self.variant == "segment_stbox" and predicate["table"] == self.segments.table
                          and predicate["kind"] in SEGMENT_SPACE_TIME
                          and (predicate["kind"] != "temporal"
                               or predicate["attribute"] == self.segments.temporal_attribute))
                if is_box:
                    conditions.append(f"{segment_geometry} && {self._box(predicate)}")
                    diagnostic = "approx:segment_stbox"
                else:
                    conditions.append(predicate_sql(self.settings, predicate))
        source, _ = from_clause(self.settings, tables)
        where = " AND ".join(f"({condition})" for condition in conditions)
        if record.relation == "single":
            sql = f"SELECT 1\n{source}\nWHERE {where}"
        else:
            entity = self.settings.entity_expression
            sql = f"SELECT {entity}\n{source}\nWHERE {where}\nGROUP BY {entity}"
        return sql, diagnostic


class PlannerEstimator:
    def __init__(self, connection: Any, renderer: PlannerSql, leader_participation: bool = True):
        self.connection = connection
        self.renderer = renderer
        self.leader_participation = leader_participation
        self.details: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def estimate(self, record: Any) -> Estimate:
        sql, note = self.renderer.render(record)
        if sql is None:
            return "unsupported", None, note
        with self.connection.cursor() as cursor:
            cursor.execute(f"EXPLAIN (FORMAT JSON, COSTS TRUE) {sql}")
            payload = cursor.fetchone()[0]
        rows, node, input_rows, input_node, workers = plan_rows(payload, self.leader_participation)
        self.details.setdefault((record.workload, record.query_id), {
            "workload": record.workload, "query_id": record.query_id, "top_node": node, "plan_rows": rows,
            "input_node": input_node, "input_plan_rows": input_rows, "parallel_workers": workers,
            "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        })
        return "ok", rows, note


def _expand(value: str) -> str:
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"unexpanded environment variable in {value!r}")
    return expanded


class MobilityDbAdapter(InProcessTrajectoryAdapter):
    latency_scope = "sql_rendering_explain_planning_and_plan_rows_conversion"
    timing_role = "in_process_mobilitydb_planner"

    @property
    def variant(self) -> str:
        return str(self.config.adapter.get("variant", "v3_sql"))

    def _connect(self) -> Any:
        import psycopg

        dsn = self.config.adapter.get("dsn")
        if not dsn:
            raise ValueError("adapter.dsn is required")
        connection = psycopg.connect(_expand(str(dsn)), autocommit=True)
        with connection.cursor() as cursor:
            cursor.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
            timezone = self.config.adapter.get("session_timezone")
            if timezone:
                cursor.execute("SELECT set_config('TimeZone', %s, false)", (str(timezone),))
        return connection

    def _estimator_for(self, connection: Any) -> PlannerEstimator:
        with connection.cursor() as cursor:
            cursor.execute("SHOW parallel_leader_participation")
            leader = str(cursor.fetchone()[0]).lower() == "on"
        return PlannerEstimator(connection, self._renderer(), leader)

    def _renderer(self) -> PlannerSql:
        return PlannerSql(self.settings, self.variant, dict(self.config.adapter.get("stbox") or {}))

    def _tables(self) -> List[Tuple[str, str]]:
        names = []
        for table in self.settings.base["tables"].values():
            schema, _, name = str(table["name"]).partition(".")
            names.append((schema, name))
        return names

    def _identity(self, connection: Any) -> Dict[str, Any]:
        identity: Dict[str, Any] = {"settings": {}, "extensions": {}, "tables": {}}
        with connection.cursor() as cursor:
            for name in RECORDED_SETTINGS:
                try:
                    cursor.execute(f"SHOW {name}")
                    identity["settings"][name] = str(cursor.fetchone()[0])
                except Exception as exc:  # setting unknown on this server version
                    identity["settings"][name] = f"unavailable: {type(exc).__name__}"
            cursor.execute("SELECT version()")
            identity["version"] = cursor.fetchone()[0]
            cursor.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
            identity["extensions"] = {name: version for name, version in cursor.fetchall()}
            for function in ("postgis_full_version()", "mobilitydb_full_version()", "mobilitydb_version()"):
                try:
                    cursor.execute(f"SELECT {function}")
                    identity[function] = cursor.fetchone()[0]
                except Exception:
                    identity[function] = None
            digest = hashlib.sha256()
            for schema, name in self._tables():
                cursor.execute(
                    "SELECT n_live_tup, last_analyze, last_autoanalyze FROM pg_stat_user_tables "
                    "WHERE schemaname = %s AND relname = %s", (schema, name))
                row = cursor.fetchone()
                cursor.execute(
                    "SELECT attname, null_frac, avg_width, n_distinct, "
                    "most_common_vals::text, most_common_freqs::text, histogram_bounds::text, correlation "
                    "FROM pg_stats WHERE schemaname = %s AND tablename = %s ORDER BY attname", (schema, name))
                stats = cursor.fetchall()
                digest.update(json.dumps([schema, name, [list(map(str, s)) for s in stats]]).encode("utf-8"))
                identity["tables"][f"{schema}.{name}"] = {
                    "n_live_tup": None if row is None else int(row[0]),
                    "last_analyze": None if row is None or row[1] is None else row[1].isoformat(),
                    "last_autoanalyze": None if row is None or row[2] is None else row[2].isoformat(),
                    "columns_with_statistics": [s[0] for s in stats],
                }
            identity["statistics_fingerprint"] = digest.hexdigest()
        return identity

    def _statistics_bytes(self, connection: Any) -> int:
        schemas = sorted({schema for schema, _ in self._tables()})
        names = [name for _, name in self._tables()]
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(SUM(pg_column_size(s.*)), 0) FROM pg_statistic s "
                "JOIN pg_class c ON c.oid = s.starelid JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = ANY(%s) AND c.relname = ANY(%s)", (schemas, names))
            return int(cursor.fetchone()[0])

    # -- stages ------------------------------------------------------------

    def prepare(self) -> StageMetrics:
        started = time.perf_counter()
        self._renderer()  # validate variant and templates before touching the database
        with self._connect() as connection:
            analyzed = False
            if self.config.adapter.get("analyze") is True:
                with connection.cursor() as cursor:
                    for schema, name in self._tables():
                        cursor.execute(f'ANALYZE "{schema}"."{name}"')
                analyzed = True
            identity = self._identity(connection)
            checks = []
            with connection.cursor() as cursor:
                for template, sql in self._renderer().literal_checks():
                    try:
                        cursor.execute(sql)
                        cursor.fetchall()
                    except Exception as exc:
                        raise ValueError(f"adapter.stbox.{template} does not work on this database: "
                                         f"{type(exc).__name__}: {exc}\n{sql}") from exc
                    checks.append({"template": template, "sql": sql, "ok": True})
        identity["analyzed_by_this_run"] = analyzed
        identity["stbox_literal_checks"] = checks
        (self.run_directory / "database_identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return StageMetrics(wall_seconds=time.perf_counter() - started, peak_rss_bytes=_peak_rss_bytes(),
                            detail={"variant": self.variant, "statistics_fingerprint": identity["statistics_fingerprint"],
                                    "analyzed_by_this_run": analyzed})

    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        started = time.perf_counter()
        results = []
        with self._connect() as connection:
            estimator = self._estimator_for(connection)
            for workload in workloads:
                for record in list(load_queries(workload))[:query_limit]:
                    status, estimate, note = estimator.estimate(record)
                    results.append({"workload": workload.workload_id, "query_id": record.query_id,
                                    "status": status, "estimate": estimate, "diagnostic": note})
        return StageMetrics(wall_seconds=time.perf_counter() - started, detail={"queries": results})

    def build(self) -> StageMetrics:
        started = time.perf_counter()
        with self._connect() as connection:
            identity = self._identity(connection)
            size = self._statistics_bytes(connection)
        prepared = json.loads((self.run_directory / "database_identity.json").read_text(encoding="utf-8"))
        if prepared["statistics_fingerprint"] != identity["statistics_fingerprint"]:
            raise ValueError("planner statistics changed between prepare and build")
        self.model_directory.mkdir(parents=True, exist_ok=True)
        meta = {"method": "mobilitydb_planner", "variant": self.variant, "serialized_bytes": size,
                "serialized_bytes_scope": "pg_statistic rows of the benchmark tables",
                "statistics_fingerprint": identity["statistics_fingerprint"],
                "server_version": identity["settings"].get("server_version"),
                "extensions": identity["extensions"],
                "default_statistics_target": identity["settings"].get("default_statistics_target")}
        (self.model_directory / MODEL_META).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return StageMetrics(wall_seconds=time.perf_counter() - started, detail=meta)

    def _model(self) -> PlannerEstimator:
        meta = json.loads((self.model_directory / MODEL_META).read_text(encoding="utf-8"))
        connection = self._connect()
        if self._identity(connection)["statistics_fingerprint"] != meta["statistics_fingerprint"]:
            connection.close()
            raise ValueError("planner statistics changed since build; start a new run")
        self._estimator = self._estimator_for(connection)
        return self._estimator

    def _evaluate(self, workloads, *, session: Any = None, profile: str = "") -> AdapterEvaluation:
        self._estimator = None
        try:
            return super()._evaluate(workloads, session=session, profile=profile)
        finally:
            if self._estimator is not None:
                if session is None:  # the accuracy pass; timing runs keep its file
                    self._write_details(self._estimator)
                self._estimator.connection.close()

    def _write_details(self, estimator: PlannerEstimator) -> None:
        path = self.run_directory / "planner_details.csv"
        fields = ["workload", "query_id", "top_node", "plan_rows", "input_node", "input_plan_rows",
                  "parallel_workers", "sql_sha256"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for key in sorted(estimator.details):
                writer.writerow(estimator.details[key])

    def _estimate(self, model: PlannerEstimator, record: Any) -> Estimate:
        return model.estimate(record)
