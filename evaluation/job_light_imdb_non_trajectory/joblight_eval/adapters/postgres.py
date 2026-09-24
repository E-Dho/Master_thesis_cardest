from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterEvaluation, StageMetrics, TimingOutcome
from ..config import WorkloadConfig
from ..records import LatencyRecord, PredictionRecord
from ..resources import run_logged_command
from ..workloads import load_workload, query_to_sql


# Settings that influence EXPLAIN estimates, plan shape, or timing noise.
RECORDED_SETTINGS = (
    "server_version", "default_statistics_target", "max_parallel_workers_per_gather",
    "max_parallel_workers", "jit", "work_mem", "shared_buffers", "effective_cache_size",
    "random_page_cost", "seq_page_cost", "plan_cache_mode", "autovacuum",
    "geqo_threshold", "join_collapse_limit", "from_collapse_limit",
)


class PostgresAdapter(Adapter):
    def prepare(self) -> StageMetrics:
        return self._optional_command("prepare")

    def build(self) -> StageMetrics:
        return self._optional_command("build")

    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        start = time.perf_counter()
        limited = self._evaluate(workloads, query_limit=query_limit, warmup_passes=0, repetitions=1)
        if any(record.status != "ok" for record in limited.predictions):
            raise ValueError("PostgreSQL smoke produced a non-success status")
        return StageMetrics(
            wall_seconds=time.perf_counter() - start,
            detail={"query_count": len(limited.predictions)},
        )

    def evaluate(self, workloads: tuple[WorkloadConfig, ...]) -> AdapterEvaluation:
        return self._evaluate(
            workloads,
            query_limit=None,
            warmup_passes=int(self.config.timing.get("warmup_passes", 1)),
            repetitions=int(self.config.timing.get("repetitions", 10)),
        )

    def time(
        self,
        profile: str,
        workloads: tuple[WorkloadConfig, ...],
        output_directory: Path,
        environment: dict[str, str],
    ) -> TimingOutcome:
        """Profile-controlled timing; this process is the pinned client."""
        from ..timing_guard import TimingSession

        report = output_directory / "postgres_timing_environment.json"
        session = TimingSession(profile, report, role="postgres_client")
        session.configure()
        session.verify("pre_connect")
        evaluation = self._evaluate(
            workloads,
            query_limit=None,
            warmup_passes=int(self.config.timing.get("warmup_passes", 1)),
            repetitions=int(self.config.timing.get("repetitions", 10)),
            session=session,
            profile=profile,
        )
        session.verify("post_timing")
        session.write()
        return TimingOutcome(
            predictions=evaluation.predictions,
            latencies=evaluation.latencies,
            guard_reports={workload.workload_id: report for workload in workloads},
            detail=evaluation.detail,
        )

    def _evaluate(
        self,
        workloads: tuple[WorkloadConfig, ...],
        *,
        query_limit: int | None,
        warmup_passes: int,
        repetitions: int,
        session: Any = None,
        profile: str = "",
    ) -> AdapterEvaluation:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL adapter requires psycopg>=3") from exc
        dsn = str(self.config.adapter.get("dsn", "dbname=imdb"))
        predictions: list[PredictionRecord] = []
        latencies: list[LatencyRecord] = []
        detail: dict[str, Any] = {}
        with psycopg.connect(dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                detail["postgres_settings"] = _settings(cursor)
                cursor.execute("SELECT pg_backend_pid()")
                backend_pid = int(cursor.fetchone()[0])
                detail["backend_pid"] = backend_pid
                if session is not None:
                    from ..timing_guard import allowed_cpus

                    session.note(postgres_settings=detail["postgres_settings"],
                                 backend_pid=backend_pid)
                    if session.profile.server_physical_cores:
                        detail["backend_cpus"] = session.verify_process_affinity(
                            backend_pid,
                            stage="pre_timing",
                            label="postgres_backend",
                            expected_physical_cores=session.profile.server_physical_cores,
                            disjoint_from=allowed_cpus(),
                        )
                    session.verify("pre_timing")
                measured = session.measure("explain_workloads") if session is not None else nullcontext()
                with measured:
                    for workload in workloads:
                        queries = load_workload(workload.queries_csv, workload.workload_id)
                        if query_limit is not None:
                            queries = queries[:query_limit]
                        for _ in range(warmup_passes):
                            for query in queries:
                                cursor.execute(_explain_statement(query))
                                extract_plan_rows(cursor.fetchone()[0])
                        estimates: list[float] = []
                        for repetition in range(repetitions):
                            for index, query in enumerate(queries):
                                # Timed: SQL rendering (encoding), planning via
                                # EXPLAIN, and conversion of Plan Rows to float.
                                start = time.perf_counter()
                                cursor.execute(_explain_statement(query))
                                estimate = extract_plan_rows(cursor.fetchone()[0])
                                elapsed = (time.perf_counter() - start) * 1000.0
                                if repetition == 0:
                                    estimates.append(estimate)
                                elif estimate != estimates[index]:
                                    raise RuntimeError(
                                        f"PostgreSQL estimate changed between repetitions "
                                        f"for query {query.query_id}"
                                    )
                                latencies.append(
                                    LatencyRecord(
                                        workload.workload_id,
                                        query.query_id,
                                        repetition,
                                        elapsed,
                                        scope="sql_rendering_explain_planning_and_plan_rows_conversion",
                                        device="cpu",
                                        profile=profile,
                                    )
                                )
                        for query, estimate in zip(queries, estimates):
                            predictions.append(
                                PredictionRecord(
                                    workload.workload_id,
                                    query.query_id,
                                    "ok",
                                    query.true_cardinality,
                                    estimate,
                                )
                            )
        return AdapterEvaluation(tuple(predictions), tuple(latencies), detail=detail)

    def artifact_metadata(self) -> dict[str, Any]:
        metadata = {
            "parameter_count": None,
            "serialized_model_mb": None,
            "postgres_version": self.config.adapter.get("postgres_version", "16.10"),
            "parameter_count_reason": "not applicable to PostgreSQL planner statistics",
        }
        manifest_value = self.config.adapter.get("manifest_path")
        if manifest_value:
            manifest_path = Path(str(manifest_value)).expanduser()
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                metadata["statistics_storage_mb"] = (
                    manifest.get("statistics_storage_bytes", 0) / 1_000_000.0
                )
                metadata["database_manifest"] = manifest
        return metadata

    def _optional_command(self, stage: str) -> StageMetrics:
        command = self.config.adapter.get(f"{stage}_command")
        if not command:
            return StageMetrics(detail={"status": "not_applicable", "stage": stage})
        values = {
            "seed": self.seed,
            "run_directory": self.run_directory,
        }
        rendered = str(command).format(**values)
        cwd = Path(str(self.config.adapter.get("working_directory", self.config.source_path.parent)))
        if not cwd.is_absolute():
            cwd = (self.config.source_path.parent / cwd).resolve()
        result = run_logged_command(
            rendered,
            cwd=cwd,
            env={
                str(key): (None if value is None else str(value))
                for key, value in self.config.adapter.get("environment", {}).items()
            },
            stdout_path=self.run_directory / "logs" / f"{stage}.out",
            stderr_path=self.run_directory / "logs" / f"{stage}.err",
        )
        return StageMetrics(
            wall_seconds=result.wall_seconds,
            peak_rss_bytes=result.peak_rss_bytes,
            detail={"command": list(result.command)},
        )


def extract_plan_rows(payload: Any) -> float:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        if not payload:
            raise ValueError("empty EXPLAIN JSON payload")
        payload = payload[0]
    if not isinstance(payload, dict) or "Plan" not in payload:
        raise ValueError("EXPLAIN JSON payload does not contain Plan")
    plan = payload["Plan"]
    if not isinstance(plan, dict) or "Plan Rows" not in plan:
        raise ValueError("top-level plan does not contain Plan Rows")
    return float(plan["Plan Rows"])


def _settings(cursor: Any) -> dict[str, str]:
    values = {}
    for name in RECORDED_SETTINGS:
        try:
            cursor.execute(f"SHOW {name}")
            values[name] = str(cursor.fetchone()[0])
        except Exception as exc:  # unknown setting on another server version
            values[name] = f"unavailable: {type(exc).__name__}"
    return values


def _explain_statement(query: Any) -> str:
    # SQL rendering is intentionally inside the timed region as predicate encoding.
    return "EXPLAIN (FORMAT JSON, COSTS TRUE) " + query_to_sql(query, count=False)
