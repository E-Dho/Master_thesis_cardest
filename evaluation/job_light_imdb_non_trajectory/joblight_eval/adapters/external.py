from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from .base import Adapter, AdapterEvaluation, StageMetrics
from ..config import WorkloadConfig
from ..records import LatencyRecord, PredictionRecord
from ..resources import run_logged_command
from ..workloads import load_workload


class ExternalCommandAdapter(Adapter):
    """Adapter for baselines isolated in their own process/environment.

    The configured evaluation command writes one CSV per workload to the
    exchange directory. Required columns are ``query_id`` and
    ``estimated_cardinality``. Optional columns are ``status``, ``diagnostic``,
    ``repetition``, and ``latency_ms``.
    """

    def prepare(self) -> StageMetrics:
        _verify_source_checkout(self.config.source)
        return self._run_stage("prepare")

    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        smoke_command = self.config.adapter.get("smoke_command")
        require_smoke = bool(self.config.adapter.get("require_smoke_before_build", False))
        if smoke_command:
            smoke = self._run_command(
                "smoke", str(smoke_command), extra={"query_limit": str(query_limit)}
            )
            return StageMetrics(
                wall_seconds=smoke.wall_seconds,
                peak_rss_bytes=smoke.peak_rss_bytes,
                detail={"command": list(smoke.command), "query_limit": query_limit},
            )
        if require_smoke:
            raise ValueError("full build is gated by adapter.smoke_command")
        return StageMetrics(detail={"status": "not_applicable", "query_limit": query_limit})

    def build(self) -> StageMetrics:
        if bool(self.config.adapter.get("require_smoke_before_build", False)):
            smoke_path = self.run_directory / "smoke_metrics.json"
            if not smoke_path.exists():
                raise ValueError("full build requires a completed smoke_metrics.json")
        build = self._run_stage("build")
        return build

    def evaluate(self, workloads: tuple[WorkloadConfig, ...]) -> AdapterEvaluation:
        exchange = self.run_directory / "exchange"
        exchange.mkdir(exist_ok=True)
        total_wall = 0.0
        timing_outputs: dict[str, list[tuple[Path, str]]] = {}
        for workload in workloads:
            command = self.config.adapter.get("evaluate_command")
            if not command:
                raise ValueError("external adapter requires adapter.evaluate_command")
            result = self._run_command(
                "evaluate",
                str(command),
                extra={
                    "workload_id": workload.workload_id,
                    "queries_csv": str(workload.queries_csv),
                    "queries_sql": str(workload.queries_sql or ""),
                    "predictions_csv": str(exchange / f"{workload.workload_id}_predictions.csv"),
                    "latency_csv": str(exchange / f"{workload.workload_id}_latency.csv"),
                },
            )
            total_wall += result.wall_seconds
            timing_outputs[workload.workload_id] = [
                (
                    exchange / f"{workload.workload_id}_latency.csv",
                    str(self.config.timing.get("device", "cpu")),
                )
            ]
            supplementary = self.config.adapter.get(
                "supplementary_evaluate_commands", []
            )
            if not isinstance(supplementary, list):
                raise ValueError(
                    "adapter.supplementary_evaluate_commands must be a list"
                )
            for profile in supplementary:
                if (
                    not isinstance(profile, dict)
                    or not profile.get("command")
                    or not profile.get("device")
                ):
                    raise ValueError(
                        "each supplementary evaluation needs command and device"
                    )
                profile_id = str(profile.get("profile_id") or profile["device"])
                latency_path = exchange / (
                    f"{workload.workload_id}_{profile_id}_latency.csv"
                )
                supplemental = self._run_command(
                    f"evaluate_{profile_id}",
                    str(profile["command"]),
                    extra={
                        "workload_id": workload.workload_id,
                        "queries_csv": str(workload.queries_csv),
                        "queries_sql": str(workload.queries_sql or ""),
                        "predictions_csv": str(
                            exchange
                            / f"{workload.workload_id}_{profile_id}_predictions.csv"
                        ),
                        "latency_csv": str(latency_path),
                    },
                )
                total_wall += supplemental.wall_seconds
                timing_outputs[workload.workload_id].append(
                    (latency_path, str(profile["device"]))
                )
        predictions: list[PredictionRecord] = []
        latencies: list[LatencyRecord] = []
        for workload in workloads:
            truth = {query.query_id: query.true_cardinality for query in load_workload(
                workload.queries_csv, workload.workload_id
            )}
            prediction_path = exchange / f"{workload.workload_id}_predictions.csv"
            rows = _read_csv(prediction_path)
            seen: set[int] = set()
            for row in rows:
                query_id = int(row["query_id"])
                seen.add(query_id)
                estimate_text = row.get("estimated_cardinality", "")
                status = row.get("status") or ("ok" if estimate_text != "" else "failed")
                predictions.append(
                    PredictionRecord(
                        workload=workload.workload_id,
                        query_id=query_id,
                        status=status,
                        true_cardinality=truth[query_id],
                        estimated_cardinality=(
                            float(estimate_text) if status == "ok" and estimate_text != "" else None
                        ),
                        diagnostic=row.get("diagnostic", ""),
                    )
                )
            for query_id, cardinality in truth.items():
                if query_id not in seen:
                    predictions.append(
                        PredictionRecord(
                            workload.workload_id,
                            query_id,
                            "failed",
                            cardinality,
                            None,
                            diagnostic="external adapter produced no row",
                        )
                    )
            for latency_path, configured_device in timing_outputs[workload.workload_id]:
                if not latency_path.exists():
                    continue
                for row in _read_csv(latency_path):
                    latencies.append(
                        LatencyRecord(
                            workload=workload.workload_id,
                            query_id=int(row["query_id"]),
                            repetition=int(row["repetition"]),
                            latency_ms=float(row["latency_ms"]),
                            scope=row.get("scope", "end_to_end"),
                            device=row.get("device") or configured_device,
                            device_name=row.get("device_name", ""),
                        )
                    )
        return AdapterEvaluation(
            predictions=tuple(predictions),
            latencies=tuple(latencies),
            detail={
                "evaluation_subprocess_wall_seconds": total_wall,
                "timing_profiles": {
                    workload_id: [device for _path, device in outputs]
                    for workload_id, outputs in timing_outputs.items()
                },
            },
        )

    def artifact_metadata(self) -> dict[str, Any]:
        manifest_path = self.run_directory / "exchange" / "artifact_manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "parameter_count": None,
            "serialized_model_mb": None,
            "reason": "external adapter did not provide artifact_manifest.json",
        }

    def _run_stage(self, stage: str) -> StageMetrics:
        command = self.config.adapter.get(f"{stage}_command")
        if not command:
            return StageMetrics(detail={"status": "not_applicable", "stage": stage})
        result = self._run_command(stage, str(command))
        bridge_metrics_path = (
            self.run_directory / "exchange" / f"{stage}_stage_metrics.json"
        )
        bridge_metrics = (
            json.loads(bridge_metrics_path.read_text(encoding="utf-8"))
            if bridge_metrics_path.exists()
            else {}
        )
        return StageMetrics(
            wall_seconds=result.wall_seconds,
            peak_rss_bytes=result.peak_rss_bytes,
            peak_gpu_allocated_bytes=bridge_metrics.get(
                "peak_training_gpu_allocated_bytes"
            ),
            peak_gpu_reserved_bytes=bridge_metrics.get(
                "peak_training_gpu_reserved_bytes"
            ),
            detail={
                "command": list(result.command),
                "bridge_metrics": bridge_metrics,
            },
        )

    def _run_command(self, stage: str, command: str, extra: dict[str, str] | None = None):
        exchange = self.run_directory / "exchange"
        exchange.mkdir(exist_ok=True)
        values = {
            "seed": str(self.seed),
            "run_directory": str(self.run_directory),
            "exchange_directory": str(exchange),
            "method_id": self.config.method_id,
            "variant_id": self.config.variant_id,
        }
        values.update(extra or {})
        rendered = command.format(**values)
        cwd_value = self.config.adapter.get("working_directory", self.config.source_path.parent)
        cwd = Path(str(cwd_value)).expanduser()
        if not cwd.is_absolute():
            cwd = (self.config.source_path.parent / cwd).resolve()
        env = {
            str(key): (None if value is None else str(value).format(**values))
            for key, value in self.config.adapter.get("environment", {}).items()
        }
        env.update(
            {
                "JOBLIGHT_SEED": str(self.seed),
                "JOBLIGHT_TIMING_WARMUP_PASSES": str(self.config.timing.get("warmup_passes", 1)),
                "JOBLIGHT_TIMING_REPETITIONS": str(self.config.timing.get("repetitions", 10)),
            }
        )
        return run_logged_command(
            rendered,
            cwd=cwd,
            env=env,
            stdout_path=self.run_directory / "logs" / f"{stage}.out",
            stderr_path=self.run_directory / "logs" / f"{stage}.err",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_source_checkout(source: dict[str, Any]) -> None:
    checkout = source.get("checkout_path")
    revision = source.get("revision")
    if not checkout or not revision:
        return
    observed = subprocess.check_output(
        ["git", "-C", str(Path(str(checkout)).expanduser()), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed != str(revision):
        raise ValueError(
            f"source checkout revision mismatch: expected {revision}, observed {observed}"
        )
