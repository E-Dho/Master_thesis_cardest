from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters.registry import create_adapter
from .artifacts import (
    create_run_directory,
    read_latencies,
    read_predictions,
    run_manifest,
    write_json,
    write_latencies,
    write_predictions,
    write_resolved_config,
)
from .config import ExperimentConfig
from .metrics import attach_q_errors, summarize_latency, summarize_predictions
from .records import LatencyRecord, PredictionRecord
from .workloads import load_workload, sha256_file, workload_statistics


REQUIRED_ARTIFACTS = (
    "resolved_config.json",
    "run_manifest.json",
    "smoke_metrics.json",
    "predictions.csv",
    "latency.csv",
    "summary.json",
    "build_metrics.json",
    "resource_metrics.json",
)


def run_seed(
    config: ExperimentConfig,
    seed: int,
    *,
    run_id: str | None = None,
) -> Path:
    run_directory = create_run_directory(config, seed, run_id)
    manifest = run_manifest(config, seed, run_directory)
    manifest["command"] = "run"
    manifest["status"] = "running"
    write_resolved_config(run_directory / "resolved_config.json", config)
    write_json(run_directory / "run_manifest.json", manifest)

    adapter = create_adapter(config, seed, run_directory)
    started = datetime.now(timezone.utc)
    try:
        prepare = adapter.prepare()
        smoke = adapter.smoke(config.workloads, query_limit=2)
        write_json(run_directory / "smoke_metrics.json", smoke.to_dict())
        build = adapter.build()
        evaluation = adapter.evaluate(config.workloads)
        predictions = tuple(attach_q_errors(record) for record in evaluation.predictions)
        write_predictions(run_directory / "predictions.csv", predictions)
        write_latencies(run_directory / "latency.csv", evaluation.latencies)

        workload_metadata = _workload_metadata(config)
        summary = summarize_run(
            config,
            seed,
            predictions,
            evaluation.latencies,
            workload_metadata,
            evaluation.detail,
        )
        artifact_metadata = adapter.artifact_metadata()
        build_metrics = {
            "preprocessing_seconds": prepare.wall_seconds,
            "training_or_statistics_seconds": build.wall_seconds,
            "total_build_seconds": prepare.wall_seconds + build.wall_seconds,
            "prepare": prepare.to_dict(),
            "smoke": smoke.to_dict(),
            "build": build.to_dict(),
            "artifacts": artifact_metadata,
        }
        resource_metrics = {
            "peak_process_rss_bytes": _max_optional(
                prepare.peak_rss_bytes, build.peak_rss_bytes
            ),
            "peak_training_gpu_allocated_bytes": (
                build.peak_gpu_allocated_bytes
                if build.peak_gpu_allocated_bytes is not None
                else artifact_metadata.get("peak_training_gpu_allocated_bytes")
            ),
            "peak_training_gpu_reserved_bytes": (
                build.peak_gpu_reserved_bytes
                if build.peak_gpu_reserved_bytes is not None
                else artifact_metadata.get("peak_training_gpu_reserved_bytes")
            ),
            "measurement_scope": "adapter-reported process/stage maxima",
        }
        summary["build"] = build_metrics
        summary["resources"] = resource_metrics
        write_json(run_directory / "summary.json", summary)
        write_json(run_directory / "build_metrics.json", build_metrics)
        write_json(run_directory / "resource_metrics.json", resource_metrics)

        _validate_complete_run(run_directory, summary)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": (
                    datetime.now(timezone.utc) - started
                ).total_seconds(),
                "workloads": workload_metadata,
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        write_json(run_directory / "run_manifest.json", manifest)
        raise
    write_json(run_directory / "run_manifest.json", manifest)
    return run_directory


def initialize_staged_run(
    config: ExperimentConfig, seed: int, run_id: str | None = None
) -> Path:
    run_directory = create_run_directory(config, seed, run_id)
    write_resolved_config(run_directory / "resolved_config.json", config)
    manifest = run_manifest(config, seed, run_directory)
    manifest.update({"command": "staged", "status": "initialized"})
    write_json(run_directory / "run_manifest.json", manifest)
    return run_directory


def run_stage(
    config: ExperimentConfig,
    seed: int,
    run_directory: Path,
    stage: str,
) -> None:
    run_directory = run_directory.resolve()
    _assert_run_identity(config, seed, run_directory)
    adapter = create_adapter(config, seed, run_directory)
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stage == "prepare":
        write_json(run_directory / "prepare_metrics.json", adapter.prepare().to_dict())
        manifest["status"] = "prepared"
    elif stage == "smoke":
        write_json(
            run_directory / "smoke_metrics.json",
            adapter.smoke(config.workloads, query_limit=2).to_dict(),
        )
        manifest["status"] = "smoke_passed"
    elif stage == "build":
        write_json(run_directory / "build_stage_metrics.json", adapter.build().to_dict())
        manifest["status"] = "built"
    elif stage == "evaluate":
        evaluation = adapter.evaluate(config.workloads)
        predictions = tuple(attach_q_errors(record) for record in evaluation.predictions)
        write_predictions(run_directory / "predictions.csv", predictions)
        write_latencies(run_directory / "latency.csv", evaluation.latencies)
        write_json(run_directory / "evaluation_detail.json", evaluation.detail)
        write_json(run_directory / "artifact_metadata.json", adapter.artifact_metadata())
        manifest["status"] = "evaluated"
    elif stage == "summarize":
        _summarize_staged_run(config, seed, run_directory)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        raise ValueError(f"unknown stage {stage!r}")
    write_json(manifest_path, manifest)


def _summarize_staged_run(
    config: ExperimentConfig, seed: int, run_directory: Path
) -> None:
    predictions = read_predictions(run_directory / "predictions.csv")
    latencies = read_latencies(run_directory / "latency.csv")
    detail_path = run_directory / "evaluation_detail.json"
    detail = json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.exists() else {}
    summary = summarize_run(
        config, seed, predictions, latencies, _workload_metadata(config), detail
    )
    prepare = _read_stage_metrics(run_directory / "prepare_metrics.json")
    smoke = _read_stage_metrics(run_directory / "smoke_metrics.json")
    build = _read_stage_metrics(run_directory / "build_stage_metrics.json")
    artifacts_path = run_directory / "artifact_metadata.json"
    artifacts = (
        json.loads(artifacts_path.read_text(encoding="utf-8"))
        if artifacts_path.exists()
        else {}
    )
    build_metrics = {
        "preprocessing_seconds": prepare.get("wall_seconds", 0.0),
        "training_or_statistics_seconds": build.get("wall_seconds", 0.0),
        "total_build_seconds": prepare.get("wall_seconds", 0.0)
        + build.get("wall_seconds", 0.0),
        "prepare": prepare,
        "smoke": smoke,
        "build": build,
        "artifacts": artifacts,
    }
    resource_metrics = {
        "peak_process_rss_bytes": _max_optional(
            prepare.get("peak_rss_bytes"), build.get("peak_rss_bytes")
        ),
        "peak_training_gpu_allocated_bytes": build.get("peak_gpu_allocated_bytes"),
        "peak_training_gpu_reserved_bytes": build.get("peak_gpu_reserved_bytes"),
        "measurement_scope": "adapter-reported process/stage maxima",
    }
    summary["build"] = build_metrics
    summary["resources"] = resource_metrics
    write_json(run_directory / "summary.json", summary)
    write_json(run_directory / "build_metrics.json", build_metrics)
    write_json(run_directory / "resource_metrics.json", resource_metrics)
    _validate_complete_run(run_directory, summary)


def _assert_run_identity(
    config: ExperimentConfig, seed: int, run_directory: Path
) -> None:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (config.experiment_id, config.method_id, config.variant_id, seed)
    observed = (
        manifest.get("experiment_id"),
        manifest.get("method_id"),
        manifest.get("variant_id"),
        int(manifest.get("seed")),
    )
    if observed != expected or manifest.get("config_hash") != config.config_hash:
        raise ValueError("run directory identity does not match config and seed")


def _read_stage_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"wall_seconds": 0.0, "detail": {"status": "not_run"}}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(
    config: ExperimentConfig,
    seed: int,
    predictions: tuple[PredictionRecord, ...],
    latencies: tuple[LatencyRecord, ...],
    workload_metadata: dict[str, Any],
    evaluation_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "method_id": config.method_id,
        "variant_id": config.variant_id,
        "display_name": config.display_name,
        "protocol": config.protocol,
        "adaptation": config.adaptation,
        "seed": seed,
        "config_hash": config.config_hash,
        "evaluation_detail": evaluation_detail or {},
        "workloads": {},
    }
    for workload in config.workloads:
        workload_predictions = tuple(
            record for record in predictions if record.workload == workload.workload_id
        )
        workload_latencies = tuple(
            record for record in latencies if record.workload == workload.workload_id
        )
        expected_queries = load_workload(workload.queries_csv, workload.workload_id)
        _validate_prediction_contract(workload_predictions, expected_queries)
        result["workloads"][workload.workload_id] = {
            "accuracy": summarize_predictions(list(workload_predictions)),
            "inference": summarize_latency(list(workload_latencies)),
            "workload": workload_metadata[workload.workload_id],
        }
    return result


def _workload_metadata(config: ExperimentConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for workload in config.workloads:
        queries = load_workload(workload.queries_csv, workload.workload_id)
        metadata[workload.workload_id] = {
            "queries_csv": str(workload.queries_csv),
            "queries_csv_sha256": sha256_file(workload.queries_csv),
            "queries_sql": str(workload.queries_sql) if workload.queries_sql else None,
            "queries_sql_sha256": (
                sha256_file(workload.queries_sql) if workload.queries_sql else None
            ),
            **workload_statistics(queries),
        }
    return metadata


def _validate_complete_run(run_directory: Path, summary: dict[str, Any]) -> None:
    for workload_id, workload in summary["workloads"].items():
        accuracy = workload["accuracy"]
        if accuracy["query_count"] != workload["workload"]["query_count"]:
            raise ValueError(f"{workload_id} prediction row count is incomplete")
        for metric_family in ("raw_q_error", "raw_q_error_true_positive", "smoothed_q_error_true_zero", "smoothed_q_error"):
            for value in accuracy[metric_family].values():
                if value is not None and not math.isfinite(float(value)):
                    raise ValueError(f"{workload_id} contains non-finite {metric_family}")
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_directory / name).exists()]
    # Validation runs before the final manifest rewrite, but every path already exists.
    if missing:
        raise ValueError(f"run is missing required artifacts: {missing}")


def _max_optional(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _validate_prediction_contract(
    predictions: tuple[PredictionRecord, ...], expected_queries: list[Any]
) -> None:
    expected = {query.query_id: query.true_cardinality for query in expected_queries}
    observed: dict[int, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.query_id in observed:
            raise ValueError(f"duplicate prediction for query {prediction.query_id}")
        observed[prediction.query_id] = prediction
        if prediction.status not in {"ok", "unsupported", "failed"}:
            raise ValueError(f"invalid prediction status {prediction.status!r}")
        if prediction.status == "ok" and prediction.estimated_cardinality is None:
            raise ValueError("successful prediction is missing an estimate")
    if set(observed) != set(expected):
        raise ValueError(
            f"prediction IDs differ: missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    for query_id, truth in expected.items():
        if observed[query_id].true_cardinality != truth:
            raise ValueError(f"true cardinality mismatch for query {query_id}")
