from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
from importlib import metadata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig
from .records import LatencyRecord, PredictionRecord


def create_run_directory(config: ExperimentConfig, seed: int, run_id: str | None = None) -> Path:
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not configured")
    timestamp = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{timestamp}-{config.config_hash}"
    path = (
        config.results_root
        / config.experiment_id
        / config.method_id
        / config.variant_id
        / f"seed_{seed}"
        / name
    )
    path.mkdir(parents=True, exist_ok=False)
    (path / "logs").mkdir()
    return path


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_resolved_config(path: Path, config: ExperimentConfig) -> None:
    # JSON is valid YAML and avoids introducing a second serializer.
    write_json(path, config.raw)


def write_predictions(path: Path, records: Iterable[PredictionRecord]) -> None:
    rows = [record.to_dict() for record in records]
    _write_rows(path, rows)


def write_latencies(path: Path, records: Iterable[LatencyRecord]) -> None:
    rows = [record.to_dict() for record in records]
    _write_rows(path, rows)


def read_predictions(path: Path) -> tuple[PredictionRecord, ...]:
    rows = _read_rows(path)
    return tuple(
        PredictionRecord(
            workload=row["workload"],
            query_id=int(row["query_id"]),
            status=row["status"],
            true_cardinality=int(row["true_cardinality"]),
            estimated_cardinality=_optional_float(row["estimated_cardinality"]),
            raw_q_error=_optional_float(row["raw_q_error"]),
            smoothed_q_error=_optional_float(row["smoothed_q_error"]),
            diagnostic=row.get("diagnostic", ""),
        )
        for row in rows
    )


def read_latencies(path: Path) -> tuple[LatencyRecord, ...]:
    return tuple(
        LatencyRecord(
            workload=row["workload"],
            query_id=int(row["query_id"]),
            repetition=int(row["repetition"]),
            latency_ms=float(row["latency_ms"]),
            scope=row.get("scope", "end_to_end"),
            device=row.get("device", "unspecified"),
            device_name=row.get("device_name", ""),
            profile=row.get("profile", ""),
            model_core_ms=_optional_float(row.get("model_core_ms", "")),
        )
        for row in _read_rows(path)
    )


def run_manifest(config: ExperimentConfig, seed: int, run_directory: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "method_id": config.method_id,
        "variant_id": config.variant_id,
        "display_name": config.display_name,
        "protocol": config.protocol,
        "adaptation": config.adaptation,
        "seed": seed,
        "config_hash": config.config_hash,
        "config_path": str(config.source_path),
        "source": config.source,
        "resource_profile": config.resources,
        "artifact_paths": config.artifacts,
        "dataset": _dataset_identity(config.artifacts),
        "run_directory": str(run_directory),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "software_versions": _software_versions(),
        "git": git_identity(config.source_path.parent),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
        "invocation": sys.argv,
    }


def git_identity(start: Path) -> dict[str, Any]:
    try:
        root = subprocess.check_output(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"], text=True
        )
        return {"root": root, "sha": sha, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "sha": None, "dirty": None}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "psycopg", "torch"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _dataset_identity(artifacts: dict[str, Any]) -> dict[str, Any]:
    manifest_value = artifacts.get("dataset_manifest")
    if not manifest_value:
        return {
            "dataset_root": artifacts.get("dataset_root"),
            "checksum_status": "not_configured",
        }
    manifest_path = Path(str(manifest_value)).expanduser()
    if not manifest_path.exists():
        return {
            "dataset_root": artifacts.get("dataset_root"),
            "dataset_manifest": str(manifest_path),
            "checksum_status": "manifest_missing",
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["dataset_manifest"] = str(manifest_path)
    payload["checksum_status"] = "verified_manifest_loaded"
    return payload
