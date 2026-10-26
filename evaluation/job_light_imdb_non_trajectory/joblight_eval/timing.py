"""Profile-controlled, timing-only stage linked to an existing accuracy run.

Accuracy and latency are decoupled: the ``evaluate`` stage produces the
predictions used for accuracy, while ``time`` re-measures latency under one
named profile (``timing_guard.PROFILES``) in its own, correctly pinned job,
reusing the run's checkpoints.  Each timing result lives in
``<run>/timing/<profile>/<utc>-<hash>/`` and references the accuracy run.

Reuse is checked, not assumed:

* the run's resolved config must equal the current config after removing
  timing-only keys; any other difference must be allowed explicitly and is
  recorded as config drift;
* the timing run's estimates are compared with the accuracy run's; methods
  declare ``deterministic`` (tolerances enforced) or ``stochastic`` (differences
  reported) in ``timing.estimate_consistency``;
* every measuring process must write a passing timing-guard report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .adapters.registry import create_adapter
from .artifacts import git_identity, read_predictions, write_json, write_latencies, write_predictions
from .config import TIMING_ONLY_CONFIG_PATHS, ExperimentConfig, accuracy_config_hash, strip_paths  # noqa: F401 (re-exported)
from .metrics import attach_q_errors, summarize_latency, summarize_model_core
from .records import LatencyRecord, PredictionRecord
from .timing_guard import (
    EXPECTED_HARDWARE_ENV,
    LATENCY_DEFINITION,
    PROFILES,
    UNPINNED_DEBUG_ENV,
    environment_snapshot,
    expected_hardware,
    get_profile,
    hardware_violations,
    launch_violations,
    nvidia_smi_gpu_names,
    profile_environment,
)


DEFAULT_CONSISTENCY = {"mode": "deterministic", "relative_tolerance": 1e-9, "absolute_tolerance": 0.0}
ACCURACY_STATUSES = {"evaluated", "complete"}


class TimingProtocolError(RuntimeError):
    """A timing run finished but violates the timing protocol."""


def declared_profiles(config: ExperimentConfig) -> list[str]:
    profiles = config.timing.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("timing.profiles must be a list")
    unknown = [profile for profile in profiles if profile not in PROFILES]
    if unknown:
        raise ValueError(f"unknown timing profiles {unknown}; known: {sorted(PROFILES)}")
    return [str(profile) for profile in profiles]


def primary_profile(config: ExperimentConfig) -> str | None:
    value = config.timing.get("primary_profile")
    return None if value in (None, "") else str(value)


def consistency_settings(config: ExperimentConfig) -> dict[str, Any]:
    settings = dict(DEFAULT_CONSISTENCY)
    settings.update(config.timing.get("estimate_consistency", {}) or {})
    if settings["mode"] not in {"deterministic", "stochastic"}:
        raise ValueError("timing.estimate_consistency.mode must be deterministic or stochastic")
    settings["relative_tolerance"] = float(settings["relative_tolerance"])
    settings["absolute_tolerance"] = float(settings["absolute_tolerance"])
    return settings


def config_drift(before: Any, after: Any, prefix: str = "") -> list[str]:
    """Dotted paths whose values differ between two config mappings."""
    if isinstance(before, dict) and isinstance(after, dict):
        drift: list[str] = []
        for key in sorted(set(before) | set(after), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                drift.append(path)
            else:
                drift.extend(config_drift(before[key], after[key], path))
        return drift
    return [] if before == after else [prefix or "<root>"]


def _allowed(path: str, allowances: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item + ".") for item in allowances)


def estimate_consistency(
    accuracy: Sequence[PredictionRecord],
    timing: Sequence[PredictionRecord],
    settings: dict[str, Any],
) -> dict[str, Any]:
    reference = {(record.workload, record.query_id): record for record in accuracy}
    relative = settings["relative_tolerance"]
    absolute = settings["absolute_tolerance"]
    status_mismatches: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []
    max_abs = 0.0
    max_rel = 0.0
    compared = 0
    for record in timing:
        key = (record.workload, record.query_id)
        base = reference.get(key)
        if base is None:
            missing.append(key)
            continue
        if base.status != record.status:
            status_mismatches.append({"workload": key[0], "query_id": key[1],
                                      "accuracy": base.status, "timing": record.status})
            continue
        if base.estimated_cardinality is None or record.estimated_cardinality is None:
            continue
        compared += 1
        a = float(base.estimated_cardinality)
        b = float(record.estimated_cardinality)
        difference = abs(a - b)
        scale = max(abs(a), abs(b))
        rel = difference / scale if scale > 0 else 0.0
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, rel)
        if not math.isclose(a, b, rel_tol=relative, abs_tol=absolute):
            violations.append({"workload": key[0], "query_id": key[1], "accuracy": a, "timing": b})
    deterministic = settings["mode"] == "deterministic"
    passed = not status_mismatches and not missing and (not deterministic or not violations)
    return {
        "mode": settings["mode"],
        "relative_tolerance": relative,
        "absolute_tolerance": absolute,
        "compared_estimates": compared,
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "violation_count": len(violations),
        "violations": violations[:25],
        "status_mismatches": status_mismatches[:25],
        "missing_in_accuracy_run": [list(item) for item in missing[:25]],
        "passed": passed,
    }


def _guard_digest(report: dict[str, Any]) -> dict[str, Any]:
    snapshot = report.get("snapshot", {})
    torch = snapshot.get("torch", {})
    return {
        "passed": bool(report.get("passed")),
        "profile": report.get("profile", {}).get("profile_id"),
        "role": report.get("role"),
        "node": snapshot.get("node", {}).get("hostname"),
        "cpu_model": snapshot.get("cpu", {}).get("model_name"),
        "cpu_flags": snapshot.get("cpu", {}).get("flags_of_interest", []),
        "logical_cpus": snapshot.get("affinity", {}).get("logical_cpu_list"),
        "physical_core_count": len(snapshot.get("affinity", {}).get("physical_cores", [])),
        "gpu_names": torch.get("cuda_device_names", []),
        "slurm_job_id": snapshot.get("slurm", {}).get("SLURM_JOB_ID"),
        "compute_threads": report.get("compute_threads"),
        "hard_failures": report.get("hard_failures", []),
        "warnings": report.get("warnings", []),
        "cpu_to_wall_ratio": [item.get("cpu_to_wall_ratio") for item in report.get("measurements", [])],
    }


def run_timing_stage(
    config: ExperimentConfig,
    seed: int,
    run_directory: Path,
    profile: str,
    *,
    allow_config_drift: Sequence[str] = (),
    allow_estimate_drift: bool = False,
) -> Path:
    get_profile(profile)
    if profile not in declared_profiles(config):
        raise ValueError(f"profile {profile} is not declared in timing.profiles of {config.source_path}")
    run_directory = run_directory.resolve()
    manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    expected = (config.experiment_id, config.method_id, config.variant_id, seed)
    observed = (manifest.get("experiment_id"), manifest.get("method_id"),
                manifest.get("variant_id"), int(manifest.get("seed")))
    if observed != expected:
        raise ValueError(f"accuracy run identity {observed} differs from config/seed {expected}")
    if manifest.get("status") not in ACCURACY_STATUSES:
        raise ValueError(f"accuracy run status is {manifest.get('status')!r}; evaluate it first")
    if not (run_directory / "predictions.csv").exists():
        raise FileNotFoundError(run_directory / "predictions.csv")
    accuracy_raw = json.loads((run_directory / "resolved_config.json").read_text(encoding="utf-8"))
    drift = config_drift(strip_paths(accuracy_raw), strip_paths(config.raw))
    disallowed = [path for path in drift if not _allowed(path, allow_config_drift)]
    if disallowed:
        raise ValueError(
            "the accuracy run was produced with a different non-timing configuration: "
            + ", ".join(disallowed)
            + "; pass --allow-config-drift for differences that cannot change estimates"
        )

    violations = launch_violations(profile)
    debug_unpinned = os.environ.get(UNPINNED_DEBUG_ENV) == "1"
    if violations and not debug_unpinned:
        raise TimingProtocolError("; ".join(violations))

    # Hardware class: one declared CPU/GPU model per profile for all methods.
    # Checked here before measuring, inside every measuring process, and again
    # on the guard reports below.
    launcher_snapshot = environment_snapshot()
    cuda_profile = PROFILES[profile].device == "cuda"
    smi_names = nvidia_smi_gpu_names() if cuda_profile else None
    detected = {"cpu_model": launcher_snapshot["cpu"]["model_name"], "gpu_names": smi_names or []}
    hardware = expected_hardware(profile, detected=detected)
    preflight_expected = dict(hardware) if smi_names else {**hardware, "gpu_name": None}
    hardware_problems = hardware_violations(preflight_expected, detected["cpu_model"], smi_names)
    if hardware_problems and not debug_unpinned:
        raise TimingProtocolError("hardware class violated: " + "; ".join(hardware_problems))

    commands = config.adapter.get("timing_commands", {}) or {}
    identity = json.dumps(
        {"profile": profile, "timing": config.timing,
         "command": commands.get(profile) if isinstance(commands, dict) else None},
        sort_keys=True, default=str,
    )
    timing_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = run_directory / "timing" / profile / f"{stamp}-{timing_hash}"
    directory.mkdir(parents=True, exist_ok=False)
    timing_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "profile": profile,
        "profile_definition": asdict(PROFILES[profile]),
        "latency_definition": LATENCY_DEFINITION,
        "timing_hash": timing_hash,
        "accuracy_run": str(run_directory),
        "accuracy_run_config_hash": manifest.get("config_hash"),
        "accuracy_config_hash": accuracy_config_hash(accuracy_raw),
        "current_config_hash": config.config_hash,
        "current_accuracy_config_hash": config.accuracy_config_hash,
        "config_path": str(config.source_path),
        "config_drift": drift,
        "allowed_config_drift": list(allow_config_drift),
        "timing_config": config.timing,
        "timing_command": commands.get(profile) if isinstance(commands, dict) else None,
        "launch_violations": violations,
        "hardware_class": hardware,
        "hardware_violations": hardware_problems,
        "debug_unpinned": bool((violations or hardware_problems) and debug_unpinned),
        "launcher_snapshot": launcher_snapshot,
        "git": git_identity(config.source_path.parent),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(directory / "timing_manifest.json", timing_manifest)
    failures: list[str] = []
    try:
        adapter = create_adapter(config, seed, run_directory)
        expected_json = json.dumps(
            {key: hardware[key] for key in ("profile", "cpu_model", "gpu_name")}, sort_keys=True
        )
        environment = {**profile_environment(profile), EXPECTED_HARDWARE_ENV: expected_json}
        previous = os.environ.get(EXPECTED_HARDWARE_ENV)
        os.environ[EXPECTED_HARDWARE_ENV] = expected_json  # in-process adapters
        try:
            outcome = adapter.time(profile, config.workloads, directory, environment)
        finally:
            if previous is None:
                os.environ.pop(EXPECTED_HARDWARE_ENV, None)
            else:
                os.environ[EXPECTED_HARDWARE_ENV] = previous
        guards: dict[str, Any] = {}
        for workload_id, path in outcome.guard_reports.items():
            if not Path(path).exists():
                failures.append(f"{workload_id}: the measuring process wrote no timing-guard report")
                continue
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            digest = _guard_digest(report)
            guards[workload_id] = digest
            if digest["profile"] != profile:
                failures.append(f"{workload_id}: guard ran profile {digest['profile']!r}")
            if not digest["passed"]:
                failures.append(f"{workload_id}: timing guard failed {digest['hard_failures']}")
            observed_problems = hardware_violations(
                hardware, digest["cpu_model"], digest["gpu_names"] if cuda_profile else None
            )
            if observed_problems:
                hardware_problems.extend(f"{workload_id}: {item}" for item in observed_problems)
                if not debug_unpinned:
                    failures.append(f"{workload_id}: hardware class violated: {observed_problems}")
        accuracy = read_predictions(run_directory / "predictions.csv")
        consistency = estimate_consistency(accuracy, outcome.predictions, consistency_settings(config))
        if not consistency["passed"] and not allow_estimate_drift:
            failures.append(
                f"timing estimates disagree with the accuracy run "
                f"({consistency['violation_count']} violations, "
                f"{len(consistency['status_mismatches'])} status mismatches)"
            )
        predictions = tuple(attach_q_errors(record) for record in outcome.predictions)
        write_predictions(directory / "predictions.csv", predictions)
        write_latencies(directory / "latency.csv", outcome.latencies)
        workloads: dict[str, Any] = {}
        for workload in config.workloads:
            records = [record for record in outcome.latencies if record.workload == workload.workload_id]
            if not records:
                failures.append(f"{workload.workload_id}: no latency observations")
            workloads[workload.workload_id] = {
                "inference": summarize_latency(records),
                "model_core": summarize_model_core(records),
                "scopes": sorted({record.scope for record in records}),
                "device_names": sorted({record.device_name for record in records if record.device_name}),
                "guard": guards.get(workload.workload_id),
            }
        status = "failed" if failures else "complete"
        summary = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "method_id": config.method_id,
            "variant_id": config.variant_id,
            "seed": seed,
            "profile": profile,
            "report_table": PROFILES[profile].report_table,
            "latency_definition": LATENCY_DEFINITION,
            "status": status,
            "reportable": (status == "complete" and not timing_manifest["debug_unpinned"]
                           and not hardware_problems),
            "accuracy_run": str(run_directory),
            "accuracy_run_config_hash": manifest.get("config_hash"),
            "accuracy_config_hash": timing_manifest["accuracy_config_hash"],
            "config_drift": drift,
            "hardware_class": hardware,
            "hardware_violations": hardware_problems,
            "estimate_consistency": consistency,
            "estimate_drift_allowed": bool(allow_estimate_drift),
            "warmup_passes": int(config.timing.get("warmup_passes", 1)),
            "repetitions": int(config.timing.get("repetitions", 10)),
            "workloads": workloads,
            "adapter_detail": outcome.detail,
            "failures": failures,
        }
        write_json(directory / "timing_summary.json", summary)
        timing_manifest["status"] = status
        timing_manifest["failures"] = failures
    except Exception as exc:
        timing_manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        timing_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "timing_manifest.json", timing_manifest)
    if failures:
        raise TimingProtocolError(f"timing run {directory} failed: " + "; ".join(failures))
    return directory


def latest_reportable_timing(run_directory: Path, profile: str) -> Path | None:
    root = Path(run_directory) / "timing" / profile
    if not root.exists():
        return None
    for candidate in sorted(root.iterdir(), reverse=True):
        summary = candidate / "timing_summary.json"
        manifest = candidate / "timing_manifest.json"
        if not summary.exists() or not manifest.exists():
            continue
        if json.loads(manifest.read_text(encoding="utf-8")).get("status") != "complete":
            continue
        if json.loads(summary.read_text(encoding="utf-8")).get("reportable"):
            return candidate
    return None


def load_timing_summaries(run_directory: Path) -> dict[str, dict[str, Any]]:
    """Latest reportable timing summary per profile of one accuracy run."""
    root = Path(run_directory) / "timing"
    results: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return results
    for profile_directory in sorted(root.iterdir()):
        if profile_directory.name not in PROFILES:
            continue
        latest = latest_reportable_timing(run_directory, profile_directory.name)
        if latest is not None:
            summary = json.loads((latest / "timing_summary.json").read_text(encoding="utf-8"))
            summary["timing_directory"] = str(latest)
            results[profile_directory.name] = summary
    return results

