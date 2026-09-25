from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import write_json
from .config import accuracy_config_hash, strip_paths
from .timing import config_drift, load_timing_summaries
from .timing_guard import PROFILES


HEADLINE_METRICS = (
    "raw_q_error.p50",
    "raw_q_error.p90",
    "raw_q_error.p95",
    "raw_q_error.p99",
    "raw_q_error.max",
    "raw_q_error_true_positive.p50",
    "raw_q_error_true_positive.p90",
    "raw_q_error_true_positive.p95",
    "raw_q_error_true_positive.p99",
    "raw_q_error_true_positive.max",
    "smoothed_q_error_true_zero.p50",
    "smoothed_q_error_true_zero.p90",
    "smoothed_q_error_true_zero.p95",
    "smoothed_q_error_true_zero.p99",
    "smoothed_q_error_true_zero.max",
    "smoothed_q_error.p50",
    "smoothed_q_error.p90",
    "smoothed_q_error.p95",
    "smoothed_q_error.p99",
    "smoothed_q_error.max",
)

Q_ERROR_REPORT_FAMILIES = (
    ("Raw, all scored queries", "raw_q_error"),
    ("Raw, true cardinality > 0", "raw_q_error_true_positive"),
    ("Smoothed, true cardinality = 0", "smoothed_q_error_true_zero"),
    ("Smoothed, all scored queries", "smoothed_q_error"),
)
Q_ERROR_PERCENTILES = ("p50", "p90", "p95", "p99", "max")
LATENCY_METRICS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_queries_per_second")
MODEL_CORE_METRICS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
EVALUATE_STAGE_TIMING_NOTE = (
    "Evaluate-stage timing is not profile-controlled (thread limits, pinning, and "
    "hardware class are not enforced); report the standardized profile tables instead."
)


def aggregate_runs(
    run_directories: Iterable[Path],
    output_directory: Path,
    *,
    allow_hardware_mismatch: bool = False,
) -> dict[str, Any]:
    directories = tuple(Path(path).resolve() for path in run_directories)
    if not directories:
        raise ValueError("at least one run directory is required")
    summaries = [_load_complete_summary(path) for path in directories]
    resolved = [_resolved_config(path) for path in directories]
    accuracy_hashes = [
        _accuracy_hash(path, summary, raw)
        for path, summary, raw in zip(directories, summaries, resolved)
    ]
    identities = {
        (item["experiment_id"], item["method_id"], item["variant_id"], accuracy_hash)
        for item, accuracy_hash in zip(summaries, accuracy_hashes)
    }
    if len(identities) != 1:
        raise ValueError(_identity_mismatch_message(directories, summaries, resolved, accuracy_hashes))
    seeds = [int(item["seed"]) for item in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed in aggregate input")

    experiment_id, method_id, variant_id, accuracy_hash = identities.pop()
    config_hashes = sorted({str(item["config_hash"]) for item in summaries})
    timing_only_differences = sorted({
        path
        for raw in resolved[1:]
        if raw is not None and resolved[0] is not None
        for path in config_drift(resolved[0], raw)
    })
    timings = [load_timing_summaries(path) for path in directories]
    all_profiles = sorted({profile for timing in timings for profile in timing})
    complete_profiles = [
        profile for profile in all_profiles if all(profile in timing for timing in timings)
    ]
    workload_ids = set(summaries[0]["workloads"])
    if any(set(item["workloads"]) != workload_ids for item in summaries):
        raise ValueError("seed summaries have different workload coverage")
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "method_id": method_id,
        "variant_id": variant_id,
        "display_name": summaries[0]["display_name"],
        "protocol": summaries[0].get("protocol", "native"),
        "adaptation": summaries[0].get("adaptation", ""),
        # Seeds aggregate on the accuracy-only hash; full hashes may differ in
        # timing-only keys (TIMING_ONLY_CONFIG_PATHS), which are listed here.
        "accuracy_config_hash": accuracy_hash,
        "config_hash": config_hashes[0] if len(config_hashes) == 1 else None,
        "config_hashes": config_hashes,
        "timing_only_config_differences": timing_only_differences,
        "seeds": sorted(seeds),
        "seed_count": len(seeds),
        "runs": [str(path) for path in directories],
        "standardized_timing_profiles": complete_profiles,
        "incomplete_timing_profiles": {
            profile: sorted(
                int(summary["seed"])
                for summary, timing in zip(summaries, timings)
                if profile in timing
            )
            for profile in all_profiles
            if profile not in complete_profiles
        },
        "workloads": {},
    }
    rows: list[dict[str, Any]] = []
    seed_hardware_mismatches: dict[str, list[dict[str, Any]]] = {}
    for workload_id in sorted(workload_ids):
        workload_result: dict[str, Any] = {
            "coverage_complete_all_seeds": True,
            "metrics": {},
        }
        for summary in summaries:
            accuracy = summary["workloads"][workload_id]["accuracy"]
            workload_result["coverage_complete_all_seeds"] &= (
                accuracy["scored_query_count"] == accuracy["query_count"]
            )
        for path in HEADLINE_METRICS:
            family, name = path.split(".")
            values = [
                summary["workloads"][workload_id]["accuracy"][family][name]
                for summary in summaries
            ]
            workload_result["metrics"][path] = _mean_std(values)
        for name in (
            "true_zero_matching_count",
            "estimate_lt_1_count",
            "estimate_lt_0_1_count",
            "estimate_lt_0_01_count",
            "zero_estimate_count",
            "coverage_fraction",
        ):
            values = [
                summary["workloads"][workload_id]["accuracy"][name]
                for summary in summaries
            ]
            workload_result["metrics"][name] = _mean_std(values)
        for name in (
            "mean_ms",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "throughput_queries_per_second",
        ):
            values = [
                summary["workloads"][workload_id]["inference"][name]
                for summary in summaries
            ]
            workload_result["metrics"][f"inference.{name}"] = _mean_std(values)
        timing_protocol = summaries[0]["workloads"][workload_id].get(
            "timing_protocol", {}
        )
        profile_sets = [
            set(_inference_profiles(summary, workload_id)) for summary in summaries
        ]
        if any(devices != profile_sets[0] for devices in profile_sets[1:]):
            raise ValueError("seed summaries have different inference timing profiles")
        workload_result["timing_protocol"] = timing_protocol
        workload_result["inference_profiles"] = {}
        for device in sorted(profile_sets[0]):
            profiles = [
                _inference_profiles(summary, workload_id)[device]
                for summary in summaries
            ]
            workload_result["inference_profiles"][device] = {
                "device_names": sorted(
                    {
                        name
                        for profile in profiles
                        for name in profile.get("device_names", [])
                    }
                ),
                "metrics": {
                    name: _mean_std([profile[name] for profile in profiles])
                    for name in (
                        "mean_ms",
                        "p50_ms",
                        "p95_ms",
                        "p99_ms",
                        "throughput_queries_per_second",
                    )
                },
            }
        workload_result["standardized_timing"] = {
            profile: _aggregate_profile(
                [timing[profile] for timing in timings], workload_id, profile
            )
            for profile in complete_profiles
        }
        aggregate["workloads"][workload_id] = workload_result
        for profile, result in workload_result["standardized_timing"].items():
            if result.get("available") and not result["hardware_consistent"]:
                seed_hardware_mismatches.setdefault(profile, []).append(
                    {"workload": workload_id, "cpu_models": result["cpu_models"],
                     "gpu_names": result["gpu_names"], "declared_hardware": result["declared_hardware"]}
                )
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "method_id": method_id,
            "variant_id": variant_id,
            "display_name": summaries[0]["display_name"],
            "protocol": aggregate["protocol"],
            "workload": workload_id,
            "seed_count": len(seeds),
            "coverage_complete_all_seeds": workload_result["coverage_complete_all_seeds"],
            "inference_device": timing_protocol.get("primary_device", timing_protocol.get("device", "unspecified")),
            "published_reference_hardware": timing_protocol.get("published_reference_hardware", ""),
            "actual_hardware": ", ".join(
                workload_result["inference_profiles"]
                .get(timing_protocol.get("primary_device", timing_protocol.get("device", "")), {})
                .get("device_names", [])
            ),
            "inference_profiles_json": json.dumps(
                workload_result["inference_profiles"], sort_keys=True
            ),
        }
        for name, value in workload_result["metrics"].items():
            row[f"{name}.mean"] = value["mean"]
            row[f"{name}.std"] = value["std"]
        row.update(_timing_columns(workload_result["standardized_timing"]))
        rows.append(row)

    if seed_hardware_mismatches and not allow_hardware_mismatch:
        raise ValueError(
            "seeds were timed on different hardware within one profile: "
            + json.dumps(seed_hardware_mismatches, sort_keys=True)
            + "; rerun timing on the declared class or pass --allow-hardware-mismatch"
        )
    aggregate["hardware_mismatches"] = seed_hardware_mismatches
    aggregate["hardware_mismatch_allowed"] = bool(seed_hardware_mismatches)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "comparison.json", aggregate)
    _write_csv(output_directory / "comparison.csv", rows)
    (output_directory / "comparison.md").write_text(
        _markdown_table(aggregate), encoding="utf-8"
    )
    return aggregate


def compare_aggregates(
    aggregate_paths: Iterable[Path],
    output_directory: Path,
    *,
    allow_hardware_mismatch: bool = False,
) -> dict[str, Any]:
    paths = tuple(Path(path).resolve() for path in aggregate_paths)
    if not paths:
        raise ValueError("at least one aggregate JSON is required")
    aggregates = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows: list[dict[str, Any]] = []
    for aggregate in aggregates:
        for workload_id, workload in sorted(aggregate["workloads"].items()):
            row: dict[str, Any] = {
                "experiment_id": aggregate["experiment_id"],
                "method_id": aggregate["method_id"],
                "variant_id": aggregate["variant_id"],
                "display_name": aggregate["display_name"],
                "protocol": aggregate.get("protocol", "native"),
                "workload": workload_id,
                "seed_count": aggregate["seed_count"],
                "coverage_complete_all_seeds": workload["coverage_complete_all_seeds"],
                "inference_device": workload.get("timing_protocol", {}).get(
                    "primary_device",
                    workload.get("timing_protocol", {}).get("device", "unspecified"),
                ),
                "published_reference_hardware": workload.get(
                    "timing_protocol", {}
                ).get("published_reference_hardware", ""),
                "inference_profiles": workload.get("inference_profiles", {}),
            }
            for name, metric in workload["metrics"].items():
                row[f"{name}.mean"] = metric["mean"]
                row[f"{name}.std"] = metric["std"]
            primary_profile = row["inference_profiles"].get(
                row["inference_device"], {}
            )
            row["actual_hardware"] = ", ".join(
                primary_profile.get("device_names", [])
            )
            row["inference_profiles_json"] = json.dumps(
                row["inference_profiles"], sort_keys=True
            )
            row["standardized_timing"] = workload.get("standardized_timing", {})
            row.update(_timing_columns(row["standardized_timing"]))
            rows.append(row)
    mismatches = hardware_mismatches(rows)
    if mismatches and not allow_hardware_mismatch:
        raise ValueError(
            "standardized latency of different methods was measured on different hardware: "
            + json.dumps(mismatches, sort_keys=True)
            + "; time every method on the declared hardware class "
            "(configs/timing_hardware.json) or pass --allow-hardware-mismatch "
            "to write a report that marks these tables as not comparable"
        )
    comparison = {
        "schema_version": 1,
        "aggregate_count": len(aggregates),
        "inputs": [str(path) for path in paths],
        "hardware_mismatches": mismatches,
        "hardware_mismatch_allowed": bool(mismatches),
        "results": rows,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "comparison.json", comparison)
    _write_csv(output_directory / "comparison.csv", rows)
    (output_directory / "comparison.md").write_text(
        _multi_method_markdown(rows, mismatches), encoding="utf-8"
    )
    return comparison


def hardware_mismatches(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Per profile: observed hardware -> rows, when rows do not share one class."""
    result: dict[str, dict[str, list[str]]] = {}
    for profile in PROFILES:
        classes: dict[str, list[str]] = {}
        for row in rows:
            timing = (row.get("standardized_timing") or {}).get(profile)
            if not (timing and timing.get("available")):
                continue
            key = "; ".join(timing["cpu_models"] + timing["gpu_names"]) or "unknown"
            if not timing.get("hardware_consistent", False):
                key += " (MIXED across seeds)"
            classes.setdefault(key, []).append(
                f"{row['method_id']}/{row['variant_id']}/{row['workload']}"
            )
        if len(classes) > 1 or any(key.endswith("(MIXED across seeds)") for key in classes):
            result[profile] = classes
    return result


def discover_latest_complete_runs(config_root: Path, seeds: Iterable[int]) -> list[Path]:
    result: list[Path] = []
    for seed in seeds:
        parent = config_root / f"seed_{seed}"
        candidates = sorted(parent.glob("*"), reverse=True) if parent.exists() else []
        selected = next((path for path in candidates if _is_complete(path)), None)
        if selected is None:
            raise FileNotFoundError(f"no complete run found under {parent}")
        result.append(selected)
    return result


def _resolved_config(path: Path) -> dict[str, Any] | None:
    resolved = path / "resolved_config.json"
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _accuracy_hash(path: Path, summary: dict[str, Any], raw: dict[str, Any] | None) -> str:
    """Accuracy-only config hash of a run, recomputed from its resolved config.

    Recomputing (rather than trusting the recorded value) applies the current
    TIMING_ONLY_CONFIG_PATHS uniformly to runs created before the hash existed.
    """
    if raw is not None:
        return accuracy_config_hash(raw)
    recorded = summary.get("accuracy_config_hash")
    if recorded:
        return str(recorded)
    # Neither available: fall back to the full hash, i.e. the strict pre-hash rule.
    return f"full:{summary['config_hash']}"


def _identity_mismatch_message(
    directories: tuple[Path, ...],
    summaries: list[dict[str, Any]],
    resolved: list[dict[str, Any] | None],
    accuracy_hashes: list[str],
) -> str:
    lines = ["only runs with identical method, variant, and accuracy config hash may aggregate"]
    reference = resolved[0]
    for path, summary, raw, accuracy_hash in zip(directories, summaries, resolved, accuracy_hashes):
        identity = (summary["experiment_id"], summary["method_id"], summary["variant_id"])
        detail = f"  {path}: {identity} accuracy_hash={accuracy_hash[:16]}"
        if reference is not None and raw is not None and accuracy_hash != accuracy_hashes[0]:
            differing = config_drift(strip_paths(reference), strip_paths(raw))
            detail += " non-timing differences to the first run: " + ", ".join(differing[:20])
        lines.append(detail)
    return "\n".join(lines)


def _load_complete_summary(path: Path) -> dict[str, Any]:
    if not _is_complete(path):
        raise ValueError(f"run is incomplete: {path}")
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _is_complete(path: Path) -> bool:
    manifest_path = path / "run_manifest.json"
    summary_path = path / "summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("status") == "complete"


def _mean_std(values: list[float | int | None]) -> dict[str, float | None]:
    if any(value is None for value in values):
        return {"mean": None, "std": None}
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
    }


def _inference_profiles(summary: dict[str, Any], workload_id: str) -> dict[str, Any]:
    workload = summary["workloads"][workload_id]
    profiles = workload.get("inference_by_device")
    if profiles:
        return profiles
    protocol = workload.get("timing_protocol", {})
    device = str(protocol.get("primary_device", protocol.get("device", "unspecified")))
    return {device: {**workload["inference"], "device_names": [], "scopes": []}}


def _aggregate_profile(
    summaries: list[dict[str, Any]], workload_id: str, profile: str
) -> dict[str, Any]:
    workloads = [summary["workloads"].get(workload_id) for summary in summaries]
    if any(workload is None for workload in workloads):
        return {"available": False}
    guards = [workload.get("guard") or {} for workload in workloads]
    cpu_models = sorted({guard.get("cpu_model") or "unknown" for guard in guards})
    gpu_names = sorted({name for guard in guards for name in guard.get("gpu_names", [])})
    nodes = sorted({guard.get("node") or "unknown" for guard in guards})
    declared = sorted({
        json.dumps(
            {key: (summary.get("hardware_class") or {}).get(key) for key in ("cpu_model", "gpu_name")},
            sort_keys=True,
        )
        for summary in summaries
    })
    hardware_declared = all(bool(summary.get("hardware_class")) for summary in summaries)
    model_core = _aggregate_model_core(workloads)
    result: dict[str, Any] = {
        "available": True,
        "report_table": PROFILES[profile].report_table,
        "seed_count": len(summaries),
        "metrics": {
            name: _mean_std([workload["inference"].get(name) for workload in workloads])
            for name in LATENCY_METRICS
        },
        "model_core": model_core,
        # Only populated when every timed query of every seed has a model-core
        # observation; partial coverage would silently average a subset.
        "model_core_metrics": model_core["metrics"],
        "scopes": sorted({scope for workload in workloads for scope in workload.get("scopes", [])}),
        "cpu_models": cpu_models,
        "gpu_names": gpu_names,
        "nodes": nodes,
        "declared_hardware": [json.loads(item) for item in declared],
        "hardware_declared": hardware_declared,
        # Every timing run must have passed hardware-class enforcement. Legacy
        # summaries with matching observed names are not equivalent because
        # their allocation was not checked against the declared profile class.
        "hardware_consistent": (
            hardware_declared
            and len(cpu_models) == 1 and "unknown" not in cpu_models
            and len(gpu_names) <= 1 and len(declared) == 1
        ),
        "guard_warning_count": sum(len(guard.get("warnings", [])) for guard in guards),
        "estimate_consistency": [
            {
                "seed": summary["seed"],
                "mode": summary["estimate_consistency"]["mode"],
                "passed": summary["estimate_consistency"]["passed"],
                "max_relative_difference": summary["estimate_consistency"]["max_relative_difference"],
            }
            for summary in summaries
        ],
        "config_drift": sorted({path for summary in summaries for path in summary.get("config_drift", [])}),
        "timing_directories": [summary.get("timing_directory") for summary in summaries],
    }
    return result


def _aggregate_model_core(workloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Model-core latency across seeds, with its query coverage.

    Means are reported only when coverage is complete in every seed (each
    latency observation has a model-core value); otherwise the metric is
    suppressed and only the coverage is shown.
    """
    cores = [workload.get("model_core") or None for workload in workloads]
    latency_counts = [int(workload["inference"].get("observation_count") or 0) for workload in workloads]
    core_counts = [int(core.get("observation_count") or 0) if core else 0 for core in cores]
    coverage = [
        (count / total) if total else 0.0 for count, total in zip(core_counts, latency_counts)
    ]
    exposed = any(core is not None for core in cores)
    complete = exposed and all(
        core is not None and total > 0 and count == total
        for core, count, total in zip(cores, core_counts, latency_counts)
    )
    if not exposed:
        reason = "method exposes no model-core boundary"
    elif not complete:
        reason = f"partial coverage (min {min(coverage):.1%} of timed queries); mean suppressed"
    else:
        reason = None
    return {
        "exposed": exposed,
        "complete": complete,
        "coverage_fraction_min": min(coverage) if coverage else 0.0,
        "coverage_fraction_mean": float(np.mean(coverage)) if coverage else 0.0,
        "observation_counts": core_counts,
        "latency_observation_counts": latency_counts,
        "metrics": (
            {name: _mean_std([core.get(name) for core in cores]) for name in MODEL_CORE_METRICS}
            if complete
            else None
        ),
        "suppressed_reason": reason,
    }


def _model_core_cell(result: dict[str, Any]) -> str:
    core = result.get("model_core")
    if core is None:  # aggregates written before coverage was tracked
        metrics = result.get("model_core_metrics")
        return _format_mean_std(metrics["mean_ms"]) + " (coverage unknown)" if metrics else "n/a"
    if core["complete"]:
        return _format_mean_std(core["metrics"]["mean_ms"])
    if not core["exposed"]:
        return "n/a"
    return f"suppressed: coverage {core['coverage_fraction_min']:.1%}"


def _timing_columns(standardized: dict[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for profile, result in sorted(standardized.items()):
        if not result.get("available"):
            continue
        for name, metric in result["metrics"].items():
            columns[f"timing.{profile}.{name}.mean"] = metric["mean"]
            columns[f"timing.{profile}.{name}.std"] = metric["std"]
        core = result.get("model_core") or {}
        if core.get("exposed"):
            columns[f"timing.{profile}.model_core.complete"] = core["complete"]
            columns[f"timing.{profile}.model_core.coverage_fraction_min"] = core["coverage_fraction_min"]
        if result.get("model_core_metrics"):
            for name, metric in result["model_core_metrics"].items():
                columns[f"timing.{profile}.model_core.{name}.mean"] = metric["mean"]
                columns[f"timing.{profile}.model_core.{name}.std"] = metric["std"]
        columns[f"timing.{profile}.cpu_models"] = "; ".join(result["cpu_models"])
        columns[f"timing.{profile}.gpu_names"] = "; ".join(result["gpu_names"])
        columns[f"timing.{profile}.hardware_consistent"] = result["hardware_consistent"]
    return columns


def _profile_table_lines(entries: list[tuple[list[str], dict[str, Any]]], identity_header: list[str]) -> list[str]:
    header = identity_header + [
        "Hardware", "Mean (ms)", "p50 (ms)", "p95 (ms)", "p99 (ms)",
        "Throughput (queries/s)", "Model-core mean (ms)", "Estimates vs accuracy run",
    ]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(identity_header) + ["---"] + ["---:"] * 6 + ["---"]) + " |"]
    for identity, result in entries:
        hardware = "; ".join(result["cpu_models"] + result["gpu_names"])
        if not result["hardware_consistent"]:
            hardware += " (MIXED)"
        checks = result["estimate_consistency"]
        consistency = (
            f"{checks[0]['mode']}, max rel. diff "
            f"{max(item['max_relative_difference'] for item in checks):.2g}"
            if checks else "n/a"
        )
        lines.append("| " + " | ".join(
            identity
            + [hardware]
            + [_format_mean_std(result["metrics"][name]) for name in LATENCY_METRICS]
            + [_model_core_cell(result), consistency]
        ) + " |")
    return lines


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def _markdown_table(aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {aggregate['display_name']}",
        "",
        f"Protocol: **{aggregate.get('protocol', 'native')}**",
    ]
    if aggregate.get("adaptation"):
        lines.extend(["", f"Adaptation: {aggregate['adaptation']}"])
    lines.extend(["", "Values are mean +/- sample standard deviation across seed-level summaries."])
    lines.extend(["", f"Accuracy config hash: `{str(aggregate.get('accuracy_config_hash', ''))[:16]}`"])
    if aggregate.get("timing_only_config_differences"):
        lines.extend(["", "Seed configs differ only in timing-only keys: "
                      + ", ".join(f"`{path}`" for path in aggregate["timing_only_config_differences"])])
    if aggregate.get("hardware_mismatches"):
        lines.extend(["", "**WARNING:** seeds were timed on different hardware in profiles "
                      + ", ".join(sorted(aggregate["hardware_mismatches"])) + "."])
    for workload_id, workload in aggregate["workloads"].items():
        metrics = workload["metrics"]
        coverage = "complete" if workload["coverage_complete_all_seeds"] else "incomplete"
        lines.extend(
            [
                "",
                f"## {workload_id}",
                "",
                f"Coverage: **{coverage}**",
                "",
                "| Q-error family | p50 | p90 | p95 | p99 | max |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, family in Q_ERROR_REPORT_FAMILIES:
            values = [
                _format_mean_std(metrics[f"{family}.{name}"])
                for name in Q_ERROR_PERCENTILES
            ]
            lines.append("| " + " | ".join([label, *values]) + " |")
        protocol = workload.get("timing_protocol", {})
        primary = protocol.get("primary_device", protocol.get("device", "unspecified"))
        lines.extend(
            [
                "",
                "### Evaluate-stage timing (not profile-controlled)",
                "",
                EVALUATE_STAGE_TIMING_NOTE,
                "",
                f"Primary timing device: **{primary}**",
                "",
                f"Published reference hardware: {protocol.get('published_reference_hardware', 'not specified')}",
                "",
                "| Device | Actual hardware | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (queries/s) |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for device, profile in workload.get("inference_profiles", {}).items():
            profile_metrics = profile["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        device,
                        ", ".join(profile.get("device_names", [])) or "not recorded",
                        *(
                            _format_mean_std(profile_metrics[name])
                            for name in (
                                "mean_ms",
                                "p50_ms",
                                "p95_ms",
                                "p99_ms",
                                "throughput_queries_per_second",
                            )
                        ),
                    ]
                )
                + " |"
            )
        for profile, result in workload.get("standardized_timing", {}).items():
            if not result.get("available"):
                continue
            lines.extend(["", f"### Standardized latency: {result['report_table']} (`{profile}`)", ""])
            lines.extend(_profile_table_lines([([profile], result)], ["Profile"]))
    missing = aggregate.get("incomplete_timing_profiles", {})
    if missing:
        lines.extend(["", "Incomplete timing profiles (not all seeds timed): "
                      + ", ".join(f"{profile} (seeds {seeds})" for profile, seeds in missing.items())])
    return "\n".join(lines) + "\n"


def _format_mean_std(metric: dict[str, float | None]) -> str:
    if metric["mean"] is None:
        return "N/A"
    return f"{metric['mean']:.4g} +/- {metric['std']:.3g}"


def _multi_method_markdown(
    rows: list[dict[str, Any]], mismatches: dict[str, dict[str, list[str]]] | None = None
) -> str:
    mismatches = mismatches or {}
    lines = [
        "# JOB-light method comparison",
        "",
        "Values are mean +/- sample standard deviation across seed-level summaries.",
        "",
        "## Q-error",
        "",
        "Rows with protocol `adapted` are project-modified variants, not the "
        "method's published configuration.",
        "",
        "| Method | Variant | Display name | Protocol | Workload | Coverage | Family | p50 | p90 | p95 | p99 | max |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def metric(name: str) -> str:
            return _format_mean_std(
                {"mean": row[f"{name}.mean"], "std": row[f"{name}.std"]}
            )

        identity = [
            str(row["method_id"]),
            str(row["variant_id"]),
            str(row["display_name"]),
            str(row.get("protocol", "native")),
            str(row["workload"]),
            "complete" if row["coverage_complete_all_seeds"] else "incomplete",
        ]
        for label, family in Q_ERROR_REPORT_FAMILIES:
            values = [metric(f"{family}.{name}") for name in Q_ERROR_PERCENTILES]
            lines.append("| " + " | ".join([*identity, label, *values]) + " |")
    profiles = sorted({profile for row in rows for profile, result in row.get("standardized_timing", {}).items()
                       if result.get("available")},
                      key=lambda name: list(PROFILES).index(name))
    for profile in profiles:
        entries = []
        absent = []
        for row in rows:
            result = row.get("standardized_timing", {}).get(profile)
            identity = [str(row["method_id"]), str(row["variant_id"]), str(row["display_name"]),
                        str(row.get("protocol", "native")), str(row["workload"])]
            if result and result.get("available"):
                entries.append((identity, result))
            else:
                absent.append(" / ".join(identity[:2] + identity[4:]))
        lines.extend(["", f"## Standardized latency: {PROFILES[profile].report_table} (`{profile}`)", "",
                      PROFILES[profile].description + ".", ""])
        if profile in mismatches:
            lines.extend([
                "**WARNING: NOT COMPARABLE.** Rows of this table were measured on different "
                "hardware (report written with --allow-hardware-mismatch):", "",
                *[f"- {hardware}: {', '.join(names)}" for hardware, names in sorted(mismatches[profile].items())],
                "",
            ])
        lines.extend(_profile_table_lines(entries, ["Method", "Variant", "Display name", "Protocol", "Workload"]))
        if absent:
            lines.extend(["", "Not measured under this profile: " + "; ".join(absent)])
    lines.extend(
        [
            "",
            "## Evaluate-stage timing (not profile-controlled)",
            "",
            EVALUATE_STAGE_TIMING_NOTE,
            "",
            "| Method | Variant | Display name | Protocol | Workload | Coverage | Device | Actual hardware | "
            "Published reference | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (queries/s) |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        def metric(name: str) -> str:
            return _format_mean_std(
                {"mean": row[f"{name}.mean"], "std": row[f"{name}.std"]}
            )

        profiles = row.get("inference_profiles", {})
        if not profiles:
            profiles = {
                row.get("inference_device", "unspecified"): {
                    "device_names": [row.get("actual_hardware", "")],
                    "metrics": {
                        name: {
                            "mean": row[f"inference.{name}.mean"],
                            "std": row[f"inference.{name}.std"],
                        }
                        for name in (
                            "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                            "throughput_queries_per_second",
                        )
                    },
                }
            }
        for device, profile in profiles.items():
            profile_metrics = profile["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method_id"]),
                        str(row["variant_id"]),
                        str(row["display_name"]),
                        str(row.get("protocol", "native")),
                        str(row["workload"]),
                        "complete" if row["coverage_complete_all_seeds"] else "incomplete",
                        device,
                        ", ".join(profile.get("device_names", [])) or "not recorded",
                        str(row.get("published_reference_hardware", "")),
                        *(
                            _format_mean_std(profile_metrics[name])
                            for name in (
                                "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                                "throughput_queries_per_second",
                            )
                        ),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"
