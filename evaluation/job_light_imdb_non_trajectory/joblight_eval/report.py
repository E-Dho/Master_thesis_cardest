from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import write_json


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


def aggregate_runs(run_directories: Iterable[Path], output_directory: Path) -> dict[str, Any]:
    directories = tuple(Path(path).resolve() for path in run_directories)
    if not directories:
        raise ValueError("at least one run directory is required")
    summaries = [_load_complete_summary(path) for path in directories]
    identities = {
        (item["experiment_id"], item["method_id"], item["variant_id"], item["config_hash"])
        for item in summaries
    }
    if len(identities) != 1:
        raise ValueError("only runs with identical method, variant, and config hash may aggregate")
    seeds = [int(item["seed"]) for item in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seed in aggregate input")

    experiment_id, method_id, variant_id, config_hash = identities.pop()
    workload_ids = set(summaries[0]["workloads"])
    if any(set(item["workloads"]) != workload_ids for item in summaries):
        raise ValueError("seed summaries have different workload coverage")
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "method_id": method_id,
        "variant_id": variant_id,
        "display_name": summaries[0]["display_name"],
        "config_hash": config_hash,
        "seeds": sorted(seeds),
        "seed_count": len(seeds),
        "runs": [str(path) for path in directories],
        "workloads": {},
    }
    rows: list[dict[str, Any]] = []
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
        aggregate["workloads"][workload_id] = workload_result
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "method_id": method_id,
            "variant_id": variant_id,
            "display_name": summaries[0]["display_name"],
            "workload": workload_id,
            "seed_count": len(seeds),
            "coverage_complete_all_seeds": workload_result["coverage_complete_all_seeds"],
        }
        for name, value in workload_result["metrics"].items():
            row[f"{name}.mean"] = value["mean"]
            row[f"{name}.std"] = value["std"]
        rows.append(row)

    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "comparison.json", aggregate)
    _write_csv(output_directory / "comparison.csv", rows)
    (output_directory / "comparison.md").write_text(
        _markdown_table(aggregate), encoding="utf-8"
    )
    return aggregate


def compare_aggregates(
    aggregate_paths: Iterable[Path], output_directory: Path
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
                "workload": workload_id,
                "seed_count": aggregate["seed_count"],
                "coverage_complete_all_seeds": workload["coverage_complete_all_seeds"],
            }
            for name, metric in workload["metrics"].items():
                row[f"{name}.mean"] = metric["mean"]
                row[f"{name}.std"] = metric["std"]
            rows.append(row)
    comparison = {
        "schema_version": 1,
        "aggregate_count": len(aggregates),
        "inputs": [str(path) for path in paths],
        "results": rows,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "comparison.json", comparison)
    _write_csv(output_directory / "comparison.csv", rows)
    (output_directory / "comparison.md").write_text(
        _multi_method_markdown(rows), encoding="utf-8"
    )
    return comparison


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {aggregate['display_name']}",
        "",
        "Values are mean +/- sample standard deviation across seed-level summaries.",
    ]
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
        lines.extend(
            [
                "",
                "| Inference mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (queries/s) |",
                "| ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    _format_mean_std(metrics[name])
                    for name in (
                        "inference.mean_ms",
                        "inference.p50_ms",
                        "inference.p95_ms",
                        "inference.p99_ms",
                        "inference.throughput_queries_per_second",
                    )
                )
                + " |",
            ]
        )
    return "\n".join(lines) + "\n"


def _format_mean_std(metric: dict[str, float | None]) -> str:
    if metric["mean"] is None:
        return "N/A"
    return f"{metric['mean']:.4g} +/- {metric['std']:.3g}"


def _multi_method_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# JOB-light method comparison",
        "",
        "Values are mean +/- sample standard deviation across seed-level summaries.",
        "",
        "## Q-error",
        "",
        "| Method | Variant | Workload | Coverage | Family | p50 | p90 | p95 | p99 | max |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def metric(name: str) -> str:
            return _format_mean_std(
                {"mean": row[f"{name}.mean"], "std": row[f"{name}.std"]}
            )

        identity = [
            str(row["method_id"]),
            str(row["variant_id"]),
            str(row["workload"]),
            "complete" if row["coverage_complete_all_seeds"] else "incomplete",
        ]
        for label, family in Q_ERROR_REPORT_FAMILIES:
            values = [metric(f"{family}.{name}") for name in Q_ERROR_PERCENTILES]
            lines.append("| " + " | ".join([*identity, label, *values]) + " |")
    lines.extend(
        [
            "",
            "## Inference",
            "",
            "| Method | Variant | Workload | Coverage | Mean (ms) | p50 (ms) | "
            "p95 (ms) | p99 (ms) | Throughput (queries/s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        def metric(name: str) -> str:
            return _format_mean_std(
                {"mean": row[f"{name}.mean"], "std": row[f"{name}.std"]}
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method_id"]),
                    str(row["variant_id"]),
                    str(row["workload"]),
                    "complete" if row["coverage_complete_all_seeds"] else "incomplete",
                    metric("inference.mean_ms"),
                    metric("inference.p50_ms"),
                    metric("inference.p95_ms"),
                    metric("inference.p99_ms"),
                    metric("inference.throughput_queries_per_second"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
