#!/usr/bin/env python3
"""Write a separate rule-versus-final own-model sensitivity aggregate.

The final-checkpoint evaluations deliberately do not enter ``final_comparison``.
This script reads their immutable per-run summaries alongside the fixed
step-11,000 rule runs and emits JSON plus a compact Markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("p50", "p90", "p99", "max")


def _summary(path: Path, expected_queries: int) -> dict[str, Any]:
    document = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    workloads = document.get("workloads") or {}
    if len(workloads) != 1:
        raise ValueError(f"{path} must contain exactly one workload")
    workload, value = next(iter(workloads.items()))
    accuracy = value.get("accuracy") or {}
    if accuracy.get("query_count") != expected_queries:
        raise ValueError(f"{path} has {accuracy.get('query_count')} queries, expected {expected_queries}")
    if accuracy.get("scored_query_count") != expected_queries:
        raise ValueError(f"{path} has incomplete coverage")
    return {
        "seed": int(document["seed"]),
        "run_directory": str(path),
        "workload": workload,
        "metrics": {name: float(accuracy["smoothed_q_error"][name]) for name in METRICS},
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": statistics.fmean(row["metrics"][name] for row in rows),
            "std": statistics.stdev(row["metrics"][name] for row in rows),
        }
        for name in METRICS
    }


def _render(label: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {label}", "", "| Seed | Smoothed p50 | p90 | p99 | max | Run directory |",
             "|---:|---:|---:|---:|---:|---|"]
    for row in sorted(rows, key=lambda item: item["seed"]):
        metric = row["metrics"]
        lines.append(
            f"| {row['seed']} | {metric['p50']:.6g} | {metric['p90']:.6g} | "
            f"{metric['p99']:.6g} | {metric['max']:.6g} | `{row['run_directory']}` |"
        )
    aggregate = _aggregate(rows)
    lines.append(
        "| mean +/- sample std | "
        + " | ".join(f"{aggregate[name]['mean']:.6g} +/- {aggregate[name]['std']:.6g}" for name in METRICS)
        + " | |"
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-light-rule", action="append", required=True, type=Path)
    parser.add_argument("--job-light-final", action="append", required=True, type=Path)
    parser.add_argument("--ranges-rule", action="append", required=True, type=Path)
    parser.add_argument("--ranges-final", action="append", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    groups = {
        "JOB-light step 11,000 rule": [_summary(path, 70) for path in args.job_light_rule],
        "JOB-light final checkpoints": [_summary(path, 70) for path in args.job_light_final],
        "JOB-light-ranges step 11,000 rule": [_summary(path, 1000) for path in args.ranges_rule],
        "JOB-light-ranges final checkpoints": [_summary(path, 1000) for path in args.ranges_final],
    }
    if any(len(rows) != 3 or {row["seed"] for row in rows} != {0, 1, 2} for rows in groups.values()):
        raise ValueError("each sensitivity group must contain complete seeds 0, 1, and 2")
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    payload = {label: {"per_seed": rows, "aggregate": _aggregate(rows)} for label, rows in groups.items()}
    (output / "sensitivity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Own-Model Final-Checkpoint Sensitivity", "",
             "Separate from `final_comparison`; all metrics are smoothed q-error.", ""]
    for label, rows in groups.items():
        lines.extend(_render(label, rows))
        lines.append("")
    (output / "sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
