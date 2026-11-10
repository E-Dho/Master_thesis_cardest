#!/usr/bin/env python3
"""Refuse table inputs that do not match their declared evaluation configs.

The final comparison is intentionally assembled from independently produced
aggregate files.  This audit makes the selection explicit before publication:
every selected seed run must have the aggregate's current accuracy identity and
must be the newest matching completed run for that seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import (
    accuracy_config_hash,
    load_experiment_config,
)


SCRIPT_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_accuracy_hash(run: Path) -> str:
    resolved = run / "resolved_config.json"
    if resolved.exists():
        return accuracy_config_hash(_read_json(resolved))
    summary = _read_json(run / "summary.json")
    recorded = summary.get("accuracy_config_hash")
    if recorded:
        return str(recorded)
    return f"full:{summary['config_hash']}"


def _config_for_aggregate(aggregate: dict[str, Any], runs: list[Path]) -> Path:
    paths = {
        Path(str(_read_json(run / "run_manifest.json").get("config_path")))
        for run in runs
        if _read_json(run / "run_manifest.json").get("config_path")
    }
    existing = [path for path in paths if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise ValueError(f"aggregate has multiple existing config paths: {sorted(map(str, existing))}")

    candidates = []
    for path in sorted((SCRIPT_ROOT / "configs").glob("*.yaml")):
        try:
            config = load_experiment_config(path)
        except (FileNotFoundError, ValueError):
            continue
        if (
            config.experiment_id == aggregate["experiment_id"]
            and config.method_id == aggregate["method_id"]
            and config.variant_id == aggregate["variant_id"]
            and config.accuracy_config_hash == aggregate["accuracy_config_hash"]
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            "could not uniquely resolve config for "
            f"{aggregate['method_id']}/{aggregate['variant_id']}: {candidates}"
        )
    return candidates[0]


def _complete_matching_runs(selected: Path, config_hash: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for run in selected.parent.iterdir():
        manifest_path = run / "run_manifest.json"
        summary_path = run / "summary.json"
        if not (run.is_dir() and manifest_path.exists() and summary_path.exists()):
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            continue
        completed_at = manifest.get("completed_at_utc")
        if not isinstance(completed_at, str):
            continue
        candidates.append({
            "path": str(run),
            "completed_at_utc": completed_at,
            "accuracy_config_hash": _run_accuracy_hash(run),
            "matches_config": _run_accuracy_hash(run) == config_hash,
            "selected": run == selected,
        })
    return sorted(candidates, key=lambda item: (item["completed_at_utc"], item["path"]))


def audit(comparison_path: Path) -> dict[str, Any]:
    comparison = _read_json(comparison_path)
    findings: list[str] = []
    aggregates: list[dict[str, Any]] = []

    row_keys = [
        (row.get("method_id"), row.get("variant_id"), row.get("workload"))
        for row in comparison.get("results", [])
    ]
    duplicates = sorted({key for key in row_keys if row_keys.count(key) > 1})
    if duplicates:
        findings.append(f"duplicate method/variant/workload rows: {duplicates}")

    for raw_input in comparison.get("inputs", []):
        aggregate_path = Path(raw_input)
        if not aggregate_path.exists():
            findings.append(f"aggregate input does not exist: {aggregate_path}")
            continue
        aggregate = _read_json(aggregate_path)
        runs = [Path(path) for path in aggregate.get("runs", [])]
        if not runs:
            findings.append(f"aggregate has no selected runs: {aggregate_path}")
            continue
        try:
            config_path = _config_for_aggregate(aggregate, runs)
            config = load_experiment_config(config_path)
        except (FileNotFoundError, ValueError) as exc:
            findings.append(f"{aggregate_path}: config resolution failed: {exc}")
            continue

        expected_hash = config.accuracy_config_hash
        if aggregate.get("accuracy_config_hash") != expected_hash:
            findings.append(
                f"{aggregate_path}: aggregate accuracy hash differs from config "
                f"({aggregate.get('accuracy_config_hash')} != {expected_hash})"
            )
        if aggregate.get("method_id") == "own_model" and not str(aggregate.get("variant_id", "")).endswith(
            "_corrected_model_code"
        ):
            findings.append(f"{aggregate_path}: non-corrected own-model variant is an input")

        seeds: list[dict[str, Any]] = []
        for run in runs:
            if not run.exists():
                findings.append(f"{aggregate_path}: selected run does not exist: {run}")
                continue
            run_hash = _run_accuracy_hash(run)
            if run_hash != expected_hash:
                findings.append(
                    f"{aggregate_path}: selected run hash differs from config: {run} "
                    f"({run_hash} != {expected_hash})"
                )
            manifest = _read_json(run / "run_manifest.json")
            complete = _complete_matching_runs(run, expected_hash)
            matching = [item for item in complete if item["matches_config"]]
            newest = matching[-1]["path"] if matching else None
            if newest != str(run):
                findings.append(
                    f"{aggregate_path}: seed {manifest.get('seed')} selected run is not newest matching: "
                    f"{run}; newest={newest}"
                )
            seeds.append({
                "seed": manifest.get("seed"),
                "selected_run": str(run),
                "complete_runs": complete,
                "newest_matching_run": newest,
            })
        aggregates.append({
            "aggregate": str(aggregate_path),
            "config": str(config_path),
            "accuracy_config_hash": expected_hash,
            "method_id": aggregate.get("method_id"),
            "variant_id": aggregate.get("variant_id"),
            "seeds": seeds,
        })

    return {
        "comparison": str(comparison_path),
        "input_count": len(comparison.get("inputs", [])),
        "aggregates": aggregates,
        "findings": findings,
        "passed": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.comparison)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
