#!/usr/bin/env python3
"""Import complete own-model artifacts when only reporting identity changes.

The source and target configurations must be identical after removing the
experiment variant/display fields.  This preserves existing accuracy, timing,
and build artifacts while giving a rule-based reporting variant its own
provenance-correct manifests.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import load_experiment_config


def _without_reporting_identity(raw: dict) -> dict:
    value = copy.deepcopy(raw)
    experiment = value.get("experiment", {})
    for key in ("variant_id", "display_name"):
        experiment.pop(key, None)
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--target-config", required=True, type=Path)
    parser.add_argument("--run-id", default="reused_rule_step11000")
    args = parser.parse_args()
    source = load_experiment_config(args.source_config)
    target = load_experiment_config(args.target_config)
    if _without_reporting_identity(source.raw) != _without_reporting_identity(target.raw):
        raise ValueError("source and target configs differ beyond reporting identity")
    if source.seeds != target.seeds:
        raise ValueError("source and target seed sets differ")
    source_base = source.results_root / source.experiment_id / source.method_id / source.variant_id
    target_base = target.results_root / target.experiment_id / target.method_id / target.variant_id
    for seed in target.seeds:
        candidates = sorted(
            path for path in (source_base / f"seed_{seed}").iterdir()
            if path.is_dir() and (path / "summary.json").exists() and (path / "run_manifest.json").exists()
        )
        if len(candidates) != 1:
            raise ValueError(f"seed {seed} needs exactly one complete source run, found {len(candidates)}")
        destination = target_base / f"seed_{seed}" / f"{args.run_id}-{target.config_hash}"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(candidates[0], destination)
        resolved = target.raw
        _write(destination / "resolved_config.json", resolved)
        manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
        manifest.update({
            "variant_id": target.variant_id,
            "display_name": target.display_name,
            "config_hash": target.config_hash,
            "accuracy_config_hash": target.accuracy_config_hash,
            "config_path": str(target.source_path),
            "run_directory": str(destination),
        })
        _write(destination / "run_manifest.json", manifest)
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        summary.update({
            "variant_id": target.variant_id,
            "display_name": target.display_name,
            "config_hash": target.config_hash,
            "accuracy_config_hash": target.accuracy_config_hash,
        })
        _write(destination / "summary.json", summary)
        for timing_manifest in destination.glob("timing/*/*/timing_manifest.json"):
            timing = json.loads(timing_manifest.read_text(encoding="utf-8"))
            timing.update({
                "accuracy_run": str(destination),
                "accuracy_run_config_hash": target.config_hash,
                "accuracy_config_hash": target.accuracy_config_hash,
                "current_config_hash": target.config_hash,
                "current_accuracy_config_hash": target.accuracy_config_hash,
                "config_path": str(target.source_path),
                "config_drift": [],
            })
            _write(timing_manifest, timing)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
