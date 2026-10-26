#!/usr/bin/env python3
"""Recompute existing aggregates and comparisons and diff them with the stored ones.

Run this on the cluster after a change to the evaluation core, against the
finished IMDB results. For every ``--aggregate`` directory it re-aggregates
the run directories listed in its ``comparison.json`` into a temporary
directory; with ``--comparison`` it also re-runs ``compare`` over the
recomputed aggregates. Every JSON, CSV and Markdown output must match the
stored one after masking timestamps, durations and host names; otherwise the
script lists the differing files and exits 1.

    python evaluation/eval_core/scripts/check_aggregate_regression.py \
        --aggregate RESULTS/joblight_phase1/postgresql/pg16_10_default_stats100/aggregate \
        --aggregate RESULTS/joblight_phase1/mscn/published_job_light/aggregate \
        --comparison RESULTS/joblight_phase1/final_comparison

The stored outputs must have been written from the same run directories (the
script does not rerun ``summarize``). Pass ``--allow-hardware-mismatch`` if the
stored outputs were written with it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from evaluation.eval_core.report import aggregate_runs, compare_aggregates  # noqa: E402

VOLATILE_KEY = re.compile(
    r"(_utc|timestamp|hostname|elapsed|wall_seconds|_seconds$|generated|started|completed)", re.I
)


def _normalize(value, mapping):
    if isinstance(value, dict):
        return {k: ("<masked>" if VOLATILE_KEY.search(k) else _normalize(v, mapping)) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v, mapping) for v in value]
    if isinstance(value, str):
        for old, new in mapping.items():
            value = value.replace(old, new)
    return value


def _diff(stored: Path, fresh: Path, mapping) -> list[str]:
    problems = []
    for path in sorted(stored.iterdir()):
        if path.suffix not in (".json", ".csv", ".md"):
            continue
        other = fresh / path.name
        if not other.exists():
            problems.append(f"{path} has no recomputed counterpart")
            continue
        if path.suffix == ".json":
            a = _normalize(json.loads(path.read_text(encoding="utf-8")), mapping)
            b = _normalize(json.loads(other.read_text(encoding="utf-8")), mapping)
            same = a == b
        else:
            text_a, text_b = path.read_text(encoding="utf-8"), other.read_text(encoding="utf-8")
            for old, new in mapping.items():
                text_a, text_b = text_a.replace(old, new), text_b.replace(old, new)
            same = text_a == text_b
        if not same:
            problems.append(f"{path} differs from {other}")
    for path in sorted(fresh.iterdir()):
        if not (stored / path.name).exists():
            problems.append(f"recomputed {path.name} is new (not in {stored})")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregate", action="append", default=[], type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--allow-hardware-mismatch", action="store_true")
    args = parser.parse_args(argv)
    if not args.aggregate:
        parser.error("at least one --aggregate directory is required")
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh_root = Path(tmp)
        fresh_jsons = []
        mapping = {}
        for index, stored in enumerate(args.aggregate):
            stored = stored.resolve()
            runs = [Path(p) for p in json.loads((stored / "comparison.json").read_text())["runs"]]
            fresh = fresh_root / f"aggregate_{index}"
            aggregate_runs(runs, fresh, allow_hardware_mismatch=args.allow_hardware_mismatch)
            mapping[str(fresh / "comparison.json")] = str(stored / "comparison.json")
            problems += _diff(stored, fresh, {str(fresh): str(stored)})
            fresh_jsons.append(fresh / "comparison.json")
            print(f"aggregate {stored}: {'ok' if not problems else 'see below'}")
        if args.comparison is not None:
            fresh = fresh_root / "comparison"
            compare_aggregates(fresh_jsons, fresh, allow_hardware_mismatch=args.allow_hardware_mismatch)
            problems += _diff(args.comparison.resolve(), fresh, mapping)
    for problem in problems:
        print("DIFF", problem)
    print("identical" if not problems else f"{len(problems)} difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
