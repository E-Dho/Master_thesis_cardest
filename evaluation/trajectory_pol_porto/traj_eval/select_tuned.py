"""Choose tuned settings from validation sweeps (``tune.py``'s ``tuning.csv`` files).

Two picks per method (decided 2026-11-05):

- ``budget``: the most accurate setting whose ``serialized_bytes`` fit the 1x
  memory budget (the own model's inference-only size);
- ``best``: the most accurate setting regardless of size, to show how much
  more the method needs to reach its best and how it compares at equal size.

"Most accurate" = lowest geometric mean of the smoothed Q-error on the
validation multi queries (decided 2026-11-05: one number over the whole
distribution, so a heavy tail counts; p90 alone picked a TraHist setting with
p99 11,111), ties broken by p90, p99, then the smaller model. Files written
before ``q_gmean`` existed get it from the run's ``predictions.csv``. All rows of one
method must have the same coverage (the settings do not change which queries
are supported); a method with no setting under the budget gets ``None`` and
the smallest setting is reported for reference.

    python -m evaluation.trajectory_pol_porto.traj_eval.select_tuned \\
        --method trahist_temporal=.../tuning/trahist_temporal/tuning.csv,.../buckets_ext/trahist_temporal/tuning.csv \\
        --method case_plus=.../tuning/case_plus/tuning.csv \\
        --budget-bytes 21400000 --output selection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .tune import COVERAGE_VERSION, coverage_sha256, geometric_mean_q

NUMERIC = ("query_count", "scored_count", "coverage_fraction", "q_p50", "q_p90", "q_p95", "q_p99", "q_max", "q_gmean",
           "zero_estimate_count", "serialized_bytes", "theoretical_bytes", "build_seconds")


COMPARED = NUMERIC + ("variant_id", "coverage_sha256")


def read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Rows of all files, one per setting; a setting repeated with any differing field is an error.

    ``coverage_sha256`` (which queries a run answered) is taken from the file
    only in the current format (``cov2:`` prefix, bound to the benchmark);
    otherwise - no column, or a value from an older version - it is recomputed
    from the run's ``predictions.csv`` and ``run_manifest.json``. A row whose
    coverage cannot be established is refused.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in NUMERIC:
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                row["parameters"] = json.loads(row["parameters"])
                row["source"] = str(path)
                if not str(row.get("coverage_sha256") or "").startswith(f"{COVERAGE_VERSION}:"):
                    # missing, or an older fingerprint format: recompute from the run directory
                    run = Path(row.get("run_directory") or "")
                    if not ((run / "predictions.csv").is_file() and (run / "run_manifest.json").is_file()):
                        raise ValueError(f"{path}: no coverage_sha256 and no predictions.csv / run_manifest.json "
                                         f"for {row['parameters']} (run directory {run})")
                    row["coverage_sha256"] = coverage_sha256(run)
                if row.get("q_gmean") in (None, ""):  # files written before the column existed
                    run = Path(row.get("run_directory") or "")
                    if not (run / "predictions.csv").is_file():
                        raise ValueError(f"{path}: no q_gmean and no predictions.csv for {row['parameters']} "
                                         f"(run directory {run})")
                    row["q_gmean"] = geometric_mean_q(run)
                key = json.dumps(row["parameters"], sort_keys=True)
                if key in rows:
                    differing = [k for k in COMPARED if rows[key].get(k) != row.get(k)]
                    if differing:
                        raise ValueError(f"setting {key} appears twice with different {differing}")
                rows[key] = row
    return list(rows.values())


def _rank(row: Dict[str, Any]):
    return (row["q_gmean"], row["q_p90"], row["q_p99"], row["serialized_bytes"])


def select(rows: Sequence[Dict[str, Any]], budget_bytes: Optional[float]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("no tuning rows")
    coverage = {(row["scored_count"], row["coverage_sha256"]) for row in rows}
    if len(coverage) != 1:
        raise ValueError(f"settings answer different query sets: {sorted(coverage)}")
    best = min(rows, key=_rank)
    fitting = [row for row in rows if budget_bytes is not None and row["serialized_bytes"] <= budget_bytes]
    budget = min(fitting, key=_rank) if fitting else None
    smallest = min(rows, key=lambda row: row["serialized_bytes"])

    def brief(row):
        if row is None:
            return None
        return {key: row[key] for key in ("parameters", "variant_id", "scored_count", "coverage_fraction", "q_p50",
                                          "q_p90", "q_p99", "q_max", "q_gmean", "zero_estimate_count",
                                          "serialized_bytes", "theoretical_bytes", "coverage_sha256", "source")}

    return {"settings": len(rows), "budget_bytes": budget_bytes, "budget": brief(budget), "best": brief(best),
            "smallest": brief(smallest), "best_over_budget_factor":
                (best["serialized_bytes"] / budget_bytes) if budget_bytes else None}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", required=True,
                        help="name=tuning.csv[,tuning.csv...] (repeatable)")
    parser.add_argument("--budget-bytes", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    out: Dict[str, Any] = {"rule": "min validation geometric-mean smoothed Q-error, then q_p90, q_p99, serialized_bytes",
                           "budget_bytes": args.budget_bytes, "methods": {}}
    for item in args.method:
        name, _, paths = item.partition("=")
        out["methods"][name] = select(read_rows(Path(p) for p in paths.split(",") if p), args.budget_bytes)
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
