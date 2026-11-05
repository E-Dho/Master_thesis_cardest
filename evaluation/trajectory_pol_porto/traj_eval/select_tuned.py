"""Choose tuned settings from validation sweeps (``tune.py``'s ``tuning.csv`` files).

Two picks per method (decided 2026-11-05):

- ``budget``: the most accurate setting whose ``serialized_bytes`` fit the 1x
  memory budget (the own model's inference-only size);
- ``best``: the most accurate setting regardless of size, to show how much
  more the method needs to reach its best and how it compares at equal size.

"Most accurate" = lowest smoothed Q-error p90 on the validation multi
queries, ties broken by p99, then p50, then the smaller model. All rows of one
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

NUMERIC = ("query_count", "scored_count", "coverage_fraction", "q_p50", "q_p90", "q_p95", "q_p99", "q_max",
           "zero_estimate_count", "serialized_bytes", "theoretical_bytes", "build_seconds")


def read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in NUMERIC:
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                row["parameters"] = json.loads(row["parameters"])
                row["source"] = str(path)
                key = json.dumps(row["parameters"], sort_keys=True)
                if key in rows and rows[key]["q_p90"] != row["q_p90"]:
                    raise ValueError(f"setting {key} appears twice with different results")
                rows[key] = row
    return list(rows.values())


def _rank(row: Dict[str, Any]):
    return (row["q_p90"], row["q_p99"], row["q_p50"], row["serialized_bytes"])


def select(rows: Sequence[Dict[str, Any]], budget_bytes: Optional[float]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("no tuning rows")
    coverage = {row["scored_count"] for row in rows}
    if len(coverage) != 1:
        raise ValueError(f"settings answer different query sets: scored counts {sorted(coverage)}")
    best = min(rows, key=_rank)
    fitting = [row for row in rows if budget_bytes is not None and row["serialized_bytes"] <= budget_bytes]
    budget = min(fitting, key=_rank) if fitting else None
    smallest = min(rows, key=lambda row: row["serialized_bytes"])

    def brief(row):
        if row is None:
            return None
        return {key: row[key] for key in ("parameters", "variant_id", "scored_count", "coverage_fraction", "q_p50",
                                          "q_p90", "q_p99", "q_max", "zero_estimate_count", "serialized_bytes",
                                          "theoretical_bytes", "source")}

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
    out: Dict[str, Any] = {"rule": "min validation smoothed q_p90, then q_p99, q_p50, serialized_bytes",
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
