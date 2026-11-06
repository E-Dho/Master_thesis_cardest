"""Parameter sweep of a trajectory method on the validation workload (never the test set).

For every combination of ``--grid key=v1,v2,...`` adapter values, the method's
config is re-run (prepare, smoke, build, evaluate, summarize) on the
validation benchmark's multi queries only, under the experiment id
``<experiment>_tuning`` (so tuning runs never mix with the test runs), and one
row per combination is written to ``tuning.csv`` / ``tuning.json``:
coverage, smoothed Q-error p50/p90/p95/p99/max and its geometric mean on the
supported queries (zero estimates included), the zero-estimate count, a fingerprint of the
answered query set (``coverage_sha256``), the
serialized model size and the method's theoretical size, and build seconds.
Choosing a configuration (under the memory budget) is a separate, documented
decision.

    python -m evaluation.trajectory_pol_porto.traj_eval.tune \\
        --config evaluation/trajectory_pol_porto/configs/pol_v3/trahist_temporal.yaml \\
        --validation-jsonl .../qgen_pol_v3_validation_seed20261026/benchmark.jsonl \\
        --expected-queries 1800 \\
        --grid unit_size_m=25,50,100,200 --grid time_buckets=25,50,100,200,400 \\
        --output evaluation/trajectory_pol_porto/results/tuning/trahist_temporal
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.eval_core.config import experiment_config_from_raw, load_raw_config
from evaluation.eval_core.runner import run_seed

COLUMNS = ("variant_id", "parameters", "query_count", "scored_count", "coverage_fraction",
           "q_p50", "q_p90", "q_p95", "q_p99", "q_max", "q_gmean", "zero_estimate_count", "serialized_bytes",
           "theoretical_bytes", "build_seconds", "coverage_sha256", "run_directory")


def _parse_value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def parse_grid(items: Sequence[str]) -> List[Tuple[str, List[Any]]]:
    grid = []
    for item in items:
        key, _, values = item.partition("=")
        if not key or not values:
            raise ValueError(f"--grid expects key=v1,v2,...; got {item!r}")
        grid.append((key.strip(), [_parse_value(v.strip()) for v in values.split(",") if v.strip()]))
    return grid


def _slug(parameters: Dict[str, Any]) -> str:
    text = "__".join(f"{key}{value}" for key, value in parameters.items())
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).replace(".", "p")


def tuning_raw(raw: Dict[str, Any], parameters: Dict[str, Any], validation_jsonl: Path,
               expected_queries: Optional[int]) -> Dict[str, Any]:
    """The method's merged config, re-pointed at the validation multi workload."""
    raw = copy.deepcopy(raw)
    experiment = raw["experiment"]
    experiment["experiment_id"] = f"{experiment['experiment_id']}_tuning"
    experiment["variant_id"] = f"{experiment['variant_id']}__{_slug(parameters)}"
    experiment["seeds"] = [0]
    workloads = raw["workloads"]
    template = next((dict(value) for value in workloads.values()
                     if dict(value.get("select") or {}).get("relation") == "multi"), None)
    if template is None:
        raise ValueError("the config has no multi workload to take the format settings from")
    template["queries_jsonl"] = str(validation_jsonl)
    template["select"] = {"relation": "multi"}
    if expected_queries is None:
        template.pop("expected_query_count", None)
    else:
        template["expected_query_count"] = int(expected_queries)
    raw["workloads"] = {"validation_multi": template}
    raw.setdefault("adapter", {}).update(parameters)
    return raw


BENCHMARK_FIELDS = ("queries_jsonl_sha256", "select", "query_id_rule", "query_count")
# The fingerprint format; values without this prefix (written before it was bound to the benchmark,
# 9612b2f) are not trusted and are recomputed from the run directory.
COVERAGE_VERSION = "cov2"


def coverage_sha256(run: Path, workload: str = "validation_multi") -> str:
    """Fingerprint of which queries a run answered, bound to the benchmark they come from.

    Query ids are positions within the workload selection, so the same ids of
    another benchmark file (or selection) are other queries. The fingerprint
    hashes the workload's identity from ``run_manifest.json`` (benchmark file
    sha256, selection, id rule, query count) followed by the sorted
    ``query_id:status`` lines of ``predictions.csv``.
    """
    import hashlib

    manifest = json.loads((Path(run) / "run_manifest.json").read_text(encoding="utf-8"))
    identity = (manifest.get("workloads") or {}).get(workload) or {}
    missing = [key for key in BENCHMARK_FIELDS if identity.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{run}: run_manifest.json has no {missing} for workload {workload}")
    header = json.dumps({key: identity[key] for key in BENCHMARK_FIELDS}, sort_keys=True)
    lines = []
    with (Path(run) / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["workload"] == workload:
                lines.append((int(row["query_id"]), row["status"]))
    if not lines:
        raise ValueError(f"no {workload} predictions in {run}")
    if len(lines) != int(identity["query_count"]):
        raise ValueError(f"{run}: {len(lines)} predictions for {identity['query_count']} queries")
    text = header + "\n" + "".join(f"{query_id}:{status}\n" for query_id, status in sorted(lines))
    return f"{COVERAGE_VERSION}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def geometric_mean_q(run: Path, workload: str = "validation_multi") -> float:
    """Geometric mean of the smoothed Q-error over the answered (``ok``) queries of a run."""
    import math

    logs = []
    with (Path(run) / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["workload"] == workload and row["status"] == "ok":
                value = float(row["smoothed_q_error"])
                if not (value >= 1.0 and math.isfinite(value)):
                    raise ValueError(f"{run}: smoothed Q-error {value!r} for query {row['query_id']}")
                logs.append(math.log(value))
    if not logs:
        raise ValueError(f"no answered {workload} queries in {run}")
    return math.exp(math.fsum(logs) / len(logs))


def row_of(run: Path, parameters: Dict[str, Any], variant_id: str) -> Dict[str, Any]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    accuracy = summary["workloads"]["validation_multi"]["accuracy"]
    q = accuracy.get("smoothed_q_error") or {}  # zero estimates stay finite (floor 1)
    artifacts = (summary.get("build") or {}).get("artifacts") or {}
    build = summary.get("build") or {}
    return {
        "variant_id": variant_id, "parameters": json.dumps(parameters, sort_keys=True),
        "query_count": accuracy.get("query_count"), "scored_count": accuracy.get("scored_query_count"),
        "coverage_fraction": accuracy.get("coverage_fraction"),
        "q_p50": q.get("p50"), "q_p90": q.get("p90"), "q_p95": q.get("p95"), "q_p99": q.get("p99"),
        "q_max": q.get("max"), "q_gmean": geometric_mean_q(run),
        "zero_estimate_count": accuracy.get("zero_estimate_count"),
        "serialized_bytes": artifacts.get("serialized_bytes"),
        "theoretical_bytes": artifacts.get("theoretical_bytes"),
        "build_seconds": build.get("total_build_seconds"),
        "coverage_sha256": coverage_sha256(run),
        "run_directory": str(run),
    }


def sweep(config_path: Path, validation_jsonl: Path, grid: List[Tuple[str, List[Any]]],
          expected_queries: Optional[int], output: Path, log=print) -> List[Dict[str, Any]]:
    base = load_raw_config(config_path)
    keys = [key for key, _ in grid]
    rows: List[Dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for values in itertools.product(*(values for _, values in grid)):
        parameters = dict(zip(keys, values))
        raw = tuning_raw(base, parameters, validation_jsonl, expected_queries)
        config = experiment_config_from_raw(raw, config_path)
        log(f"tuning {config.variant_id}")
        run = run_seed(config, 0)
        rows.append(row_of(run, parameters, config.variant_id))
        _write(rows, output)
    return rows


def _write(rows: Iterable[Dict[str, Any]], output: Path) -> None:
    rows = list(rows)
    with (output / "tuning.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "tuning.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validation-jsonl", required=True, type=Path)
    parser.add_argument("--expected-queries", type=int)
    parser.add_argument("--grid", action="append", default=[], help="adapter key=v1,v2,... (repeatable)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.grid:
        parser.error("at least one --grid is required")
    rows = sweep(args.config.resolve(), args.validation_jsonl.resolve(), parse_grid(args.grid),
                 args.expected_queries, args.output)
    print(json.dumps({"combinations": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
