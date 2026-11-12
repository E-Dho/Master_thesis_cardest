"""Same-query comparison: every method scored on the queries all of them answer, by query type.

The comparison of the core scores each method on the queries *it* answers
(``status = ok``), so methods with different coverage are compared on
different queries (DeepDB-MBR answers 74 % of POL multi, TraHist only the
space/time queries). This module restricts the scoring to common queries:

- **intersection** (per workload): the queries answered by every run (every
  seed) of every method in the comparison that answers any query of that
  workload at all (a method that answers none of a workload, such as
  Exact-MBR on single queries, is listed under ``not_compared``);
- **pairs** (``--reference``): for each reference method and each other
  method, the queries both answer, so the reference is compared with every
  baseline on as many queries as possible.

For every scope the smoothed Q-error (the ``predictions.csv`` column, floor 1)
is summarised per seed (n, p50, p90, p95, p99, max, geometric mean) and then
averaged over seeds (mean and sample std, as the core aggregates), overall and
per query type (``category``: dimension.interval.relation). ``wins`` is the
share of the scope's queries on which the method has the lowest per-query
error (geometric mean over its seeds; ties split).

Input: the aggregate ``comparison.json`` files (``cli aggregate``), which list
the run directories, so the same runs as the comparison are used; or
``--config``, which reads the config's current aggregate
(``<results_root>/<experiment>/<method>/<variant>/aggregate/comparison.json``,
written by ``cli aggregate`` and by every ``aggregate_compare.sbatch``). All
runs of a workload must have evaluated the same benchmark file (sha256) and
selection. ``aggregate_compare.sbatch`` runs it after every comparison
(``<comparison>/same_query``, reference ``own_model:segments_first_dedup``
when present).

    python -m evaluation.trajectory_pol_porto.traj_eval.same_query \\
        --aggregate-json A/comparison.json --config configs/pol_v3/mscn.yaml \\
        --reference own_model:segments_first_dedup --output DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .workloads import load_traj_workload

SCHEMA = "trajectory_same_query_v1"
STATISTICS = ("p50", "p90", "p95", "p99", "max", "gmean")
CSV_FIELDS = ("workload", "scope", "stratum", "method", "queries", "seeds",
              *(f"{name}_{part}" for name in STATISTICS for part in ("mean", "std")), "wins")


class Method:
    """One aggregate: ``method_id:variant_id`` and its runs (one per seed)."""

    def __init__(self, aggregate_json: Path) -> None:
        raw = json.loads(Path(aggregate_json).read_text(encoding="utf-8"))
        self.key = f"{raw['method_id']}:{raw['variant_id']}"
        self.display_name = raw.get("display_name", self.key)
        self.runs = [Path(path) for path in raw["runs"]]
        if not self.runs:
            raise ValueError(f"{aggregate_json}: no runs")
        self.manifests = [json.loads((run / "run_manifest.json").read_text(encoding="utf-8")) for run in self.runs]
        self.seeds = [int(manifest["seed"]) for manifest in self.manifests]
        # workload -> per run: query_id -> smoothed Q-error (ok only)
        self.errors: Dict[str, List[Dict[str, float]]] = {}
        for run in self.runs:
            with (run / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            per_workload: Dict[str, Dict[str, float]] = {}
            for row in rows:
                per_workload.setdefault(row["workload"], {})
                if row["status"] == "ok":
                    value = float(row["smoothed_q_error"])
                    if not math.isfinite(value) or value < 1.0:
                        raise ValueError(f"{run}: smoothed Q-error {row['smoothed_q_error']!r} "
                                         f"for {row['workload']} query {row['query_id']}")
                    per_workload[row["workload"]][row["query_id"]] = value
            for workload, errors in per_workload.items():
                self.errors.setdefault(workload, []).append(errors)
        for workload, runs in self.errors.items():
            if len(runs) != len(self.runs):
                raise ValueError(f"{self.key}: workload {workload} is missing in some runs")

    def identity(self, workload: str) -> List[Tuple[str, str, int]]:
        out = []
        for manifest in self.manifests:
            entry = manifest["workloads"][workload]
            out.append((entry["queries_jsonl_sha256"], json.dumps(entry.get("select"), sort_keys=True),
                        int(entry["query_count"])))
        return out

    def answered(self, workload: str) -> Set[str]:
        """Queries every seed answered."""
        runs = self.errors.get(workload, [])
        return set.intersection(*(set(run) for run in runs)) if runs else set()


def _summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {name: None for name in STATISTICS}
    array = np.asarray(values, dtype=np.float64)
    out = {f"p{q}": float(np.percentile(array, q)) for q in (50, 90, 95, 99)}
    out["max"] = float(array.max())
    out["gmean"] = float(np.exp(np.mean(np.log(array))))
    return out


def _mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    present = [v for v in values if v is not None]
    if not present:
        return None, None
    mean = float(np.mean(present))
    return mean, (float(np.std(present, ddof=1)) if len(present) > 1 else 0.0)


def score(methods: Sequence[Method], workload: str, queries: Iterable[str],
          categories: Mapping[str, str]) -> Dict[str, Dict[str, Any]]:
    """stratum -> method -> metrics on exactly ``queries``."""
    queries = sorted(queries, key=lambda q: (len(q), q))
    strata: Dict[str, List[str]] = OrderedDict([("all", queries)])
    for query in queries:
        strata.setdefault(f"category:{categories[query]}", []).append(query)
    out: Dict[str, Dict[str, Any]] = OrderedDict()
    for stratum in ["all"] + sorted(name for name in strata if name != "all"):
        members = strata[stratum]
        per_query = {}
        for method in methods:
            runs = method.errors[workload]
            per_query[method.key] = np.exp(np.mean([[math.log(run[q]) for q in members] for run in runs], axis=0)) \
                if members else np.zeros(0)
        wins = {method.key: 0.0 for method in methods}
        if members:
            stacked = np.vstack([per_query[method.key] for method in methods])
            best = stacked.min(axis=0)
            winners = np.isclose(stacked, best, rtol=1e-12, atol=0.0)
            share = winners / winners.sum(axis=0)
            for index, method in enumerate(methods):
                wins[method.key] = float(share[index].sum() / len(members))
        cell: Dict[str, Any] = OrderedDict()
        for method in methods:
            seeds = [_summary([run[q] for q in members]) for run in method.errors[workload]]
            metrics: Dict[str, Any] = {"queries": len(members), "seeds": len(seeds), "wins": wins[method.key]}
            for name in STATISTICS:
                metrics[f"{name}_mean"], metrics[f"{name}_std"] = _mean_std([s[name] for s in seeds])
            cell[method.key] = metrics
        out[stratum] = cell
    return out


def compare(aggregate_jsons: Sequence[Path], references: Sequence[str] = ()) -> Dict[str, Any]:
    methods = [Method(path) for path in aggregate_jsons]
    keys = [method.key for method in methods]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate method in the input: {keys}")
    by_key = {method.key: method for method in methods}
    workloads = sorted({workload for method in methods for workload in method.errors})
    result: Dict[str, Any] = {"schema_version": SCHEMA, "methods": {m.key: {"display_name": m.display_name,
                                                                          "seeds": m.seeds} for m in methods},
                              "references": list(references),
                              "missing_references": [r for r in references if r not in by_key],
                              "workloads": OrderedDict()}
    for workload in workloads:
        present = [method for method in methods if workload in method.errors]
        identities = {key for method in present for key in method.identity(workload)}
        if len(identities) != 1:
            raise ValueError(f"workload {workload}: the runs evaluated different benchmarks or selections "
                             f"{sorted(identities)}")
        sha, select, count = identities.pop()
        entry = present[0].manifests[0]["workloads"][workload]
        records = load_traj_workload(entry["queries_jsonl"], workload, select=entry.get("select"))
        if len(records) != count:
            raise ValueError(f"workload {workload}: {len(records)} records, the runs had {count}")
        categories = {str(record.query_id): record.category for record in records}
        compared = [method for method in present if method.answered(workload)]
        common = set.intersection(*(method.answered(workload) for method in compared)) if compared else set()
        block: Dict[str, Any] = {
            "queries": count, "queries_jsonl_sha256": sha,
            "coverage": {m.key: len(m.answered(workload)) for m in present},
            "not_compared": [m.key for m in present if not m.answered(workload)],
            "intersection": {"methods": [m.key for m in compared], "queries": len(common),
                             "strata": score(compared, workload, common, categories) if common else {}},
            "pairs": OrderedDict(),
        }
        for reference in references:
            if reference not in by_key or by_key[reference] not in compared:
                continue
            ref = by_key[reference]
            for other in compared:
                if other is ref:
                    continue
                both = ref.answered(workload) & other.answered(workload)
                block["pairs"][f"{reference}|{other.key}"] = {
                    "methods": [ref.key, other.key], "queries": len(both),
                    "strata": score([ref, other], workload, both, categories) if both else {}}
        result["workloads"][workload] = block
    return result


def csv_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for workload, block in result["workloads"].items():
        scopes = [("intersection", block["intersection"])] + [(f"pair:{name}", pair)
                                                              for name, pair in block["pairs"].items()]
        for scope, data in scopes:
            for stratum, cell in data["strata"].items():
                for method, metrics in cell.items():
                    rows.append({"workload": workload, "scope": scope, "stratum": stratum, "method": method,
                                 **{field: metrics.get(field) for field in CSV_FIELDS[4:]}})
    return rows


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}" if value < 10 else f"{value:.1f}"


def _cell(metrics: Mapping[str, Any], name: str) -> str:
    mean, std = metrics.get(f"{name}_mean"), metrics.get(f"{name}_std")
    if mean is None:
        return "-"
    return _fmt(mean) if not std else f"{_fmt(mean)} ± {_fmt(std)}"


def markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Same-query comparison", "",
             "Smoothed Q-error on the queries every compared method answers (all seeds; mean ± std over "
             "seeds). `wins`: share of those queries on which the method has the lowest error.", ""]
    if result["missing_references"]:
        lines += [f"References not in this comparison: {', '.join(result['missing_references'])}", ""]
    for workload, block in result["workloads"].items():
        inter = block["intersection"]
        lines += [f"## {workload}", "",
                  f"{inter['queries']} of {block['queries']} queries answered by all "
                  f"{len(inter['methods'])} methods. Coverage: "
                  + ", ".join(f"{k} {v}" for k, v in block["coverage"].items()) + "."]
        if block["not_compared"]:
            lines.append(f"Not compared (answer no query): {', '.join(block['not_compared'])}.")
        lines.append("")
        if inter["strata"]:
            lines += ["| method | p50 | p90 | p99 | max | gmean | wins |", "|---|---|---|---|---|---|---|"]
            for method, metrics in inter["strata"]["all"].items():
                lines.append(f"| {method} | {_cell(metrics, 'p50')} | {_cell(metrics, 'p90')} | "
                             f"{_cell(metrics, 'p99')} | {_cell(metrics, 'max')} | {_cell(metrics, 'gmean')} | "
                             f"{metrics['wins']:.0%} |")
            categories = [name for name in inter["strata"] if name != "all"]
            if categories:
                lines += ["", "Median (p90) per query type, same queries:", "",
                          "| method | " + " | ".join(f"{c.split(':', 1)[1]} (n={next(iter(inter['strata'][c].values()))['queries']})"
                                                     for c in categories) + " |",
                          "|---|" + "---|" * len(categories)]
                for method in inter["methods"]:
                    lines.append(f"| {method} | " + " | ".join(
                        f"{_cell(inter['strata'][c][method], 'p50')} ({_cell(inter['strata'][c][method], 'p90')})"
                        for c in categories) + " |")
            lines.append("")
        if block["pairs"]:
            lines += ["Pairs (each on the queries both answer): p50 / p90 / p99 / gmean of each, and the "
                      "reference's share of wins",
                      "", "| pair | queries | reference | other | reference wins |", "|---|---|---|---|---|"]
            for name, pair in block["pairs"].items():
                if not pair["strata"]:
                    lines.append(f"| {name} | 0 | - | - | - |")
                    continue
                ref, other = pair["methods"]
                cell = pair["strata"]["all"]
                text = lambda m: " / ".join(_cell(cell[m], s) for s in ("p50", "p90", "p99", "gmean"))  # noqa: E731
                lines.append(f"| {ref} vs {other} | {pair['queries']} | {text(ref)} | {text(other)} | "
                             f"{cell[ref]['wins']:.0%} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def write(result: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "same_query.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "same_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(result))
    (output / "same_query.md").write_text(markdown(result), encoding="utf-8")


def aggregate_of(config_path: Path) -> Path:
    """The current aggregate ``comparison.json`` of a method config (as ``cli aggregate`` writes it)."""
    from evaluation.eval_core.config import load_experiment_config

    config = load_experiment_config(config_path)
    path = (config.results_root / config.experiment_id / config.method_id / config.variant_id
            / "aggregate" / "comparison.json")
    if not path.exists():
        raise FileNotFoundError(f"{config_path}: no aggregate at {path} (run cli aggregate first)")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--aggregate-json", action="append", default=[], type=Path)
    parser.add_argument("--config", action="append", default=[], type=Path,
                        help="a method config; its current aggregate (cli aggregate) is used")
    parser.add_argument("--reference", action="append", default=[],
                        help="method_id:variant_id compared pairwise with every other method (repeatable; "
                             "ignored, and listed, when not in the comparison)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    aggregates = list(args.aggregate_json) + [aggregate_of(path) for path in args.config]
    if not aggregates:
        parser.error("give --aggregate-json or --config")
    result = compare(aggregates, args.reference)
    write(result, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
