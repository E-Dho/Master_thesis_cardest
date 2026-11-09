"""Split the own model's distinct-trajectory error into its three factors (P7).

For a multi query joining segments the model estimates ``D_hat = M_hat * a_hat``.
With the true counts of ``segment_truth.py`` (``M`` = matching segments and
``D`` = distinct trajectories under the semantics the model answers: segment
MBR, space and time separate; ``D_exact`` = the label):

    D_hat / D_exact = (M_hat / M) * (a_hat / a) * (D / D_exact),   a = D / M

- segment factor ``M_hat / M``: error of the learned matching-segment count;
- dedup factor ``a_hat / a``: error of the dedup factor the reported estimate
  used, ``a_hat = D_hat / M_hat`` taken from the reported estimate, not from the
  stored head output: ``traj_dedup_factor`` for ``own_model``, 1 for the
  ablation ``own_model_no_dedup`` (which reports ``M_hat``; its whole
  duplication error lands in this term);
- semantics factor ``D / D_exact``: what answering the MBR / decoupled question
  instead of the exact one costs (1 for queries without a spatial window).

The three log10 terms add up to ``log10(D_hat / D_exact)`` exactly. Queries with
a zero estimate or zero support have no logarithm and are counted separately.
Inputs: the runs' ``exchange/<workload>_predictions_components.csv`` (written
by ``scripts/own_model_traj_bridge.py``), the benchmark (query positions ->
ids) and ``segment_truth``'s per-query CSV.

    python -m evaluation.trajectory_pol_porto.traj_eval.own_model_error_split \\
        --benchmark .../benchmark.jsonl --segment-truth .../test_per_query.csv \\
        --run RUN_SEED0 --run RUN_SEED1 --run RUN_SEED2 --output-prefix .../own_model_error_split
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TERMS = ("segment", "dedup", "semantics")


def _csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_query(component: Dict[str, str], truth: Dict[str, str]) -> Optional[Dict[str, float]]:
    """The three log10 terms of one query, or None (zero estimate / zero support / not a distinct route)."""

    if component["route"] != "distinct" or component["status"] != "ok" or component["zero_support"] != "False":
        return None
    m_hat = float(component["matching_segment_estimate"])
    reported = float(component["estimated_cardinality"])
    kind = component.get("estimate_kind")
    if kind not in ("dedup", "matching_segments"):
        raise ValueError(f"components row {component.get('query_id')} has no known estimate_kind ({kind!r})")
    if m_hat <= 0 or reported <= 0:
        return None
    a_hat = reported / m_hat  # the factor the reported estimate actually applied
    expected = float(component["traj_dedup_factor"]) if kind == "dedup" else 1.0
    if not math.isclose(a_hat, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"components row {component.get('query_id')}: reported estimate {reported} is not "
                         f"M_hat x {'traj_dedup_factor' if kind == 'dedup' else '1'}")
    m, d = int(truth["matching_segments_model"]), int(truth["distinct_model"])
    d_exact = int(truth["distinct_exact"])
    if min(m_hat, a_hat, m, d, d_exact) <= 0:
        return None
    a = d / m
    terms = {"segment": math.log10(m_hat / m), "dedup": math.log10(a_hat / a),
             "semantics": math.log10(d / d_exact)}
    terms["total"] = math.log10(reported / d_exact)
    if abs(sum(terms[t] for t in TERMS) - terms["total"]) > 1e-9:
        raise ArithmeticError(f"terms do not add up for {truth['query_id']}")
    return terms


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(items: List[Dict[str, float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"queries": len(items)}
    for term in (*TERMS, "total"):
        values = [item[term] for item in items]
        magnitudes = [abs(v) for v in values]
        out[term] = {"mean_log10": (sum(values) / len(values)) if values else None,
                     "median_abs_log10": _quantile(magnitudes, 0.5), "p90_abs_log10": _quantile(magnitudes, 0.9)}
    dominant = defaultdict(int)
    for item in items:
        dominant[max(TERMS, key=lambda term: abs(item[term]))] += 1
    out["dominant_share"] = {term: (dominant[term] / len(items)) if items else None for term in TERMS}
    return out


def split_runs(benchmark: Path, truth_csv: Path, runs: Iterable[Path], workload: str = "pol_v3_multi",
               select: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .workloads import select_rows

    positions = [row["query_id"] for row in select_rows(benchmark, select or {"relation": "multi"})]
    truth = {row["query_id"]: row for row in _csv(truth_csv)}
    per_run: Dict[str, Any] = {}
    pooled: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    rows_out: List[Dict[str, Any]] = []
    for run in runs:
        components = _csv(Path(run) / "exchange" / f"{workload}_predictions_components.csv")
        if len(components) != len(positions):
            raise ValueError(f"{run}: {len(components)} component rows for {len(positions)} workload queries")
        kinds = {component.get("estimate_kind") for component in components if component["route"] == "distinct"}
        if len(kinds) != 1:
            raise ValueError(f"{run}: distinct routes with estimate kinds {sorted(map(str, kinds))}")
        groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        excluded = 0
        for component in components:
            source_id = positions[int(component["query_id"])]
            if component["route"] != "distinct":
                continue
            if source_id not in truth:
                raise ValueError(f"no segment truth for {source_id}")
            terms = split_query(component, truth[source_id])
            if terms is None:
                excluded += 1
                continue
            dimension = truth[source_id]["dimension"]
            groups["all"].append(terms)
            groups[dimension].append(terms)
            pooled["all"].append(terms)
            pooled[dimension].append(terms)
            rows_out.append({"run": str(run), "query_id": source_id, "dimension": dimension, **terms})
        per_run[str(run)] = {"estimate_kind": kinds.pop(), "excluded_zero_or_unsupported": excluded,
                             "groups": {name: summarize(items) for name, items in sorted(groups.items())}}
    run_kinds = {value["estimate_kind"] for value in per_run.values()}
    if len(run_kinds) > 1:
        raise ValueError(f"runs of different estimate kinds cannot be pooled: {sorted(run_kinds)}")
    return {"estimate_kind": run_kinds.pop() if run_kinds else None,
            "definition": "D_hat/D_exact = (M_hat/M) * (a_hat/a) * (D/D_exact), a_hat = D_hat/M_hat of the "
                          "reported estimate; log10 terms; M, D under the model's "
                          "semantics (segment MBR, separate space and time), D_exact = label",
            "runs": per_run, "pooled": {name: summarize(items) for name, items in sorted(pooled.items())},
            "rows": rows_out}


def markdown(report: Dict[str, Any]) -> str:
    lines = [f"# Own model: error split (log10 terms, pooled over runs; estimate: {report.get('estimate_kind')})", "",
             "| group | queries | segment med abs | dedup med abs | semantics med abs | total med abs | "
             "segment mean | dedup mean | semantics mean | dominant segment / dedup / semantics |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    fmt = lambda v: "" if v is None else f"{v:.3f}"  # noqa: E731
    for name, group in report["pooled"].items():
        share = group["dominant_share"]
        lines.append(f"| {name} | {group['queries']} | " + " | ".join(
            fmt(group[t]["median_abs_log10"]) for t in (*TERMS, "total")) + " | " + " | ".join(
            fmt(group[t]["mean_log10"]) for t in TERMS) + " | " + " / ".join(fmt(share[t]) for t in TERMS) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--segment-truth", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--workload", default="pol_v3_multi")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    report = split_runs(args.benchmark, args.segment_truth, args.run, workload=args.workload)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = report.pop("rows")
    Path(f"{args.output_prefix}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    Path(f"{args.output_prefix}.md").write_text(markdown(report), encoding="utf-8")
    with Path(f"{args.output_prefix}_per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "query_id", "dimension", *TERMS, "total"))
        writer.writeheader()
        writer.writerows(rows)
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
