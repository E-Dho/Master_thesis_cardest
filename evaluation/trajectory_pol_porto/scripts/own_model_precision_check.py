#!/usr/bin/env python3
"""Diagnostic: does float32 accumulation of factorized-column CDFs explain GPU/CPU estimate drift?

P7 GPU timing (job 22995629) disagreed with the CPU accuracy run on one POL
single query (1819: 241.02 vs 246.79, 2.39 %), the same relative deviation as
the worst case of the first GPU timing (22923145). Hypothesis: a range
predicate on a factorized column is evaluated as ``P(X <= U) - P(X <= L)``
(``FactorizedColumnProbabilityEvaluator.predicate_mass``), both CDFs summed in
float32 in different orders on CPU and GPU; for a narrow range high in the
domain the difference cancels most digits, so both devices are off by up to a
few percent (and a negative difference is clamped to 0).

For every query of a workload selection this script computes the reported
estimate (as the bridge does) on each requested device twice: with the
current float32 accumulation and with the factor distributions accumulated in
float64 (the network itself stays float32). It writes:

- ``estimates.csv``: per query the route, diagnostic and the estimate per
  ``<device>_<precision>``;
- ``summary.json``: for each pair of columns the number of queries outside the
  GPU timing tolerance (0.1 % relative or one tuple), relative-difference
  quantiles, the largest differences, and zero estimates that become
  positive in float64;
- ``detail.json`` for ``--detail`` queries: per column the predicate token and
  the factor in each setting, and every CDF evaluated for it.

    python evaluation/trajectory_pol_porto/scripts/own_model_precision_check.py \\
        --training-config .../training_config.yaml --checkpoint .../checkpoint_best.pt \\
        --queries-jsonl .../benchmark.jsonl --select-json '{"relation": "single"}' \\
        --devices cpu,cuda --detail 1819 --output DIR

Read-only: nothing in the repository or the results changes.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
TOLERANCE = (1.0e-3, 1.0)  # the gpu_single_query estimate-consistency tolerance (relative, absolute)


def _bridge():
    spec = importlib.util.spec_from_file_location(
        "own_model_traj_bridge", REPO_ROOT / "evaluation/trajectory_pol_porto/scripts/own_model_traj_bridge.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def accumulation(precision: str, record: Optional[List[Dict[str, Any]]] = None):
    """float32: the code as it is; float64: factor distributions (and all masses built from them) in float64."""
    import torch

    from model.src.model import output_adapter

    cls = output_adapter.FactorizedColumnProbabilityEvaluator
    saved = {name: cls.__dict__[name] for name in ("conditional_distribution", "dtype", "less_equal_mass")}
    try:
        if precision == "float64":
            original = saved["conditional_distribution"]
            cls.conditional_distribution = lambda self, index, prefixes: original(self, index, prefixes).double()
            cls.dtype = property(lambda self: torch.float64)
        elif precision != "float32":
            raise ValueError(precision)
        if record is not None:
            inner = cls.less_equal_mass

            def less_equal_mass(self, encoded_id):
                value = inner(self, encoded_id)
                record.append({"encoded_id": int(encoded_id), "cdf": float(value.reshape(-1)[0]),
                               "dtype": str(value.dtype)})
                return value

            cls.less_equal_mass = less_equal_mass
        yield
    finally:
        for name, value in saved.items():
            setattr(cls, name, value)


def _estimate(bridge, model, translation, estimate_kind: str) -> float:
    parts = model.estimate(translation)
    return float(bridge.reported_estimate(translation.route, parts, estimate_kind))


def _token_text(token) -> Dict[str, Any]:
    return {key: (str(value) if value is not None else None)
            for key, value in (("op", getattr(token, "op", None)), ("value", getattr(token, "value", None)),
                               ("upper", getattr(token, "upper", None)),
                               ("lower_inclusive", getattr(token, "lower_inclusive", None)),
                               ("upper_inclusive", getattr(token, "upper_inclusive", None)))}


def detail(bridge, model, translation, precision: str) -> Dict[str, Any]:
    from model.src.evaluation.pol_query_adapter import pol_workload_record_to_context

    context = pol_workload_record_to_context(translation.record, model.metadata, srid=model.srid,
                                             trajectory_spatial=model.trajectory_spatial)
    tokens = list(context.tokens)
    calls: List[Dict[str, Any]] = []
    with accumulation(precision, calls):
        factors = model.wrapped.predict_column_factors(tokens)
    columns = []
    for index, (column, token) in enumerate(zip(model.metadata.columns, tokens)):
        if str(getattr(token, "op", "")).endswith("WILDCARD"):
            continue
        columns.append({"index": index, "column": getattr(column, "name", str(index)),
                        "domain_size": len(getattr(column, "domain", ()) or ()),
                        "token": _token_text(token), "factor": float(factors[index])})
    return {"precision": precision, "full_join_cardinality": float(model.metadata.full_join_cardinality),
            "constrained_columns": columns, "cdf_calls": calls}


def _quantiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]  # noqa: E731
    return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": values[-1]}


def compare(rows: List[Dict[str, Any]], a: str, b: str) -> Dict[str, Any]:
    relative, outside, largest, zero_to_positive = [], [], [], 0
    for row in rows:
        x, y = row.get(a), row.get(b)
        if x is None or y is None:
            continue
        difference = abs(x - y)
        rel = difference / max(abs(x), 1e-300) if x else (0.0 if y == 0 else math.inf)
        relative.append(rel)
        if difference > max(TOLERANCE[0] * abs(x), TOLERANCE[1]):
            outside.append(row["query_id"])
        largest.append((rel, row["query_id"], x, y))
        if x == 0 and y > 0:
            zero_to_positive += 1
    largest.sort(reverse=True)
    finite = [value for value in relative if math.isfinite(value)]
    return {"queries": len(relative), "outside_gpu_tolerance": len(outside), "outside_query_ids": outside[:50],
            "relative_difference": _quantiles(finite),
            "relative_above": {str(t): sum(1 for v in relative if v > t) for t in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)},
            "zero_in_first_positive_in_second": zero_to_positive,
            "largest": [{"query_id": q, a: x, b: y, "relative": r} for r, q, x, y in largest[:10]]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries-jsonl", required=True, type=Path)
    parser.add_argument("--select-json", required=True)
    parser.add_argument("--estimate", choices=("dedup", "matching_segments"), default="dedup")
    parser.add_argument("--devices", default="cpu", help="comma-separated: cpu, cuda")
    parser.add_argument("--detail", action="append", type=int, default=[], help="query id (workload position)")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    bridge = _bridge()
    bridge._install_paths(REPO_ROOT)
    import torch

    rows = bridge._rows(args.queries_jsonl, args.select_json, args.query_limit)
    devices = [d for d in args.devices.split(",") if d]
    table: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    environment = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                   "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                   "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                   "float32_matmul_precision": torch.get_float32_matmul_precision()}
    for device in devices:
        model = bridge.OwnModel(args.training_config, args.checkpoint, device=device)
        environment[f"device_name_{device}"] = model.device_name()
        translations = {}
        for query_id, row in enumerate(rows):
            try:
                translations[query_id] = model.translate_row(row)
            except model.translate.TranslationError:
                continue
        if not table:
            table = [{"query_id": q, "route": t.route, "diagnostic": ";".join(t.notes)} for q, t in translations.items()]
        by_id = {row["query_id"]: row for row in table}
        for precision in ("float32", "float64"):
            column = f"{device}_{precision}"
            with torch.no_grad(), accumulation(precision):
                for query_id, translation in translations.items():
                    try:
                        by_id[query_id][column] = _estimate(bridge, model, translation, args.estimate)
                    except Exception as exc:  # noqa: BLE001 - recorded per query
                        by_id[query_id][column] = None
                        by_id[query_id][f"{column}_error"] = f"{type(exc).__name__}: {exc}"
            for query_id in args.detail:
                if query_id in translations:
                    with torch.no_grad():
                        details.setdefault(str(query_id), {"row": rows[query_id].get("query_id"),
                                                           "route": translations[query_id].route,
                                                           "record": translations[query_id].record,
                                                           "settings": {}})
                        details[str(query_id)]["settings"][column] = detail(bridge, model, translations[query_id],
                                                                              precision)
    columns = [f"{d}_{p}" for d in devices for p in ("float32", "float64")]
    pairs = []
    for device in devices:
        pairs.append((f"{device}_float32", f"{device}_float64"))  # accuracy effect of the precision
    if len(devices) > 1:
        pairs.append((f"{devices[0]}_float32", f"{devices[1]}_float32"))  # the observed drift
        pairs.append((f"{devices[0]}_float64", f"{devices[1]}_float64"))  # drift left with float64 accumulation
    summary = {"environment": environment, "query_count": len(rows), "translated": len(table),
               "tolerance": {"relative": TOLERANCE[0], "absolute": TOLERANCE[1]},
               "comparisons": {f"{a} vs {b}": compare(table, a, b) for a, b in pairs}}
    args.output.mkdir(parents=True, exist_ok=True)
    import csv

    fields = ["query_id", "route", "diagnostic", *columns]
    with (args.output / "estimates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({k: ("" if v is None else (repr(v) if isinstance(v, float) else v)) for k, v in row.items()}
                         for row in table)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.output / "detail.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["comparisons"], indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
