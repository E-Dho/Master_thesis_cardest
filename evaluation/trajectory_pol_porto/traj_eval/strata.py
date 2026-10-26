"""Coverage and stratified accuracy for trajectory workloads.

Per run (one seed, one workload) :func:`trajectory_summary` adds an
``extension`` block to ``summary.json``:

- **coverage**: ``ok`` split into exact and approximate estimates (an adapter
  marks an approximation with ``status = ok`` and a diagnostic starting with
  ``approx:``), plus ``unsupported`` and ``failed``;
- **strata**: raw and smoothed Q-error percentiles, percentile-bootstrap
  confidence intervals, and the signed error ``log10(estimate / truth)`` per
  stratum: overall, exact estimates only, category, true-cardinality decade,
  window area and interval length (as fractions of the data domain), and
  number of attribute add-ons.

Across seeds, :class:`TrajectorySummaryExtension` (registered with the core
through the plugin) averages every stratum metric (mean and sample standard
deviation; confidence-interval bounds are averaged too) and writes
``strata.csv`` / ``strata.md`` next to the aggregate and the comparison.

Q-errors follow ``evaluation.eval_core.metrics`` exactly (1e-12 guard for the
raw metric, floor 1 for the smoothed one). Only estimates with status ``ok``
are scored; coverage says how many that is.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.eval_core.metrics import raw_q_error, smoothed_q_error

SCHEMA = "trajectory_summary_v1"
APPROX_PREFIX = "approx:"
PERCENTILES = (50, 90, 95, 99)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
MIN_QUERIES_FOR_CI = 10
#: decade edges for domain fractions: <1e-6, [1e-6, 1e-5), ..., [1e-1, 1]
FRACTION_DECADES = tuple(range(-6, 0))


def is_approximate(prediction: Any) -> bool:
    return prediction.status == "ok" and str(prediction.diagnostic or "").startswith(APPROX_PREFIX)


# ---------------------------------------------------------------------------
# bucketing
# ---------------------------------------------------------------------------


def truth_decade(value: int) -> str:
    if value <= 0:
        return "0"
    decade = int(math.floor(math.log10(value)))
    return f"[1e{decade},1e{decade + 1})"


def fraction_bucket(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value <= 0:
        return "0"
    if value < 10.0 ** FRACTION_DECADES[0]:
        return f"<1e{FRACTION_DECADES[0]}"
    decade = int(math.floor(math.log10(value)))
    if decade >= -1:
        return "[1e-1,1]"
    return f"[1e{decade},1e{decade + 1})"


def strata_of(record: Any) -> List[str]:
    """All strata a query belongs to (besides ``all``)."""

    features = record.features
    names = [
        f"category:{record.category}",
        f"truth_decade:{truth_decade(record.true_cardinality)}",
        f"addons:{record.addon_count}",
    ]
    area = fraction_bucket(features.get("window_area_fraction"))
    if area is not None:
        names.append(f"window_area_fraction:{area}")
    interval = fraction_bucket(features.get("interval_fraction"))
    if interval is not None:
        names.append(f"interval_fraction:{interval}")
    return names


# ---------------------------------------------------------------------------
# per-run summary
# ---------------------------------------------------------------------------


def _percentiles(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    result = {f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES}
    result["max"] = float(values.max())
    return result


def _bootstrap(values: np.ndarray, seed_text: str) -> Dict[str, Optional[List[float]]]:
    if values.size < MIN_QUERIES_FOR_CI:
        return {f"p{q}": None for q in PERCENTILES}
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    estimates = {q: [] for q in PERCENTILES}
    chunk = max(1, min(BOOTSTRAP_RESAMPLES, 2_000_000 // values.size))
    done = 0
    while done < BOOTSTRAP_RESAMPLES:
        size = min(chunk, BOOTSTRAP_RESAMPLES - done)
        sample = values[rng.integers(0, values.size, size=(size, values.size))]
        for q in PERCENTILES:
            estimates[q].append(np.percentile(sample, q, axis=1))
        done += size
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    result = {}
    for q in PERCENTILES:
        draws = np.concatenate(estimates[q])
        result[f"p{q}"] = [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))]
    return result


def _stratum(name: str, items: Sequence[Tuple[Any, Any]], seed_text: str) -> Dict[str, Any]:
    statuses = {"ok_exact": 0, "ok_approx": 0, "unsupported": 0, "failed": 0}
    raw, smooth, log_ratio = [], [], []
    for record, prediction in items:
        if prediction.status == "ok" and prediction.estimated_cardinality is not None:
            statuses["ok_approx" if is_approximate(prediction) else "ok_exact"] += 1
            estimate = float(prediction.estimated_cardinality)
            truth = record.true_cardinality
            raw.append(raw_q_error(estimate, truth))
            smooth.append(smoothed_q_error(estimate, truth))
            log_ratio.append(math.log10(max(estimate, 1.0) / max(truth, 1.0)))
        elif prediction.status in ("unsupported", "failed"):
            statuses[prediction.status] += 1
        else:
            statuses["failed"] += 1
    n = len(items)
    scored = statuses["ok_exact"] + statuses["ok_approx"]
    raw_array, log_array = np.asarray(raw, dtype=float), np.asarray(log_ratio, dtype=float)
    return {
        "query_count": n,
        "scored_count": scored,
        **{f"{key}_count": value for key, value in statuses.items()},
        "coverage_fraction": scored / n if n else None,
        "raw_q_error": _percentiles(raw_array),
        "raw_q_error_ci": _bootstrap(raw_array, f"{seed_text}|{name}"),
        "smoothed_q_error": _percentiles(np.asarray(smooth, dtype=float)),
        "log10_ratio": {
            "p10": float(np.percentile(log_array, 10)) if log_array.size else None,
            "p50": float(np.percentile(log_array, 50)) if log_array.size else None,
            "p90": float(np.percentile(log_array, 90)) if log_array.size else None,
            "mean": float(log_array.mean()) if log_array.size else None,
        },
        "overestimate_fraction": float((log_array > 0).mean()) if log_array.size else None,
    }


def trajectory_summary(workload_id: str, records: Sequence[Any], predictions: Sequence[Any]) -> Dict[str, Any]:
    by_id = {prediction.query_id: prediction for prediction in predictions}
    pairs = [(record, by_id[record.query_id]) for record in records]
    groups: "OrderedDict[str, List[Tuple[Any, Any]]]" = OrderedDict()
    groups["all"] = pairs
    groups["all:exact_only"] = [
        pair for pair in pairs if pair[1].status == "ok" and not is_approximate(pair[1])
    ]
    keyed: Dict[str, List[Tuple[Any, Any]]] = {}
    for pair in pairs:
        for name in strata_of(pair[0]):
            keyed.setdefault(name, []).append(pair)
    for name in sorted(keyed):
        groups[name] = keyed[name]
    strata = OrderedDict((name, _stratum(name, items, workload_id)) for name, items in groups.items())
    overall = strata["all"]
    return {
        "schema": SCHEMA,
        "coverage": {
            key: overall[key]
            for key in ("query_count", "scored_count", "ok_exact_count", "ok_approx_count",
                        "unsupported_count", "failed_count", "coverage_fraction")
        },
        "strata": strata,
        "bootstrap": {
            "method": "percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "min_queries": MIN_QUERIES_FOR_CI,
            "statistic": "raw Q-error percentiles over scored queries",
        },
        "notes": {
            "approximate": f"status ok with a diagnostic starting with {APPROX_PREFIX!r}",
            "all:exact_only": "scored queries without approximate estimates",
            "log10_ratio": "log10(max(estimate,1) / max(truth,1)); > 0 is an overestimate",
        },
    }


# ---------------------------------------------------------------------------
# aggregation across seeds and methods (called by evaluation.eval_core.report)
# ---------------------------------------------------------------------------


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    if not values or any(value is None for value in values):
        return {"mean": None, "std": None}
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()), "std": float(data.std(ddof=1)) if data.size > 1 else 0.0}


def _flatten(stratum: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    flat: Dict[str, Optional[float]] = {}
    for key, value in stratum.items():
        if isinstance(value, Mapping) and key.endswith("_ci"):
            for inner, bounds in value.items():
                flat[f"{key}.{inner}.low"] = bounds[0] if bounds else None
                flat[f"{key}.{inner}.high"] = bounds[1] if bounds else None
        elif isinstance(value, Mapping):
            for inner, inner_value in value.items():
                if inner_value is None or isinstance(inner_value, (int, float)):
                    flat[f"{key}.{inner}"] = inner_value
        elif value is None or isinstance(value, (int, float)):
            flat[key] = value
    return flat


MARKDOWN_COLUMNS = (
    ("n", "query_count"),
    ("coverage", "coverage_fraction"),
    ("approx", "ok_approx_count"),
    ("raw p50", "raw_q_error.p50"),
    ("raw p90", "raw_q_error.p90"),
    ("raw p99", "raw_q_error.p99"),
    ("raw max", "raw_q_error.max"),
    ("log10 ratio p50", "log10_ratio.p50"),
    ("overestimated", "overestimate_fraction"),
)


def _cell(metric: Mapping[str, Optional[float]]) -> str:
    if metric is None or metric.get("mean") is None:
        return "N/A"
    mean, std = metric["mean"], metric.get("std") or 0.0
    # identical seed values can leave floating-point noise in the std
    if std <= 1e-9 * max(1.0, abs(mean)):
        return f"{mean:.4g}"
    return f"{mean:.4g} +/- {std:.2g}"


class TrajectorySummaryExtension:
    schema = SCHEMA

    def aggregate(self, extensions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        names = list(extensions[0]["strata"])
        if any(list(item["strata"]) != names for item in extensions[1:]):
            raise ValueError("seed summaries have different trajectory strata")
        strata = OrderedDict()
        for name in names:
            flats = [_flatten(item["strata"][name]) for item in extensions]
            keys = list(flats[0])
            strata[name] = {key: _mean_std([flat.get(key) for flat in flats]) for key in keys}
        return {
            "schema": SCHEMA,
            "seed_count": len(extensions),
            "strata": strata,
            "bootstrap": extensions[0].get("bootstrap"),
            "ci_aggregation": "per-seed percentile-bootstrap bounds, averaged across seeds",
        }

    def rows(self, identity: Mapping[str, Any], workload_id: str, aggregated: Mapping[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        for name, metrics in aggregated["strata"].items():
            dimension, _, value = name.partition(":")
            row: Dict[str, Any] = {**identity, "workload": workload_id, "stratum": name,
                                   "stratum_dimension": dimension, "stratum_value": value}
            for key, metric in metrics.items():
                row[f"{key}.mean"] = metric["mean"]
                row[f"{key}.std"] = metric["std"]
            rows.append(row)
        return rows

    def markdown(self, title: str, entries: Iterable[Tuple[Mapping[str, Any], str, Mapping[str, Any]]]) -> str:
        lines = [f"# {title}", "",
                 "Q-error over scored (`ok`) queries; values are mean +/- sample std across seeds. "
                 "`log10 ratio` > 0 means overestimation. Coverage counts `ok` (exact and approximate) "
                 "estimates; `approx` is the mean number of approximate ones.", ""]
        header = ["Method", "Variant", "Workload", "Stratum"] + [label for label, _ in MARKDOWN_COLUMNS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for identity, workload_id, aggregated in entries:
            for name, metrics in aggregated["strata"].items():
                cells = [str(identity.get("method_id", "")), str(identity.get("variant_id", "")),
                         workload_id, name]
                cells += [_cell(metrics.get(key)) for _, key in MARKDOWN_COLUMNS]
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


TRAJECTORY_SUMMARY_EXTENSION = TrajectorySummaryExtension()
