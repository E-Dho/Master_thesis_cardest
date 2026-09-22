from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .records import LatencyRecord, PredictionRecord


EPSILON = 1.0e-12
PERCENTILES = (50, 90, 95, 99)

# ---------------------------------------------------------------------------
# Q-error flavours
# ---------------------------------------------------------------------------
# Two q-error variants are tracked because zero-truth queries cause a blow-up
# in raw q-error: raw_q_error(est, 0) ≈ est / EPSILON, which dominates every
# aggregate metric and makes models with many empty-result queries incomparable.
#
# Convention used throughout this module:
#   raw_q_error          — max(est, ε) / max(truth, ε) or its reciprocal;
#                          reliable only when truth > 0.  Use this as the
#                          primary accuracy metric for **true-positive queries**
#                          (i.e. queries whose actual result set is non-empty).
#
#   smoothed_q_error     — max(est, 1.0) / max(truth, 1.0) or its reciprocal;
#                          treats both sides as "at least 1 row", so a zero-
#                          truth query with a zero estimate scores 1.0 (perfect)
#                          instead of 1/ε (catastrophic).  Use this as the
#                          primary accuracy metric for **true-zero queries**
#                          (queries whose actual result set is empty) and as a
#                          secondary overall metric that is comparable across
#                          workloads with varying fractions of empty results.
#
# The summary produced by summarize_predictions() therefore contains four
# separate q-error blocks:
#   raw_q_error                — all scored queries   (primary: compare true-positives)
#   raw_q_error_true_positive  — truth > 0 only       (isolated true-positive view)
#   smoothed_q_error_true_zero — truth == 0 only      (isolated empty-result view)
#   smoothed_q_error           — all scored queries   (cross-workload comparable)
# ---------------------------------------------------------------------------


def raw_q_error(estimate: float, truth: float) -> float:
    estimate = max(float(estimate), EPSILON)
    truth = max(float(truth), EPSILON)
    return max(estimate / truth, truth / estimate)


def smoothed_q_error(estimate: float, truth: float) -> float:
    estimate = max(float(estimate), 1.0)
    truth = max(float(truth), 1.0)
    return max(estimate / truth, truth / estimate)


def attach_q_errors(record: PredictionRecord) -> PredictionRecord:
    if record.status != "ok" or record.estimated_cardinality is None:
        return record
    estimate = float(record.estimated_cardinality)
    return PredictionRecord(
        workload=record.workload,
        query_id=record.query_id,
        status=record.status,
        true_cardinality=record.true_cardinality,
        estimated_cardinality=estimate,
        raw_q_error=raw_q_error(estimate, record.true_cardinality),
        smoothed_q_error=smoothed_q_error(estimate, record.true_cardinality),
        diagnostic=record.diagnostic,
    )


def percentile_summary(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def summarize_predictions(records: list[PredictionRecord]) -> dict[str, Any]:
    completed = [attach_q_errors(record) for record in records]
    scored = [record for record in completed if record.status == "ok"]

    # Split scored records by whether the ground-truth cardinality is zero.
    # See module-level docstring for the rationale.
    true_positive = [r for r in scored if r.true_cardinality > 0]
    true_zero = [r for r in scored if r.true_cardinality == 0]

    estimates = [float(record.estimated_cardinality) for record in scored]
    statuses: dict[str, int] = defaultdict(int)
    for record in completed:
        statuses[record.status] += 1
    denominator = max(len(scored), 1)
    return {
        "query_count": len(completed),
        "scored_query_count": len(scored),
        "true_zero_matching_count": len(true_zero),
        "coverage_fraction": len(scored) / max(len(completed), 1),
        "status_counts": dict(sorted(statuses.items())),
        # --- q-error metrics (see module-level note for which to use when) ---
        "raw_q_error": percentile_summary(
            float(record.raw_q_error) for record in scored if record.raw_q_error is not None
        ),
        "raw_q_error_true_positive": percentile_summary(
            float(record.raw_q_error)
            for record in true_positive
            if record.raw_q_error is not None
        ),
        "smoothed_q_error_true_zero": percentile_summary(
            float(record.smoothed_q_error)
            for record in true_zero
            if record.smoothed_q_error is not None
        ),
        "smoothed_q_error": percentile_summary(
            float(record.smoothed_q_error)
            for record in scored
            if record.smoothed_q_error is not None
        ),
        # --- estimate sanity counters ---
        "estimate_lt_1_count": sum(value < 1.0 for value in estimates),
        "estimate_lt_1_fraction": sum(value < 1.0 for value in estimates) / denominator,
        "estimate_lt_0_1_count": sum(value < 0.1 for value in estimates),
        "estimate_lt_0_1_fraction": sum(value < 0.1 for value in estimates) / denominator,
        "estimate_lt_0_01_count": sum(value < 0.01 for value in estimates),
        "estimate_lt_0_01_fraction": sum(value < 0.01 for value in estimates) / denominator,
        "zero_estimate_count": sum(value == 0.0 for value in estimates),
    }


def summarize_latency(records: list[LatencyRecord]) -> dict[str, Any]:
    values = np.asarray([record.latency_ms for record in records], dtype=float)
    if values.size == 0:
        return {
            "observation_count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "throughput_queries_per_second": None,
        }
    total_seconds = float(values.sum() / 1000.0)
    return {
        "observation_count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_queries_per_second": float(values.size / total_seconds),
    }


# Metric families surfaced per seed and aggregated across seeds.
# Each entry is (summary_key, metric_name) where summary_key is the top-level
# key inside the per-workload accuracy dict.
_Q_ERROR_AGGREGATE_PATHS = [
    (section, metric)
    for section in (
        "raw_q_error",
        "raw_q_error_true_positive",
        "smoothed_q_error_true_zero",
        "smoothed_q_error",
    )
    for metric in ("p50", "p90", "p95", "p99", "max")
]


def aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("at least one seed summary is required")
    result: dict[str, Any] = {"seed_count": len(summaries), "metrics": {}}
    for section, metric in _Q_ERROR_AGGREGATE_PATHS:
        values = [summary[section][metric] for summary in summaries]
        if any(value is None for value in values):
            result["metrics"][f"{section}.{metric}"] = {"mean": None, "std": None}
            continue
        data = np.asarray(values, dtype=float)
        result["metrics"][f"{section}.{metric}"] = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        }
    return result
