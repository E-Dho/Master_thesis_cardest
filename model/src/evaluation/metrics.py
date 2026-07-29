from __future__ import annotations

import numpy as np


def q_error(estimated: float, true: float, *, epsilon: float = 1.0e-12) -> float:
    """Symmetric q-error with epsilon handling for zero-cardinality queries."""

    estimated = max(float(estimated), epsilon)
    true = max(float(true), epsilon)
    return max(estimated / true, true / estimated)


def q_error_summary(estimates: list[float], truths: list[float]) -> dict[str, float]:
    errors = np.array([q_error(e, t) for e, t in zip(estimates, truths)], dtype=float)
    return {
        "median": float(np.percentile(errors, 50)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
    }

