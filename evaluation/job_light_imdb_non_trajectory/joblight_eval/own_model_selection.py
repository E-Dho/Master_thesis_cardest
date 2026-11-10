"""Eligibility checks for selected own-model checkpoint reports.

Checkpoint sweeps can contain smoke, partial, or unsupported evaluations.  A
candidate is eligible for a final A/B table only when it covers the complete
canonical workload with a score for every query.
"""

from __future__ import annotations

from typing import Any


EXPECTED_QUERY_COUNTS = {
    "job_light": 70,
    "job_light_ranges": 1000,
}


class CandidateRejected(ValueError):
    """Raised when a checkpoint result is not a complete canonical workload."""


def require_complete_candidate(workload_id: str, accuracy: dict[str, Any]) -> None:
    """Reject partial results and non-canonical workload sizes for A/B selection."""
    try:
        expected = EXPECTED_QUERY_COUNTS[workload_id]
    except KeyError as exc:
        raise CandidateRejected(f"unsupported selection workload: {workload_id}") from exc

    query_count = accuracy.get("query_count")
    scored_query_count = accuracy.get("scored_query_count")
    coverage_fraction = accuracy.get("coverage_fraction")
    if query_count != expected:
        raise CandidateRejected(
            f"{workload_id} candidate has {query_count!r} queries; expected {expected}"
        )
    if scored_query_count != expected:
        raise CandidateRejected(
            f"{workload_id} candidate scored {scored_query_count!r} queries; expected {expected}"
        )
    if coverage_fraction != 1.0:
        raise CandidateRejected(
            f"{workload_id} candidate coverage is {coverage_fraction!r}; expected 1.0"
        )
