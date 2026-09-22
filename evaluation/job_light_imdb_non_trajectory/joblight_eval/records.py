from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TableRef:
    name: str
    alias: str


@dataclass(frozen=True)
class JoinPredicate:
    left: str
    operator: str
    right: str


@dataclass(frozen=True)
class FilterPredicate:
    column: str
    operator: str
    value: Any


@dataclass(frozen=True)
class QueryRecord:
    workload: str
    query_id: int
    tables: tuple[TableRef, ...]
    joins: tuple[JoinPredicate, ...]
    filters: tuple[FilterPredicate, ...]
    true_cardinality: int
    source_line: str


@dataclass(frozen=True)
class PredictionRecord:
    workload: str
    query_id: int
    status: str
    true_cardinality: int
    estimated_cardinality: float | None
    raw_q_error: float | None = None
    smoothed_q_error: float | None = None
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyRecord:
    workload: str
    query_id: int
    repetition: int
    latency_ms: float
    scope: str = "end_to_end"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

