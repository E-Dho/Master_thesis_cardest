from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, WorkloadConfig
from ..records import LatencyRecord, PredictionRecord


@dataclass(frozen=True)
class StageMetrics:
    wall_seconds: float = 0.0
    peak_rss_bytes: int | None = None
    peak_gpu_allocated_bytes: int | None = None
    peak_gpu_reserved_bytes: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingOutcome:
    """Result of a profile-controlled, timing-only evaluation."""

    predictions: tuple[PredictionRecord, ...]
    latencies: tuple[LatencyRecord, ...]
    #: workload id -> timing-guard report written by the measuring process
    guard_reports: dict[str, Path]
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterEvaluation:
    predictions: tuple[PredictionRecord, ...]
    latencies: tuple[LatencyRecord, ...]
    detail: dict[str, Any] = field(default_factory=dict)


class Adapter(ABC):
    def __init__(self, config: ExperimentConfig, seed: int, run_directory: Path) -> None:
        self.config = config
        self.seed = seed
        self.run_directory = run_directory

    @abstractmethod
    def prepare(self) -> StageMetrics:
        raise NotImplementedError

    @abstractmethod
    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        raise NotImplementedError

    @abstractmethod
    def build(self) -> StageMetrics:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, workloads: tuple[WorkloadConfig, ...]) -> AdapterEvaluation:
        raise NotImplementedError

    @abstractmethod
    def artifact_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def time(
        self,
        profile: str,
        workloads: tuple[WorkloadConfig, ...],
        output_directory: Path,
        environment: dict[str, str],
    ) -> TimingOutcome:
        """Re-measure latency under ``profile`` without rebuilding the model."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support profile-controlled timing"
        )
