from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from time import perf_counter

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.data.trajectory_distinct import (
    TrajectoryDistinctNotApplicable,
    TrajectoryDistinctRuntimeConfig,
    trajectory_distinct_context_eligibility,
)
from model.src.predicates.generation import GeneratedTrainingContext
from model.src.inference.masks import factors_from_distributions
from model.src.predicates.operators import PredicateToken


@dataclass(frozen=True)
class EstimateResult:
    estimated_cardinality: float
    factors: np.ndarray
    latency_seconds: float


@dataclass(frozen=True)
class DistinctTrajectoryEstimate:
    matching_segment_estimate: float
    traj_dedup_factor: float
    distinct_trajectory_estimate: float
    model_forward_calls: int
    latency_seconds: float


class OnePassEstimator:
    """Sampling-free estimator using one predicate-conditioned model pass."""

    def __init__(self, model: object, metadata: ModelMetadata) -> None:
        self.model = model
        self.metadata = metadata

    def estimate(
        self,
        tokens: list[PredicateToken],
        *,
        use_log_space_product: bool = True,
    ) -> EstimateResult:
        """Return |J| prod_i a_i, optionally accumulated as log|J|+sum log a_i."""

        start = perf_counter()
        if hasattr(self.model, "predict_column_factors"):
            factors = np.asarray(self.model.predict_column_factors(tokens), dtype=float)
        else:
            distributions = self.model.predict_distributions(tokens)
            factors = factors_from_distributions(distributions, self.metadata.columns, tokens)
        if use_log_space_product:
            if self.metadata.full_join_cardinality == 0 or np.any(factors == 0):
                estimate = 0.0
            else:
                estimate = exp(
                    log(self.metadata.full_join_cardinality)
                    + float(np.sum(np.log(factors)))
                )
        else:
            estimate = float(self.metadata.full_join_cardinality * np.prod(factors))
        latency = perf_counter() - start
        if estimate < 0 or not np.isfinite(estimate):
            raise ValueError(f"invalid cardinality estimate {estimate!r}")
        return EstimateResult(estimate, factors, latency)

    def estimate_distinct_trajectories(
        self,
        tokens: list[PredicateToken],
        *,
        context: GeneratedTrainingContext | None = None,
        included_tables: set[str] | frozenset[str] | None = None,
        trajectory_config: TrajectoryDistinctRuntimeConfig | None = None,
        use_log_space_product: bool = True,
    ) -> DistinctTrajectoryEstimate:
        """Return D_hat = M_hat * traj_dedup_factor for segment-query workloads."""

        if not hasattr(self.model, "predict_column_factors_and_traj_dedup"):
            raise ValueError("model does not expose traj_dedup_factor inference")
        runtime_config = trajectory_config or TrajectoryDistinctRuntimeConfig()
        if context is None:
            if included_tables is None:
                raise TrajectoryDistinctNotApplicable(
                    "distinct trajectory inference requires query included_tables"
                )
            context = GeneratedTrainingContext(
                tokens=tuple(tokens),
                included_tables=frozenset(included_tables),
                inverse_fanout_columns=frozenset(),
                ordinary_predicates={},
            )
        eligibility = trajectory_distinct_context_eligibility(
            context,
            self.metadata,
            runtime_config,
        )
        if not eligibility.eligible:
            raise TrajectoryDistinctNotApplicable(
                eligibility.reason or "trajectory distinct is not applicable"
            )
        start_calls = int(getattr(getattr(self.model, "resmade", self.model), "forward_calls", 0))
        start = perf_counter()
        factors, traj_dedup_factor = self.model.predict_column_factors_and_traj_dedup(tokens)
        if use_log_space_product:
            if self.metadata.full_join_cardinality == 0 or np.any(factors == 0):
                matching_segment_estimate = 0.0
            else:
                matching_segment_estimate = exp(
                    log(self.metadata.full_join_cardinality)
                    + float(np.sum(np.log(factors)))
                )
        else:
            matching_segment_estimate = float(
                self.metadata.full_join_cardinality * np.prod(factors)
            )
        distinct = float(matching_segment_estimate * traj_dedup_factor)
        latency = perf_counter() - start
        end_calls = int(getattr(getattr(self.model, "resmade", self.model), "forward_calls", 0))
        if not (0.0 <= traj_dedup_factor <= 1.0):
            raise ValueError(f"invalid traj_dedup_factor {traj_dedup_factor!r}")
        if distinct < -1.0e-12 or distinct > matching_segment_estimate + 1.0e-9:
            raise ValueError("distinct trajectory estimate violates 0 <= D_hat <= M_hat")
        return DistinctTrajectoryEstimate(
            matching_segment_estimate=matching_segment_estimate,
            traj_dedup_factor=traj_dedup_factor,
            distinct_trajectory_estimate=distinct,
            model_forward_calls=end_calls - start_calls,
            latency_seconds=latency,
        )
