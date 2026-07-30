from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from time import perf_counter

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.inference.masks import factors_from_distributions
from model.src.predicates.operators import PredicateToken


@dataclass(frozen=True)
class EstimateResult:
    estimated_cardinality: float
    factors: np.ndarray
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
