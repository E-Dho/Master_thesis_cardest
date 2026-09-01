from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.inference.masks import factors_from_distributions
from model.src.predicates.encoding import column_factor
from model.src.predicates.operators import PredicateToken
from model.src.predicates.generation import GeneratedTrainingContext, context_satisfies_row


@dataclass(frozen=True)
class ExactDistinctTrajectoryResult:
    matching_segments_true: int
    distinct_trajectories_true: int
    a_true: float


@dataclass(frozen=True)
class ExactOracle:
    """Exact empirical oracle over a materialized full-outer-join sample."""

    metadata: ModelMetadata
    encoded_rows: np.ndarray

    def marginal_distribution(self, column_index: int, row_weights: np.ndarray | None = None) -> np.ndarray:
        weights = (
            np.ones(len(self.encoded_rows), dtype=float)
            if row_weights is None
            else np.asarray(row_weights, dtype=float)
        )
        column = self.metadata.columns[column_index]
        counts = np.zeros(column.domain_size, dtype=float)
        for encoded_value, weight in zip(self.encoded_rows[:, column_index], weights):
            counts[encoded_value] += weight
        total = np.sum(counts)
        if total == 0:
            return np.full(column.domain_size, 1.0 / column.domain_size)
        return counts / total

    def independent_factor_estimate(self, tokens: list[PredicateToken]) -> float:
        """Compute |J|prod_i a_i from ordinary empirical marginals."""

        distributions = [
            self.marginal_distribution(column_index)
            for column_index in range(len(self.metadata.columns))
        ]
        factors = factors_from_distributions(distributions, self.metadata.columns, tokens)
        return float(self.metadata.full_join_cardinality * np.prod(factors))

    def exact_weighted_product_for_fanouts(
        self,
        fanout_column_names: tuple[str, ...],
    ) -> float:
        """Compute E[prod_j 1/F_j] directly from materialized encoded rows."""

        product_values = np.ones(len(self.encoded_rows), dtype=float)
        for column_name in fanout_column_names:
            column_index = self.metadata.column_index(column_name)
            column = self.metadata.columns[column_index]
            for row_index, encoded_value in enumerate(self.encoded_rows[:, column_index]):
                product_values[row_index] *= 1.0 / float(column.domain[encoded_value])
        return float(np.mean(product_values))

    def factor_from_oracle_distribution(
        self,
        column_index: int,
        token: PredicateToken,
        row_weights: np.ndarray | None = None,
    ) -> float:
        distribution = self.marginal_distribution(column_index, row_weights)
        return column_factor(distribution, self.metadata.columns[column_index], token)

    def exact_distinct_trajectory_count(
        self,
        context: GeneratedTrainingContext,
        *,
        trajectory_ids: tuple[object, ...] | list[object] | np.ndarray,
        segment_ids: tuple[object, ...] | list[object] | np.ndarray | None = None,
    ) -> ExactDistinctTrajectoryResult:
        """Evaluate logical M_true, D_true, and a_true over materialized rows."""

        if len(trajectory_ids) != len(self.encoded_rows):
            raise ValueError("trajectory_ids must align with encoded_rows")
        if segment_ids is not None and len(segment_ids) != len(self.encoded_rows):
            raise ValueError("segment_ids must align with encoded_rows")
        matching_segments = set()
        matching_trajectories = set()
        for row_index, (row, trajectory_id) in enumerate(zip(self.encoded_rows, trajectory_ids)):
            if context_satisfies_row(context, row, self.metadata):
                segment_id = (
                    segment_ids[row_index] if segment_ids is not None else row_index
                )
                matching_segments.add(segment_id)
                matching_trajectories.add(trajectory_id)
        matching = len(matching_segments)
        distinct = len(matching_trajectories)
        return ExactDistinctTrajectoryResult(
            matching_segments_true=matching,
            distinct_trajectories_true=distinct,
            a_true=0.0 if matching == 0 else float(distinct / matching),
        )
