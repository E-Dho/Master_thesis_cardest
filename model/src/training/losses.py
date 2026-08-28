from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken


@dataclass(frozen=True)
class WeightStats:
    """Monitoring summary for weighted fanout-head objectives."""

    minimum: float
    maximum: float
    mean: float
    effective_sample_size: float


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS=(sum_b w_b)^2/sum_b w_b^2 for a nonnegative weight vector."""

    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights * weights))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.sum(weights)) ** 2
    return numerator / denominator


def summarize_weights(weights: np.ndarray) -> WeightStats:
    weights = np.asarray(weights, dtype=float)
    return WeightStats(
        minimum=float(np.min(weights)),
        maximum=float(np.max(weights)),
        mean=float(np.mean(weights)),
        effective_sample_size=effective_sample_size(weights),
    )


def cumulative_inverse_fanout_weights(
    encoded_rows: np.ndarray,
    tokens: list[list[PredicateToken]],
    metadata: ModelMetadata,
    *,
    compute_in_log_space: bool = True,
) -> np.ndarray:
    """Build per-head weights product_{r<i,T_r=INV_FANOUT} 1/f_r^(b).

    The current fanout's inverse never weights its own loss; it starts affecting
    later heads. Wildcard fanouts contribute one and are excluded from the
    cumulative product.
    """

    encoded_rows = np.asarray(encoded_rows, dtype=int)
    batch_size, num_columns = encoded_rows.shape
    if num_columns != len(metadata.columns):
        raise ValueError("encoded row width does not match metadata")
    if len(tokens) != batch_size or any(len(row) != num_columns for row in tokens):
        raise ValueError("tokens must have shape [batch_size, number_of_columns]")

    weights = np.ones((batch_size, num_columns), dtype=float)
    if compute_in_log_space:
        cumulative_log = np.zeros(batch_size, dtype=float)
        for column_index, column in enumerate(metadata.columns):
            weights[:, column_index] = np.exp(cumulative_log)
            if column.kind == ColumnKind.FANOUT:
                for row_index, token_row in enumerate(tokens):
                    if token_row[column_index].op == PredicateOp.INV_FANOUT:
                        encoded_value = encoded_rows[row_index, column_index]
                        fanout_value = float(column.domain[encoded_value])
                        if fanout_value <= 0:
                            raise ValueError("fanout values used in inverse weights must be positive")
                        cumulative_log[row_index] -= np.log(fanout_value)
        return weights

    cumulative = np.ones(batch_size, dtype=float)
    for column_index, column in enumerate(metadata.columns):
        weights[:, column_index] = cumulative
        if column.kind == ColumnKind.FANOUT:
            for row_index, token_row in enumerate(tokens):
                if token_row[column_index].op == PredicateOp.INV_FANOUT:
                    encoded_value = encoded_rows[row_index, column_index]
                    fanout_value = float(column.domain[encoded_value])
                    if fanout_value <= 0:
                        raise ValueError("fanout values used in inverse weights must be positive")
                    cumulative[row_index] *= 1.0 / fanout_value
    return weights


def stable_combine_importance_and_inverse_weights(
    inv_weights: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Return weights proportional to rho(x) times cumulative INV weights.

    Target tuple-sampling correction ``rho=p/q`` and query-measure INV
    reweighting serve different purposes, so the final WCE head weight is their
    product.  The normalized WCE is invariant to a common positive scale per
    head, so this combines in log space and subtracts the per-head maximum log
    weight before exponentiating.
    """

    inv = np.asarray(inv_weights, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if np.any(inv <= 0.0) or not np.all(np.isfinite(inv)):
        raise ValueError("inverse fanout weights must be finite and positive")
    if np.any(rho <= 0.0) or not np.all(np.isfinite(rho)):
        raise ValueError("importance weights rho must be finite and positive")
    log_weights = np.log(inv) + np.log(rho)[:, None]
    log_weights = log_weights - np.max(log_weights, axis=0, keepdims=True)
    return np.exp(log_weights)


def importance_weights_for_generated_contexts(
    importance_weights: object,
    generated_row_count: int,
    generation_stats: object,
) -> np.ndarray:
    """Return per-context rho values aligned with accepted generated contexts."""

    rho = np.asarray(importance_weights, dtype=float)
    if rho.ndim != 1:
        raise ValueError("importance_weights must be a one-dimensional [batch] array")
    source_row_indices = tuple(getattr(generation_stats, "source_row_indices", ()) or ())
    if source_row_indices:
        indices = np.asarray(source_row_indices, dtype=int)
        if indices.shape != (generated_row_count,):
            raise ValueError("predicate source_row_indices must match generated context count")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= rho.shape[0]):
            raise ValueError("predicate source_row_indices reference rows outside importance_weights")
        return rho[indices]
    if rho.shape == (generated_row_count,):
        return rho
    if generated_row_count % rho.shape[0] == 0:
        repeats = generated_row_count // rho.shape[0]
        return np.repeat(rho, repeats)
    raise ValueError("importance_weights cannot be aligned to generated predicate contexts")


def weighted_cross_entropy(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> float:
    """Normalized WCE=sum_b w_b[-log q_b(y_b)]/(sum_b w_b+eps)."""

    probabilities = np.asarray(probabilities, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch_size, domain_size]")
    batch_size = probabilities.shape[0]
    if target_indices.shape != (batch_size,) or weights.shape != (batch_size,):
        raise ValueError("targets and weights must have shape [batch_size]")
    selected = probabilities[np.arange(batch_size), target_indices]
    selected = np.maximum(selected, epsilon)
    denominator = float(np.sum(weights)) + epsilon
    return float(np.sum(weights * (-np.log(selected))) / denominator)


def per_head_weighted_cross_entropy(
    distributions: list[np.ndarray],
    encoded_rows: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> np.ndarray:
    """Apply normalized weighted CE independently to every output head."""

    encoded_rows = np.asarray(encoded_rows, dtype=int)
    losses = []
    for column_index, distribution in enumerate(distributions):
        losses.append(
            weighted_cross_entropy(
                distribution,
                encoded_rows[:, column_index],
                weights[:, column_index],
                epsilon=epsilon,
            )
        )
    return np.array(losses, dtype=float)
