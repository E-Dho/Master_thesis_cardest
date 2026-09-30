from __future__ import annotations

import numpy as np

from model.src.data.null_sentinel import null_sentinel_index
from model.src.data.schema import ColumnKind, ColumnMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken


def predicate_mask(column: ColumnMetadata, token: PredicateToken) -> np.ndarray:
    """Build m_i(v)=1[v satisfies token] for ordinary and indicator heads."""

    if token.op == PredicateOp.INV_FANOUT:
        raise ValueError("use reciprocal_fanout_mask for INV_FANOUT tokens")
    mask = np.array([token.satisfies(value) for value in column.domain], dtype=float)
    sentinel = null_sentinel_index(column)
    if sentinel is not None:
        # The void sentinel stands for "no such row"; it satisfies nothing, not
        # even a wildcard, so it can never contribute to a column factor.
        mask[sentinel] = 0.0
    return mask


def reciprocal_fanout_mask(column: ColumnMetadata) -> np.ndarray:
    """Return the exact fanout potential r_i(f)=1/f over the encoded domain."""

    if column.kind != ColumnKind.FANOUT:
        raise ValueError(f"column {column.name!r} is not a fanout column")
    values = np.array(column.domain, dtype=float)
    if np.any(values <= 0):
        raise ValueError(f"fanout column {column.name!r} contains non-positive values")
    return 1.0 / values


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable per-slice softmax."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def column_factor(
    distribution: np.ndarray,
    column: ColumnMetadata,
    token: PredicateToken,
) -> float:
    """Compute a_i=sum_v q_i(v)m_i(v), using 1/f for active fanout heads."""

    distribution = np.asarray(distribution, dtype=float)
    if token.op == PredicateOp.WILDCARD:
        sentinel = null_sentinel_index(column)
        if sentinel is None:
            return 1.0
        # An unpredicated column still excludes the void sentinel, so a context
        # the model believes is impossible keeps collapsing the product past the
        # bottleneck head instead of being rescued by a wildcard factor of one.
        return float(1.0 - distribution[sentinel])
    if token.op == PredicateOp.INV_FANOUT:
        mask = reciprocal_fanout_mask(column)
    else:
        mask = predicate_mask(column, token)
    return float(np.dot(distribution, mask))

