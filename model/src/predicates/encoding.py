from __future__ import annotations

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken


def predicate_mask(column: ColumnMetadata, token: PredicateToken) -> np.ndarray:
    """Build m_i(v)=1[v satisfies token] for ordinary and indicator heads."""

    if token.op == PredicateOp.INV_FANOUT:
        raise ValueError("use reciprocal_fanout_mask for INV_FANOUT tokens")
    return np.array([token.satisfies(value) for value in column.domain], dtype=float)


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
        return 1.0
    if token.op == PredicateOp.INV_FANOUT:
        mask = reciprocal_fanout_mask(column)
    else:
        mask = predicate_mask(column, token)
    return float(np.dot(distribution, mask))

