from __future__ import annotations

import numpy as np

from model.src.data.schema import ColumnMetadata
from model.src.predicates.encoding import column_factor
from model.src.predicates.operators import PredicateToken


def factors_from_distributions(
    distributions: list[np.ndarray],
    columns: tuple[ColumnMetadata, ...],
    tokens: list[PredicateToken],
) -> np.ndarray:
    """Convert each q_i distribution and token T_i into scalar factor a_i."""

    if len(distributions) != len(columns) or len(tokens) != len(columns):
        raise ValueError("distributions, columns, and tokens must have equal length")
    return np.array(
        [
            column_factor(distribution, column, token)
            for distribution, column, token in zip(distributions, columns, tokens)
        ],
        dtype=float,
    )

