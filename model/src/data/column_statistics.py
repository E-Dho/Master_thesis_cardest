"""Column statistics for choosing an autoregressive order.

The ordering ablation needs a reason to put one column before another, and the
reason has to come from the data the model is trained on -- full-outer-join
rows -- rather than from the benchmark it is scored on.  Ordering columns by
anything read off a previous run's errors would make a positive result
unfalsifiable, so nothing here looks at a query, a result or a cardinality.

Two statistics are computed, both from sampled join rows:

``marginal_entropy``
    H(X_i) over the column's full domain.  How much there is to predict about
    a column before any conditioning.

``pairwise_mutual_information``
    I(X_i; X_j) between binned columns.  How much knowing one column says
    about another, which is what an autoregressive order can exploit: the
    model conditions column i on the predicates of columns before it, so a
    column that is informative about many others earns its place early.

Binning matters because ``movie_keyword:keyword_id`` has 134,171 values and a
sample of a few million rows leaves almost every cell of a 134k x 134k table
empty, where mutual information estimates noise.  Columns are binned by
frequency rank -- the ``max_bins - 1`` most common values keep their own bin
and the remaining mass shares one.  Mutual information depends on the
partition and not on the labels, so this neither needs nor benefits from
knowing which columns are ordinal, and giving every column the same bin count
keeps the upward small-sample bias comparable between them.  That bias is real
and is why the reported figures are only ever used to rank columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata


@dataclass(frozen=True)
class BinnedColumns:
    """Frequency-rank bin ids per column, with the bin count actually used."""

    assignments: np.ndarray
    bin_counts: tuple[int, ...]
    max_bins: int


def marginal_entropies(
    rows: np.ndarray, metadata: ModelMetadata, *, base: float = 2.0
) -> tuple[float, ...]:
    """Return H(X_i) per column, over each column's full domain."""

    _validate_rows(rows, metadata)
    entropies = []
    for index, column in enumerate(metadata.columns):
        counts = np.bincount(rows[:, index], minlength=column.domain_size)
        entropies.append(_entropy_from_counts(counts, base))
    return tuple(entropies)


def bin_columns(
    rows: np.ndarray, metadata: ModelMetadata, *, max_bins: int = 64
) -> BinnedColumns:
    """Bin each column by frequency rank, pooling the tail into one bin."""

    _validate_rows(rows, metadata)
    if max_bins < 2:
        raise ValueError("max_bins must be at least two for a column to vary")
    assignments = np.zeros(rows.shape, dtype=np.int32)
    bin_counts = []
    for index, column in enumerate(metadata.columns):
        counts = np.bincount(rows[:, index], minlength=column.domain_size)
        observed = int(np.count_nonzero(counts))
        if observed <= max_bins:
            # Everything present fits, so no value is pooled with another and
            # the partition is the column itself.
            keep = np.flatnonzero(counts)
        else:
            # The tail shares the last bin; argpartition keeps this linear in
            # the domain rather than sorting 134k counts.
            keep = np.argpartition(counts, -(max_bins - 1))[-(max_bins - 1):]
            keep = keep[np.argsort(-counts[keep], kind="stable")]
        lookup = np.full(column.domain_size, len(keep), dtype=np.int32)
        lookup[keep] = np.arange(len(keep), dtype=np.int32)
        assignments[:, index] = lookup[rows[:, index]]
        bin_counts.append(int(lookup.max()) + 1)
    return BinnedColumns(assignments, tuple(bin_counts), int(max_bins))


def pairwise_mutual_information(
    binned: BinnedColumns, indices: Sequence[int], *, base: float = 2.0
) -> np.ndarray:
    """Return the symmetric I(X_i; X_j) matrix over ``indices``, zero on the diagonal."""

    selected = list(int(index) for index in indices)
    matrix = np.zeros((len(selected), len(selected)), dtype=float)
    for first, left in enumerate(selected):
        for second in range(first + 1, len(selected)):
            right = selected[second]
            value = _mutual_information(
                binned.assignments[:, left],
                binned.assignments[:, right],
                binned.bin_counts[left],
                binned.bin_counts[right],
                base,
            )
            matrix[first, second] = value
            matrix[second, first] = value
    return matrix


def data_column_indices(metadata: ModelMetadata) -> tuple[int, ...]:
    """Return the positions of the DATA columns, in their current order."""

    return tuple(
        index
        for index, column in enumerate(metadata.columns)
        if column.kind == ColumnKind.DATA
    )


def reversed_ordering(order: Sequence[str], slots: Sequence[int]) -> list[str]:
    """Return ``order`` with the entries at ``slots`` in the opposite order.

    The falsification arm has to be the exact inverse of the arm it falsifies.
    Ranking the same scores ascending instead does not give that: ties break on
    current position in both directions, so two equally coupled columns keep
    their relative order in the "reversed" arm and the two arms stop being
    opposites in precisely the case where the statistic had nothing to say.
    """

    positions = sorted(int(slot) for slot in slots)
    if any(slot < 0 or slot >= len(order) for slot in positions):
        raise ValueError("slots name a position outside the order")
    moved = [order[slot] for slot in positions][::-1]
    result = list(order)
    for slot, name in zip(positions, moved):
        result[slot] = name
    return result


def ordering_from_scores(
    metadata: ModelMetadata,
    scores: Mapping[int, float],
) -> list[str]:
    """Return a full column order: scored columns ranked, the rest left in place.

    Only the scored columns move.  The indicator and fanout blocks keep their
    positions, so an arm differs from the control in the DATA order alone and
    not in where the join structure sits.  Ties break on the current position,
    so the control's order is what an uninformative statistic reproduces.
    """

    names = [column.name for column in metadata.columns]
    movable = sorted(scores, key=lambda index: (-scores[index], index))
    if any(index < 0 or index >= len(names) for index in movable):
        raise ValueError("scores name a column outside the schema")
    slots = sorted(scores)
    ordered = list(names)
    for slot, index in zip(slots, movable):
        ordered[slot] = names[index]
    return ordered


def _mutual_information(
    left: np.ndarray,
    right: np.ndarray,
    left_bins: int,
    right_bins: int,
    base: float,
) -> float:
    joint = np.bincount(
        left.astype(np.int64) * right_bins + right.astype(np.int64),
        minlength=left_bins * right_bins,
    ).reshape(left_bins, right_bins)
    total = joint.sum()
    if total == 0:
        return 0.0
    joint_probability = joint / total
    left_marginal = joint_probability.sum(axis=1, keepdims=True)
    right_marginal = joint_probability.sum(axis=0, keepdims=True)
    expected = left_marginal * right_marginal
    support = joint_probability > 0
    value = float(
        np.sum(
            joint_probability[support]
            * np.log(joint_probability[support] / expected[support])
        )
        / np.log(base)
    )
    # Mutual information is nonnegative; floating-point sums can land just
    # below zero when two columns are independent.
    return max(value, 0.0)


def _entropy_from_counts(counts: np.ndarray, base: float) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(base))


def _validate_rows(rows: np.ndarray, metadata: ModelMetadata) -> None:
    if rows.ndim != 2:
        raise ValueError(f"rows must be two-dimensional, got shape {rows.shape}")
    if rows.shape[1] != len(metadata.columns):
        raise ValueError(
            f"rows have {rows.shape[1]} columns but the schema declares "
            f"{len(metadata.columns)}"
        )
    if rows.shape[0] == 0:
        raise ValueError("no rows were drawn; statistics would describe nothing")
