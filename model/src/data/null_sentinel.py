"""Null-token (void sentinel) support for zero-cardinality training.

The autoregressive estimator is trained exclusively on predicate contexts that
are satisfied by a sampled full-outer-join row, so it never observes a context
whose conjunction matches nothing.  At inference the per-column factors are
therefore free to stay comfortably above zero for impossible queries, and the
product ``N * prod_i a_i`` inherits that optimism.

Single-column impossibility is already exact: ``_canonical_data_token`` in the
POL evaluation adapter rewrites a literal that no domain value satisfies into
``GREATER_THAN max(domain)``, whose mask is identically zero.  What remains are
*conjunction voids* -- every conjunct individually satisfiable, the conjunction
empty.  Those dominate the true-zero tail of the trajectory benchmark.

This module supplies the two halves of the null-token remedy:

``apply_null_sentinel_to_metadata``
    Appends one extra value to every eligible DATA column domain.  The token is
    ``"__VOID__"``; the leading underscores mean the existing sentinel handling
    in :mod:`model.src.model.output_adapter` (``_non_sentinel_bounds``,
    ``_subtract_excluded_edge_sentinels``) already excludes it from ordered
    interval arithmetic without further changes.

``VoidPairCatalog``
    A co-occurrence catalog over the materialized sample rows.  A pair of
    encoded values that never co-occurs, where both values are individually
    well supported, is a provable conjunction void and can be turned into a
    training context whose target is the sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata

NULL_SENTINEL_TOKEN = "__VOID__"

# Chunk size for the co-occurrence scan.  Bounded so a memory-mapped 50M-row
# fixture is never materialized in full.
_CATALOG_CHUNK_ROWS = 1_000_000


@dataclass(frozen=True)
class NullSentinelConfig:
    """Runtime configuration for null-token zero-cardinality training."""

    enabled: bool = False
    void_probability: float = 0.1
    chain_extension_probability: float = 0.2
    max_chain_extensions: int = 5
    max_catalog_domain_size: int = 512
    max_catalog_columns: int = 16
    min_marginal_count: int = 32
    min_anchor_partners: int = 4
    max_catalog_rows: int | None = None
    cascade: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NullSentinelConfig":
        data = data or {}
        raw_rows = data.get("max_catalog_rows")
        return cls(
            enabled=bool(data.get("enabled", False)),
            void_probability=float(data.get("void_probability", 0.1)),
            chain_extension_probability=float(
                data.get("chain_extension_probability", 0.2)
            ),
            max_chain_extensions=int(data.get("max_chain_extensions", 5)),
            max_catalog_domain_size=int(data.get("max_catalog_domain_size", 512)),
            max_catalog_columns=int(data.get("max_catalog_columns", 16)),
            min_marginal_count=int(data.get("min_marginal_count", 32)),
            min_anchor_partners=int(data.get("min_anchor_partners", 4)),
            max_catalog_rows=None if raw_rows is None else int(raw_rows),
            cascade=bool(data.get("cascade", True)),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not 0.0 <= self.void_probability <= 1.0:
            raise ValueError("null_sentinel.void_probability must lie in [0, 1]")
        if not 0.0 <= self.chain_extension_probability < 1.0:
            raise ValueError(
                "null_sentinel.chain_extension_probability must lie in [0, 1)"
            )
        if self.max_chain_extensions < 0:
            raise ValueError("null_sentinel.max_chain_extensions must be nonnegative")
        if self.max_catalog_domain_size < 2:
            raise ValueError("null_sentinel.max_catalog_domain_size must exceed one")
        if self.max_catalog_columns < 2:
            raise ValueError(
                "null_sentinel.max_catalog_columns must admit at least one pair"
            )
        if self.min_marginal_count < 1:
            raise ValueError("null_sentinel.min_marginal_count must be positive")
        if self.min_anchor_partners < 1:
            raise ValueError("null_sentinel.min_anchor_partners must be positive")
        if self.max_catalog_rows is not None and self.max_catalog_rows < 1:
            raise ValueError("null_sentinel.max_catalog_rows must be positive or null")


def column_has_null_sentinel(column: ColumnMetadata) -> bool:
    """Report whether this column carries the void sentinel as its last value."""

    return bool(column.domain) and column.domain[-1] == NULL_SENTINEL_TOKEN


def null_sentinel_index(column: ColumnMetadata) -> int | None:
    """Return the encoded id of the void sentinel, or None when absent."""

    if not column_has_null_sentinel(column):
        return None
    return column.domain_size - 1


def null_sentinel_indices(metadata: ModelMetadata) -> tuple[int | None, ...]:
    """Return per-column sentinel ids aligned with ``metadata.columns``."""

    return tuple(null_sentinel_index(column) for column in metadata.columns)


def _column_is_eligible(column: ColumnMetadata) -> bool:
    if column.kind != ColumnKind.DATA:
        return False
    return not column_has_null_sentinel(column)


def apply_null_sentinel_to_metadata(
    metadata: ModelMetadata,
    config: NullSentinelConfig,
) -> ModelMetadata:
    """Return metadata whose DATA column domains carry a trailing void sentinel.

    The sentinel is appended rather than inserted so every pre-existing encoded
    id keeps its meaning; prepared fixtures written before this feature stay
    valid and need no re-preparation.  ``predicate_domain`` is left untouched
    because the sentinel is an output-side target, never a predicate literal.
    """

    config.validate()
    if not config.enabled:
        return metadata
    columns = tuple(
        ColumnMetadata(
            name=column.name,
            kind=column.kind,
            domain=column.domain + (NULL_SENTINEL_TOKEN,),
            table=column.table,
            predicate_domain=column.predicate_domain,
            fanout_source=column.fanout_source,
            factorization=column.factorization,
        )
        if _column_is_eligible(column)
        else column
        for column in metadata.columns
    )
    return ModelMetadata(
        columns=columns,
        full_join_cardinality=metadata.full_join_cardinality,
        column_order=metadata.column_order,
        upstream_attribution=metadata.upstream_attribution,
        schema_hash=None,
        factorization_plan=metadata.factorization_plan,
        join_root=metadata.join_root,
        join_tables=metadata.join_tables,
        join_edges=metadata.join_edges,
    )


def sentinel_width_changes(metadata: ModelMetadata) -> tuple[str, ...]:
    """Name DATA columns whose bit width grew when the sentinel was appended.

    Appending one value is free whenever the real domain size is not an exact
    power of two, because the bitwise factorization already carries slack that
    the sentinel simply occupies.  When it *is* a power of two the column gains
    a bit, which widens heads and can even pull a previously atomic column onto
    the factorized path.  That is not an error, but it changes the checkpoint
    layout, so the run log should say so rather than let it surprise someone
    comparing parameter counts against a baseline.
    """

    changed: list[str] = []
    for column in metadata.columns:
        if not column_has_null_sentinel(column):
            continue
        real_size = column.domain_size - 1
        if real_size < 1:
            continue
        if (real_size & (real_size - 1)) == 0 and real_size > 1:
            changed.append(column.name)
    return tuple(changed)


@dataclass(frozen=True)
class VoidPair:
    """One provable conjunction void drawn from the co-occurrence catalog."""

    anchor_column_index: int
    anchor_value_id: int
    void_column_index: int
    void_value_id: int


class VoidPairCatalog:
    """Boolean co-occurrence matrices over small-domain DATA columns.

    For every eligible ordered column pair ``(i, j)`` with ``i < j`` the catalog
    stores ``present[u, v]``: whether any scanned row carries encoded value
    ``u`` in column ``i`` and ``v`` in column ``j``.  A cell that stayed False
    while both ``u`` and ``v`` are individually well supported is a conjunction
    that no row satisfies, which is exactly the training signal the model is
    missing.

    The marginal-support floor matters.  Without it a False cell may simply mean
    that ``u`` or ``v`` is itself rare, and the resulting "void" would teach the
    model that a merely uncommon combination is impossible.
    """

    def __init__(
        self,
        *,
        column_indices: tuple[int, ...],
        domain_sizes: dict[int, int],
        present: dict[tuple[int, int], np.ndarray],
        marginals: dict[int, np.ndarray],
        scanned_rows: int,
        min_marginal_count: int,
        min_anchor_partners: int = 1,
    ) -> None:
        self.column_indices = column_indices
        self.domain_sizes = domain_sizes
        self._present = present
        self._marginals = marginals
        self.scanned_rows = int(scanned_rows)
        self.min_marginal_count = int(min_marginal_count)
        self.min_anchor_partners = int(min_anchor_partners)
        self._supported: dict[int, np.ndarray] = {
            index: marginal >= min_marginal_count
            for index, marginal in marginals.items()
        }
        self._pairs: tuple[tuple[int, int], ...] = tuple(sorted(present.keys()))

    @property
    def pair_count(self) -> int:
        return len(self._pairs)

    @property
    def void_cell_count(self) -> int:
        """Count catalog cells that are both absent and well supported."""

        total = 0
        for (left, right), present in self._present.items():
            supported = np.outer(self._supported[left], self._supported[right])
            total += int(np.count_nonzero(supported & ~present))
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "scanned_rows": self.scanned_rows,
            "catalog_columns": len(self.column_indices),
            "catalog_pairs": self.pair_count,
            "void_cells": self.void_cell_count,
            "min_marginal_count": self.min_marginal_count,
            "min_anchor_partners": self.min_anchor_partners,
        }

    def present_partner_count(
        self,
        anchor_column_index: int,
        anchor_value_id: int,
        void_column_index: int,
    ) -> int:
        """Count supported void-column values this anchor is seen together with."""

        row = self._co_occurrence_row(
            anchor_column_index, anchor_value_id, void_column_index
        )
        if row is None:
            return 0
        return int(np.count_nonzero(self._supported[void_column_index] & row))

    def _co_occurrence_row(
        self,
        anchor_column_index: int,
        anchor_value_id: int,
        void_column_index: int,
    ) -> np.ndarray | None:
        key = (
            (anchor_column_index, void_column_index)
            if anchor_column_index < void_column_index
            else (void_column_index, anchor_column_index)
        )
        present = self._present.get(key)
        if present is None:
            return None
        if anchor_column_index < void_column_index:
            if anchor_value_id >= present.shape[0]:
                return None
            return present[anchor_value_id]
        if anchor_value_id >= present.shape[1]:
            return None
        return present[:, anchor_value_id]

    def absent_values_for(
        self,
        anchor_column_index: int,
        anchor_value_id: int,
        void_column_index: int,
    ) -> np.ndarray:
        """Return void-column values that never co-occur with the anchor value."""

        row = self._co_occurrence_row(
            anchor_column_index, anchor_value_id, void_column_index
        )
        if row is None:
            return np.empty(0, dtype=np.int64)
        supported = self._supported[void_column_index]
        return np.flatnonzero(supported & ~row).astype(np.int64, copy=False)

    def sample_void_pair(
        self,
        encoded_row: Sequence[int],
        rng: np.random.Generator,
        *,
        max_attempts: int = 8,
    ) -> VoidPair | None:
        """Draw a void whose anchor value is the one this row actually carries.

        Anchoring on the row keeps the prefix genuine: columns before the void
        column keep their real values, so the only thing the model is asked to
        learn is that the void column cannot follow that prefix under the void
        predicate.
        """

        if not self._pairs:
            return None
        for _ in range(max_attempts):
            pair_index = int(rng.integers(0, len(self._pairs)))
            left, right = self._pairs[pair_index]
            # Either column of the pair may serve as the anchor; the other one
            # becomes the void column.  Prefer the later column as the void so
            # the bottleneck sits as deep in the chain as possible.
            if bool(rng.integers(0, 2)):
                anchor_index, void_index = left, right
            else:
                anchor_index, void_index = right, left
            anchor_value = int(encoded_row[anchor_index])
            if anchor_value >= self.domain_sizes[anchor_index]:
                continue
            if not bool(self._supported[anchor_index][anchor_value]):
                continue
            # An anchor that co-occurs with only a handful of partners would be
            # void for almost every alternative, and the head learns the cheaper
            # rule "this anchor implies void" instead of "this pair is
            # impossible" -- which then suppresses the anchor's genuine
            # combinations too.  Require a branch wide enough that voiding one
            # partner cannot stand in for voiding the anchor.
            if (
                self.present_partner_count(anchor_index, anchor_value, void_index)
                < self.min_anchor_partners
            ):
                continue
            candidates = self.absent_values_for(anchor_index, anchor_value, void_index)
            if candidates.size == 0:
                continue
            void_value = int(candidates[int(rng.integers(0, candidates.size))])
            return VoidPair(
                anchor_column_index=anchor_index,
                anchor_value_id=anchor_value,
                void_column_index=void_index,
                void_value_id=void_value,
            )
        return None


def eligible_catalog_columns(
    metadata: ModelMetadata,
    config: NullSentinelConfig,
) -> tuple[int, ...]:
    """Select DATA columns small enough to carry a dense co-occurrence matrix.

    Columns are ranked by domain size ascending so the cheapest and most
    strongly supported columns win the budget; those are also the nominal
    columns that dominate the benchmark's true-zero conjunctions.
    """

    candidates: list[tuple[int, int]] = []
    for index, column in enumerate(metadata.columns):
        if column.kind != ColumnKind.DATA:
            continue
        size = _real_domain_size(column)
        if size < 2 or size > config.max_catalog_domain_size:
            continue
        candidates.append((size, index))
    candidates.sort()
    selected = [index for _, index in candidates[: config.max_catalog_columns]]
    return tuple(sorted(selected))


def _real_domain_size(column: ColumnMetadata) -> int:
    """Domain size excluding a trailing void sentinel."""

    if column_has_null_sentinel(column):
        return column.domain_size - 1
    return column.domain_size


def _row_chunks(
    rows: np.ndarray,
    max_rows: int | None,
) -> Iterator[np.ndarray]:
    total = int(rows.shape[0])
    if max_rows is not None:
        total = min(total, max_rows)
    start = 0
    while start < total:
        stop = min(start + _CATALOG_CHUNK_ROWS, total)
        yield np.asarray(rows[start:stop])
        start = stop


def build_void_pair_catalog(
    rows: np.ndarray,
    metadata: ModelMetadata,
    config: NullSentinelConfig,
) -> VoidPairCatalog | None:
    """Scan materialized rows once and record which value pairs ever co-occur.

    Returns None when no column pair is eligible, which makes void injection a
    no-op rather than an error: a schema without small categorical columns has
    nothing this catalog can prove.
    """

    config.validate()
    if rows is None or rows.size == 0:
        return None
    column_indices = eligible_catalog_columns(metadata, config)
    if len(column_indices) < 2:
        return None
    domain_sizes = {
        index: _real_domain_size(metadata.columns[index]) for index in column_indices
    }
    marginals = {
        index: np.zeros(domain_sizes[index], dtype=np.int64) for index in column_indices
    }
    pairs = [
        (left, right)
        for position, left in enumerate(column_indices)
        for right in column_indices[position + 1 :]
    ]
    present = {
        (left, right): np.zeros(
            (domain_sizes[left], domain_sizes[right]), dtype=bool
        )
        for left, right in pairs
    }
    scanned = 0
    for chunk in _row_chunks(rows, config.max_catalog_rows):
        scanned += int(chunk.shape[0])
        columns = {index: chunk[:, index].astype(np.int64, copy=False) for index in column_indices}
        valid = {
            index: (values >= 0) & (values < domain_sizes[index])
            for index, values in columns.items()
        }
        for index in column_indices:
            selected = columns[index][valid[index]]
            if selected.size:
                marginals[index] += np.bincount(
                    selected, minlength=domain_sizes[index]
                ).astype(np.int64, copy=False)
        for left, right in pairs:
            mask = valid[left] & valid[right]
            if not np.any(mask):
                continue
            width = domain_sizes[right]
            flat = columns[left][mask] * width + columns[right][mask]
            seen = np.bincount(flat, minlength=domain_sizes[left] * width) > 0
            present[(left, right)] |= seen.reshape(domain_sizes[left], width)
    catalog = VoidPairCatalog(
        column_indices=column_indices,
        domain_sizes=domain_sizes,
        present=present,
        marginals=marginals,
        scanned_rows=scanned,
        min_marginal_count=config.min_marginal_count,
        min_anchor_partners=config.min_anchor_partners,
    )
    if catalog.void_cell_count == 0:
        return None
    return catalog


def sample_chain_extension_count(
    rng: np.random.Generator,
    config: NullSentinelConfig,
) -> int:
    """Draw how many extra row-satisfied predicates ride along with the void.

    Each step extends with probability ``chain_extension_probability``, so short
    voids dominate while longer ones still appear -- matching the benchmark,
    where true-zero queries carry two to seven predicates.  Adding further
    row-satisfied conjuncts can never make an empty conjunction non-empty, so
    the extension preserves provability.
    """

    if config.chain_extension_probability <= 0.0:
        return 0
    count = 0
    while count < config.max_chain_extensions:
        if rng.random() >= config.chain_extension_probability:
            break
        count += 1
    return count
