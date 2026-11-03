"""Permute the column order of a prepared schema.

The model's autoregressive ordering *is* the order of ``ModelMetadata.columns``.
ResMADE derives its input and hidden degrees from column position, so column i
contributes ``P(col_i in R_i | predicates on columns < i)``: a column early in
the order has its predicate conditioning every later factor while its own factor
is an unconditioned marginal, and a column late in the order has the reverse.
Nothing else in the pipeline depends on that order being the one
``build_complete_metadata`` emits -- every consumer takes its index from
``enumerate(metadata.columns)`` or from a name-to-index map built the same way,
and no site assumes the DATA/INDICATOR/FANOUT grouping -- so varying the
ordering is a matter of rewriting the prepared artifacts rather than threading
an ordering through the model.

Rewriting the artifacts is also the only safe place to do it.  ``sample_rows.npy``
is aligned to the manifest by position and is loaded raw next to
``source.metadata`` in more than one place, so a permutation applied to metadata
at load time would misalign rows with no error raised.  Applied here, the
manifest and the rows move together, and ``validate_prepared_manifest`` -- which
checks every column's encoded values against that column's domain and runs on
every sample-source construction -- is what says so.

An identity permutation is a no-op by construction: it leaves the schema hash
unchanged, so permuting a directory by its own current order reproduces it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from model.src.data.schema import FactorizationPlan, ModelMetadata

PERMUTED_COLUMN_ORDER = "explicit_permutation"


def column_permutation(
    metadata: ModelMetadata, order: Sequence[Any]
) -> tuple[int, ...]:
    """Return each requested column's position in ``metadata``, in the new order."""

    names = tuple(column.name for column in metadata.columns)
    requested = tuple(str(name) for name in order)
    if len(set(requested)) != len(requested):
        repeated = sorted({name for name in requested if requested.count(name) > 1})
        raise ValueError(
            f"column order repeats {repeated}; a permutation names each column once"
        )
    known = set(names)
    asked = set(requested)
    missing = tuple(name for name in names if name not in asked)
    unexpected = tuple(name for name in requested if name not in known)
    if missing or unexpected:
        raise ValueError(
            "column order is not a permutation of the schema's columns: "
            f"missing {missing}, unexpected {unexpected}"
        )
    positions = {name: index for index, name in enumerate(names)}
    return tuple(positions[name] for name in requested)


def permute_metadata_columns(
    metadata: ModelMetadata, order: Sequence[Any]
) -> ModelMetadata:
    """Return ``metadata`` with its columns reordered, and the hash recomputed."""

    permutation = column_permutation(metadata, order)
    if metadata.factorization_plan.enabled:
        # The plan's output heads carry source column indices, so reordering the
        # columns under a built plan would leave every head pointing at the
        # wrong column.  Permute the prepared manifest, which carries no plan,
        # and let apply_factorization_to_metadata build one for the new order.
        raise ValueError(
            "cannot permute metadata that already carries an enabled "
            "factorization plan; permute the prepared manifest instead"
        )
    identity = permutation == tuple(range(len(permutation)))
    return ModelMetadata(
        columns=tuple(metadata.columns[index] for index in permutation),
        full_join_cardinality=metadata.full_join_cardinality,
        # Only a label -- nothing reads it -- but it is hashed, so leaving it
        # alone for the identity keeps "permute by the current order" a no-op.
        column_order=metadata.column_order if identity else PERMUTED_COLUMN_ORDER,
        upstream_attribution=dict(metadata.upstream_attribution),
        schema_hash=None,
        factorization_plan=FactorizationPlan(),
        join_root=metadata.join_root,
        join_tables=metadata.join_tables,
        join_edges=metadata.join_edges,
    )


def permute_encoded_rows(
    rows: np.ndarray, permutation: Sequence[int]
) -> np.ndarray:
    """Return ``rows`` with its columns taken in ``permutation`` order."""

    array = np.asarray(rows)
    if array.ndim != 2:
        raise ValueError(f"encoded rows must be two-dimensional, got shape {array.shape}")
    if array.shape[1] != len(permutation):
        raise ValueError(
            f"encoded rows have {array.shape[1]} columns but the permutation "
            f"names {len(permutation)}"
        )
    return array[:, list(int(index) for index in permutation)]
