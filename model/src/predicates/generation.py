from __future__ import annotations

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.operators import PredicateToken


def tokens_for_query_tables(
    metadata: ModelMetadata,
    included_tables: set[str],
    inverse_fanout_columns: set[str],
    ordinary_predicates: dict[str, PredicateToken] | None = None,
) -> list[PredicateToken]:
    """Create a consistent virtual-token row for a query context.

    Included table indicators are constrained to I_T=1. Fanout columns listed in
    inverse_fanout_columns receive INV_FANOUT; all other unconstrained positions
    receive WILDCARD.
    """

    ordinary_predicates = ordinary_predicates or {}
    tokens: list[PredicateToken] = []
    for column in metadata.columns:
        if column.kind == ColumnKind.DATA:
            tokens.append(ordinary_predicates.get(column.name, PredicateToken.wildcard()))
        elif column.kind == ColumnKind.INDICATOR:
            if column.table in included_tables:
                tokens.append(PredicateToken.equal(1))
            else:
                tokens.append(PredicateToken.wildcard())
        elif column.kind == ColumnKind.FANOUT:
            if column.name in inverse_fanout_columns:
                tokens.append(PredicateToken.inv_fanout())
            else:
                tokens.append(PredicateToken.wildcard())
        else:
            raise ValueError(f"unsupported column kind {column.kind!r}")
    return tokens


def satisfied_training_tokens(
    metadata: ModelMetadata,
    decoded_row: tuple[object, ...],
    included_tables: set[str],
    inverse_fanout_columns: set[str],
) -> list[PredicateToken]:
    """Generate simple Duet-style predicates that the sampled row satisfies."""

    ordinary = {}
    for column, value in zip(metadata.columns, decoded_row):
        if column.kind == ColumnKind.DATA and value is not None:
            ordinary[column.name] = PredicateToken.equal(value)
    return tokens_for_query_tables(
        metadata,
        included_tables,
        inverse_fanout_columns,
        ordinary_predicates=ordinary,
    )

