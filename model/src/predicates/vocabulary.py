from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken


def token_to_key(token: PredicateToken) -> str:
    """Serialize a predicate token without collapsing operator identity."""

    return json.dumps(token.stable_key(), sort_keys=True)


def token_to_legacy_key(token: PredicateToken) -> str:
    """Serialize the pre-native-range token key used by older checkpoints."""

    return json.dumps((token.op.value, token.value, token.upper), sort_keys=True)


def key_to_token(key: str) -> PredicateToken:
    pieces = json.loads(key)
    if len(pieces) == 3:
        op_value, value, upper = pieces
        return PredicateToken(PredicateOp(op_value), value=value, upper=upper)
    op_value, value, upper, lower_inclusive, upper_inclusive = pieces
    return PredicateToken(
        PredicateOp(op_value),
        value=value,
        upper=upper,
        lower_inclusive=bool(lower_inclusive),
        upper_inclusive=bool(upper_inclusive),
    )


def default_predicate_tokens(
    column: ColumnMetadata,
    *,
    include_native_ranges: bool = False,
    native_range_max_domain_size: int = 512,
) -> tuple[PredicateToken, ...]:
    """Build a conservative predicate-token domain for one modeled column."""

    if column.kind == ColumnKind.FANOUT:
        return (PredicateToken.wildcard(), PredicateToken.inv_fanout())
    if column.kind == ColumnKind.INDICATOR:
        return (PredicateToken.wildcard(), PredicateToken.equal(0), PredicateToken.equal(1))
    tokens = [PredicateToken.wildcard()]
    for value in column.domain:
        tokens.append(PredicateToken.equal(value))
        tokens.append(PredicateToken(PredicateOp.LESS_EQUAL, value=value))
        tokens.append(PredicateToken(PredicateOp.GREATER_EQUAL, value=value))
    if include_native_ranges and len(column.domain) <= native_range_max_domain_size:
        comparable_values = _comparable_domain_values(column.domain)
        for lower_index, lower in enumerate(comparable_values):
            for upper in comparable_values[lower_index:]:
                tokens.append(PredicateToken.range(lower, upper))
    return tuple(tokens)


@dataclass(frozen=True)
class PredicateVocabularies:
    """Per-column token vocabularies for ResMADE predicate inputs."""

    token_keys_by_column: tuple[tuple[str, ...], ...]

    @classmethod
    def from_metadata(
        cls,
        metadata: ModelMetadata,
        *,
        include_native_ranges: bool = False,
        native_range_max_domain_size: int = 512,
    ) -> "PredicateVocabularies":
        token_columns = []
        for column in metadata.columns:
            if column.predicate_domain is not None:
                token_columns.append(tuple(str(value) for value in column.predicate_domain))
            else:
                token_columns.append(
                    tuple(
                        token_to_key(token)
                        for token in default_predicate_tokens(
                            column,
                            include_native_ranges=include_native_ranges,
                            native_range_max_domain_size=native_range_max_domain_size,
                        )
                    )
                )
        return cls(tuple(token_columns))

    @property
    def input_bins(self) -> tuple[int, ...]:
        return tuple(len(tokens) for tokens in self.token_keys_by_column)

    def encode_token(self, column_index: int, token: PredicateToken) -> int:
        key = token_to_key(token)
        try:
            return self.token_keys_by_column[column_index].index(key)
        except ValueError as exc:
            legacy_key = token_to_legacy_key(token)
            try:
                return self.token_keys_by_column[column_index].index(legacy_key)
            except ValueError:
                raise ValueError(
                    f"token {token!r} is outside predicate vocabulary for column {column_index}"
                ) from exc

    def encode_rows(self, token_rows: list[list[PredicateToken]]) -> list[list[int]]:
        """Encode virtual query tuples as integer token IDs."""

        encoded_rows = []
        for tokens in token_rows:
            if len(tokens) != len(self.token_keys_by_column):
                raise ValueError("token row width does not match predicate vocabularies")
            encoded_rows.append(
                [self.encode_token(index, token) for index, token in enumerate(tokens)]
            )
        return encoded_rows

    def to_json_dict(self) -> dict[str, Any]:
        return {"token_keys_by_column": self.token_keys_by_column}

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "PredicateVocabularies":
        return cls(tuple(tuple(values) for values in data["token_keys_by_column"]))


def _comparable_domain_values(domain: tuple[Any, ...]) -> list[Any]:
    values = []
    for value in domain:
        if isinstance(value, str) and value.startswith("__"):
            continue
        try:
            _ = value <= value
        except TypeError:
            continue
        values.append(value)
    try:
        return sorted(values)
    except TypeError:
        return []
