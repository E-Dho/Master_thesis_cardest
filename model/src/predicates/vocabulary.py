from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken

TWO_SLOT_EMPTY_OPERATOR_ID = 0
TWO_SLOT_OP_TO_ID = {
    op: index + 1 for index, op in enumerate(PredicateOp)
}
TWO_SLOT_OPERATOR_BINS = len(TWO_SLOT_OP_TO_ID) + 1


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
    encoding_mode: str = "categorical"
    domains_by_column: tuple[tuple[Any, ...], ...] = ()

    @classmethod
    def from_metadata(
        cls,
        metadata: ModelMetadata,
        *,
        include_native_ranges: bool = False,
        native_range_max_domain_size: int = 512,
        encoding_mode: str = "categorical",
    ) -> "PredicateVocabularies":
        if encoding_mode not in {
            "categorical",
            "two_slot",
            "two_slot_binary_duet",
        }:
            raise ValueError(
                "predicate vocabulary encoding_mode must be categorical, two_slot, "
                "or two_slot_binary_duet"
            )
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
        return cls(
            tuple(token_columns),
            encoding_mode=encoding_mode,
            domains_by_column=tuple(column.domain for column in metadata.columns),
        )

    @property
    def input_bins(self) -> tuple[int, ...]:
        if self.encoding_mode in {"two_slot", "two_slot_binary_duet"}:
            return tuple(1 for _ in self.token_keys_by_column)
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

        if self.encoding_mode in {"two_slot", "two_slot_binary_duet"}:
            return self.encode_rows_two_slot(token_rows)  # type: ignore[return-value]
        encoded_rows = []
        for tokens in token_rows:
            if len(tokens) != len(self.token_keys_by_column):
                raise ValueError("token row width does not match predicate vocabularies")
            encoded_rows.append(
                [self.encode_token(index, token) for index, token in enumerate(tokens)]
            )
        return encoded_rows

    def encode_rows_two_slot(
        self,
        token_rows: list[list[PredicateToken]],
    ) -> list[list[list[int]]]:
        """Encode rows as [column][op1,value1,op2,value2] Duet-style slots."""

        encoded_rows = []
        for tokens in token_rows:
            if len(tokens) != len(self.token_keys_by_column):
                raise ValueError("token row width does not match predicate vocabularies")
            encoded_rows.append(
                [self.encode_token_two_slot(index, token) for index, token in enumerate(tokens)]
            )
        return encoded_rows

    def encode_token_two_slot(self, column_index: int, token: PredicateToken) -> list[int]:
        domain = self.domains_by_column[column_index]
        missing_value_id = len(domain)
        empty = [TWO_SLOT_EMPTY_OPERATOR_ID, missing_value_id]
        if token.op == PredicateOp.WILDCARD:
            return [*empty, *empty]
        if token.op == PredicateOp.RANGE:
            lower_op = (
                PredicateOp.GREATER_EQUAL
                if token.lower_inclusive
                else PredicateOp.GREATER_THAN
            )
            upper_op = (
                PredicateOp.LESS_EQUAL
                if token.upper_inclusive
                else PredicateOp.LESS_THAN
            )
            return [
                TWO_SLOT_OP_TO_ID[lower_op],
                _value_id(domain, token.value),
                TWO_SLOT_OP_TO_ID[upper_op],
                _value_id(domain, token.upper),
            ]
        if token.op in TWO_SLOT_OP_TO_ID:
            value_id = (
                missing_value_id
                if token.op == PredicateOp.INV_FANOUT
                else _value_id(domain, token.value)
            )
            return [TWO_SLOT_OP_TO_ID[token.op], value_id, *empty]
        raise ValueError(f"unsupported predicate token {token!r}")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "token_keys_by_column": self.token_keys_by_column,
            "encoding_mode": self.encoding_mode,
            "domains_by_column": self.domains_by_column,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "PredicateVocabularies":
        return cls(
            tuple(tuple(values) for values in data["token_keys_by_column"]),
            encoding_mode=str(data.get("encoding_mode", "categorical")),
            domains_by_column=tuple(
                tuple(values) for values in data.get("domains_by_column", ())
            ),
        )


def two_slot_value_bins_by_column(metadata: ModelMetadata) -> tuple[int, ...]:
    return tuple(column.domain_size + 1 for column in metadata.columns)


def binary_literal_width(domain_size: int) -> int:
    """Return Duet-style LSB-first binary width for a column domain."""

    if domain_size <= 0:
        raise ValueError("domain_size must be positive")
    return max(1, (int(domain_size) - 1).bit_length())


def two_slot_binary_widths_by_column(metadata: ModelMetadata) -> tuple[int, ...]:
    return tuple(binary_literal_width(column.domain_size) for column in metadata.columns)


def binary_bits_lsb(value_id: int, width: int) -> tuple[int, ...]:
    """Encode a nonnegative integer as least-significant-bit-first bits."""

    if value_id < 0:
        raise ValueError("value_id must be nonnegative")
    if width <= 0:
        raise ValueError("width must be positive")
    return tuple((int(value_id) >> bit) & 1 for bit in range(width))


def _value_id(domain: tuple[Any, ...], value: Any) -> int:
    try:
        return domain.index(value)
    except ValueError as exc:
        raise ValueError(f"value {value!r} is outside predicate slot domain") from exc


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
