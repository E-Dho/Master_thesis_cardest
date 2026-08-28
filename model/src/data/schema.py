from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any


class ColumnKind(str, Enum):
    """Modeled column categories used by the estimator and loss weighting."""

    DATA = "data"
    INDICATOR = "indicator"
    FANOUT = "fanout"


@dataclass(frozen=True)
class OutputHeadSpec:
    """One model output head, possibly representing a factor of an original column."""

    name: str
    source_column_index: int
    factor_index: int | None
    domain_size: int


@dataclass(frozen=True)
class FactorColumnMetadata:
    """Metadata for one lossless bit factor of an original dictionary ID."""

    name: str
    original_column_index: int
    factor_index: int
    domain_size: int
    bit_width: int
    bit_offset: int
    radix: int
    is_last_factor: bool


@dataclass(frozen=True)
class OriginalColumnFactorization:
    """Lossless mapping between one original column and its factor output heads."""

    original_column_index: int
    factor_column_indices: tuple[int, ...]
    original_domain_size: int
    factor_order: str
    factor_domains: tuple[int, ...]
    bit_widths: tuple[int, ...]
    bit_offsets: tuple[int, ...]
    radices: tuple[int, ...]
    invalid_combination_count: int = 0


@dataclass(frozen=True)
class FactorizationPlan:
    """Immutable schema-level plan for original-column-preserving factorization."""

    enabled: bool = False
    strategy: str = "none"
    word_size_bits: int = 0
    minimum_domain_size: int = 0
    factor_order: str = "none"
    blacklist_columns: tuple[str, ...] = ()
    blacklist_kinds: tuple[str, ...] = ()
    original_column_factorizations: tuple[OriginalColumnFactorization, ...] = ()
    factor_columns: tuple[FactorColumnMetadata, ...] = ()
    output_head_specs: tuple[OutputHeadSpec, ...] = ()
    original_output_width: int = 0
    factorized_output_width: int = 0
    metadata_version: int = 1

    def factorization_for_column(
        self, original_column_index: int
    ) -> OriginalColumnFactorization | None:
        for factorization in self.original_column_factorizations:
            if factorization.original_column_index == original_column_index:
                return factorization
        return None

    def output_heads_for_column(self, original_column_index: int) -> tuple[int, ...]:
        return tuple(
            index
            for index, spec in enumerate(self.output_head_specs)
            if spec.source_column_index == original_column_index
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any] | None) -> "FactorizationPlan":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            strategy=str(data.get("strategy", "none")),
            word_size_bits=int(data.get("word_size_bits", 0)),
            minimum_domain_size=int(data.get("minimum_domain_size", 0)),
            factor_order=str(data.get("factor_order", "none")),
            blacklist_columns=tuple(data.get("blacklist_columns", ())),
            blacklist_kinds=tuple(data.get("blacklist_kinds", ())),
            original_column_factorizations=tuple(
                OriginalColumnFactorization(
                    original_column_index=int(item["original_column_index"]),
                    factor_column_indices=tuple(item["factor_column_indices"]),
                    original_domain_size=int(item["original_domain_size"]),
                    factor_order=str(item["factor_order"]),
                    factor_domains=tuple(item["factor_domains"]),
                    bit_widths=tuple(item["bit_widths"]),
                    bit_offsets=tuple(item["bit_offsets"]),
                    radices=tuple(item["radices"]),
                    invalid_combination_count=int(item.get("invalid_combination_count", 0)),
                )
                for item in data.get("original_column_factorizations", ())
            ),
            factor_columns=tuple(
                FactorColumnMetadata(
                    name=str(item["name"]),
                    original_column_index=int(item["original_column_index"]),
                    factor_index=int(item["factor_index"]),
                    domain_size=int(item["domain_size"]),
                    bit_width=int(item["bit_width"]),
                    bit_offset=int(item["bit_offset"]),
                    radix=int(item["radix"]),
                    is_last_factor=bool(item["is_last_factor"]),
                )
                for item in data.get("factor_columns", ())
            ),
            output_head_specs=tuple(
                OutputHeadSpec(
                    name=str(item["name"]),
                    source_column_index=int(item["source_column_index"]),
                    factor_index=(
                        None if item.get("factor_index") is None else int(item["factor_index"])
                    ),
                    domain_size=int(item["domain_size"]),
                )
                for item in data.get("output_head_specs", ())
            ),
            original_output_width=int(data.get("original_output_width", 0)),
            factorized_output_width=int(data.get("factorized_output_width", 0)),
            metadata_version=int(data.get("metadata_version", 1)),
        )


@dataclass(frozen=True)
class FactorizationMetadata:
    """Future-proof metadata for lossless factorization without enabling it."""

    original_column: str | None = None
    subcolumns: tuple[str, ...] = ()
    subcolumn_domain_sizes: tuple[int, ...] = ()
    reconstruction_order: tuple[int, ...] = ()
    radix: int | None = None
    mapping_name: str | None = None


@dataclass(frozen=True)
class ColumnMetadata:
    """Domain and ordering metadata for one modeled output head."""

    name: str
    kind: ColumnKind
    domain: tuple[Any, ...]
    table: str | None = None
    predicate_domain: tuple[Any, ...] | None = None
    fanout_source: str | None = None
    factorization: FactorizationMetadata = field(default_factory=FactorizationMetadata)

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError(f"column {self.name!r} must have a non-empty domain")
        if self.kind == ColumnKind.FANOUT:
            for fanout_value in self.domain:
                if float(fanout_value) <= 0:
                    raise ValueError(
                        f"fanout column {self.name!r} contains non-positive value "
                        f"{fanout_value!r}"
                    )

    @property
    def domain_size(self) -> int:
        return len(self.domain)

    @property
    def predicate_domain_size(self) -> int:
        if self.predicate_domain is not None:
            return len(self.predicate_domain)
        return len(self.domain)

    @cached_property
    def _hashable_value_to_id(self) -> dict[Any, int]:
        lookup: dict[Any, int] = {}
        for index, value in enumerate(self.domain):
            try:
                hash(value)
            except TypeError:
                continue
            lookup[value] = index
        return lookup

    def encode_value(self, value: Any) -> int:
        try:
            return self._hashable_value_to_id[value]
        except TypeError:
            pass
        except KeyError:
            pass
        for index, candidate in enumerate(self.domain):
            if candidate == value:
                return index
        raise ValueError(
            f"value {value!r} is outside domain for column {self.name!r}"
        )


@dataclass(frozen=True)
class ModelMetadata:
    """Checkpoint metadata required to preserve column order and domains."""

    columns: tuple[ColumnMetadata, ...]
    full_join_cardinality: float
    column_order: str = "data_indicators_fanouts"
    upstream_attribution: dict[str, str] = field(default_factory=dict)
    schema_hash: str | None = None
    factorization_plan: FactorizationPlan = field(default_factory=FactorizationPlan)
    join_root: str | None = None
    join_tables: tuple[str, ...] = ()
    join_edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.full_join_cardinality < 0:
            raise ValueError("full_join_cardinality must be nonnegative")
        seen: set[str] = set()
        for column in self.columns:
            if column.name in seen:
                raise ValueError(f"duplicate modeled column name {column.name!r}")
            seen.add(column.name)

    def column_index(self, name: str) -> int:
        for index, column in enumerate(self.columns):
            if column.name == name:
                return index
        raise KeyError(name)

    def fanout_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, column in enumerate(self.columns)
            if column.kind == ColumnKind.FANOUT
        )

    @property
    def data_output_bins(self) -> tuple[int, ...]:
        return tuple(column.domain_size for column in self.columns)

    @property
    def model_output_bins(self) -> tuple[int, ...]:
        if self.factorization_plan.enabled:
            return tuple(spec.domain_size for spec in self.factorization_plan.output_head_specs)
        return self.data_output_bins

    @property
    def predicate_input_bins(self) -> tuple[int, ...]:
        return tuple(column.predicate_domain_size for column in self.columns)

    @property
    def output_slices(self) -> tuple[tuple[int, int], ...]:
        slices = []
        start = 0
        for column in self.columns:
            stop = start + column.domain_size
            slices.append((start, stop))
            start = stop
        return tuple(slices)

    @property
    def model_output_slices(self) -> tuple[tuple[int, int], ...]:
        widths = self.model_output_bins
        slices = []
        start = 0
        for width in widths:
            stop = start + width
            slices.append((start, stop))
            start = stop
        return tuple(slices)

    def stable_schema_hash(self) -> str:
        """Hash the ordering, domains, predicate domains, and join cardinality."""

        payload = json.dumps(self.to_json_dict(include_hash=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for column in data["columns"]:
            column["kind"] = column["kind"].value
        data["factorization_plan"] = self.factorization_plan.to_json_dict()
        if not include_hash:
            data["schema_hash"] = None
        if include_hash and data.get("schema_hash") is None:
            data["schema_hash"] = self.stable_schema_hash()
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        columns = []
        for raw_column in data["columns"]:
            raw_factorization = raw_column.get("factorization", {})
            columns.append(
                ColumnMetadata(
                    name=raw_column["name"],
                    kind=ColumnKind(raw_column["kind"]),
                    domain=tuple(raw_column["domain"]),
                    table=raw_column.get("table"),
                    predicate_domain=(
                        tuple(raw_column["predicate_domain"])
                        if raw_column.get("predicate_domain") is not None
                        else None
                    ),
                    fanout_source=raw_column.get("fanout_source"),
                    factorization=FactorizationMetadata(
                        original_column=raw_factorization.get("original_column"),
                        subcolumns=tuple(raw_factorization.get("subcolumns", ())),
                        subcolumn_domain_sizes=tuple(
                            raw_factorization.get("subcolumn_domain_sizes", ())
                        ),
                        reconstruction_order=tuple(
                            raw_factorization.get("reconstruction_order", ())
                        ),
                        radix=raw_factorization.get("radix"),
                        mapping_name=raw_factorization.get("mapping_name"),
                    ),
                )
            )
        return cls(
            columns=tuple(columns),
            full_join_cardinality=float(data["full_join_cardinality"]),
            column_order=data.get("column_order", "data_indicators_fanouts"),
            upstream_attribution=dict(data.get("upstream_attribution", {})),
            schema_hash=data.get("schema_hash"),
            factorization_plan=FactorizationPlan.from_json_dict(
                data.get("factorization_plan")
            ),
            join_root=data.get("join_root"),
            join_tables=tuple(data.get("join_tables", ())),
            join_edges=tuple(
                (str(left), str(right))
                for left, right in data.get("join_edges", ())
            ),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelMetadata":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Table:
    """Small schema descriptor for acyclic synthetic join tests."""

    name: str
    primary_key: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ForeignKey:
    """Equality join edge from one table column to another table column."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
