from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ColumnKind(str, Enum):
    """Modeled column categories used by the estimator and loss weighting."""

    DATA = "data"
    INDICATOR = "indicator"
    FANOUT = "fanout"


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

    def encode_value(self, value: Any) -> int:
        try:
            return self.domain.index(value)
        except ValueError as exc:
            raise ValueError(
                f"value {value!r} is outside domain for column {self.name!r}"
            ) from exc


@dataclass(frozen=True)
class ModelMetadata:
    """Checkpoint metadata required to preserve column order and domains."""

    columns: tuple[ColumnMetadata, ...]
    full_join_cardinality: float
    column_order: str = "data_indicators_fanouts"
    upstream_attribution: dict[str, str] = field(default_factory=dict)
    schema_hash: str | None = None

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

    def stable_schema_hash(self) -> str:
        """Hash the ordering, domains, predicate domains, and join cardinality."""

        payload = json.dumps(self.to_json_dict(include_hash=False), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        data = asdict(self)
        for column in data["columns"]:
            column["kind"] = column["kind"].value
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
