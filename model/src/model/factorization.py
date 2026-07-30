from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import ceil, log2, prod
from typing import Any, Sequence

import numpy as np

from model.src.data.schema import (
    ColumnKind,
    FactorColumnMetadata,
    FactorizationPlan,
    ModelMetadata,
    OriginalColumnFactorization,
    OutputHeadSpec,
)


@dataclass(frozen=True)
class FactorizationConfig:
    """Runtime configuration for lossless bitwise column factorization."""

    enabled: bool = False
    strategy: str = "none"
    word_size_bits: int = 11
    minimum_domain_size: int = 2048
    factor_order: str = "most_significant_first"
    blacklist_columns: tuple[str, ...] = ()
    blacklist_kinds: tuple[str, ...] = ("indicator", "fanout")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FactorizationConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            strategy=str(data.get("strategy", "none")),
            word_size_bits=int(data.get("word_size_bits", 11)),
            minimum_domain_size=int(data.get("minimum_domain_size", 2048)),
            factor_order=str(data.get("factor_order", "most_significant_first")),
            blacklist_columns=tuple(data.get("blacklist_columns", ())),
            blacklist_kinds=tuple(data.get("blacklist_kinds", ("indicator", "fanout"))),
        )

    def validate(self) -> None:
        if not self.enabled:
            if self.strategy != "none":
                raise ValueError("factorization.strategy must be 'none' when disabled")
            return
        if self.strategy != "bitwise_lossless":
            raise ValueError("factorization.strategy must be 'bitwise_lossless' when enabled")
        if self.word_size_bits <= 0:
            raise ValueError("factorization.word_size_bits must be positive")
        if self.minimum_domain_size <= 1:
            raise ValueError("factorization.minimum_domain_size must exceed one")
        if self.factor_order != "most_significant_first":
            raise ValueError("only factor_order='most_significant_first' is implemented")


def apply_factorization_to_metadata(
    metadata: ModelMetadata, config: FactorizationConfig
) -> ModelMetadata:
    """Return metadata with a deterministic factorization plan attached."""

    config.validate()
    plan = build_factorization_plan(metadata, config)
    return ModelMetadata(
        columns=metadata.columns,
        full_join_cardinality=metadata.full_join_cardinality,
        column_order=metadata.column_order,
        upstream_attribution=metadata.upstream_attribution,
        schema_hash=None,
        factorization_plan=plan,
    )


def build_factorization_plan(
    metadata: ModelMetadata, config: FactorizationConfig
) -> FactorizationPlan:
    """Build output heads from complete original domains, never sampled rows."""

    original_output_width = sum(column.domain_size for column in metadata.columns)
    output_heads: list[OutputHeadSpec] = []
    factor_columns: list[FactorColumnMetadata] = []
    original_factorizations: list[OriginalColumnFactorization] = []
    if not config.enabled:
        return FactorizationPlan(
            enabled=False,
            strategy="none",
            original_output_width=original_output_width,
            factorized_output_width=original_output_width,
            output_head_specs=tuple(
                OutputHeadSpec(column.name, index, None, column.domain_size)
                for index, column in enumerate(metadata.columns)
            ),
        )

    for column_index, column in enumerate(metadata.columns):
        if not _should_factorize(column, config):
            output_heads.append(
                OutputHeadSpec(column.name, column_index, None, column.domain_size)
            )
            continue

        widths = _factor_bit_widths(column.domain_size, config.word_size_bits)
        offsets = _factor_bit_offsets(widths)
        domains = tuple(1 << width for width in widths)
        head_indices = []
        for factor_index, (width, offset, domain_size) in enumerate(
            zip(widths, offsets, domains)
        ):
            name = f"{column.name}__fact_{factor_index}"
            head_indices.append(len(output_heads))
            output_heads.append(
                OutputHeadSpec(name, column_index, factor_index, domain_size)
            )
            factor_columns.append(
                FactorColumnMetadata(
                    name=name,
                    original_column_index=column_index,
                    factor_index=factor_index,
                    domain_size=domain_size,
                    bit_width=width,
                    bit_offset=offset,
                    radix=2,
                    is_last_factor=factor_index == len(widths) - 1,
                )
            )
        product = int(prod(domains))
        original_factorizations.append(
            OriginalColumnFactorization(
                original_column_index=column_index,
                factor_column_indices=tuple(head_indices),
                original_domain_size=column.domain_size,
                factor_order=config.factor_order,
                factor_domains=domains,
                bit_widths=tuple(widths),
                bit_offsets=tuple(offsets),
                radices=tuple(2 for _ in widths),
                invalid_combination_count=product - column.domain_size,
            )
        )

    return FactorizationPlan(
        enabled=True,
        strategy=config.strategy,
        word_size_bits=config.word_size_bits,
        minimum_domain_size=config.minimum_domain_size,
        factor_order=config.factor_order,
        blacklist_columns=config.blacklist_columns,
        blacklist_kinds=config.blacklist_kinds,
        original_column_factorizations=tuple(original_factorizations),
        factor_columns=tuple(factor_columns),
        output_head_specs=tuple(output_heads),
        original_output_width=original_output_width,
        factorized_output_width=sum(head.domain_size for head in output_heads),
    )


def factorize_value(
    encoded_value: int, plan: OriginalColumnFactorization
) -> tuple[int, ...]:
    """Map an original dictionary ID to its most-significant-first factors."""

    value = int(encoded_value)
    if value < 0 or value >= plan.original_domain_size:
        raise ValueError(
            f"encoded value {encoded_value!r} outside original domain "
            f"[0,{plan.original_domain_size})"
        )
    return tuple(
        (value >> offset) & ((1 << width) - 1)
        for width, offset in zip(plan.bit_widths, plan.bit_offsets)
    )


def decode_factors(
    factors: Sequence[int], plan: OriginalColumnFactorization
) -> int:
    """Decode factor digits and reject invalid combinations beyond the domain."""

    if len(factors) != len(plan.factor_domains):
        raise ValueError("factor width does not match factorization plan")
    value = 0
    for factor, width, offset, domain_size in zip(
        factors, plan.bit_widths, plan.bit_offsets, plan.factor_domains
    ):
        factor_value = int(factor)
        if factor_value < 0 or factor_value >= domain_size:
            raise ValueError(f"factor value {factor!r} outside factor domain {domain_size}")
        value |= factor_value << offset
    if value >= plan.original_domain_size:
        raise ValueError(
            f"factor tuple {tuple(factors)!r} decodes to invalid original ID {value}"
        )
    return value


def factorize_rows(encoded_rows: Any, metadata: ModelMetadata) -> Any:
    """Convert original encoded rows into targets aligned with model output heads."""

    if not metadata.factorization_plan.enabled:
        return encoded_rows
    try:
        import torch

        if isinstance(encoded_rows, torch.Tensor):
            return _factorize_rows_torch(encoded_rows, metadata)
    except ModuleNotFoundError:
        pass
    return _factorize_rows_numpy(np.asarray(encoded_rows, dtype=int), metadata)


def factorization_plan_hash(plan: FactorizationPlan) -> str:
    """Hash a factorization plan for manifest/checkpoint compatibility checks."""

    payload = json.dumps(plan.to_json_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _factorize_rows_numpy(encoded_rows: np.ndarray, metadata: ModelMetadata) -> np.ndarray:
    plan = metadata.factorization_plan
    output = np.zeros((encoded_rows.shape[0], len(plan.output_head_specs)), dtype=int)
    by_original = {
        factorization.original_column_index: factorization
        for factorization in plan.original_column_factorizations
    }
    for head_index, head in enumerate(plan.output_head_specs):
        values = encoded_rows[:, head.source_column_index]
        factorization = by_original.get(head.source_column_index)
        if factorization is None:
            output[:, head_index] = values
        else:
            width = factorization.bit_widths[int(head.factor_index)]
            offset = factorization.bit_offsets[int(head.factor_index)]
            output[:, head_index] = (values >> offset) & ((1 << width) - 1)
    return output


def _factorize_rows_torch(encoded_rows: Any, metadata: ModelMetadata) -> Any:
    import torch

    plan = metadata.factorization_plan
    pieces = []
    by_original = {
        factorization.original_column_index: factorization
        for factorization in plan.original_column_factorizations
    }
    for head in plan.output_head_specs:
        values = encoded_rows[:, head.source_column_index].long()
        factorization = by_original.get(head.source_column_index)
        if factorization is None:
            pieces.append(values)
        else:
            width = factorization.bit_widths[int(head.factor_index)]
            offset = factorization.bit_offsets[int(head.factor_index)]
            pieces.append((values >> offset) & ((1 << width) - 1))
    return torch.stack(pieces, dim=1)


def _should_factorize(column: Any, config: FactorizationConfig) -> bool:
    if column.kind != ColumnKind.DATA:
        return False
    if column.kind.value in config.blacklist_kinds:
        return False
    if column.name in config.blacklist_columns:
        return False
    if column.domain_size < config.minimum_domain_size:
        return False
    return len(_factor_bit_widths(column.domain_size, config.word_size_bits)) > 1


def _factor_bit_widths(domain_size: int, word_size_bits: int) -> tuple[int, ...]:
    total_bits = max(1, ceil(log2(domain_size)))
    factor_count = ceil(total_bits / word_size_bits)
    first_width = total_bits - word_size_bits * (factor_count - 1)
    return tuple([first_width] + [word_size_bits] * (factor_count - 1))


def _factor_bit_offsets(widths: Sequence[int]) -> tuple[int, ...]:
    remaining = sum(widths)
    offsets = []
    for width in widths:
        remaining -= width
        offsets.append(remaining)
    return tuple(offsets)
