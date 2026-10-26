from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .records import FilterPredicate, QueryRecord
from .workloads import query_to_sql


MSCN_RANGE_OPERATORS = ("=", "<", ">", "<=", ">=")


@dataclass(frozen=True)
class DomainRankEncoder:
    """Order-preserving literal encoder for the adapted MSCN range model."""

    domain: tuple[Any, ...]

    @classmethod
    def from_values(cls, values: Iterable[Any]) -> "DomainRankEncoder":
        unique = set(values)
        try:
            domain = tuple(sorted(unique))
        except TypeError as exc:
            raise ValueError("domain values must be mutually comparable") from exc
        if not domain:
            raise ValueError("domain must not be empty")
        return cls(domain)

    def encode(self, value: Any) -> float:
        try:
            index = self.domain.index(value)
        except ValueError as exc:
            raise KeyError(value) from exc
        if len(self.domain) == 1:
            return 0.0
        return index / (len(self.domain) - 1)


def canonical_query_fingerprint(query: QueryRecord) -> str:
    canonical = "|".join(
        [
            ",".join(sorted(f"{table.name}:{table.alias}" for table in query.tables)),
            ",".join(sorted(f"{join.left}{join.operator}{join.right}" for join in query.joins)),
            ",".join(
                sorted(
                    f"{predicate.column}{predicate.operator}{predicate.value!r}"
                    for predicate in query.filters
                )
            ),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_test_disjoint(
    training_queries: Iterable[QueryRecord], evaluation_queries: Iterable[QueryRecord]
) -> None:
    training = {canonical_query_fingerprint(query) for query in training_queries}
    evaluation = {canonical_query_fingerprint(query) for query in evaluation_queries}
    overlap = training & evaluation
    if overlap:
        raise ValueError(f"training workload overlaps evaluation by {len(overlap)} queries")


def generate_range_training_queries(
    templates: Sequence[QueryRecord],
    domains_by_column: Mapping[str, Sequence[Any]],
    *,
    count: int = 100_000,
    seed: int,
) -> list[QueryRecord]:
    """Create a deterministic, evaluation-disjoint MSCN range workload.

    Join graphs and predicate columns follow the evaluation schema, while
    operators and complete-domain literals are sampled independently.
    Cardinalities remain placeholders until ``label_queries`` is called.
    """

    if not templates:
        raise ValueError("at least one query template is required")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    forbidden = {canonical_query_fingerprint(query) for query in templates}
    observed = set(forbidden)
    generated: list[QueryRecord] = []
    max_attempts = count * 100
    attempts = 0
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        template = templates[int(rng.integers(0, len(templates)))]
        filters: list[FilterPredicate] = []
        for predicate in template.filters:
            domain = domains_by_column.get(predicate.column)
            if not domain:
                raise ValueError(f"complete domain is missing for {predicate.column}")
            operator = MSCN_RANGE_OPERATORS[int(rng.integers(0, len(MSCN_RANGE_OPERATORS)))]
            literal = domain[int(rng.integers(0, len(domain)))]
            filters.append(FilterPredicate(predicate.column, operator, literal))
        candidate = replace(
            template,
            workload="mscn_job_light_ranges_training",
            query_id=len(generated),
            filters=tuple(filters),
            true_cardinality=0,
            source_line="",
        )
        fingerprint = canonical_query_fingerprint(candidate)
        if fingerprint in observed:
            continue
        observed.add(fingerprint)
        generated.append(candidate)
    if len(generated) != count:
        raise RuntimeError(
            f"could generate only {len(generated)} distinct queries after {attempts} attempts"
        )
    assert_test_disjoint(generated, templates)
    return generated


def label_queries(
    queries: Iterable[QueryRecord], execute_count: Callable[[str], int]
) -> list[QueryRecord]:
    return [
        replace(query, true_cardinality=int(execute_count(query_to_sql(query))))
        for query in queries
    ]


def regenerate_table_bitmaps(
    queries: Sequence[QueryRecord],
    sampled_rows_by_table: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[tuple[int, str], np.ndarray]:
    """Regenerate MSCN's materialized per-table sample bits.

    ``sampled_rows_by_table`` should contain the fixed 1,000 rows selected for
    each table. Predicates are applied with ordered Python comparisons, which
    preserves complete-domain rank semantics for numeric and string values.
    """

    result: dict[tuple[int, str], np.ndarray] = {}
    for query in queries:
        alias_to_table = {table.alias: table.name for table in query.tables}
        predicates_by_table: dict[str, list[FilterPredicate]] = {}
        for predicate in query.filters:
            alias, column = predicate.column.split(".", 1)
            table = alias_to_table.get(alias, alias)
            predicates_by_table.setdefault(table, []).append(
                FilterPredicate(column, predicate.operator, predicate.value)
            )
        for table in {table.name for table in query.tables}:
            rows = sampled_rows_by_table.get(table)
            if rows is None:
                raise ValueError(f"sample rows are missing for table {table}")
            predicates = predicates_by_table.get(table, [])
            bitmap = np.asarray(
                [all(_matches(row.get(p.column), p.operator, p.value) for p in predicates) for row in rows],
                dtype=bool,
            )
            result[(query.query_id, table)] = bitmap
    return result


def _matches(value: Any, operator: str, literal: Any) -> bool:
    if value is None:
        return False
    if operator == "=":
        return value == literal
    if operator == "<":
        return value < literal
    if operator == ">":
        return value > literal
    if operator == "<=":
        return value <= literal
    if operator == ">=":
        return value >= literal
    raise ValueError(f"unsupported MSCN operator {operator!r}")
