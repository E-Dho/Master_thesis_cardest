from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken


@dataclass(frozen=True)
class JoinGraphMetadata:
    """Join-tree topology used to sample valid connected training queries."""

    root_table: str
    tables: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class GeneratedTrainingContext:
    """One row-satisfied predicate context for a sampled full-join tuple."""

    tokens: tuple[PredicateToken, ...]
    included_tables: frozenset[str]
    inverse_fanout_columns: frozenset[str]
    ordinary_predicates: Mapping[str, PredicateToken]


@dataclass(frozen=True)
class PredicateGenerationStats:
    """Counters emitted while generating row-specific training contexts."""

    generated_contexts: int = 0
    rejected_unsatisfied_contexts: int = 0
    included_indicator_contradictions: int = 0

    def to_json_dict(self) -> dict[str, int]:
        return {
            "generated_contexts": int(self.generated_contexts),
            "rejected_unsatisfied_contexts": int(self.rejected_unsatisfied_contexts),
            "included_indicator_contradictions": int(self.included_indicator_contradictions),
        }


@dataclass(frozen=True)
class _ColumnPredicateCache:
    comparable_values: tuple[Any, ...]
    comparable_set: frozenset[Any]


_TOKEN_COVERAGE_KEYS = (
    "wildcard",
    "equal",
    "less_than",
    "less_equal",
    "greater_than",
    "greater_equal",
    "range",
    "indicator_equal_1",
    "indicator_wildcard",
    "fanout_inv",
    "fanout_wildcard",
)


class PredicateTrainingContextGenerator:
    """Generate query contexts whose predicates are true for sampled rows.

    Training must expose the predicate-conditioned network to the same token
    semantics used at evaluation time. The default strategy samples one
    row-satisfied context per encoded full-join tuple. The Duet-style strategy
    samples one shared context from batch-level bounds and applies it to every
    row in the optimizer batch.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.legacy_fixed_context = bool(self.config.get("legacy_fixed_context", False))
        self.per_row_contexts = int(self.config.get("per_row_contexts", 1))
        if self.per_row_contexts <= 0:
            raise ValueError("predicate_generation.per_row_contexts must be positive")
        self.wildcard_probability = float(self.config.get("wildcard_probability", 0.2))
        self.equality_probability = float(self.config.get("equality_probability", 0.4))
        self.lower_bound_probability = float(self.config.get("lower_bound_probability", 0.2))
        self.upper_bound_probability = float(self.config.get("upper_bound_probability", 0.2))
        self.native_range_probability = float(self.config.get("native_range_probability", 0.0))
        self.strategy = str(self.config.get("strategy", "row_satisfied"))
        if self.strategy not in {"row_satisfied", "duet_batch_bounds"}:
            raise ValueError(
                "predicate_generation.strategy must be row_satisfied or duet_batch_bounds"
            )
        self.enable_native_range_tokens = bool(
            self.config.get("enable_native_range_tokens", False)
        )
        self.native_range_max_domain_size = int(
            self.config.get("native_range_max_domain_size", 512)
        )
        if self.native_range_max_domain_size <= 0:
            raise ValueError("predicate_generation.native_range_max_domain_size must be positive")
        probabilities = (
            self.wildcard_probability,
            self.equality_probability,
            self.lower_bound_probability,
            self.upper_bound_probability,
            self.native_range_probability,
        )
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("predicate_generation probabilities must be nonnegative")
        self._probability_total = float(sum(probabilities))
        if self._probability_total <= 0.0:
            raise ValueError("predicate_generation probabilities must have positive total")
        self.table_subset_sampling = str(self.config.get("table_subset_sampling", "full"))
        self._cache_by_metadata_id: dict[int, tuple[_ColumnPredicateCache | None, ...]] = {}

    def generate_batch(
        self,
        *,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> tuple[list[GeneratedTrainingContext], np.ndarray, PredicateGenerationStats]:
        """Generate contexts and row targets, repeating rows per context."""

        encoded_rows = np.asarray(encoded_rows, dtype=int)
        if self.strategy == "duet_batch_bounds" and not self.legacy_fixed_context:
            return self._generate_duet_batch_bounds(
                encoded_rows=encoded_rows,
                metadata=metadata,
                rng=rng,
            )
        contexts: list[GeneratedTrainingContext] = []
        repeated_rows: list[np.ndarray] = []
        rejected = 0
        contradictions = 0
        for row in encoded_rows:
            for _ in range(self.per_row_contexts if self.enabled else 1):
                context = (
                    self._generate_legacy_fixed_context(metadata)
                    if self.legacy_fixed_context
                    else self._generate_one(row, metadata, rng)
                )
                if self.legacy_fixed_context:
                    contradictions += included_indicator_contradictions(context, row, metadata)
                    contexts.append(context)
                    repeated_rows.append(row)
                    continue
                if not context_satisfies_row(context, row, metadata):
                    rejected += 1
                    continue
                contexts.append(context)
                repeated_rows.append(row)
        if not contexts:
            raise ValueError("predicate generation rejected every sampled context")
        return (
            contexts,
            np.stack(repeated_rows, axis=0),
            PredicateGenerationStats(
                generated_contexts=len(contexts),
                rejected_unsatisfied_contexts=rejected,
                included_indicator_contradictions=contradictions,
            ),
        )

    def _generate_duet_batch_bounds(
        self,
        *,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> tuple[list[GeneratedTrainingContext], np.ndarray, PredicateGenerationStats]:
        """Generate shared Duet-style contexts from batch-level value bounds."""

        contexts: list[GeneratedTrainingContext] = []
        repeated_rows: list[np.ndarray] = []
        rejected = 0
        for _ in range(self.per_row_contexts if self.enabled else 1):
            context = self._generate_one_for_batch(encoded_rows, metadata, rng)
            for row in encoded_rows:
                if not context_satisfies_row(context, row, metadata):
                    rejected += 1
                    continue
                contexts.append(context)
                repeated_rows.append(row)
        if not contexts:
            raise ValueError("predicate generation rejected every sampled context")
        return (
            contexts,
            np.stack(repeated_rows, axis=0),
            PredicateGenerationStats(
                generated_contexts=len(contexts),
                rejected_unsatisfied_contexts=rejected,
                included_indicator_contradictions=0,
            ),
        )

    def _generate_one_for_batch(
        self,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> GeneratedTrainingContext:
        included_tables = self._sample_batch_included_tables(encoded_rows, metadata, rng)
        inverse_fanouts = inverse_fanouts_for_table_subset(metadata, included_tables)
        ordinary = self._ordinary_predicates_for_batch(encoded_rows, metadata, rng)
        tokens = tuple(
            tokens_for_query_tables(
                metadata,
                set(included_tables),
                set(inverse_fanouts),
                dict(ordinary),
            )
        )
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=frozenset(included_tables),
            inverse_fanout_columns=frozenset(inverse_fanouts),
            ordinary_predicates=ordinary,
        )

    def _generate_one(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> GeneratedTrainingContext:
        included_tables = self._sample_included_tables(encoded_row, metadata, rng)
        inverse_fanouts = inverse_fanouts_for_table_subset(metadata, included_tables)
        ordinary = self._ordinary_predicates(encoded_row, metadata, rng)
        tokens = tuple(
            tokens_for_query_tables(
                metadata,
                set(included_tables),
                set(inverse_fanouts),
                dict(ordinary),
            )
        )
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=frozenset(included_tables),
            inverse_fanout_columns=frozenset(inverse_fanouts),
            ordinary_predicates=ordinary,
        )

    def _generate_legacy_fixed_context(
        self,
        metadata: ModelMetadata,
    ) -> GeneratedTrainingContext:
        included_tables = frozenset(
            column.table
            for column in metadata.columns
            if column.table is not None
        )
        inverse_fanouts = frozenset(
            column.name
            for column in metadata.columns
            if column.kind == ColumnKind.FANOUT
        )
        tokens = tuple(tokens_for_query_tables(metadata, set(included_tables), set(inverse_fanouts)))
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=included_tables,
            inverse_fanout_columns=inverse_fanouts,
            ordinary_predicates={},
        )

    def _sample_included_tables(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> frozenset[str]:
        present = present_tables_for_row(encoded_row, metadata)
        if not present:
            return frozenset()
        if self.table_subset_sampling == "full" or not self.enabled:
            return frozenset(present)
        if self.table_subset_sampling != "connected":
            raise ValueError(
                f"unsupported predicate_generation.table_subset_sampling "
                f"{self.table_subset_sampling!r}"
            )
        candidates = connected_table_subsets(metadata, allowed_tables=present)
        if not candidates:
            return frozenset(present)
        index = int(rng.integers(0, len(candidates)))
        return frozenset(candidates[index])

    def _sample_batch_included_tables(
        self,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> frozenset[str]:
        present_by_row = [present_tables_for_row(row, metadata) for row in encoded_rows]
        if not present_by_row:
            return frozenset()
        common_present = set(present_by_row[0])
        for present in present_by_row[1:]:
            common_present &= set(present)
        if not common_present:
            return frozenset()
        if self.table_subset_sampling == "full" or not self.enabled:
            return frozenset(common_present)
        if self.table_subset_sampling != "connected":
            raise ValueError(
                f"unsupported predicate_generation.table_subset_sampling "
                f"{self.table_subset_sampling!r}"
            )
        candidates = connected_table_subsets(metadata, allowed_tables=frozenset(common_present))
        if not candidates:
            return frozenset(common_present)
        index = int(rng.integers(0, len(candidates)))
        return frozenset(candidates[index])

    def _ordinary_predicates(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> dict[str, PredicateToken]:
        ordinary: dict[str, PredicateToken] = {}
        caches = self._column_caches(metadata)
        for column_index, column in enumerate(metadata.columns):
            if column.kind != ColumnKind.DATA:
                continue
            value = column.domain[int(encoded_row[column_index])]
            token = self._sample_satisfied_predicate(caches[column_index], value, rng)
            if token.op != PredicateOp.WILDCARD:
                ordinary[column.name] = token
        return ordinary

    def _ordinary_predicates_for_batch(
        self,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
    ) -> dict[str, PredicateToken]:
        ordinary: dict[str, PredicateToken] = {}
        caches = self._column_caches(metadata)
        for column_index, column in enumerate(metadata.columns):
            if column.kind != ColumnKind.DATA:
                continue
            cache = caches[column_index]
            if cache is None:
                continue
            values = [
                column.domain[int(row[column_index])]
                for row in encoded_rows
            ]
            if any(value not in cache.comparable_set for value in values):
                continue
            token = self._sample_batch_satisfied_predicate(
                cache,
                column_domain_size=column.domain_size,
                values=values,
                rng=rng,
            )
            if token.op != PredicateOp.WILDCARD:
                ordinary[column.name] = token
        return ordinary

    def _sample_satisfied_predicate(
        self,
        cache: _ColumnPredicateCache | None,
        value: Any,
        rng: np.random.Generator,
    ) -> PredicateToken:
        if not self.enabled:
            return PredicateToken.wildcard()
        if cache is None or value not in cache.comparable_set:
            return PredicateToken.wildcard()
        comparable_values = cache.comparable_values
        roll = float(rng.random() * self._probability_total)
        if roll < self.wildcard_probability:
            return PredicateToken.wildcard()
        roll -= self.wildcard_probability
        if roll < self.equality_probability:
            return PredicateToken.equal(value)
        roll -= self.equality_probability
        if roll < self.lower_bound_probability:
            stop = bisect_right(comparable_values, value)
            if stop <= 0:
                return PredicateToken.wildcard()
            threshold = comparable_values[int(rng.integers(0, stop))]
            return PredicateToken(PredicateOp.GREATER_EQUAL, value=threshold)
        roll -= self.lower_bound_probability
        if roll < self.upper_bound_probability:
            start = bisect_left(comparable_values, value)
            if start >= len(comparable_values):
                return PredicateToken.wildcard()
            threshold = comparable_values[int(rng.integers(start, len(comparable_values)))]
            return PredicateToken(PredicateOp.LESS_EQUAL, value=threshold)
        return PredicateToken.wildcard()

    def _sample_batch_satisfied_predicate(
        self,
        cache: _ColumnPredicateCache,
        *,
        column_domain_size: int,
        values: list[Any],
        rng: np.random.Generator,
    ) -> PredicateToken:
        comparable_values = cache.comparable_values
        batch_min = min(values)
        batch_max = max(values)
        roll = float(rng.random() * self._probability_total)
        if roll < self.wildcard_probability:
            return PredicateToken.wildcard()
        roll -= self.wildcard_probability
        if roll < self.equality_probability:
            if all(value == values[0] for value in values):
                return PredicateToken.equal(values[0])
            return self._sample_batch_range_style_predicate(
                comparable_values,
                batch_min=batch_min,
                batch_max=batch_max,
                column_domain_size=column_domain_size,
                rng=rng,
            )
        roll -= self.equality_probability
        if roll < self.lower_bound_probability:
            return self._sample_batch_lower_bound(
                comparable_values,
                batch_min=batch_min,
                rng=rng,
            )
        roll -= self.lower_bound_probability
        if roll < self.upper_bound_probability:
            return self._sample_batch_upper_bound(
                comparable_values,
                batch_max=batch_max,
                rng=rng,
            )
        return self._sample_batch_range_style_predicate(
            comparable_values,
            batch_min=batch_min,
            batch_max=batch_max,
            column_domain_size=column_domain_size,
            rng=rng,
        )

    def _sample_batch_range_style_predicate(
        self,
        comparable_values: tuple[Any, ...],
        *,
        batch_min: Any,
        batch_max: Any,
        column_domain_size: int,
        rng: np.random.Generator,
    ) -> PredicateToken:
        range_allowed = (
            self.enable_native_range_tokens
            and self.native_range_probability > 0.0
            and column_domain_size <= self.native_range_max_domain_size
        )
        choices: list[tuple[str, float]] = [
            ("lower", max(0.0, self.lower_bound_probability)),
            ("upper", max(0.0, self.upper_bound_probability)),
        ]
        if range_allowed:
            choices.append(("range", max(0.0, self.native_range_probability)))
        total = sum(weight for _, weight in choices)
        if total <= 0.0:
            choices = [("lower", 1.0), ("upper", 1.0)]
            total = 2.0
        roll = float(rng.random() * total)
        for name, weight in choices:
            if roll < weight:
                if name == "lower":
                    return self._sample_batch_lower_bound(
                        comparable_values,
                        batch_min=batch_min,
                        rng=rng,
                    )
                if name == "upper":
                    return self._sample_batch_upper_bound(
                        comparable_values,
                        batch_max=batch_max,
                        rng=rng,
                    )
                return self._sample_batch_native_range(
                    comparable_values,
                    batch_min=batch_min,
                    batch_max=batch_max,
                    rng=rng,
                )
            roll -= weight
        return self._sample_batch_upper_bound(
            comparable_values,
            batch_max=batch_max,
            rng=rng,
        )

    @staticmethod
    def _sample_batch_lower_bound(
        comparable_values: tuple[Any, ...],
        *,
        batch_min: Any,
        rng: np.random.Generator,
    ) -> PredicateToken:
        stop = bisect_right(comparable_values, batch_min)
        if stop <= 0:
            return PredicateToken.wildcard()
        threshold = comparable_values[int(rng.integers(0, stop))]
        return PredicateToken(PredicateOp.GREATER_EQUAL, value=threshold)

    @staticmethod
    def _sample_batch_upper_bound(
        comparable_values: tuple[Any, ...],
        *,
        batch_max: Any,
        rng: np.random.Generator,
    ) -> PredicateToken:
        start = bisect_left(comparable_values, batch_max)
        if start >= len(comparable_values):
            return PredicateToken.wildcard()
        threshold = comparable_values[int(rng.integers(start, len(comparable_values)))]
        return PredicateToken(PredicateOp.LESS_EQUAL, value=threshold)

    def _sample_batch_native_range(
        self,
        comparable_values: tuple[Any, ...],
        *,
        batch_min: Any,
        batch_max: Any,
        rng: np.random.Generator,
    ) -> PredicateToken:
        lower = self._sample_batch_lower_bound(
            comparable_values,
            batch_min=batch_min,
            rng=rng,
        )
        upper = self._sample_batch_upper_bound(
            comparable_values,
            batch_max=batch_max,
            rng=rng,
        )
        if lower.op == PredicateOp.WILDCARD or upper.op == PredicateOp.WILDCARD:
            return PredicateToken.wildcard()
        return PredicateToken.range(lower.value, upper.value)

    def _column_caches(
        self,
        metadata: ModelMetadata,
    ) -> tuple[_ColumnPredicateCache | None, ...]:
        key = id(metadata)
        cached = self._cache_by_metadata_id.get(key)
        if cached is not None:
            return cached
        caches = []
        for column in metadata.columns:
            if column.kind != ColumnKind.DATA:
                caches.append(None)
                continue
            values = comparable_domain_values(column.domain)
            try:
                values = sorted(values)
            except TypeError:
                values = []
            caches.append(
                _ColumnPredicateCache(
                    comparable_values=tuple(values),
                    comparable_set=frozenset(values),
                )
                if values
                else None
            )
        result = tuple(caches)
        self._cache_by_metadata_id[key] = result
        return result


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


def present_tables_for_row(encoded_row: np.ndarray, metadata: ModelMetadata) -> frozenset[str]:
    """Return tables whose indicator column is 1 in the encoded full-join row."""

    indicator_tables = {
        column.table
        for column in metadata.columns
        if column.kind == ColumnKind.INDICATOR and column.table is not None
    }
    if not indicator_tables:
        return frozenset(
            column.table
            for column in metadata.columns
            if column.table is not None
        )
    present: set[str] = set()
    for column_index, column in enumerate(metadata.columns):
        if column.kind != ColumnKind.INDICATOR or column.table is None:
            continue
        decoded = column.domain[int(encoded_row[column_index])]
        if decoded == 1:
            present.add(column.table)
    return frozenset(present)


def infer_join_graph(metadata: ModelMetadata) -> JoinGraphMetadata:
    """Infer a join graph from fanout source strings such as ``A->B``."""

    tables = tuple(
        dict.fromkeys(
            column.table
            for column in metadata.columns
            if column.table is not None and column.kind != ColumnKind.FANOUT
        )
    )
    edges = []
    for column in metadata.columns:
        if column.kind != ColumnKind.FANOUT or not column.fanout_source:
            continue
        if "->" in column.fanout_source:
            left, right = column.fanout_source.split("->", 1)
            edges.append((left.strip(), right.strip()))
    root = tables[0] if tables else ""
    return JoinGraphMetadata(root_table=root, tables=tables, edges=tuple(edges))


def connected_table_subsets(
    metadata: ModelMetadata,
    *,
    allowed_tables: frozenset[str] | set[str] | None = None,
) -> tuple[frozenset[str], ...]:
    """Enumerate nonempty connected table subsets within the join graph."""

    graph = infer_join_graph(metadata)
    allowed = set(allowed_tables if allowed_tables is not None else graph.tables)
    tables = tuple(table for table in graph.tables if table in allowed)
    if not graph.edges:
        return tuple(frozenset((table,)) for table in tables)
    subsets: list[frozenset[str]] = []
    for size in range(1, len(tables) + 1):
        for combo in combinations(tables, size):
            subset = frozenset(combo)
            if _is_connected_subset(subset, graph.edges):
                subsets.append(subset)
    return tuple(subsets)


def inverse_fanouts_for_table_subset(
    metadata: ModelMetadata,
    included_tables: frozenset[str] | set[str],
) -> frozenset[str]:
    """Choose INV_FANOUT tokens from the included/excluded child-table semantics.

    A fanout column ``A->B`` removes duplication introduced by the child table
    ``B`` when that child is not part of the query. If no child can be inferred,
    the column's table metadata is used as a conservative fallback.
    """

    included = set(included_tables)
    inverse: set[str] = set()
    for column in metadata.columns:
        if column.kind != ColumnKind.FANOUT:
            continue
        child_table = _fanout_child_table(column.fanout_source) or column.table
        if child_table is not None and child_table not in included:
            inverse.add(column.name)
    return frozenset(inverse)


def token_coverage(
    token_rows: list[list[PredicateToken]] | list[tuple[PredicateToken, ...]],
    metadata: ModelMetadata,
) -> dict[str, dict[str, int]]:
    """Count token operators by column for training/evaluation diagnostics."""

    coverage = {
        column.name: {key: 0 for key in _TOKEN_COVERAGE_KEYS}
        for column in metadata.columns
    }
    for token_row in token_rows:
        for column, token in zip(metadata.columns, token_row):
            key = _coverage_key(column, token)
            coverage[column.name][key] += 1
    return coverage


def literal_token_occurrences(
    token_rows: list[list[PredicateToken]] | list[tuple[PredicateToken, ...]],
    metadata: ModelMetadata,
) -> dict[str, dict[str, dict[str, int]]]:
    """Count observed literal-bearing predicate tokens by column and operator."""

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for token_row in token_rows:
        for column, token in zip(metadata.columns, token_row):
            if column.kind != ColumnKind.DATA:
                continue
            if token.op in {PredicateOp.WILDCARD, PredicateOp.INV_FANOUT}:
                continue
            key = _literal_key(token.value, token.upper)
            column_counts = counts.setdefault(column.name, {})
            op_counts = column_counts.setdefault(token.op.value, {})
            op_counts[key] = int(op_counts.get(key, 0)) + 1
    return counts


def literal_token_stats(
    occurrence_counts: dict[str, dict[str, dict[str, int]]],
    metadata: ModelMetadata,
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    """Summarize observed literal-token sparsity against available vocab tokens."""

    stats: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for column in metadata.columns:
        if column.kind != ColumnKind.DATA:
            continue
        column_stats: dict[str, dict[str, int | float | None]] = {}
        observed_by_op = occurrence_counts.get(column.name, {})
        for op in (
            PredicateOp.EQUAL,
            PredicateOp.LESS_EQUAL,
            PredicateOp.GREATER_EQUAL,
            PredicateOp.LESS_THAN,
            PredicateOp.GREATER_THAN,
            PredicateOp.RANGE,
        ):
            available = _available_literal_token_count(column.domain, op)
            observed_counts = list(observed_by_op.get(op.value, {}).values())
            column_stats[op.value] = {
                "unique_literal_tokens_observed": len(observed_counts),
                "total_literal_tokens_available": available,
                "minimum_occurrence_count": min(observed_counts) if observed_counts else None,
                "median_occurrence_count": _percentile(observed_counts, 50),
                "p95_occurrence_count": _percentile(observed_counts, 95),
                "number_of_unseen_tokens": max(0, available - len(observed_counts)),
            }
        stats[column.name] = column_stats
    return stats


def context_satisfies_row(
    context: GeneratedTrainingContext,
    encoded_row: np.ndarray,
    metadata: ModelMetadata,
) -> bool:
    """Validate that a generated training context is true for its target row."""

    for column_index, (column, token) in enumerate(zip(metadata.columns, context.tokens)):
        value = column.domain[int(encoded_row[column_index])]
        if column.kind == ColumnKind.DATA and not token.satisfies(value):
            return False
        if column.kind == ColumnKind.INDICATOR and token.op == PredicateOp.EQUAL and token.value != value:
            return False
    present = present_tables_for_row(encoded_row, metadata)
    return set(context.included_tables).issubset(present)


def included_indicator_contradictions(
    context: GeneratedTrainingContext,
    encoded_row: np.ndarray,
    metadata: ModelMetadata,
) -> int:
    """Count included-table indicators that contradict the sampled row."""

    contradictions = 0
    for table in context.included_tables:
        for column_index, column in enumerate(metadata.columns):
            if column.kind == ColumnKind.INDICATOR and column.table == table:
                value = column.domain[int(encoded_row[column_index])]
                if value != 1:
                    contradictions += 1
                break
    return contradictions


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


def comparable_domain_values(domain: tuple[Any, ...]) -> list[Any]:
    values = []
    for value in domain:
        if isinstance(value, str) and value.startswith("__"):
            continue
        values.append(value)
    comparable = []
    for value in values:
        try:
            _ = value <= value
        except TypeError:
            continue
        comparable.append(value)
    return comparable


def _fanout_child_table(fanout_source: str | None) -> str | None:
    if fanout_source and "->" in fanout_source:
        return fanout_source.split("->", 1)[1].strip()
    return None


def _is_connected_subset(subset: frozenset[str], edges: tuple[tuple[str, str], ...]) -> bool:
    if len(subset) <= 1:
        return True
    adjacency = {table: set() for table in subset}
    for left, right in edges:
        if left in subset and right in subset:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = set()
    stack = [next(iter(subset))]
    while stack:
        table = stack.pop()
        if table in seen:
            continue
        seen.add(table)
        stack.extend(adjacency[table] - seen)
    return seen == set(subset)


def _coverage_key(column: Any, token: PredicateToken) -> str:
    if column.kind == ColumnKind.INDICATOR:
        if token.op == PredicateOp.EQUAL and token.value == 1:
            return "indicator_equal_1"
        return "indicator_wildcard"
    if column.kind == ColumnKind.FANOUT:
        if token.op == PredicateOp.INV_FANOUT:
            return "fanout_inv"
        return "fanout_wildcard"
    return token.op.value


def _literal_key(value: Any, upper: Any) -> str:
    if upper is None:
        return repr(value)
    return f"{value!r}..{upper!r}"


def _available_literal_token_count(domain: tuple[Any, ...], op: PredicateOp) -> int:
    if op in {PredicateOp.EQUAL, PredicateOp.LESS_EQUAL, PredicateOp.GREATER_EQUAL}:
        return len(domain)
    if op == PredicateOp.RANGE:
        value_count = len(comparable_domain_values(domain))
        return value_count * (value_count + 1) // 2
    return 0


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _leq(left: Any, right: Any) -> bool:
    try:
        return bool(left <= right)
    except TypeError:
        return False


def _geq(left: Any, right: Any) -> bool:
    try:
        return bool(left >= right)
    except TypeError:
        return False
