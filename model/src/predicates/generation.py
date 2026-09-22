from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.vocabulary import TWO_SLOT_EMPTY_OPERATOR_ID, TWO_SLOT_OP_TO_ID


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
    trajectory_query: Any | None = None
    encoded_two_slot_row: tuple[tuple[int, int, int, int], ...] | None = None


@dataclass(frozen=True)
class GeneratedPhysicalQuery:
    included_tables: frozenset[str]
    inverse_fanout_columns: frozenset[str]
    ordinary_predicates: Mapping[str, PredicateToken]
    trajectory_query: Any | None = None


@dataclass(frozen=True)
class PredicateGenerationStats:
    """Counters emitted while generating row-specific training contexts."""

    generated_contexts: int = 0
    rejected_unsatisfied_contexts: int = 0
    included_indicator_contradictions: int = 0
    source_row_indices: tuple[int, ...] = ()

    def to_json_dict(self) -> dict[str, int]:
        return {
            "generated_contexts": int(self.generated_contexts),
            "rejected_unsatisfied_contexts": int(self.rejected_unsatisfied_contexts),
            "included_indicator_contradictions": int(self.included_indicator_contradictions),
        }


@dataclass(frozen=True)
class _ColumnPredicateCache:
    comparable_values: tuple[Any, ...]
    domain_id_to_left_rank: tuple[int, ...]
    domain_id_to_right_rank: tuple[int, ...]


@dataclass(frozen=True)
class _NumericLiteralCache:
    values: tuple[float, ...]
    literals: tuple[Any, ...]


@dataclass(frozen=True)
class _CompiledGenerationMetadata:
    column_name_to_index: dict[str, int]
    data_indices: tuple[int, ...]
    indicator_indices: tuple[int, ...]
    fanout_indices: tuple[int, ...]
    table_indicator_indices: dict[str, int]
    graph: JoinGraphMetadata
    fanout_child_by_name: dict[str, str | None]
    column_caches: tuple[_ColumnPredicateCache | None, ...]
    pol_required_available: bool
    temporal_start_index: int | None
    temporal_end_index: int | None
    endpoint_spatial_indices: dict[str, int]
    mbr_spatial_indices: dict[str, int]
    endpoint_x_domain: tuple[float, ...]
    endpoint_y_domain: tuple[float, ...]
    numeric_literals_by_column: tuple[_NumericLiteralCache | None, ...]
    table_to_bit: dict[str, int]
    bit_to_table: tuple[str, ...]
    data_table_bits: tuple[int, ...]
    fanout_child_bits: tuple[int, ...]


_WILDCARD_TOKEN = PredicateToken.wildcard()
_INDICATOR_INCLUDED_TOKEN = PredicateToken.equal(1)
_INV_FANOUT_TOKEN = PredicateToken.inv_fanout()


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
    is batch-vectorized by the trainer but still emits row-specific predicates
    using thresholds sampled relative to each tuple's value.
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
        self.normalize_predicate_probabilities = bool(
            self.config.get("normalize_predicate_probabilities", True)
        )
        self._probability_total = float(sum(probabilities))
        if self._probability_total <= 0.0:
            raise ValueError("predicate_generation probabilities must have positive total")
        if (
            not self.normalize_predicate_probabilities
            and abs(self._probability_total - 1.0) > 1.0e-8
        ):
            raise ValueError(
                "predicate_generation probabilities must sum to 1.0 when "
                "normalize_predicate_probabilities=false"
            )
        self.table_subset_sampling = str(self.config.get("table_subset_sampling", "full"))
        self.trajectory_query_semantics = str(
            self.config.get("trajectory_query_semantics", "auto_pol_segments")
        )
        self.trajectory_temporal_probability = float(
            self.config.get("trajectory_temporal_probability", 0.0)
        )
        self.trajectory_spatial_probability = float(
            self.config.get("trajectory_spatial_probability", 0.0)
        )
        trajectory_spatial = self.config.get("trajectory_spatial", {})
        if trajectory_spatial is None:
            trajectory_spatial = {}
        self.trajectory_spatial_enabled = bool(
            trajectory_spatial.get("enabled", False)
        )
        self.trajectory_spatial_representation = str(
            trajectory_spatial.get("representation", "")
        )
        if (
            self.trajectory_spatial_enabled
            and self.trajectory_spatial_representation != "segment_mbr"
        ):
            raise ValueError(
                "predicate_generation.trajectory_spatial.representation must be "
                "segment_mbr when enabled"
        )
        self._cache_by_metadata_id: dict[int, tuple[_ColumnPredicateCache | None, ...]] = {}
        self._compiled_by_metadata_id: dict[int, _CompiledGenerationMetadata] = {}

    def probability_diagnostics(self) -> dict[str, float | bool]:
        return {
            "wildcard_probability": self.wildcard_probability,
            "equality_probability": self.equality_probability,
            "lower_bound_probability": self.lower_bound_probability,
            "upper_bound_probability": self.upper_bound_probability,
            "native_range_probability": self.native_range_probability,
            "probability_total": self._probability_total,
            "normalize_predicate_probabilities": self.normalize_predicate_probabilities,
        }

    def generate_batch(
        self,
        *,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        validate_contexts: bool = True,
        validation_sample_size: int | None = None,
    ) -> tuple[list[GeneratedTrainingContext], np.ndarray, PredicateGenerationStats]:
        """Generate contexts and row targets, repeating rows per context."""

        encoded_rows = np.asarray(encoded_rows, dtype=int)
        compiled = self._compiled(metadata)
        if self.strategy == "duet_batch_bounds" and not self.legacy_fixed_context:
            # IMPORTANT:
            # `duet_batch_bounds` is intentionally ROW-SPECIFIC in this project.
            #
            # Do not construct one predicate context from optimizer-batch min/max
            # values or from the intersection of table presence across the batch.
            #
            # That historical implementation is incorrect for our
            # predicate-conditioned training semantics because it:
            #   1. removes almost all equality predicates on heterogeneous batches,
            #   2. creates artificially broad ranges,
            #   3. lets one OUTER_MISSING row wildcard a whole batch column, and
            #   4. collapses NeuroCard table-subset diversity through batch-wide
            #      table-presence intersection.
            #
            # Each sampled FOJ row must receive an independently sampled query
            # context that is satisfied by that same row.
            return self._generate_duet_row_specific(
                encoded_rows=encoded_rows,
                metadata=metadata,
                rng=rng,
                compiled=compiled,
                validate_contexts=validate_contexts,
                validation_sample_size=validation_sample_size,
            )
        contexts: list[GeneratedTrainingContext] = []
        repeated_rows: list[np.ndarray] = []
        source_row_indices: list[int] = []
        rejected = 0
        contradictions = 0
        for row_index, row in enumerate(encoded_rows):
            for _ in range(self.per_row_contexts if self.enabled else 1):
                context = (
                    self._generate_legacy_fixed_context(metadata)
                    if self.legacy_fixed_context
                    else self._generate_one(row, metadata, rng, compiled)
                )
                if self.legacy_fixed_context:
                    contradictions += included_indicator_contradictions(context, row, metadata)
                    contexts.append(context)
                    repeated_rows.append(row)
                    source_row_indices.append(row_index)
                    continue
                if self._requires_root(metadata, compiled) and not context.included_tables:
                    rejected += 1
                    continue
                if not context_satisfies_row(context, row, metadata):
                    rejected += 1
                    continue
                contexts.append(context)
                repeated_rows.append(row)
                source_row_indices.append(row_index)
        if not contexts:
            raise ValueError("predicate generation rejected every sampled context")
        return (
            contexts,
            np.stack(repeated_rows, axis=0),
            PredicateGenerationStats(
                generated_contexts=len(contexts),
                rejected_unsatisfied_contexts=rejected,
                included_indicator_contradictions=contradictions,
                source_row_indices=tuple(source_row_indices),
            ),
        )

    def generate_forced_stratum_batch(
        self,
        *,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        strata: tuple[Any, ...] | list[Any],
        rng: np.random.Generator,
        debug_allow_row_dependent_native_range_tail: bool = False,
    ) -> tuple[list[GeneratedTrainingContext], np.ndarray, PredicateGenerationStats]:
        """Generate row-satisfied contexts with an exact stratum predicate forced.

        The table subset and all non-stratum ordinary predicates follow the
        normal row-specific generator.  The stratum's DATA column is then forced
        to the support-deficient event that caused the rare row to be sampled.
        """

        encoded_rows = np.asarray(encoded_rows, dtype=int)
        strata = tuple(strata)
        compiled = self._compiled(metadata)
        if encoded_rows.shape[0] != len(strata):
            raise ValueError("forced stratum generation requires one stratum per row")
        contexts: list[GeneratedTrainingContext] = []
        repeated_rows: list[np.ndarray] = []
        source_row_indices: list[int] = []
        rejected = 0
        contradictions = 0
        for row_index, (row, stratum) in enumerate(zip(encoded_rows, strata)):
            context = self._generate_forced_stratum_context(
                row,
                metadata,
                stratum,
                rng,
                compiled,
                debug_allow_row_dependent_native_range_tail=(
                    debug_allow_row_dependent_native_range_tail
                ),
            )
            contradictions += included_indicator_contradictions(context, row, metadata)
            if not context_satisfies_row(context, row, metadata):
                rejected += 1
                continue
            contexts.append(context)
            repeated_rows.append(row)
            source_row_indices.append(row_index)
        if not contexts:
            raise ValueError("forced stratum predicate generation rejected every context")
        return (
            contexts,
            np.stack(repeated_rows, axis=0),
            PredicateGenerationStats(
                generated_contexts=len(contexts),
                rejected_unsatisfied_contexts=rejected,
                included_indicator_contradictions=contradictions,
                source_row_indices=tuple(source_row_indices),
            ),
        )

    def _generate_forced_stratum_context(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        stratum: Any,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
        *,
        debug_allow_row_dependent_native_range_tail: bool = False,
    ) -> GeneratedTrainingContext:
        compiled = compiled or self._compiled(metadata)
        included_tables = set(self._sample_included_tables(encoded_row, metadata, rng, compiled))
        column = metadata.columns[int(stratum.column_index)]
        if column.table is not None:
            present = self._present_tables_for_row(encoded_row, metadata, compiled)
            if column.table not in present:
                tokens = tuple([_WILDCARD_TOKEN] * len(metadata.columns))
                return GeneratedTrainingContext(
                    tokens=tokens,
                    included_tables=frozenset(),
                    inverse_fanout_columns=frozenset(),
                    ordinary_predicates={},
                    encoded_two_slot_row=_wildcard_two_slot_row(metadata),
                )
            included_tables.add(column.table)
            if self.table_subset_sampling == "neurocard_table_dropout_rooted":
                graph = compiled.graph
                included_tables = _root_connected_component(
                    included_tables,
                    graph.root_table,
                    graph.edges,
                )
                included_tables.add(column.table)
        inverse_fanouts = self._inverse_fanouts_for_table_subset(compiled, included_tables)
        ordinary = self._ordinary_predicates(
            encoded_row,
            metadata,
            rng,
            frozenset(included_tables),
            compiled=compiled,
        )
        ordinary[column.name] = forced_predicate_for_stratum(
            stratum,
            encoded_row=encoded_row,
            metadata=metadata,
            debug_allow_row_dependent_native_range_tail=(
                debug_allow_row_dependent_native_range_tail
            ),
        )
        tokens, encoded_two_slot_row = self._tokens_and_two_slot_for_query_tables(
            metadata,
            compiled,
            set(included_tables),
            set(inverse_fanouts),
            dict(ordinary),
        )
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=frozenset(included_tables),
            inverse_fanout_columns=frozenset(inverse_fanouts),
            ordinary_predicates=ordinary,
            encoded_two_slot_row=encoded_two_slot_row,
        )

    def _generate_duet_row_specific(
        self,
        *,
        encoded_rows: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
        validate_contexts: bool = True,
        validation_sample_size: int | None = None,
    ) -> tuple[list[GeneratedTrainingContext], np.ndarray, PredicateGenerationStats]:
        """Generate independent Duet-style query contexts for each sampled row."""

        compiled = compiled or self._compiled(metadata)
        contexts: list[GeneratedTrainingContext] = []
        repeated_rows: list[np.ndarray] = []
        source_row_indices: list[int] = []
        rejected = 0
        contradictions = 0
        repeats = self.per_row_contexts if self.enabled else 1
        for row_index, row in enumerate(encoded_rows):
            for _ in range(repeats):
                context = self._generate_one(row, metadata, rng, compiled)
                if self._requires_root(metadata, compiled) and not context.included_tables:
                    rejected += 1
                    continue
                if validate_contexts:
                    contradictions += included_indicator_contradictions(
                        context,
                        row,
                        metadata,
                    )
                    if not context_satisfies_row(context, row, metadata):
                        rejected += 1
                        continue
                contexts.append(context)
                repeated_rows.append(row)
                source_row_indices.append(row_index)
        if not contexts:
            raise ValueError("predicate generation rejected every sampled context")
        if not validate_contexts and validation_sample_size:
            sample_count = min(int(validation_sample_size), len(contexts))
            for validation_index in _deterministic_validation_indices(
                len(contexts),
                sample_count,
            ):
                context = contexts[validation_index]
                row = repeated_rows[validation_index]
                contradiction_count = included_indicator_contradictions(
                    context,
                    row,
                    metadata,
                )
                contradictions += contradiction_count
                if contradiction_count or not context_satisfies_row(context, row, metadata):
                    raise ValueError(
                        "row-satisfied predicate generation validation failed "
                        f"for context index {validation_index}"
                    )
        return (
            contexts,
            np.stack(repeated_rows, axis=0),
            PredicateGenerationStats(
                generated_contexts=len(contexts),
                rejected_unsatisfied_contexts=rejected,
                included_indicator_contradictions=contradictions,
                source_row_indices=tuple(source_row_indices),
            ),
        )

    def _generate_one(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> GeneratedTrainingContext:
        compiled = compiled or self._compiled(metadata)
        query = self._generate_physical_query(encoded_row, metadata, rng, compiled)
        tokens, encoded_two_slot_row = self._tokens_and_two_slot_for_query_tables(
            metadata,
            compiled,
            set(query.included_tables),
            set(query.inverse_fanout_columns),
            dict(query.ordinary_predicates),
        )
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=query.included_tables,
            inverse_fanout_columns=query.inverse_fanout_columns,
            ordinary_predicates=query.ordinary_predicates,
            trajectory_query=query.trajectory_query,
            encoded_two_slot_row=encoded_two_slot_row,
        )

    def _generate_physical_query(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> GeneratedPhysicalQuery:
        compiled = compiled or self._compiled(metadata)
        included_table_mask = self._sample_included_table_mask(
            encoded_row,
            metadata,
            rng,
            compiled,
        )
        included_tables = self._tables_from_mask(compiled, included_table_mask)
        inverse_fanouts = self._inverse_fanouts_for_table_mask(
            metadata,
            compiled,
            included_table_mask,
        )
        semantic_owned: set[str] = set()
        trajectory_query = self._pol_trajectory_query_for_row(
            encoded_row,
            metadata,
            rng,
            included_tables,
            compiled,
        )
        if trajectory_query is not None:
            semantic_owned.update(_semantic_owned_column_names(trajectory_query))
        ordinary = self._ordinary_predicates(
            encoded_row,
            metadata,
            rng,
            included_tables,
            excluded_columns=frozenset(semantic_owned),
            compiled=compiled,
            included_table_mask=included_table_mask,
        )
        if trajectory_query is not None:
            ordinary.update(dict(getattr(trajectory_query, "scalar_predicates", ())))
        return GeneratedPhysicalQuery(
            included_tables=frozenset(included_tables),
            inverse_fanout_columns=frozenset(inverse_fanouts),
            ordinary_predicates=ordinary,
            trajectory_query=trajectory_query,
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
        compiled = self._compiled(metadata)
        tokens, encoded_two_slot_row = self._tokens_and_two_slot_for_query_tables(
            metadata,
            compiled,
            set(included_tables),
            set(inverse_fanouts),
        )
        return GeneratedTrainingContext(
            tokens=tokens,
            included_tables=included_tables,
            inverse_fanout_columns=inverse_fanouts,
            ordinary_predicates={},
            encoded_two_slot_row=encoded_two_slot_row,
        )

    def _sample_included_tables(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> frozenset[str]:
        compiled = compiled or self._compiled(metadata)
        present = self._present_tables_for_row(encoded_row, metadata, compiled)
        return self._sample_included_from_present(present, metadata, rng, compiled)

    def _sample_included_from_present(
        self,
        present: frozenset[str],
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> frozenset[str]:
        compiled = compiled or self._compiled(metadata)
        if not present:
            return frozenset()
        if self.table_subset_sampling == "full" or not self.enabled:
            return frozenset(present)
        if self.table_subset_sampling in {
            "neurocard_rooted_connected",
            "rooted_connected_uniform_legacy",
        }:
            graph = compiled.graph
            if graph.root_table not in present:
                return frozenset()
            candidates = connected_table_subsets(
                metadata,
                allowed_tables=present,
                required_root=graph.root_table,
            )
            if not candidates:
                return frozenset({graph.root_table})
            index = int(rng.integers(0, len(candidates)))
            return frozenset(candidates[index])
        if self.table_subset_sampling == "neurocard_table_dropout_rooted":
            return self._neurocard_table_dropout_rooted_subset(compiled, present, rng)
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

    def _requires_root(
        self,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> bool:
        compiled = compiled or self._compiled(metadata)
        return (
            self.enabled
            and self.table_subset_sampling in {
                "neurocard_rooted_connected",
                "rooted_connected_uniform_legacy",
                "neurocard_table_dropout_rooted",
            }
            and bool(compiled.graph.root_table)
        )

    def _ordinary_predicates(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        included_tables: frozenset[str],
        excluded_columns: frozenset[str] = frozenset(),
        compiled: _CompiledGenerationMetadata | None = None,
        included_table_mask: int | None = None,
    ) -> dict[str, PredicateToken]:
        compiled = compiled or self._compiled(metadata)
        if included_table_mask is None:
            included_table_mask = _table_mask_from_tables(included_tables, compiled)
        ordinary: dict[str, PredicateToken] = {}
        for column_index in compiled.data_indices:
            column = metadata.columns[column_index]
            if column.name in excluded_columns:
                continue
            table_bit = compiled.data_table_bits[column_index]
            if table_bit and not (included_table_mask & table_bit):
                continue
            value = column.domain[int(encoded_row[column_index])]
            token = self._sample_satisfied_predicate(
                compiled.column_caches[column_index],
                value,
                rng,
                encoded_value_id=int(encoded_row[column_index]),
            )
            if token.op != PredicateOp.WILDCARD:
                ordinary[column.name] = token
        return ordinary

    def _pol_trajectory_query_for_row(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        included_tables: frozenset[str],
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> Any | None:
        compiled = compiled or self._compiled(metadata)
        if self.trajectory_query_semantics not in {"auto_pol_segments", "pol_segments"}:
            return None
        if "segments" not in included_tables:
            return None
        required = {
            "segments:t_s",
            "segments:t_e",
            "segments:s_x",
            "segments:s_y",
            "segments:e_x",
            "segments:e_y",
        }
        if not compiled.pol_required_available:
            if self.trajectory_query_semantics == "pol_segments":
                raise ValueError("pol_segments trajectory query semantics require POL segment columns")
            return None
        scalar_predicates: list[tuple[str, PredicateToken]] = []
        temporal_predicates = []
        spatial_predicates = []
        if float(rng.random()) < self.trajectory_temporal_probability:
            generated = self._sample_pol_temporal_query(encoded_row, metadata, rng, compiled)
            if generated is not None:
                scalar_predicates.extend(generated[0])
                temporal_predicates.append(generated[1])
        if float(rng.random()) < self.trajectory_spatial_probability:
            generated = self._sample_pol_spatial_query(encoded_row, metadata, rng, compiled)
            if generated is not None:
                scalar_predicates.extend(generated[0])
                spatial_predicates.append(generated[1])
        if not temporal_predicates and not spatial_predicates and not scalar_predicates:
            return None
        from model.src.data.trajectory_distinct import TrajectoryQuerySemantics

        return TrajectoryQuerySemantics(
            scalar_predicates=tuple(scalar_predicates),
            temporal_predicates=tuple(temporal_predicates),
            spatial_predicates=tuple(spatial_predicates),
        )

    def _sample_pol_temporal_query(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> tuple[list[tuple[str, PredicateToken]], Any] | None:
        from model.src.data.trajectory_distinct import SegmentTemporalPredicate

        compiled = compiled or self._compiled(metadata)
        if compiled.temporal_start_index is None or compiled.temporal_end_index is None:
            return None
        start_index = compiled.temporal_start_index
        end_index = compiled.temporal_end_index
        start_value = metadata.columns[start_index].domain[int(encoded_row[start_index])]
        end_value = metadata.columns[end_index].domain[int(encoded_row[end_index])]
        start_cache = compiled.column_caches[start_index]
        end_cache = compiled.column_caches[end_index]
        if start_cache is None or end_cache is None:
            return None
        lower_stop = end_cache.domain_id_to_right_rank[int(encoded_row[end_index])]
        upper_start = start_cache.domain_id_to_right_rank[int(encoded_row[start_index])]
        if lower_stop <= 0 or upper_start >= len(start_cache.comparable_values):
            return None
        lower = end_cache.comparable_values[int(rng.integers(0, lower_stop))]
        upper = start_cache.comparable_values[
            int(rng.integers(upper_start, len(start_cache.comparable_values)))
        ]
        scalar = [
            ("segments:t_s", PredicateToken(PredicateOp.LESS_THAN, value=upper)),
            ("segments:t_e", PredicateToken(PredicateOp.GREATER_EQUAL, value=lower)),
        ]
        return scalar, SegmentTemporalPredicate(
            "segments:t_s",
            "segments:t_e",
            lower=lower,
            upper=upper,
            semantics="overlap",
        )

    def _sample_pol_spatial_query(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata | None = None,
    ) -> tuple[list[tuple[str, PredicateToken]], Any] | None:
        """Sample a row-satisfied POL spatial rectangle context."""

        from model.src.data.trajectory_distinct import (
            SegmentMbrSpatialPredicate,
            SegmentSpatialPredicate,
        )

        compiled = compiled or self._compiled(metadata)
        if self.trajectory_spatial_enabled:
            mbr_columns = (
                "segments:seg_min_x",
                "segments:seg_max_x",
                "segments:seg_min_y",
                "segments:seg_max_y",
            )
            if len(compiled.mbr_spatial_indices) != len(mbr_columns):
                if self.trajectory_query_semantics == "pol_segments":
                    raise ValueError(
                        "trajectory_spatial.segment_mbr requires prepared POL MBR columns"
                    )
                return None
            values = {
                name: float(
                    metadata.columns[compiled.mbr_spatial_indices[name]].domain[
                        int(encoded_row[compiled.mbr_spatial_indices[name]])
                    ]
                )
                for name in mbr_columns
            }
            cache_by_name = {
                name: compiled.column_caches[compiled.mbr_spatial_indices[name]]
                for name in mbr_columns
            }
            if any(cache is None for cache in cache_by_name.values()):
                return None
            seg_min_x_values = cache_by_name["segments:seg_min_x"].comparable_values  # type: ignore[union-attr]
            seg_max_x_values = cache_by_name["segments:seg_max_x"].comparable_values  # type: ignore[union-attr]
            seg_min_y_values = cache_by_name["segments:seg_min_y"].comparable_values  # type: ignore[union-attr]
            seg_max_y_values = cache_by_name["segments:seg_max_y"].comparable_values  # type: ignore[union-attr]
            x_upper_start = cache_by_name["segments:seg_min_x"].domain_id_to_left_rank[  # type: ignore[union-attr]
                int(encoded_row[compiled.mbr_spatial_indices["segments:seg_min_x"]])
            ]
            x_lower_stop = cache_by_name["segments:seg_max_x"].domain_id_to_right_rank[  # type: ignore[union-attr]
                int(encoded_row[compiled.mbr_spatial_indices["segments:seg_max_x"]])
            ]
            y_upper_start = cache_by_name["segments:seg_min_y"].domain_id_to_left_rank[  # type: ignore[union-attr]
                int(encoded_row[compiled.mbr_spatial_indices["segments:seg_min_y"]])
            ]
            y_lower_stop = cache_by_name["segments:seg_max_y"].domain_id_to_right_rank[  # type: ignore[union-attr]
                int(encoded_row[compiled.mbr_spatial_indices["segments:seg_max_y"]])
            ]
            if (
                x_upper_start >= len(seg_min_x_values)
                or x_lower_stop <= 0
                or y_upper_start >= len(seg_min_y_values)
                or y_lower_stop <= 0
            ):
                return None
            raw_min_x = float(seg_max_x_values[int(rng.integers(0, x_lower_stop))])
            raw_max_x = float(
                seg_min_x_values[int(rng.integers(x_upper_start, len(seg_min_x_values)))]
            )
            raw_min_y = float(seg_max_y_values[int(rng.integers(0, y_lower_stop))])
            raw_max_y = float(
                seg_min_y_values[int(rng.integers(y_upper_start, len(seg_min_y_values)))]
            )
            min_x, max_x = sorted((raw_min_x, raw_max_x))
            min_y, max_y = sorted((raw_min_y, raw_max_y))
            canonical = self._canonicalize_segment_mbr_predicate_cached(
                compiled,
                min_x_column="segments:seg_min_x",
                max_x_column="segments:seg_max_x",
                min_y_column="segments:seg_min_y",
                max_y_column="segments:seg_max_y",
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
            )
            if canonical.zero_support:
                return None
            scalar = [
                (
                    "segments:seg_min_x",
                    PredicateToken(PredicateOp.LESS_EQUAL, value=canonical.max_x_literal),
                ),
                (
                    "segments:seg_max_x",
                    PredicateToken(PredicateOp.GREATER_EQUAL, value=canonical.min_x_literal),
                ),
                (
                    "segments:seg_min_y",
                    PredicateToken(PredicateOp.LESS_EQUAL, value=canonical.max_y_literal),
                ),
                (
                    "segments:seg_max_y",
                    PredicateToken(PredicateOp.GREATER_EQUAL, value=canonical.min_y_literal),
                ),
            ]
            return scalar, SegmentMbrSpatialPredicate(
                min_x=canonical.min_x,
                min_y=canonical.min_y,
                max_x=canonical.max_x,
                max_y=canonical.max_y,
                srid=int(self.config.get("trajectory_srid", 26916)),
                physical_min_x=canonical.physical_min_x,
                physical_min_y=canonical.physical_min_y,
                physical_max_x=canonical.physical_max_x,
                physical_max_y=canonical.physical_max_y,
            )

        column_names = ("segments:s_x", "segments:s_y", "segments:e_x", "segments:e_y")
        if len(compiled.endpoint_spatial_indices) != len(column_names):
            return None
        values = {
            name: float(metadata.columns[compiled.endpoint_spatial_indices[name]].domain[int(encoded_row[compiled.endpoint_spatial_indices[name]])])
            for name in column_names
        }
        x_low_anchor = min(values["segments:s_x"], values["segments:e_x"])
        x_high_anchor = max(values["segments:s_x"], values["segments:e_x"])
        y_low_anchor = min(values["segments:s_y"], values["segments:e_y"])
        y_high_anchor = max(values["segments:s_y"], values["segments:e_y"])
        x_domain = compiled.endpoint_x_domain
        y_domain = compiled.endpoint_y_domain
        x_lower_candidates = [value for value in x_domain if value <= x_low_anchor]
        x_upper_candidates = [value for value in x_domain if value >= x_high_anchor]
        y_lower_candidates = [value for value in y_domain if value <= y_low_anchor]
        y_upper_candidates = [value for value in y_domain if value >= y_high_anchor]
        if not (x_lower_candidates and x_upper_candidates and y_lower_candidates and y_upper_candidates):
            return None
        min_x = float(x_lower_candidates[int(rng.integers(0, len(x_lower_candidates)))])
        max_x = float(x_upper_candidates[int(rng.integers(0, len(x_upper_candidates)))])
        min_y = float(y_lower_candidates[int(rng.integers(0, len(y_lower_candidates)))])
        max_y = float(y_upper_candidates[int(rng.integers(0, len(y_upper_candidates)))])
        scalar = [
            ("segments:s_x", PredicateToken.range(min_x, max_x)),
            ("segments:e_x", PredicateToken.range(min_x, max_x)),
            ("segments:s_y", PredicateToken.range(min_y, max_y)),
            ("segments:e_y", PredicateToken.range(min_y, max_y)),
        ]
        return scalar, SegmentSpatialPredicate(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            srid=int(self.config.get("trajectory_srid", 26916)),
        )

    def _canonicalize_segment_mbr_predicate_cached(
        self,
        compiled: _CompiledGenerationMetadata,
        *,
        min_x_column: str,
        max_x_column: str,
        min_y_column: str,
        max_y_column: str,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> Any:
        from model.src.data.trajectory_distinct import CanonicalSegmentMbrPredicate

        physical_min_x, physical_max_x = sorted((float(min_x), float(max_x)))
        physical_min_y, physical_max_y = sorted((float(min_y), float(max_y)))
        min_x_index = compiled.column_name_to_index[min_x_column]
        max_x_index = compiled.column_name_to_index[max_x_column]
        min_y_index = compiled.column_name_to_index[min_y_column]
        max_y_index = compiled.column_name_to_index[max_y_column]
        upper_x = _floor_numeric_literal(
            compiled.numeric_literals_by_column[min_x_index],
            physical_max_x,
        )
        lower_x = _ceil_numeric_literal(
            compiled.numeric_literals_by_column[max_x_index],
            physical_min_x,
        )
        upper_y = _floor_numeric_literal(
            compiled.numeric_literals_by_column[min_y_index],
            physical_max_y,
        )
        lower_y = _ceil_numeric_literal(
            compiled.numeric_literals_by_column[max_y_index],
            physical_min_y,
        )
        if upper_x is None:
            return _zero_support_mbr_predicate_cached(
                compiled.numeric_literals_by_column[min_x_index],
                min_x_column,
                PredicateOp.LESS_THAN,
                physical_min_x,
                physical_min_y,
                physical_max_x,
                physical_max_y,
            )
        if lower_x is None:
            return _zero_support_mbr_predicate_cached(
                compiled.numeric_literals_by_column[max_x_index],
                max_x_column,
                PredicateOp.GREATER_THAN,
                physical_min_x,
                physical_min_y,
                physical_max_x,
                physical_max_y,
            )
        if upper_y is None:
            return _zero_support_mbr_predicate_cached(
                compiled.numeric_literals_by_column[min_y_index],
                min_y_column,
                PredicateOp.LESS_THAN,
                physical_min_x,
                physical_min_y,
                physical_max_x,
                physical_max_y,
            )
        if lower_y is None:
            return _zero_support_mbr_predicate_cached(
                compiled.numeric_literals_by_column[max_y_index],
                max_y_column,
                PredicateOp.GREATER_THAN,
                physical_min_x,
                physical_min_y,
                physical_max_x,
                physical_max_y,
            )
        lower_x_value, lower_x_literal = lower_x
        upper_x_value, upper_x_literal = upper_x
        lower_y_value, lower_y_literal = lower_y
        upper_y_value, upper_y_literal = upper_y
        return CanonicalSegmentMbrPredicate(
            min_x=lower_x_value,
            min_y=lower_y_value,
            max_x=upper_x_value,
            max_y=upper_y_value,
            min_x_literal=lower_x_literal,
            min_y_literal=lower_y_literal,
            max_x_literal=upper_x_literal,
            max_y_literal=upper_y_literal,
            physical_min_x=physical_min_x,
            physical_min_y=physical_min_y,
            physical_max_x=physical_max_x,
            physical_max_y=physical_max_y,
        )

    def _sample_satisfied_predicate(
        self,
        cache: _ColumnPredicateCache | None,
        value: Any,
        rng: np.random.Generator,
        encoded_value_id: int | None = None,
    ) -> PredicateToken:
        if not self.enabled:
            return PredicateToken.wildcard()
        if cache is None or not _is_comparable_value(value):
            return PredicateToken.wildcard()
        comparable_values = cache.comparable_values
        left_rank = (
            cache.domain_id_to_left_rank[encoded_value_id]
            if encoded_value_id is not None
            else bisect_left(comparable_values, value)
        )
        right_rank = (
            cache.domain_id_to_right_rank[encoded_value_id]
            if encoded_value_id is not None
            else bisect_right(comparable_values, value)
        )
        roll = float(rng.random() * self._probability_total)
        if roll < self.wildcard_probability:
            return PredicateToken.wildcard()
        roll -= self.wildcard_probability
        if roll < self.equality_probability:
            return PredicateToken.equal(value)
        roll -= self.equality_probability
        if roll < self.lower_bound_probability:
            stop = right_rank
            if stop <= 0:
                return PredicateToken.wildcard()
            threshold = comparable_values[int(rng.integers(0, stop))]
            return PredicateToken(PredicateOp.GREATER_EQUAL, value=threshold)
        roll -= self.lower_bound_probability
        if roll < self.upper_bound_probability:
            start = left_rank
            if start >= len(comparable_values):
                return PredicateToken.wildcard()
            threshold = comparable_values[int(rng.integers(start, len(comparable_values)))]
            return PredicateToken(PredicateOp.LESS_EQUAL, value=threshold)
        return self._sample_row_range_style_predicate(
            comparable_values,
            left_rank=left_rank,
            right_rank=right_rank,
            domain_size=len(cache.comparable_values),
            rng=rng,
        )

    def _sample_row_range_style_predicate(
        self,
        comparable_values: tuple[Any, ...],
        *,
        left_rank: int,
        right_rank: int,
        domain_size: int,
        rng: np.random.Generator,
    ) -> PredicateToken:
        if not (
            self.enable_native_range_tokens
            and self.native_range_probability > 0.0
        ):
            if bool(rng.integers(0, 2)):
                stop = right_rank
                if stop <= 0:
                    return PredicateToken.wildcard()
                threshold = comparable_values[int(rng.integers(0, stop))]
                return PredicateToken(PredicateOp.GREATER_EQUAL, value=threshold)
            start = left_rank
            if start >= len(comparable_values):
                return PredicateToken.wildcard()
            threshold = comparable_values[int(rng.integers(start, len(comparable_values)))]
            return PredicateToken(PredicateOp.LESS_EQUAL, value=threshold)
        stop = right_rank
        start = left_rank
        if stop <= 0 or start >= len(comparable_values):
            return PredicateToken.wildcard()
        lower = comparable_values[int(rng.integers(0, stop))]
        upper = comparable_values[int(rng.integers(start, len(comparable_values)))]
        return PredicateToken.range(lower, upper)

    def _compiled(self, metadata: ModelMetadata) -> _CompiledGenerationMetadata:
        key = id(metadata)
        cached = self._compiled_by_metadata_id.get(key)
        if cached is not None:
            return cached
        column_name_to_index = {
            column.name: index for index, column in enumerate(metadata.columns)
        }
        data_indices = tuple(
            index
            for index, column in enumerate(metadata.columns)
            if column.kind == ColumnKind.DATA
        )
        indicator_indices = tuple(
            index
            for index, column in enumerate(metadata.columns)
            if column.kind == ColumnKind.INDICATOR
        )
        fanout_indices = tuple(
            index
            for index, column in enumerate(metadata.columns)
            if column.kind == ColumnKind.FANOUT
        )
        table_indicator_indices = {
            metadata.columns[index].table: index
            for index in indicator_indices
            if metadata.columns[index].table is not None
        }
        graph = infer_join_graph(metadata)
        table_to_bit = {table: 1 << index for index, table in enumerate(graph.tables)}
        bit_to_table = tuple(graph.tables)
        fanout_child_by_name = {
            metadata.columns[index].name: (
                _fanout_child_table(metadata.columns[index].fanout_source)
                or metadata.columns[index].table
            )
            for index in fanout_indices
        }
        column_caches = self._column_caches(metadata)
        numeric_literals_by_column = tuple(
            _numeric_domain_literals_from_cache(cache)
            for cache in column_caches
        )
        pol_required = (
            "segments:t_s",
            "segments:t_e",
            "segments:s_x",
            "segments:s_y",
            "segments:e_x",
            "segments:e_y",
        )
        endpoint_names = ("segments:s_x", "segments:s_y", "segments:e_x", "segments:e_y")
        mbr_names = (
            "segments:seg_min_x",
            "segments:seg_max_x",
            "segments:seg_min_y",
            "segments:seg_max_y",
        )
        compiled = _CompiledGenerationMetadata(
            column_name_to_index=column_name_to_index,
            data_indices=data_indices,
            indicator_indices=indicator_indices,
            fanout_indices=fanout_indices,
            table_indicator_indices=table_indicator_indices,
            graph=graph,
            fanout_child_by_name=fanout_child_by_name,
            column_caches=column_caches,
            pol_required_available=all(name in column_name_to_index for name in pol_required),
            temporal_start_index=column_name_to_index.get("segments:t_s"),
            temporal_end_index=column_name_to_index.get("segments:t_e"),
            endpoint_spatial_indices={
                name: column_name_to_index[name]
                for name in endpoint_names
                if name in column_name_to_index
            },
            mbr_spatial_indices={
                name: column_name_to_index[name]
                for name in mbr_names
                if name in column_name_to_index
            },
            endpoint_x_domain=_numeric_values_for_columns_from_cache(
                metadata,
                column_caches,
                ("segments:s_x", "segments:e_x"),
                column_name_to_index,
            ),
            endpoint_y_domain=_numeric_values_for_columns_from_cache(
                metadata,
                column_caches,
                ("segments:s_y", "segments:e_y"),
                column_name_to_index,
            ),
            numeric_literals_by_column=numeric_literals_by_column,
            table_to_bit=table_to_bit,
            bit_to_table=bit_to_table,
            data_table_bits=tuple(
                table_to_bit.get(column.table or "", 0) for column in metadata.columns
            ),
            fanout_child_bits=tuple(
                table_to_bit.get(
                    (
                        _fanout_child_table(column.fanout_source)
                        or column.table
                        or ""
                    ),
                    0,
                )
                if column.kind == ColumnKind.FANOUT
                else 0
                for column in metadata.columns
            ),
        )
        self._compiled_by_metadata_id[key] = compiled
        return compiled

    def _tokens_for_query_tables(
        self,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata,
        included_tables: set[str],
        inverse_fanout_columns: set[str],
        ordinary_predicates: dict[str, PredicateToken] | None = None,
    ) -> list[PredicateToken]:
        tokens, _ = self._tokens_and_two_slot_for_query_tables(
            metadata,
            compiled,
            included_tables,
            inverse_fanout_columns,
            ordinary_predicates,
        )
        return list(tokens)

    def _tokens_and_two_slot_for_query_tables(
        self,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata,
        included_tables: set[str],
        inverse_fanout_columns: set[str],
        ordinary_predicates: dict[str, PredicateToken] | None = None,
    ) -> tuple[tuple[PredicateToken, ...], tuple[tuple[int, int, int, int], ...] | None]:
        tokens = [_WILDCARD_TOKEN] * len(metadata.columns)
        encoded = list(_wildcard_two_slot_row(metadata))
        preencoding_supported = True
        for column_name, token in (ordinary_predicates or {}).items():
            column_index = compiled.column_name_to_index.get(column_name)
            if column_index is not None:
                tokens[column_index] = token
                if preencoding_supported:
                    try:
                        encoded[column_index] = _encode_token_two_slot_for_metadata(
                            metadata,
                            column_index,
                            token,
                        )
                    except ValueError:
                        preencoding_supported = False
        for table in included_tables:
            indicator_index = compiled.table_indicator_indices.get(table)
            if indicator_index is not None:
                tokens[indicator_index] = _INDICATOR_INCLUDED_TOKEN
                if preencoding_supported:
                    encoded[indicator_index] = _encode_token_two_slot_for_metadata(
                        metadata,
                        indicator_index,
                        _INDICATOR_INCLUDED_TOKEN,
                    )
        for column_index in compiled.fanout_indices:
            column = metadata.columns[column_index]
            if column.name in inverse_fanout_columns:
                tokens[column_index] = _INV_FANOUT_TOKEN
                if preencoding_supported:
                    encoded[column_index] = _encode_token_two_slot_for_metadata(
                        metadata,
                        column_index,
                        _INV_FANOUT_TOKEN,
                    )
        return tuple(tokens), tuple(encoded) if preencoding_supported else None

    def _present_tables_for_row(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata,
    ) -> frozenset[str]:
        if not compiled.table_indicator_indices:
            return frozenset(
                column.table
                for column in metadata.columns
                if column.table is not None
            )
        present: set[str] = set()
        for table, column_index in compiled.table_indicator_indices.items():
            column = metadata.columns[column_index]
            if column.domain[int(encoded_row[column_index])] == 1:
                present.add(table)
        return frozenset(present)

    def _present_table_mask_for_row(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata,
    ) -> int:
        if not compiled.table_indicator_indices:
            mask = 0
            for table in compiled.graph.tables:
                mask |= compiled.table_to_bit.get(table, 0)
            return mask
        mask = 0
        for table, column_index in compiled.table_indicator_indices.items():
            column = metadata.columns[column_index]
            if column.domain[int(encoded_row[column_index])] == 1:
                mask |= compiled.table_to_bit.get(table, 0)
        return mask

    def _tables_from_mask(self, compiled: _CompiledGenerationMetadata, mask: int) -> frozenset[str]:
        return frozenset(
            table for table in compiled.bit_to_table if mask & compiled.table_to_bit[table]
        )

    def _inverse_fanouts_for_table_subset(
        self,
        compiled: _CompiledGenerationMetadata,
        included_tables: frozenset[str] | set[str],
    ) -> frozenset[str]:
        included = set(included_tables)
        return frozenset(
            fanout_name
            for fanout_name, child_table in compiled.fanout_child_by_name.items()
            if child_table is not None and child_table not in included
        )

    def _inverse_fanouts_for_table_mask(
        self,
        metadata: ModelMetadata,
        compiled: _CompiledGenerationMetadata,
        included_mask: int,
    ) -> frozenset[str]:
        inverse: set[str] = set()
        for column_index in compiled.fanout_indices:
            child_bit = compiled.fanout_child_bits[column_index]
            if child_bit and not (included_mask & child_bit):
                inverse.add(metadata.columns[column_index].name)
        return frozenset(inverse)

    def _sample_included_table_mask(
        self,
        encoded_row: np.ndarray,
        metadata: ModelMetadata,
        rng: np.random.Generator,
        compiled: _CompiledGenerationMetadata,
    ) -> int:
        present_mask = self._present_table_mask_for_row(encoded_row, metadata, compiled)
        if not present_mask:
            return 0
        if self.table_subset_sampling == "full" or not self.enabled:
            return present_mask
        if self.table_subset_sampling == "neurocard_table_dropout_rooted":
            return self._neurocard_table_dropout_rooted_mask(compiled, present_mask, rng)
        return _table_mask_from_tables(
            self._sample_included_from_present(
                self._tables_from_mask(compiled, present_mask),
                metadata,
                rng,
                compiled,
            ),
            compiled,
        )

    def _neurocard_table_dropout_rooted_subset(
        self,
        compiled: _CompiledGenerationMetadata,
        present_tables: frozenset[str] | set[str],
        rng: np.random.Generator,
    ) -> frozenset[str]:
        graph = compiled.graph
        tables = tuple(graph.tables)
        present = set(present_tables)
        if not tables or graph.root_table not in present:
            return frozenset()
        if len(tables) <= 1:
            return frozenset({graph.root_table})
        dropped_count = int(rng.integers(1, len(tables)))
        drop_probability = dropped_count / len(tables)
        proposed = {
            table
            for table in tables
            if table == graph.root_table or float(rng.random()) > drop_probability
        }
        proposed.intersection_update(present)
        proposed.add(graph.root_table)
        return frozenset(
            _root_connected_component(proposed, graph.root_table, graph.edges)
        )

    def _neurocard_table_dropout_rooted_mask(
        self,
        compiled: _CompiledGenerationMetadata,
        present_mask: int,
        rng: np.random.Generator,
    ) -> int:
        graph = compiled.graph
        tables = tuple(graph.tables)
        root_bit = compiled.table_to_bit.get(graph.root_table, 0)
        if not tables or not (present_mask & root_bit):
            return 0
        if len(tables) <= 1:
            return root_bit
        dropped_count = int(rng.integers(1, len(tables)))
        drop_probability = dropped_count / len(tables)
        proposed_mask = root_bit
        for table in tables:
            if table == graph.root_table:
                continue
            table_bit = compiled.table_to_bit[table]
            if float(rng.random()) > drop_probability:
                proposed_mask |= table_bit
        proposed_mask &= present_mask | root_bit
        proposed_mask |= root_bit
        return _root_connected_component_mask(proposed_mask, compiled, graph)

    def _column_caches(
        self,
        metadata: ModelMetadata,
    ) -> tuple[_ColumnPredicateCache | None, ...]:
        key = id(metadata)
        cached = self._cache_by_metadata_id.get(key)
        if cached is not None:
            return cached
        caches = [self._build_column_cache(column) for column in metadata.columns]
        result = tuple(caches)
        self._cache_by_metadata_id[key] = result
        return result

    def _column_cache(
        self,
        metadata: ModelMetadata,
        column_index: int,
    ) -> _ColumnPredicateCache | None:
        key = id(metadata)
        cached = self._cache_by_metadata_id.get(key)
        if cached is not None:
            return cached[column_index]
        single_cache = getattr(self, "_single_column_cache_by_metadata_id", None)
        if single_cache is None:
            single_cache = {}
            self._single_column_cache_by_metadata_id = single_cache
        column_key = (key, int(column_index))
        if column_key not in single_cache:
            single_cache[column_key] = self._build_column_cache(metadata.columns[column_index])
        return single_cache[column_key]

    def _build_column_cache(self, column: Any) -> _ColumnPredicateCache | None:
        if column.kind != ColumnKind.DATA:
            return None
        return _build_column_predicate_cache(column.domain)


def _build_column_predicate_cache(domain: tuple[Any, ...]) -> _ColumnPredicateCache | None:
    values = _sorted_comparable_domain_values(domain)
    if not values:
        return None
    rank_by_value: dict[Any, tuple[int, int]] = {}
    for rank, value in enumerate(values):
        try:
            previous = rank_by_value.get(value)
            left = rank if previous is None else previous[0]
            rank_by_value[value] = (left, rank + 1)
        except TypeError:
            continue
    left_ranks: list[int] = []
    right_ranks: list[int] = []
    for value in domain:
        if _is_comparable_value(value):
            try:
                ranks = rank_by_value.get(value)
            except TypeError:
                ranks = None
            if ranks is None:
                ranks = (bisect_left(values, value), bisect_right(values, value))
            left_ranks.append(ranks[0])
            right_ranks.append(ranks[1])
        else:
            left_ranks.append(-1)
            right_ranks.append(0)
    return _ColumnPredicateCache(
        comparable_values=values,
        domain_id_to_left_rank=tuple(left_ranks),
        domain_id_to_right_rank=tuple(right_ranks),
    )


def _numeric_domain_literals_from_cache(
    cache: _ColumnPredicateCache | None,
) -> _NumericLiteralCache | None:
    if cache is None:
        return None
    values: list[float] = []
    literals: list[Any] = []
    seen: set[float] = set()
    for literal in cache.comparable_values:
        if isinstance(literal, str) and literal.startswith("__"):
            continue
        try:
            numeric = float(literal)
        except (TypeError, ValueError):
            continue
        if numeric in seen:
            continue
        seen.add(numeric)
        values.append(numeric)
        literals.append(literal)
    if not values:
        return None
    return _NumericLiteralCache(values=tuple(values), literals=tuple(literals))


def _floor_numeric_literal(
    cache: _NumericLiteralCache | None,
    upper: float,
) -> tuple[float, Any] | None:
    if cache is None:
        return None
    index = bisect_right(cache.values, float(upper)) - 1
    if index < 0:
        return None
    return cache.values[index], cache.literals[index]


def _ceil_numeric_literal(
    cache: _NumericLiteralCache | None,
    lower: float,
) -> tuple[float, Any] | None:
    if cache is None:
        return None
    index = bisect_left(cache.values, float(lower))
    if index >= len(cache.values):
        return None
    return cache.values[index], cache.literals[index]


def _zero_support_mbr_predicate_cached(
    cache: _NumericLiteralCache | None,
    column_name: str,
    op: PredicateOp,
    physical_min_x: float,
    physical_min_y: float,
    physical_max_x: float,
    physical_max_y: float,
) -> Any:
    from model.src.data.trajectory_distinct import CanonicalSegmentMbrPredicate

    if cache is None or not cache.values:
        raise ValueError(f"MBR column {column_name!r} has no numeric domain literals")
    index = 0 if op == PredicateOp.LESS_THAN else len(cache.values) - 1
    token = PredicateToken(op, value=cache.literals[index])
    return CanonicalSegmentMbrPredicate(
        min_x=physical_min_x,
        min_y=physical_min_y,
        max_x=physical_max_x,
        max_y=physical_max_y,
        min_x_literal=None,
        min_y_literal=None,
        max_x_literal=None,
        max_y_literal=None,
        physical_min_x=physical_min_x,
        physical_min_y=physical_min_y,
        physical_max_x=physical_max_x,
        physical_max_y=physical_max_y,
        zero_support=True,
        zero_support_column=column_name,
        zero_support_token=token,
    )


def _wildcard_two_slot_row(metadata: ModelMetadata) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            TWO_SLOT_EMPTY_OPERATOR_ID,
            len(column.domain),
            TWO_SLOT_EMPTY_OPERATOR_ID,
            len(column.domain),
        )
        for column in metadata.columns
    )


def _encode_token_two_slot_for_metadata(
    metadata: ModelMetadata,
    column_index: int,
    token: PredicateToken,
) -> tuple[int, int, int, int]:
    column = metadata.columns[column_index]
    missing_value_id = len(column.domain)
    empty = (TWO_SLOT_EMPTY_OPERATOR_ID, missing_value_id)
    if token.op == PredicateOp.WILDCARD:
        return (*empty, *empty)
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
        return (
            TWO_SLOT_OP_TO_ID[lower_op],
            column.encode_value(token.value),
            TWO_SLOT_OP_TO_ID[upper_op],
            column.encode_value(token.upper),
        )
    if token.op == PredicateOp.INV_FANOUT:
        return (TWO_SLOT_OP_TO_ID[token.op], missing_value_id, *empty)
    if token.op in TWO_SLOT_OP_TO_ID:
        return (TWO_SLOT_OP_TO_ID[token.op], column.encode_value(token.value), *empty)
    raise ValueError(f"unsupported predicate token {token!r}")


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
    """Return the persisted join graph, with conservative legacy fallbacks."""

    if metadata.join_tables and metadata.join_edges:
        return JoinGraphMetadata(
            root_table=metadata.join_root or metadata.join_tables[0],
            tables=tuple(metadata.join_tables),
            edges=tuple(metadata.join_edges),
        )
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
    if not edges and "title" in tables and _looks_like_job_light_tables(tables):
        root = "title"
        edges = [(root, table) for table in tables if table != root]
    return JoinGraphMetadata(root_table=root, tables=tables, edges=tuple(edges))


def _looks_like_job_light_tables(tables: tuple[str, ...]) -> bool:
    job_light_tables = {
        "title",
        "cast_info",
        "movie_companies",
        "movie_info",
        "movie_info_idx",
        "movie_keyword",
    }
    return set(tables).issubset(job_light_tables) and len(set(tables)) > 1


def connected_table_subsets(
    metadata: ModelMetadata,
    *,
    allowed_tables: frozenset[str] | set[str] | None = None,
    required_root: str | None = None,
) -> tuple[frozenset[str], ...]:
    """Enumerate nonempty connected table subsets within the join graph."""

    graph = infer_join_graph(metadata)
    allowed = set(allowed_tables if allowed_tables is not None else graph.tables)
    tables = tuple(table for table in graph.tables if table in allowed)
    if not graph.edges:
        return tuple(
            frozenset((table,))
            for table in tables
            if required_root is None or table == required_root
        )
    subsets: list[frozenset[str]] = []
    for size in range(1, len(tables) + 1):
        for combo in combinations(tables, size):
            subset = frozenset(combo)
            if required_root is not None and required_root not in subset:
                continue
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


def neurocard_table_dropout_rooted_subset(
    metadata: ModelMetadata,
    present_tables: frozenset[str] | set[str],
    rng: np.random.Generator,
) -> frozenset[str]:
    """Map NeuroCard's root-protected table dropout law to query subsets."""

    graph = infer_join_graph(metadata)
    tables = tuple(graph.tables)
    present = set(present_tables)
    if not tables or graph.root_table not in present:
        return frozenset()
    if len(tables) <= 1:
        return frozenset({graph.root_table})
    dropped_count = int(rng.integers(1, len(tables)))
    drop_probability = dropped_count / len(tables)
    proposed = {
        table
        for table in tables
        if table == graph.root_table or float(rng.random()) > drop_probability
    }
    proposed.intersection_update(present)
    proposed.add(graph.root_table)
    return frozenset(_root_connected_component(proposed, graph.root_table, graph.edges))


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


def predicate_context_diagnostics(
    contexts: list[GeneratedTrainingContext],
    metadata: ModelMetadata,
) -> dict[str, Any]:
    """Summarize predicate diversity and row-local Duet token coverage."""

    unique_keys = {
        tuple(token.stable_key() for token in context.tokens)
        for context in contexts
    }
    per_column: dict[str, dict[str, Any]] = {
        column.name: {
            "equality_rows": 0,
            "lower_bound_rows": 0,
            "upper_bound_rows": 0,
            "two_sided_range_rows": 0,
            "wildcard_rows": 0,
            "unique_equality_literals": 0,
            "unique_lower_literals": 0,
            "unique_upper_literals": 0,
            "_equality_literals": set(),
            "_lower_literals": set(),
            "_upper_literals": set(),
        }
        for column in metadata.columns
    }
    table_subset_distribution: dict[str, int] = {}
    tables_for_diagnostics = tuple(
        metadata.join_tables
        or tuple(
            dict.fromkeys(
                column.table
                for column in metadata.columns
                if column.table is not None and column.kind != ColumnKind.FANOUT
            )
        )
    )
    per_table_inclusion_counts: dict[str, int] = {
        table: 0 for table in tables_for_diagnostics
    }
    predicate_choice_counts = {
        "wildcard": 0,
        "equality": 0,
        "lower": 0,
        "upper": 0,
        "two_sided_range": 0,
    }
    full_table_count = len(
        tables_for_diagnostics
    )
    for context in contexts:
        table_count = len(context.included_tables)
        key = "full-query" if full_table_count and table_count == full_table_count else f"{table_count}-table"
        table_subset_distribution[key] = int(table_subset_distribution.get(key, 0)) + 1
        for table in context.included_tables:
            if table in per_table_inclusion_counts:
                per_table_inclusion_counts[table] += 1
        for column, token in zip(metadata.columns, context.tokens):
            column_stats = per_column[column.name]
            if column.kind == ColumnKind.DATA:
                if token.op == PredicateOp.WILDCARD:
                    predicate_choice_counts["wildcard"] += 1
                elif token.op == PredicateOp.EQUAL:
                    predicate_choice_counts["equality"] += 1
                elif token.op in {PredicateOp.GREATER_EQUAL, PredicateOp.GREATER_THAN}:
                    predicate_choice_counts["lower"] += 1
                elif token.op in {PredicateOp.LESS_EQUAL, PredicateOp.LESS_THAN}:
                    predicate_choice_counts["upper"] += 1
                elif token.op == PredicateOp.RANGE:
                    predicate_choice_counts["two_sided_range"] += 1
            if token.op == PredicateOp.WILDCARD:
                column_stats["wildcard_rows"] += 1
            elif token.op == PredicateOp.EQUAL:
                column_stats["equality_rows"] += 1
                column_stats["_equality_literals"].add(repr(token.value))
            elif token.op in {PredicateOp.GREATER_EQUAL, PredicateOp.GREATER_THAN}:
                column_stats["lower_bound_rows"] += 1
                column_stats["_lower_literals"].add(repr(token.value))
            elif token.op in {PredicateOp.LESS_EQUAL, PredicateOp.LESS_THAN}:
                column_stats["upper_bound_rows"] += 1
                column_stats["_upper_literals"].add(repr(token.value))
            elif token.op == PredicateOp.RANGE:
                column_stats["two_sided_range_rows"] += 1
                column_stats["_lower_literals"].add(repr(token.value))
                column_stats["_upper_literals"].add(repr(token.upper))
    for column_stats in per_column.values():
        column_stats["unique_equality_literals"] = len(column_stats.pop("_equality_literals"))
        column_stats["unique_lower_literals"] = len(column_stats.pop("_lower_literals"))
        column_stats["unique_upper_literals"] = len(column_stats.pop("_upper_literals"))
    return {
        "unique_predicate_context_rows": len(unique_keys),
        "unique_predicate_context_fraction": (
            len(unique_keys) / max(len(contexts), 1)
        ),
        "per_column": per_column,
        "table_subset_cardinality_distribution": table_subset_distribution,
        "table_inclusion_probability": {
            table: count / max(len(contexts), 1)
            for table, count in per_table_inclusion_counts.items()
        },
        "empirical_predicate_choice_counts": predicate_choice_counts,
        "empirical_predicate_choice_frequencies": {
            key: count / max(sum(predicate_choice_counts.values()), 1)
            for key, count in predicate_choice_counts.items()
        },
    }


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


def forced_predicate_for_stratum(
    stratum: Any,
    *,
    encoded_row: np.ndarray,
    metadata: ModelMetadata,
    debug_allow_row_dependent_native_range_tail: bool = False,
) -> PredicateToken:
    column_index = int(stratum.column_index)
    column = metadata.columns[column_index]
    row_value = column.domain[int(encoded_row[column_index])]
    if getattr(stratum, "support_bottleneck", None) == "native_range":
        if stratum.region_type == "equality":
            return PredicateToken.range(stratum.value, stratum.value)
        if stratum.region_type == "lower_tail":
            if debug_allow_row_dependent_native_range_tail:
                return PredicateToken.range(stratum.lower, row_value)
            singleton = _native_range_singleton_support_value(stratum, column.domain)
            if singleton is not None:
                return PredicateToken.range(singleton, singleton)
            raise ValueError(
                "non-singleton native-range lower-tail rare strata cannot use "
                "a row-dependent boundary unless "
                "debug_allow_row_dependent_native_range_tail=true"
            )
        if stratum.region_type == "upper_tail":
            if debug_allow_row_dependent_native_range_tail:
                return PredicateToken.range(row_value, stratum.upper)
            singleton = _native_range_singleton_support_value(stratum, column.domain)
            if singleton is not None:
                return PredicateToken.range(singleton, singleton)
            raise ValueError(
                "non-singleton native-range upper-tail rare strata cannot use "
                "a row-dependent boundary unless "
                "debug_allow_row_dependent_native_range_tail=true"
            )
    if stratum.region_type == "equality":
        return PredicateToken.equal(stratum.value)
    if stratum.region_type == "lower_tail":
        return PredicateToken(PredicateOp.GREATER_EQUAL, value=stratum.lower)
    if stratum.region_type == "upper_tail":
        return PredicateToken(PredicateOp.LESS_EQUAL, value=stratum.upper)
    if stratum.region_type == "range":
        return PredicateToken.range(stratum.lower, stratum.upper)
    raise ValueError(f"unsupported rare stratum region_type {stratum.region_type!r}")


def _native_range_singleton_support_value(
    stratum: Any,
    domain: tuple[Any, ...],
) -> Any | None:
    matching_values = []
    for value in domain:
        try:
            if stratum.contains_value(value):
                matching_values.append(value)
        except (TypeError, ValueError):
            continue
        if len(matching_values) > 1:
            return None
    if len(matching_values) == 1:
        return matching_values[0]
    return None


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


def _is_comparable_value(value: Any) -> bool:
    if isinstance(value, str) and value.startswith("__"):
        return False
    try:
        _ = value <= value
    except TypeError:
        return False
    return True


def _sorted_comparable_domain_values(domain: tuple[Any, ...]) -> tuple[Any, ...]:
    if _domain_is_already_sorted_comparable(domain):
        return domain
    values = comparable_domain_values(domain)
    try:
        return tuple(sorted(values))
    except TypeError:
        return ()


def _domain_is_already_sorted_comparable(domain: tuple[Any, ...]) -> bool:
    previous = None
    has_value = False
    for value in domain:
        if not _is_comparable_value(value):
            return False
        if previous is not None and value < previous:
            return False
        previous = value
        has_value = True
    return has_value


def _semantic_owned_column_names(trajectory_query: Any) -> frozenset[str]:
    owned: set[str] = set()
    for temporal in getattr(trajectory_query, "temporal_predicates", ()):
        owned.add(temporal.start_column)
        owned.add(temporal.end_column)
    for spatial in getattr(trajectory_query, "spatial_predicates", ()):
        if hasattr(spatial, "start_x_column"):
            owned.update(
                (
                    spatial.start_x_column,
                    spatial.start_y_column,
                    spatial.end_x_column,
                    spatial.end_y_column,
                )
            )
        else:
            owned.update(
                (
                    spatial.min_x_column,
                    spatial.max_x_column,
                    spatial.min_y_column,
                    spatial.max_y_column,
                    "segments:s_x",
                    "segments:s_y",
                    "segments:e_x",
                    "segments:e_y",
                )
            )
    return frozenset(owned)


def _numeric_values_for_columns(
    metadata: ModelMetadata,
    column_names: tuple[str, ...],
) -> tuple[float, ...]:
    column_name_to_index = {
        column.name: index for index, column in enumerate(metadata.columns)
    }
    caches = tuple(
        _build_column_predicate_cache(column.domain)
        if column.kind == ColumnKind.DATA
        else None
        for column in metadata.columns
    )
    return _numeric_values_for_columns_from_cache(
        metadata,
        caches,
        column_names,
        column_name_to_index,
    )


def _numeric_values_for_columns_from_cache(
    metadata: ModelMetadata,
    column_caches: tuple[_ColumnPredicateCache | None, ...],
    column_names: tuple[str, ...],
    column_name_to_index: dict[str, int],
) -> tuple[float, ...]:
    values: set[float] = set()
    for column_name in column_names:
        column_index = column_name_to_index.get(column_name)
        if column_index is None:
            continue
        cache = column_caches[column_index]
        if cache is None:
            continue
        for value in cache.comparable_values:
            try:
                values.add(float(value))
            except (TypeError, ValueError):
                continue
    return tuple(sorted(values))


def _deterministic_validation_indices(total_count: int, sample_count: int) -> tuple[int, ...]:
    if total_count <= 0 or sample_count <= 0:
        return ()
    if sample_count >= total_count:
        return tuple(range(total_count))
    return tuple(
        int(index)
        for index in np.linspace(0, total_count - 1, num=sample_count, dtype=int)
    )


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


def _root_connected_component(
    tables: set[str],
    root_table: str,
    edges: tuple[tuple[str, str], ...],
) -> set[str]:
    if root_table not in tables:
        return set()
    if not edges:
        return {root_table}
    adjacency = {table: set() for table in tables}
    for left, right in edges:
        if left in tables and right in tables:
            adjacency[left].add(right)
            adjacency[right].add(left)
    seen = {root_table}
    stack = [root_table]
    while stack:
        table = stack.pop()
        for neighbor in adjacency.get(table, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _table_mask_from_tables(
    tables: frozenset[str] | set[str],
    compiled: _CompiledGenerationMetadata,
) -> int:
    mask = 0
    for table in tables:
        mask |= compiled.table_to_bit.get(table, 0)
    return mask


def _root_connected_component_mask(
    table_mask: int,
    compiled: _CompiledGenerationMetadata,
    graph: JoinGraphMetadata,
) -> int:
    root_bit = compiled.table_to_bit.get(graph.root_table, 0)
    if not (table_mask & root_bit):
        return 0
    if not graph.edges:
        return root_bit
    adjacency: dict[str, set[str]] = {
        table: set()
        for table in compiled.bit_to_table
        if table_mask & compiled.table_to_bit[table]
    }
    for left, right in graph.edges:
        left_bit = compiled.table_to_bit.get(left, 0)
        right_bit = compiled.table_to_bit.get(right, 0)
        if (table_mask & left_bit) and (table_mask & right_bit):
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
    seen = {graph.root_table}
    stack = [graph.root_table]
    while stack:
        table = stack.pop()
        for neighbor in adjacency.get(table, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return _table_mask_from_tables(seen, compiled)


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
