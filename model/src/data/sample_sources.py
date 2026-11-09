from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from model.src.data.full_join_sampler import (
    FullJoinBatch,
    LiveNeuroCardFullJoinSampleSource,
    NeuroCardFullJoinSampleSource,
    SyntheticFullJoinSampleSource,
)
from model.src.data.importance_sampling import (
    ImportanceSamplingSampleSource,
    RareSupportSampleSource,
)
from model.src.data.trajectory_distinct import TrajectoryDistinctRuntimeConfig
from model.src.data.schema import ColumnKind, FactorizationPlan, ModelMetadata
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
)
from model.src.predicates.generation import GeneratedTrainingContext


def sample_source_from_config(
    config: dict[str, Any],
    *,
    startup_callback: object | None = None,
) -> object:
    """Construct a sample source without coupling trainers to concrete classes."""

    dataset = config.get("dataset", {})
    dataset_type = dataset.get("type", "synthetic_full_join")
    if dataset_type == "synthetic_full_join":
        source = SyntheticFullJoinSampleSource()
    elif dataset_type == "neurocard_full_join":
        sampling_mode = str(dataset.get("sampling_mode", "fixture"))
        if bool(config.get("trajectory_distinct", {}).get("enabled", False)) and sampling_mode == "live":
            raise ValueError(
                "trajectory_distinct.enabled=true with sampling_mode=live is unsupported "
                "because live sampler provenance is unavailable"
            )
        if sampling_mode == "live":
            source = LiveNeuroCardFullJoinSampleSource(
                Path(dataset["prepared_directory"]),
                csv_directory=Path(dataset["csv_directory"]),
                neurocard_path=dataset.get("neurocard_path"),
                sampler_batch_size=int(
                    dataset.get("sampler_batch_size", dataset.get("sample_batch_size", 16384))
                ),
                seed=int(dataset.get("sampler_seed", config.get("training", {}).get("seed", 0))),
                startup_callback=startup_callback,
            )
        else:
            source = NeuroCardFullJoinSampleSource(
                Path(dataset["prepared_directory"]),
                sampling_mode=sampling_mode,
                trajectory_ids_path=dataset.get("trajectory_ids_path"),
                segment_ids_path=dataset.get("segment_ids_path"),
                trajectory_index_path=dataset.get("trajectory_index_path"),
                preload_trajectory_index=bool(dataset.get("preload_trajectory_index", False)),
            )
    elif dataset_type == "pol_trajectory_full_join":
        if str(dataset.get("sampling_mode", "fixture")) == "live":
            raise ValueError(
                "pol_trajectory_full_join trajectory distinct currently supports fixture "
                "mode only; live provenance must be emitted with sampled rows first"
            )
        source = NeuroCardFullJoinSampleSource(
            Path(dataset["prepared_directory"]),
            sampling_mode=str(dataset.get("sampling_mode", "fixture")),
            trajectory_ids_path=dataset.get("trajectory_ids_path"),
            segment_ids_path=dataset.get("segment_ids_path"),
            trajectory_index_path=dataset.get("trajectory_index_path"),
            preload_trajectory_index=bool(dataset.get("preload_trajectory_index", False)),
        )
    else:
        raise ValueError(f"unsupported dataset.type {dataset_type!r}")
    if bool(config.get("trajectory_distinct", {}).get("enabled", False)):
        validate = getattr(source, "validate_trajectory_distinct", None)
        if validate is None:
            raise ValueError(
                "trajectory_distinct.enabled=true requires a sample source with "
                "trajectory distinct startup validation"
            )
        validate(
            runtime_config=TrajectoryDistinctRuntimeConfig.from_dict(
                config.get("trajectory_distinct", {})
            )
        )
    factorization = FactorizationConfig.from_dict(config.get("factorization", {}))
    if not factorization.enabled:
        wrapped = source
    else:
        wrapped = FactorizedMetadataSampleSource(source, factorization)
    importance = config.get("importance_sampling", {})
    if bool(importance.get("enabled", False)):
        wrapped = ImportanceSamplingSampleSource(wrapped, config)
    rare_support = config.get("rare_support", {})
    if bool(rare_support.get("enabled", False)):
        wrapped = RareSupportSampleSource(wrapped, config)
    table_order = tuple(config.get("schema", {}).get("table_order", ()))
    if table_order:
        wrapped = ColumnReorderedSampleSource(
            wrapped,
            table_order=table_order,
            factorization=factorization,
        )
    return wrapped


class ColumnReorderedSampleSource:
    """Expose a table-major AR order without rewriting prepared row artifacts."""

    _KIND_ORDER = {
        ColumnKind.DATA: 0,
        ColumnKind.INDICATOR: 1,
        ColumnKind.FANOUT: 2,
    }

    def __init__(
        self,
        base_source: object,
        *,
        table_order: tuple[str, ...],
        factorization: FactorizationConfig,
    ) -> None:
        self.base_source = base_source
        old_metadata = base_source.metadata  # type: ignore[attr-defined]
        if len(table_order) != len(set(table_order)):
            raise ValueError("schema.table_order must not contain duplicate tables")
        present_tables = {
            column.table for column in old_metadata.columns if column.table is not None
        }
        if set(table_order) != present_tables:
            raise ValueError(
                "schema.table_order must list every modeled table exactly once: "
                f"expected={sorted(present_tables)}, got={list(table_order)}"
            )
        table_rank = {table: index for index, table in enumerate(table_order)}
        self._permutation = np.asarray(
            sorted(
                range(len(old_metadata.columns)),
                key=lambda index: (
                    self._KIND_ORDER[old_metadata.columns[index].kind],
                    table_rank[old_metadata.columns[index].table],
                    index,
                ),
            ),
            dtype=np.int64,
        )
        self._old_to_new = {
            int(old_index): new_index
            for new_index, old_index in enumerate(self._permutation.tolist())
        }
        self._inverse_permutation = np.argsort(self._permutation)
        self._trajectory_provider: object | None = None
        reordered = ModelMetadata(
            columns=tuple(old_metadata.columns[index] for index in self._permutation),
            full_join_cardinality=old_metadata.full_join_cardinality,
            column_order=old_metadata.column_order,
            upstream_attribution=dict(old_metadata.upstream_attribution),
            schema_hash=None,
            factorization_plan=FactorizationPlan(),
            join_root=old_metadata.join_root,
            join_tables=old_metadata.join_tables,
            join_edges=old_metadata.join_edges,
        )
        self._metadata = (
            apply_factorization_to_metadata(reordered, factorization)
            if factorization.enabled
            else reordered
        )

    @property
    def join_cardinality(self) -> int:
        return int(self.base_source.join_cardinality)  # type: ignore[attr-defined]

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def trajectory_multiplicity_provider(self) -> object:
        if self._trajectory_provider is None:
            self._trajectory_provider = _ColumnOrderTrajectoryMultiplicityProvider(
                getattr(self.base_source, "trajectory_multiplicity_provider"),
                self._context_in_base_order,
            )
        return self._trajectory_provider

    @property
    def sampler_run_calls(self) -> int | None:
        return getattr(self.base_source, "sampler_run_calls", None)

    @property
    def distinct_original_rows_seen_estimate(self) -> object:
        return getattr(self.base_source, "distinct_original_rows_seen_estimate", None)

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        return self._reorder_batch(
            self.base_source.batches(batch_size, seed=seed)  # type: ignore[attr-defined]
        )

    def rare_batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        sample = getattr(self.base_source, "rare_batches", None)
        if sample is None:
            raise AttributeError("base source does not provide rare auxiliary batches")
        return self._reorder_batch(sample(batch_size, seed=seed), remap_strata=True)

    def discard_buffer(self) -> None:
        discard = getattr(self.base_source, "discard_buffer", None)
        if discard is not None:
            discard()

    def importance_sampling_summary(self, *args: Any, **kwargs: Any) -> object:
        summary = getattr(self.base_source, "importance_sampling_summary", None)
        return {"enabled": False} if summary is None else summary(*args, **kwargs)

    def rare_support_summary(self, *args: Any, **kwargs: Any) -> object:
        summary = getattr(self.base_source, "rare_support_summary", None)
        return {"enabled": False} if summary is None else summary(*args, **kwargs)

    def conjunctive_summary(self) -> dict[str, Any]:
        summary = getattr(self.base_source, "conjunctive_summary", None)
        return {} if summary is None else summary()

    def update_importance_context_statistics(self, *args: Any, **kwargs: Any) -> None:
        update = getattr(self.base_source, "update_importance_context_statistics", None)
        if update is not None:
            update(*args, **kwargs)

    def _reorder_batch(
        self,
        batch: FullJoinBatch,
        *,
        remap_strata: bool = False,
    ) -> FullJoinBatch:
        weights = batch.importance_weights
        if weights is not None and np.asarray(weights).ndim == 2:
            weights = np.asarray(weights)[:, self._permutation]
        metadata = batch.importance_metadata
        if remap_strata and metadata is not None:
            metadata = dict(metadata)
            strata = tuple(
                self._remap_stratum(stratum)
                for stratum in metadata.get("selected_strata", ())
            )
            metadata["selected_strata"] = strata
            metadata["selected_stratum_column_index"] = [
                int(stratum.column_index) for stratum in strata
            ]
        return FullJoinBatch(
            encoded_values=np.asarray(batch.encoded_values[:, self._permutation]),
            column_metadata=self._metadata.columns,
            raw_values=None,
            trajectory_ids=batch.trajectory_ids,
            segment_ids=batch.segment_ids,
            fresh_rows_drawn=batch.fresh_rows_drawn,
            fixture_rows_reused=batch.fixture_rows_reused,
            importance_weights=weights,
            importance_metadata=metadata,
        )

    def _remap_stratum(self, stratum: object) -> object:
        changes: dict[str, object] = {
            "column_index": self._old_to_new[int(stratum.column_index)],  # type: ignore[attr-defined]
        }
        terms = getattr(stratum, "terms", None)
        if terms is not None:
            changes["terms"] = tuple(
                replace(term, column_index=self._old_to_new[int(term.column_index)])
                for term in terms
            )
        return replace(stratum, **changes)

    def _context_in_base_order(
        self,
        context: GeneratedTrainingContext,
    ) -> GeneratedTrainingContext:
        tokens = tuple(context.tokens[index] for index in self._inverse_permutation)
        encoded = context.encoded_two_slot_row
        if encoded is not None:
            encoded = tuple(encoded[index] for index in self._inverse_permutation)
        return replace(context, tokens=tokens, encoded_two_slot_row=encoded)


class _ColumnOrderTrajectoryMultiplicityProvider:
    """Restore prepared-index column order at the trajectory provider boundary."""

    def __init__(
        self,
        base_provider: object,
        restore_context: Callable[[GeneratedTrainingContext], GeneratedTrainingContext],
    ) -> None:
        self.base_provider = base_provider
        self.restore_context = restore_context

    def evaluate_batch(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> object:
        return self.base_provider.evaluate_batch(  # type: ignore[attr-defined]
            anchor_trajectory_ids=anchor_trajectory_ids,
            contexts=tuple(self.restore_context(context) for context in contexts),
        )

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        return self.base_provider.matching_multiplicities(  # type: ignore[attr-defined]
            anchor_trajectory_ids=anchor_trajectory_ids,
            contexts=tuple(self.restore_context(context) for context in contexts),
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self.base_provider, name)


class FactorizedMetadataSampleSource:
    """Expose factorized metadata while keeping sample rows in original form."""

    def __init__(self, base_source: object, factorization: FactorizationConfig) -> None:
        self.base_source = base_source
        self._metadata = apply_factorization_to_metadata(
            base_source.metadata, factorization  # type: ignore[attr-defined]
        )

    @property
    def join_cardinality(self) -> int:
        return int(self.base_source.join_cardinality)  # type: ignore[attr-defined]

    @property
    def metadata(self) -> object:
        return self._metadata

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        batch = self.base_source.batches(batch_size, seed=seed)  # type: ignore[attr-defined]
        return FullJoinBatch(
            encoded_values=batch.encoded_values,
            column_metadata=self._metadata.columns,
            raw_values=batch.raw_values,
            trajectory_ids=batch.trajectory_ids,
            segment_ids=batch.segment_ids,
            fresh_rows_drawn=batch.fresh_rows_drawn,
            fixture_rows_reused=batch.fixture_rows_reused,
            importance_weights=batch.importance_weights,
            importance_metadata=batch.importance_metadata,
        )

    @property
    def sampler_run_calls(self) -> int | None:
        return getattr(self.base_source, "sampler_run_calls", None)

    @property
    def distinct_original_rows_seen_estimate(self) -> object:
        return getattr(self.base_source, "distinct_original_rows_seen_estimate", None)

    @property
    def trajectory_multiplicity_provider(self) -> object:
        return getattr(self.base_source, "trajectory_multiplicity_provider")

    def discard_buffer(self) -> None:
        discard = getattr(self.base_source, "discard_buffer", None)
        if discard is not None:
            discard()

    def prepare_root_strata(self, strata: object) -> None:
        prepare = getattr(self.base_source, "prepare_root_strata", None)
        if prepare is not None:
            prepare(strata)

    def sample_root_strata_rows(self, strata: object, *, rng: object) -> object:
        sample = getattr(self.base_source, "sample_root_strata_rows", None)
        if sample is None:
            raise AttributeError("base source does not support batched root strata sampling")
        return sample(strata, rng=rng)
