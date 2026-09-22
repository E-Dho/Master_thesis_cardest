from __future__ import annotations

from pathlib import Path
from typing import Any

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
from model.src.data.null_sentinel import (
    NullSentinelConfig,
    VoidPairCatalog,
    apply_null_sentinel_to_metadata,
    build_void_pair_catalog,
    sentinel_width_changes,
)
from model.src.data.trajectory_distinct import TrajectoryDistinctRuntimeConfig
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
)


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
    null_sentinel = NullSentinelConfig.from_dict(config.get("null_sentinel", {}))
    if null_sentinel.enabled:
        # Must wrap before factorization so the bitwise plan is built against the
        # sentinel-extended domain size rather than the original one.
        source = NullSentinelSampleSource(source, null_sentinel)
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
        return RareSupportSampleSource(wrapped, config)
    return wrapped


class NullSentinelSampleSource:
    """Extend DATA column domains with a void sentinel and build its catalog.

    Sample rows pass through untouched: real data never carries the sentinel id,
    which is exactly why the sentinel is appended rather than inserted.  Only
    void training contexts, built by the predicate generator from this source's
    catalog, ever target it.
    """

    def __init__(self, base_source: object, config: NullSentinelConfig) -> None:
        self.base_source = base_source
        self.null_sentinel_config = config
        self._metadata = apply_null_sentinel_to_metadata(
            base_source.metadata,  # type: ignore[attr-defined]
            config,
        )
        self._catalog: VoidPairCatalog | None = None
        self._catalog_built = False
        self.width_changed_columns = sentinel_width_changes(self._metadata)

    @property
    def join_cardinality(self) -> int:
        return int(self.base_source.join_cardinality)  # type: ignore[attr-defined]

    @property
    def metadata(self) -> object:
        return self._metadata

    def void_pair_catalog(self) -> VoidPairCatalog | None:
        """Build (once) the co-occurrence catalog over materialized sample rows.

        Returns None when the source cannot expose a complete row set.  Absence
        of a pair in a partial scan would not prove the pair impossible, so
        rather than manufacture unsound voids the feature stays inert and the
        trainer reports that no catalog was available.
        """

        if self._catalog_built:
            return self._catalog
        self._catalog_built = True
        rows = _materialized_encoded_rows(self.base_source)
        if rows is None:
            self._catalog = None
            return None
        self._catalog = build_void_pair_catalog(
            rows,
            self._metadata,
            self.null_sentinel_config,
        )
        return self._catalog

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

    def __getattr__(self, name: str) -> Any:
        # Everything this wrapper does not override -- trajectory providers,
        # strata preparation, sampler counters -- belongs to the base source.
        return getattr(self.base_source, name)


def _materialized_encoded_rows(source: object) -> Any | None:
    """Return the complete encoded row set a source can expose, or None."""

    dataset = getattr(source, "dataset", None)
    if dataset is not None:
        rows = getattr(dataset, "encoded_rows", None)
        if rows is not None:
            return rows
    sample_rows = getattr(source, "_sample_rows", None)
    if callable(sample_rows):
        return sample_rows()
    return None


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
