from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.data.full_join_sampler import (
    FullJoinBatch,
    LiveNeuroCardFullJoinSampleSource,
    NeuroCardFullJoinSampleSource,
    SyntheticFullJoinSampleSource,
)
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
            )
    else:
        raise ValueError(f"unsupported dataset.type {dataset_type!r}")
    factorization = FactorizationConfig.from_dict(config.get("factorization", {}))
    if not factorization.enabled:
        return source
    return FactorizedMetadataSampleSource(source, factorization)


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
            fresh_rows_drawn=batch.fresh_rows_drawn,
            fixture_rows_reused=batch.fixture_rows_reused,
        )

    @property
    def sampler_run_calls(self) -> int | None:
        return getattr(self.base_source, "sampler_run_calls", None)

    @property
    def distinct_original_rows_seen_estimate(self) -> object:
        return getattr(self.base_source, "distinct_original_rows_seen_estimate", None)

    def discard_buffer(self) -> None:
        discard = getattr(self.base_source, "discard_buffer", None)
        if discard is not None:
            discard()
