from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.data.full_join_sampler import (
    FullJoinBatch,
    NeuroCardFullJoinSampleSource,
    SyntheticFullJoinSampleSource,
)
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
)


def sample_source_from_config(config: dict[str, Any]) -> object:
    """Construct a sample source without coupling trainers to concrete classes."""

    dataset = config.get("dataset", {})
    dataset_type = dataset.get("type", "synthetic_full_join")
    if dataset_type == "synthetic_full_join":
        source = SyntheticFullJoinSampleSource()
    elif dataset_type == "neurocard_full_join":
        source = NeuroCardFullJoinSampleSource(Path(dataset["prepared_directory"]))
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
        )
