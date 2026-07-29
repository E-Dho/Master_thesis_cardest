from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.data.full_join_sampler import (
    NeuroCardFullJoinSampleSource,
    SyntheticFullJoinSampleSource,
)


def sample_source_from_config(config: dict[str, Any]) -> object:
    """Construct a sample source without coupling trainers to concrete classes."""

    dataset = config.get("dataset", {})
    dataset_type = dataset.get("type", "synthetic_full_join")
    if dataset_type == "synthetic_full_join":
        return SyntheticFullJoinSampleSource()
    if dataset_type == "neurocard_full_join":
        return NeuroCardFullJoinSampleSource(Path(dataset["prepared_directory"]))
    raise ValueError(f"unsupported dataset.type {dataset_type!r}")

