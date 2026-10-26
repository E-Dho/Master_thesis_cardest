from __future__ import annotations

from pathlib import Path

from .base import Adapter
from .external import ExternalCommandAdapter
from .foj_sampling import FojSamplingAdapter
from .learned import (
    DeepDbAdapter,
    DistJoinAdapter,
    MscnAdapter,
    NeuroCardAdapter,
    OwnModelAdapter,
)
from .postgres import PostgresAdapter
from ..config import ExperimentConfig


def create_adapter(config: ExperimentConfig, seed: int, run_directory: Path) -> Adapter:
    adapters: dict[str, type[Adapter]] = {
        "postgres": PostgresAdapter,
        "foj_sampling": FojSamplingAdapter,
        "external": ExternalCommandAdapter,
        "mscn": MscnAdapter,
        "deepdb": DeepDbAdapter,
        "neurocard": NeuroCardAdapter,
        "distjoin": DistJoinAdapter,
        "own_model": OwnModelAdapter,
    }
    try:
        adapter_type = adapters[config.adapter_type]
    except KeyError as exc:
        raise ValueError(f"unknown adapter type {config.adapter_type!r}") from exc
    return adapter_type(config, seed, run_directory)
