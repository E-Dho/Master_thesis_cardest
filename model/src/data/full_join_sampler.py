from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata

OUTER_MISSING = "__OUTER_MISSING__"


@dataclass(frozen=True)
class SyntheticDataset:
    """Tiny three-table chain with duplicates, unmatched rows, and fanout skew."""

    metadata: ModelMetadata
    decoded_rows: tuple[tuple[object, ...], ...]
    encoded_rows: np.ndarray


@dataclass(frozen=True)
class FullJoinBatch:
    """One batch from a full-outer-join sample source in canonical ordering."""

    encoded_values: np.ndarray
    column_metadata: tuple[ColumnMetadata, ...]
    raw_values: tuple[tuple[object, ...], ...] | None = None


@dataclass(frozen=True)
class SamplerInspection:
    join_cardinality: int
    column_order: tuple[str, ...]
    column_types: tuple[str, ...]
    domain_sizes: tuple[int, ...]
    indicator_frequencies: dict[str, float]
    fanout_domains: dict[str, tuple[object, ...]]
    fanout_min_max: dict[str, tuple[float, float]]
    padded_percentages: dict[str, float]
    decoded_sample_rows: tuple[tuple[object, ...], ...]


class FullJoinSampler(Protocol):
    """Interface expected from a NeuroCard-style uniform full-join sampler."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        ...


class FullJoinSampleSource(Protocol):
    @property
    def join_cardinality(self) -> int:
        ...

    @property
    def metadata(self) -> ModelMetadata:
        ...

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        ...


class SyntheticFullJoinSampleSource:
    """Deterministic sample source over the materialized synthetic oracle rows."""

    def __init__(self, dataset: SyntheticDataset | None = None) -> None:
        self.dataset = dataset or build_synthetic_chain_dataset()

    @property
    def join_cardinality(self) -> int:
        return int(self.dataset.metadata.full_join_cardinality)

    @property
    def metadata(self) -> ModelMetadata:
        return self.dataset.metadata

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.dataset.encoded_rows), size=batch_size)
        raw_rows = tuple(self.dataset.decoded_rows[index] for index in indices)
        return FullJoinBatch(
            encoded_values=self.dataset.encoded_rows[indices],
            column_metadata=self.dataset.metadata.columns,
            raw_values=raw_rows,
        )

    def inspect(self, *, sample_rows: int = 5) -> SamplerInspection:
        return inspect_encoded_rows(self.dataset.metadata, self.dataset.encoded_rows, self.dataset.decoded_rows[:sample_rows])


class NeuroCardFullJoinSamplerAdapter:
    """Boundary for wiring NeuroCard's sampler without vendoring it here."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        raise NotImplementedError(
            "Production NeuroCard sampler integration is the next milestone. "
            "This milestone validates the tuple contract with a synthetic oracle."
        )


class NeuroCardFullJoinSampleSource:
    """Manifest-backed adapter boundary for NeuroCard Exact Weight sampling."""

    def __init__(self, prepared_directory: str | Path) -> None:
        self.prepared_directory = Path(prepared_directory)
        manifest_path = self.prepared_directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"missing NeuroCard preparation manifest: {manifest_path}. "
                "Run python3 -m model.scripts.prepare_neurocard_data --config <config>."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from model.src.data.complete_domain_preparation import validate_prepared_manifest

        validate_prepared_manifest(self.prepared_directory)
        self._metadata = ModelMetadata.from_json_dict(self.manifest["metadata"])

    @property
    def join_cardinality(self) -> int:
        return int(self.manifest["join_cardinality"])

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        sample_path = self.prepared_directory / "sample_rows.npy"
        if sample_path.exists():
            rows = np.load(sample_path)
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(rows), size=batch_size)
            return FullJoinBatch(rows[indices], self.metadata.columns)
        raise NotImplementedError(
            "No local sample_rows.npy fixture is available. Wire NeuroCard's "
            "FactorizedSampler/FactorizedSamplerIterDataset here after its Rust "
            "join-count/index artifacts exist in the prepared directory."
        )


def _encode_rows(metadata: ModelMetadata, rows: tuple[tuple[object, ...], ...]) -> np.ndarray:
    encoded = np.zeros((len(rows), len(metadata.columns)), dtype=int)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            encoded[row_index, column_index] = metadata.columns[column_index].encode_value(value)
    return encoded


def canonicalize_fanout_value(value: object, *, outer_padding: bool = False) -> int:
    """Convert known neutral outer padding to fanout 1 and reject malformed values."""

    if value is None and outer_padding:
        return 1
    fanout_value = int(value)
    if fanout_value <= 0:
        raise ValueError(f"fanout values must be strictly positive, got {value!r}")
    return fanout_value


def build_synthetic_chain_dataset() -> SyntheticDataset:
    """Materialize a small full outer join for exact validation only.

    Column order is data, indicators, then fanouts. Fanout columns are positive
    effective fanouts, including value 1 for unmatched outer-join branches.
    """

    columns = (
        ColumnMetadata("A.value", ColumnKind.DATA, ("a1", "a2", OUTER_MISSING), table="A"),
        ColumnMetadata("B.value", ColumnKind.DATA, ("b1", "b2", "b3", OUTER_MISSING), table="B"),
        ColumnMetadata("C.value", ColumnKind.DATA, ("c1", "c2", OUTER_MISSING), table="C"),
        ColumnMetadata("I_A", ColumnKind.INDICATOR, (0, 1), table="A"),
        ColumnMetadata("I_B", ColumnKind.INDICATOR, (0, 1), table="B"),
        ColumnMetadata("I_C", ColumnKind.INDICATOR, (0, 1), table="C"),
        ColumnMetadata("F_A_to_B", ColumnKind.FANOUT, (1, 2), table="A", fanout_source="A->B"),
        ColumnMetadata("F_B_to_C", ColumnKind.FANOUT, (1, 2, 10), table="B", fanout_source="B->C"),
    )
    rows = (
        ("a1", "b1", "c1", 1, 1, 1, 2, 10),
        ("a1", "b1", "c2", 1, 1, 1, 2, 10),
        ("a1", "b2", "c1", 1, 1, 1, 2, 1),
        ("a2", OUTER_MISSING, OUTER_MISSING, 1, 0, 0, 1, 1),
        (OUTER_MISSING, "b3", OUTER_MISSING, 0, 1, 0, 1, 1),
    )
    metadata = ModelMetadata(
        columns=columns,
        full_join_cardinality=float(len(rows)),
        upstream_attribution={
            "NeuroCard": "full-outer-join tuples, indicators, and fanout semantics",
            "Duet": "predicate-conditioned virtual-token inference",
            "DistJoin": "future factorized output adapter boundary only",
        },
    )
    return SyntheticDataset(metadata, rows, _encode_rows(metadata, rows))


def inspect_encoded_rows(
    metadata: ModelMetadata,
    encoded_rows: np.ndarray,
    decoded_sample_rows: tuple[tuple[object, ...], ...] = (),
) -> SamplerInspection:
    """Summarize ordering, indicators, fanouts, padding, and decoded examples."""

    indicator_frequencies = {}
    padded_percentages = {}
    fanout_domains = {}
    fanout_min_max = {}
    for column_index, column in enumerate(metadata.columns):
        values = encoded_rows[:, column_index]
        decoded_values = np.array([column.domain[index] for index in values], dtype=object)
        if column.kind == ColumnKind.INDICATOR:
            present = np.mean(decoded_values.astype(float))
            indicator_frequencies[column.name] = float(present)
            if column.table is not None:
                padded_percentages[column.table] = float(100.0 * (1.0 - present))
        elif column.kind == ColumnKind.FANOUT:
            numeric = decoded_values.astype(float)
            fanout_domains[column.name] = column.domain
            fanout_min_max[column.name] = (float(np.min(numeric)), float(np.max(numeric)))
    return SamplerInspection(
        join_cardinality=int(metadata.full_join_cardinality),
        column_order=tuple(column.name for column in metadata.columns),
        column_types=tuple(column.kind.value for column in metadata.columns),
        domain_sizes=tuple(column.domain_size for column in metadata.columns),
        indicator_frequencies=indicator_frequencies,
        fanout_domains=fanout_domains,
        fanout_min_max=fanout_min_max,
        padded_percentages=padded_percentages,
        decoded_sample_rows=decoded_sample_rows,
    )
