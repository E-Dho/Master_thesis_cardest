from __future__ import annotations

from dataclasses import dataclass
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


class FullJoinSampler(Protocol):
    """Interface expected from a NeuroCard-style uniform full-join sampler."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        ...


class NeuroCardFullJoinSamplerAdapter:
    """Boundary for wiring NeuroCard's sampler without vendoring it here."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        raise NotImplementedError(
            "Production NeuroCard sampler integration is the next milestone. "
            "This milestone validates the tuple contract with a synthetic oracle."
        )


def _encode_rows(metadata: ModelMetadata, rows: tuple[tuple[object, ...], ...]) -> np.ndarray:
    encoded = np.zeros((len(rows), len(metadata.columns)), dtype=int)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            encoded[row_index, column_index] = metadata.columns[column_index].encode_value(value)
    return encoded


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
        ColumnMetadata("F_A_to_B", ColumnKind.FANOUT, (1, 2), table="A"),
        ColumnMetadata("F_B_to_C", ColumnKind.FANOUT, (1, 2, 10), table="B"),
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
