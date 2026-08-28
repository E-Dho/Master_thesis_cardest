from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.generation import GeneratedTrainingContext, context_satisfies_row
from model.src.predicates.operators import PredicateOp


TRAJECTORY_TARGET_SEMANTICS_VERSION = "single_anchor_query_only_v1"


class UnsupportedTrajectoryContext(ValueError):
    """Raised when a context cannot be interpreted as a segment-selection query."""


class TrajectoryMultiplicityProvider(Protocol):
    """Boundary for exact local trajectory multiplicities used by training."""

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class TrajectorySegmentIndex:
    """CSR-style encoded segment rows grouped by logical trajectory id."""

    metadata: ModelMetadata
    trajectory_ids: tuple[object, ...]
    offsets: np.ndarray
    encoded_segment_rows: np.ndarray
    predicate_columns: tuple[str, ...]
    trajectory_key: str
    compatibility_hash: str

    @classmethod
    def from_rows(
        cls,
        *,
        metadata: ModelMetadata,
        trajectory_ids: Sequence[object],
        encoded_rows: np.ndarray,
        predicate_columns: Sequence[str],
        trajectory_key: str,
    ) -> "TrajectorySegmentIndex":
        encoded = np.asarray(encoded_rows, dtype=np.int64)
        if encoded.ndim != 2 or encoded.shape[1] != len(metadata.columns):
            raise ValueError("encoded_rows must have shape [rows, metadata columns]")
        if len(trajectory_ids) != encoded.shape[0]:
            raise ValueError("trajectory_ids must align with encoded_rows")
        grouped: dict[object, list[int]] = {}
        for row_index, trajectory_id in enumerate(trajectory_ids):
            grouped.setdefault(trajectory_id, []).append(row_index)
        ordered_ids = tuple(grouped)
        offsets = [0]
        pieces = []
        for trajectory_id in ordered_ids:
            indices = grouped[trajectory_id]
            pieces.append(encoded[np.asarray(indices, dtype=int)])
            offsets.append(offsets[-1] + len(indices))
        compact_rows = (
            np.concatenate(pieces, axis=0)
            if pieces
            else np.empty((0, len(metadata.columns)), dtype=np.int64)
        )
        return cls(
            metadata=metadata,
            trajectory_ids=ordered_ids,
            offsets=np.asarray(offsets, dtype=np.int64),
            encoded_segment_rows=compact_rows,
            predicate_columns=tuple(predicate_columns),
            trajectory_key=str(trajectory_key),
            compatibility_hash=trajectory_index_compatibility_hash(
                metadata,
                predicate_columns=predicate_columns,
                trajectory_key=trajectory_key,
            ),
        )

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        if len(anchor_trajectory_ids) != len(contexts):
            raise ValueError("anchor_trajectory_ids and contexts must have equal length")
        id_to_position = {trajectory_id: index for index, trajectory_id in enumerate(self.trajectory_ids)}
        values = np.zeros(len(contexts), dtype=np.int64)
        for row_index, (trajectory_id, context) in enumerate(zip(anchor_trajectory_ids, contexts)):
            position = id_to_position.get(trajectory_id)
            if position is None:
                raise KeyError(f"unknown trajectory id {trajectory_id!r}")
            self._validate_context_supported(context)
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            rows = self.encoded_segment_rows[start:stop]
            multiplicity = 0
            for candidate in rows:
                if context_satisfies_row(context, candidate, self.metadata):
                    multiplicity += 1
            if multiplicity <= 0:
                raise ValueError(
                    "row-satisfied trajectory context produced m_t(Q)=0; "
                    "anchor segment was not counted"
                )
            values[row_index] = multiplicity
        return values

    def _validate_context_supported(self, context: GeneratedTrainingContext) -> None:
        allowed = set(self.predicate_columns)
        for column, token in zip(self.metadata.columns, context.tokens):
            if column.kind != ColumnKind.DATA:
                continue
            if token.op == PredicateOp.WILDCARD:
                continue
            if column.name not in allowed:
                raise UnsupportedTrajectoryContext(
                    f"non-wildcard predicate on unsupported trajectory column "
                    f"{column.name!r}"
                )

    def local_targets(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        multiplicities = self.matching_multiplicities(
            anchor_trajectory_ids=anchor_trajectory_ids,
            contexts=contexts,
        )
        return 1.0 / multiplicities.astype(float)

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadata.to_json_dict(), sort_keys=True)
        np.savez_compressed(
            path,
            metadata_json=np.asarray(metadata_json),
            trajectory_ids=np.asarray(self.trajectory_ids, dtype=object),
            offsets=self.offsets,
            encoded_segment_rows=self.encoded_segment_rows,
            predicate_columns=np.asarray(self.predicate_columns, dtype=object),
            trajectory_key=np.asarray(self.trajectory_key),
            compatibility_hash=np.asarray(self.compatibility_hash),
            target_semantics_version=np.asarray(TRAJECTORY_TARGET_SEMANTICS_VERSION),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrajectorySegmentIndex":
        payload = np.load(path, allow_pickle=True)
        metadata = ModelMetadata.from_json_dict(json.loads(str(payload["metadata_json"])))
        predicate_columns = tuple(str(value) for value in payload["predicate_columns"].tolist())
        trajectory_key = str(payload["trajectory_key"])
        expected_hash = trajectory_index_compatibility_hash(
            metadata,
            predicate_columns=predicate_columns,
            trajectory_key=trajectory_key,
        )
        stored_hash = str(payload["compatibility_hash"])
        if stored_hash != expected_hash:
            raise ValueError(
                "trajectory index compatibility hash does not match metadata/config"
            )
        return cls(
            metadata=metadata,
            trajectory_ids=tuple(payload["trajectory_ids"].tolist()),
            offsets=np.asarray(payload["offsets"], dtype=np.int64),
            encoded_segment_rows=np.asarray(payload["encoded_segment_rows"], dtype=np.int64),
            predicate_columns=predicate_columns,
            trajectory_key=trajectory_key,
            compatibility_hash=stored_hash,
        )


def trajectory_index_compatibility_hash(
    metadata: ModelMetadata,
    *,
    predicate_columns: Sequence[str],
    trajectory_key: str,
) -> str:
    payload = {
        "schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "predicate_columns": tuple(predicate_columns),
        "trajectory_key": trajectory_key,
        "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

