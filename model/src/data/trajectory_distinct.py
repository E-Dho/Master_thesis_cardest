from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.generation import GeneratedTrainingContext
from model.src.predicates.operators import PredicateOp, PredicateToken


TRAJECTORY_TARGET_SEMANTICS_VERSION = "single_anchor_query_only_v2"
TEMPORAL_OVERLAP_SEMANTICS_VERSION = "pol_half_open_overlap_v1"
SPATIAL_INTERSECTS_SEMANTICS_VERSION = "line_segment_aabb_intersects_v1"
TRAJECTORY_INDEX_FORMAT_VERSION = 2


class UnsupportedTrajectoryContext(ValueError):
    """Raised when a context cannot be interpreted as a segment-selection query."""


@dataclass(frozen=True)
class TrajectoryDistinctNotApplicable(ValueError):
    """Raised when a query is not a matching-segment measure."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class TrajectoryDistinctEligibility:
    eligible: bool
    reason: str | None = None


@dataclass(frozen=True)
class TrajectoryDistinctRuntimeConfig:
    entity_table: str = "trips"
    segment_table: str = "segments"
    trajectory_key: str = "trip_id"
    segment_key: str = "trip_id,segment_idx"
    predicate_scope: str = "segment_query"
    trajectory_static_columns: tuple[str, ...] = ()
    segment_varying_columns: tuple[str, ...] = ()
    srid: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrajectoryDistinctRuntimeConfig":
        raw = dict(data or {})
        return cls(
            entity_table=str(raw.get("entity_table", "trips")),
            segment_table=str(raw.get("segment_table", "segments")),
            trajectory_key=str(raw.get("trajectory_key", "trip_id")),
            segment_key=str(raw.get("segment_key", "trip_id,segment_idx")),
            predicate_scope=str(raw.get("predicate_scope", "segment_query")),
            trajectory_static_columns=tuple(
                str(value) for value in raw.get("trajectory_static_columns", ())
            ),
            segment_varying_columns=tuple(
                str(value) for value in raw.get("segment_varying_columns", ())
            ),
            srid=(None if raw.get("srid") is None else int(raw.get("srid"))),
        )


@dataclass(frozen=True)
class SegmentMeasureCapability:
    """Physical segment-query semantics represented by the base estimator.

    The current one-pass AR factor product can represent scalar column tokens
    and POL temporal overlap via ``t_s < upper`` and ``t_e >= lower``. It does
    not represent physical line/rectangle ``ST_Intersects``; endpoint-coordinate
    range tokens are only endpoint conditioning and are not equivalent.
    """

    supports_scalar: bool = True
    supports_temporal_overlap: bool = True
    supports_spatial_intersects: bool = False


ONE_PASS_AR_SEGMENT_MEASURE_CAPABILITY = SegmentMeasureCapability(
    supports_scalar=True,
    supports_temporal_overlap=True,
    supports_spatial_intersects=False,
)


@dataclass(frozen=True)
class SegmentTemporalPredicate:
    start_column: str
    end_column: str
    lower: Any | None = None
    upper: Any | None = None
    semantics: Literal["overlap"] = "overlap"


@dataclass(frozen=True)
class SegmentSpatialPredicate:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    srid: int | None = None
    start_x_column: str = "segments:s_x"
    start_y_column: str = "segments:s_y"
    end_x_column: str = "segments:e_x"
    end_y_column: str = "segments:e_y"
    semantics: Literal["intersects"] = "intersects"


@dataclass(frozen=True)
class TrajectoryQuerySemantics:
    scalar_predicates: tuple[tuple[str, PredicateToken], ...] = ()
    temporal_predicates: tuple[SegmentTemporalPredicate, ...] = ()
    spatial_predicates: tuple[SegmentSpatialPredicate, ...] = ()

    @property
    def query_type(self) -> str:
        has_temporal = bool(self.temporal_predicates)
        has_spatial = bool(self.spatial_predicates)
        if has_temporal and has_spatial:
            return "spatio_temporal"
        if has_temporal:
            return "temporal"
        if has_spatial:
            return "spatial"
        return "scalar_only"


@dataclass(frozen=True)
class TrajectoryMultiplicityBatch:
    multiplicities: np.ndarray
    eligible_mask: np.ndarray
    skip_reasons: tuple[str | None, ...]
    lookup_seconds: float = 0.0
    predicate_eval_seconds: float = 0.0
    provider_seconds: float = 0.0
    segments_scanned: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


class TrajectoryMultiplicityProvider(Protocol):
    """Boundary for exact local trajectory multiplicities used by training."""

    def evaluate_batch(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> TrajectoryMultiplicityBatch:
        ...

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        ...


def semantic_owned_columns(query: TrajectoryQuerySemantics | None) -> frozenset[str]:
    if query is None:
        return frozenset()
    owned: set[str] = set()
    for temporal in query.temporal_predicates:
        owned.add(temporal.start_column)
        owned.add(temporal.end_column)
    for spatial in query.spatial_predicates:
        owned.update(
            (
                spatial.start_x_column,
                spatial.start_y_column,
                spatial.end_x_column,
                spatial.end_y_column,
            )
        )
    return frozenset(owned)


def context_satisfies_row_with_trajectory_semantics(
    context: GeneratedTrainingContext,
    encoded_row: np.ndarray,
    metadata: ModelMetadata,
) -> bool:
    """Evaluate one row against Duet tokens plus physical trajectory semantics."""

    semantic_query = getattr(context, "trajectory_query", None)
    owned = semantic_owned_columns(semantic_query)
    for column_index, (column, token) in enumerate(zip(metadata.columns, context.tokens)):
        value = column.domain[int(encoded_row[column_index])]
        if column.kind == ColumnKind.DATA:
            if column.name in owned:
                continue
            if not token.satisfies(value):
                return False
        if column.kind == ColumnKind.INDICATOR and token.op == PredicateOp.EQUAL and token.value != value:
            return False
    present_tables = {
        column.table
        for column_index, column in enumerate(metadata.columns)
        if column.kind == ColumnKind.INDICATOR
        and column.table is not None
        and column.domain[int(encoded_row[column_index])] == 1
    }
    if present_tables and not set(context.included_tables).issubset(present_tables):
        return False
    if semantic_query is None:
        return True
    if semantic_query.scalar_predicates:
        by_name = {column.name: index for index, column in enumerate(metadata.columns)}
        for column_name, token in semantic_query.scalar_predicates:
            if column_name in owned:
                continue
            column_index = by_name[column_name]
            column = metadata.columns[column_index]
            value = column.domain[int(encoded_row[column_index])]
            if not token.satisfies(value):
                return False
    for temporal in semantic_query.temporal_predicates:
        if not bool(
            temporal_overlap_mask(
                np.asarray([_decoded_row_value(metadata, encoded_row, temporal.start_column)], dtype=object),
                np.asarray([_decoded_row_value(metadata, encoded_row, temporal.end_column)], dtype=object),
                lower=temporal.lower,
                upper=temporal.upper,
            )[0]
        ):
            return False
    for spatial in semantic_query.spatial_predicates:
        if not bool(
            segment_rectangle_intersects_mask(
                np.asarray([_decoded_row_value(metadata, encoded_row, spatial.start_x_column)], dtype=float),
                np.asarray([_decoded_row_value(metadata, encoded_row, spatial.start_y_column)], dtype=float),
                np.asarray([_decoded_row_value(metadata, encoded_row, spatial.end_x_column)], dtype=float),
                np.asarray([_decoded_row_value(metadata, encoded_row, spatial.end_y_column)], dtype=float),
                spatial.min_x,
                spatial.min_y,
                spatial.max_x,
                spatial.max_y,
            )[0]
        ):
            return False
    return True


def _decoded_row_value(metadata: ModelMetadata, encoded_row: np.ndarray, column_name: str) -> Any:
    column_index = metadata.column_index(column_name)
    column = metadata.columns[column_index]
    return column.domain[int(encoded_row[column_index])]


def trajectory_distinct_context_eligibility(
    context: GeneratedTrainingContext,
    metadata: ModelMetadata,
    config: TrajectoryDistinctRuntimeConfig,
) -> TrajectoryDistinctEligibility:
    """Return whether the ordinary estimator output is a segment cardinality."""

    del metadata
    if config.predicate_scope != "segment_query":
        return TrajectoryDistinctEligibility(False, "unsupported_predicate_scope")
    if config.segment_table not in context.included_tables:
        return TrajectoryDistinctEligibility(False, "non_segment_measure")
    return TrajectoryDistinctEligibility(True, None)


def trajectory_base_measure_support(
    context: GeneratedTrainingContext,
    capability: SegmentMeasureCapability = ONE_PASS_AR_SEGMENT_MEASURE_CAPABILITY,
) -> TrajectoryDistinctEligibility:
    """Return whether the base segment estimator models the same event as Q."""

    query = getattr(context, "trajectory_query", None)
    if query is None:
        return (
            TrajectoryDistinctEligibility(True, None)
            if capability.supports_scalar
            else TrajectoryDistinctEligibility(False, "unsupported_base_segment_scalar_measure")
        )
    if getattr(query, "spatial_predicates", ()) and not capability.supports_spatial_intersects:
        return TrajectoryDistinctEligibility(False, "unsupported_base_segment_spatial_measure")
    if getattr(query, "temporal_predicates", ()) and not capability.supports_temporal_overlap:
        return TrajectoryDistinctEligibility(False, "unsupported_base_segment_temporal_measure")
    if getattr(query, "scalar_predicates", ()) and not capability.supports_scalar:
        return TrajectoryDistinctEligibility(False, "unsupported_base_segment_scalar_measure")
    return TrajectoryDistinctEligibility(True, None)


@dataclass(frozen=True)
class TrajectorySegmentIndex:
    """CSR-style segment rows grouped by logical trajectory id.

    The legacy NPZ representation stores encoded rows for compact unit tests.
    Production POL indexes should use the directory format emitted by the
    preparation script and memory-map the compact arrays.
    """

    metadata: ModelMetadata
    trajectory_ids: tuple[object, ...]
    offsets: np.ndarray
    encoded_segment_rows: np.ndarray
    predicate_columns: tuple[str, ...]
    trajectory_key: str
    compatibility_hash: str
    entity_table: str = "trips"
    segment_table: str = "segments"
    segment_key: str = "trip_id,segment_idx"
    trajectory_static_columns: tuple[str, ...] = ()
    segment_varying_columns: tuple[str, ...] = ()
    srid: int | None = None
    format_version: int = TRAJECTORY_INDEX_FORMAT_VERSION
    _trajectory_id_to_position: dict[object, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_trajectory_id_to_position",
            {trajectory_id: index for index, trajectory_id in enumerate(self.trajectory_ids)},
        )
        _validate_columns_exist(
            self.metadata,
            self.trajectory_static_columns + self.segment_varying_columns,
        )

    @classmethod
    def from_rows(
        cls,
        *,
        metadata: ModelMetadata,
        trajectory_ids: Sequence[object],
        encoded_rows: np.ndarray,
        predicate_columns: Sequence[str] = (),
        trajectory_key: str,
        entity_table: str = "trips",
        segment_table: str = "segments",
        segment_key: str = "trip_id,segment_idx",
        trajectory_static_columns: Sequence[str] = (),
        segment_varying_columns: Sequence[str] = (),
        srid: int | None = None,
    ) -> "TrajectorySegmentIndex":
        encoded = np.asarray(encoded_rows, dtype=np.int64)
        if encoded.ndim != 2 or encoded.shape[1] != len(metadata.columns):
            raise ValueError("encoded_rows must have shape [rows, metadata columns]")
        if len(trajectory_ids) != encoded.shape[0]:
            raise ValueError("trajectory_ids must align with encoded_rows")
        if not segment_varying_columns:
            segment_varying_columns = tuple(predicate_columns)
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
        compatibility_hash = trajectory_index_compatibility_hash(
            metadata,
            predicate_columns=predicate_columns,
            trajectory_key=trajectory_key,
            entity_table=entity_table,
            segment_table=segment_table,
            segment_key=segment_key,
            trajectory_static_columns=trajectory_static_columns,
            segment_varying_columns=segment_varying_columns,
            srid=srid,
            format_version=TRAJECTORY_INDEX_FORMAT_VERSION,
        )
        return cls(
            metadata=metadata,
            trajectory_ids=ordered_ids,
            offsets=np.asarray(offsets, dtype=np.int64),
            encoded_segment_rows=compact_rows,
            predicate_columns=tuple(predicate_columns),
            trajectory_key=str(trajectory_key),
            compatibility_hash=compatibility_hash,
            entity_table=str(entity_table),
            segment_table=str(segment_table),
            segment_key=str(segment_key),
            trajectory_static_columns=tuple(trajectory_static_columns),
            segment_varying_columns=tuple(segment_varying_columns),
            srid=srid,
        )

    @property
    def runtime_config(self) -> TrajectoryDistinctRuntimeConfig:
        return TrajectoryDistinctRuntimeConfig(
            entity_table=self.entity_table,
            segment_table=self.segment_table,
            trajectory_key=self.trajectory_key,
            segment_key=self.segment_key,
            trajectory_static_columns=self.trajectory_static_columns,
            segment_varying_columns=self.segment_varying_columns,
            srid=self.srid,
        )

    def validate_runtime_compatibility(
        self,
        runtime_config: TrajectoryDistinctRuntimeConfig,
        metadata: ModelMetadata,
    ) -> None:
        validate_trajectory_index_runtime_compatibility(
            index=self,
            runtime_config=runtime_config,
            metadata=metadata,
        )

    def evaluate_batch(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> TrajectoryMultiplicityBatch:
        started = time.perf_counter()
        if len(anchor_trajectory_ids) != len(contexts):
            raise ValueError("anchor_trajectory_ids and contexts must have equal length")
        values = np.zeros(len(contexts), dtype=np.int64)
        eligible = np.zeros(len(contexts), dtype=bool)
        skip_reasons: list[str | None] = [None] * len(contexts)
        segments_scanned = np.zeros(len(contexts), dtype=np.int64)
        lookup_seconds = 0.0
        eval_seconds = 0.0
        for row_index, (trajectory_id, context) in enumerate(zip(anchor_trajectory_ids, contexts)):
            eligibility = trajectory_distinct_context_eligibility(
                context,
                self.metadata,
                self.runtime_config,
            )
            if not eligibility.eligible:
                skip_reasons[row_index] = eligibility.reason
                continue
            lookup_started = time.perf_counter()
            position = self._trajectory_id_to_position.get(trajectory_id)
            lookup_seconds += time.perf_counter() - lookup_started
            if position is None:
                skip_reasons[row_index] = "missing_provenance"
                continue
            unsupported = self._unsupported_reason(context)
            if unsupported is not None:
                skip_reasons[row_index] = unsupported
                continue
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            rows = self.encoded_segment_rows[start:stop]
            segments_scanned[row_index] = int(stop - start)
            eval_started = time.perf_counter()
            mask = self._matching_mask(rows, context)
            eval_seconds += time.perf_counter() - eval_started
            multiplicity = int(np.sum(mask))
            if multiplicity <= 0:
                raise ValueError(
                    "row-satisfied trajectory context produced m_t(Q)=0; "
                    "anchor segment was not counted"
                )
            values[row_index] = multiplicity
            eligible[row_index] = True
        return TrajectoryMultiplicityBatch(
            multiplicities=values,
            eligible_mask=eligible,
            skip_reasons=tuple(skip_reasons),
            lookup_seconds=lookup_seconds,
            predicate_eval_seconds=eval_seconds,
            provider_seconds=time.perf_counter() - started,
            segments_scanned=segments_scanned,
        )

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        result = self.evaluate_batch(
            anchor_trajectory_ids=anchor_trajectory_ids,
            contexts=contexts,
        )
        if not np.all(result.eligible_mask):
            reasons = sorted({reason for reason in result.skip_reasons if reason is not None})
            raise UnsupportedTrajectoryContext(
                f"trajectory contexts are not all eligible: {reasons}"
            )
        return result.multiplicities

    def _unsupported_reason(self, context: GeneratedTrainingContext) -> str | None:
        static = set(self.trajectory_static_columns)
        varying = set(self.segment_varying_columns)
        legacy_allowed = set(self.predicate_columns)
        for column, token in zip(self.metadata.columns, context.tokens):
            if column.kind != ColumnKind.DATA or token.op == PredicateOp.WILDCARD:
                continue
            if column.name in static or column.name in varying or column.name in legacy_allowed:
                continue
            return "unsupported_semantics"
        semantic_query = getattr(context, "trajectory_query", None)
        if semantic_query is not None:
            for temporal in getattr(semantic_query, "temporal_predicates", ()):
                for name in (temporal.start_column, temporal.end_column):
                    if name not in varying:
                        return "unsupported_semantics"
            for spatial in getattr(semantic_query, "spatial_predicates", ()):
                for name in (
                    spatial.start_x_column,
                    spatial.start_y_column,
                    spatial.end_x_column,
                    spatial.end_y_column,
                ):
                    if name not in varying:
                        return "unsupported_semantics"
        return None

    def _matching_mask(
        self,
        encoded_rows: np.ndarray,
        context: GeneratedTrainingContext,
    ) -> np.ndarray:
        mask = np.ones(encoded_rows.shape[0], dtype=bool)
        varying = set(self.segment_varying_columns) | set(self.predicate_columns)
        semantic_query = getattr(context, "trajectory_query", None)
        owned = semantic_owned_columns(semantic_query)
        for column_index, (column, token) in enumerate(zip(self.metadata.columns, context.tokens)):
            if (
                column.kind != ColumnKind.DATA
                or token.op == PredicateOp.WILDCARD
                or column.name not in varying
                or column.name in owned
            ):
                continue
            decoded = np.asarray(
                [column.domain[int(value)] for value in encoded_rows[:, column_index]],
                dtype=object,
            )
            mask &= _token_satisfies_array(decoded, token)
        if semantic_query is not None:
            for temporal in semantic_query.temporal_predicates:
                start_values = self._decoded_column(encoded_rows, temporal.start_column)
                end_values = self._decoded_column(encoded_rows, temporal.end_column)
                mask &= temporal_overlap_mask(
                    start_values,
                    end_values,
                    lower=temporal.lower,
                    upper=temporal.upper,
                )
            for spatial in semantic_query.spatial_predicates:
                sx = self._decoded_column(encoded_rows, spatial.start_x_column).astype(float)
                sy = self._decoded_column(encoded_rows, spatial.start_y_column).astype(float)
                ex = self._decoded_column(encoded_rows, spatial.end_x_column).astype(float)
                ey = self._decoded_column(encoded_rows, spatial.end_y_column).astype(float)
                mask &= segment_rectangle_intersects_mask(
                    sx,
                    sy,
                    ex,
                    ey,
                    spatial.min_x,
                    spatial.min_y,
                    spatial.max_x,
                    spatial.max_y,
                )
        return mask

    def _decoded_column(self, encoded_rows: np.ndarray, column_name: str) -> np.ndarray:
        column_index = self.metadata.column_index(column_name)
        column = self.metadata.columns[column_index]
        return np.asarray(
            [column.domain[int(value)] for value in encoded_rows[:, column_index]],
            dtype=object,
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
            entity_table=np.asarray(self.entity_table),
            segment_table=np.asarray(self.segment_table),
            segment_key=np.asarray(self.segment_key),
            trajectory_static_columns=np.asarray(self.trajectory_static_columns, dtype=object),
            segment_varying_columns=np.asarray(self.segment_varying_columns, dtype=object),
            srid=np.asarray(-1 if self.srid is None else self.srid),
            format_version=np.asarray(self.format_version),
            compatibility_hash=np.asarray(self.compatibility_hash),
            target_semantics_version=np.asarray(TRAJECTORY_TARGET_SEMANTICS_VERSION),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrajectorySegmentIndex":
        payload = np.load(path, allow_pickle=True)
        metadata = ModelMetadata.from_json_dict(json.loads(str(payload["metadata_json"])))
        predicate_columns = tuple(str(value) for value in payload["predicate_columns"].tolist())
        trajectory_key = str(payload["trajectory_key"])
        entity_table = str(payload["entity_table"]) if "entity_table" in payload else "trips"
        segment_table = str(payload["segment_table"]) if "segment_table" in payload else "segments"
        segment_key = str(payload["segment_key"]) if "segment_key" in payload else "trip_id,segment_idx"
        trajectory_static_columns = (
            tuple(str(value) for value in payload["trajectory_static_columns"].tolist())
            if "trajectory_static_columns" in payload
            else ()
        )
        segment_varying_columns = (
            tuple(str(value) for value in payload["segment_varying_columns"].tolist())
            if "segment_varying_columns" in payload
            else predicate_columns
        )
        raw_srid = int(payload["srid"]) if "srid" in payload else -1
        srid = None if raw_srid < 0 else raw_srid
        format_version = int(payload["format_version"]) if "format_version" in payload else 1
        expected_hash = trajectory_index_compatibility_hash(
            metadata,
            predicate_columns=predicate_columns,
            trajectory_key=trajectory_key,
            entity_table=entity_table,
            segment_table=segment_table,
            segment_key=segment_key,
            trajectory_static_columns=trajectory_static_columns,
            segment_varying_columns=segment_varying_columns,
            srid=srid,
            format_version=format_version,
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
            entity_table=entity_table,
            segment_table=segment_table,
            segment_key=segment_key,
            trajectory_static_columns=trajectory_static_columns,
            segment_varying_columns=segment_varying_columns,
            srid=srid,
            format_version=format_version,
        )


@dataclass(frozen=True)
class CompactTrajectorySegmentIndex:
    """Memory-mapped POL trajectory index storing only segment-varying fields."""

    metadata: ModelMetadata
    trajectory_ids: np.ndarray
    offsets: np.ndarray
    segment_idx: np.ndarray
    t_s: np.ndarray
    t_e: np.ndarray
    s_x: np.ndarray
    s_y: np.ndarray
    e_x: np.ndarray
    e_y: np.ndarray
    trajectory_key: str
    compatibility_hash: str
    entity_table: str = "trips"
    segment_table: str = "segments"
    segment_key: str = "trip_id,segment_idx"
    trajectory_static_columns: tuple[str, ...] = ()
    segment_varying_columns: tuple[str, ...] = (
        "segments:segment_idx",
        "segments:t_s",
        "segments:t_e",
        "segments:s_x",
        "segments:s_y",
        "segments:e_x",
        "segments:e_y",
    )
    srid: int | None = None
    format_version: int = TRAJECTORY_INDEX_FORMAT_VERSION

    @classmethod
    def from_directory(cls, directory: str | Path) -> "CompactTrajectorySegmentIndex":
        root = Path(directory)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing compact trajectory index manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = ModelMetadata.from_json_dict(manifest["metadata"])
        predicate_columns = tuple(str(value) for value in manifest.get("predicate_columns", ()))
        trajectory_key = str(manifest["trajectory_key"])
        entity_table = str(manifest.get("entity_table", "trips"))
        segment_table = str(manifest.get("segment_table", "segments"))
        segment_key = str(manifest.get("segment_key", "trip_id,segment_idx"))
        trajectory_static_columns = tuple(
            str(value) for value in manifest.get("trajectory_static_columns", ())
        )
        segment_varying_columns = tuple(
            str(value) for value in manifest.get("segment_varying_columns", ())
        )
        srid = manifest.get("srid")
        srid = None if srid is None else int(srid)
        format_version = int(manifest.get("format_version", TRAJECTORY_INDEX_FORMAT_VERSION))
        expected_hash = trajectory_index_compatibility_hash(
            metadata,
            predicate_columns=predicate_columns,
            trajectory_key=trajectory_key,
            entity_table=entity_table,
            segment_table=segment_table,
            segment_key=segment_key,
            trajectory_static_columns=trajectory_static_columns,
            segment_varying_columns=segment_varying_columns,
            srid=srid,
            format_version=format_version,
        )
        stored_hash = str(manifest["compatibility_hash"])
        if stored_hash != expected_hash:
            raise ValueError(
                "compact trajectory index compatibility hash does not match metadata/config"
            )
        _validate_columns_exist(
            metadata,
            trajectory_static_columns + segment_varying_columns,
        )
        arrays = manifest.get("arrays", {})
        return cls(
            metadata=metadata,
            trajectory_ids=np.load(root / arrays.get("trajectory_ids", "trajectory_ids.npy"), mmap_mode="r"),
            offsets=np.load(root / arrays.get("offsets", "offsets.npy"), mmap_mode="r"),
            segment_idx=np.load(root / arrays.get("segment_idx", "segment_idx.npy"), mmap_mode="r"),
            t_s=np.load(root / arrays.get("t_s", "t_s.npy"), mmap_mode="r"),
            t_e=np.load(root / arrays.get("t_e", "t_e.npy"), mmap_mode="r"),
            s_x=np.load(root / arrays.get("s_x", "s_x.npy"), mmap_mode="r"),
            s_y=np.load(root / arrays.get("s_y", "s_y.npy"), mmap_mode="r"),
            e_x=np.load(root / arrays.get("e_x", "e_x.npy"), mmap_mode="r"),
            e_y=np.load(root / arrays.get("e_y", "e_y.npy"), mmap_mode="r"),
            trajectory_key=trajectory_key,
            compatibility_hash=stored_hash,
            entity_table=entity_table,
            segment_table=segment_table,
            segment_key=segment_key,
            trajectory_static_columns=trajectory_static_columns,
            segment_varying_columns=segment_varying_columns,
            srid=srid,
            format_version=format_version,
        )

    @property
    def runtime_config(self) -> TrajectoryDistinctRuntimeConfig:
        return TrajectoryDistinctRuntimeConfig(
            entity_table=self.entity_table,
            segment_table=self.segment_table,
            trajectory_key=self.trajectory_key,
            segment_key=self.segment_key,
            trajectory_static_columns=self.trajectory_static_columns,
            segment_varying_columns=self.segment_varying_columns,
            srid=self.srid,
        )

    def validate_runtime_compatibility(
        self,
        runtime_config: TrajectoryDistinctRuntimeConfig,
        metadata: ModelMetadata,
    ) -> None:
        validate_trajectory_index_runtime_compatibility(
            index=self,
            runtime_config=runtime_config,
            metadata=metadata,
        )

    @property
    def segment_count(self) -> int:
        return int(self.segment_idx.shape[0])

    @property
    def index_bytes(self) -> int:
        return int(
            sum(
                int(array.size) * int(array.dtype.itemsize)
                for array in (
                    self.trajectory_ids,
                    self.offsets,
                    self.segment_idx,
                    self.t_s,
                    self.t_e,
                    self.s_x,
                    self.s_y,
                    self.e_x,
                    self.e_y,
                )
            )
        )

    def storage_summary(self) -> dict[str, Any]:
        bytes_per_segment = self.index_bytes / max(1, self.segment_count)
        bytes_per_trajectory = self.index_bytes / max(1, int(self.trajectory_ids.shape[0]))
        return {
            "format_version": int(self.format_version),
            "trajectory_count": int(self.trajectory_ids.shape[0]),
            "segment_count": self.segment_count,
            "index_bytes": self.index_bytes,
            "bytes_per_segment": float(bytes_per_segment),
            "bytes_per_trajectory": float(bytes_per_trajectory),
            "estimated_bytes_1m_segments": int(round(bytes_per_segment * 1_000_000)),
            "estimated_bytes_10m_segments": int(round(bytes_per_segment * 10_000_000)),
            "estimated_bytes_50m_segments": int(round(bytes_per_segment * 50_000_000)),
        }

    def evaluate_batch(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> TrajectoryMultiplicityBatch:
        started = time.perf_counter()
        if len(anchor_trajectory_ids) != len(contexts):
            raise ValueError("anchor_trajectory_ids and contexts must have equal length")
        values = np.zeros(len(contexts), dtype=np.int64)
        eligible = np.zeros(len(contexts), dtype=bool)
        skip_reasons: list[str | None] = [None] * len(contexts)
        scanned = np.zeros(len(contexts), dtype=np.int64)
        lookup_seconds = 0.0
        eval_seconds = 0.0
        trajectory_ids = np.asarray(self.trajectory_ids)
        for row_index, (trajectory_id, context) in enumerate(zip(anchor_trajectory_ids, contexts)):
            context_ok = trajectory_distinct_context_eligibility(
                context,
                self.metadata,
                self.runtime_config,
            )
            if not context_ok.eligible:
                skip_reasons[row_index] = context_ok.reason
                continue
            unsupported = self._unsupported_reason(context)
            if unsupported is not None:
                skip_reasons[row_index] = unsupported
                continue
            lookup_started = time.perf_counter()
            try:
                lookup_value = int(trajectory_id)
            except (TypeError, ValueError):
                skip_reasons[row_index] = "missing_provenance"
                continue
            position = int(np.searchsorted(trajectory_ids, lookup_value))
            lookup_seconds += time.perf_counter() - lookup_started
            if position >= len(trajectory_ids) or int(trajectory_ids[position]) != lookup_value:
                skip_reasons[row_index] = "missing_provenance"
                continue
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            scanned[row_index] = stop - start
            eval_started = time.perf_counter()
            mask = self._matching_mask_slice(start, stop, context)
            eval_seconds += time.perf_counter() - eval_started
            multiplicity = int(np.sum(mask))
            if multiplicity <= 0:
                raise ValueError(
                    "row-satisfied trajectory context produced m_t(Q)=0; "
                    "anchor segment was not counted"
                )
            values[row_index] = multiplicity
            eligible[row_index] = True
        return TrajectoryMultiplicityBatch(
            multiplicities=values,
            eligible_mask=eligible,
            skip_reasons=tuple(skip_reasons),
            lookup_seconds=lookup_seconds,
            predicate_eval_seconds=eval_seconds,
            provider_seconds=time.perf_counter() - started,
            segments_scanned=scanned,
        )

    def matching_multiplicities(
        self,
        *,
        anchor_trajectory_ids: Sequence[object],
        contexts: Sequence[GeneratedTrainingContext],
    ) -> np.ndarray:
        result = self.evaluate_batch(
            anchor_trajectory_ids=anchor_trajectory_ids,
            contexts=contexts,
        )
        if not np.all(result.eligible_mask):
            reasons = sorted({reason for reason in result.skip_reasons if reason is not None})
            raise UnsupportedTrajectoryContext(
                f"trajectory contexts are not all eligible: {reasons}"
            )
        return result.multiplicities

    def _unsupported_reason(self, context: GeneratedTrainingContext) -> str | None:
        static = set(self.trajectory_static_columns)
        varying = set(self.segment_varying_columns)
        for column, token in zip(self.metadata.columns, context.tokens):
            if column.kind != ColumnKind.DATA or token.op == PredicateOp.WILDCARD:
                continue
            if column.name in static or column.name in varying:
                continue
            return "unsupported_semantics"
        semantic_query = getattr(context, "trajectory_query", None)
        if semantic_query is not None:
            for temporal in getattr(semantic_query, "temporal_predicates", ()):
                if temporal.semantics != "overlap":
                    return "unsupported_semantics"
                for name in (temporal.start_column, temporal.end_column):
                    if name not in varying:
                        return "unsupported_semantics"
            for spatial in getattr(semantic_query, "spatial_predicates", ()):
                if spatial.semantics != "intersects":
                    return "unsupported_semantics"
                if spatial.srid is not None and self.srid is not None and spatial.srid != self.srid:
                    return "unsupported_semantics"
                for name in (
                    spatial.start_x_column,
                    spatial.start_y_column,
                    spatial.end_x_column,
                    spatial.end_y_column,
                ):
                    if name not in varying:
                        return "unsupported_semantics"
        return None

    def _matching_mask_slice(
        self,
        start: int,
        stop: int,
        context: GeneratedTrainingContext,
    ) -> np.ndarray:
        mask = np.ones(stop - start, dtype=bool)
        semantic_query = getattr(context, "trajectory_query", None)
        owned = semantic_owned_columns(semantic_query)
        for column, token in zip(self.metadata.columns, context.tokens):
            if column.kind != ColumnKind.DATA or token.op == PredicateOp.WILDCARD:
                continue
            if column.name in owned:
                continue
            if column.name not in self.segment_varying_columns:
                continue
            values = self._segment_values(column.name, start, stop)
            if values is None:
                continue
            mask &= _token_satisfies_numeric_array(values, token)
        if semantic_query is not None:
            for temporal in semantic_query.temporal_predicates:
                mask &= temporal_overlap_mask(
                    self._required_segment_values(temporal.start_column, start, stop),
                    self._required_segment_values(temporal.end_column, start, stop),
                    lower=temporal.lower,
                    upper=temporal.upper,
                )
            for spatial in semantic_query.spatial_predicates:
                mask &= segment_rectangle_intersects_mask(
                    self._required_segment_values(spatial.start_x_column, start, stop),
                    self._required_segment_values(spatial.start_y_column, start, stop),
                    self._required_segment_values(spatial.end_x_column, start, stop),
                    self._required_segment_values(spatial.end_y_column, start, stop),
                    spatial.min_x,
                    spatial.min_y,
                    spatial.max_x,
                    spatial.max_y,
                )
        return mask

    def _required_segment_values(self, column_name: str, start: int, stop: int) -> np.ndarray:
        values = self._segment_values(column_name, start, stop)
        if values is None:
            raise KeyError(column_name)
        return values

    def _segment_values(self, column_name: str, start: int, stop: int) -> np.ndarray | None:
        by_column: Mapping[str, np.ndarray] = {
            "segments:segment_idx": self.segment_idx,
            "segments:t_s": self.t_s,
            "segments:t_e": self.t_e,
            "segments:s_x": self.s_x,
            "segments:s_y": self.s_y,
            "segments:e_x": self.e_x,
            "segments:e_y": self.e_y,
            "segment_idx": self.segment_idx,
            "t_s": self.t_s,
            "t_e": self.t_e,
            "s_x": self.s_x,
            "s_y": self.s_y,
            "e_x": self.e_x,
            "e_y": self.e_y,
        }
        array = by_column.get(column_name)
        if array is None:
            return None
        return np.asarray(array[start:stop])


def write_compact_trajectory_index(
    directory: str | Path,
    *,
    metadata: ModelMetadata,
    trip_ids: Sequence[object],
    segment_idx: Sequence[object],
    t_s: Sequence[object],
    t_e: Sequence[object],
    s_x: Sequence[object],
    s_y: Sequence[object],
    e_x: Sequence[object],
    e_y: Sequence[object],
    trajectory_key: str = "trip_id",
    entity_table: str = "trips",
    segment_table: str = "segments",
    segment_key: str = "trip_id,segment_idx",
    trajectory_static_columns: Sequence[str] = (),
    segment_varying_columns: Sequence[str] = (
        "segments:segment_idx",
        "segments:t_s",
        "segments:t_e",
        "segments:s_x",
        "segments:s_y",
        "segments:e_x",
        "segments:e_y",
    ),
    srid: int | None = None,
) -> dict[str, Any]:
    """Write a sorted, memory-mappable trajectory index for POL segments."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trip_ids_array = np.asarray([int(value) for value in trip_ids], dtype=np.int64)
    segment_idx_array = np.asarray([int(value) for value in segment_idx], dtype=np.int32)
    t_s_array = np.asarray(t_s, dtype=object)
    t_e_array = np.asarray(t_e, dtype=object)
    s_x_array = np.asarray(s_x, dtype=float)
    s_y_array = np.asarray(s_y, dtype=float)
    e_x_array = np.asarray(e_x, dtype=float)
    e_y_array = np.asarray(e_y, dtype=float)
    expected_shape = trip_ids_array.shape
    if any(
        array.shape != expected_shape
        for array in (
            segment_idx_array,
            t_s_array,
            t_e_array,
            s_x_array,
            s_y_array,
            e_x_array,
            e_y_array,
        )
    ):
        raise ValueError("segment arrays must have identical lengths")
    order = np.lexsort((segment_idx_array, trip_ids_array))
    sorted_trip_ids = trip_ids_array[order]
    sorted_segment_idx = segment_idx_array[order]
    unique_trip_ids, starts = np.unique(sorted_trip_ids, return_index=True)
    offsets = np.concatenate(
        [starts.astype(np.int64), np.asarray([len(sorted_trip_ids)], dtype=np.int64)]
    )
    arrays = {
        "trajectory_ids": unique_trip_ids.astype(np.int64),
        "offsets": offsets,
        "segment_idx": sorted_segment_idx.astype(np.int32),
        "t_s": np.asarray(
            [_timestamp_to_float(value) for value in t_s_array[order]],
            dtype=np.float64,
        ),
        "t_e": np.asarray(
            [_timestamp_to_float(value) for value in t_e_array[order]],
            dtype=np.float64,
        ),
        "s_x": np.asarray(s_x_array[order], dtype=np.float64),
        "s_y": np.asarray(s_y_array[order], dtype=np.float64),
        "e_x": np.asarray(e_x_array[order], dtype=np.float64),
        "e_y": np.asarray(e_y_array[order], dtype=np.float64),
    }
    filenames = {name: f"{name}.npy" for name in arrays}
    for name, values in arrays.items():
        np.save(root / filenames[name], values)
    compatibility_hash = trajectory_index_compatibility_hash(
        metadata,
        predicate_columns=(),
        trajectory_key=trajectory_key,
        entity_table=entity_table,
        segment_table=segment_table,
        segment_key=segment_key,
        trajectory_static_columns=trajectory_static_columns,
        segment_varying_columns=segment_varying_columns,
        srid=srid,
        format_version=TRAJECTORY_INDEX_FORMAT_VERSION,
    )
    index_bytes = int(sum(values.size * values.dtype.itemsize for values in arrays.values()))
    manifest = {
        "format_version": TRAJECTORY_INDEX_FORMAT_VERSION,
        "index_type": "compact_pol_segment_mmap",
        "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
        "temporal_semantics_version": TEMPORAL_OVERLAP_SEMANTICS_VERSION,
        "spatial_semantics_version": SPATIAL_INTERSECTS_SEMANTICS_VERSION,
        "metadata": metadata.to_json_dict(),
        "compatibility_hash": compatibility_hash,
        "trajectory_key": trajectory_key,
        "entity_table": entity_table,
        "segment_table": segment_table,
        "segment_key": segment_key,
        "predicate_columns": [],
        "trajectory_static_columns": list(trajectory_static_columns),
        "segment_varying_columns": list(segment_varying_columns),
        "srid": srid,
        "trajectory_count": int(unique_trip_ids.shape[0]),
        "segment_count": int(sorted_trip_ids.shape[0]),
        "index_bytes": index_bytes,
        "bytes_per_segment": float(index_bytes / max(1, int(sorted_trip_ids.shape[0]))),
        "arrays": filenames,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _token_satisfies_array(values: np.ndarray, token: PredicateToken) -> np.ndarray:
    if token.op == PredicateOp.WILDCARD:
        return np.ones(values.shape[0], dtype=bool)
    if token.op == PredicateOp.EQUAL:
        return values == token.value
    result = np.zeros(values.shape[0], dtype=bool)
    for index, value in enumerate(values):
        try:
            if token.op == PredicateOp.LESS_EQUAL:
                result[index] = bool(value <= token.value)
            elif token.op == PredicateOp.LESS_THAN:
                result[index] = bool(value < token.value)
            elif token.op == PredicateOp.GREATER_EQUAL:
                result[index] = bool(value >= token.value)
            elif token.op == PredicateOp.GREATER_THAN:
                result[index] = bool(value > token.value)
            elif token.op == PredicateOp.RANGE:
                result[index] = bool(token.value <= value <= token.upper)
        except TypeError:
            result[index] = False
    return result


def validate_trajectory_index_runtime_compatibility(
    *,
    index: object,
    runtime_config: TrajectoryDistinctRuntimeConfig,
    metadata: ModelMetadata,
) -> None:
    mismatches: list[str] = []
    for field_name in (
        "entity_table",
        "segment_table",
        "trajectory_key",
        "segment_key",
        "trajectory_static_columns",
        "segment_varying_columns",
        "srid",
    ):
        expected = getattr(runtime_config, field_name)
        observed = getattr(index, field_name)
        if observed != expected:
            mismatches.append(f"{field_name}: index={observed!r}, config={expected!r}")
    index_metadata = getattr(index, "metadata")
    expected_schema = metadata.schema_hash or metadata.stable_schema_hash()
    observed_schema = index_metadata.schema_hash or index_metadata.stable_schema_hash()
    if observed_schema != expected_schema:
        mismatches.append(
            f"metadata_schema_hash: index={observed_schema!r}, config={expected_schema!r}"
        )
    if getattr(index, "format_version") != TRAJECTORY_INDEX_FORMAT_VERSION:
        mismatches.append(
            "index_format_version: "
            f"index={getattr(index, 'format_version')!r}, "
            f"expected={TRAJECTORY_INDEX_FORMAT_VERSION!r}"
        )
    expected_hash = trajectory_index_compatibility_hash(
        metadata,
        predicate_columns=getattr(index, "predicate_columns", ()),
        trajectory_key=runtime_config.trajectory_key,
        entity_table=runtime_config.entity_table,
        segment_table=runtime_config.segment_table,
        segment_key=runtime_config.segment_key,
        trajectory_static_columns=runtime_config.trajectory_static_columns,
        segment_varying_columns=runtime_config.segment_varying_columns,
        srid=runtime_config.srid,
        format_version=TRAJECTORY_INDEX_FORMAT_VERSION,
    )
    if getattr(index, "compatibility_hash") != expected_hash:
        mismatches.append("compatibility_hash does not match current runtime config")
    if mismatches:
        raise ValueError(
            "trajectory index is incompatible with current trajectory_distinct config: "
            + "; ".join(mismatches)
        )


def _token_satisfies_numeric_array(values: np.ndarray, token: PredicateToken) -> np.ndarray:
    if token.op == PredicateOp.WILDCARD:
        return np.ones(values.shape[0], dtype=bool)
    array = np.asarray(values)
    try:
        value = None if token.value is None else _literal_to_numeric(token.value)
        upper = None if token.upper is None else _literal_to_numeric(token.upper)
    except ValueError:
        return _token_satisfies_array(array.astype(object), token)
    if token.op != PredicateOp.RANGE and value is None:
        return np.zeros(array.shape[0], dtype=bool)
    if token.op == PredicateOp.RANGE and (value is None or upper is None):
        return np.zeros(array.shape[0], dtype=bool)
    if token.op == PredicateOp.EQUAL:
        return array == value
    if token.op == PredicateOp.LESS_EQUAL:
        return array <= value
    if token.op == PredicateOp.LESS_THAN:
        return array < value
    if token.op == PredicateOp.GREATER_EQUAL:
        return array >= value
    if token.op == PredicateOp.GREATER_THAN:
        return array > value
    if token.op == PredicateOp.RANGE:
        return (array >= value) & (array <= upper)
    return np.zeros(array.shape[0], dtype=bool)


def _literal_to_numeric(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value)
    try:
        return float(text)
    except ValueError:
        return _timestamp_to_float(text)


def temporal_overlap_mask(
    start_values: np.ndarray,
    end_values: np.ndarray,
    *,
    lower: Any | None,
    upper: Any | None,
) -> np.ndarray:
    """POL interval overlap: segment_start < q_upper and segment_end >= q_lower."""

    starts = np.asarray([_timestamp_to_float(value) for value in start_values], dtype=float)
    ends = np.asarray([_timestamp_to_float(value) for value in end_values], dtype=float)
    mask = np.ones(starts.shape[0], dtype=bool)
    if upper is not None:
        mask &= starts < _timestamp_to_float(upper)
    if lower is not None:
        mask &= ends >= _timestamp_to_float(lower)
    return mask


def segment_rectangle_intersects_mask(
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> np.ndarray:
    """Exact line-segment/AABB intersection including boundary touches."""

    sx = np.asarray(sx, dtype=float)
    sy = np.asarray(sy, dtype=float)
    ex = np.asarray(ex, dtype=float)
    ey = np.asarray(ey, dtype=float)
    min_x, max_x = sorted((float(min_x), float(max_x)))
    min_y, max_y = sorted((float(min_y), float(max_y)))
    inside_start = (min_x <= sx) & (sx <= max_x) & (min_y <= sy) & (sy <= max_y)
    inside_end = (min_x <= ex) & (ex <= max_x) & (min_y <= ey) & (ey <= max_y)
    intersects = inside_start | inside_end
    for x in (min_x, max_x):
        intersects |= _crosses_vertical_edge(sx, sy, ex, ey, x, min_y, max_y)
    for y in (min_y, max_y):
        intersects |= _crosses_horizontal_edge(sx, sy, ex, ey, y, min_x, max_x)
    return intersects


def _crosses_vertical_edge(
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    x: float,
    min_y: float,
    max_y: float,
) -> np.ndarray:
    dx = ex - sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(x) - sx) / dx
        y = sy + t * (ey - sy)
    vertical_segment = dx == 0.0
    on_vertical = vertical_segment & (sx == float(x))
    overlap_y = np.maximum(np.minimum(sy, ey), min_y) <= np.minimum(np.maximum(sy, ey), max_y)
    return ((0.0 <= t) & (t <= 1.0) & (min_y <= y) & (y <= max_y)) | (
        on_vertical & overlap_y
    )


def _crosses_horizontal_edge(
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    y: float,
    min_x: float,
    max_x: float,
) -> np.ndarray:
    dy = ey - sy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(y) - sy) / dy
        x = sx + t * (ex - sx)
    horizontal_segment = dy == 0.0
    on_horizontal = horizontal_segment & (sy == float(y))
    overlap_x = np.maximum(np.minimum(sx, ex), min_x) <= np.minimum(np.maximum(sx, ex), max_x)
    return ((0.0 <= t) & (t <= 1.0) & (min_x <= x) & (x <= max_x)) | (
        on_horizontal & overlap_x
    )


def _timestamp_to_float(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value)
        dt = None
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                pass
        if dt is None:
            raise ValueError(f"unsupported timestamp literal {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def trajectory_index_compatibility_hash(
    metadata: ModelMetadata,
    *,
    predicate_columns: Sequence[str] = (),
    trajectory_key: str,
    entity_table: str = "trips",
    segment_table: str = "segments",
    segment_key: str = "trip_id,segment_idx",
    trajectory_static_columns: Sequence[str] = (),
    segment_varying_columns: Sequence[str] = (),
    srid: int | None = None,
    format_version: int = TRAJECTORY_INDEX_FORMAT_VERSION,
) -> str:
    payload = {
        "schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "predicate_columns": tuple(predicate_columns),
        "trajectory_key": trajectory_key,
        "entity_table": entity_table,
        "segment_table": segment_table,
        "segment_key": segment_key,
        "trajectory_static_columns": tuple(trajectory_static_columns),
        "segment_varying_columns": tuple(segment_varying_columns),
        "temporal_semantics_version": TEMPORAL_OVERLAP_SEMANTICS_VERSION,
        "spatial_semantics_version": SPATIAL_INTERSECTS_SEMANTICS_VERSION,
        "srid": srid,
        "index_format_version": int(format_version),
        "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_columns_exist(metadata: ModelMetadata, column_names: Sequence[str]) -> None:
    known = {column.name for column in metadata.columns}
    missing = [name for name in column_names if name not in known]
    if missing:
        raise ValueError(f"trajectory distinct columns missing from metadata: {missing}")
