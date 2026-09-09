from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.data.trajectory_distinct import (
    PhysicalSpatialPredicate,
    SegmentMbrSpatialPredicate,
    TRAJECTORY_TARGET_SEMANTICS_VERSION,
    TrajectoryDistinctNotApplicable,
    TrajectoryDistinctRuntimeConfig,
    TrajectoryQuerySemantics,
    SegmentSpatialPredicate,
    SegmentTemporalPredicate,
    canonicalize_segment_mbr_predicate,
    trajectory_base_measure_support,
    trajectory_query_has_zero_support,
)
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.inference.estimator import OnePassEstimator
from model.src.predicates.generation import (
    GeneratedTrainingContext,
    inverse_fanouts_for_table_subset,
    tokens_for_query_tables,
)
from model.src.predicates.operators import PredicateOp, PredicateToken


@dataclass(frozen=True)
class PolDistinctEvaluation:
    query_id: str | None
    distinct_estimate_status: str
    matching_segments_true: int | None = None
    matching_segment_estimate: float | None = None
    matching_segment_qerror: float | None = None
    distinct_trajectories_true: int | None = None
    distinct_trajectory_estimate: float | None = None
    distinct_trajectory_qerror: float | None = None
    a_true: float | None = None
    a_hat: float | None = None
    a_abs_error: float | None = None
    model_forward_calls: int | None = None
    latency_seconds: float | None = None
    database_matching_segments_true: int | None = None
    database_distinct_trajectories_true: int | None = None
    database_a_true: float | None = None
    database_truth_status: str = "missing_database_truth"
    fixture_matching_segments: int | None = None
    fixture_distinct_trajectories: int | None = None
    fixture_a: float | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DatabaseTrajectoryTruth:
    available: bool
    status: str
    matching_segments_true: int | None = None
    distinct_trajectories_true: int | None = None
    a_true: float | None = None


def assert_checkpoint_trajectory_config_compatible(
    checkpoint_payload: Mapping[str, Any],
    runtime_config: TrajectoryDistinctRuntimeConfig,
    trajectory_spatial: Mapping[str, Any] | None = None,
) -> None:
    """Reject evaluating checkpoints under incompatible trajectory semantics."""

    stored = checkpoint_payload.get("trajectory_distinct")
    if stored is None:
        stored = checkpoint_payload.get("model_configuration", {}).get("trajectory_distinct", {})
    stored = dict(stored or {})
    if not bool(stored.get("enabled", False)):
        raise ValueError("checkpoint was not trained with trajectory_distinct.enabled=true")
    stored_runtime = TrajectoryDistinctRuntimeConfig.from_dict(stored)
    for field in (
        "entity_table",
        "segment_table",
        "trajectory_key",
        "segment_key",
        "predicate_scope",
        "srid",
        "trajectory_static_columns",
        "segment_varying_columns",
    ):
        if getattr(stored_runtime, field) != getattr(runtime_config, field):
            raise ValueError(
                "checkpoint trajectory_distinct config does not match runtime "
                f"field {field}: checkpoint={getattr(stored_runtime, field)!r}, "
                f"runtime={getattr(runtime_config, field)!r}"
            )
    stored_version = checkpoint_payload.get("trajectory_target_semantics_version")
    if stored_version != TRAJECTORY_TARGET_SEMANTICS_VERSION:
        raise ValueError(
            "checkpoint trajectory target semantics version does not match runtime: "
            f"checkpoint={stored_version!r}, runtime={TRAJECTORY_TARGET_SEMANTICS_VERSION!r}"
        )
    if trajectory_spatial is not None:
        stored_spatial = dict(
            checkpoint_payload.get("model_configuration", {}).get(
                "trajectory_spatial",
                {},
            )
        )
        runtime_spatial = dict(trajectory_spatial or {})
        if bool(stored_spatial.get("enabled", False)) != bool(
            runtime_spatial.get("enabled", False)
        ):
            raise ValueError(
                "checkpoint trajectory_spatial.enabled does not match runtime"
            )
        if bool(runtime_spatial.get("enabled", False)) and str(
            stored_spatial.get("representation", "")
        ) != str(runtime_spatial.get("representation", "")):
            raise ValueError(
                "checkpoint trajectory_spatial.representation does not match runtime"
            )


def pol_workload_record_to_context(
    record: Mapping[str, Any],
    metadata: ModelMetadata,
    *,
    srid: int | None = None,
    trajectory_spatial: Mapping[str, Any] | None = None,
) -> GeneratedTrainingContext:
    """Convert a structured POL workload JSON record into model query context."""

    included_tables = frozenset(str(table) for table in record.get("tables", ()))
    if not included_tables:
        included_tables = frozenset(
            column.table
            for column in metadata.columns
            if column.table is not None and column.kind != ColumnKind.FANOUT
        )
    ordinary: dict[str, PredicateToken] = {}
    temporal_predicates: list[SegmentTemporalPredicate] = []
    spatial_predicates: list[object] = []
    for predicate in record.get("predicates", ()):
        table = str(predicate["table"])
        attribute = str(predicate["attribute"])
        mode = str(predicate["mode"])
        column_name = f"{table}:{attribute}"
        if mode == "nominal_eq":
            ordinary[column_name] = _canonical_data_token(
                metadata,
                column_name,
                PredicateToken.equal(predicate["value"]),
            )
        elif mode == "range":
            ordinary[column_name] = _canonical_data_token(
                metadata,
                column_name,
                PredicateToken.range(
                    predicate["lower"],
                    predicate["upper"],
                ),
            )
        elif mode == "unbounded":
            ordinary[column_name] = _canonical_data_token(
                metadata,
                column_name,
                _operator_token(
                    str(predicate["operator"]),
                    predicate["value"],
                ),
            )
        elif mode in {"temporal_overlap", "temporal_unbounded"}:
            start_column, end_column = _temporal_columns(table, attribute)
            if mode == "temporal_overlap":
                lower = predicate["lower"]
                upper = predicate["upper"]
                ordinary[start_column] = _canonical_data_token(
                    metadata,
                    start_column,
                    PredicateToken(PredicateOp.LESS_THAN, value=upper),
                )
                ordinary[end_column] = _canonical_data_token(
                    metadata,
                    end_column,
                    PredicateToken(PredicateOp.GREATER_EQUAL, value=lower),
                )
                temporal_predicates.append(
                    SegmentTemporalPredicate(start_column, end_column, lower=lower, upper=upper)
                )
            else:
                op = str(predicate["operator"])
                value = predicate["value"]
                column_name = start_column if op in {"<", "<="} else end_column
                ordinary[column_name] = _canonical_data_token(
                    metadata,
                    column_name,
                    _operator_token(op, value),
                )
        elif mode in {"spatial_intersects", "spatial_unbounded"}:
            min_x = float(predicate["min_x"])
            min_y = float(predicate["min_y"])
            max_x = float(predicate["max_x"])
            max_y = float(predicate["max_y"])
            if _uses_segment_mbr_spatial(trajectory_spatial, table, attribute, metadata):
                min_x_column, max_x_column, min_y_column, max_y_column = _mbr_spatial_columns(
                    table,
                    attribute,
                )
                canonical = canonicalize_segment_mbr_predicate(
                    metadata,
                    min_x_column=min_x_column,
                    max_x_column=max_x_column,
                    min_y_column=min_y_column,
                    max_y_column=max_y_column,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                )
                if canonical.zero_support:
                    if canonical.zero_support_column is None or canonical.zero_support_token is None:
                        raise ValueError("zero-support MBR canonicalization did not provide a token")
                    ordinary[canonical.zero_support_column] = canonical.zero_support_token
                else:
                    ordinary[min_x_column] = PredicateToken(
                        PredicateOp.LESS_EQUAL,
                        value=canonical.max_x_literal,
                    )
                    ordinary[max_x_column] = PredicateToken(
                        PredicateOp.GREATER_EQUAL,
                        value=canonical.min_x_literal,
                    )
                    ordinary[min_y_column] = PredicateToken(
                        PredicateOp.LESS_EQUAL,
                        value=canonical.max_y_literal,
                    )
                    ordinary[max_y_column] = PredicateToken(
                        PredicateOp.GREATER_EQUAL,
                        value=canonical.min_y_literal,
                    )
                spatial_predicates.append(
                    SegmentMbrSpatialPredicate(
                        canonical.min_x,
                        canonical.min_y,
                        canonical.max_x,
                        canonical.max_y,
                        srid=(srid if srid is not None else int(predicate.get("srid", 26916))),
                        min_x_column=min_x_column,
                        max_x_column=max_x_column,
                        min_y_column=min_y_column,
                        max_y_column=max_y_column,
                        physical_min_x=canonical.physical_min_x,
                        physical_min_y=canonical.physical_min_y,
                        physical_max_x=canonical.physical_max_x,
                        physical_max_y=canonical.physical_max_y,
                        zero_support=canonical.zero_support,
                    )
                )
            else:
                columns = _spatial_columns(table, attribute)
                has_endpoint_columns = _metadata_has_columns(metadata, columns)
                if has_endpoint_columns:
                    for column_name in columns:
                        if column_name.endswith(":s_x") or column_name.endswith(":e_x"):
                            ordinary[column_name] = _canonical_data_token(
                                metadata,
                                column_name,
                                PredicateToken.range(min_x, max_x),
                            )
                        else:
                            ordinary[column_name] = _canonical_data_token(
                                metadata,
                                column_name,
                                PredicateToken.range(min_y, max_y),
                            )
                    spatial_predicates.append(
                        SegmentSpatialPredicate(
                            min_x,
                            min_y,
                            max_x,
                            max_y,
                            srid=(srid if srid is not None else int(predicate.get("srid", 26916))),
                            start_x_column=columns[0],
                            start_y_column=columns[1],
                            end_x_column=columns[2],
                            end_y_column=columns[3],
                        )
                    )
                else:
                    spatial_predicates.append(
                        PhysicalSpatialPredicate(
                            table=table,
                            attribute=attribute,
                            min_x=min_x,
                            min_y=min_y,
                            max_x=max_x,
                            max_y=max_y,
                            srid=(srid if srid is not None else int(predicate.get("srid", 26916))),
                            geometry_column=_geometry_column(table, attribute),
                        )
                    )
        else:
            raise ValueError(f"unsupported POL predicate mode {mode!r}")
    inverse_fanouts = inverse_fanouts_for_table_subset(metadata, included_tables)
    tokens = tokens_for_query_tables(metadata, set(included_tables), set(inverse_fanouts), ordinary)
    trajectory_query = (
        TrajectoryQuerySemantics(
            scalar_predicates=tuple(ordinary.items()),
            temporal_predicates=tuple(temporal_predicates),
            spatial_predicates=tuple(spatial_predicates),
        )
        if temporal_predicates or spatial_predicates
        else None
    )
    return GeneratedTrainingContext(
        tokens=tuple(tokens),
        included_tables=included_tables,
        inverse_fanout_columns=inverse_fanouts,
        ordinary_predicates=ordinary,
        trajectory_query=trajectory_query,
    )


def database_truth_for_trajectory_query(
    record: Mapping[str, Any],
    context: GeneratedTrainingContext,
    *,
    trajectory_key: str = "trip_id",
    segment_table: str = "segments",
) -> DatabaseTrajectoryTruth:
    """Return population truth from executed POL workload fields when eligible."""

    if segment_table not in context.included_tables:
        return DatabaseTrajectoryTruth(False, "non_segment_measure")
    raw_join = record.get("join_cardinality")
    raw_entity = record.get("entity_cardinality")
    if raw_join is None or raw_entity is None:
        return DatabaseTrajectoryTruth(False, "missing_database_truth")
    entity_key = record.get("entity_key")
    if entity_key is not None and str(entity_key) != trajectory_key:
        return DatabaseTrajectoryTruth(False, "entity_cardinality_not_trajectory_key")
    entity_sql = record.get("entity_sql")
    if entity_sql is not None and trajectory_key not in str(entity_sql):
        return DatabaseTrajectoryTruth(False, "entity_cardinality_not_trajectory_key")
    matching = int(raw_join)
    distinct = int(raw_entity)
    if matching < 0 or distinct < 0:
        return DatabaseTrajectoryTruth(False, "negative_database_truth")
    if matching == 0 and distinct != 0:
        return DatabaseTrajectoryTruth(False, "inconsistent_database_truth")
    return DatabaseTrajectoryTruth(
        True,
        "database_truth_available",
        matching_segments_true=matching,
        distinct_trajectories_true=distinct,
        a_true=None if matching == 0 else float(distinct / matching),
    )


def evaluate_pol_distinct_record(
    record: Mapping[str, Any],
    *,
    metadata: ModelMetadata,
    estimator: OnePassEstimator,
    oracle: ExactOracle | None = None,
    trajectory_ids: tuple[object, ...] | list[object] | None = None,
    segment_ids: tuple[object, ...] | list[object] | None = None,
    trajectory_config: TrajectoryDistinctRuntimeConfig | None = None,
    trajectory_spatial: Mapping[str, Any] | None = None,
) -> PolDistinctEvaluation:
    """Evaluate one structured POL workload record when semantics are supported."""

    context = pol_workload_record_to_context(
        record,
        metadata,
        trajectory_spatial=trajectory_spatial,
    )
    support = trajectory_base_measure_support(context)
    database_truth = database_truth_for_trajectory_query(
        record,
        context,
        trajectory_key=(
            trajectory_config.trajectory_key
            if trajectory_config is not None
            else "trip_id"
        ),
        segment_table=(
            trajectory_config.segment_table
            if trajectory_config is not None
            else "segments"
        ),
    )
    exact = None
    if _is_physical_segment_spatial_record(record) and _mbr_spatial_enabled(trajectory_spatial):
        common_truth = _truth_payload(database_truth, exact)
        if trajectory_query_has_zero_support(context.trajectory_query):
            matching_qerror = (
                None
                if not database_truth.available
                else _q_error(0.0, database_truth.matching_segments_true)
            )
            return PolDistinctEvaluation(
                query_id=record.get("query_id"),
                distinct_estimate_status="unsupported_physical_spatial_distinct_mbr_approximation",
                matching_segment_estimate=0.0,
                matching_segment_qerror=matching_qerror,
                model_forward_calls=0,
                latency_seconds=0.0,
                **common_truth,
            )
        try:
            estimate = estimator.estimate_distinct_trajectories(
                list(context.tokens),
                context=context,
                trajectory_config=trajectory_config,
            )
        except TrajectoryDistinctNotApplicable as exc:
            return PolDistinctEvaluation(
                query_id=record.get("query_id"),
                distinct_estimate_status=str(exc),
                **common_truth,
            )
        matching_qerror = (
            None
            if not database_truth.available
            else _q_error(
                estimate.matching_segment_estimate,
                database_truth.matching_segments_true,
            )
        )
        return PolDistinctEvaluation(
            query_id=record.get("query_id"),
            distinct_estimate_status="unsupported_physical_spatial_distinct_mbr_approximation",
            matching_segment_estimate=estimate.matching_segment_estimate,
            matching_segment_qerror=matching_qerror,
            model_forward_calls=estimate.model_forward_calls,
            latency_seconds=estimate.latency_seconds,
            **common_truth,
        )
    if support.eligible and oracle is not None and trajectory_ids is not None:
        exact = oracle.exact_distinct_trajectory_count(
            context,
            trajectory_ids=trajectory_ids,
            segment_ids=segment_ids,
        )
    common_truth = _truth_payload(database_truth, exact)
    if not support.eligible:
        return PolDistinctEvaluation(
            query_id=record.get("query_id"),
            distinct_estimate_status=support.reason or "unsupported_base_segment_measure",
            **common_truth,
        )
    try:
        estimate = estimator.estimate_distinct_trajectories(
            list(context.tokens),
            context=context,
            trajectory_config=trajectory_config,
        )
    except TrajectoryDistinctNotApplicable as exc:
        return PolDistinctEvaluation(
            query_id=record.get("query_id"),
            distinct_estimate_status=str(exc),
            **common_truth,
        )
    matching_qerror = (
        None
        if not database_truth.available
        else _q_error(
            estimate.matching_segment_estimate,
            database_truth.matching_segments_true,
        )
    )
    distinct_qerror = (
        None
        if not database_truth.available
        else _q_error(
            estimate.distinct_trajectory_estimate,
            database_truth.distinct_trajectories_true,
        )
    )
    a_abs_error = (
        None
        if not database_truth.available or database_truth.a_true is None
        else abs(float(estimate.traj_dedup_factor) - float(database_truth.a_true))
    )
    return PolDistinctEvaluation(
        query_id=record.get("query_id"),
        distinct_estimate_status="ok",
        matching_segment_estimate=estimate.matching_segment_estimate,
        matching_segment_qerror=matching_qerror,
        distinct_trajectory_estimate=estimate.distinct_trajectory_estimate,
        distinct_trajectory_qerror=distinct_qerror,
        a_hat=estimate.traj_dedup_factor,
        a_abs_error=a_abs_error,
        model_forward_calls=estimate.model_forward_calls,
        latency_seconds=estimate.latency_seconds,
        **common_truth,
    )


def _truth_payload(
    database_truth: DatabaseTrajectoryTruth,
    exact: Any | None,
) -> dict[str, Any]:
    return {
        "matching_segments_true": database_truth.matching_segments_true,
        "distinct_trajectories_true": database_truth.distinct_trajectories_true,
        "a_true": database_truth.a_true,
        "database_matching_segments_true": database_truth.matching_segments_true,
        "database_distinct_trajectories_true": database_truth.distinct_trajectories_true,
        "database_a_true": database_truth.a_true,
        "database_truth_status": database_truth.status,
        "fixture_matching_segments": (
            None if exact is None else exact.matching_segments_true
        ),
        "fixture_distinct_trajectories": (
            None if exact is None else exact.distinct_trajectories_true
        ),
        "fixture_a": None if exact is None else exact.a_true,
    }


def _operator_token(operator: str, value: Any) -> PredicateToken:
    op_map = {
        "<": PredicateOp.LESS_THAN,
        "<=": PredicateOp.LESS_EQUAL,
        ">": PredicateOp.GREATER_THAN,
        ">=": PredicateOp.GREATER_EQUAL,
    }
    if operator not in op_map:
        raise ValueError(f"unsupported operator {operator!r}")
    return PredicateToken(op_map[operator], value=value)


def _canonical_data_token(
    metadata: ModelMetadata,
    column_name: str,
    token: PredicateToken,
) -> PredicateToken:
    """Map workload thresholds onto exact literals from the encoded column domain."""

    try:
        column = metadata.columns[metadata.column_index(column_name)]
    except KeyError:
        return token
    if column.kind != ColumnKind.DATA:
        return token
    domain = _comparable_domain(column.domain)
    if not domain:
        return token
    if token.op == PredicateOp.WILDCARD:
        return token
    if token.op == PredicateOp.EQUAL:
        for literal in domain:
            if literal == token.value:
                return PredicateToken.equal(literal)
        return _zero_support_data_token(domain)
    if token.op == PredicateOp.RANGE:
        if _contains_literal(domain, token.value) and _contains_literal(domain, token.upper):
            return token if _leq(token.value, token.upper) else _zero_support_data_token(domain)
        lower = _min_satisfying(domain, token.value, PredicateOp.GREATER_EQUAL)
        upper = _max_satisfying(domain, token.upper, PredicateOp.LESS_EQUAL)
        if lower is None or upper is None or not _leq(lower, upper):
            return _zero_support_data_token(domain)
        return PredicateToken.range(lower, upper)
    if token.op == PredicateOp.GREATER_THAN:
        if _contains_literal(domain, token.value):
            return token
        literal = _min_satisfying(domain, token.value, PredicateOp.GREATER_THAN)
        return _zero_support_data_token(domain) if literal is None else PredicateToken(
            PredicateOp.GREATER_EQUAL,
            value=literal,
        )
    if token.op == PredicateOp.GREATER_EQUAL:
        if _contains_literal(domain, token.value):
            return token
        literal = _min_satisfying(domain, token.value, PredicateOp.GREATER_EQUAL)
        return _zero_support_data_token(domain) if literal is None else PredicateToken(
            PredicateOp.GREATER_EQUAL,
            value=literal,
        )
    if token.op == PredicateOp.LESS_THAN:
        if _contains_literal(domain, token.value):
            return token
        literal = _max_satisfying(domain, token.value, PredicateOp.LESS_THAN)
        return _zero_support_data_token(domain) if literal is None else PredicateToken(
            PredicateOp.LESS_EQUAL,
            value=literal,
        )
    if token.op == PredicateOp.LESS_EQUAL:
        if _contains_literal(domain, token.value):
            return token
        literal = _max_satisfying(domain, token.value, PredicateOp.LESS_EQUAL)
        return _zero_support_data_token(domain) if literal is None else PredicateToken(
            PredicateOp.LESS_EQUAL,
            value=literal,
        )
    return token


def _comparable_domain(domain: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        value
        for value in domain
        if not (isinstance(value, str) and value.startswith("__"))
    )


def _contains_literal(domain: tuple[Any, ...], literal: Any) -> bool:
    return any(value == literal for value in domain)


def _min_satisfying(
    domain: tuple[Any, ...],
    threshold: Any,
    op: PredicateOp,
) -> Any | None:
    candidates = [value for value in domain if _satisfies_op(value, threshold, op)]
    if not candidates:
        return None
    try:
        return min(candidates)
    except TypeError:
        return candidates[0]


def _max_satisfying(
    domain: tuple[Any, ...],
    threshold: Any,
    op: PredicateOp,
) -> Any | None:
    candidates = [value for value in domain if _satisfies_op(value, threshold, op)]
    if not candidates:
        return None
    try:
        return max(candidates)
    except TypeError:
        return candidates[-1]


def _satisfies_op(value: Any, threshold: Any, op: PredicateOp) -> bool:
    try:
        if op == PredicateOp.GREATER_THAN:
            return value > threshold
        if op == PredicateOp.GREATER_EQUAL:
            return value >= threshold
        if op == PredicateOp.LESS_THAN:
            return value < threshold
        if op == PredicateOp.LESS_EQUAL:
            return value <= threshold
    except TypeError:
        return False
    return False


def _leq(left: Any, right: Any) -> bool:
    try:
        return left <= right
    except TypeError:
        return False


def _zero_support_data_token(domain: tuple[Any, ...]) -> PredicateToken:
    try:
        maximum = max(domain)
    except TypeError:
        maximum = domain[-1]
    return PredicateToken(PredicateOp.GREATER_THAN, value=maximum)


def _temporal_columns(table: str, attribute: str) -> tuple[str, str]:
    if table == "segments" and attribute == "segment_time":
        return "segments:t_s", "segments:t_e"
    if table == "trips" and attribute == "trip_time":
        return "trips:start_time", "trips:end_time"
    return f"{table}:start_time", f"{table}:end_time"


def _spatial_columns(table: str, attribute: str) -> tuple[str, str, str, str]:
    if table == "segments" and attribute == "segment_geom":
        return "segments:s_x", "segments:s_y", "segments:e_x", "segments:e_y"
    if table == "trips" and attribute == "trip_geom":
        return "trips:s_x", "trips:s_y", "trips:e_x", "trips:e_y"
    return f"{table}:s_x", f"{table}:s_y", f"{table}:e_x", f"{table}:e_y"


def _mbr_spatial_columns(table: str, attribute: str) -> tuple[str, str, str, str]:
    if table == "segments" and attribute == "segment_geom":
        return (
            "segments:seg_min_x",
            "segments:seg_max_x",
            "segments:seg_min_y",
            "segments:seg_max_y",
        )
    return (
        f"{table}:seg_min_x",
        f"{table}:seg_max_x",
        f"{table}:seg_min_y",
        f"{table}:seg_max_y",
    )


def _uses_segment_mbr_spatial(
    trajectory_spatial: Mapping[str, Any] | None,
    table: str,
    attribute: str,
    metadata: ModelMetadata,
) -> bool:
    if not trajectory_spatial or not bool(trajectory_spatial.get("enabled", False)):
        return False
    if str(trajectory_spatial.get("representation", "")) != "segment_mbr":
        raise ValueError(
            "trajectory_spatial.representation must be segment_mbr when enabled"
        )
    columns = _mbr_spatial_columns(table, attribute)
    if table != "segments" or attribute != "segment_geom":
        return False
    if not _metadata_has_columns(metadata, columns):
        raise ValueError(
            "trajectory_spatial.segment_mbr requires prepared metadata columns "
            f"{columns}"
        )
    return True


def _mbr_spatial_enabled(trajectory_spatial: Mapping[str, Any] | None) -> bool:
    return bool(trajectory_spatial and trajectory_spatial.get("enabled", False)) and str(
        trajectory_spatial.get("representation", "")
    ) == "segment_mbr"


def _is_physical_segment_spatial_record(record: Mapping[str, Any]) -> bool:
    for predicate in record.get("predicates", ()):
        if (
            str(predicate.get("table")) == "segments"
            and str(predicate.get("attribute")) == "segment_geom"
            and str(predicate.get("mode")) in {"spatial_intersects", "spatial_unbounded"}
        ):
            return True
    return False


def _geometry_column(table: str, attribute: str) -> str:
    if attribute == "trip_geom":
        return f"{table}:trip_geom"
    if attribute == "segment_geom":
        return f"{table}:segment_geom"
    return f"{table}:{attribute}"


def _metadata_has_columns(metadata: ModelMetadata, names: tuple[str, ...]) -> bool:
    available = {column.name for column in metadata.columns}
    return all(name in available for name in names)


def _q_error(estimate: float, true_value: float) -> float:
    eps = 1.0e-12
    estimate = max(float(estimate), eps)
    true_value = max(float(true_value), eps)
    return max(estimate / true_value, true_value / estimate)
