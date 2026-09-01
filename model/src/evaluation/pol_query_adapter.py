from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.data.trajectory_distinct import (
    TrajectoryDistinctNotApplicable,
    TrajectoryDistinctRuntimeConfig,
    TrajectoryQuerySemantics,
    SegmentSpatialPredicate,
    SegmentTemporalPredicate,
    trajectory_base_measure_support,
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

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def pol_workload_record_to_context(
    record: Mapping[str, Any],
    metadata: ModelMetadata,
    *,
    srid: int | None = None,
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
    spatial_predicates: list[SegmentSpatialPredicate] = []
    for predicate in record.get("predicates", ()):
        table = str(predicate["table"])
        attribute = str(predicate["attribute"])
        mode = str(predicate["mode"])
        if mode == "nominal_eq":
            ordinary[f"{table}:{attribute}"] = PredicateToken.equal(predicate["value"])
        elif mode == "range":
            ordinary[f"{table}:{attribute}"] = PredicateToken.range(
                predicate["lower"],
                predicate["upper"],
            )
        elif mode == "unbounded":
            ordinary[f"{table}:{attribute}"] = _operator_token(
                str(predicate["operator"]),
                predicate["value"],
            )
        elif mode in {"temporal_overlap", "temporal_unbounded"}:
            start_column, end_column = _temporal_columns(table, attribute)
            if mode == "temporal_overlap":
                lower = predicate["lower"]
                upper = predicate["upper"]
                ordinary[start_column] = PredicateToken(PredicateOp.LESS_THAN, value=upper)
                ordinary[end_column] = PredicateToken(PredicateOp.GREATER_EQUAL, value=lower)
                temporal_predicates.append(
                    SegmentTemporalPredicate(start_column, end_column, lower=lower, upper=upper)
                )
            else:
                op = str(predicate["operator"])
                value = predicate["value"]
                column_name = start_column if op in {"<", "<="} else end_column
                ordinary[column_name] = _operator_token(op, value)
        elif mode in {"spatial_intersects", "spatial_unbounded"}:
            columns = _spatial_columns(table, attribute)
            min_x = float(predicate["min_x"])
            min_y = float(predicate["min_y"])
            max_x = float(predicate["max_x"])
            max_y = float(predicate["max_y"])
            for column_name in columns:
                if column_name.endswith(":s_x") or column_name.endswith(":e_x"):
                    ordinary[column_name] = PredicateToken.range(min_x, max_x)
                else:
                    ordinary[column_name] = PredicateToken.range(min_y, max_y)
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


def evaluate_pol_distinct_record(
    record: Mapping[str, Any],
    *,
    metadata: ModelMetadata,
    estimator: OnePassEstimator,
    oracle: ExactOracle | None = None,
    trajectory_ids: tuple[object, ...] | list[object] | None = None,
    segment_ids: tuple[object, ...] | list[object] | None = None,
    trajectory_config: TrajectoryDistinctRuntimeConfig | None = None,
) -> PolDistinctEvaluation:
    """Evaluate one structured POL workload record when semantics are supported."""

    context = pol_workload_record_to_context(record, metadata)
    exact = None
    if oracle is not None and trajectory_ids is not None:
        exact = oracle.exact_distinct_trajectory_count(
            context,
            trajectory_ids=trajectory_ids,
            segment_ids=segment_ids,
        )
    support = trajectory_base_measure_support(context)
    if not support.eligible:
        return PolDistinctEvaluation(
            query_id=record.get("query_id"),
            distinct_estimate_status=support.reason or "unsupported_base_segment_measure",
            matching_segments_true=(None if exact is None else exact.matching_segments_true),
            distinct_trajectories_true=(None if exact is None else exact.distinct_trajectories_true),
            a_true=(None if exact is None else exact.a_true),
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
            matching_segments_true=(None if exact is None else exact.matching_segments_true),
            distinct_trajectories_true=(None if exact is None else exact.distinct_trajectories_true),
            a_true=(None if exact is None else exact.a_true),
        )
    return PolDistinctEvaluation(
        query_id=record.get("query_id"),
        distinct_estimate_status="ok",
        matching_segments_true=(None if exact is None else exact.matching_segments_true),
        matching_segment_estimate=estimate.matching_segment_estimate,
        matching_segment_qerror=(
            None
            if exact is None
            else _q_error(estimate.matching_segment_estimate, exact.matching_segments_true)
        ),
        distinct_trajectories_true=(None if exact is None else exact.distinct_trajectories_true),
        distinct_trajectory_estimate=estimate.distinct_trajectory_estimate,
        distinct_trajectory_qerror=(
            None
            if exact is None
            else _q_error(estimate.distinct_trajectory_estimate, exact.distinct_trajectories_true)
        ),
        a_true=(None if exact is None else exact.a_true),
        a_hat=estimate.traj_dedup_factor,
        a_abs_error=(
            None if exact is None else abs(float(estimate.traj_dedup_factor) - exact.a_true)
        ),
        model_forward_calls=estimate.model_forward_calls,
    )


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


def _q_error(estimate: float, true_value: float) -> float:
    eps = 1.0e-12
    estimate = max(float(estimate), eps)
    true_value = max(float(true_value), eps)
    return max(estimate / true_value, true_value / estimate)
