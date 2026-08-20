from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.evaluation.metrics import q_error, q_error_floor_one
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.generation import inverse_fanouts_for_table_subset
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.sets import ColumnPredicateSet
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.vocabulary import PredicateVocabularies

OP_MAP = {
    "=": PredicateOp.EQUAL,
    ">": PredicateOp.GREATER_THAN,
    ">=": PredicateOp.GREATER_EQUAL,
    "<": PredicateOp.LESS_THAN,
    "<=": PredicateOp.LESS_EQUAL,
}
_MISSING = object()


@dataclass(frozen=True)
class RawPredicate:
    column: str
    op: str
    literal: Any
    text: str


@dataclass(frozen=True)
class TokenBuild:
    token: PredicateToken | None
    zero: bool = False
    all_values: bool = False
    missing_column: str | None = None
    unsupported: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuroCard JOB-light CSV queries.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    checkpoint = Path(args.checkpoint)
    queries_path = Path(args.queries)
    results_path = Path(args.results)
    summary_path = Path(args.summary)

    from model.src.inference.estimator import OnePassEstimator
    from model.src.inference.torch_estimator import TorchDistributionModel
    from model.src.model.checkpoint import load_resmade_checkpoint

    model, payload = load_resmade_checkpoint(checkpoint, map_location="cpu")
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    vocabularies = PredicateVocabularies.from_json_dict(
        payload["predicate_vocabularies"],
        metadata,
    )
    wrapped = TorchDistributionModel(model, metadata, vocabularies)
    estimator = OnePassEstimator(wrapped, metadata)

    rows = []
    start = perf_counter()
    for query_id, line in enumerate(queries_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        included, predicates, true_cardinality = parse_query(line)
        query_eval = eval_query(
            estimator,
            wrapped,
            model,
            vocabularies,
            metadata,
            included,
            predicates,
        )
        status, estimate, calls, query_wall, model_latency, inverse, branches, detail = query_eval[:8]
        factors, native_range_count, predicate_columns, predicate_operators = query_eval[8:]
        qe = q_error(estimate, true_cardinality)
        qe_floor_one = q_error_floor_one(estimate, true_cardinality)
        if len(factors):
            smallest_factor_index = int(np.argmin(factors))
            smallest_factor = float(factors[smallest_factor_index])
            smallest_factor_name = metadata.columns[smallest_factor_index].name
        else:
            smallest_factor_index = -1
            smallest_factor = float("nan")
            smallest_factor_name = ""
        rows.append(
            {
                "query_id": query_id,
                "status": status,
                "tables": ",".join(sorted(included)),
                "filters": ";".join(predicate.text for predicate in predicates),
                "true_cardinality": true_cardinality,
                "estimated_cardinality": estimate,
                "q_error": qe,
                "q_error_epsilon": qe,
                "q_error_floor_one": qe_floor_one,
                "zero_estimate": estimate == 0.0,
                "native_range_count": native_range_count,
                "legacy_external_subtraction_used": False,
                "number_of_predicates": len(predicates),
                "number_of_tables": len(included),
                "predicate_columns": ",".join(predicate_columns),
                "predicate_operators": ",".join(predicate_operators),
                "model_latency_seconds": model_latency,
                "query_wall_seconds": query_wall,
                "model_forward_calls": calls,
                "inverse_fanouts": inverse,
                "per_column_factors": ";".join(f"{value:.12g}" for value in factors),
                "log_per_column_factors": ";".join(
                    "-inf" if value <= 0.0 else f"{log(value):.12g}" for value in factors
                ),
                "smallest_column_factor": smallest_factor,
                "smallest_column_factor_index": smallest_factor_index,
                "smallest_column_factor_name": smallest_factor_name,
                "branch_estimates": branches,
                "missing_domain_predicates": detail if status == "zero_due_to_missing_domain" else "",
                "unsupported_predicates": detail if status == "unsupported" else "",
            }
        )
        print(f"query_id={query_id} status={status} q_error={qe:.6g}", flush=True)

    elapsed = perf_counter() - start
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, checkpoint, queries_path, results_path, summary_path, model, elapsed)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_literal(text: str) -> Any:
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text.strip("'\"")


def parse_query(line: str) -> tuple[set[str], list[RawPredicate], int]:
    tables_part, _joins_part, predicates_part, cardinality = line.strip().split("#")
    alias_to_table = {}
    tables = set()
    for item in tables_part.split(","):
        pieces = item.strip().split()
        table = pieces[0]
        alias = pieces[1] if len(pieces) > 1 else table
        alias_to_table[alias] = table
        tables.add(table)
    raw = [part.strip() for part in predicates_part.split(",") if part.strip()]
    predicates = []
    for index in range(0, len(raw), 3):
        colref, op, literal_text = raw[index : index + 3]
        alias, source_col = colref.split(".", 1)
        table = alias_to_table[alias]
        column = f"{table}:{source_col}"
        literal = parse_literal(literal_text)
        predicates.append(RawPredicate(column, op, literal, f"{column}{op}{literal_text}"))
    return tables, predicates, int(cardinality)


def comparable_domain_values(column: Any) -> list[Any]:
    values = []
    for value in column.domain:
        if isinstance(value, str) and value.startswith("__"):
            continue
        values.append(value)
    return values


def matching_domain_value(domain: tuple[Any, ...], literal: Any) -> Any:
    for value in domain:
        if value == literal:
            return value
    return _MISSING


def build_token(metadata: ModelMetadata, predicate: RawPredicate) -> TokenBuild:
    try:
        column = metadata.columns[metadata.column_index(predicate.column)]
    except KeyError:
        return TokenBuild(None, unsupported=f"unknown_column:{predicate.column}")
    if column.kind != ColumnKind.DATA:
        return TokenBuild(None, unsupported=f"non_data_predicate:{predicate.column}")
    op = OP_MAP[predicate.op]
    literal = predicate.literal
    domain = comparable_domain_values(column)
    if op == PredicateOp.EQUAL:
        canonical = matching_domain_value(column.domain, literal)
        if canonical is _MISSING:
            return TokenBuild(None, zero=True, missing_column=predicate.column)
        return TokenBuild(PredicateToken.equal(canonical))
    try:
        _ = [value for value in domain if value == literal or value < literal or value > literal]
    except TypeError:
        return TokenBuild(None, unsupported=f"type_mismatch:{predicate.column}")
    if op == PredicateOp.GREATER_THAN:
        candidates = [value for value in domain if value > literal]
        if not candidates:
            return TokenBuild(None, zero=True, missing_column=predicate.column)
        return _range_token_or_wildcard(domain, PredicateOp.GREATER_EQUAL, min(candidates))
    if op == PredicateOp.GREATER_EQUAL:
        candidates = [value for value in domain if value >= literal]
        if not candidates:
            return TokenBuild(None, zero=True, missing_column=predicate.column)
        return _range_token_or_wildcard(domain, PredicateOp.GREATER_EQUAL, min(candidates))
    if op == PredicateOp.LESS_THAN:
        candidates = [value for value in domain if value < literal]
        if not candidates:
            return TokenBuild(None, zero=True, missing_column=predicate.column)
        return _range_token_or_wildcard(domain, PredicateOp.LESS_EQUAL, max(candidates))
    if op == PredicateOp.LESS_EQUAL:
        candidates = [value for value in domain if value <= literal]
        if not candidates:
            return TokenBuild(None, zero=True, missing_column=predicate.column)
        return _range_token_or_wildcard(domain, PredicateOp.LESS_EQUAL, max(candidates))
    return TokenBuild(None, unsupported=f"unsupported_op:{predicate.op}")


def build_normalized_token(
    metadata: ModelMetadata,
    column_name: str,
    predicates: list[RawPredicate],
) -> TokenBuild:
    """Canonicalize all SQL predicates on a column against model domains."""

    tokens = []
    missing = []
    for predicate in predicates:
        built = build_token(metadata, predicate)
        if built.unsupported:
            return built
        if built.zero:
            missing.append(predicate.column)
            continue
        if built.token is not None:
            tokens.append(built.token)
    if missing:
        return TokenBuild(None, zero=True, missing_column=column_name)
    if not tokens:
        return TokenBuild(PredicateToken.wildcard(), all_values=True)
    try:
        normalized = ColumnPredicateSet(tuple(tokens)).normalize(max_predicates=2)
    except (TypeError, ValueError) as exc:
        return TokenBuild(None, unsupported=f"{column_name}:{exc}")
    if normalized.contradiction:
        return TokenBuild(None, zero=True, missing_column=column_name)
    return TokenBuild(normalized.output_token())


def raw_predicate_token(predicate: RawPredicate) -> PredicateToken:
    op = OP_MAP[predicate.op]
    return PredicateToken(op, value=predicate.literal)


def _range_token_or_wildcard(
    domain: list[Any],
    op: PredicateOp,
    threshold: Any,
) -> TokenBuild:
    if op == PredicateOp.GREATER_EQUAL and all(value >= threshold for value in domain):
        return TokenBuild(None, all_values=True)
    if op == PredicateOp.LESS_EQUAL and all(value <= threshold for value in domain):
        return TokenBuild(None, all_values=True)
    return TokenBuild(PredicateToken(op, value=threshold))


def lower_complement(predicate: RawPredicate) -> RawPredicate:
    if predicate.op == ">":
        return RawPredicate(predicate.column, "<=", predicate.literal, f"{predicate.column}<={predicate.literal}")
    if predicate.op == ">=":
        return RawPredicate(predicate.column, "<", predicate.literal, f"{predicate.column}<{predicate.literal}")
    raise ValueError(predicate)


def eval_query(
    estimator: OnePassEstimator,
    wrapped: TorchDistributionModel,
    model: Any,
    vocabularies: PredicateVocabularies,
    metadata: ModelMetadata,
    included_tables: set[str],
    predicates: list[RawPredicate],
) -> tuple[str, float, int, float, float, str, str, str, np.ndarray, int, list[str], list[str]]:
    return eval_query_native(
        wrapped,
        model,
        vocabularies,
        metadata,
        included_tables,
        predicates,
    )


def eval_query_native(
    wrapped: TorchDistributionModel,
    model: Any,
    vocabularies: PredicateVocabularies,
    metadata: ModelMetadata,
    included_tables: set[str],
    predicates: list[RawPredicate],
) -> tuple[str, float, int, float, float, str, str, str, np.ndarray, int, list[str], list[str]]:
    import torch
    from model.src.model.output_adapter import TorchBackboneOutputs

    by_col: dict[str, list[RawPredicate]] = {}
    for predicate in predicates:
        by_col.setdefault(predicate.column, []).append(predicate)
    output_tokens: dict[str, PredicateToken] = {}
    conditioning_tokens: dict[str, PredicateToken] = {}
    missing = []
    unsupported = []
    native_range_count = 0
    for column_name, col_preds in by_col.items():
        try:
            column = metadata.columns[metadata.column_index(column_name)]
        except KeyError:
            unsupported.append(f"unknown_column:{column_name}")
            continue
        if column.kind != ColumnKind.DATA:
            unsupported.append(f"non_data_predicate:{column_name}")
            continue
        if any(predicate.op == "=" for predicate in col_preds):
            for predicate in col_preds:
                if predicate.op == "=":
                    canonical = matching_domain_value(column.domain, predicate.literal)
                    if canonical is _MISSING:
                        missing.append(column_name)
                    break
        built_normalized = build_normalized_token(metadata, column_name, col_preds)
        if built_normalized.unsupported:
            unsupported.append(built_normalized.unsupported)
            continue
        if built_normalized.zero:
            missing.append(column_name)
            continue
        assert built_normalized.token is not None
        normalized = ColumnPredicateSet((built_normalized.token,)).normalize(max_predicates=2)
        if normalized.contradiction:
            zero_factors = np.zeros(len(metadata.columns), dtype=float)
            return (
                "zero_due_to_contradiction",
                0.0,
                0,
                0.0,
                0.0,
                ",".join(sorted(inverse_fanouts_for_table_subset(metadata, included_tables))),
                "",
                column_name,
                zero_factors,
                int(normalized.is_native_range),
                sorted(by_col),
                sorted(predicate.op for predicate in predicates),
            )
        output_token = normalized.output_token()
        if output_token.op != PredicateOp.WILDCARD:
            output_tokens[column_name] = output_token
            conditioning_tokens[column_name] = output_token
        if normalized.is_native_range:
            native_range_count += 1
    if unsupported:
        return _empty_eval("unsupported", ";".join(sorted(set(unsupported))), metadata, predicates, included_tables)
    if missing:
        return _empty_eval("zero_due_to_missing_domain", ";".join(sorted(set(missing))), metadata, predicates, included_tables)

    inverse_fanouts = set(inverse_fanouts_for_table_subset(metadata, included_tables))
    tokens = tokens_for_query_tables(metadata, included_tables, inverse_fanouts, conditioning_tokens)
    before = model.forward_calls
    start = perf_counter()
    token_ids = encode_tokens_tensor([tokens], vocabularies, device=wrapped.device)
    model.eval()
    with torch.no_grad():
        backbone_start = perf_counter()
        logits = model(token_ids)
        wrapped.last_backbone_seconds = perf_counter() - backbone_start
        outputs = TorchBackboneOutputs(
            logits=logits,
            split_logits=model.split_head_outputs(logits),
            output_embeddings=(
                [embedding.weight for embedding in model.output_embeddings]
                if getattr(model.config, "output_encoding", "one_hot") == "embed"
                else None
            ),
        )
        decode_start = perf_counter()
        factor_values = []
        for column_index, column in enumerate(metadata.columns):
            output_token = output_tokens.get(column.name, tokens[column_index])
            if output_token.op == PredicateOp.RANGE:
                if metadata.factorization_plan.enabled:
                    factor = wrapped.output_adapter.interval_mass(
                        original_column_index=column_index,
                        backbone_outputs=outputs,
                        lower_literal=output_token.value,
                        upper_literal=output_token.upper,
                        lower_inclusive=output_token.lower_inclusive,
                        upper_inclusive=output_token.upper_inclusive,
                    )
                else:
                    factor = wrapped.output_adapter.interval_mass(
                        original_column_index=column_index,
                        backbone_outputs=outputs,
                        metadata=metadata,
                        lower_literal=output_token.value,
                        upper_literal=output_token.upper,
                        lower_inclusive=output_token.lower_inclusive,
                        upper_inclusive=output_token.upper_inclusive,
                    )
            elif metadata.factorization_plan.enabled:
                factor = wrapped.output_adapter.column_factor(
                    original_column_index=column_index,
                    backbone_outputs=outputs,
                    predicate_token=output_token,
                )
            else:
                factor = wrapped.output_adapter.column_factor(
                    original_column_index=column_index,
                    backbone_outputs=outputs,
                    metadata=metadata,
                    predicate_token=output_token,
                )
            factor_values.append(float(factor[0].detach().cpu()))
        wrapped.last_decode_seconds = perf_counter() - decode_start
    factors = np.array(factor_values, dtype=float)
    calls = model.forward_calls - before
    if metadata.full_join_cardinality == 0 or np.any(factors == 0):
        estimate = 0.0
    else:
        estimate = exp(log(metadata.full_join_cardinality) + float(np.sum(np.log(factors))))
    wall = perf_counter() - start
    if estimate < 0 or not np.isfinite(estimate):
        raise ValueError(f"invalid cardinality estimate {estimate!r}")
    status = "ok_native_range" if native_range_count else "ok"
    model_latency = float((wrapped.last_backbone_seconds or 0.0) + (wrapped.last_decode_seconds or 0.0))
    return (
        status,
        estimate,
        calls,
        wall,
        model_latency,
        ",".join(sorted(inverse_fanouts)),
        "",
        "",
        factors,
        native_range_count,
        sorted(by_col),
        sorted(predicate.op for predicate in predicates),
    )


def _empty_eval(
    status: str,
    detail: str,
    metadata: ModelMetadata,
    predicates: list[RawPredicate],
    included_tables: set[str],
) -> tuple[str, float, int, float, float, str, str, str, np.ndarray, int, list[str], list[str]]:
    return (
        status,
        0.0,
        0,
        0.0,
        0.0,
        ",".join(sorted(inverse_fanouts_for_table_subset(metadata, included_tables))),
        "",
        detail,
        np.zeros(len(metadata.columns), dtype=float),
        0,
        sorted({predicate.column for predicate in predicates}),
        sorted(predicate.op for predicate in predicates),
    )


def build_ordinary(
    metadata: ModelMetadata,
    predicates: list[RawPredicate],
) -> tuple[dict[str, PredicateToken], list[str], list[str]]:
    ordinary = {}
    missing = []
    unsupported = []
    for predicate in predicates:
        built = build_token(metadata, predicate)
        if built.unsupported:
            unsupported.append(predicate.column)
        elif built.zero:
            missing.append(predicate.column)
        elif built.token is not None:
            if predicate.column in ordinary:
                unsupported.append(f"multiple_predicates:{predicate.column}")
                continue
            ordinary[predicate.column] = built.token
    return ordinary, missing, unsupported


def estimate_once(
    estimator: OnePassEstimator,
    wrapped: TorchDistributionModel,
    model: Any,
    metadata: ModelMetadata,
    included_tables: set[str],
    ordinary: dict[str, PredicateToken],
) -> tuple[float, int, float, float, set[str]]:
    inverse_fanouts = set(inverse_fanouts_for_table_subset(metadata, included_tables))
    tokens = tokens_for_query_tables(metadata, included_tables, inverse_fanouts, ordinary)
    before = model.forward_calls
    result = estimator.estimate(tokens, use_log_space_product=True)
    calls = model.forward_calls - before
    model_latency = float((wrapped.last_backbone_seconds or 0.0) + (wrapped.last_decode_seconds or 0.0))
    return result.estimated_cardinality, calls, float(result.latency_seconds), model_latency, inverse_fanouts


def summarize(
    rows: list[dict[str, Any]],
    checkpoint: Path,
    queries_path: Path,
    results_path: Path,
    summary_path: Path,
    model: Any,
    elapsed: float,
) -> dict[str, Any]:
    scored = [
        row for row in rows
        if row["status"] in {"ok", "ok_native_range", "zero_due_to_missing_domain", "zero_due_to_contradiction"}
    ]
    ok_rows = [row for row in rows if row["status"] in {"ok", "ok_native_range"}]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    all_q = [float(row["q_error"]) for row in scored]
    all_q_floor_one = [float(row["q_error_floor_one"]) for row in scored]
    estimates = [float(row["estimated_cardinality"]) for row in scored]
    ok_q = [float(row["q_error"]) for row in ok_rows]
    native_q = [float(row["q_error"]) for row in rows if int(row["native_range_count"]) > 0]
    normal_q = [float(row["q_error"]) for row in rows if int(row["native_range_count"]) == 0 and row["status"] == "ok"]
    parameter_size = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    estimate_lt_1_count = sum(1 for value in estimates if value < 1.0)
    estimate_lt_0_1_count = sum(1 for value in estimates if value < 0.1)
    estimate_lt_0_01_count = sum(1 for value in estimates if value < 0.01)
    sub_one_estimate_queries = [
        {
            "query_id": row["query_id"],
            "true_cardinality": row["true_cardinality"],
            "estimated_cardinality": row["estimated_cardinality"],
            "raw_q_error": row["q_error"],
            "floor_one_q_error": row["q_error_floor_one"],
            "tables": row["tables"],
            "filters": row["filters"],
            "smallest_column_factor": row.get("smallest_column_factor"),
            "smallest_column_factor_index": row.get("smallest_column_factor_index"),
            "smallest_column_factor_name": row.get("smallest_column_factor_name"),
        }
        for row in scored
        if float(row["estimated_cardinality"]) < 1.0
    ]
    sub_one_estimate_queries.sort(
        key=lambda row: (
            float(row["estimated_cardinality"]),
            -float(row["raw_q_error"]),
        )
    )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_size_mib": checkpoint.stat().st_size / (1024 * 1024),
        "evaluation_wall_seconds": elapsed,
        "mean_model_latency_seconds": _mean([float(row["model_latency_seconds"]) for row in rows]),
        "mean_query_wall_seconds": _mean([float(row["query_wall_seconds"]) for row in rows]),
        "median_model_latency_seconds": _percentile([float(row["model_latency_seconds"]) for row in rows], 50),
        "median_query_wall_seconds": _percentile([float(row["query_wall_seconds"]) for row in rows], 50),
        "total_model_latency_seconds": sum(float(row["model_latency_seconds"]) for row in rows),
        "total_query_wall_seconds": sum(float(row["query_wall_seconds"]) for row in rows),
        "total_model_forward_calls": sum(int(row["model_forward_calls"]) for row in rows),
        "parameter_count": model.parameter_count(),
        "parameter_size_bytes": int(parameter_size),
        "parameter_size_mib": parameter_size / (1024 * 1024),
        "queries_path": str(queries_path),
        "queries_total": len(rows),
        "queries_scored": len(scored),
        "queries_ok_or_inclusion_exclusion": len(ok_rows),
        "queries_ok_or_native_range": len(ok_rows),
        "zero_estimate_count": sum(1 for row in rows if str(row["zero_estimate"]) == "True"),
        "estimate_lt_1_count": estimate_lt_1_count,
        "estimate_lt_0_1_count": estimate_lt_0_1_count,
        "estimate_lt_0_01_count": estimate_lt_0_01_count,
        "estimate_lt_1_fraction": estimate_lt_1_count / max(len(scored), 1),
        "estimate_lt_0_1_fraction": estimate_lt_0_1_count / max(len(scored), 1),
        "estimate_lt_0_01_fraction": estimate_lt_0_01_count / max(len(scored), 1),
        "results_path": str(results_path),
        "status_counts": status_counts,
        "summary_path": str(summary_path),
        "all_scored_median_q_error": _percentile(all_q, 50),
        "all_scored_p90_q_error": _percentile(all_q, 90),
        "all_scored_p95_q_error": _percentile(all_q, 95),
        "all_scored_p99_q_error": _percentile(all_q, 99),
        "all_scored_max_q_error": max(all_q) if all_q else None,
        "all_scored_median_q_error_floor_one": _percentile(all_q_floor_one, 50),
        "all_scored_p90_q_error_floor_one": _percentile(all_q_floor_one, 90),
        "all_scored_p95_q_error_floor_one": _percentile(all_q_floor_one, 95),
        "all_scored_p99_q_error_floor_one": _percentile(all_q_floor_one, 99),
        "all_scored_max_q_error_floor_one": max(all_q_floor_one) if all_q_floor_one else None,
        "raw": {
            "median": _percentile(all_q, 50),
            "p90": _percentile(all_q, 90),
            "p95": _percentile(all_q, 95),
            "p99": _percentile(all_q, 99),
            "max": max(all_q) if all_q else None,
        },
        "floor_one": {
            "median": _percentile(all_q_floor_one, 50),
            "p90": _percentile(all_q_floor_one, 90),
            "p95": _percentile(all_q_floor_one, 95),
            "p99": _percentile(all_q_floor_one, 99),
            "max": max(all_q_floor_one) if all_q_floor_one else None,
        },
        "ok_median_q_error": _percentile(ok_q, 50),
        "ok_p90_q_error": _percentile(ok_q, 90),
        "ok_p95_q_error": _percentile(ok_q, 95),
        "ok_max_q_error": max(ok_q) if ok_q else None,
        "native_range_median_q_error": _percentile(native_q, 50),
        "native_range_p95_q_error": _percentile(native_q, 95),
        "native_range_max_q_error": max(native_q) if native_q else None,
        "normal_median_q_error": _percentile(normal_q, 50),
        "normal_p95_q_error": _percentile(normal_q, 95),
        "normal_max_q_error": max(normal_q) if normal_q else None,
        "worst_20_queries": sorted(
            rows,
            key=lambda row: float(row["q_error"]),
            reverse=True,
        )[:20],
        "sub_one_estimate_queries": sub_one_estimate_queries,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.array(values, dtype=float))) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.array(values, dtype=float), q)) if values else None


if __name__ == "__main__":
    main()
