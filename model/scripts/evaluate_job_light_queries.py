from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.evaluation.metrics import q_error
from model.src.inference.estimator import OnePassEstimator
from model.src.inference.torch_estimator import TorchDistributionModel
from model.src.model.checkpoint import load_resmade_checkpoint
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.operators import PredicateOp, PredicateToken
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

    model, payload = load_resmade_checkpoint(checkpoint, map_location="cpu")
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    vocabularies = PredicateVocabularies.from_json_dict(payload["predicate_vocabularies"])
    wrapped = TorchDistributionModel(model, metadata, vocabularies)
    estimator = OnePassEstimator(wrapped, metadata)

    rows = []
    start = perf_counter()
    for query_id, line in enumerate(queries_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        included, predicates, true_cardinality = parse_query(line)
        status, estimate, calls, query_wall, model_latency, inverse, branches, detail = eval_query(
            estimator,
            wrapped,
            model,
            metadata,
            included,
            predicates,
        )
        qe = q_error(estimate, true_cardinality)
        rows.append(
            {
                "query_id": query_id,
                "status": status,
                "tables": ",".join(sorted(included)),
                "filters": ";".join(predicate.text for predicate in predicates),
                "true_cardinality": true_cardinality,
                "estimated_cardinality": estimate,
                "q_error": qe,
                "model_latency_seconds": model_latency,
                "query_wall_seconds": query_wall,
                "model_forward_calls": calls,
                "inverse_fanouts": inverse,
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
    metadata: ModelMetadata,
    included_tables: set[str],
    predicates: list[RawPredicate],
) -> tuple[str, float, int, float, float, str, str, str]:
    by_col: dict[str, list[RawPredicate]] = {}
    for predicate in predicates:
        by_col.setdefault(predicate.column, []).append(predicate)
    two_sided = None
    for column, col_preds in by_col.items():
        lowers = [predicate for predicate in col_preds if predicate.op in {">", ">="}]
        uppers = [predicate for predicate in col_preds if predicate.op in {"<", "<="}]
        if lowers and uppers:
            two_sided = (column, lowers[0], uppers[0])
            break

    if two_sided is None:
        ordinary, missing, unsupported = build_ordinary(metadata, predicates)
        if unsupported:
            return "unsupported", 0.0, 0, 0.0, 0.0, "", "", ";".join(sorted(set(unsupported)))
        if missing:
            return "zero_due_to_missing_domain", 0.0, 0, 0.0, 0.0, "", "", ";".join(sorted(set(missing)))
        estimate, calls, wall, model_latency, inverse = estimate_once(
            estimator, wrapped, model, metadata, included_tables, ordinary
        )
        return "ok", estimate, calls, wall, model_latency, ",".join(sorted(inverse)), "", ""

    two_col, lower, upper = two_sided
    base_predicates = [predicate for predicate in predicates if predicate.column != two_col or predicate is upper]
    ordinary, missing, unsupported = build_ordinary(metadata, base_predicates)
    if unsupported:
        return "unsupported", 0.0, 0, 0.0, 0.0, "", "", ";".join(sorted(set(unsupported)))
    if missing:
        return "zero_due_to_missing_domain", 0.0, 0, 0.0, 0.0, "", "", ";".join(sorted(set(missing)))
    upper_est, upper_calls, upper_wall, upper_model, inverse = estimate_once(
        estimator, wrapped, model, metadata, included_tables, ordinary
    )

    subtract_pred = lower_complement(lower)
    subtract_ordinary = dict(ordinary)
    built = build_token(metadata, subtract_pred)
    if built.unsupported:
        return "unsupported", 0.0, upper_calls, upper_wall, upper_model, ",".join(sorted(inverse)), "", built.unsupported
    if built.zero:
        lower_est = 0.0
        lower_calls = 0
        lower_wall = 0.0
        lower_model = 0.0
    else:
        if built.token is None:
            subtract_ordinary.pop(two_col, None)
        else:
            subtract_ordinary[two_col] = built.token
        lower_est, lower_calls, lower_wall, lower_model, _ = estimate_once(
            estimator, wrapped, model, metadata, included_tables, subtract_ordinary
        )
    estimate = max(0.0, upper_est - lower_est)
    branches = f"upper={upper_est};lower={lower_est}"
    return (
        "ok_inclusion_exclusion",
        estimate,
        upper_calls + lower_calls,
        upper_wall + lower_wall,
        upper_model + lower_model,
        ",".join(sorted(inverse)),
        branches,
        "",
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
    inverse_fanouts = {
        column.name
        for column in metadata.columns
        if column.kind == ColumnKind.FANOUT and column.table not in included_tables
    }
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
        if row["status"] in {"ok", "ok_inclusion_exclusion", "zero_due_to_missing_domain"}
    ]
    ok_rows = [row for row in rows if row["status"] in {"ok", "ok_inclusion_exclusion"}]
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    all_q = [float(row["q_error"]) for row in scored]
    ok_q = [float(row["q_error"]) for row in ok_rows]
    parameter_size = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
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
        "results_path": str(results_path),
        "status_counts": status_counts,
        "summary_path": str(summary_path),
        "all_scored_median_q_error": _percentile(all_q, 50),
        "all_scored_p90_q_error": _percentile(all_q, 90),
        "all_scored_p95_q_error": _percentile(all_q, 95),
        "all_scored_p99_q_error": _percentile(all_q, 99),
        "all_scored_max_q_error": max(all_q) if all_q else None,
        "ok_median_q_error": _percentile(ok_q, 50),
        "ok_p90_q_error": _percentile(ok_q, 90),
        "ok_p95_q_error": _percentile(ok_q, 95),
        "ok_max_q_error": max(ok_q) if ok_q else None,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(np.array(values, dtype=float))) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.array(values, dtype=float), q)) if values else None


if __name__ == "__main__":
    main()
