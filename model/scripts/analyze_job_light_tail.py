from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from model.src.data.schema import ColumnKind, ModelMetadata


Q_ERROR_KEYS = (
    "all_scored_median_q_error",
    "all_scored_p90_q_error",
    "all_scored_p95_q_error",
    "all_scored_p99_q_error",
    "all_scored_max_q_error",
    "zero_estimate_count",
    "evaluation_wall_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze JOB-light tail behavior from evaluator CSV/JSON outputs."
    )
    parser.add_argument("--results", help="JOB-light per-query results CSV.")
    parser.add_argument("--checkpoint", help="Checkpoint used for the results CSV.")
    parser.add_argument("--training-metrics", help="training_metrics.jsonl for correlations.")
    parser.add_argument("--sweep-dir", help="Directory containing checkpoint sweep summaries.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(Path(args.checkpoint)) if args.checkpoint else None
    if args.results:
        rows = load_rows(Path(args.results))
        grouped = grouped_summaries(rows, metadata)
        worst = worst_query_attribution(rows, metadata)
        write_json(output_dir / "grouped_q_error_summary.json", grouped)
        write_json(output_dir / "worst_20_tail_attribution.json", worst)
        write_grouped_csv(output_dir / "grouped_q_error_summary.csv", grouped)

    if args.sweep_dir:
        sweep = checkpoint_sweep_summary(
            Path(args.sweep_dir),
            Path(args.training_metrics) if args.training_metrics else None,
        )
        write_json(output_dir / "checkpoint_sweep_summary.json", sweep)
        write_sweep_csv(output_dir / "checkpoint_sweep_summary.csv", sweep)


def load_metadata(checkpoint: Path) -> ModelMetadata:
    from model.src.model.checkpoint import load_resmade_checkpoint

    _model, payload = load_resmade_checkpoint(checkpoint, map_location="cpu")
    return ModelMetadata.from_json_dict(payload["metadata"])


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in [
            "query_id",
            "true_cardinality",
            "number_of_predicates",
            "number_of_tables",
            "native_range_count",
            "model_forward_calls",
        ]:
            row[key] = int(row[key])
        for key in [
            "estimated_cardinality",
            "q_error",
            "q_error_floor_one",
            "model_latency_seconds",
            "query_wall_seconds",
        ]:
            row[key] = float(row[key])
        row["zero_estimate"] = str(row["zero_estimate"]).lower() == "true"
    return rows


def grouped_summaries(
    rows: list[dict[str, Any]],
    metadata: ModelMetadata | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in scored_rows(rows):
        columns = split_list(row["predicate_columns"])
        operators = split_list(row["predicate_operators"])
        grouped["predicate_type"][predicate_type(operators)].append(row)
        grouped["factorized_predicate_columns"][
            factorized_predicate_group(columns, metadata)
        ].append(row)
        for column in columns or ["<none>"]:
            grouped["predicate_column"][column].append(row)
        grouped["true_cardinality_bucket"][true_cardinality_bucket(row["true_cardinality"])].append(row)
        grouped["number_of_predicates"][str(row["number_of_predicates"])].append(row)
        grouped["number_of_tables"][str(row["number_of_tables"])].append(row)
        grouped["inverse_fanout_pattern"][row.get("inverse_fanouts") or "<none>"].append(row)
    return {
        group_name: [
            {"group": key, **q_error_summary(group_rows)}
            for key, group_rows in sorted(groups.items())
        ]
        for group_name, groups in sorted(grouped.items())
    }


def worst_query_attribution(
    rows: list[dict[str, Any]],
    metadata: ModelMetadata | None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    worst = sorted(scored_rows(rows), key=lambda row: row["q_error"], reverse=True)[:limit]
    output = []
    for row in worst:
        columns = split_list(row["predicate_columns"])
        operators = split_list(row["predicate_operators"])
        factor_details = per_column_details(row, metadata)
        output.append(
            {
                "query_id": row["query_id"],
                "status": row["status"],
                "tables": row["tables"],
                "filters": row["filters"],
                "true_cardinality": row["true_cardinality"],
                "estimated_cardinality": row["estimated_cardinality"],
                "q_error": row["q_error"],
                "predicate_columns": columns,
                "predicate_operators": operators,
                "factorized_predicate_columns": {
                    column: is_factorized_column(column, metadata) for column in columns
                },
                "inverse_fanouts": split_list(row.get("inverse_fanouts", "")),
                "dominant_error_pattern": dominant_factor_pattern(factor_details),
                "per_column_factors": factor_details,
            }
        )
    return output


def per_column_details(
    row: dict[str, Any],
    metadata: ModelMetadata | None,
) -> list[dict[str, Any]]:
    factors = split_floats(row.get("per_column_factors", ""))
    log_factors = split_floats(row.get("log_per_column_factors", ""))
    details = []
    for index, factor in enumerate(factors):
        column_name = f"column_{index}"
        kind = None
        factorized = False
        if metadata is not None and index < len(metadata.columns):
            column = metadata.columns[index]
            column_name = column.name
            kind = column.kind.value
            factorized = (
                metadata.factorization_plan.factorization_for_column(index) is not None
            )
        details.append(
            {
                "column": column_name,
                "kind": kind,
                "factorized": factorized,
                "factor": factor,
                "log_factor": log_factors[index] if index < len(log_factors) else None,
            }
        )
    return details


def dominant_factor_pattern(details: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [
        item
        for item in details
        if item["log_factor"] is not None and math.isfinite(float(item["log_factor"]))
    ]
    weights = [abs(float(item["log_factor"])) for item in finite]
    total = sum(weights)
    if not finite or total == 0.0:
        return {"classification": "undetermined", "dominant_column": None, "share": None}
    max_index = max(range(len(finite)), key=lambda index: weights[index])
    share = weights[max_index] / total
    classification = (
        "one_catastrophic_factor"
        if share >= 0.5 and weights[max_index] >= 5.0
        else "several_moderate_factors"
    )
    return {
        "classification": classification,
        "dominant_column": finite[max_index]["column"],
        "dominant_log_factor_abs": weights[max_index],
        "dominant_share_of_abs_log_mass": share,
    }


def checkpoint_sweep_summary(
    sweep_dir: Path,
    training_metrics_path: Path | None,
) -> dict[str, Any]:
    metrics_by_step = load_training_metrics(training_metrics_path)
    checkpoints = []
    for summary_path in sorted(sweep_dir.glob("*summary*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        step = checkpoint_step(summary.get("checkpoint", "") or summary_path.name)
        row = {
            "step": step,
            "summary_path": str(summary_path),
            "results_path": summary.get("results_path"),
        }
        for key in Q_ERROR_KEYS:
            row[key] = summary.get(key)
        if step in metrics_by_step:
            row.update(metrics_by_step[step])
        checkpoints.append(row)
    checkpoints = sorted(checkpoints, key=lambda row: row["step"] or -1)
    return {
        "checkpoints": checkpoints,
        "best_by_median": best_checkpoint(checkpoints, "all_scored_median_q_error"),
        "best_by_p95": best_checkpoint(checkpoints, "all_scored_p95_q_error"),
        "best_by_max": best_checkpoint(checkpoints, "all_scored_max_q_error"),
        "spearman_correlations": spearman_correlations(checkpoints),
    }


def load_training_metrics(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    by_step: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        metric = json.loads(line)
        step = int(metric["step"])
        by_step[step] = {
            key: float(metric[key])
            for key in [
                "loss",
                "validation_nll",
                "validation_weighted_nll",
                "validation_indicator_nll",
                "validation_ordinary_nll",
            ]
            if key in metric and metric[key] is not None
        }
    return by_step


def spearman_correlations(checkpoints: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    metric_keys = sorted(
        {
            key
            for row in checkpoints
            for key, value in row.items()
            if key.startswith("validation_") or key == "loss"
            if isinstance(value, (int, float))
        }
    )
    target_keys = [
        "all_scored_median_q_error",
        "all_scored_p95_q_error",
        "all_scored_max_q_error",
    ]
    output: dict[str, dict[str, float | None]] = {}
    for metric in metric_keys:
        output[metric] = {}
        for target in target_keys:
            pairs = [
                (float(row[metric]), float(row[target]))
                for row in checkpoints
                if isinstance(row.get(metric), (int, float))
                and isinstance(row.get(target), (int, float))
            ]
            output[metric][target] = spearman(pairs)
    return output


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    left, right = zip(*pairs)
    return pearson(rank(list(left)), rank(list(right)))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    left_mean = mean(left)
    right_mean = mean(right)
    num = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    den_left = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    den_right = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if den_left == 0.0 or den_right == 0.0:
        return None
    return num / (den_left * den_right)


def best_checkpoint(checkpoints: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in checkpoints if isinstance(row.get(key), (int, float))]
    if not candidates:
        return None
    row = min(candidates, key=lambda item: float(item[key]))
    return {"step": row["step"], key: row[key], "summary_path": row["summary_path"]}


def q_error_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([float(row["q_error"]) for row in rows], dtype=float)
    return {
        "count": int(values.size),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(values.max()) if values.size else None,
        "zero_estimates": sum(1 for row in rows if row["zero_estimate"]),
        "mean_query_wall_seconds": (
            float(np.mean([row["query_wall_seconds"] for row in rows])) if rows else None
        ),
    }


def percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def scored_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["status"]
        in {"ok", "ok_native_range", "zero_due_to_missing_domain", "zero_due_to_contradiction"}
    ]


def predicate_type(operators: list[str]) -> str:
    has_range = any(operator in {"<", "<=", ">", ">="} for operator in operators)
    has_equal = any(operator == "=" for operator in operators)
    if has_equal and has_range:
        return "mixed_equality_range"
    if has_range:
        return "range_only"
    if has_equal:
        return "equality_only"
    return "no_predicate"


def factorized_predicate_group(
    columns: list[str],
    metadata: ModelMetadata | None,
) -> str:
    if not columns:
        return "no_predicate"
    states = [is_factorized_column(column, metadata) for column in columns]
    if all(states):
        return "all_factorized"
    if any(states):
        return "mixed_factorized_unfactorized"
    return "all_unfactorized"


def is_factorized_column(column: str, metadata: ModelMetadata | None) -> bool:
    if metadata is None:
        return False
    try:
        index = metadata.column_index(column)
    except KeyError:
        return False
    return metadata.factorization_plan.factorization_for_column(index) is not None


def true_cardinality_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    for upper in [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]:
        if value <= upper:
            return f"1..{upper}"
    return ">10000000"


def split_list(value: str) -> list[str]:
    return [item for item in str(value).split(",") if item]


def split_floats(value: str) -> list[float]:
    output = []
    for item in str(value).split(";"):
        if not item:
            continue
        output.append(float("-inf") if item == "-inf" else float(item))
    return output


def checkpoint_step(text: str) -> int | None:
    match = re.search(r"checkpoint_step_(\d+)\.pt", text)
    return int(match.group(1)) if match else None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_grouped_csv(path: Path, grouped: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["group_type", "group", "count", "median", "p90", "p95", "p99", "max", "zero_estimates", "mean_query_wall_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group_type, rows in grouped.items():
            for row in rows:
                writer.writerow({"group_type": group_type, **row})


def write_sweep_csv(path: Path, sweep: dict[str, Any]) -> None:
    rows = sweep.get("checkpoints", [])
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
