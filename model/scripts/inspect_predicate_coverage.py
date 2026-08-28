from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.generation import inverse_fanouts_for_table_subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare training predicate-token coverage with JOB-light query requirements."
    )
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = json.loads(Path(args.training_summary).read_text(encoding="utf-8"))
    training_coverage = summary.get("predicate_token_coverage", {})
    from model.src.model.checkpoint import load_resmade_checkpoint

    _model, payload = load_resmade_checkpoint(Path(args.checkpoint), map_location="cpu")
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    requirements = evaluation_token_requirements(Path(args.queries), metadata)
    missing = unseen_required_token_types(training_coverage, requirements)
    report = {
        "training_summary": args.training_summary,
        "checkpoint": args.checkpoint,
        "queries": args.queries,
        "evaluation_token_requirements": requirements,
        "columns_with_unseen_evaluation_token_types": missing,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


def evaluation_token_requirements(
    queries_path: Path,
    metadata: ModelMetadata,
) -> dict[str, dict[str, int]]:
    """Count token types required by a JOB-light workload without model calls."""

    from model.scripts.evaluate_job_light_queries import build_token, parse_query

    counts = {
        column.name: {
            "wildcard": 0,
            "equal": 0,
            "less_than": 0,
            "less_equal": 0,
            "greater_than": 0,
            "greater_equal": 0,
            "range": 0,
            "indicator_equal_1": 0,
            "indicator_wildcard": 0,
            "fanout_inv": 0,
            "fanout_wildcard": 0,
        }
        for column in metadata.columns
    }
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        included, predicates, _true_cardinality = parse_query(line)
        predicate_columns: set[str] = set()
        for predicate in predicates:
            built = build_token(metadata, predicate)
            if built.token is not None:
                counts[predicate.column][built.token.op.value] += 1
                predicate_columns.add(predicate.column)
        inverse_fanouts = inverse_fanouts_for_table_subset(metadata, included)
        for column in metadata.columns:
            if column.kind == ColumnKind.DATA and column.name not in predicate_columns:
                counts[column.name]["wildcard"] += 1
            elif column.kind == ColumnKind.INDICATOR:
                if column.table in included:
                    counts[column.name]["indicator_equal_1"] += 1
                else:
                    counts[column.name]["indicator_wildcard"] += 1
            elif column.kind == ColumnKind.FANOUT:
                if column.name in inverse_fanouts:
                    counts[column.name]["fanout_inv"] += 1
                else:
                    counts[column.name]["fanout_wildcard"] += 1
    return counts


def unseen_required_token_types(
    training_coverage: dict[str, dict[str, Any]],
    requirements: dict[str, dict[str, int]],
) -> list[str]:
    """Return ``column:token_type`` entries required by evaluation but unseen in training."""

    missing = []
    for column_name, required_counts in requirements.items():
        trained_counts = training_coverage.get(column_name, {})
        for token_type, required_count in required_counts.items():
            if required_count <= 0:
                continue
            if int(trained_counts.get(token_type, 0) or 0) == 0:
                missing.append(f"{column_name}:{token_type}")
    return sorted(missing)


if __name__ == "__main__":
    main()
