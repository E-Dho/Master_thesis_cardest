from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import _load_segment_ids_array, _segment_id_rows_to_tuples
from model.src.data.schema import ModelMetadata
from model.src.data.trajectory_distinct import TrajectoryDistinctRuntimeConfig
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.evaluation.pol_query_adapter import (
    assert_checkpoint_trajectory_config_compatible,
    evaluate_pol_distinct_record,
)
from model.src.inference.estimator import OnePassEstimator
from model.src.predicates.vocabulary import PredicateVocabularies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicate-conditioned ResMADE on structured POL distinct queries."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queries", required=True, help="JSONL produced by query_generation/")
    parser.add_argument("--output", required=True, help="Per-query JSONL result path")
    parser.add_argument(
        "--exact-fixture-dir",
        default=None,
        help=(
            "Optional prepared POL fixture directory for fixture-only semantic "
            "counts; production q-error uses workload database truth only."
        ),
    )
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    try:
        from model.src.inference.torch_estimator import TorchDistributionModel
        from model.src.model.checkpoint import load_resmade_checkpoint
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    model, payload = load_resmade_checkpoint(args.checkpoint, map_location="cpu")
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    vocabularies = PredicateVocabularies.from_json_dict(
        payload["predicate_vocabularies"],
        metadata,
    )
    wrapped = TorchDistributionModel(model, metadata, vocabularies)
    estimator = OnePassEstimator(wrapped, metadata)
    oracle, trajectory_ids, segment_ids = _load_exact_fixture(args, config, metadata)
    runtime_config = TrajectoryDistinctRuntimeConfig.from_dict(
        config.get("trajectory_distinct", {})
    )
    assert_checkpoint_trajectory_config_compatible(
        payload,
        runtime_config,
        config.get("trajectory_spatial", {}),
    )
    queries_path = Path(args.queries)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    database_truth_status_counts: dict[str, int] = {}
    database_truth_count = 0
    missing_database_truth_count = 0
    supported_count = 0
    unsupported_count = 0
    matching_qerrors: list[float] = []
    distinct_qerrors: list[float] = []
    a_abs_errors: list[float] = []
    total = 0
    with queries_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            result = evaluate_pol_distinct_record(
                record,
                metadata=metadata,
                estimator=estimator,
                oracle=oracle,
                trajectory_ids=trajectory_ids,
                segment_ids=segment_ids,
                trajectory_config=runtime_config,
                trajectory_spatial=config.get("trajectory_spatial", {}),
            )
            payload = result.to_json_dict()
            sink.write(json.dumps(payload, sort_keys=True) + "\n")
            total += 1
            status = str(payload["distinct_estimate_status"])
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            truth_status = str(payload.get("database_truth_status", "missing_database_truth"))
            database_truth_status_counts[truth_status] = int(
                database_truth_status_counts.get(truth_status, 0)
            ) + 1
            if truth_status == "database_truth_available":
                database_truth_count += 1
            else:
                missing_database_truth_count += 1
            if status == "ok":
                supported_count += 1
            else:
                unsupported_count += 1
            for key, sink_values in [
                ("matching_segment_qerror", matching_qerrors),
                ("distinct_trajectory_qerror", distinct_qerrors),
                ("a_abs_error", a_abs_errors),
            ]:
                value = payload.get(key)
                if value is not None:
                    sink_values.append(float(value))
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "queries_path": str(queries_path),
                "output_path": str(output_path),
                "queries_total": total,
                "status_counts": status_counts,
                "database_truth_status_counts": database_truth_status_counts,
                "supported_query_count": supported_count,
                "unsupported_query_count": unsupported_count,
                "queries_with_database_truth": database_truth_count,
                "queries_without_database_truth": missing_database_truth_count,
                "matching_segment_qerror": _percentile_summary(matching_qerrors),
                "distinct_trajectory_qerror": _percentile_summary(distinct_qerrors),
                "a_abs_error": _error_summary(a_abs_errors),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"evaluated {total} POL queries to {output_path}")
    print(f"summary={summary_path}")


def _load_exact_fixture(
    args: argparse.Namespace,
    config: dict,
    metadata: ModelMetadata,
) -> tuple[ExactOracle | None, tuple[object, ...] | None, tuple[object, ...] | None]:
    fixture_dir = args.exact_fixture_dir or config.get("dataset", {}).get("prepared_directory")
    if not fixture_dir:
        return None, None, None
    root = Path(fixture_dir)
    rows_path = root / "sample_rows.npy"
    trajectory_ids_path = root / "sample_trajectory_ids.npy"
    segment_ids_path = root / "sample_segment_ids.npy"
    if not (rows_path.exists() and trajectory_ids_path.exists() and segment_ids_path.exists()):
        return None, None, None
    rows = np.load(rows_path, mmap_mode="r")
    trajectory_ids = tuple(np.load(trajectory_ids_path, allow_pickle=True).tolist())
    segment_ids = _segment_id_rows_to_tuples(_load_segment_ids_array(segment_ids_path))
    return ExactOracle(metadata, rows), trajectory_ids, segment_ids


def _percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _error_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


if __name__ == "__main__":
    main()
