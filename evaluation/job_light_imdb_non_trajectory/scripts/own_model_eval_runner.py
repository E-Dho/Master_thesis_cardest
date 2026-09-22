#!/usr/bin/env python3
"""Canonical CPU evaluator for this repository's predicate ResMADE models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from model.scripts.evaluate_job_light_queries import eval_query, parse_query
from model.src.data.schema import ModelMetadata
from model.src.inference.estimator import OnePassEstimator
from model.src.inference.torch_estimator import TorchDistributionModel
from model.src.model.checkpoint import load_resmade_checkpoint
from model.src.predicates.vocabulary import PredicateVocabularies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--latencies", required=True, type=Path)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    model, payload = load_resmade_checkpoint(args.checkpoint, map_location="cpu")
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    vocabularies = PredicateVocabularies.from_json_dict(payload["predicate_vocabularies"], metadata)
    wrapped = TorchDistributionModel(model, metadata, vocabularies)
    estimator = OnePassEstimator(wrapped, metadata)
    queries = [
        (query_id, *parse_query(line))
        for query_id, line in enumerate(args.queries.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]

    for _ in range(args.warmup_passes):
        for _query_id, included, predicates, _truth in queries:
            eval_query(estimator, wrapped, model, vocabularies, metadata, included, predicates)

    predictions: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    reference: dict[int, float] = {}
    for repetition in range(args.repetitions):
        for query_id, included, predicates, _truth in queries:
            started = time.perf_counter()
            result = eval_query(
                estimator, wrapped, model, vocabularies, metadata, included, predicates
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            status, estimate = str(result[0]), float(result[1])
            if repetition == 0:
                reference[query_id] = estimate
                supported = status != "unsupported"
                predictions.append(
                    {
                        "query_id": query_id,
                        "estimated_cardinality": estimate if supported else "",
                        "status": "ok" if supported else "unsupported",
                        "diagnostic": status,
                    }
                )
            elif reference[query_id] != estimate:
                raise RuntimeError(f"non-deterministic estimate for query {query_id}")
            latencies.append(
                {
                    "query_id": query_id,
                    "repetition": repetition,
                    "latency_ms": latency_ms,
                    "scope": "predicate_encoding_and_estimation",
                }
            )
    _write_csv(args.predictions, predictions)
    _write_csv(args.latencies, latencies)
    return 0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty result {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
