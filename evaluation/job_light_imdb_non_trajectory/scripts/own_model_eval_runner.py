#!/usr/bin/env python3
"""Canonical CPU/CUDA evaluator for this repository's predicate ResMADE models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    timing = _timing()
    session = timing.session_from_environment("own_model_eval_runner", device=args.device)
    if session is not None:
        session.configure(torch=torch)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    device = torch.device(args.device)
    model, payload = load_resmade_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    vocabularies = PredicateVocabularies.from_json_dict(payload["predicate_vocabularies"], metadata)
    wrapped = TorchDistributionModel(model, metadata, vocabularies, device=str(device))
    estimator = OnePassEstimator(wrapped, metadata)
    queries = [
        (query_id, *parse_query(line))
        for query_id, line in enumerate(args.queries.read_text(encoding="utf-8").splitlines())
        if line.strip()
    ]

    if session is not None:
        session.configure_torch(torch)
        session.note(query_count=len(queries))
        session.verify("pre_timing")
    measured = timing.measured(session, "own_model_workload")
    measured.__enter__()
    for _ in range(args.warmup_passes):
        for _query_id, included, predicates, _truth in queries:
            eval_query(estimator, wrapped, model, vocabularies, metadata, included, predicates)
    _synchronize(device)

    predictions: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    reference: dict[int, float] = {}
    for repetition in range(args.repetitions):
        for query_id, included, predicates, _truth in queries:
            _synchronize(device)
            started = time.perf_counter()
            result = eval_query(
                estimator, wrapped, model, vocabularies, metadata, included, predicates
            )
            _synchronize(device)
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
            elif not np.isclose(reference[query_id], estimate, rtol=1e-7, atol=1e-9):
                raise RuntimeError(f"non-deterministic estimate for query {query_id}")
            latencies.append(
                {
                    "query_id": query_id,
                    "repetition": repetition,
                    "latency_ms": latency_ms,
                    "scope": "predicate_encoding_and_estimation",
                    "device": args.device,
                    "device_name": _device_name(device),
                }
            )
    measured.__exit__(None, None, None)
    if session is not None:
        session.verify("post_timing")
        session.write()
    _write_csv(args.predictions, predictions)
    _write_csv(args.latencies, latencies)
    return 0


def _timing():
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import _timing

    return _timing


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return "CPU"


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
