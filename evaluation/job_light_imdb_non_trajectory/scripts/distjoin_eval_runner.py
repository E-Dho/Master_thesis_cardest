#!/usr/bin/env python3
"""Evaluate a trained pinned DistJoin model without its broken polling loop."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--experiment-mark", default="production")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--base-cardinalities", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--latencies", required=True, type=Path)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--initialize-missing-checkpoints", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    device = torch.device(args.device)

    source = args.source_root.resolve()
    overlay = args.overlay_root.resolve()
    os.chdir(overlay)
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "eval-IMDB.py"), "--config", "IMDB", "--no_wandb"]
    module = _load_module(source / "eval-IMDB.py")
    module.util.get_device = lambda: device
    _install_missing_tesseract_transformer(sys.modules["estimator"])
    module.JoinOrderBenchmark.LoadTrueBaseCard(module.raw_config["tag"])
    tables, _, estimator, _ = _build_estimator_cpu_compatible(
        module, args.experiment_mark
    )
    if args.initialize_missing_checkpoints:
        _initialize_checkpoints(module, estimator, overlay, args.experiment_mark)
    _load_checkpoints(module, estimator, overlay, args.experiment_mark)

    query_rows, _ = module.utils.util.JobToQuery(str(args.queries))
    loaded = module.utils.util.UnpackQueries(tables, query_rows)
    base_cards = json.loads(args.base_cardinalities.read_text(encoding="utf-8"))
    estimator.cache = {}

    for _ in range(args.warmup_passes):
        for query in loaded:
            _estimate(module, estimator, tables, query, base_cards)
        _synchronize(device)

    predictions: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    reference: dict[int, float] = {}
    for repetition in range(args.repetitions):
        for query_id, query in enumerate(loaded):
            _synchronize(device)
            started = time.perf_counter()
            estimate = _estimate(module, estimator, tables, query, base_cards)
            _synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            if repetition == 0:
                reference[query_id] = estimate
                predictions.append(
                    {
                        "query_id": query_id,
                        "estimated_cardinality": estimate,
                        "status": "ok",
                        "diagnostic": "native DistJoin DirectEstimator",
                    }
                )
            elif not np.isclose(reference[query_id], estimate, rtol=1e-6, atol=1e-6):
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
    _write_csv(args.predictions, predictions)
    _write_csv(args.latencies, latencies)
    return 0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return str(torch.cuda.get_device_name(device))
    return "CPU"


def _load_module(path: Path):
    # Upstream imports one constant through a workload generator whose module
    # import initializes a broken vendored watchdog package. Evaluation never
    # uses that generator, so provide only the imported constant.
    workload_module = types.ModuleType("queries.GenerateMSCNWorkload")
    workload_module.workload_num = 1_000
    sys.modules.setdefault("queries.GenerateMSCNWorkload", workload_module)
    spec = importlib.util.spec_from_file_location("distjoin_upstream_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_missing_tesseract_transformer(module: Any) -> None:
    """Repair an optional upstream class reference left undefined in IMDB mode."""

    if not hasattr(module, "TesseractTransformer"):
        module.TesseractTransformer = type("TesseractTransformer", (), {})


def _build_estimator_cpu_compatible(module: Any, experiment_mark: str):
    if torch.cuda.is_available():
        return module.build_estimator(experiment_mark, True)
    original = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
    try:
        return module.build_estimator(experiment_mark, True)
    finally:
        torch.Tensor.pin_memory = original


def _load_checkpoints(module: Any, estimator: Any, overlay: Path, mark: str) -> None:
    from model import made

    for dataset, model in estimator.models.items():
        filename = module.config["glob"].format(dataset, mark)
        checkpoint = overlay / "Configs" / "IMDB" / "model" / mark / filename
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location=module.util.get_device())
        for key in list(state):
            if "multi_pred_embed_nn" in key:
                del state[key]
        if isinstance(model, made.MADE):
            model.ConvertToUnensemble()
        model.load_state_dict(state)
        model.eval()
        if isinstance(model, made.MADE):
            model.ConvertToEnsemble()


def _initialize_checkpoints(module: Any, estimator: Any, overlay: Path, mark: str) -> None:
    """Write random fixture checkpoints solely for evaluator integration tests."""

    directory = overlay / "Configs" / "IMDB" / "model" / mark
    directory.mkdir(parents=True, exist_ok=True)
    for dataset, model in estimator.models.items():
        filename = module.config["glob"].format(dataset, mark)
        checkpoint = directory / filename
        if not checkpoint.exists():
            torch.save(model.state_dict(), checkpoint)


def _estimate(module: Any, estimator: Any, tables: dict[str, Any], query: Any,
              base_cards: dict[str, int]) -> float:
    join_tables, _join_keys, predicates_by_table, _truth = query
    predicates = []
    for table_name, predicate in predicates_by_table.items():
        columns, operators, values = module.AQP_estimator.FillInUnqueriedColumns(
            estimator.base_tables[table_name],
            predicate["cols"], predicate["ops"], predicate["vals"],
        )
        projected = module.AQP_estimator.ProjectQuery(
            estimator.fact_tables[table_name], columns, operators, values
        )
        fact_cols, fact_ops, fact_values, _ = projected
        table = estimator.fact_tables[table_name]
        for fact_col, operations, literals in zip(fact_cols, fact_ops, fact_values):
            if operations is None:
                continue
            encoded_ops = [module.AQP_estimator.OPS_dict[value] for value in operations]
            encoded_values = table.columns_dict[fact_col.name].ValToBin(literals)
            column_index = table.base_table.columns_name_to_idx[fact_col.raw_col_name]
            original_ops = [module.AQP_estimator.OPS_dict[value] for value in operators[column_index]]
            original_values = [
                table.base_table.columns_dict[fact_col.raw_col_name].ValToBin(value)
                for value in values[column_index]
            ]
            predicates.append(
                module.Predicate(
                    table_name, fact_col.name,
                    table.map_from_fact_col_to_col[fact_col.name],
                    encoded_ops, encoded_values, original_ops, original_values,
                    operators[column_index], values[column_index],
                )
            )
    for table_name in join_tables:
        predicates.append(
            module.DirectEstimator.GetKeyPredicate(
                table_name, module.DirectEstimator.GetJoinKeyColumn(tables[table_name])
            )
        )
    condition = estimator.get_prob_of_predicate_tree(
        predicates, join_tables, tables, module.config["how"], real=module.config["real"]
    )
    key = ",".join(sorted(join_tables))
    if key not in base_cards:
        raise KeyError(f"missing exact base cardinality for {key}")
    return float(np.ceil(float(condition) * int(base_cards[key])))


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
