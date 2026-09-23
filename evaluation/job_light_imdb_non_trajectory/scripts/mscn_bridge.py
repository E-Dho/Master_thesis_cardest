#!/usr/bin/env python3
"""Compatibility and instrumentation bridge for the upstream MSCN baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MATERIALIZED_SAMPLES = 1_000


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_common(prepare)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--output", required=True)

    train = subparsers.add_parser("train")
    _add_common(train)
    train.add_argument("--checkpoint-output", required=True)
    train.add_argument("--metadata-output", required=True)
    train.add_argument("--artifact-manifest", required=True)
    train.add_argument("--training-metrics", required=True)
    train.add_argument("--queries", type=int, default=100_000)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=1_024)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--max-batches", type=int)

    evaluate = subparsers.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--metadata", required=True)
    evaluate.add_argument("--queries-prefix", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--latency", required=True)
    evaluate.add_argument("--query-limit", type=int)
    evaluate.add_argument("--warmup-passes", type=int)
    evaluate.add_argument("--repetitions", type=int)
    evaluate.add_argument("--smoke", action="store_true")

    smoke = subparsers.add_parser("smoke")
    _add_common(smoke)
    smoke.add_argument("--output-directory", required=True)
    smoke.add_argument("--queries-prefix", required=True)
    smoke.add_argument("--query-limit", type=int, default=2)

    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "train":
        _train(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "smoke":
        _smoke(args)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--runtime-root",
        help="Directory containing data/train.* and data/column_min_max_vals.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)


def _prepare(args: argparse.Namespace) -> None:
    source = Path(args.source_root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != args.revision:
        raise ValueError(f"MSCN revision mismatch: expected {args.revision}, got {revision}")
    required = (
        source / "train.py",
        source / "mscn" / "model.py",
        source / "data" / "train.csv",
        source / "data" / "train.bitmaps",
        source / "data" / "column_min_max_vals.csv",
        source / "workloads" / "job-light.csv",
        source / "workloads" / "job-light.bitmaps",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    training_queries = _line_count(source / "data" / "train.csv")
    evaluation_queries = _line_count(source / "workloads" / "job-light.csv")
    if training_queries != 100_000 or evaluation_queries != 70:
        raise ValueError(
            f"unexpected MSCN workloads: train={training_queries}, JOB-light={evaluation_queries}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--short"], text=True
    ).strip()
    payload = {
        "status": "ready",
        "source_root": str(source),
        "source_revision": revision,
        "source_dirty": bool(status),
        "source_status": status.splitlines(),
        "training_query_count": training_queries,
        "evaluation_query_count": evaluation_queries,
        "materialized_samples": MATERIALIZED_SAMPLES,
        "training_csv_sha256": _sha256(source / "data" / "train.csv"),
        "training_bitmaps_sha256": _sha256(source / "data" / "train.bitmaps"),
        "workload_csv_sha256": _sha256(source / "workloads" / "job-light.csv"),
        "workload_bitmaps_sha256": _sha256(source / "workloads" / "job-light.bitmaps"),
    }
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _train(args: argparse.Namespace) -> None:
    runtime = _load_upstream(
        Path(args.source_root), args.device, args.cpu_threads,
        Path(args.runtime_root) if args.runtime_root else None,
    )
    torch = runtime["torch"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime["np"].random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    setup_started = time.perf_counter()
    loaded = runtime["data"].get_train_datasets(args.queries, MATERIALIZED_SAMPLES)
    (
        dictionaries,
        column_min_max,
        min_log_cardinality,
        max_log_cardinality,
        labels_train,
        labels_validation,
        max_num_joins,
        max_num_predicates,
        train_dataset,
        validation_dataset,
    ) = loaded
    table2vec, column2vec, op2vec, join2vec = dictionaries
    sample_features = len(table2vec) + MATERIALIZED_SAMPLES
    predicate_features = len(column2vec) + len(op2vec) + 1
    join_features = len(join2vec)
    model = runtime["SetConv"](
        sample_features, predicate_features, join_features, args.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    setup_seconds = time.perf_counter() - setup_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    epoch_losses: list[float] = []
    optimizer_steps = 0
    training_started = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        batches = 0
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            tensors = [tensor.to(device) for tensor in batch]
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = tensors
            optimizer.zero_grad(set_to_none=True)
            output = model(
                samples, predicates, joins, sample_masks, predicate_masks, join_masks
            )
            loss = _qerror_loss(
                torch, output, targets.float(), min_log_cardinality, max_log_cardinality
            )
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite MSCN loss at epoch={epoch} batch={batch_index}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            optimizer_steps += 1
        if not batches:
            raise ValueError("MSCN training produced no optimizer batches")
        epoch_loss = loss_sum / batches
        epoch_losses.append(epoch_loss)
        print(f"epoch={epoch} batches={batches} qerror_loss={epoch_loss:.8f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - training_started

    checkpoint = Path(args.checkpoint_output).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    metadata = {
        "format_version": 1,
        "seed": args.seed,
        "materialized_samples": MATERIALIZED_SAMPLES,
        "hidden_size": args.hidden_size,
        "sample_features": sample_features,
        "predicate_features": predicate_features,
        "join_features": join_features,
        "table_keys": sorted(table2vec),
        "column_keys": sorted(column2vec),
        "operator_keys": sorted(op2vec),
        "join_keys": sorted(join2vec),
        "column_min_max": {
            key: [float(value[0]), float(value[1])]
            for key, value in column_min_max.items()
        },
        "min_log_cardinality": float(min_log_cardinality),
        "max_log_cardinality": float(max_log_cardinality),
        "max_num_joins": int(max_num_joins),
        "max_num_predicates": int(max_num_predicates),
    }
    _write_json(Path(args.metadata_output), metadata)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    peak_allocated = peak_reserved = None
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    metrics = {
        "device": str(device),
        "seed": args.seed,
        "training_queries_requested": args.queries,
        "training_examples": len(labels_train),
        "validation_examples": len(labels_validation),
        "materialized_samples": MATERIALIZED_SAMPLES,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "optimizer_steps": optimizer_steps,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "total_build_seconds": setup_seconds + training_seconds,
        "loss_first": epoch_losses[0],
        "loss_last": epoch_losses[-1],
        "loss_by_epoch": epoch_losses,
        "parameter_count": parameter_count,
        "peak_training_gpu_allocated_bytes": peak_allocated,
        "peak_training_gpu_reserved_bytes": peak_reserved,
    }
    artifact = {
        "parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "serialized_model_mb": checkpoint.stat().st_size / 1_000_000.0,
        "full_checkpoint_mb": (
            checkpoint.stat().st_size + Path(args.metadata_output).resolve().stat().st_size
        ) / 1_000_000.0,
        "checkpoint_format": "PyTorch state_dict plus JSON encoding metadata",
        "checkpoint": str(checkpoint),
        "metadata": str(Path(args.metadata_output).resolve()),
        "training": metrics,
        "peak_training_gpu_allocated_bytes": peak_allocated,
        "peak_training_gpu_reserved_bytes": peak_reserved,
    }
    _write_json(Path(args.training_metrics), metrics)
    _write_json(Path(args.artifact_manifest), artifact)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    runtime = _load_upstream(
        Path(args.source_root), args.device, args.cpu_threads,
        Path(args.runtime_root) if args.runtime_root else None,
    )
    torch = runtime["torch"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable")
    device = torch.device(args.device)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    model = runtime["SetConv"](
        metadata["sample_features"],
        metadata["predicate_features"],
        metadata["join_features"],
        metadata["hidden_size"],
    ).to(device)
    state = torch.load(
        Path(args.checkpoint), map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    parsed = runtime["data"].load_data(
        str(Path(args.queries_prefix).resolve()), metadata["materialized_samples"]
    )
    joins, predicates, tables, samples, labels = parsed
    if args.query_limit is not None:
        joins = joins[: args.query_limit]
        predicates = predicates[: args.query_limit]
        tables = tables[: args.query_limit]
        samples = samples[: args.query_limit]
        labels = labels[: args.query_limit]
    query_count = len(labels)
    warmup_passes = (
        args.warmup_passes
        if args.warmup_passes is not None
        else int(os.environ.get("JOBLIGHT_TIMING_WARMUP_PASSES", "1"))
    )
    repetitions = (
        args.repetitions
        if args.repetitions is not None
        else int(os.environ.get("JOBLIGHT_TIMING_REPETITIONS", "10"))
    )
    if warmup_passes < 0 or repetitions <= 0:
        raise ValueError("warmup passes must be nonnegative and repetitions positive")

    dictionaries = _dictionaries(runtime["util"], metadata)
    failures: dict[int, str] = {}
    with torch.inference_mode():
        for _ in range(warmup_passes):
            for query_id in range(query_count):
                try:
                    _estimate_one(
                        runtime, model, device, metadata, dictionaries,
                        joins[query_id], predicates[query_id], tables[query_id], samples[query_id]
                    )
                except Exception as exc:
                    failures[query_id] = f"{type(exc).__name__}: {exc}"

        estimates: dict[int, float] = {}
        latency_rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            for query_id in range(query_count):
                if query_id in failures:
                    continue
                try:
                    _synchronize(torch, device)
                    started = time.perf_counter()
                    estimate = _estimate_one(
                        runtime, model, device, metadata, dictionaries,
                        joins[query_id], predicates[query_id], tables[query_id], samples[query_id]
                    )
                    _synchronize(torch, device)
                    latency_ms = (time.perf_counter() - started) * 1_000.0
                    if not math.isfinite(estimate) or estimate < 0:
                        raise ValueError(f"invalid estimate {estimate}")
                    if repetition == 0:
                        estimates[query_id] = estimate
                    latency_rows.append({
                        "query_id": query_id,
                        "repetition": repetition,
                        "latency_ms": latency_ms,
                        "scope": "predicate_bitmap_encoding_and_setconv_inference",
                        "device": args.device,
                        "device_name": (
                            str(torch.cuda.get_device_name(device))
                            if device.type == "cuda"
                            else "CPU"
                        ),
                    })
                except Exception as exc:
                    failures[query_id] = f"{type(exc).__name__}: {exc}"

    prediction_rows = [
        {
            "query_id": query_id,
            "estimated_cardinality": "" if query_id in failures else estimates[query_id],
            "status": "failed" if query_id in failures else "ok",
            "diagnostic": failures.get(query_id, ""),
        }
        for query_id in range(query_count)
    ]
    _write_csv(
        Path(args.predictions), prediction_rows,
        ("query_id", "estimated_cardinality", "status", "diagnostic")
    )
    _write_csv(
        Path(args.latency), latency_rows,
        ("query_id", "repetition", "latency_ms", "scope", "device", "device_name")
    )
    payload = {
        "status": "ok" if not failures else "partial_failure",
        "query_count": query_count,
        "success_count": query_count - len(failures),
        "failure_count": len(failures),
        "warmup_passes": warmup_passes,
        "repetitions": repetitions,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.smoke and failures:
        raise RuntimeError(f"MSCN smoke inference failed: {failures}")


def _smoke(args: argparse.Namespace) -> None:
    output = Path(args.output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        checkpoint_output=str(output / "smoke_model.pt"),
        metadata_output=str(output / "smoke_model_metadata.json"),
        artifact_manifest=str(output / "smoke_artifact_manifest.json"),
        training_metrics=str(output / "smoke_training_metrics.json"),
        queries=1_000,
        epochs=1,
        batch_size=64,
        hidden_size=256,
        max_batches=2,
    )
    _train(train_args)
    evaluate_args = argparse.Namespace(
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        checkpoint=str(output / "smoke_model.pt"),
        metadata=str(output / "smoke_model_metadata.json"),
        queries_prefix=args.queries_prefix,
        predictions=str(output / "smoke_predictions.csv"),
        latency=str(output / "smoke_latency.csv"),
        query_limit=args.query_limit,
        warmup_passes=0,
        repetitions=1,
        smoke=True,
    )
    _evaluate(evaluate_args)


def _load_upstream(
    source_root: Path,
    device: str,
    cpu_threads: int,
    runtime_root: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.chdir(source_root if runtime_root is None else runtime_root.resolve())
    import numpy as np
    import torch
    import mscn.data as data
    import mscn.util as util
    from mscn.model import SetConv

    def compatible_open(file, mode="r", *args, **kwargs):
        return open(file, mode.replace("U", ""), *args, **kwargs)

    # Python 3.11 removed universal-newline mode. Keep upstream parsing intact.
    data.open = compatible_open
    torch.set_num_threads(max(1, int(cpu_threads)))
    return {"np": np, "torch": torch, "data": data, "util": util, "SetConv": SetConv}


def _qerror_loss(torch, predictions, targets, minimum, maximum):
    predicted = torch.exp(predictions * (maximum - minimum) + minimum)
    actual = torch.exp(targets.reshape(-1, 1) * (maximum - minimum) + minimum)
    return torch.maximum(predicted / actual, actual / predicted).mean()


def _dictionaries(util, metadata):
    return (
        util.get_set_encoding(set(metadata["table_keys"]))[0],
        util.get_set_encoding(set(metadata["column_keys"]))[0],
        util.get_set_encoding(set(metadata["operator_keys"]))[0],
        util.get_set_encoding(set(metadata["join_keys"]))[0],
    )


def _estimate_one(runtime, model, device, metadata, dictionaries,
                  joins, predicates, tables, samples) -> float:
    table2vec, column2vec, op2vec, join2vec = dictionaries
    encoded_samples = runtime["util"].encode_samples([tables], [samples], table2vec)
    encoded_predicates, encoded_joins = runtime["util"].encode_data(
        [predicates], [joins], metadata["column_min_max"],
        column2vec, op2vec, join2vec
    )
    dataset = runtime["data"].make_dataset(
        encoded_samples,
        encoded_predicates,
        encoded_joins,
        [0.0],
        max(len(joins), 1),
        max(len(predicates), 1),
    )
    tensors = [tensor.to(device) for tensor in dataset.tensors]
    sample, predicate, join, _, sample_mask, predicate_mask, join_mask = tensors
    output = model(sample, predicate, join, sample_mask, predicate_mask, join_mask)
    normalized = float(output.detach().cpu().reshape(-1)[0])
    log_cardinality = (
        normalized
        * (metadata["max_log_cardinality"] - metadata["min_log_cardinality"])
        + metadata["min_log_cardinality"]
    )
    return float(round(math.exp(log_cardinality)))


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
