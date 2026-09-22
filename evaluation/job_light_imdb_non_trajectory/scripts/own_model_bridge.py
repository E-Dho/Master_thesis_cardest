#!/usr/bin/env python3
"""Adapter bridge for arbitrary checkpoints and configs of the own approach."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "build", "smoke", "evaluate"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--latencies", type=Path)
    parser.add_argument("--query-limit", type=int, default=2)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    _validate(args.config, args.checkpoint)
    if args.command == "prepare":
        _write_json(args.output_directory / "prepare_manifest.json", _manifest(args))
    elif args.command == "build":
        _build(args)
    else:
        _evaluate(args, smoke=args.command == "smoke")
    return 0


def _validate(config: Path, checkpoint: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)


def _manifest(args: argparse.Namespace) -> dict:
    return {
        "status": "ready",
        "model_config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
    }


def _build(args: argparse.Namespace) -> None:
    from model.src.model.checkpoint import load_resmade_checkpoint

    model, _ = load_resmade_checkpoint(args.checkpoint, map_location="cpu")
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    checkpoint_mb = args.checkpoint.stat().st_size / 1_000_000
    artifact = {
        "parameter_count": parameter_count,
        "serialized_model_mb": checkpoint_mb,
        "full_checkpoint_mb": checkpoint_mb,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_config": str(args.config.resolve()),
    }
    _write_json(args.output_directory / "artifact_manifest.json", artifact)
    _write_json(
        args.output_directory / "build_stage_metrics.json",
        {"status": "pretrained_checkpoint", "total_build_seconds": None},
    )


def _evaluate(args: argparse.Namespace, *, smoke: bool) -> None:
    if args.queries is None:
        raise ValueError("--queries is required")
    queries = args.queries
    temporary = None
    if smoke:
        temporary = args.output_directory / "own_model_smoke_queries.csv"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        lines = [line for line in queries.read_text(encoding="utf-8").splitlines() if line.strip()]
        temporary.write_text("\n".join(lines[: args.query_limit]) + "\n", encoding="utf-8")
        queries = temporary
    predictions = args.predictions or args.output_directory / "smoke_predictions.csv"
    latencies = args.latencies or args.output_directory / "smoke_latency.csv"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "own_model_eval_runner.py"),
        "--checkpoint", str(args.checkpoint), "--queries", str(queries),
        "--predictions", str(predictions), "--latencies", str(latencies),
        "--warmup-passes", str(args.warmup_passes),
        "--repetitions", str(args.repetitions),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    if smoke:
        _write_json(
            args.output_directory / "smoke_metrics.json",
            {
                "status": "ok",
                "query_count": args.query_limit,
                "wall_seconds": time.perf_counter() - started,
            },
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
