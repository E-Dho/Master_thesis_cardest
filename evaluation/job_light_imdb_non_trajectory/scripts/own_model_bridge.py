#!/usr/bin/env python3
"""Adapter bridge for arbitrary checkpoints and configs of the own approach."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model_source = _activate_model_source()
    _validate(args.config, args.checkpoint)
    if args.command == "prepare":
        _write_json(
            args.output_directory / "prepare_manifest.json",
            _manifest(args, model_source),
        )
    elif args.command == "build":
        _build(args, model_source)
    else:
        _evaluate(args, model_source=model_source, smoke=args.command == "smoke")
    return 0


def _validate(config: Path, checkpoint: Path) -> None:
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)


def _activate_model_source() -> dict[str, str | None]:
    """Put the configured thesis checkout ahead of the evaluation checkout.

    The evaluation repository intentionally does not carry a second editable
    model tree.  Every own-model invocation must instead import the recorded
    source checkout, and reject a checkout that does not match that revision.
    """

    source_text = os.environ.get("OWN_MODEL_SOURCE")
    expected_revision = os.environ.get("OWN_MODEL_SOURCE_REVISION")
    if not source_text:
        raise ValueError("OWN_MODEL_SOURCE is required for own-model evaluation")
    source = Path(source_text).expanduser().resolve()
    if not (source / "model").is_dir():
        raise FileNotFoundError(f"OWN_MODEL_SOURCE has no model package: {source}")
    observed_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected_revision and observed_revision != expected_revision:
        raise ValueError(
            "OWN_MODEL_SOURCE_REVISION mismatch: "
            f"expected {expected_revision}, observed {observed_revision}"
        )
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    inherited = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [source_text, *[entry for entry in inherited if entry != source_text]]
    )
    return {"path": source_text, "revision": observed_revision}


def _manifest(args: argparse.Namespace, model_source: dict[str, str | None]) -> dict:
    return {
        "status": "ready",
        "model_config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "model_source": model_source,
    }


def _build(args: argparse.Namespace, model_source: dict[str, str | None]) -> None:
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
        "model_source": model_source,
    }
    _write_json(args.output_directory / "artifact_manifest.json", artifact)
    _write_json(
        args.output_directory / "build_stage_metrics.json",
        {"status": "pretrained_checkpoint", "total_build_seconds": None},
    )


def _evaluate(
    args: argparse.Namespace,
    *,
    model_source: dict[str, str | None],
    smoke: bool,
) -> None:
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
        "--device", args.device,
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True, env=os.environ.copy())
    if smoke:
        _write_json(
            args.output_directory / "smoke_metrics.json",
            {
                "status": "ok",
                "query_count": args.query_limit,
                "wall_seconds": time.perf_counter() - started,
                "model_source": model_source,
            },
        )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
