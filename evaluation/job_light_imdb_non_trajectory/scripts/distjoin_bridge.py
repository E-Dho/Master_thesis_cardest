#!/usr/bin/env python3
"""Isolation, compatibility, and instrumentation bridge for DistJoin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

JOB_LIGHT_TABLES = (
    "title",
    "cast_info",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_companies",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "evaluator-smoke", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-root", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--output-directory", required=True)
        command.add_argument("--python", default=sys.executable)
        command.add_argument("--seed", type=int, default=42)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source-root", required=True)
    evaluate.add_argument("--revision", required=True)
    evaluate.add_argument("--output-directory", required=True)
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--base-cardinalities", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--latencies", required=True)
    evaluate.add_argument("--python", default=sys.executable)
    evaluate.add_argument("--warmup-passes", type=int, default=1)
    evaluate.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "smoke":
        _smoke(args)
    elif args.command == "evaluator-smoke":
        _evaluator_smoke(args)
    elif args.command == "train":
        _train(args)
    else:
        _evaluate(args)
    return 0


def _prepare(args: argparse.Namespace) -> None:
    source = _validate_source(Path(args.source_root), args.revision)
    dataset = Path(args.dataset_root).resolve()
    tables = _validate_dataset(dataset)
    probe = _run_import_probe(source, Path(args.python))
    payload = {
        "status": "ready",
        "source_root": str(source),
        "source_revision": args.revision,
        "source_dirty": bool(_git_status(source)),
        "source_status": _git_status(source),
        "upstream_rename_regression": "AQP_estimator.py renamed to estimator.py at pinned HEAD",
        "compatibility_alias": str(_compat_directory()),
        "native_sampler": probe,
        "tables": tables,
    }
    _write_json(Path(args.output_directory) / "prepare_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _smoke(args: argparse.Namespace) -> None:
    source = _validate_source(Path(args.source_root), args.revision)
    _validate_dataset(Path(args.dataset_root).resolve())
    output = Path(args.output_directory).resolve()
    fixture = output / "fixture"
    overlay = output / "overlay"
    fixture.mkdir(parents=True, exist_ok=True)
    _write_fixture(fixture)
    config = _production_config(source, fixture, args.seed)
    config["num_gpu"] = 1
    config["train"].update(
        {
            "epochs": 1,
            "bs": 16,
            "sample_bs": 16,
            "max_steps": 1,
            "warmups": 1,
            "patient": 2,
        }
    )
    effective_warmup_steps = _effective_warmup_steps(
        config["train"]["warmups"],
        batches_per_epoch=config["train"]["max_steps"],
        epochs=config["train"]["epochs"],
    )
    config["test"]["glob"] = f"{{}}-seed{args.seed}-0.pt"
    config["test"]["glob_epoch"] = f"{{}}-seed{args.seed}-{{}}.pt"
    _write_overlay(overlay, config)
    started = time.perf_counter()
    completed = _run_upstream_train(source, overlay, Path(args.python), "smoke")
    wall_seconds = time.perf_counter() - started
    (output / "upstream_train.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / "upstream_train.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    checkpoints = sorted((overlay / "Configs" / "IMDB" / "model" / "smoke").glob("*.pt"))
    checkpoint_tables = {
        path.name.split(f"-seed{args.seed}-", 1)[0]
        for path in checkpoints
        if f"-seed{args.seed}-" in path.name
    }
    if completed.returncode != 0 or checkpoint_tables != set(JOB_LIGHT_TABLES):
        raise RuntimeError(
            "DistJoin smoke failed or produced no checkpoint\n"
            f"stdout:\n{completed.stdout[-8000:]}\nstderr:\n{completed.stderr[-8000:]}"
        )
    workload = output / "smoke_queries.csv"
    base_cards = output / "smoke_base_cardinalities.json"
    predictions = output / "smoke_predictions.csv"
    latencies = output / "smoke_latency.csv"
    _write_smoke_workload(workload)
    _write_json(base_cards, {"movie_keyword,title": 128, "movie_info_idx,title": 128})
    evaluation = _run_distjoin_evaluator(
        source=source,
        overlay=overlay,
        python=Path(args.python),
        queries=workload,
        base_cardinalities=base_cards,
        predictions=predictions,
        latencies=latencies,
        warmup_passes=1,
        repetitions=2,
        experiment_mark="smoke",
    )
    with predictions.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with latencies.open(newline="", encoding="utf-8") as handle:
        latency_rows = list(csv.DictReader(handle))
    if len(prediction_rows) != 2 or len(latency_rows) != 4:
        raise RuntimeError(
            f"DistJoin evaluator smoke produced {len(prediction_rows)} predictions "
            f"and {len(latency_rows)} latency rows\n{evaluation.stderr[-8000:]}"
        )
    payload = {
        "status": "ok",
        "wall_seconds": wall_seconds,
        "checkpoint_count": len(checkpoints),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        "native_dynamic_sampler_exercised": "using data generator" in completed.stdout,
        "anpm_enabled": bool(config["train"]["use_ANPM"]),
        "factorization_enabled": bool(config["factorize"]),
        "evaluator_integration": "ok",
        "evaluation_query_count": len(prediction_rows),
        "evaluation_timing_rows": len(latency_rows),
        "effective_warmup_steps": effective_warmup_steps,
        "stdout_tail": completed.stdout[-4000:],
    }
    _write_json(output / "smoke_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _train(args: argparse.Namespace) -> None:
    source = _validate_source(Path(args.source_root), args.revision)
    dataset = Path(args.dataset_root).resolve()
    _validate_dataset(dataset)
    output = Path(args.output_directory).resolve()
    overlay = output / "overlay"
    config = _production_config(source, dataset, args.seed)
    _write_overlay(overlay, config)
    started = time.perf_counter()
    completed = _run_upstream_train(source, overlay, Path(args.python), "production")
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"DistJoin training failed\nstdout:\n{completed.stdout[-12000:]}\n"
            f"stderr:\n{completed.stderr[-12000:]}"
        )
    checkpoints = sorted((overlay / "Configs" / "IMDB" / "model" / "production").glob("*.pt"))
    if not checkpoints:
        raise RuntimeError("DistJoin training produced no checkpoints")
    metrics = {
        "training_seconds": wall_seconds,
        "total_build_seconds": wall_seconds,
        "checkpoint_count": len(checkpoints),
        "full_checkpoint_mb": sum(path.stat().st_size for path in checkpoints) / 1_000_000,
        "upstream_config": config,
    }
    # DistJoin registers autoregressive masks as state-dict buffers. Exclude
    # those buffers so this remains a parameter count rather than a checkpoint
    # tensor-element count.
    try:
        import torch as _torch
        parameter_count = sum(
            _parameter_count_from_state_dict(
                _torch.load(ckpt, map_location="cpu", weights_only=True)
            )
            for ckpt in checkpoints
        )
    except Exception:
        parameter_count = None
    artifact = {
        "parameter_count": parameter_count,
        "serialized_model_mb": metrics["full_checkpoint_mb"],
        "full_checkpoint_mb": metrics["full_checkpoint_mb"],
        "checkpoint_directory": str(checkpoints[0].parent),
        "training": metrics,
    }
    _write_json(output / "build_stage_metrics.json", metrics)
    _write_json(output / "artifact_manifest.json", artifact)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _parameter_count_from_state_dict(state: dict[str, Any]) -> int:
    """Count parameters in pinned DistJoin checkpoints, excluding mask buffers."""

    return sum(
        int(value.numel())
        for name, value in state.items()
        if not name.endswith(".mask")
    )


def _effective_warmup_steps(
    warmups: float | int,
    *,
    batches_per_epoch: int,
    epochs: int,
) -> int:
    """Mirror DistJoin's warmup conversion and reject a zero denominator."""
    steps = (
        int(float(warmups) * batches_per_epoch * epochs)
        if float(warmups) < 1
        else int(warmups)
    )
    if steps < 1:
        raise ValueError(
            "DistJoin warmup resolves to zero steps; use at least one absolute "
            "warmup step for a short smoke run"
        )
    return steps


def _evaluator_smoke(args: argparse.Namespace) -> None:
    source = _validate_source(Path(args.source_root), args.revision)
    output = Path(args.output_directory).resolve()
    fixture = output / "fixture"
    overlay = output / "overlay"
    fixture.mkdir(parents=True, exist_ok=True)
    _write_fixture(fixture)
    config = _production_config(source, fixture, args.seed)
    config["test"]["glob"] = f"{{}}-seed{args.seed}-0.pt"
    config["test"]["glob_epoch"] = f"{{}}-seed{args.seed}-{{}}.pt"
    _write_overlay(overlay, config)
    workload = output / "smoke_queries.csv"
    base_cards = output / "smoke_base_cardinalities.json"
    predictions = output / "smoke_predictions.csv"
    latencies = output / "smoke_latency.csv"
    _write_smoke_workload(workload)
    _write_json(base_cards, {"movie_keyword,title": 128, "movie_info_idx,title": 128})
    started = time.perf_counter()
    _run_distjoin_evaluator(
        source=source,
        overlay=overlay,
        python=Path(args.python),
        queries=workload,
        base_cardinalities=base_cards,
        predictions=predictions,
        latencies=latencies,
        warmup_passes=1,
        repetitions=2,
        initialize_missing_checkpoints=True,
    )
    with predictions.open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with latencies.open(newline="", encoding="utf-8") as handle:
        latency_rows = list(csv.DictReader(handle))
    if len(prediction_rows) != 2 or len(latency_rows) != 4:
        raise RuntimeError("DistJoin evaluator-only fixture produced incomplete outputs")
    payload = {
        "status": "ok",
        "checkpoint_kind": "untrained_architecture_fixture",
        "query_count": len(prediction_rows),
        "latency_row_count": len(latency_rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "evaluator_smoke_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
    source = _validate_source(Path(args.source_root), args.revision)
    output = Path(args.output_directory).resolve()
    overlay = output / "overlay"
    completed = _run_distjoin_evaluator(
        source=source,
        overlay=overlay,
        python=Path(args.python),
        queries=Path(args.queries).resolve(),
        base_cardinalities=Path(args.base_cardinalities).resolve(),
        predictions=Path(args.predictions).resolve(),
        latencies=Path(args.latencies).resolve(),
        warmup_passes=args.warmup_passes,
        repetitions=args.repetitions,
    )
    print(completed.stdout)


def _run_distjoin_evaluator(
    *, source: Path, overlay: Path, python: Path, queries: Path,
    base_cardinalities: Path, predictions: Path, latencies: Path,
    warmup_passes: int, repetitions: int,
    experiment_mark: str = "production",
    initialize_missing_checkpoints: bool = False,
):
    command = [
        str(python), str(Path(__file__).resolve().parent / "distjoin_eval_runner.py"),
        "--source-root", str(source), "--overlay-root", str(overlay),
        "--experiment-mark", experiment_mark,
        "--queries", str(queries),
        "--base-cardinalities", str(base_cardinalities),
        "--predictions", str(predictions),
        "--latencies", str(latencies),
        "--warmup-passes", str(warmup_passes),
        "--repetitions", str(repetitions),
    ]
    if initialize_missing_checkpoints:
        command.append("--initialize-missing-checkpoints")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_compat_directory()), str(source), str(source / "MySampler"), environment.get("PYTHONPATH", "")]
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        command, cwd=overlay, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"DistJoin evaluation failed\nstdout:\n{completed.stdout[-12000:]}\n"
            f"stderr:\n{completed.stderr[-12000:]}"
        )
    return completed


def _production_config(source: Path, dataset: Path, seed: int) -> dict[str, Any]:
    config = _yaml_module().safe_load(
        (source / "Configs" / "IMDB" / "IMDB.yaml").read_text(encoding="utf-8")
    )
    config["seed"] = int(seed)
    config["data_dir"] = str(dataset) + "/"
    # Upstream uses this sentinel to select the IMDB-specific factorized
    # ordering. TrainTask obtains the six actual tables from JOB metadata.
    config["datasets"] = ["imdb"]
    config["excludes"] = []
    config["exclude_gpus"] = []
    config["num_gpu"] = 1
    config["tag"] = ""
    # Preserve the published architecture and dynamic-sampling recipe.
    required = {
        "use_pregen_data": False,
        "use_ANPM": True,
        "model_type": "MADE",
        "bs": 16_384,
        "sample_bs": 4_096,
        "max_steps": 256,
        "epochs": 20,
    }
    for key, expected in required.items():
        if config["train"].get(key) != expected:
            raise ValueError(f"unexpected upstream DistJoin setting {key}={config['train'].get(key)!r}")
    if "test" not in config:
        raise ValueError("upstream DistJoin config has no test section")
    config["test"]["glob"] = f"{{}}-seed{seed}-19.pt"
    config["test"]["glob_epoch"] = f"{{}}-seed{seed}-{{}}.pt"
    return config


def _write_overlay(overlay: Path, config: dict[str, Any]) -> None:
    config_dir = overlay / "Configs" / "IMDB"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "IMDB.yaml").write_text(
        _yaml_module().safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    for relative in ("train_log", "results", "log"):
        (overlay / relative).mkdir(parents=True, exist_ok=True)


def _run_upstream_train(
    source: Path,
    overlay: Path,
    python: Path,
    mark: str,
    tables: tuple[str, ...] = (),
):
    environment = os.environ.copy()
    paths = [str(_compat_directory()), str(source), str(source / "MySampler")]
    existing = environment.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment.setdefault("WANDB_MODE", "disabled")
    command = [
        str(python), str(Path(__file__).resolve().parent / "distjoin_train_runner.py"),
        "--source-root", str(source), "--exp-mark", mark,
    ]
    if tables:
        command.extend(["--tables", *tables])
    return subprocess.run(
        command,
        cwd=overlay,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_import_probe(source: Path, python: Path) -> dict[str, Any]:
    code = (
        "import json,torch,estimator; import AQP_estimator; import mysamplerAQP; "
        "assert AQP_estimator.DirectEstimator is estimator.DirectEstimator; "
        "print(json.dumps({'torch':torch.__version__, 'cuda':torch.cuda.is_available(), "
        "'sampler_module':mysamplerAQP.__file__, 'alias_ok':True}))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_compat_directory()), str(source), str(source / "MySampler"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [str(python), "-c", code], env=environment, text=True,
        cwd=source, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"DistJoin import probe failed: {result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _validate_source(source: Path, revision: str) -> Path:
    source = source.resolve()
    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != revision:
        raise ValueError(f"DistJoin revision mismatch: expected {revision}, got {observed}")
    for path in (source / "train.py", source / "estimator.py", source / "MySampler" / "setup.py"):
        if not path.exists():
            raise FileNotFoundError(path)
    return source


def _validate_dataset(dataset: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table in JOB_LIGHT_TABLES:
        path = dataset / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if "id" not in header:
            raise ValueError(f"{path} does not contain a CSV header")
        result[table] = {"path": str(path), "header": header, "sha256": _sha256(path)}
    return result


def _write_fixture(directory: Path) -> None:
    schemas = {
        "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
        "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
    }
    for table, header in schemas.items():
        path = directory / f"{table}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in range(1, 129):
                values = {
                    "id": index, "title": f"title-{index}",
                    "imdb_index": f"I{index % 5}",
                    "kind_id": index % 7 + 1, "production_year": 1950 + index % 75,
                    "imdb_id": index, "phonetic_code": f"P{index % 17}", "episode_of_id": "",
                    "season_nr": index % 10, "episode_nr": index % 50,
                    "series_years": f"{1950 + index % 20}-{1955 + index % 20}",
                    "md5sum": f"hash-{index}", "person_id": index, "movie_id": index,
                    "person_role_id": index, "note": "", "nr_order": index % 20,
                    "role_id": index % 11 + 1, "info_type_id": index % 71 + 1,
                    "info": str(index * 10), "keyword_id": index % 101 + 1,
                    "company_id": index % 53 + 1, "company_type_id": index % 2 + 1,
                }
                writer.writerow([values[column] for column in header])


def _write_smoke_workload(path: Path) -> None:
    path.write_text(
        "title t,movie_keyword mk#t.id=mk.movie_id#t.kind_id,=,2#19\n"
        "title t,movie_info_idx mi_idx#t.id=mi_idx.movie_id#mi_idx.info_type_id,=,5#2\n",
        encoding="utf-8",
    )


def _git_status(source: Path) -> list[str]:
    return subprocess.check_output(["git", "-C", str(source), "status", "--short"], text=True).splitlines()


def _compat_directory() -> Path:
    return Path(__file__).resolve().parent / "distjoin_compat"


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("DistJoin bridge requires PyYAML in its isolated environment") from exc
    return yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
