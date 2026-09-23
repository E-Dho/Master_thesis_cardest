#!/usr/bin/env python3
"""Isolated native NeuroCard training and evaluation bridge."""

from __future__ import annotations

import argparse
import collections
import copy
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


MODEL_CONFIGS = {
    "job_light": {
        "experiment": "job-light",
        "pretrained": "job-light-pretrained.pt",
    },
    "job_light_ranges": {
        "experiment": "job-light-ranges",
        "pretrained": "job-light-ranges-pretrained.pt",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_common_arguments(prepare)
    prepare.add_argument("--revision", required=True)
    prepare.add_argument("--checkpoint")
    prepare.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate")
    _add_common_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--latency", required=True)
    evaluate.add_argument("--psamples", type=int, default=8000)
    evaluate.add_argument("--query-limit", type=int)
    evaluate.add_argument("--warmup-passes", type=int)
    evaluate.add_argument("--repetitions", type=int)
    evaluate.add_argument("--smoke", action="store_true")

    train = subparsers.add_parser("train")
    _add_common_arguments(train)
    train.add_argument("--checkpoint-output", required=True)
    train.add_argument("--artifact-manifest", required=True)
    train.add_argument("--training-metrics", required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--loader-workers", type=int)

    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "evaluate":
        _evaluate(args)
    elif args.command == "train":
        _train(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-kind", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)


def _prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    package_root = source_root / "neurocard"
    observed_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed_revision != args.revision:
        raise ValueError(
            f"NeuroCard revision mismatch: expected {args.revision}, "
            f"observed {observed_revision}"
        )
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(source_root), "diff", "--no-ext-diff", "HEAD"]
    )
    tracked_diff_sha256 = hashlib.sha256(tracked_diff).hexdigest()
    runtime_extension = package_root / "factorized_sampler_lib" / "rustlib.so"
    required_source_files = (
        package_root / "run.py",
        package_root / "experiments.py",
        package_root / "estimators.py",
    )
    for path in required_source_files:
        if not path.exists():
            raise FileNotFoundError(path)
    required_tables = (
        "cast_info",
        "movie_companies",
        "movie_info",
        "movie_keyword",
        "title",
        "movie_info_idx",
    )
    missing_csv = [str(dataset_root / f"{table}.csv") for table in required_tables
                   if not (dataset_root / f"{table}.csv").exists()]
    if missing_csv:
        raise FileNotFoundError(f"missing IMDB CSV files: {missing_csv}")
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else package_root / "models" / MODEL_CONFIGS[args.model_kind]["pretrained"]
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    modules = _load_upstream(source_root, args.device, args.cpu_threads)
    experiment = MODEL_CONFIGS[args.model_kind]["experiment"]
    config = _resolved_experiment_config(modules["experiments"], experiment)
    spec = modules["join_utils"].get_join_spec(config)
    prepare_utils = modules["prepare_utils"]
    cache_hit = bool(prepare_utils.check_required_files(spec))
    if not cache_hit:
        raise FileNotFoundError(
            f"native NeuroCard sampler cache is incomplete for {spec.join_name}"
        )
    payload = {
        "status": "ready",
        "source_root": str(source_root),
        "source_revision": observed_revision,
        "source_dirty": bool(source_status),
        "source_status": source_status.splitlines(),
        "source_tracked_diff_sha256": tracked_diff_sha256,
        "source_tracked_diff_bytes": len(tracked_diff),
        "sampler_runtime_extension_sha256": (
            _sha256_file(runtime_extension) if runtime_extension.exists() else None
        ),
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "model_kind": args.model_kind,
        "upstream_experiment": experiment,
        "sampler_cache_hit": cache_hit,
        "sampler_join_name": spec.join_name,
        "versions": _versions(modules),
    }
    output = Path(args.output)
    _write_json(output, payload)
    if tracked_diff:
        patch_path = output.resolve().with_name("neurocard_source_tracked_diff.patch")
        patch_path.write_bytes(tracked_diff)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _evaluate(args: argparse.Namespace) -> None:
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
    if args.psamples <= 0:
        raise ValueError("psamples must be positive")

    setup_started = time.perf_counter()
    runtime = _NativeRuntime(
        source_root=Path(args.source_root),
        dataset_root=Path(args.dataset_root),
        model_kind=args.model_kind,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        checkpoint=Path(args.checkpoint),
        training=False,
    )
    setup_seconds = time.perf_counter() - setup_started
    queries, _ = runtime.load_queries(Path(args.queries))
    if args.query_limit is not None:
        queries = queries[: args.query_limit]
    estimator = runtime.make_estimator(args.psamples)

    failures: dict[int, str] = {}
    for _ in range(warmup_passes):
        for query_id, query in enumerate(queries):
            if query_id in failures:
                continue
            try:
                estimator.Query(*query)
                runtime.synchronize()
            except Exception as exc:  # keep failed queries explicit
                failures[query_id] = f"{type(exc).__name__}: {exc}"

    estimates: dict[int, float] = {}
    latency_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for query_id, query in enumerate(queries):
            if query_id in failures:
                continue
            try:
                runtime.synchronize()
                started = time.perf_counter()
                estimate = float(estimator.Query(*query))
                runtime.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if not math.isfinite(estimate) or estimate < 0:
                    raise ValueError(f"invalid estimate {estimate}")
                if repetition == 0:
                    estimates[query_id] = estimate
                latency_rows.append(
                    {
                        "query_id": query_id,
                        "repetition": repetition,
                        "latency_ms": elapsed_ms,
                        "scope": "predicate_encoding_and_native_progressive_sampling",
                        "device": args.device,
                        "device_name": (
                            str(runtime.torch.cuda.get_device_name(runtime.device))
                            if runtime.device.type == "cuda"
                            else "CPU"
                        ),
                    }
                )
            except Exception as exc:
                failures[query_id] = f"{type(exc).__name__}: {exc}"

    prediction_rows = []
    for query_id in range(len(queries)):
        if query_id in failures:
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "estimated_cardinality": "",
                    "status": "failed",
                    "diagnostic": failures[query_id],
                }
            )
        else:
            prediction_rows.append(
                {
                    "query_id": query_id,
                    "estimated_cardinality": estimates[query_id],
                    "status": "ok",
                    "diagnostic": "",
                }
            )
    _write_csv(
        Path(args.predictions),
        prediction_rows,
        ("query_id", "estimated_cardinality", "status", "diagnostic"),
    )
    _write_csv(
        Path(args.latency),
        latency_rows,
        ("query_id", "repetition", "latency_ms", "scope", "device", "device_name"),
    )
    payload = {
        "status": "ok" if not failures else "partial_failure",
        "query_count": len(queries),
        "success_count": len(queries) - len(failures),
        "failure_count": len(failures),
        "psamples": args.psamples,
        "warmup_passes": warmup_passes,
        "repetitions": repetitions,
        "setup_seconds_excluded_from_timing": setup_seconds,
        "parameter_count": runtime.parameter_count,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.smoke and failures:
        raise RuntimeError(f"NeuroCard smoke failed: {failures}")


def _train(args: argparse.Namespace) -> None:
    setup_started = time.perf_counter()
    runtime = _NativeRuntime(
        source_root=Path(args.source_root),
        dataset_root=Path(args.dataset_root),
        model_kind=args.model_kind,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        checkpoint=None,
        training=True,
        overrides={
            key: value
            for key, value in {
                "epochs": args.epochs,
                "max_steps": args.max_steps,
                "bs": args.batch_size,
                "loader_workers": args.loader_workers,
            }.items()
            if value is not None
        },
    )
    setup_seconds = time.perf_counter() - setup_started
    if runtime.device.type == "cuda":
        runtime.torch.cuda.reset_peak_memory_stats(runtime.device)

    optimizer, scheduler, custom_lr = runtime.make_optimizer()
    configured_warmups = runtime.config["warmups"]
    effective_warmups = configured_warmups
    if scheduler is None and custom_lr is None and configured_warmups < 1:
        planned_warmup_steps = int(
            configured_warmups
            * int(runtime.config["max_steps"])
            * int(runtime.config["epochs"])
        )
        if planned_warmup_steps == 0:
            # Upstream's inverse-power schedule cannot evaluate t**-1.5 at
            # t=0. This only changes tiny smoke overrides; published budgets
            # have hundreds of warmup steps and retain their exact schedule.
            effective_warmups = 1
    losses: list[float] = []
    train_started = time.perf_counter()
    for epoch in range(runtime.config["epochs"]):
        loss = runtime.run_module.run_epoch(
            "train",
            runtime.model,
            optimizer,
            upto=runtime.config["max_steps"],
            train_data=runtime.train_data,
            val_data=runtime.train_data,
            batch_size=runtime.config["bs"],
            epoch_num=epoch,
            epochs=runtime.config["epochs"],
            log_every=100,
            table_bits=0,
            warmups=effective_warmups,
            loader=runtime.loader,
            constant_lr=runtime.config["constant_lr"],
            summary_writer=None,
            lr_scheduler=scheduler,
            custom_lr_lambda=custom_lr,
            label_smoothing=runtime.config["label_smoothing"],
        )
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError(f"non-finite NeuroCard training loss at epoch {epoch}: {loss}")
        losses.append(loss)
    train_seconds = time.perf_counter() - train_started

    checkpoint = Path(args.checkpoint_output).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime.torch.save(runtime.model.state_dict(), checkpoint)
    peak_allocated = peak_reserved = None
    if runtime.device.type == "cuda":
        peak_allocated = int(runtime.torch.cuda.max_memory_allocated(runtime.device))
        peak_reserved = int(runtime.torch.cuda.max_memory_reserved(runtime.device))
    nominal_tuples = (
        int(runtime.config["bs"])
        * int(runtime.config["max_steps"])
        * int(runtime.config["epochs"])
    )
    metrics = {
        "model_kind": args.model_kind,
        "seed": args.seed,
        "device": str(runtime.device),
        "setup_seconds": setup_seconds,
        "training_seconds": train_seconds,
        "total_build_seconds": setup_seconds + train_seconds,
        "epochs": int(runtime.config["epochs"]),
        "max_steps": int(runtime.config["max_steps"]),
        "batch_size": int(runtime.config["bs"]),
        "optimizer_steps": int(runtime.config["epochs"])
        * int(runtime.config["max_steps"]),
        "nominal_sampled_tuples": nominal_tuples,
        "configured_warmups": configured_warmups,
        "effective_warmups": effective_warmups,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_by_epoch": losses,
        "parameter_count": runtime.parameter_count,
        "peak_training_gpu_allocated_bytes": peak_allocated,
        "peak_training_gpu_reserved_bytes": peak_reserved,
    }
    artifact = {
        "parameter_count": runtime.parameter_count,
        "trainable_parameter_count": runtime.trainable_parameter_count,
        "serialized_model_mb": checkpoint.stat().st_size / 1_000_000.0,
        "full_checkpoint_mb": checkpoint.stat().st_size / 1_000_000.0,
        "checkpoint_format": "PyTorch state_dict",
        "checkpoint_includes_optimizer": False,
        "checkpoint": str(checkpoint),
        "upstream_experiment": MODEL_CONFIGS[args.model_kind]["experiment"],
        "progressive_samples": 8000,
        "training": metrics,
        "peak_training_gpu_allocated_bytes": peak_allocated,
        "peak_training_gpu_reserved_bytes": peak_reserved,
    }
    _write_json(Path(args.training_metrics), metrics)
    _write_json(Path(args.artifact_manifest), artifact)
    print(json.dumps(metrics, indent=2, sort_keys=True))


class _NativeRuntime:
    def __init__(
        self,
        *,
        source_root: Path,
        dataset_root: Path,
        model_kind: str,
        seed: int,
        device: str,
        cpu_threads: int,
        checkpoint: Path | None,
        training: bool,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.source_root = source_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.modules = _load_upstream(self.source_root, device, cpu_threads)
        self.torch = self.modules["torch"]
        if device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        self.device = self.torch.device(device)
        self.modules["train_utils"].get_device = lambda: self.device
        self.run_module = self.modules["run"]
        self.config = _resolved_experiment_config(
            self.modules["experiments"], MODEL_CONFIGS[model_kind]["experiment"]
        )
        self.config.update(overrides or {})
        self.config.update(
            {
                "seed": int(seed),
                "cwd": str(self.source_root),
                "checkpoint_to_load": str(checkpoint.resolve()) if checkpoint else None,
                "__run": MODEL_CONFIGS[model_kind]["experiment"],
                "__gpu": 1 if self.device.type == "cuda" else 0,
                "__cpu": int(cpu_threads),
            }
        )
        self.torch.set_num_threads(max(1, int(cpu_threads)))
        import random as _random
        _random.seed(int(seed))
        self.torch.manual_seed(int(seed))
        self.modules["np"].random.seed(int(seed))
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(int(seed))

        runner = object.__new__(self.run_module.NeuroCard)
        runner.config = self.config
        for key, value in self.config.items():
            setattr(runner, key, value)
        loaded_tables = [
            self.modules["datasets"].LoadImdb(
                table,
                data_dir=str(self.dataset_root) + "/",
                use_cols=self.config["use_cols"],
                try_load_parsed=True,
            )
            for table in self.config["join_tables"]
        ]
        join_spec = self.modules["join_utils"].get_join_spec(self.config)
        prepare_utils = self.modules["prepare_utils"]
        if not prepare_utils.check_required_files(join_spec):
            raise FileNotFoundError(
                f"native NeuroCard cache is incomplete for {join_spec.join_name}"
            )
        # Upstream prepare() initializes Ray before testing its immutable cache.
        self.modules["factorized_sampler"].prepare_utils.prepare = lambda _: None
        join_spec, train_data, loader, table = runner.MakeSamplerDatasetLoader(loaded_tables)
        table.cardinality = self.modules[
            "datasets"
        ].JoinOrderBenchmark.GetFullOuterCardinalityOrFail(self.config["join_tables"])
        train_data.cardinality = table.cardinality
        runner.join_spec = join_spec
        runner.train_data = train_data
        runner.loader = loader
        runner.table = table
        runner.fixed_ordering = runner.MakeOrdering(table)
        title_index = [table.name for table in loaded_tables].index("title")
        model = runner.MakeModel(table, train_data, table_primary_index=title_index)
        if not isinstance(model, self.modules["transformer"].Transformer):
            model.apply(self.modules["train_utils"].weight_init)
        runner.model = model
        if checkpoint is not None:
            _load_checkpoint(self.torch, model, checkpoint.resolve())
        self.runner = runner
        self.model = model
        self.table = table
        self.train_data = train_data
        self.loader = loader
        self.join_spec = join_spec
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )

    def load_queries(self, path: Path):
        raw = self.modules["utils"].JobToQuery(str(path.resolve()))
        return self.modules["utils"].UnpackQueries(self.table, raw)

    def make_estimator(self, progressive_samples: int):
        original = self.runner.eval_psamples
        self.runner.eval_psamples = [int(progressive_samples)]
        try:
            estimator = self.runner.MakeProgressiveSamplers(
                self.model,
                self.train_data,
                do_fanout_scaling=True,
            )[0]
            # Current PyTorch requires reusable ``out=`` buffers to be empty
            # before their first resize. This preserves upstream semantics and
            # avoids the deprecated implicit resize from the one-row probe.
            for output in estimator.logits_outs:
                output.resize_(0)
            return estimator
        finally:
            self.runner.eval_psamples = original

    def make_optimizer(self):
        torch = self.torch
        if self.config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(list(self.model.parameters()), 2e-4)
        else:
            optimizer = torch.optim.Adagrad(list(self.model.parameters()), 2e-4)
        total_steps = int(self.config["epochs"]) * int(self.config["max_steps"])
        scheduler_name = self.config["lr_scheduler"]
        scheduler = custom_lr = None
        if scheduler_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        elif scheduler_name == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=2e-3, total_steps=total_steps
            )
        elif scheduler_name and scheduler_name.startswith("OneCycleLR-"):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=2e-3,
                total_steps=total_steps,
                pct_start=float(scheduler_name.split("-")[-1]),
            )
        elif scheduler_name and scheduler_name.startswith("wd_"):
            _, learning_rate, warmup_fraction = scheduler_name.split("_")
            custom_lr = self.modules["train_utils"].get_cosine_learning_rate_fn(
                total_steps,
                learning_rate=float(learning_rate),
                min_learning_rate_mult=1e-5,
                constant_fraction=0.0,
                warmup_fraction=float(warmup_fraction),
            )
            scheduler = object()
        elif scheduler_name is not None:
            raise ValueError(f"unsupported NeuroCard scheduler {scheduler_name}")
        return optimizer, scheduler, custom_lr

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


def _load_upstream(source_root: Path, device: str, cpu_threads: int) -> dict[str, Any]:
    package_root = source_root.resolve() / "neurocard"
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    os.chdir(source_root.resolve())
    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
    old_argv = sys.argv
    sys.argv = [str(package_root / "run.py")]
    try:
        import datasets
        import experiments
        import factorized_sampler
        import join_utils
        import run
        import train_utils
        import transformer
        import utils
        from factorized_sampler_lib import prepare_utils
        import torch
    finally:
        sys.argv = old_argv
    torch.set_num_threads(max(1, int(cpu_threads)))
    return {
        "datasets": datasets,
        "experiments": experiments,
        "factorized_sampler": factorized_sampler,
        "join_utils": join_utils,
        "np": np,
        "prepare_utils": prepare_utils,
        "run": run,
        "torch": torch,
        "train_utils": train_utils,
        "transformer": transformer,
        "utils": utils,
    }


def _resolved_experiment_config(experiments: Any, name: str) -> dict[str, Any]:
    return _resolve_grid_values(copy.deepcopy(experiments.EXPERIMENT_CONFIGS[name]))


def _resolve_grid_values(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"grid_search"}:
            choices = value["grid_search"]
            if len(choices) != 1:
                raise ValueError(f"expected one upstream grid choice, received {choices}")
            return _resolve_grid_values(choices[0])
        return {key: _resolve_grid_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_grid_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_grid_values(item) for item in value)
    return value


def _load_checkpoint(torch: Any, model: Any, path: Path) -> None:
    try:
        state = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
    except TypeError:  # PyTorch before weights_only was added.
        state = torch.load(path, map_location=next(model.parameters()).device)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError as original_error:
        renamed = collections.OrderedDict(
            (
                key.replace("embedding_networks", "embeddings")
                if key.startswith("embedding_networks")
                else key,
                value,
            )
            for key, value in state.items()
        )
        modules = list(model.net.children())
        if len(modules) < 2 or modules[-2].__class__.__name__ != "ReLU":
            raise original_error
        import warnings
        warnings.warn(
            f"Checkpoint at {path} used an older architecture (trailing ReLU + "
            "'embedding_networks' key names).  Applying compatibility patch: "
            "removing the trailing ReLU and renaming embedding keys.  "
            "Re-save the checkpoint with the current architecture to silence this.",
            stacklevel=2,
        )
        modules.pop(-2)
        model.net = torch.nn.Sequential(*modules)
        model.load_state_dict(renamed)


def _versions(modules: dict[str, Any]) -> dict[str, str]:
    import pandas
    import ray

    return {
        "numpy": modules["np"].__version__,
        "pandas": pandas.__version__,
        "ray": ray.__version__,
        "torch": modules["torch"].__version__,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
