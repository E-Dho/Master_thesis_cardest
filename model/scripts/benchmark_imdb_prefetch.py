"""IMDB step prefetch on the real live sampler: identical training, supply rate, memory.

1. Identity: trains ``--identity-steps`` steps on the CPU three times, each in a
   fresh Python process (as a training job starts): twice without workers and
   once with ``--identity-workers``, and compares losses, metrics (timings aside),
   source counters and the final checkpoint (model and optimizer state). The two
   serial runs are the control for run-to-run reproducibility of the sampler.
   Exit code 0: identical; 1: the prefetch run differs; 2: inconclusive (the
   serial runs already differ).
2. Supply rate and memory: for each worker count, prepares ``--steps`` steps (main
   batch, contexts, weights, rare batch, token IDs) exactly as
   ``training.prefetch_workers`` does, without the model, and reports seconds per
   step after a warm-up and the memory (PSS) of the training process and its
   prefetch processes. Worker count 0 is the serial preparation for reference. On
   the GPU, a step then takes about max(prefetch time per step, model time per step).

As in ``profile_training_phases``, the sample sources are built from the
configuration as is (rare-support planning sees the configured step count); only
the training length, device, checkpoints and output directory are overridden.

    python -m model.scripts.benchmark_imdb_prefetch --config <training_config.yaml> \\
        --workers 0,2,3,4 --steps 400 --output-directory <empty dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _pss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


def _children(pid: int) -> list[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as handle:
            return [int(value) for value in handle.read().split()]
    except OSError:
        return []


def _memory() -> dict[str, float]:
    pid = os.getpid()
    children = _children(pid)
    return {"parent_pss_mb": round(_pss_mb(pid), 1),
            "prefetch_pss_mb": round(sum(_pss_mb(child) for child in children), 1),
            "prefetch_processes": len(children)}


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items()
            if "timing" not in k and "per_second" not in k and "seconds" not in k
            and k not in ("prefetch_workers", "checkpoint", "metrics_path", "summary_path",
                          "checkpoint_path", "best_checkpoint_path")}


def _identity_run(args) -> int:
    """One identity training in this (fresh) process; writes ``identity_result.json``."""

    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.training import resmade_trainer as trainer

    config = load_simple_yaml(args.config)
    validate_config(config)
    source = sample_source_from_config(config)
    directory = args.output_directory
    config["training"].update({
        "steps_per_epoch": int(args.identity_steps), "epochs": 1, "device": "cpu",
        "checkpoint_interval_steps": 0, "early_stopping_patience_steps": 0,
        "prefetch_workers": int(args.identity_run_workers),
    })
    config.setdefault("logging", {})["output_directory"] = str(directory)
    started = time.perf_counter()
    result = trainer.train_resmade_sample_source(source, config)
    (directory / "identity_result.json").write_text(json.dumps({
        "workers": int(args.identity_run_workers),
        "pid": os.getpid(),
        "first_loss": result.first_loss, "last_loss": result.last_loss,
        "checkpoint_path": str(result.checkpoint_path),
        "counters": trainer._source_counters(source),
        "seconds": time.perf_counter() - started,
    }, indent=2) + "\n")
    return 0


def _compare_runs(left: Path, right: Path) -> list[str]:
    import torch

    a_info = json.loads((left / "identity_result.json").read_text())
    b_info = json.loads((right / "identity_result.json").read_text())
    mismatches = []
    if (a_info["first_loss"], a_info["last_loss"]) != (b_info["first_loss"], b_info["last_loss"]):
        mismatches.append("first/last loss")
    rows = [[_strip(json.loads(line)) for line in (d / "training_metrics.jsonl").read_text().splitlines()]
            for d in (left, right)]
    if rows[0] != rows[1]:
        mismatches.append("training_metrics.jsonl")
    if a_info["counters"] != b_info["counters"]:
        mismatches.append(f"source counters {a_info['counters']} vs {b_info['counters']}")
    a = torch.load(a_info["checkpoint_path"], weights_only=False)
    b = torch.load(b_info["checkpoint_path"], weights_only=False)
    if not all(torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]) for k in a["model_state_dict"]):
        mismatches.append("checkpoint model_state_dict")
    for index, state in a["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.equal(value, b["optimizer_state_dict"]["state"][index][key]):
                mismatches.append(f"optimizer state {index}.{key}")
    return mismatches


def _identity_verdict(serial_mismatches: list[str], prefetch_mismatches: list[str]) -> str:
    """identical / different / inconclusive (two serial runs already differ)."""

    if serial_mismatches:
        return "inconclusive"
    return "different" if prefetch_mismatches else "identical"


def _identity(args, config_path: str) -> dict:
    """Two serial runs and one prefetch run, each in a fresh Python process.

    A fresh process per run starts every random state the way a training job
    does, including state that lives inside NeuroCard's modules and objects
    (which resetting Python's, numpy's and torch's global generators would not
    reach). The second serial run is the control: if it already differs from the
    first, the sampler is not reproducible run to run and the comparison with the
    prefetch run says nothing (``inconclusive``).
    """

    import subprocess
    import sys

    runs = {}
    for label, workers in (("serial_a", 0), ("serial_b", 0), ("prefetch", int(args.identity_workers))):
        directory = args.output_directory / f"identity_{label}"
        directory.mkdir(parents=True)
        command = [sys.executable, "-m", "model.scripts.benchmark_imdb_prefetch", "--config", config_path,
                   "--identity-steps", str(int(args.identity_steps)), "--identity-run-workers", str(workers),
                   "--output-directory", str(directory)]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        (directory / "run.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            raise SystemExit(f"identity run {label} failed (exit {completed.returncode}); see {directory}/run.log")
        runs[label] = {"directory": directory, "seconds": time.perf_counter() - started}
    serial_mismatches = _compare_runs(runs["serial_a"]["directory"], runs["serial_b"]["directory"])
    prefetch_mismatches = _compare_runs(runs["serial_a"]["directory"], runs["prefetch"]["directory"])
    losses = {label: [json.loads((run["directory"] / "identity_result.json").read_text())[key]
                      for key in ("first_loss", "last_loss")] for label, run in runs.items()}
    verdict = _identity_verdict(serial_mismatches, prefetch_mismatches)
    return {
        "steps": int(args.identity_steps), "workers": int(args.identity_workers),
        "verdict": verdict, "identical": verdict == "identical",
        "serial_control_mismatches": serial_mismatches, "mismatches": prefetch_mismatches,
        "losses": losses, "wall_seconds": {label: run["seconds"] for label, run in runs.items()},
    }


def _supply(args, config_path: str) -> dict:
    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.predicates.generation import PredicateTrainingContextGenerator
    from model.src.training import resmade_trainer as trainer
    config = load_simple_yaml(config_path)
    validate_config(config)
    started = time.perf_counter()
    source = sample_source_from_config(config)
    metadata = source.metadata
    vocabularies = trainer.predicate_vocabularies_from_config(metadata, config)
    generator = PredicateTrainingContextGenerator(config.get("predicate_generation", {}))
    setup_seconds = time.perf_counter() - started
    batch_size = int(config["training"]["batch_size"])
    seed = int(config["training"].get("seed", 0))

    def prepare(batch, step: int, rare_batch=None):
        return trainer.prepare_step_inputs(batch, metadata=metadata, vocabularies=vocabularies, config=config,
                                           context_generator=generator, sample_source=source, global_step=step,
                                           compute_expensive_diagnostics=False, rare_batch=rare_batch)

    draw_rare = ((lambda step: trainer._draw_rare_batch(source, config, step))
                 if trainer._rare_auxiliary_enabled(config) else None)

    report = {"batch_size": batch_size, "setup_seconds": setup_seconds, "memory_after_setup": _memory(), "runs": []}
    print(json.dumps({"setup_seconds": setup_seconds, **_memory()}), flush=True)
    warmup = int(args.warmup_steps)
    for workers in [int(value) for value in args.workers.split(",") if value != ""]:
        samples = []
        steps = int(args.serial_steps if workers == 0 else args.steps)
        if workers == 0:
            begin = None
            for step in range(steps):
                if step == warmup:
                    begin = time.perf_counter()
                batch = source.batches(batch_size, seed=seed + step)
                prepare(batch, step)
        else:
            pipeline = trainer._StepPipeline(sample_source=source, prepare=prepare, metadata=metadata,
                                             batch_size=batch_size, seed=seed, total_steps=steps, workers=workers,
                                             depth=int(args.prefetch_factor), draw_rare=draw_rare)
            begin = None
            try:
                for step in range(steps):
                    if step == warmup:
                        begin = time.perf_counter()
                    pipeline.next(step)
                    if step % max(1, steps // 10) == 0 or step == steps - 1:
                        samples.append({"step": step, **_memory()})
            finally:
                pipeline.close()
        elapsed = time.perf_counter() - begin
        run = {"workers": workers, "steps": steps, "timed_steps": steps - warmup,
               "seconds_per_step": elapsed / max(1, steps - warmup), "memory_samples": samples}
        report["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "memory_samples"}),
              json.dumps(max(samples, key=lambda s: s["parent_pss_mb"] + s["prefetch_pss_mb"])) if samples else "",
              flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", default="0,2,3,4")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--serial-steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--identity-steps", type=int, default=40)
    parser.add_argument("--identity-workers", type=int, default=2)
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--identity-run-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    if args.identity_run_workers is not None:
        return _identity_run(args)
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"{args.output_directory} is not empty; refusing")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    report = {"config": args.config}
    if not args.skip_identity:
        report["identity"] = _identity(args, args.config)
        print(json.dumps({"identity": report["identity"]}), flush=True)
    report["supply"] = _supply(args, args.config)
    (args.output_directory / "prefetch_benchmark.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    verdict = report.get("identity", {}).get("verdict", "identical")
    return {"identical": 0, "different": 1, "inconclusive": 2}[verdict]


if __name__ == "__main__":
    raise SystemExit(main())
