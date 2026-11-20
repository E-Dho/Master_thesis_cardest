"""Throughput and memory of the step-prefetch workers on a real preparation (no GPU, no training).

For each worker count, prepares ``--steps`` training steps (the CPU part of a step:
batch, query contexts, inverse-fanout weights, rare batch, dedup-target lookups)
exactly as ``training.prefetch_workers`` does, and reports steps per second and
the memory of the training process plus its workers (PSS: shared pages counted
once) over time. Worker count 0 is the serial step for reference. The model's own
part of a step (forward, backward, optimizer) is not included: training runs at
about min(prefetch rate, 1 / GPU time per step).

    python -m model.scripts.benchmark_prefetch --config model/configs/<run>.yaml \\
        --workers 0,2,4,6 --steps 400 --output prefetch_benchmark.json
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
    workers = _children(pid)
    return {"parent_pss_mb": _pss_mb(pid), "workers_pss_mb": sum(_pss_mb(w) for w in workers),
            "worker_processes": len(workers)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", default="0,2,4,6")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--serial-steps", type=int, default=50)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite")

    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.predicates.generation import PredicateTrainingContextGenerator
    from model.src.training import resmade_trainer as trainer

    config = load_simple_yaml(args.config)
    validate_config(config)
    started = time.perf_counter()
    source = sample_source_from_config(config)
    metadata = source.metadata
    generator = PredicateTrainingContextGenerator(trainer._predicate_generation_config(config))
    generator.warm_caches(metadata)
    setup_seconds = time.perf_counter() - started
    batch_size = int(config["training"]["batch_size"])

    def prepare(step: int):
        return trainer.prepare_step_inputs(sample_source=source, metadata=metadata, config=config,
                                           context_generator=generator, global_step=step,
                                           batch_size=batch_size, compute_expensive_diagnostics=False)

    report = {"config": args.config, "batch_size": batch_size, "setup_seconds": setup_seconds,
              "memory_after_setup": _memory(), "runs": []}
    print(json.dumps({"setup_seconds": setup_seconds, **_memory()}), flush=True)
    for workers in [int(value) for value in args.workers.split(",") if value != ""]:
        samples = []
        if workers == 0:
            steps = args.serial_steps
            begin = time.perf_counter()
            for step in range(steps):
                prepare(step)
            elapsed = time.perf_counter() - begin
        else:
            steps = args.steps
            iterator = trainer._prefetched_steps(prepare, total_steps=steps, workers=workers,
                                                 prefetch_factor=args.prefetch_factor)
            begin = time.perf_counter()
            for index in range(steps):
                prepared = trainer._with_column_metadata(next(iterator), metadata)
                if prepared.global_step != index:
                    raise SystemExit(f"step {prepared.global_step} arrived for {index}")
                if index % max(1, steps // 10) == 0 or index == steps - 1:
                    samples.append({"step": index, "seconds": time.perf_counter() - begin, **_memory()})
            elapsed = time.perf_counter() - begin
            del iterator
        run = {"workers": workers, "steps": steps, "seconds": elapsed,
               "seconds_per_step": elapsed / steps, "memory_samples": samples}
        report["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "memory_samples"}),
              json.dumps(samples[-1]) if samples else "", flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
