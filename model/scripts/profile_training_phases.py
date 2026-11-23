"""Where a training step spends its time: wall-clock totals per function over a short run.

Trains ``--steps`` steps of a real configuration (rare-support planning still sees the
configured step count) and times the step's parts by wrapping the functions the
trainer calls: batch sampling, query-context generation (and the per-column cache
builds inside it), diagnostics, weights, rare batches, token encoding, loss, backward,
optimizer, checkpoint saves, validation and metrics. Totals are inclusive (nested
calls are counted in both). Works for any trainer version: functions it does not
find are skipped. On a CPU node the model's forward/backward run on the CPU; the
CPU-side parts are what this is for.

    python -m model.scripts.profile_training_phases --config <training_config.yaml> \\
        --steps 200 --device cpu --checkpoint-interval 100 --output-directory <empty dir>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def _write_cprofile(profiler, directory: Path) -> None:
    """Top functions by cumulative time, and the call tree below the sampling functions."""

    import io
    import pstats

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative").print_stats(80)
    (directory / "cprofile_top.txt").write_text(stream.getvalue())
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    for pattern in ("batches", "rare_batches", "_sample_conditionals_from_source", "sample_root_strata_rows",
                    "_encode", "encode"):
        stats.print_callees(pattern)
    (directory / "cprofile_sampling.txt").write_text(stream.getvalue())
    profiler.dump_stats(str(directory / "training.pstats"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--cprofile", action="store_true",
                        help="also run cProfile over training; writes cprofile_top.txt and cprofile_sampling.txt")
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"{args.output_directory} is not empty; refusing")

    import torch

    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.predicates import generation
    from model.src.training import resmade_trainer as trainer

    totals: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)

    def timed(name, function):
        def wrapper(*a, **k):
            started = time.perf_counter()
            try:
                return function(*a, **k)
            finally:
                totals[name] += time.perf_counter() - started
                calls[name] += 1
        return wrapper

    for name in ("token_coverage", "literal_token_occurrences", "predicate_context_diagnostics",
                 "cumulative_inverse_fanout_weights", "rare_batches", "rare_auxiliary_diagnostics",
                 "encode_tokens_tensor", "encode_contexts_tensor", "torch_weighted_per_head_cross_entropy",
                 "_save_checkpoint", "_run_validation", "literal_token_stats", "_append_metrics",
                 "_predicate_embedding_gradient_coverage", "_merge_literal_occurrences", "_merge_coverage",
                 "trajectory_dedup_loss_for_batch", "PredicateTrainingContextGenerator"):
        if hasattr(trainer, name) and name != "PredicateTrainingContextGenerator":
            setattr(trainer, name, timed(name, getattr(trainer, name)))
    generator_class = generation.PredicateTrainingContextGenerator
    for method in ("generate_batch", "generate_forced_stratum_batch", "_column_caches", "_compiled"):
        if hasattr(generator_class, method):
            setattr(generator_class, method, timed(f"generator.{method}", getattr(generator_class, method)))
    original_init = generator_class.__init__
    def counting_init(self, *a, **k):
        calls["generator.__init__ (new generator objects)"] += 1
        return original_init(self, *a, **k)
    generator_class.__init__ = counting_init
    torch.Tensor.backward = timed("backward", torch.Tensor.backward)
    for optimizer_class in (torch.optim.Adam, torch.optim.AdamW):
        optimizer_class.step = timed("optimizer.step", optimizer_class.step)

    config = load_simple_yaml(args.config)
    validate_config(config)
    started = time.perf_counter()
    source = sample_source_from_config(config)
    setup_seconds = time.perf_counter() - started
    source.batches = timed("sample_source.batches", source.batches)
    if hasattr(source, "rare_batches"):
        source.rare_batches = timed("sample_source.rare_batches", source.rare_batches)
    config["training"]["steps_per_epoch"] = int(args.steps)
    config["training"]["epochs"] = 1
    config["training"]["device"] = args.device
    config["training"]["checkpoint_interval_steps"] = int(args.checkpoint_interval)
    config["training"]["early_stopping_patience_steps"] = 0
    config.setdefault("logging", {})["output_directory"] = str(args.output_directory)
    profiler = None
    if args.cprofile:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    started = time.perf_counter()
    result = trainer.train_resmade_sample_source(source, config)
    wall = time.perf_counter() - started
    if profiler is not None:
        profiler.disable()
        _write_cprofile(profiler, args.output_directory)
    report = {
        "config": args.config,
        "steps": args.steps,
        "device": args.device,
        "setup_seconds": setup_seconds,
        "train_wall_seconds": wall,
        "training_seconds": getattr(result, "training_seconds", None),
        "seconds_per_step": wall / max(1, args.steps),
        "per_function": {
            name: {"total_seconds": round(totals[name], 3), "calls": calls[name],
                   "seconds_per_step": round(totals[name] / max(1, args.steps), 4)}
            for name in sorted(totals, key=totals.get, reverse=True)
        },
        "counts_only": {k: v for k, v in calls.items() if k not in totals},
    }
    (args.output_directory / "phase_profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
