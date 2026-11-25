"""Widths of training intervals and windows with predicate_generation.log_width_gate.

Draws main batches from the configured sample source (as training does), generates
their contexts with the trainer's predicate generator configuration (training's
_predicate_generation_config, which adds the top-level trajectory_spatial section,
i.e. the segment-MBR windows) at each gate probability (the config's log_width_gate
with only the probability replaced), validating every context, and reports per
probability: generation seconds per batch, rejected contexts, and the width
distribution of
  - segment temporal contexts (t_s < U, t_e >= L): U - L,
  - trip-time ordinary predicates (start_time <= U and end_time >= L): U - L,
  - segment windows: max_x - min_x and max_y - min_y.
Rare support is switched off (contexts only).

    python -m model.scripts.diagnose_log_width_gate --config <training config> \\
        --batches 50 --probabilities 0,0.5,1 --output <file.json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BINS = {
    "time": [("< 1 min", 60.0), ("1 min-1 h", 3600.0), ("1 h-1 day", 86400.0),
             ("1-30 days", 30 * 86400.0), ("> 30 days", float("inf"))],
    "space": [("< 10 m", 10.0), ("10-100 m", 100.0), ("100 m-1 km", 1000.0),
              ("1-10 km", 10000.0), ("> 10 km", float("inf"))],
}


def _seconds(value) -> float:
    from model.src.data.trajectory_distinct import _timestamp_to_float

    return _timestamp_to_float(value)


def _collect(contexts, widths: dict[str, list[float]]) -> None:
    from model.src.predicates.operators import PredicateOp

    for context in contexts:
        query = context.trajectory_query
        if query is not None:
            for temporal in getattr(query, "temporal_predicates", ()):
                widths["segment_interval"].append(_seconds(temporal.upper) - _seconds(temporal.lower))
            for spatial in getattr(query, "spatial_predicates", ()):
                widths["window_x"].append(float(spatial.max_x) - float(spatial.min_x))
                widths["window_y"].append(float(spatial.max_y) - float(spatial.min_y))
        start = context.ordinary_predicates.get("trips:start_time")
        end = context.ordinary_predicates.get("trips:end_time")
        if (start is not None and end is not None and start.op == PredicateOp.LESS_EQUAL
                and end.op == PredicateOp.GREATER_EQUAL):
            widths["trip_interval"].append(_seconds(start.value) - _seconds(end.value))


def _summary(values: list[float], kind: str) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"count": 0}
    result = {"count": int(array.size),
              "quantiles": {f"p{q}": float(np.percentile(array, q)) for q in (5, 25, 50, 75, 95)},
              "inside (U < L)": float(np.mean(array < 0))}
    previous = 0.0
    for label, limit in BINS[kind]:
        result[label] = float(np.mean((array >= previous) & (array < limit)))
        previous = limit
    return result


def main() -> int:
    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.predicates.generation import PredicateTrainingContextGenerator
    from model.src.training.resmade_trainer import _predicate_generation_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--probabilities", default="0,0.5,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    gate = dict(config["predicate_generation"].get("log_width_gate") or {})
    if not gate.get("dimensions"):
        raise SystemExit("the config has no predicate_generation.log_width_gate.dimensions")
    config.setdefault("rare_support", {})["enabled"] = False
    config.setdefault("rare_auxiliary", {})["enabled"] = False
    started = time.perf_counter()
    source = sample_source_from_config(config)
    metadata = source.metadata
    batch_size = int(config["training"]["batch_size"])
    batches = [source.batches(batch_size, seed=args.seed + index).encoded_values for index in range(args.batches)]
    report = {"config": args.config, "batches": args.batches, "batch_size": batch_size,
              "setup_seconds": time.perf_counter() - started, "gate": gate,
              "trajectory_spatial": _predicate_generation_config(config).get("trajectory_spatial"),
              "probabilities": {}}
    for probability in [float(value) for value in args.probabilities.split(",")]:
        predicate_config = _predicate_generation_config(config)
        predicate_config["log_width_gate"] = {**gate, "probability": probability}
        generator = PredicateTrainingContextGenerator(predicate_config)
        warm = time.perf_counter()
        generator.warm_caches(metadata)
        warm_seconds = time.perf_counter() - warm
        widths = {"segment_interval": [], "trip_interval": [], "window_x": [], "window_y": []}
        rejected = generated = 0
        seconds = 0.0
        rng = np.random.default_rng(args.seed)
        for rows in batches:
            begin = time.perf_counter()
            contexts, _, stats = generator.generate_batch(encoded_rows=rows, metadata=metadata, rng=rng)
            seconds += time.perf_counter() - begin
            rejected += stats.rejected_unsatisfied_contexts
            generated += stats.generated_contexts
            _collect(contexts, widths)
        report["probabilities"][str(probability)] = {
            "warm_cache_seconds": warm_seconds,
            "generation_seconds_per_batch": seconds / max(1, len(batches)),
            "generated_contexts": generated,
            "rejected_unsatisfied_contexts": rejected,
            "segment_interval_seconds": _summary(widths["segment_interval"], "time"),
            "trip_interval_seconds": _summary(widths["trip_interval"], "time"),
            "window_x_meters": _summary(widths["window_x"], "space"),
            "window_y_meters": _summary(widths["window_y"], "space"),
        }
        print(json.dumps({"probability": probability,
                          "seconds_per_batch": report["probabilities"][str(probability)]["generation_seconds_per_batch"],
                          "rejected": rejected}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(entry["rejected_unsatisfied_contexts"] == 0 for entry in report["probabilities"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
