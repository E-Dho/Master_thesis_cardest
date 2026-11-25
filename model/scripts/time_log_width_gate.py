"""Startup and per-batch cost of predicate_generation.log_width_gate on real data.

Prints, flushed as it goes: sample-source setup time; the generator's compile time
(warm_caches at p = 0); per gated column the domain length, value type and the
seconds to build its numeric view (_numeric_rank_values); then the context
generation time of a few batches at p = 0, 0.5 and 1.

    python -m model.scripts.time_log_width_gate --config <training config> --batches 3
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    import numpy as np

    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.predicates.generation import PredicateTrainingContextGenerator, _numeric_rank_values

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    def say(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    config = load_simple_yaml(args.config)
    validate_config(config)
    config.setdefault("rare_support", {})["enabled"] = False
    config.setdefault("rare_auxiliary", {})["enabled"] = False
    started = time.perf_counter()
    source = sample_source_from_config(config)
    metadata = source.metadata
    say(f"source setup {time.perf_counter() - started:.1f} s")
    batch_size = int(config["training"]["batch_size"])
    batches = [source.batches(batch_size, seed=index).encoded_values for index in range(args.batches)]
    say(f"{len(batches)} batches drawn")

    base = dict(config["predicate_generation"])
    gate = dict(base.get("log_width_gate") or {})
    uniform = PredicateTrainingContextGenerator({**base, "log_width_gate": {**gate, "probability": 0.0}})
    started = time.perf_counter()
    compiled = uniform._compiled(metadata)
    say(f"compile (p = 0 warm_caches) {time.perf_counter() - started:.1f} s")

    gated = PredicateTrainingContextGenerator({**base, "log_width_gate": {**gate, "probability": 0.5}})
    gated._compiled_by_metadata_id = uniform._compiled_by_metadata_id
    gated._cache_by_metadata_id = uniform._cache_by_metadata_id
    for name in sorted(gated.log_width_column_dimension):
        index = compiled.column_name_to_index.get(name)
        if index is None:
            say(f"{name}: not in metadata")
            continue
        cache = compiled.column_caches[index]
        values = cache.comparable_values if cache is not None else ()
        sample = values[len(values) // 2] if values else None
        started = time.perf_counter()
        numeric = _numeric_rank_values(values) if cache is not None else None
        seconds = time.perf_counter() - started
        block = "none" if numeric is None else f"ranks {numeric[1]}..{numeric[2]}"
        say(f"{name}: {len(values)} values, type {type(sample).__name__} ({sample!r}), numeric view {seconds:.1f} s, {block}")

    for probability, generator in ((0.0, uniform), (0.5, gated),
                                   (1.0, PredicateTrainingContextGenerator({**base, "log_width_gate": {**gate, "probability": 1.0}}))):
        if probability == 1.0:
            generator._compiled_by_metadata_id = uniform._compiled_by_metadata_id
            generator._cache_by_metadata_id = uniform._cache_by_metadata_id
            generator._log_width_numeric = gated._log_width_numeric
        elif probability == 0.5:
            started = time.perf_counter()
            generator.warm_caches(metadata)
            say(f"p = 0.5 warm_caches (numeric views) {time.perf_counter() - started:.1f} s")
        rng = np.random.default_rng(0)
        started = time.perf_counter()
        for rows in batches:
            generator.generate_batch(encoded_rows=rows, metadata=metadata, rng=rng)
        say(f"p = {probability}: {(time.perf_counter() - started) / len(batches):.2f} s per batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
