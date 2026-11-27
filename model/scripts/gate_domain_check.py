"""Log-width gate diagnostic: domain spans and snapping of the gate columns.

Porto's gated windows came out wider than uniform ones (job 23516479); this reports the
span, quantiles and largest gaps of each gate column and simulates the snapped distances.

    python -m model.scripts.gate_domain_check <training config> <output.json>
"""
import json
import math
import sys
import time

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.sample_sources import sample_source_from_config
from model.src.predicates.generation import PredicateTrainingContextGenerator
from model.src.training.resmade_trainer import _predicate_generation_config

config = load_simple_yaml(sys.argv[1])
validate_config(config)
config.setdefault("rare_support", {})["enabled"] = False
config.setdefault("rare_auxiliary", {})["enabled"] = False
started = time.perf_counter()
source = sample_source_from_config(config)
metadata = source.metadata
generator = PredicateTrainingContextGenerator(_predicate_generation_config(config))
compiled = generator._compiled(metadata)
rows = np.concatenate([np.asarray(source.batches(2048, seed=index).encoded_values) for index in range(5)])
rng = np.random.default_rng(0)
report = {"config": sys.argv[1], "setup_seconds": time.perf_counter() - started, "columns": {}}
for column, dimension in sorted(generator.log_width_column_dimension.items()):
    if column not in compiled.column_name_to_index:
        report["columns"][column] = "not in metadata"
        continue
    index = compiled.column_name_to_index[column]
    view = generator._log_width_values(compiled, index)
    if view is None:
        report["columns"][column] = "not numeric"
        continue
    block, first, log_span = view
    gaps = np.diff(block)
    order = np.argsort(gaps)[::-1][:5]
    quantiles = {f"q{q}": float(np.quantile(block, q / 1000)) for q in (0, 1, 10, 50, 500, 950, 990, 999, 1000)}
    # row values (frequency-weighted), from 10,240 sampled rows
    ranks = rows[:, index].astype(np.int64)
    valid = (ranks >= first) & (ranks < first + len(block))
    row_values = block[ranks[valid] - first]
    row_quantiles = {f"q{q}": float(np.quantile(row_values, q / 1000)) for q in (1, 10, 500, 990, 999)} if row_values.size else {}
    # simulate the upper bound at p = 1 from sampled rows
    anchors = ranks[valid][:5000]
    distances, snapped, clamped = [], [], 0
    for anchor in anchors:
        distance = generator._log_distance(dimension, log_span, rng)
        target = float(block[anchor - first]) + distance
        position = int(block.searchsorted(target, side="left"))
        if position >= len(block):
            clamped += 1
            position = len(block) - 1
        distances.append(distance)
        snapped.append(float(block[position]) - float(block[anchor - first]))
    distances, snapped = np.asarray(distances), np.asarray(snapped)
    report["columns"][column] = {
        "dimension": dimension,
        "domain_values": int(len(block)),
        "span": float(math.exp(log_span)) if math.isfinite(log_span) else None,
        "domain_quantiles": quantiles,
        "row_quantiles": row_quantiles,
        "largest_gaps": [{"gap": float(gaps[i]), "from": float(block[i]), "to": float(block[i + 1])} for i in order],
        "drawn_distance_p50": float(np.median(distances)) if distances.size else None,
        "snapped_distance_quantiles": {f"p{q}": float(np.percentile(snapped, q)) for q in (10, 50, 90)} if snapped.size else {},
        "clamped_to_domain_end": clamped / max(1, len(anchors)),
        "snapped_over_2x_drawn": float(np.mean(snapped > 2 * distances)) if snapped.size else None,
    }
    print(column, json.dumps(report["columns"][column]["span"]), report["columns"][column]["snapped_distance_quantiles"], flush=True)
with open(sys.argv[2], "w") as handle:
    json.dump(report, handle, indent=2)
