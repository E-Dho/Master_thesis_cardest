"""A real v3 benchmark on synthetic POL data, built with the v3 generator itself."""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import generate_category
from query_generation.benchmark_v3.merge import merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend

REPO = Path(__file__).resolve().parents[3]
V3_CONFIG = REPO / "query_generation" / "benchmark_v3" / "pol_v3_config.json"
SINGLE, MULTI = 6, 8  # per category: 8 x 6 single + 8 x 8 multi


@lru_cache(maxsize=None)
def benchmark_rows() -> tuple:
    settings = load_settings(V3_CONFIG, single_count=SINGLE, multi_count=MULTI)
    data, rows = synthetic.build(settings)
    pools = synthetic.pools_from(settings, rows)
    backend = NumpyBackend(settings, data)
    with tempfile.TemporaryDirectory() as tmp:
        for category in settings.categories():
            generate_category(settings, pools, category, base_seed=9, count_fn=backend.count,
                              output_dir=Path(tmp), progress_interval=0, log=lambda message: None)
        return tuple(merge(settings, Path(tmp))["rows"])


def write_benchmark(path: Path, rows=None) -> Path:
    rows = benchmark_rows() if rows is None else rows
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path
