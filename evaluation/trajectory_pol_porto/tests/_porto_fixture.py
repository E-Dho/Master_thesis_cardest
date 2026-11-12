"""A real v3 Porto benchmark on synthetic Porto-shaped data (the v3 generator's own Porto test data).

The data has what the Porto probes found: call types A/B/C with origin_stand
NULL (NaN in the snapshot) for A, C and some B, stationary segments and GPS
jumps 500 km outside the window domain. Two tables: ``trips`` and
``segments``, no agents.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import generate_category, load_excluded_hashes
from query_generation.benchmark_v3.merge import merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.benchmark_v3.synthetic import pools_from
from query_generation.benchmark_v3.test_porto_end_to_end import porto_data, porto_rows

from ._fixture import REPO, write_synthetic_snapshot

PORTO_CONFIG = REPO / "query_generation" / "benchmark_v3" / "porto_v3_config.json"
SINGLE, MULTI = 6, 8  # per category: 8 x 6 single + 8 x 8 multi
TRIPS = 240


@lru_cache(maxsize=None)
def porto_settings():
    return load_settings(PORTO_CONFIG, single_count=SINGLE, multi_count=MULTI)


def porto_synthetic_data():
    return porto_data(porto_settings(), porto_rows(trips=TRIPS))


@lru_cache(maxsize=None)
def porto_benchmark_rows() -> tuple:
    settings = porto_settings()
    rows = porto_rows(trips=TRIPS)
    pools = pools_from(settings, rows)
    backend = NumpyBackend(settings, porto_data(settings, rows))
    with tempfile.TemporaryDirectory() as tmp:
        for category in settings.categories():
            summary = generate_category(settings, pools, category, base_seed=13, count_fn=backend.count,
                                        output_dir=Path(tmp), progress_interval=0, log=lambda message: None)
            assert summary["finished"], category.key
        return tuple(merge(settings, Path(tmp))["rows"])


def write_porto_benchmark(path: Path, rows=None) -> Path:
    rows = porto_benchmark_rows() if rows is None else rows
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def write_porto_snapshot(root: Path) -> Path:
    return write_synthetic_snapshot(root, porto_synthetic_data(), settings=porto_settings(), config=PORTO_CONFIG)


def write_porto_training(root: Path, benchmark: Path) -> Path:
    """A disjoint Porto workload on the same data (another seed, ``benchmark`` excluded), for MSCN."""
    settings = load_settings(PORTO_CONFIG, single_count=6, multi_count=10)
    rows = porto_rows(trips=TRIPS)
    pools = pools_from(settings, rows)
    backend = NumpyBackend(settings, porto_data(settings, rows))
    hashes, sources = load_excluded_hashes([benchmark])
    for category in settings.categories():
        generate_category(settings, pools, category, base_seed=123, count_fn=backend.count,
                          output_dir=root / "train", progress_interval=0, log=lambda message: None,
                          excluded_hashes=hashes, excluded_sources=sources)
    path = root / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in merge(settings, root / "train", [benchmark])["rows"]),
                    encoding="utf-8")
    return path
