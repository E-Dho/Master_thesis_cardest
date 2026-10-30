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

import numpy as np

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


@lru_cache(maxsize=None)
def synthetic_settings():
    return load_settings(V3_CONFIG, single_count=SINGLE, multi_count=MULTI)


def synthetic_data():
    """The synthetic POL data the fixture benchmark was labelled on."""
    data, _ = synthetic.build(synthetic_settings())
    return data


def write_synthetic_snapshot(root: Path, data=None) -> Path:
    """A P2-format snapshot of the synthetic data (no database needed)."""
    from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap

    data = synthetic_data() if data is None else data
    tables = {}
    for table, arrays in data.items():
        columns = {}
        for name, array in arrays.items():
            if table == "segments" and name.startswith("segment_time."):
                continue
            path = root / "tables" / table / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(array))
            columns[name] = {}
        tables[table] = {"columns": columns}
    snap.derive_trips(synthetic_settings(), root)
    manifest = {"snapshot_version": snap.SNAPSHOT_VERSION, "config_sha256": load_settings(V3_CONFIG).sha256,
                "tables": tables, "aliases": {"segments": {"segment_time.start": "ts", "segment_time.end": "te"}},
                "files": {}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root
