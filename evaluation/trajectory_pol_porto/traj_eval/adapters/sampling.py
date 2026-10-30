"""Trajectory sampling: exact evaluation on a uniform sample of whole trips.

The model is a seeded simple random sample (without replacement) of ``n`` of
the ``N`` trips, stored with all their segments and their agent rows. A query
is evaluated exactly on the sample with the v3 ``NumpyBackend`` (the code that
labelled the benchmark), and the count is scaled by ``N / n``:

- multi (distinct trips): matching sampled trips x N/n;
- single on the trips table: matching sampled trips x N/n;
- single on the segments table: matching sampled segments x N/n (cluster
  sampling with whole trips as clusters; still unbiased);
- single on any other table (agents): unsupported, because a trip sample is
  not a sample of that table.

These are Horvitz-Thompson estimators, unbiased for every query, including
attribute add-ons. A zero count gives the estimate 0; it is kept as is (the
core scores it with the smoothed Q-error; no hidden fallback). With ``n = N``
every estimate equals its label (tested), which is this method's correctness
gate.

Sample size, exactly one of:

``adapter.sample_trips``
    n directly.
``adapter.sample_fraction``
    n = round(fraction x N).
``adapter.budget_bytes``
    n = floor(budget / mean bytes per trip), the mean taken over the whole
    snapshot (trip row + mean segments x segment row + agent share). n is
    fixed before drawing, so the sample stays a simple random sample; the
    actual serialized size is reported and may deviate slightly from the budget.

The random stream depends on the run seed and the method name only, so seeds
0, 1, 2 give three independent samples and the same seed gives the same sample.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .base import Estimate, InProcessTrajectoryAdapter, query_row

STREAM_SALT = zlib.crc32(b"traj_sampling")


def _row_bytes(columns: Mapping[str, np.ndarray]) -> int:
    return int(sum(np.asarray(values).dtype.itemsize for values in columns.values()))


def choose_sample_size(adapter_config: Mapping[str, Any], data, trips) -> Tuple[int, Dict[str, Any]]:
    total = int(trips["trip_id"].size)
    given = [key for key in ("sample_trips", "sample_fraction", "budget_bytes")
             if adapter_config.get(key) not in (None, "")]
    if len(given) != 1:
        raise ValueError("set exactly one of adapter.sample_trips, sample_fraction, budget_bytes")
    key = given[0]
    detail: Dict[str, Any] = {"sample_size_rule": key}
    if key == "sample_trips":
        n = int(adapter_config["sample_trips"])
    elif key == "sample_fraction":
        fraction = float(adapter_config["sample_fraction"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        n = int(round(fraction * total))
    else:
        budget = int(adapter_config["budget_bytes"])
        segment_columns = {k: v for k, v in data["segments"].items() if not k.startswith("segment_time.")}
        segment_count = int(next(iter(segment_columns.values())).size)
        per_trip = _row_bytes(data["trips"]) + (segment_count / total) * _row_bytes(segment_columns)
        others = sum(_row_bytes(c) * int(next(iter(c.values())).size)
                     for t, c in data.items() if t not in ("trips", "segments"))
        per_trip += others / total
        n = int(math.floor(budget / per_trip))
        detail.update({"budget_bytes": budget, "mean_bytes_per_trip": per_trip})
    if not 1 <= n <= total:
        raise ValueError(f"sample size {n} outside [1, {total}]")
    detail.update({"sample_trips": n, "population_trips": total, "sample_fraction": n / total})
    return n, detail


def draw_sample(settings: Any, data, trips, n: int, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Whole trips ``n`` of ``N``, with their segments and agent rows."""

    from query_generation.benchmark_v3.numpy_backend import NumpyBackend

    total = int(trips["trip_id"].size)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_SALT]))
    positions = np.sort(rng.choice(total, size=n, replace=False))
    entity = settings.base["entity"]
    entity_table, key = entity["table"], entity["key"]
    if not np.array_equal(np.asarray(data[entity_table][key]), np.asarray(trips["trip_id"])):
        raise ValueError("entity table and derived trips differ; re-export the snapshot")
    sample: Dict[str, Dict[str, np.ndarray]] = {
        entity_table: {name: np.asarray(values)[positions] for name, values in data[entity_table].items()}
    }
    offsets = np.asarray(trips["offset"])
    starts, stops = offsets[positions], offsets[positions + 1]
    lengths = stops - starts
    segment_rows = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths) + np.arange(int(lengths.sum()))
    segments = settings.segments.table
    sample[segments] = {name: np.asarray(values)[segment_rows] for name, values in data[segments].items()
                        if not name.startswith(f"{settings.segments.temporal_attribute}.")}
    join_keys = NumpyBackend(settings, {}).join_keys
    for table_id, columns in data.items():
        if table_id in (entity_table, segments):
            continue
        entity_column, table_column = join_keys[(entity_table, table_id)]
        wanted = np.unique(sample[entity_table][entity_column])
        keep = np.isin(np.asarray(columns[table_column]), wanted)
        sample[table_id] = {name: np.asarray(values)[keep] for name, values in columns.items()}
    return sample


class SampleModel:
    def __init__(self, settings: Any, sample: Dict[str, Dict[str, np.ndarray]], population: int):
        from query_generation.benchmark_v3.numpy_backend import NumpyBackend

        seg = settings.segments
        sample[seg.table][f"{seg.temporal_attribute}.start"] = sample[seg.table]["ts"]
        sample[seg.table][f"{seg.temporal_attribute}.end"] = sample[seg.table]["te"]
        self.settings = settings
        self.backend = NumpyBackend(settings, sample)
        self.n = int(next(iter(sample[settings.base["entity"]["table"]].values())).size)
        self.scale = population / self.n
        self.sampled_tables = {settings.base["entity"]["table"], seg.table}

    def estimate(self, record: Any) -> Estimate:
        if record.relation == "single" and record.tables[0] not in self.sampled_tables:
            return "unsupported", None, f"{record.tables[0]} rows are not sampled (trip sample)"
        return "ok", float(self.backend.count(query_row(record))) * self.scale, ""


class TrajectorySamplingAdapter(InProcessTrajectoryAdapter):
    latency_scope = "exact_predicate_evaluation_on_trip_sample"
    timing_role = "in_process_trajectory_sampler"

    def _build_model(self, data, trips):
        n, detail = choose_sample_size(self.config.adapter, data, trips)
        sample = draw_sample(self.settings, data, trips, n, self.seed)
        arrays = {f"table/{table}/{column}": values
                  for table, columns in sample.items() for column, values in columns.items()}
        segments = next(iter(sample[self.settings.segments.table].values())).size
        return arrays, {"method": "trajectory_sampling", "seed": self.seed,
                        "sampled_segments": int(segments), **detail}

    def _load_model(self, arrays, meta) -> SampleModel:
        sample: Dict[str, Dict[str, np.ndarray]] = {}
        for name, values in arrays.items():
            _, table_id, column = name.split("/", 2)
            sample.setdefault(table_id, {})[column] = values
        return SampleModel(self.settings, sample, int(meta["population_trips"]))

    def _estimate(self, model: SampleModel, record: Any) -> Estimate:
        return model.estimate(record)
