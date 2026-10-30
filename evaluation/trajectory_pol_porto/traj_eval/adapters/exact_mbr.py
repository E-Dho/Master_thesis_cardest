"""Exact-MBR: the exact count of trajectories whose bounding box matches.

One box per trajectory, from the snapshot's ``derived/trips``: spatial extent
over all segment endpoints and the time span ``[tmin, tmax]``. For a multi
(distinct-trip) query a trajectory counts when

- its trip and agent rows satisfy their predicates (evaluated exactly, with
  the v3 ``NumpyBackend`` predicate code on per-trip aligned arrays), and
- its box overlaps the query: the spatial window (closed), and the interval
  ``tmin <= upper`` and ``tmax >= lower`` (missing bounds are open). A
  spatio-temporal predicate tests space and time separately, which is
  exactly the MBR representation.

This is the representation error of every one-box-per-trajectory method
(DeepDB-MBR, TraHist/CASE+ at trajectory level) without any model error.
Because a segment that matches lies inside its trajectory's box, the result
is never below the true count for supported queries (tested).

When a query has a segment window or interval, the answer is to the box
question, not the exact one, so it carries the diagnostic
``approx:trajectory_box_overlap`` (status ``ok``): the coverage block counts
it as approximate and ``all:exact_only`` leaves it out, as for the own
model's MBR surrogate. Queries with only trip and agent predicates are
answered exactly and carry no diagnostic.

Unsupported (status ``unsupported``):

- single-table queries (row-count labels; not a trajectory-level question);
- standard predicates on segment rows (e.g. ``segment_idx``): a box cannot
  express which segments satisfy them (decided for all per-trajectory
  methods).

The estimator is deterministic; one seed is enough.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .base import Estimate, InProcessTrajectoryAdapter

SEGMENT_SPACE_TIME = ("spatial", "spatio_temporal", "temporal")
#: diagnostic of an estimate that answers the box question instead of the exact one
APPROX_DIAGNOSTIC = "approx:trajectory_box_overlap"


def _bound_us(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    from query_generation.benchmark_v3 import exact
    from query_generation.query_generator import parse_timestamp

    return exact.to_us(parse_timestamp(value))


def align_trip_tables(settings: Any, data: Mapping[str, Mapping[str, np.ndarray]],
                      trips: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """Every non-segment table's columns aligned with the snapshot's trip order.

    The entity table (trips) must hold exactly the trips of ``derived/trips``
    (the snapshot's consistency checks guarantee it); other tables join it
    through the v3 join keys (agents via ``trips.agent_id``).
    """

    from query_generation.benchmark_v3.numpy_backend import NumpyBackend

    entity = settings.base["entity"]
    entity_table, key = entity["table"], entity["key"]
    trip_ids = np.asarray(trips["trip_id"])
    entity_ids = np.asarray(data[entity_table][key])
    if not np.array_equal(entity_ids, trip_ids):
        raise ValueError("entity table and derived trips differ; re-export the snapshot")
    aligned: Dict[str, Dict[str, np.ndarray]] = {
        entity_table: {name: np.asarray(values) for name, values in data[entity_table].items()}
    }
    join_keys = NumpyBackend(settings, {}).join_keys
    segments = settings.segments.table
    for table_id, columns in data.items():
        if table_id in (entity_table, segments):
            continue
        if (entity_table, table_id) not in join_keys:
            raise ValueError(f"{table_id} does not join {entity_table} directly")
        entity_column, table_column = join_keys[(entity_table, table_id)]
        keys = np.asarray(columns[table_column])
        order = np.argsort(keys, kind="stable")
        wanted = np.asarray(data[entity_table][entity_column])
        position = np.searchsorted(keys, wanted, sorter=order)
        position = np.clip(position, 0, keys.size - 1)
        rows = order[position]
        if not np.array_equal(keys[rows], wanted):
            raise ValueError(f"some {entity_table} rows have no {table_id} row")
        aligned[table_id] = {name: np.asarray(values)[rows] for name, values in columns.items()}
    return aligned


class ExactMbrModel:
    def __init__(self, settings: Any, arrays: Mapping[str, np.ndarray]):
        from query_generation.benchmark_v3.numpy_backend import NumpyBackend

        self.settings = settings
        self.segments = settings.segments
        self.box = {name: arrays[f"box/{name}"] for name in ("xmin", "ymin", "xmax", "ymax", "tmin_us", "tmax_us")}
        tables: Dict[str, Dict[str, np.ndarray]] = {}
        for name, values in arrays.items():
            if name.startswith("table/"):
                _, table_id, column = name.split("/", 2)
                tables.setdefault(table_id, {})[column] = values
        self.backend = NumpyBackend(settings, tables)
        self.trip_count = int(self.box["xmin"].size)

    def estimate(self, record: Any) -> Estimate:
        if record.relation != "multi":
            return "unsupported", None, "single-table query (Exact-MBR answers distinct-trip queries only)"
        mask = np.ones(self.trip_count, dtype=bool)
        surrogate = False
        for predicate in record.predicates:
            if predicate["table"] != self.segments.table:
                mask &= self.backend.predicate_mask(predicate)
                continue
            kind = predicate["kind"]
            if kind not in SEGMENT_SPACE_TIME:
                return "unsupported", None, f"segment-level attribute predicate on {predicate['attribute']}"
            if kind == "temporal" and predicate["attribute"] != self.segments.temporal_attribute:
                return "unsupported", None, f"segment temporal attribute {predicate['attribute']}"
            surrogate = True
            if kind in ("spatial", "spatio_temporal"):
                mask &= self.box["xmin"] <= float(predicate["max_x"])
                mask &= self.box["xmax"] >= float(predicate["min_x"])
                mask &= self.box["ymin"] <= float(predicate["max_y"])
                mask &= self.box["ymax"] >= float(predicate["min_y"])
            if kind in ("temporal", "spatio_temporal"):
                upper, lower = _bound_us(predicate.get("upper")), _bound_us(predicate.get("lower"))
                if upper is not None:
                    mask &= self.box["tmin_us"] <= upper
                if lower is not None:
                    mask &= self.box["tmax_us"] >= lower
        return "ok", float(np.count_nonzero(mask)), APPROX_DIAGNOSTIC if surrogate else ""


class ExactMbrAdapter(InProcessTrajectoryAdapter):
    latency_scope = "predicate_translation_and_trip_box_scan"
    timing_role = "in_process_exact_mbr"

    def _build_model(self, data, trips) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        aligned = align_trip_tables(self.settings, data, trips)
        arrays: Dict[str, np.ndarray] = {
            f"box/{name}": np.asarray(trips[name])
            for name in ("xmin", "ymin", "xmax", "ymax", "tmin_us", "tmax_us")
        }
        for table_id, columns in aligned.items():
            for column, values in columns.items():
                arrays[f"table/{table_id}/{column}"] = np.asarray(values)
        return arrays, {"method": "exact_mbr", "trip_count": int(trips["trip_id"].size)}

    def _load_model(self, arrays, meta) -> ExactMbrModel:
        return ExactMbrModel(self.settings, arrays)

    def _estimate(self, model: ExactMbrModel, record: Any) -> Estimate:
        return model.estimate(record)
