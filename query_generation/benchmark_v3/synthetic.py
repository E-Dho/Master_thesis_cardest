"""Small synthetic POL-shaped database for tests and local dry runs.

The data follows the base config's tables and domains and deliberately
contains the segment shapes the exact predicate treats specially: instants,
stationary segments and axis-parallel segments.
"""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

from query_generation.query_generator import parse_timestamp

from .config import Settings
from .exact import to_us
from .pools import Pools, pool_columns, subsample
from .query_model import timestamp_text


def build(settings: Settings, *, agents: int = 40, trips_per_agent: int = 6, seed: int = 7) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[Dict[str, Any]]]]:
    rng = random.Random(seed)
    seg = settings.segments
    geo = {k: float(v) for k, v in settings.attr(seg.table, seg.spatial_attribute)["domain"].items()}
    t_domain = settings.attr(seg.table, seg.temporal_attribute)["domain"]
    t_min, t_max = parse_timestamp(t_domain["min"]), parse_timestamp(t_domain["max"])
    agent_attrs = {a["name"]: a for a in settings.table("agents")["attributes"]}

    agent_rows, trip_rows, segment_rows = [], [], []
    trip_id = 0
    for agent_id in range(1, agents + 1):
        row = {"agent_id": agent_id}
        for name, attr in agent_attrs.items():
            if attr["type"] == "nominal":
                row[name] = rng.choice(attr["values"])
            elif attr["type"] == "integer":
                row[name] = rng.randint(int(attr["domain"]["min"]), int(attr["domain"]["max"]))
            else:
                row[name] = rng.uniform(float(attr["domain"]["min"]), float(attr["domain"]["max"]))
        agent_rows.append(row)
        for _ in range(trips_per_agent):
            trip_id += 1
            start = t_min + timedelta(seconds=rng.uniform(0, (t_max - t_min).total_seconds() - 86400))
            start = start.replace(microsecond=(start.microsecond // 1000) * 1000)
            x = rng.uniform(geo["min_x"] + 200, geo["max_x"] - 200)
            y = rng.uniform(geo["min_y"] + 200, geo["max_y"] - 200)
            t = start
            n = rng.randint(1, 8)
            for idx in range(n):
                shape = rng.random()
                if shape < 0.1:  # instant
                    nx, ny, dt = x, y, 0.0
                elif shape < 0.2:  # stationary
                    nx, ny, dt = x, y, rng.uniform(10, 600)
                elif shape < 0.3:  # axis-parallel
                    step = rng.uniform(-300, 300)
                    nx, ny = (x + step, y) if rng.random() < 0.5 else (x, y + step)
                    dt = rng.uniform(10, 600)
                else:
                    nx, ny = x + rng.uniform(-300, 300), y + rng.uniform(-300, 300)
                    dt = rng.uniform(10, 600)
                nx = min(max(nx, geo["min_x"]), geo["max_x"])
                ny = min(max(ny, geo["min_y"]), geo["max_y"])
                end = t + timedelta(milliseconds=round(dt * 1000))
                segment_rows.append({
                    "trip_id": trip_id, "segment_idx": idx,
                    "sx": x, "sy": y, "ex": nx, "ey": ny, "ts": t, "te": end,
                })
                x, y, t = nx, ny, end
            trip_rows.append({
                "trip_id": trip_id, "agent_id": agent_id, "num_of_segments": n,
                "start": start, "end": t,
            })

    data = {
        "agents": {key: np.asarray([r[key] for r in agent_rows]) for key in agent_rows[0]},
        "trips": {
            "trip_id": np.asarray([r["trip_id"] for r in trip_rows]),
            "agent_id": np.asarray([r["agent_id"] for r in trip_rows]),
            "num_of_segments": np.asarray([r["num_of_segments"] for r in trip_rows]),
            "trip_time.start": np.asarray([to_us(r["start"]) for r in trip_rows], dtype=np.int64),
            "trip_time.end": np.asarray([to_us(r["end"]) for r in trip_rows], dtype=np.int64),
        },
        "segments": {
            "trip_id": np.asarray([r["trip_id"] for r in segment_rows]),
            "segment_idx": np.asarray([r["segment_idx"] for r in segment_rows]),
            **{key: np.asarray([r[key] for r in segment_rows], dtype=np.float64) for key in ("sx", "sy", "ex", "ey")},
            "ts": np.asarray([to_us(r["ts"]) for r in segment_rows], dtype=np.int64),
            "te": np.asarray([to_us(r["te"]) for r in segment_rows], dtype=np.int64),
        },
    }
    data["segments"][f"{seg.temporal_attribute}.start"] = data["segments"]["ts"]
    data["segments"][f"{seg.temporal_attribute}.end"] = data["segments"]["te"]
    rows = {"agents": agent_rows, "trips": trip_rows, "segments": segment_rows}
    return data, rows


def pools_from(settings: Settings, rows: Dict[str, List[Dict[str, Any]]], size: int = 500, seed: int = 3) -> Pools:
    """Pools with the same columns ``build_pools.py`` stores, taken from synthetic rows."""

    tables = {}
    for table_id, table_rows in rows.items():
        columns = pool_columns(settings, table_id)
        names = [name for name, _ in columns]

        def value(row, name):
            if name.endswith(".start"):
                return timestamp_text(row["start"])
            if name.endswith(".end"):
                return timestamp_text(row["end"])
            if name in ("ts", "te"):
                return timestamp_text(row[name])
            return row[name]

        tables[table_id] = {
            "columns": names,
            "rows": subsample([[value(r, n) for n in names] for r in table_rows], size, seed),
            "sampled_rows": len(table_rows),
            "sql": "synthetic",
        }
    return Pools(tables, {"synthetic": True})
