"""Which v3 queries the spatial trajectory histograms (TraHist, CASE+) can answer.

Both cover space and time only (decided 2026-10-26: attribute add-ons are
unsupported). A supported multi query has at most one spatial window on the
segments and at most one time interval: on the segment time, a
spatio-temporal predicate, or the trip time (answered with the segment or
point times, marked ``approx:``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .exact_mbr import _bound_us


@dataclass(frozen=True)
class SpaceTimeQuery:
    window: Optional[Tuple[float, float, float, float]]  # min_x, min_y, max_x, max_y
    interval: Optional[Tuple[Optional[int], Optional[int]]]  # lower_us, upper_us (None: open)
    trip_time: bool  # the interval comes from the trip table


def parse_space_time(settings: Any, record: Any, method: str) -> Tuple[Optional[SpaceTimeQuery], str]:
    """``(query, "")`` or ``(None, reason it is unsupported)``."""
    if record.relation != "multi":
        return None, f"single-table query ({method} answers distinct-trajectory queries)"
    segments = settings.segments
    entity = settings.base["entity"]["table"]
    window = interval = None
    trip_time = False
    for predicate in record.predicates:
        table, kind, attribute = predicate["table"], predicate["kind"], predicate["attribute"]
        spatial = table == segments.table and kind in ("spatial", "spatio_temporal")
        segment_time = (table == segments.table and kind in ("temporal", "spatio_temporal")
                        and (kind == "spatio_temporal" or attribute == segments.temporal_attribute))
        on_trip = (kind == "temporal" and table == entity
                   and settings.attr(table, attribute)["type"] == "temporal_interval")
        if not (spatial or segment_time or on_trip):
            return None, f"attribute predicate on {table}.{attribute} ({method} covers space and time only)"
        if spatial:
            if window is not None:
                return None, "more than one spatial window"
            window = tuple(float(predicate[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
        if segment_time or on_trip:
            if interval is not None:
                return None, "more than one time interval"
            interval = (_bound_us(predicate.get("lower")), _bound_us(predicate.get("upper")))
            trip_time = on_trip
    return SpaceTimeQuery(window=window, interval=interval, trip_time=trip_time), ""
