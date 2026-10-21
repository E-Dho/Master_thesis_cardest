"""Numpy reference implementation of the v3 predicates (SEMANTICS.md, section 2).

Every function mirrors the SQL rendered by ``query_model.py`` operation by
operation, so both evaluate the same IEEE-754 doubles and must agree exactly.
Timestamps are int64 microseconds since 1970-01-01 of the stored
``timestamp without time zone`` value (no time-zone conversion).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np

EPOCH = datetime(1970, 1, 1)
MICROSECOND = timedelta(microseconds=1)


def to_us(value: datetime) -> int:
    """Microseconds since the epoch for a naive timestamp."""

    if value.tzinfo is not None:
        raise ValueError("v3 timestamps are naive; got a time-zone aware value")
    return (value - EPOCH) // MICROSECOND


def from_us(value: int) -> datetime:
    return EPOCH + timedelta(microseconds=int(value))


def _axis(s: np.ndarray, e: np.ndarray, a0: float, a1: float) -> Tuple[np.ndarray, np.ndarray]:
    """Parameter range of one axis: CASE WHEN e = s THEN ... ELSE LEAST/GREATEST(...) END."""

    d = e - s
    flat = d == 0.0
    inside = (s >= a0) & (s <= a1)
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (a0 - s) / d
        second = (a1 - s) / d
    lo = np.where(flat, np.where(inside, 0.0, 2.0), np.minimum(first, second))
    hi = np.where(flat, np.where(inside, 1.0, -1.0), np.maximum(first, second))
    return lo, hi


def clip_parameters(
    sx: np.ndarray,
    sy: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    window: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Liang-Barsky clip: ``u0 = GREATEST(0, lo_x, lo_y)``, ``u1 = LEAST(1, hi_x, hi_y)``."""

    x0, y0, x1, y1 = (float(value) for value in window)
    if x0 > x1 or y0 > y1:
        raise ValueError(f"window has min > max: {window}")
    sx, sy, ex, ey = (np.asarray(value, dtype=np.float64) for value in (sx, sy, ex, ey))
    lo_x, hi_x = _axis(sx, ex, x0, x1)
    lo_y, hi_y = _axis(sy, ey, y0, y1)
    u0 = np.maximum(np.maximum(0.0, lo_x), lo_y)
    u1 = np.minimum(np.minimum(1.0, hi_x), hi_y)
    return u0, u1


def spatial_mask(sx, sy, ex, ey, window) -> np.ndarray:
    """Segment line intersects the closed window."""

    u0, u1 = clip_parameters(sx, sy, ex, ey, window)
    return u0 <= u1


def temporal_mask(
    start_us: np.ndarray,
    end_us: np.ndarray,
    lower_us: Optional[int],
    upper_us: Optional[int],
) -> np.ndarray:
    """Closed interval overlap: ``start <= U AND end >= L`` (missing bounds are ignored)."""

    check_interval(lower_us, upper_us)
    start_us = np.asarray(start_us, dtype=np.int64)
    end_us = np.asarray(end_us, dtype=np.int64)
    mask = np.ones(start_us.shape, dtype=bool)
    if upper_us is not None:
        mask &= start_us <= np.int64(upper_us)
    if lower_us is not None:
        mask &= end_us >= np.int64(lower_us)
    return mask


def check_interval(lower_us: Optional[int], upper_us: Optional[int]) -> None:
    """The clipping formulas assume a non-empty query interval (L <= U)."""

    if lower_us is not None and upper_us is not None and lower_us > upper_us:
        raise ValueError(f"empty query interval: lower {lower_us} > upper {upper_us}")


def _time_parameter(bound_us: int, ts_us: np.ndarray, duration_s: np.ndarray) -> np.ndarray:
    """``EXTRACT(EPOCH FROM (bound - t_s))::float8 / EXTRACT(EPOCH FROM (t_e - t_s))::float8``."""

    with np.errstate(divide="ignore", invalid="ignore"):
        return ((np.int64(bound_us) - ts_us) / 1e6) / duration_s


def spatio_temporal_mask(
    sx,
    sy,
    ex,
    ey,
    ts_us,
    te_us,
    window: Tuple[float, float, float, float],
    lower_us: Optional[int],
    upper_us: Optional[int],
) -> np.ndarray:
    """There is a u in [0, 1] with t(u) in [L, U] and p(u) in the window."""

    ts_us = np.asarray(ts_us, dtype=np.int64)
    te_us = np.asarray(te_us, dtype=np.int64)
    u0, u1 = clip_parameters(sx, sy, ex, ey, window)
    mask = temporal_mask(ts_us, te_us, lower_us, upper_us)  # index prefilter, implied
    mask &= u0 <= u1
    instant = te_us == ts_us
    duration_s = (te_us - ts_us) / 1e6
    timed = np.ones(ts_us.shape, dtype=bool)
    if upper_us is not None:
        timed &= u0 <= _time_parameter(upper_us, ts_us, duration_s)
    if lower_us is not None:
        timed &= u1 >= _time_parameter(lower_us, ts_us, duration_s)
    return mask & (instant | timed)


def decoupled_mask(sx, sy, ex, ey, ts_us, te_us, window, lower_us, upper_us) -> np.ndarray:
    """The v2 semantics, for diagnostics only.

    v2 tested the segment line against the window and the segment's time span
    against the interval separately (``t_s < U AND t_e >= L``), so a segment
    could match although it was inside the window only outside the interval.
    """

    ts_us = np.asarray(ts_us, dtype=np.int64)
    te_us = np.asarray(te_us, dtype=np.int64)
    mask = spatial_mask(sx, sy, ex, ey, window)
    if upper_us is not None:
        mask &= ts_us < np.int64(upper_us)
    if lower_us is not None:
        mask &= te_us >= np.int64(lower_us)
    return mask
