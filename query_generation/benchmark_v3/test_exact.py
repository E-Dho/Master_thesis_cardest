"""The exact predicate: hand-made edge cases, exact rational arithmetic, and
agreement with the existing segment/rectangle test of the model code."""

import random
import unittest
from datetime import datetime
from fractions import Fraction

import numpy as np

from query_generation.benchmark_v3 import exact

S = 1_000_000  # microseconds per second
WINDOW = (0.0, 0.0, 10.0, 10.0)


def st(sx, sy, ex, ey, ts, te, window=WINDOW, lower=None, upper=None):
    return bool(exact.spatio_temporal_mask([sx], [sy], [ex], [ey], [ts], [te], window, lower, upper)[0])


def sp(sx, sy, ex, ey, window=WINDOW):
    return bool(exact.spatial_mask([sx], [sy], [ex], [ey], window)[0])


class EdgeCases(unittest.TestCase):
    # A segment that is inside the window only during [10 s, 20 s].
    seg = (-10.0, 5.0, 10.0, 5.0, 0, 20 * S)

    def test_crossing_outside_interval_is_not_a_match(self):
        self.assertTrue(sp(*self.seg[:4]))
        self.assertFalse(st(*self.seg, lower=0, upper=5 * S))
        decoupled = exact.decoupled_mask(*([v] for v in self.seg), WINDOW, 0, 5 * S)[0]
        self.assertTrue(decoupled, "v2 semantics counted this segment")

    def test_interval_touching_entry_time_matches(self):
        self.assertTrue(st(*self.seg, lower=9 * S, upper=10 * S))
        self.assertFalse(st(*self.seg, lower=9 * S, upper=10 * S - 1))

    def test_inside_during_interval(self):
        self.assertTrue(st(*self.seg, lower=12 * S, upper=13 * S))

    def test_unbounded_sides(self):
        self.assertFalse(st(*self.seg, upper=5 * S))
        self.assertTrue(st(*self.seg, upper=10 * S))
        self.assertTrue(st(*self.seg, lower=15 * S))
        self.assertFalse(st(*self.seg, lower=20 * S + 1))

    def test_instant(self):
        self.assertTrue(st(5, 5, 5, 5, 100 * S, 100 * S, lower=100 * S, upper=100 * S))
        self.assertFalse(st(5, 5, 5, 5, 100 * S, 100 * S, lower=101 * S, upper=200 * S))
        self.assertTrue(st(10, 5, 10, 5, 100 * S, 100 * S, lower=0, upper=200 * S), "boundary point")
        self.assertFalse(st(11, 5, 11, 5, 100 * S, 100 * S, lower=0, upper=200 * S))

    def test_stationary(self):
        self.assertTrue(st(5, 5, 5, 5, 0, 100 * S, lower=50 * S, upper=60 * S))
        self.assertFalse(st(15, 5, 15, 5, 0, 100 * S, lower=50 * S, upper=60 * S))

    def test_axis_parallel_and_boundaries(self):
        self.assertFalse(sp(11, -5, 11, 15))
        self.assertTrue(sp(10, -5, 10, 15), "on the right edge")
        self.assertTrue(sp(-5, 5, 5, 15), "touches the corner (0, 10)")
        self.assertFalse(sp(-5, 6, 4, 15), "passes just outside the corner")

    def test_empty_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            st(*self.seg, lower=28 * S, upper=26 * S)

    def test_time_bounds_are_closed(self):
        mask = exact.temporal_mask([10], [20], 20, 30)
        self.assertTrue(mask[0])
        self.assertFalse(exact.temporal_mask([10], [20], 21, 30)[0])


def rational_st(sx, sy, ex, ey, ts, te, window, lower, upper):
    """Exact ground truth with fractions (no rounding anywhere)."""

    x0, y0, x1, y1 = (Fraction(v) for v in window)
    lo, hi = Fraction(0), Fraction(1)
    for s, e, a0, a1 in ((sx, ex, x0, x1), (sy, ey, y0, y1)):
        s, e = Fraction(s), Fraction(e)
        if e == s:
            if not (a0 <= s <= a1):
                return False
            continue
        p, q = (a0 - s) / (e - s), (a1 - s) / (e - s)
        lo, hi = max(lo, min(p, q)), min(hi, max(p, q))
    if lo > hi:
        return False
    if lower is None and upper is None:
        return True
    if te == ts:
        return (lower is None or lower <= ts) and (upper is None or ts <= upper)
    if lower is not None:
        lo = max(lo, Fraction(lower - ts, te - ts))
    if upper is not None:
        hi = min(hi, Fraction(upper - ts, te - ts))
    return lo <= hi


class RationalGroundTruth(unittest.TestCase):
    """Integer coordinates and times create many exact touches and degenerate cases."""

    def test_spatio_temporal_matches_rational_arithmetic(self):
        rng = random.Random(11)
        n = 20000
        rows = []
        for _ in range(n):
            sx, sy, ex, ey = (rng.randint(-4, 14) for _ in range(4))
            ts = rng.randint(0, 20) * S
            te = ts + rng.choice([0, 0, 1, 2, 5, 10]) * S
            x0, x1 = sorted(rng.randint(-2, 12) for _ in range(2))
            y0, y1 = sorted(rng.randint(-2, 12) for _ in range(2))
            lower = rng.choice([None, rng.randint(0, 30) * S])
            upper = rng.choice([None, rng.randint(0, 30) * S])
            if lower is not None and upper is not None and lower > upper:
                lower, upper = upper, lower
            rows.append((sx, sy, ex, ey, ts, te, (x0, y0, x1, y1), lower, upper))
        mismatches = 0
        for sx, sy, ex, ey, ts, te, window, lower, upper in rows:
            expected = rational_st(sx, sy, ex, ey, ts, te, window, lower, upper)
            if lower is None and upper is None:
                got = sp(float(sx), float(sy), float(ex), float(ey), tuple(map(float, window)))
            else:
                got = st(float(sx), float(sy), float(ex), float(ey), ts, te, tuple(map(float, window)), lower, upper)
            mismatches += expected != got
        self.assertEqual(mismatches, 0)


class AgreesWithModelImplementation(unittest.TestCase):
    def test_spatial_mask_equals_segment_rectangle_intersects_mask(self):
        from model.src.data.trajectory_distinct import segment_rectangle_intersects_mask

        rng = np.random.default_rng(5)
        n = 50000
        sx, sy, ex, ey = (rng.uniform(-50, 150, n) for _ in range(4))
        # a share of axis-parallel and degenerate segments
        ex[: n // 10] = sx[: n // 10]
        ey[n // 10 : n // 5] = sy[n // 10 : n // 5]
        ex[n // 5 : n // 4], ey[n // 5 : n // 4] = sx[n // 5 : n // 4], sy[n // 5 : n // 4]
        for window in ((0.0, 0.0, 100.0, 100.0), (37.5, -10.25, 41.0, 120.0), (20.0, 20.0, 20.5, 80.0)):
            ours = exact.spatial_mask(sx, sy, ex, ey, window)
            theirs = segment_rectangle_intersects_mask(sx, sy, ex, ey, *window)
            self.assertEqual(int(np.sum(ours != theirs)), 0, window)


class Timestamps(unittest.TestCase):
    def test_microsecond_roundtrip(self):
        value = datetime(2019, 7, 1, 6, 25, 0, 123456)
        self.assertEqual(exact.from_us(exact.to_us(value)), value)


if __name__ == "__main__":
    unittest.main()
