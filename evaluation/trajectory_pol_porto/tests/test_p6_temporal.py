"""P6 shared temporal extension: D([a, b]) is exact per statistic (gate 7 of the spec)."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.trajectory_pol_porto.traj_eval.temporal import (
    BucketCounts,
    Occurrences,
    TemporalDistinct,
    TimeBuckets,
    brute_force_distinct,
)


def _random_events(rng, n_units=40, n_traj=60, n_buckets=12, n_events=3000):
    unit = rng.integers(0, n_units, n_events)
    trajectory = rng.integers(0, n_traj, n_events)
    # trajectories occupy a few buckets with gaps, re-enter units, etc.
    bucket = rng.integers(0, n_buckets, n_events)
    return unit, trajectory, bucket


class TemporalDistinctTest(unittest.TestCase):
    def test_distinct_is_exact_for_every_interval(self) -> None:
        rng = np.random.default_rng(7)
        n_units, n_traj, n_buckets = 40, 60, 12
        unit, trajectory, bucket = _random_events(rng, n_units, n_traj, n_buckets)
        occurrences = Occurrences.from_events(unit, trajectory, bucket, n_units, n_traj, n_buckets)
        identity = np.arange(n_units)
        per_unit = TemporalDistinct.build(occurrences)
        # aggregated statistics: units grouped into 7 statistics (TraHist buckets)
        owner = rng.integers(0, 7, n_units)
        grouped = TemporalDistinct.build(occurrences, owner_of_unit=owner, n_stats=7)
        stats_units = np.arange(n_units)
        for a in range(n_buckets):
            for b in range(a, n_buckets):
                got = per_unit.distinct(stats_units, a, b)
                want = [brute_force_distinct(unit, trajectory, bucket, identity, s, a, b) for s in stats_units]
                np.testing.assert_array_equal(got, want, err_msg=f"units {a}..{b}")
                got = grouped.distinct(np.arange(7), a, b)
                want = [brute_force_distinct(unit, trajectory, bucket, owner, s, a, b) for s in range(7)]
                np.testing.assert_array_equal(got, want, err_msg=f"groups {a}..{b}")

    def test_full_range_is_the_plain_distinct_count(self) -> None:
        rng = np.random.default_rng(3)
        unit, trajectory, bucket = _random_events(rng)
        occurrences = Occurrences.from_events(unit, trajectory, bucket, 40, 60, 12)
        model = TemporalDistinct.build(occurrences)
        plain = [len(set(trajectory[unit == u].tolist())) for u in range(40)]
        np.testing.assert_array_equal(model.distinct(np.arange(40), 0, 11), plain)

    def test_no_false_occupancy_between_occupied_buckets(self) -> None:
        # one trajectory in buckets 1 and 5 of one unit: absent from 2..4
        occurrences = Occurrences.from_events(np.array([0, 0]), np.array([0, 0]), np.array([1, 5]), 1, 1, 8)
        model = TemporalDistinct.build(occurrences)
        self.assertEqual(model.distinct(np.array([0]), 2, 4).tolist(), [0])
        self.assertEqual(model.distinct(np.array([0]), 1, 5).tolist(), [1])
        self.assertEqual(model.distinct(np.array([0]), 0, 7).tolist(), [1])
        self.assertEqual(model.links.total, 1)

    def test_arrays_round_trip_and_repeated_stats(self) -> None:
        rng = np.random.default_rng(11)
        unit, trajectory, bucket = _random_events(rng)
        model = TemporalDistinct.build(Occurrences.from_events(unit, trajectory, bucket, 40, 60, 12))
        again = TemporalDistinct.from_arrays(model.arrays("t"), "t")
        stats = np.array([3, 3, 0, 39])
        np.testing.assert_array_equal(again.distinct(stats, 2, 9), model.distinct(stats, 2, 9))

    def test_bucket_counts_with_weights(self) -> None:
        counts = BucketCounts.from_pairs(np.array([0, 0, 2, 2, 2]), np.array([1, 1, 0, 3, 3]), 3, 4,
                                         weight=np.array([2, 3, 1, 1, 1]))
        self.assertEqual(counts.range_sum(np.array([0, 1, 2]), 0, 3).tolist(), [5, 0, 3])
        self.assertEqual(counts.range_sum(np.array([2]), 1, 3).tolist(), [2])
        self.assertEqual(counts.row_totals().tolist(), [5, 0, 3])


class TimeBucketsTest(unittest.TestCase):
    def test_equi_depth_and_interval(self) -> None:
        rng = np.random.default_rng(5)
        midpoints = np.sort(rng.integers(1_000, 1_000_000, 10_000))
        buckets = TimeBuckets.equi_depth(midpoints, 10, 0, 1_000_000)
        self.assertEqual(buckets.count, 10)
        sizes = np.bincount(buckets.assign(midpoints), minlength=10)
        self.assertLessEqual(sizes.max() - sizes.min(), 2)  # equal depth up to ties
        self.assertEqual(buckets.boundaries[0], 0)
        self.assertEqual(buckets.boundaries[-1], 1_000_001)
        self.assertEqual(buckets.interval(None, None), (0, 9))
        inner = int(buckets.boundaries[4])
        self.assertEqual(buckets.interval(inner, inner), (4, 4))  # a boundary belongs to the bucket it starts
        self.assertEqual(buckets.interval(inner - 1, inner), (3, 4))
        self.assertIsNone(buckets.interval(2_000_000, None))
        self.assertIsNone(buckets.interval(None, -5))
        self.assertEqual(buckets.interval(-5, 0), (0, 0))

    def test_ties_merge_buckets(self) -> None:
        midpoints = np.array([5] * 90 + list(range(10, 20)))
        buckets = TimeBuckets.equi_depth(midpoints, 10, 0, 100)
        self.assertLess(buckets.count, 10)
        self.assertTrue(np.all(np.diff(buckets.boundaries) > 0))


if __name__ == "__main__":
    unittest.main()
