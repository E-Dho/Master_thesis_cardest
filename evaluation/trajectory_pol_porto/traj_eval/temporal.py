"""Shared temporal extension of the spatial trajectory histograms (CASE+, TraHist).

Spec: ``claude/baseline-research-trajectories.md``, "Temporal extension for
CASE+ and TraHist" (decided 2026-10-26/30). A spatial histogram keeps its 2D
statistics; each statistic additionally gets per-time-bucket counts, and the
number of distinct trajectories of one statistic over a bucket-aligned
interval ``a..b`` is

    D([a, b]) = sum_{k=a..b} F_k  -  sum_{a <= i < j <= b} L_{i,j}

- ``F_k``: distinct trajectories contributing to the statistic in bucket ``k``;
- ``L_{i,j}``: trajectories whose *consecutive occupied* buckets for that
  statistic are ``i`` and ``j`` (empty buckets may lie between them).

A trajectory occupying buckets ``k1 < ... < km`` of a statistic has its
occupied buckets inside ``a..b`` as a contiguous run of that sequence, so all
links between them lie inside the range: it contributes ``r - (r - 1) = 1``
for ``r >= 1`` buckets in range and 0 otherwise. The formula is therefore
exact for every statistic and every bucket-aligned interval. The link term is
a dominance count (pairs with ``i >= a`` and ``j <= b``).

Time buckets are equi-depth: each holds about the same number of segments,
counted by their time midpoint (decided 2026-10-30); the boundaries are
computed on the full data and stored with the histogram. A query interval
selects every bucket it intersects (no fractional weighting).

Everything here is plain numpy; statistics are integer ids ``0..n_stats-1``
chosen by the histogram (cells, buckets, edges, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

KEY_LIMIT = 1 << 62  # combined integer keys must stay below this


class SettingTooLarge(MemoryError, OverflowError):
    """A histogram setting refused by an explicit size guard before anything that large is allocated.

    ``tune.sweep`` skips such a setting and goes on with the next one (a real
    allocation failure or any other error still stops the sweep). Subclasses
    both MemoryError and OverflowError, the types these guards raised before.
    """


# ---------------------------------------------------------------------------
# Time buckets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeBuckets:
    """Half-open buckets ``[boundaries[k], boundaries[k+1])``, k = 0..count-1.

    ``boundaries[0]`` is the earliest and ``boundaries[-1]`` one microsecond
    past the latest time of the data, so every data time falls in a bucket.
    """

    boundaries: np.ndarray  # int64, strictly increasing, length count + 1

    @property
    def count(self) -> int:
        return int(self.boundaries.size - 1)

    @classmethod
    def equi_depth(cls, midpoints_us: np.ndarray, requested: int, t_min_us: int, t_max_us: int) -> "TimeBuckets":
        """``requested`` buckets with about equal numbers of midpoints each.

        Interior boundaries are the ``k / requested`` quantiles of the
        midpoints (lower order statistic, so they are data values). Equal
        quantiles (many midpoints at one time) merge, so fewer buckets can
        result; :attr:`count` is the effective number.
        """

        if requested < 1:
            raise ValueError("at least one time bucket is required")
        midpoints = np.sort(np.asarray(midpoints_us, dtype=np.int64))
        if midpoints.size == 0:
            raise ValueError("no midpoints")
        t_min, t_max = int(t_min_us), int(t_max_us)
        if not (t_min <= midpoints[0] and midpoints[-1] <= t_max):
            raise ValueError("midpoints outside [t_min, t_max]")
        positions = (np.arange(1, requested, dtype=np.int64) * midpoints.size) // requested
        interior = midpoints[positions] if positions.size else np.zeros(0, dtype=np.int64)
        boundaries = np.unique(np.concatenate(([t_min], interior, [t_max + 1])).astype(np.int64))
        boundaries = boundaries[(boundaries >= t_min) & (boundaries <= t_max + 1)]
        if boundaries[0] != t_min:
            boundaries = np.concatenate(([t_min], boundaries))
        return cls(boundaries=boundaries)

    def assign(self, times_us: np.ndarray) -> np.ndarray:
        """Bucket of every time (times outside the data range are clipped)."""
        index = np.searchsorted(self.boundaries, np.asarray(times_us, dtype=np.int64), side="right") - 1
        return np.clip(index, 0, self.count - 1).astype(np.int32)

    def interval(self, lower_us: Optional[int], upper_us: Optional[int]) -> Optional[Tuple[int, int]]:
        """Buckets ``a..b`` intersecting the closed interval; ``None`` if none does."""
        first, last = int(self.boundaries[0]), int(self.boundaries[-1]) - 1
        lower = first if lower_us is None else max(int(lower_us), first)
        upper = last if upper_us is None else min(int(upper_us), last)
        if lower > upper:
            return None
        a = int(np.searchsorted(self.boundaries, lower, side="right") - 1)
        b = int(np.searchsorted(self.boundaries, upper, side="right") - 1)
        return a, b


# ---------------------------------------------------------------------------
# Sparse per-statistic bucket counts and links
# ---------------------------------------------------------------------------


def _check_key_space(*sizes: int) -> None:
    product = 1
    for size in sizes:
        product *= max(int(size), 1)
    if product >= KEY_LIMIT:
        raise SettingTooLarge(f"key space {sizes} exceeds 2^62; use fewer cells or buckets")


def _csr(stat: np.ndarray, n_stats: int) -> np.ndarray:
    """Row offsets for entries already sorted by ``stat``."""
    return np.concatenate(([0], np.cumsum(np.bincount(stat, minlength=n_stats)))).astype(np.int64)


def _gather(offsets: np.ndarray, stats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Entry indices of the rows ``stats`` and, for each entry, its position in ``stats``."""
    stats = np.asarray(stats, dtype=np.int64)
    starts, ends = offsets[stats], offsets[stats + 1]
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(stats.size, dtype=np.int64), lengths)
    first = np.cumsum(lengths) - lengths  # first output position of each row
    index = np.arange(total, dtype=np.int64) - first[owner] + starts[owner]
    return index, owner


@dataclass(frozen=True)
class BucketCounts:
    """A sparse ``n_stats x n_buckets`` count matrix (CSR by statistic)."""

    offsets: np.ndarray  # int64, n_stats + 1
    bucket: np.ndarray  # int32, sorted within each row
    value: np.ndarray  # int64

    @classmethod
    def from_pairs(cls, stat: np.ndarray, bucket: np.ndarray, n_stats: int, n_buckets: int,
                   weight: Optional[np.ndarray] = None) -> "BucketCounts":
        """Sum of ``weight`` (default 1) per (stat, bucket)."""
        stat = np.asarray(stat, dtype=np.int64)
        bucket = np.asarray(bucket, dtype=np.int64)
        _check_key_space(n_stats, n_buckets)
        keys = stat * n_buckets + bucket
        unique, inverse = np.unique(keys, return_inverse=True)
        value = np.bincount(inverse, weights=None if weight is None else np.asarray(weight, dtype=np.float64),
                            minlength=unique.size)
        value = np.rint(value).astype(np.int64)
        rows = unique // n_buckets
        return cls(offsets=_csr(rows, n_stats), bucket=(unique % n_buckets).astype(np.int32), value=value)

    def range_sum(self, stats: np.ndarray, a: int, b: int) -> np.ndarray:
        """``sum_{k=a..b}`` of every row in ``stats`` (int64 array, one per stat)."""
        index, owner = _gather(self.offsets, stats)
        out = np.zeros(len(stats), dtype=np.int64)
        if index.size:
            keep = (self.bucket[index] >= a) & (self.bucket[index] <= b)
            np.add.at(out, owner[keep], self.value[index[keep]])
        return out

    def row_totals(self) -> np.ndarray:
        n_stats = self.offsets.size - 1
        rows = np.repeat(np.arange(n_stats), np.diff(self.offsets))
        return np.bincount(rows, weights=self.value.astype(np.float64), minlength=n_stats).astype(np.int64)

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {f"{prefix}/offsets": self.offsets, f"{prefix}/bucket": self.bucket, f"{prefix}/value": self.value}

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str) -> "BucketCounts":
        return cls(offsets=np.asarray(arrays[f"{prefix}/offsets"], dtype=np.int64),
                   bucket=np.asarray(arrays[f"{prefix}/bucket"], dtype=np.int32),
                   value=np.asarray(arrays[f"{prefix}/value"], dtype=np.int64))


@dataclass(frozen=True)
class Links:
    """Per statistic, the multiset of consecutive occupied bucket pairs ``(i, j)``, ``i < j``."""

    offsets: np.ndarray  # int64, n_stats + 1
    first: np.ndarray  # int32 i, sorted within each row
    second: np.ndarray  # int32 j
    count: np.ndarray  # int64 multiplicity

    def dominated(self, stats: np.ndarray, a: int, b: int) -> np.ndarray:
        """Per stat, the number of links with ``i >= a`` and ``j <= b``."""
        index, owner = _gather(self.offsets, stats)
        out = np.zeros(len(stats), dtype=np.int64)
        if index.size:
            keep = (self.first[index] >= a) & (self.second[index] <= b)
            np.add.at(out, owner[keep], self.count[index[keep]])
        return out

    @property
    def total(self) -> int:
        return int(self.count.sum())

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {f"{prefix}/offsets": self.offsets, f"{prefix}/first": self.first,
                f"{prefix}/second": self.second, f"{prefix}/count": self.count}

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str) -> "Links":
        return cls(offsets=np.asarray(arrays[f"{prefix}/offsets"], dtype=np.int64),
                   first=np.asarray(arrays[f"{prefix}/first"], dtype=np.int32),
                   second=np.asarray(arrays[f"{prefix}/second"], dtype=np.int32),
                   count=np.asarray(arrays[f"{prefix}/count"], dtype=np.int64))


@dataclass(frozen=True)
class TemporalDistinct:
    """``F`` and ``L`` of a set of statistics; :meth:`distinct` is ``D([a, b])``."""

    occupancy: BucketCounts  # F
    links: Links  # L

    @classmethod
    def build(cls, occurrences: "Occurrences", owner_of_unit: Optional[np.ndarray] = None,
              n_stats: Optional[int] = None) -> "TemporalDistinct":
        """From unique (unit, trajectory, bucket) occurrences.

        ``owner_of_unit`` maps each unit (e.g. grid cell) to the statistic it
        is aggregated into (e.g. a TraHist bucket); the statistic's ``F`` and
        ``L`` are then the *sums* of its units' ``F`` and ``L``, so its
        ``D`` is the sum of the units' distinct counts. Without it every unit
        is its own statistic.
        """

        unit, bucket = occurrences.unit, occurrences.bucket
        stat = unit if owner_of_unit is None else np.asarray(owner_of_unit, dtype=np.int64)[unit]
        n_stats = int(occurrences.n_units if n_stats is None else n_stats)
        n_buckets = occurrences.n_buckets
        occupancy = BucketCounts.from_pairs(stat, bucket, n_stats, n_buckets)
        # consecutive occurrences of one (unit, trajectory) are consecutive occupied buckets
        same = (occurrences.unit[1:] == occurrences.unit[:-1]) & (occurrences.trajectory[1:] == occurrences.trajectory[:-1])
        link_stat = stat[1:][same]
        link_i = bucket[:-1][same].astype(np.int64)
        link_j = bucket[1:][same].astype(np.int64)
        _check_key_space(n_stats, n_buckets, n_buckets)
        keys = (link_stat * n_buckets + link_i) * n_buckets + link_j
        unique, counts = np.unique(keys, return_counts=True)
        rows = unique // (n_buckets * n_buckets)
        links = Links(offsets=_csr(rows, n_stats),
                      first=((unique // n_buckets) % n_buckets).astype(np.int32),
                      second=(unique % n_buckets).astype(np.int32),
                      count=counts.astype(np.int64))
        return cls(occupancy=occupancy, links=links)

    def distinct(self, stats: np.ndarray, a: int, b: int) -> np.ndarray:
        return self.occupancy.range_sum(stats, a, b) - self.links.dominated(stats, a, b)

    def arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        return {**self.occupancy.arrays(f"{prefix}/F"), **self.links.arrays(f"{prefix}/L")}

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str) -> "TemporalDistinct":
        return cls(occupancy=BucketCounts.from_arrays(arrays, f"{prefix}/F"),
                   links=Links.from_arrays(arrays, f"{prefix}/L"))


@dataclass(frozen=True)
class Occurrences:
    """Unique (unit, trajectory, bucket) triples, sorted by unit, trajectory, bucket."""

    unit: np.ndarray  # int64
    trajectory: np.ndarray  # int64 (dense index)
    bucket: np.ndarray  # int64
    n_units: int
    n_trajectories: int
    n_buckets: int

    @classmethod
    def from_events(cls, unit: np.ndarray, trajectory: np.ndarray, bucket: np.ndarray,
                    n_units: int, n_trajectories: int, n_buckets: int) -> "Occurrences":
        """Deduplicated and sorted; ``trajectory`` must be a dense index ``0..n-1``."""
        _check_key_space(n_units, n_trajectories, n_buckets)
        keys = (np.asarray(unit, dtype=np.int64) * n_trajectories
                + np.asarray(trajectory, dtype=np.int64)) * n_buckets + np.asarray(bucket, dtype=np.int64)
        keys = np.unique(keys)
        return cls(unit=keys // (n_trajectories * n_buckets),
                   trajectory=(keys // n_buckets) % n_trajectories,
                   bucket=keys % n_buckets,
                   n_units=int(n_units), n_trajectories=int(n_trajectories), n_buckets=int(n_buckets))


def brute_force_distinct(unit: np.ndarray, trajectory: np.ndarray, bucket: np.ndarray, stat_of_unit: np.ndarray,
                         stat: int, a: int, b: int) -> int:
    """Reference for tests: the sum over the stat's units of distinct trajectories in buckets ``a..b``."""
    unit = np.asarray(unit)
    mask = (np.asarray(stat_of_unit)[unit] == stat) & (np.asarray(bucket) >= a) & (np.asarray(bucket) <= b)
    pairs = set(zip(unit[mask].tolist(), np.asarray(trajectory)[mask].tolist()))
    return len(pairs)
