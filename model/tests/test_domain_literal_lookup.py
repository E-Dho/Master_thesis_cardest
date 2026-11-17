"""MBR window canonicalization on domains that are not all-numeric (snapshot preparations).

Porto's prepared domains end with ``__OUTER_MISSING__`` (and may hold ``__SQL_NULL__``),
so the sorted-numeric fast path does not apply and the floor/ceil lookup used to scan
and re-sort the whole domain per call: ~22 s per spatial query on ~10^7 coordinates.
The lookup is now an index built once per domain; it must return exactly what the scan did.
"""

from __future__ import annotations

import math
import random
import time
import unittest

import numpy as np

from model.src.data import trajectory_distinct as td


def _reference_literals(domain):
    values, seen = [], set()
    for literal in domain:
        if isinstance(literal, str) and literal.startswith("__"):
            continue
        try:
            numeric = float(literal)
        except (TypeError, ValueError):
            continue
        if numeric in seen:
            continue
        seen.add(numeric)
        values.append((numeric, literal))
    values.sort(key=lambda item: item[0])
    return tuple(values)


def _reference_floor(domain, upper):
    best = None
    for numeric, literal in _reference_literals(domain):
        if numeric <= upper and (best is None or numeric > best[0]):
            best = (numeric, literal)
    return best


def _reference_ceil(domain, lower):
    best = None
    for numeric, literal in _reference_literals(domain):
        if numeric >= lower and (best is None or numeric < best[0]):
            best = (numeric, literal)
    return best


class DomainLiteralLookupTest(unittest.TestCase):
    def test_same_answers_as_the_scan(self) -> None:
        rng = random.Random(7)
        for trial in range(300):
            size = rng.randint(0, 60)
            values = [rng.choice([rng.uniform(-50, 50), float(rng.randint(-20, 20)), rng.randint(-20, 20)])
                      for _ in range(size)]
            if rng.random() < 0.3:
                values.append(float("nan"))
            if rng.random() < 0.5:
                values.append("__SQL_NULL__")
            if rng.random() < 0.3:
                values.append("not-a-number")
            rng.shuffle(values)
            domain = tuple(values) + ("__OUTER_MISSING__",)
            bounds = [rng.uniform(-60, 60) for _ in range(10)] + [math.inf, -math.inf, float("nan")]
            bounds += [float(v) for v in values[:5] if not isinstance(v, str)]
            for bound in bounds:
                self.assertEqual(td._floor_domain_literal_by_scan(domain, bound), _reference_floor(domain, bound),
                                 (trial, bound))
                self.assertEqual(td._ceil_domain_literal_by_scan(domain, bound), _reference_ceil(domain, bound),
                                 (trial, bound))
            reference = _reference_literals(domain)
            cached = td._numeric_domain_literals(domain)
            self.assertEqual(len(cached), len(reference))
            self.assertEqual([(None if math.isnan(n) else n, l) for n, l in cached if not math.isnan(n)],
                             [(n, l) for n, l in reference if not math.isnan(n)])

    def test_literal_type_of_the_first_occurrence_is_kept(self) -> None:
        domain = (3, 3.0, 1.0, 1, "__OUTER_MISSING__")
        self.assertEqual(td._floor_domain_literal_by_scan(domain, 2.5), (1.0, 1.0))
        self.assertIsInstance(td._ceil_domain_literal_by_scan(domain, 2.5)[1], int)  # 3 came before 3.0

    def test_large_snapshot_domain_is_fast_after_the_first_lookup(self) -> None:
        coordinates = np.unique(np.random.default_rng(0).uniform(-50_000, 50_000, 2_000_000))
        domain = tuple(coordinates.tolist()) + ("__OUTER_MISSING__",)
        self.assertFalse(td._is_sorted_numeric_domain(domain))  # the case the fast path misses
        td._floor_domain_literal_by_scan(domain, 0.0)  # builds the index once
        started = time.perf_counter()
        for bound in np.linspace(-49_000, 49_000, 400):
            numeric, literal = td._floor_domain_literal_by_scan(domain, float(bound))
            self.assertLessEqual(numeric, bound)
            td._ceil_domain_literal_by_scan(domain, float(bound))
        self.assertLess(time.perf_counter() - started, 0.5)  # was seconds per call
        index = int(np.searchsorted(coordinates, 1234.5, side="right")) - 1
        self.assertEqual(td._floor_domain_literal_by_scan(domain, 1234.5), (coordinates[index], coordinates[index]))


if __name__ == "__main__":
    unittest.main()
