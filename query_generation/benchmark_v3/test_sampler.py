"""Sampling: determinism, grid, structural rules, add-on and width distributions."""

import math
import unittest
from collections import Counter
from pathlib import Path

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import finish_record
from query_generation.benchmark_v3.merge import check_row
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.benchmark_v3.sampler import Sampler
from query_generation.query_generator import Category, parse_timestamp

CONFIG = Path(__file__).with_name("pol_v3_config.json")


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG)
        cls.data, rows = synthetic.build(cls.settings)
        cls.pools = synthetic.pools_from(cls.settings, rows)
        cls.sampler = Sampler(cls.settings, cls.pools, base_seed=42)
        # A larger grid, for statistics that need more slots than one category has.
        cls.wide = Sampler(load_settings(CONFIG, single_count=2000, multi_count=2000), cls.pools, base_seed=42)
        cls.backend = NumpyBackend(cls.settings, cls.data)

    def test_grid_is_8x250_plus_8x750(self):
        counts = Counter((c.relation, self.settings.count_for(c)) for c in self.settings.categories())
        self.assertEqual(counts, {("single", 250): 8, ("multi", 750): 8})
        self.assertEqual(self.settings.total_queries, 8000)

    def test_candidates_are_deterministic_per_slot_and_attempt(self):
        category = Category.parse("spatio_temporal.range.multi")
        first = self.sampler.candidate(category, 7, 0)
        self.assertEqual(first, self.sampler.candidate(category, 7, 0))
        self.assertNotEqual(first["predicate_hash"], self.sampler.candidate(category, 7, 1)["predicate_hash"])
        other_seed = Sampler(self.settings, self.pools, base_seed=43).candidate(category, 7, 0)
        self.assertNotEqual(first["predicate_hash"], other_seed["predicate_hash"])

    def test_every_candidate_passes_the_structural_gates(self):
        for category in self.settings.categories():
            for index in range(150):
                record = self.sampler.candidate(category, index, 0)
                finish_record(record, 1, 0.0)
                self.assertEqual(check_row(self.settings, record), [], (category.key, index))

    def test_addons_only_in_non_standard_multi_with_exact_split(self):
        for category in self.settings.categories():
            n = self.settings.count_for(category)
            observed = Counter()
            for index in range(n):
                record = self.sampler.candidate(category, index, index % 3)
                addons = [p for p in record["predicates"] if p.get("role") == "addon"]
                self.assertEqual(len(addons), record["addon_target"])
                observed[len(addons)] += 1
                # the target belongs to the slot, not to the attempt
                self.assertEqual(self.sampler.candidate(category, index, 7)["addon_target"], record["addon_target"])
            if category.relation == "single" or category.dimension == "standard":
                self.assertEqual(observed, Counter({0: n}), category.key)
            else:
                self.assertEqual(observed, Counter({0: 375, 1: 188, 2: 187}), category.key)

    def test_quotas_use_largest_remainders(self):
        from query_generation.benchmark_v3.sampler import quotas
        self.assertEqual(quotas(750, [0.5, 0.25, 0.25]), [375, 188, 187])
        self.assertEqual(quotas(10, [0.5, 0.25, 0.25]), [5, 3, 2])
        self.assertEqual(sum(quotas(7, [0.5, 0.25, 0.25])), 7)

    def test_temporal_widths_are_log_uniform(self):
        attr = self.settings.attr("segments", "segment_time")
        span = (parse_timestamp(attr["domain"]["max"]) - parse_timestamp(attr["domain"]["min"])).total_seconds()
        low, high = math.log(60), math.log(span)
        positions = []
        for key in ("temporal.range.single", "spatio_temporal.range.single"):
            category = Category.parse(key)
            for index in range(1500):
                for p in self.wide.candidate(category, index, 0)["predicates"]:
                    width = p.get("width_seconds") or p.get("temporal_width_seconds")
                    if width:
                        self.assertGreaterEqual(width, 60 - 1e-9)
                        self.assertLessEqual(width, span + 1e-6)
                        positions.append((math.log(width) - low) / (high - low))
        positions.sort()
        for q in (0.25, 0.5, 0.75):
            self.assertAlmostEqual(positions[int(q * len(positions))], q, delta=0.04)

    def test_spatial_widths_are_log_uniform(self):
        domain = self.settings.attr("segments", "segment_geom")["domain"]
        extent = {"x": domain["max_x"] - domain["min_x"], "y": domain["max_y"] - domain["min_y"]}
        positions = {"x": [], "y": []}
        for key in ("spatial.range.single", "spatio_temporal.range.single"):
            category = Category.parse(key)
            for index in range(1500):
                predicate = self.wide.candidate(category, index, 0)["predicates"][0]
                for axis in ("x", "y"):
                    width = predicate[f"width_{axis}_meters"]
                    self.assertGreaterEqual(width, 10 - 1e-9)
                    self.assertLessEqual(width, extent[axis] + 1e-6)
                    self.assertLessEqual(predicate[f"max_{axis}"] - predicate[f"min_{axis}"], width + 1e-6)
                    positions[axis].append((math.log(width) - math.log(10)) / (math.log(extent[axis]) - math.log(10)))
        for axis, values in positions.items():
            values.sort()
            for q in (0.25, 0.5, 0.75):
                self.assertAlmostEqual(values[int(q * len(values))], q, delta=0.04, msg=axis)

    def test_joint_centers_make_spatio_temporal_queries_positive(self):
        for key in ("spatio_temporal.range.single", "spatio_temporal.unbounded.single"):
            category = Category.parse(key)
            joint = 0
            for index in range(300):
                record = self.wide.candidate(category, index, 0)
                if record["predicates"][0]["center_source"] != "pool_row_joint":
                    continue
                joint += 1
                self.assertGreater(self.backend.count(record), 0, record["slot_id"])
            self.assertGreater(joint, 200)

    def test_one_window_one_interval_and_no_trip_geometry(self):
        for category in self.settings.categories():
            for index in range(100):
                record = self.sampler.candidate(category, index, 0)
                kinds = Counter(p["kind"] for p in record["predicates"])
                self.assertLessEqual(kinds["spatial"] + kinds["spatio_temporal"], 1)
                self.assertLessEqual(kinds["temporal"] + kinds["spatio_temporal"], 1)
                self.assertNotIn("trip_geom", record["sql"])
                self.assertNotIn("tgeom", record["sql"])


if __name__ == "__main__":
    unittest.main()
