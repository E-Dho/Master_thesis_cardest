"""The Porto v3 config (decided 2026-11-02): loads, every category has tables."""

import json
import unittest
from pathlib import Path

from query_generation.benchmark_v3.config import eligible_tables, load_settings

CONFIG = Path(__file__).with_name("porto_v3_config.json")
# exact extent of all segment endpoints (Porto probe 22629119), GPS outliers included
EXTENT = {"min_x": -610919.605185877, "min_y": -302980.297887669,
          "max_x": 315277.573883702, "max_y": 1269994.79218329}


class PortoConfigTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(CONFIG)

    def test_schema_follows_the_decisions(self):
        self.assertEqual(sorted(self.settings.base["tables"]), ["segments", "trips"])
        names = {a["name"] for a in self.settings.table("trips")["attributes"]}
        self.assertEqual(names, {"call_type", "origin_stand", "num_of_segments", "trip_time"})
        self.assertEqual(self.settings.srid, 3763)
        self.assertEqual(self.settings.semantics_version,
                         json.loads(Path(__file__).with_name("pol_v3_config.json").read_text())["semantics_version"])

    def test_every_category_is_eligible(self):
        categories = list(self.settings.categories())
        self.assertEqual(len(categories), 16)
        for category in categories:
            self.assertTrue(eligible_tables(self.settings.base, category), category.key)

    def test_one_sided_windows_reach_every_outlier(self):
        domain = self.settings.attr("segments", "segment_geom")["domain"]
        margin = self.settings.unbounded_spatial_margin
        self.assertLessEqual(domain["min_x"] - margin, EXTENT["min_x"])
        self.assertLessEqual(domain["min_y"] - margin, EXTENT["min_y"])
        self.assertGreaterEqual(domain["max_x"] + margin, EXTENT["max_x"])
        self.assertGreaterEqual(domain["max_y"] + margin, EXTENT["max_y"])


if __name__ == "__main__":
    unittest.main()
