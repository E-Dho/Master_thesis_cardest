import unittest

from query_generation.build_porto_taxi_query_config import build_config
from query_generation.query_generator import all_valid_categories, validate_config


DOMAINS = {
    "taxi_id": {"min": 1, "max": 442},
    "origin_call": {"min": 1, "max": 20},
    "origin_stand": {"min": 1, "max": 30},
    "num_of_segments": {"min": 1, "max": 40},
    "segment_idx": {"min": 0, "max": 39},
    "call_type": ["A", "B", "C"],
    "daytype": ["A", "B", "C"],
    "segment_time": {"min": "2013-07-01 00:00:00", "max": "2014-06-30 00:00:00"},
    "segment_geom": {"min_x": 0.0, "min_y": 0.0, "max_x": 10.0, "max_y": 10.0},
}


class PortoQueryConfigTest(unittest.TestCase):
    def test_config_is_valid_and_has_full_grid(self):
        config = build_config(DOMAINS)
        validate_config(config)
        self.assertEqual(len(all_valid_categories(config)), 16)

    def test_spatial_and_temporal_attributes_are_segments_only(self):
        config = build_config(DOMAINS)
        for table_id, table in config["tables"].items():
            for attribute in table["attributes"]:
                if attribute["dimension"] in {"spatial", "temporal"}:
                    self.assertEqual(table_id, "segments")


if __name__ == "__main__":
    unittest.main()
