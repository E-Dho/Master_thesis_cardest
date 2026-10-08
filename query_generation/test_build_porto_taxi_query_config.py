import unittest
from datetime import datetime

from query_generation.build_porto_taxi_query_config import build_config, read_domains
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


MOMENT = datetime(2013, 7, 1, 0, 0, 0)


class PortoDomainTest(unittest.TestCase):
    def test_spatial_extent_covers_both_segment_endpoints(self):
        # Every segment end point is the next segment's start point except the
        # last of each trip, so a start-only extent drops one terminal point
        # per trip.  Here the extremes live only in e_x / e_y.
        class FakeCursor:
            def __init__(self):
                self.statements = []

            def execute(self, sql):
                self.statements.append(" ".join(sql.split()))

            def fetchone(self):
                sql = self.statements[-1]
                if "MIN(t_s)" in sql:
                    return (MOMENT, MOMENT, -5.0, -6.0, 7.0, 8.0)
                return (1, 2)

            def fetchall(self):
                return [("A",), ("B",)]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        cursor = FakeCursor()

        class FakeConnection:
            def cursor(self):
                return cursor

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        import query_generation.build_porto_taxi_query_config as module

        original = module.__dict__.get("psycopg")
        try:
            import sys
            import types

            fake = types.ModuleType("psycopg")
            fake.connect = lambda **kwargs: FakeConnection()
            sys.modules["psycopg"] = fake
            domains = read_domains("h", 1, "db", None)
        finally:
            sys.modules.pop("psycopg", None)
            if original is not None:
                module.__dict__["psycopg"] = original

        extent = next(s for s in cursor.statements if "MIN(t_s)" in s)
        self.assertIn("LEAST(MIN(s_x), MIN(e_x))", extent)
        self.assertIn("GREATEST(MAX(s_y), MAX(e_y))", extent)
        self.assertEqual(
            domains["segment_geom"],
            {"min_x": -5.0, "min_y": -6.0, "max_x": 7.0, "max_y": 8.0},
        )


if __name__ == "__main__":
    unittest.main()
