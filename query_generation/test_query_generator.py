import copy
import json
import tempfile
import unittest
from pathlib import Path

from query_generation.query_generator import (
    Category,
    ConfigError,
    drop_conflicting_attributes,
    max_conflict_free_count,
    LiveCenterCache,
    QueryGenerator,
    all_valid_categories,
    load_config,
    validate_config,
    write_jsonl,
)


CONFIG_PATH = Path(__file__).with_name("pol_query_config.json")


class FakeExecutor:
    def __init__(self):
        self.row_queries = []

    def rows(self, sql):
        self.row_queries.append(sql)
        return [(25.0,), (30.0,), (35.0,)]

    def scalar(self, sql):
        raise AssertionError(f"Unexpected scalar query: {sql}")


class QueryGeneratorTest(unittest.TestCase):
    def load_pol_config(self):
        return load_config(CONFIG_PATH)

    def test_invalid_config_rejects_missing_domain(self):
        config = self.load_pol_config()
        bad = copy.deepcopy(config)
        del bad["tables"]["agents"]["attributes"][0]["domain"]
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_invalid_config_rejects_disconnected_joins(self):
        config = self.load_pol_config()
        bad = copy.deepcopy(config)
        bad["joins"] = [bad["joins"][0]]
        with self.assertRaises(ConfigError):
            validate_config(bad)

    def test_deterministic_generation_for_fixed_seed(self):
        config = self.load_pol_config()
        category = Category.parse("standard.range.single")
        first = QueryGenerator(config, seed=42).generate([category], 5)
        second = QueryGenerator(config, seed=42).generate([category], 5)
        self.assertEqual([record["sql"] for record in first], [record["sql"] for record in second])
        self.assertEqual([record["predicates"] for record in first], [record["predicates"] for record in second])

    def test_valid_categories_include_expected_single_and_multi_shapes(self):
        config = self.load_pol_config()
        keys = {category.key for category in all_valid_categories(config)}
        self.assertIn("standard.range.single", keys)
        self.assertIn("temporal.range.single", keys)
        self.assertIn("spatial.range.single", keys)
        self.assertIn("spatio_temporal.range.multi", keys)

    def test_spatial_sql_uses_st_intersects(self):
        config = self.load_pol_config()
        category = Category.parse("spatial.range.single")
        records = QueryGenerator(config, seed=7).generate([category], 8)
        spatial_records = [
            record
            for record in records
            if any(predicate["dimension"] == "spatial" for predicate in record["predicates"])
        ]
        self.assertTrue(spatial_records)
        for record in spatial_records:
            self.assertIn("ST_Intersects(", record["sql"])
            self.assertIn("ST_MakeEnvelope(", record["sql"])

    def test_spatio_temporal_queries_include_both_dimensions(self):
        config = self.load_pol_config()
        category = Category.parse("spatio_temporal.range.single")
        records = QueryGenerator(config, seed=9).generate([category], 10)
        for record in records:
            dims = {predicate["dimension"] for predicate in record["predicates"]}
            self.assertIn("spatial", dims)
            self.assertIn("temporal", dims)

    def test_unbounded_queries_include_ordered_unbounded_predicate(self):
        config = self.load_pol_config()
        category = Category.parse("standard.unbounded.single")
        records = QueryGenerator(config, seed=11).generate([category], 25)
        for record in records:
            modes = {predicate["mode"] for predicate in record["predicates"]}
            self.assertTrue({"unbounded", "temporal_unbounded", "spatial_unbounded"} & modes)

    def test_jsonl_output_is_parseable(self):
        config = self.load_pol_config()
        category = Category.parse("standard.range.single")
        records = QueryGenerator(config, seed=3).generate([category], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queries.jsonl"
            write_jsonl(path, records)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn("sql", rows[0])

    def test_live_centers_are_cached_per_attribute(self):
        config = self.load_pol_config()
        executor = FakeExecutor()
        generator = QueryGenerator(config, seed=1, executor=executor, sample_cache_size=3)
        attr = config["tables"]["agents"]["attributes"][0]
        for _ in range(5):
            value, source = generator.scalar_center("agents", attr)
            self.assertIn(value, {25.0, 30.0, 35.0})
            self.assertEqual(source, "live_row")
        self.assertEqual(len(executor.row_queries), 1)




class GeometryCenterSamplingTest(unittest.TestCase):
    """A segment geometry is a two-point line; both of its ends must be drawn."""

    def sql(self, geometry="s.segment_geom"):
        return LiveCenterCache.geometry_sample_sql(
            geometry, "FROM porto.segments s ORDER BY random() LIMIT 8"
        )

    def test_two_point_lines_draw_either_endpoint(self):
        sql = self.sql()
        self.assertIn("ST_NPoints(s.segment_geom) = 2", sql)
        self.assertIn("random() < 0.5", sql)
        self.assertIn("ST_StartPoint(s.segment_geom)", sql)
        self.assertIn("ST_EndPoint(s.segment_geom)", sql)

    def test_longer_lines_keep_the_interior_point(self):
        self.assertIn("ST_PointOnSurface(t.trip_geom)", self.sql("t.trip_geom"))

    def test_the_draw_is_made_once_per_row(self):
        # ST_X and ST_Y must read one materialized point.  Evaluating the CASE
        # per output column would pair one endpoint's X with the other's Y.
        sql = self.sql()
        self.assertEqual(sql.count("random() < 0.5"), 1)
        self.assertIn("MATERIALIZED", sql)
        self.assertIn("SELECT ST_X(sampled_point), ST_Y(sampled_point) FROM sampled", sql)


EXCLUSIVE_CONFIG = {
    "srid": 1,
    "entity": {"table": "trips", "key": "id", "expression": "t.id"},
    "tables": {
        "trips": {
            "name": "trips",
            "alias": "t",
            "primary_key": "id",
            "flags": ["standard"],
            "attributes": [
                {"name": "call_type", "type": "nominal", "dimension": "standard",
                 "expression": "t.call_type", "values": ["A", "B", "C"]},
                {"name": "origin_call", "type": "integer", "dimension": "standard",
                 "expression": "t.origin_call", "domain": {"min": 1, "max": 20}},
                {"name": "origin_stand", "type": "integer", "dimension": "standard",
                 "expression": "t.origin_stand", "domain": {"min": 1, "max": 60}},
                {"name": "num_of_segments", "type": "integer", "dimension": "standard",
                 "expression": "t.num_of_segments", "domain": {"min": 1, "max": 400}},
            ],
        }
    },
    "joins": [],
    "exclusive_attribute_groups": [
        ["trips.call_type", "trips.origin_call", "trips.origin_stand"]
    ],
}

GROUP = {"trips.call_type", "trips.origin_call", "trips.origin_stand"}


def exclusive_config(**overrides):
    config = copy.deepcopy(EXCLUSIVE_CONFIG)
    config.update(overrides)
    return config


class ExclusiveAttributeGroupTest(unittest.TestCase):
    """Columns the schema makes mutually exclusive must never co-occur."""

    def generated_predicate_keys(self, config, count=400):
        rows = QueryGenerator(config, seed=11).generate(
            [Category.parse("standard.range.single"),
             Category.parse("standard.unbounded.single")],
            count,
        )
        return [
            {f"{p['table']}.{p['attribute']}" for p in row["predicates"]} for row in rows
        ]

    def test_without_the_group_the_impossible_pair_is_common(self):
        config = exclusive_config()
        config.pop("exclusive_attribute_groups")
        conflicts = sum(
            1 for keys in self.generated_predicate_keys(config) if len(keys & GROUP) >= 2
        )
        # Roughly (n+1)/(3n) of draws; the point is that it is far from zero.
        self.assertGreater(conflicts, 100)

    def test_declaring_the_group_removes_every_conflicting_query(self):
        for keys in self.generated_predicate_keys(exclusive_config()):
            self.assertLessEqual(len(keys & GROUP), 1, keys)

    def test_every_group_member_is_still_reachable(self):
        seen = set()
        for keys in self.generated_predicate_keys(exclusive_config()):
            seen |= keys & GROUP
        self.assertEqual(seen, GROUP)

    def test_reduction_keeps_one_member_and_everything_else(self):
        import random

        attrs = [
            ("trips", attr) for attr in EXCLUSIVE_CONFIG["tables"]["trips"]["attributes"]
        ]
        groups = [frozenset(GROUP)]
        self.assertEqual(max_conflict_free_count(attrs, groups), 2)
        for seed in range(20):
            retained = drop_conflicting_attributes(attrs, groups, random.Random(seed))
            keys = {f"{table}.{attr['name']}" for table, attr in retained}
            self.assertEqual(len(keys & GROUP), 1)
            self.assertIn("trips.num_of_segments", keys)

    def test_a_group_that_cannot_satisfy_the_minimum_is_rejected(self):
        # spatio_temporal needs one spatial and one temporal attribute; putting
        # both in one group makes the category unsatisfiable, which must raise
        # rather than loop.
        config = {
            "srid": 1,
            "entity": {"table": "segments", "key": "id", "expression": "s.id"},
            "tables": {
                "segments": {
                    "name": "segments", "alias": "s", "primary_key": "id",
                    "flags": ["temporal", "spatial"],
                    "attributes": [
                        {"name": "segment_time", "type": "temporal_interval",
                         "dimension": "temporal", "start_expression": "s.t_s",
                         "end_expression": "s.t_e",
                         "domain": {"min": "2013-01-01 00:00:00", "max": "2013-01-02 00:00:00"}},
                        {"name": "segment_geom", "type": "geometry", "dimension": "spatial",
                         "expression": "s.segment_geom",
                         "domain": {"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 1.0}},
                    ],
                }
            },
            "joins": [],
            "exclusive_attribute_groups": [
                ["segments.segment_time", "segments.segment_geom"]
            ],
        }
        validate_config(config)
        with self.assertRaisesRegex(ConfigError, "Not enough attributes"):
            QueryGenerator(config, seed=1).generate_one(
                Category.parse("spatio_temporal.range.single"), 1, 0
            )


class ExclusiveGroupValidationTest(unittest.TestCase):
    def test_accepts_a_well_formed_group(self):
        validate_config(exclusive_config())

    def test_rejects_an_unknown_attribute(self):
        config = exclusive_config(
            exclusive_attribute_groups=[["trips.call_type", "trips.missing"]]
        )
        with self.assertRaisesRegex(ConfigError, "unknown attribute"):
            validate_config(config)

    def test_rejects_a_single_member_group(self):
        config = exclusive_config(exclusive_attribute_groups=[["trips.call_type"]])
        with self.assertRaisesRegex(ConfigError, "at least two attributes"):
            validate_config(config)

    def test_rejects_overlapping_groups(self):
        config = exclusive_config(
            exclusive_attribute_groups=[
                ["trips.call_type", "trips.origin_call"],
                ["trips.origin_call", "trips.origin_stand"],
            ]
        )
        with self.assertRaisesRegex(ConfigError, "overlapping groups"):
            validate_config(config)

    def test_rejects_a_repeated_member(self):
        config = exclusive_config(
            exclusive_attribute_groups=[["trips.call_type", "trips.call_type"]]
        )
        with self.assertRaisesRegex(ConfigError, "repeats an attribute"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
