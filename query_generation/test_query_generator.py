import copy
import json
import tempfile
import unittest
from pathlib import Path

from query_generation.query_generator import (
    Category,
    ConfigError,
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


if __name__ == "__main__":
    unittest.main()
