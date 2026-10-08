import copy
import unittest

from query_generation.query_generator import Category, QueryGenerator
from query_generation.resample_nonzero_benchmark import (
    REPLACEMENT_NAME,
    replace_true_zeros,
    source_category,
    validate_output,
)


def config():
    return {
        "srid": 1,
        "entity": {"table": "items", "key": "id", "expression": "i.id"},
        "tables": {
            "items": {
                "name": "items",
                "alias": "i",
                "primary_key": "id",
                "flags": ["standard"],
                "attributes": [
                    {
                        "name": "value",
                        "type": "numeric",
                        "dimension": "standard",
                        "expression": "i.value",
                        "domain": {"min": 0, "max": 10},
                    }
                ],
            }
        },
        "joins": [],
    }


class FakeExecutor:
    def rows(self, sql):
        self.last_rows_sql = sql
        return [(2.0,), (4.0,), (6.0,)]

    def scalar(self, sql):
        self.last_scalar_sql = sql
        return 3


def row(query_id, cardinality):
    generated = QueryGenerator(config(), seed=7).generate(
        [Category.parse("standard.range.single")], 1
    )[0]
    generated["query_id"] = query_id
    generated["join_cardinality"] = cardinality
    generated["entity_cardinality"] = None
    return generated


class NonzeroReplacementTest(unittest.TestCase):
    def test_replaces_only_true_zero_rows_and_preserves_shape(self):
        source = [row("q00000001", 0), row("q00000002", 5)]
        original_positive = copy.deepcopy(source[1])
        output, attempts = replace_true_zeros(
            source,
            config=config(),
            executor=FakeExecutor(),
            base_seed=123,
            max_attempts=5,
            source_workload="source.jsonl",
            sample_cache_size=3,
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(output[1], original_positive)
        self.assertGreater(output[0]["join_cardinality"], 0)
        self.assertEqual(output[0]["query_id"], "q00000001")
        self.assertEqual(output[0]["nonzero_replacement"]["name"], REPLACEMENT_NAME)
        self.assertEqual(output[0]["nonzero_replacement"]["attempt"], 1)
        self.assertEqual(attempts, [1])

    def test_validation_rejects_a_remaining_zero(self):
        source = [row("q00000001", 0)]
        with self.assertRaisesRegex(ValueError, "not strictly positive"):
            validate_output(source, source)

    def test_rewritten_row_uses_its_original_sampling_category(self):
        target = row("q00000001", 0)
        target["category"] = {
            "dimension": "spatio_temporal",
            "interval": "range",
            "relation": "multi",
            "key": "spatio_temporal.range.multi",
            "category_index": 3,
        }
        target["semantic_correction"] = {
            "original": {
                "category": {
                    "dimension": "spatio_temporal",
                    "interval": "range",
                    "relation": "single",
                    "key": "spatio_temporal.range.single",
                    "category_index": 3,
                }
            }
        }

        self.assertEqual(source_category(target).key, "spatio_temporal.range.single")


if __name__ == "__main__":
    unittest.main()
