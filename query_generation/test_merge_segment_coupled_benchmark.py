import copy
import unittest

from query_generation.merge_segment_coupled_benchmark import CORRECTION_NAME, merge_rows


def original_row(query_id, dimension="spatio_temporal", table="trips", attribute="trip_geom"):
    return {
        "query_id": query_id,
        "category": {"dimension": dimension, "interval": "range", "relation": "single", "key": "test"},
        "predicates": [{"table": table, "attribute": attribute}],
        "join_cardinality": 1,
    }


def corrected_row(row):
    output = copy.deepcopy(row)
    output["join_cardinality"] = 5
    output["entity_cardinality"] = 3
    output["semantic_correction"] = {"name": CORRECTION_NAME}
    return output


class SegmentCoupledBenchmarkMergeTest(unittest.TestCase):
    def test_replaces_each_expected_corrected_row_and_preserves_order(self):
        original = [original_row("q1"), original_row("q2", dimension="spatial", attribute="trip_geom")]
        merged = merge_rows(original, {"q1": corrected_row(original[0])})
        self.assertEqual([row["query_id"] for row in merged], ["q1", "q2"])
        self.assertEqual(merged[0]["join_cardinality"], 5)
        self.assertEqual(merged[1]["join_cardinality"], 1)

    def test_rejects_missing_expected_correction(self):
        with self.assertRaises(SystemExit):
            merge_rows([original_row("q1")], {})

    def test_rejects_unexpected_correction(self):
        original = [original_row("q1", dimension="spatial", attribute="trip_geom")]
        with self.assertRaises(SystemExit):
            merge_rows(original, {"q1": corrected_row(original[0])})


if __name__ == "__main__":
    unittest.main()
