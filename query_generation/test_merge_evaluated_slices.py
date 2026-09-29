import unittest
from query_generation.merge_evaluated_slices import merge_rows, validate_segment_coupled_semantics, validate_workload_shape


def row(index, relation="single"):
    return {
        "query_id": f"q{index:08d}",
        "category": {"key": "spatio_temporal.range." + relation, "dimension": "spatio_temporal", "relation": relation},
        "predicates": [
            {"table": "segments", "dimension": "spatial"},
            {"table": "segments", "dimension": "temporal"},
        ],
    }


class EvaluatedSliceMergeTest(unittest.TestCase):
    def test_rejects_missing_slices(self):
        with self.assertRaisesRegex(SystemExit, "missing evaluated rows"):
            merge_rows([row(0)], [])

    def test_rejects_non_segment_spatiotemporal_predicate(self):
        invalid = row(0)
        invalid["predicates"][0]["table"] = "trips"
        with self.assertRaisesRegex(SystemExit, "non-segment spatial"):
            validate_segment_coupled_semantics(invalid)

    def test_validates_expected_distinct_truth_count(self):
        with self.assertRaisesRegex(SystemExit, "distinct-trajectory"):
            validate_workload_shape([row(0)], None, 1)


if __name__ == "__main__":
    unittest.main()
