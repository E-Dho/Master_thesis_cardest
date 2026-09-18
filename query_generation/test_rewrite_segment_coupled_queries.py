import unittest

from query_generation.rewrite_segment_coupled_queries import needs_segment_correction, rewrite_row


def trip_row(tables):
    return {
        "query_id": "q00000001",
        "category": {
            "dimension": "spatio_temporal",
            "interval": "range",
            "relation": "single",
            "key": "spatio_temporal.range.single",
            "category_index": 0,
        },
        "tables": tables,
        "joins": [],
        "predicates": [
            {
                "table": "trips",
                "attribute": "trip_geom",
                "dimension": "spatial",
                "mode": "spatial_intersects",
                "sql": "ST_Intersects(t.trip_geom, ST_MakeEnvelope(1, 2, 3, 4, 26916))",
            },
            {
                "table": "trips",
                "attribute": "trip_time",
                "dimension": "temporal",
                "mode": "temporal_overlap",
                "sql": "t.start_time < timestamp '2020-01-02' AND t.end_time >= timestamp '2020-01-01'",
            },
        ],
        "sql": "old sql",
        "entity_sql": None,
        "join_cardinality": 12,
        "entity_cardinality": None,
    }


class SegmentCoupledRewriteTest(unittest.TestCase):
    def test_rewrites_trip_only_query_to_matching_segment_measure(self):
        row = trip_row(["trips"])
        self.assertTrue(needs_segment_correction(row))

        rewritten = rewrite_row(row, "input.jsonl")

        self.assertEqual(rewritten["tables"], ["segments", "trips"])
        self.assertEqual(rewritten["category"]["key"], "spatio_temporal.range.multi")
        self.assertEqual(rewritten["category"]["relation"], "multi")
        self.assertIn("JOIN pol.trips t ON t.trip_id = s.trip_id", rewritten["sql"])
        self.assertIn("ST_Intersects(s.segment_geom", rewritten["sql"])
        self.assertIn("s.t_s <", rewritten["sql"])
        self.assertIn("s.t_e >=", rewritten["sql"])
        self.assertIn("COUNT(DISTINCT t.trip_id)", rewritten["entity_sql"])
        self.assertIsNone(rewritten["join_cardinality"])
        self.assertEqual(rewritten["semantic_correction"]["original"]["sql"], "old sql")

    def test_rewrites_trip_time_on_existing_segment_row(self):
        row = trip_row(["segments", "trips"])
        row["predicates"] = row["predicates"][1:] + [
            {
                "table": "segments",
                "attribute": "segment_geom",
                "dimension": "spatial",
                "mode": "spatial_intersects",
                "sql": "ST_Intersects(s.segment_geom, ST_MakeEnvelope(1, 2, 3, 4, 26916))",
            }
        ]

        rewritten = rewrite_row(row, "input.jsonl")

        self.assertEqual(rewritten["tables"], ["segments", "trips"])
        self.assertIn("s.t_s <", rewritten["sql"])
        self.assertNotIn("t.start_time", rewritten["sql"])

    def test_leaves_non_spatiotemporal_rows_out_of_scope(self):
        row = trip_row(["trips"])
        row["category"]["dimension"] = "spatial"
        self.assertFalse(needs_segment_correction(row))


if __name__ == "__main__":
    unittest.main()
