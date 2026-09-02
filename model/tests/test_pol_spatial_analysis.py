from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.scripts.analyze_pol_spatial_intersections import (
    SpatialRectangleQuery,
    analyze_rectangle_query,
    summarize_results,
)


class PolSpatialAnalysisTest(unittest.TestCase):
    def test_classifies_endpoint_and_crossing_matches(self) -> None:
        query = SpatialRectangleQuery(
            query_id="q_crossing",
            predicate_index=0,
            table="segments",
            attribute="segment_geom",
            mode="spatial_intersects",
            min_x=0.0,
            min_y=0.0,
            max_x=2.0,
            max_y=2.0,
        )
        stats = analyze_rectangle_query(
            query,
            sx=np.asarray([0.5, -1.0, -1.0, 3.0]),
            sy=np.asarray([0.5, 1.0, -1.0, 3.0]),
            ex=np.asarray([1.5, 1.0, 3.0, 4.0]),
            ey=np.asarray([1.5, 1.0, 1.0, 4.0]),
            chunk_size=2,
        )
        self.assertEqual(stats.M_Q, 3)
        self.assertEqual(stats.B_Q, 1)
        self.assertEqual(stats.O_Q, 1)
        self.assertEqual(stats.C_Q, 1)
        self.assertAlmostEqual(stats.current_endpoint_model_recall, 1.0 / 3.0)
        self.assertAlmostEqual(
            stats.mean_matching_segment_length,
            (2 ** 0.5 + 2.0 + 20 ** 0.5) / 3.0,
        )
        self.assertIsNotNone(stats.p50_matching_segment_length)
        self.assertIsNotNone(stats.p90_matching_segment_length)
        self.assertIsNotNone(stats.p95_matching_segment_length)
        self.assertIsNotNone(stats.p99_matching_segment_length)

    def test_summary_reports_global_fractions_and_buckets(self) -> None:
        results = [
            analyze_rectangle_query(
                SpatialRectangleQuery(
                    query_id="q1",
                    predicate_index=0,
                    table="segments",
                    attribute="segment_geom",
                    mode="spatial_intersects",
                    min_x=0.0,
                    min_y=0.0,
                    max_x=2.0,
                    max_y=2.0,
                ),
                sx=np.asarray([0.5, -1.0]),
                sy=np.asarray([0.5, 1.0]),
                ex=np.asarray([1.5, 3.0]),
                ey=np.asarray([1.5, 1.0]),
                chunk_size=10,
            )
        ]
        summary = summarize_results(results, bucket_edges=(0.0, 10.0, float("inf")))
        self.assertEqual(summary["spatial_queries_analyzed"], 1)
        self.assertEqual(summary["total_matching_segments"], 2)
        self.assertAlmostEqual(
            summary["cardinality_weighted_global_fractions"]["B_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["cardinality_weighted_global_fractions"]["C_fraction"],
            0.5,
        )
        self.assertTrue(summary["sanity_checks"]["all_partition_counts_match"])
        self.assertEqual(summary["normalized_window_size_buckets"][0]["query_count"], 1)
        self.assertEqual(
            summary["matching_segment_length_percentiles_by_query"]["mean"]["count"],
            1,
        )
        self.assertEqual(
            summary["normalized_window_size_percentiles"][
                "sqrt_area_over_mean_matching_segment_length"
            ]["count"],
            1,
        )

    def test_cli_reads_spatial_workload_and_segments_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queries = root / "queries.jsonl"
            queries.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "category": {"key": "spatial.range.single"},
                        "predicates": [
                            {
                                "table": "segments",
                                "attribute": "segment_geom",
                                "mode": "spatial_intersects",
                                "min_x": 0.0,
                                "min_y": 0.0,
                                "max_x": 2.0,
                                "max_y": 2.0,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            segments = root / "segments.tsv"
            segments.write_text(
                "\n".join(
                    [
                        "1\t0\t0.5\t0.5\t1.5\t1.5\t2020-01-01T00:00:00.000\t2020-01-01T00:01:00.000",
                        "1\t1\t-1.0\t1.0\t3.0\t1.0\t2020-01-01T00:01:00.000\t2020-01-01T00:02:00.000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "analysis.jsonl"
            summary = root / "summary.json"
            from model.scripts.analyze_pol_spatial_intersections import main
            import sys

            previous = sys.argv
            try:
                sys.argv = [
                    "analyze_pol_spatial_intersections",
                    "--queries",
                    str(queries),
                    "--segments-tsv",
                    str(segments),
                    "--output-jsonl",
                    str(output),
                    "--summary",
                    str(summary),
                ]
                main()
            finally:
                sys.argv = previous
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows[0]["M_Q"], 2)
            self.assertEqual(rows[0]["B_Q"], 1)
            self.assertEqual(rows[0]["C_Q"], 1)
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["spatial_queries_analyzed"], 1)


if __name__ == "__main__":
    unittest.main()
