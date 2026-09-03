from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.scripts.analyze_pol_spatial_intersections import (
    SpatialRectangleQuery,
    analyze_rectangle_query,
    bounded_range_breakdown_rows,
    summarize_results,
    _load_spatial_queries,
)


class PolSpatialAnalysisTest(unittest.TestCase):
    def test_classifies_endpoint_and_crossing_matches(self) -> None:
        query = _query("q_crossing")
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
        self.assertEqual(stats.M_MBR_Q, 3)
        self.assertEqual(stats.MBR_false_positives, 0)
        self.assertAlmostEqual(stats.MBR_precision, 1.0)
        self.assertAlmostEqual(stats.MBR_overestimation, 1.0)
        self.assertAlmostEqual(stats.current_endpoint_model_recall, 1.0 / 3.0)
        self.assertAlmostEqual(
            stats.mean_matching_segment_length,
            (2 ** 0.5 + 2.0 + 20 ** 0.5) / 3.0,
        )
        self.assertIsNotNone(stats.p50_matching_segment_length)
        self.assertIsNotNone(stats.p90_matching_segment_length)
        self.assertIsNotNone(stats.p95_matching_segment_length)
        self.assertIsNotNone(stats.p99_matching_segment_length)

    def test_counts_diagonal_mbr_false_positive(self) -> None:
        stats = analyze_rectangle_query(
            _query(
                "q_mbr_false_positive",
                min_x=0.0,
                min_y=1.5,
                max_x=0.5,
                max_y=2.0,
            ),
            sx=np.asarray([0.0, 0.25]),
            sy=np.asarray([0.0, 1.75]),
            ex=np.asarray([2.0, 0.30]),
            ey=np.asarray([2.0, 1.80]),
            chunk_size=1,
        )
        self.assertEqual(stats.M_Q, 1)
        self.assertEqual(stats.M_MBR_Q, 2)
        self.assertEqual(stats.MBR_false_positives, 1)
        self.assertAlmostEqual(stats.MBR_precision, 0.5)
        self.assertAlmostEqual(stats.MBR_overestimation, 2.0)

    def test_summary_reports_global_fractions_and_buckets(self) -> None:
        results = [
            analyze_rectangle_query(
                _query("q1"),
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
        self.assertEqual(summary["total_mbr_candidate_segments"], 2)
        self.assertEqual(summary["total_mbr_false_positives"], 0)
        self.assertAlmostEqual(
            summary["cardinality_weighted_global_fractions"]["B_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["cardinality_weighted_global_fractions"]["C_fraction"],
            0.5,
        )
        self.assertAlmostEqual(
            summary["cardinality_weighted_global_fractions"]["MBR_precision"],
            1.0,
        )
        self.assertTrue(summary["sanity_checks"]["all_partition_counts_match"])
        self.assertTrue(summary["sanity_checks"]["all_mbr_counts_cover_true_matches"])
        self.assertTrue(summary["sanity_checks"]["all_mbr_false_positives_nonnegative"])
        self.assertEqual(summary["normalized_window_size_buckets"][0]["query_count"], 1)
        self.assertEqual(
            summary["normalized_window_size_buckets"][0]["MBR_precision"]["count"],
            1,
        )
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
        interval = summary["stratified_statistics"]["interval"]["range"]
        self.assertEqual(interval["query_count"], 1)
        self.assertEqual(interval["sum_M_Q"], 2)
        self.assertAlmostEqual(
            interval["cardinality_weighted_fractions"]["C_fraction"],
            0.5,
        )

    def test_cli_reads_spatial_workload_and_segments_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queries = root / "queries.jsonl"
            queries.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "category": {
                            "key": "spatial.range.single",
                            "dimension": "spatial",
                            "interval": "range",
                            "relation": "single",
                        },
                        "predicates": [
                            {
                                "table": "segments",
                                "attribute": "segment_geom",
                                "mode": "spatial_intersects",
                                "min_x": 0.0,
                                "min_y": 0.0,
                                "max_x": 2.0,
                                "max_y": 2.0,
                                "center_source": "live_row",
                                "range_source": "uniform",
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
            breakdown = root / "bounded_range_breakdown.csv"
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
                    "--bounded-range-breakdown-csv",
                    str(breakdown),
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
            self.assertEqual(rows[0]["M_MBR_Q"], 2)
            self.assertEqual(rows[0]["MBR_false_positives"], 0)
            self.assertEqual(rows[0]["B_Q"], 1)
            self.assertEqual(rows[0]["C_Q"], 1)
            self.assertEqual(rows[0]["category_dimension"], "spatial")
            self.assertEqual(rows[0]["category_interval"], "range")
            self.assertEqual(rows[0]["category_relation"], "single")
            self.assertEqual(rows[0]["center_source"], "live_row")
            self.assertEqual(rows[0]["range_source"], "uniform")
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["spatial_queries_analyzed"], 1)
            with breakdown.open(encoding="utf-8") as handle:
                breakdown_rows = list(csv.DictReader(handle))
            self.assertEqual(len(breakdown_rows), 1)
            self.assertEqual(breakdown_rows[0]["interval"], "range")
            self.assertEqual(breakdown_rows[0]["width_source"], "uniform")
            self.assertEqual(breakdown_rows[0]["center_source"], "live_row")
            self.assertEqual(breakdown_rows[0]["median_M_Q"], "2.0")
            self.assertEqual(breakdown_rows[0]["sum_M_MBR_Q"], "2")
            self.assertEqual(breakdown_rows[0]["sum_MBR_false_positives"], "0")
            self.assertEqual(breakdown_rows[0]["MBR_precision_weighted"], "1.0")
            self.assertTrue(
                payload["predicate_filter"]["default_segments_segment_geom_only"]
            )
            self.assertIn("interval", payload["stratified_statistics"])
            self.assertIn(
                "description",
                payload["cardinality_weighted_global_fractions"],
            )

    def test_default_filter_keeps_only_segment_geometry_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.jsonl"
            record = {
                "query_id": "q1",
                "category": {
                    "dimension": "spatio_temporal",
                    "interval": "unbounded",
                    "relation": "multi",
                },
                "predicates": [
                    {
                        "table": "trips",
                        "attribute": "trip_geom",
                        "mode": "spatial_intersects",
                        "min_x": 0.0,
                        "min_y": 0.0,
                        "max_x": 1.0,
                        "max_y": 1.0,
                        "center_source": "domain",
                        "range_source": "exponential",
                    },
                    {
                        "table": "segments",
                        "attribute": "segment_geom",
                        "mode": "spatial_unbounded",
                        "min_x": 0.0,
                        "min_y": 0.0,
                        "max_x": 1.0,
                        "max_y": 1.0,
                        "center_source": "live_row",
                    },
                ],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            default_queries = list(_load_spatial_queries([path]))
            self.assertEqual(len(default_queries), 1)
            self.assertEqual(default_queries[0].table, "segments")
            self.assertEqual(default_queries[0].attribute, "segment_geom")
            self.assertEqual(default_queries[0].category_dimension, "spatio_temporal")
            self.assertEqual(default_queries[0].category_interval, "unbounded")
            self.assertEqual(default_queries[0].category_relation, "multi")
            self.assertEqual(default_queries[0].center_source, "live_row")
            self.assertIsNone(default_queries[0].range_source)

            all_queries = list(
                _load_spatial_queries(
                    [path],
                    include_other_spatial_predicates=True,
                )
            )
            self.assertEqual(len(all_queries), 2)

    def test_grouping_reports_required_workload_generation_strata(self) -> None:
        base = dict(
            sx=np.asarray([0.5, -1.0, -1.0]),
            sy=np.asarray([0.5, 1.0, -1.0]),
            ex=np.asarray([1.5, 3.0, 3.0]),
            ey=np.asarray([1.5, 1.0, 1.0]),
            chunk_size=10,
        )
        results = [
            analyze_rectangle_query(
                _query(
                    "range_live_uniform",
                    category_interval="range",
                    category_dimension="spatial",
                    center_source="live_row",
                    range_source="uniform",
                    mode="spatial_intersects",
                ),
                **base,
            ),
            analyze_rectangle_query(
                _query(
                    "range_domain_exponential",
                    category_interval="range",
                    category_dimension="spatio_temporal",
                    center_source="domain",
                    range_source="exponential",
                    mode="spatial_intersects",
                ),
                **base,
            ),
            analyze_rectangle_query(
                _query(
                    "unbounded",
                    category_interval="unbounded",
                    category_dimension="spatial",
                    center_source="live_row",
                    range_source=None,
                    mode="spatial_unbounded",
                ),
                **base,
            ),
        ]
        summary = summarize_results(results, bucket_edges=(0.0, 10.0, float("inf")))
        self.assertEqual(
            summary["stratified_statistics"]["interval"]["range"]["query_count"],
            2,
        )
        self.assertEqual(
            summary["stratified_statistics"]["interval"]["unbounded"]["query_count"],
            1,
        )
        self.assertEqual(
            summary["stratified_statistics"]["center_source"]["live_row"]["query_count"],
            2,
        )
        self.assertEqual(
            summary["stratified_statistics"]["center_source"]["domain"]["query_count"],
            1,
        )
        self.assertIn("uniform", summary["stratified_statistics"]["range_source"])
        self.assertIn("exponential", summary["stratified_statistics"]["range_source"])
        interaction = summary["stratified_statistics"][
            "bounded_range_interval_x_width_source_x_center_source"
        ]
        self.assertEqual(interaction["range x uniform x live_row"]["query_count"], 1)
        self.assertEqual(interaction["range x exponential x domain"]["query_count"], 1)
        self.assertNotIn("unbounded x missing x live_row", interaction)
        self.assertEqual(
            summary["stratified_statistics"]["category_dimension"]["spatial"][
                "query_count"
            ],
            2,
        )
        self.assertEqual(
            len(summary["normalized_window_size_buckets_for_range_queries"]),
            2,
        )
        rows = bounded_range_breakdown_rows(results)
        self.assertEqual(len(rows), 2)
        live_uniform = {
            row["width_source"]: row
            for row in rows
            if row["center_source"] == "live_row"
        }
        self.assertIn("uniform", live_uniform)
        self.assertEqual(live_uniform["uniform"]["query_count"], 1)
        self.assertEqual(live_uniform["uniform"]["median_M_Q"], 3.0)
        self.assertEqual(live_uniform["uniform"]["sum_M_MBR_Q"], 3)
        self.assertAlmostEqual(live_uniform["uniform"]["MBR_precision_weighted"], 1.0)


def _query(
    query_id: str,
    *,
    category_dimension: str = "spatial",
    category_interval: str = "range",
    category_relation: str = "single",
    center_source: str = "live_row",
    range_source: str | None = "uniform",
    mode: str = "spatial_intersects",
    min_x: float = 0.0,
    min_y: float = 0.0,
    max_x: float = 2.0,
    max_y: float = 2.0,
) -> SpatialRectangleQuery:
    return SpatialRectangleQuery(
        query_id=query_id,
        predicate_index=0,
        table="segments",
        attribute="segment_geom",
        mode=mode,
        category_dimension=category_dimension,
        category_interval=category_interval,
        category_relation=category_relation,
        center_source=center_source,
        range_source=range_source,
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
    )


if __name__ == "__main__":
    unittest.main()
