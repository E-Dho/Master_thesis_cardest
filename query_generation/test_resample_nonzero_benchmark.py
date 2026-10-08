import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from query_generation.query_generator import (
    Category,
    LiveCenterCache,
    QueryGenerator,
    decode_center_value,
    encode_center_value,
)
from query_generation import resample_nonzero_benchmark as resample
from query_generation.resample_nonzero_benchmark import (
    REPLACEMENT_NAME,
    build_summary,
    replace_true_zeros,
    source_category,
    table_subset_counts,
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
    def __init__(self):
        self.scalar_calls = 0
        self.rows_calls = 0

    def rows(self, sql):
        self.rows_calls += 1
        self.last_rows_sql = sql
        return [(2.0,), (4.0,), (6.0,)]

    def scalar(self, sql):
        self.scalar_calls += 1
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


def replace(source, executor, *, base_seed=123, progress=None, centers=None):
    return replace_true_zeros(
        source,
        config=config(),
        executor=executor,
        base_seed=base_seed,
        max_attempts=5,
        source_workload="source.jsonl",
        sample_cache_size=3,
        progress_path=progress,
        centers_cache_path=centers,
    )


class NonzeroReplacementTest(unittest.TestCase):
    def test_replaces_only_true_zero_rows_and_preserves_shape(self):
        source = [row("q00000001", 0), row("q00000002", 5)]
        original_positive = copy.deepcopy(source[1])
        output, attempts, provenance = replace(source, FakeExecutor())

        self.assertEqual(len(output), 2)
        self.assertEqual(output[1], original_positive)
        self.assertGreater(output[0]["join_cardinality"], 0)
        self.assertEqual(output[0]["query_id"], "q00000001")
        self.assertEqual(output[0]["source_row_index"], 0)
        self.assertIsNotNone(output[0]["evaluated_at"])
        self.assertEqual(output[0]["nonzero_replacement"]["name"], REPLACEMENT_NAME)
        self.assertEqual(output[0]["nonzero_replacement"]["attempt"], 1)
        self.assertEqual(attempts, [1])
        self.assertEqual(provenance["replaced_query_ids"], ["q00000001"])
        self.assertEqual(provenance["resumed_replacements"], 0)

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


class ResumeTest(unittest.TestCase):
    def test_progress_sidecar_is_reused_without_touching_the_database(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "queries.jsonl.partial.jsonl"
            source = [row("q00000001", 0), row("q00000002", 5)]

            first_executor = FakeExecutor()
            first, _, _ = replace(source, first_executor, progress=progress)
            self.assertGreater(first_executor.scalar_calls, 0)
            self.assertEqual(len(progress.read_text().splitlines()), 1)

            second_executor = FakeExecutor()
            second, attempts, provenance = replace(
                source, second_executor, progress=progress
            )
            self.assertEqual(second_executor.scalar_calls, 0)
            self.assertEqual(attempts, [1])
            self.assertEqual(provenance["resumed_replacements"], 1)
            self.assertEqual(
                json.loads(json.dumps(second, sort_keys=True, default=str)),
                json.loads(json.dumps(first, sort_keys=True, default=str)),
            )

    def test_progress_from_another_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]
            replace(source, FakeExecutor(), progress=progress)
            with self.assertRaisesRegex(SystemExit, "different --seed"):
                replace(source, FakeExecutor(), base_seed=999, progress=progress)

    def test_progress_from_another_source_workload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]
            replace(source, FakeExecutor(), progress=progress)
            moved = [row("q00000003", 0), source[0]]
            with self.assertRaisesRegex(SystemExit, "row index mismatch"):
                replace(moved, FakeExecutor(), progress=progress)


class MainOrderingTest(unittest.TestCase):
    def test_output_is_written_before_validation_runs(self):
        source = [row("q00000001", 0), row("q00000002", 5)]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source_path = base / "source.jsonl"
            source_path.write_text(
                "".join(
                    json.dumps(entry, sort_keys=True, default=str) + "\n"
                    for entry in source
                ),
                encoding="utf-8",
            )
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config()), encoding="utf-8")
            output_path = base / "queries.jsonl"
            summary_path = base / "benchmark_summary.json"

            class FakeConnection:
                def __enter__(inner):
                    return FakeExecutor()

                def __exit__(inner, *exc):
                    return False

            def exploding_validate(rows, source_rows):
                raise ValueError("boom")

            original_executor = resample.QueryExecutor
            original_validate = resample.validate_output
            argv = [
                "resample_nonzero_benchmark.py",
                "--input", str(source_path),
                "--config", str(config_path),
                "--output", str(output_path),
                "--summary", str(summary_path),
                "--seed", "123",
                "--host", "/tmp",
                "--port", "5432",
                "--dbname", "pol",
            ]
            resample.QueryExecutor = lambda **kwargs: FakeConnection()
            resample.validate_output = exploding_validate
            original_argv = sys.argv
            sys.argv = argv
            try:
                with self.assertRaisesRegex(ValueError, "boom"):
                    resample.main()
            finally:
                sys.argv = original_argv
                resample.QueryExecutor = original_executor
                resample.validate_output = original_validate

            self.assertEqual(len(output_path.read_text().splitlines()), 2)
            self.assertEqual(
                json.loads(summary_path.read_text())["replaced_true_zero_rows"], 1
            )


class CenterPoolTest(unittest.TestCase):
    def test_pools_round_trip_through_the_cache_file(self):
        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            source = [row("q00000001", 0)]

            first_executor = FakeExecutor()
            replace(source, first_executor, centers=centers)
            self.assertGreater(first_executor.rows_calls, 0)
            self.assertEqual(
                json.loads(centers.read_text())["items.value"], [2.0, 4.0, 6.0]
            )

            second_executor = FakeExecutor()
            replace(source, second_executor, centers=centers)
            self.assertEqual(second_executor.rows_calls, 0)

    def test_timestamp_and_row_pools_survive_encoding(self):
        cache = LiveCenterCache(config(), None, 0)
        moment = datetime(2026, 3, 4, 5, 6, 7, 890123)
        cache._cache[("trips", "trip_time")] = [(moment, moment)]
        restored = LiveCenterCache(config(), None, 0)
        restored.restore(json.loads(json.dumps(cache.snapshot())))
        self.assertEqual(restored._cache[("trips", "trip_time")], [(moment, moment)])
        self.assertEqual(decode_center_value(encode_center_value(moment)), moment)
        self.assertEqual(cache.pool_keys(), ["trips.trip_time"])


class SummaryTest(unittest.TestCase):
    def test_summary_counts_replacements_and_table_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.jsonl"
            source = [row("q00000001", 0), row("q00000002", 5)]
            source_path.write_text(
                "".join(
                    json.dumps(entry, sort_keys=True, default=str) + "\n"
                    for entry in source
                ),
                encoding="utf-8",
            )
            output, attempts, provenance = replace(source, FakeExecutor())
            summary = build_summary(
                output,
                source_path=source_path,
                source_rows=source,
                seed=123,
                attempts=attempts,
                provenance=provenance,
            )

        self.assertEqual(summary["replaced_true_zero_rows"], 1)
        self.assertEqual(summary["retained_positive_rows"], 1)
        self.assertEqual(summary["remaining_true_zero_rows"], 0)
        self.assertEqual(summary["resumed_replacements"], 0)
        self.assertEqual(
            summary["table_subset_counts"]["output"], table_subset_counts(output)
        )
        self.assertIn("items.value", summary["center_pools"])

    def test_retained_count_ignores_replacement_blocks_from_an_earlier_run(self):
        source = [row("q00000001", 5), row("q00000002", 5)]
        source[0]["nonzero_replacement"] = {"name": REPLACEMENT_NAME}
        output, attempts, provenance = replace(source, FakeExecutor())
        summary = build_summary(
            output,
            source_path=Path(__file__),
            source_rows=source,
            seed=123,
            attempts=attempts,
            provenance=provenance,
        )
        self.assertEqual(summary["replaced_true_zero_rows"], 0)
        self.assertEqual(summary["retained_positive_rows"], 2)


if __name__ == "__main__":
    unittest.main()
