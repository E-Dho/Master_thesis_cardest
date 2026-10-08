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
    canonical_sql_hash,
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


def replace(
    source,
    executor,
    *,
    base_seed=123,
    progress=None,
    centers=None,
    generator_config=None,
    sample_cache_size=3,
    source_sha="source-sha",
):
    return replace_true_zeros(
        source,
        config=generator_config or config(),
        executor=executor,
        base_seed=base_seed,
        max_attempts=5,
        source_workload="source.jsonl",
        sample_cache_size=sample_cache_size,
        source_workload_sha256=source_sha,
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
            lines = progress.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                json.loads(lines[0])["progress_header"]["source_workload_sha256"],
                "source-sha",
            )

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
            with self.assertRaisesRegex(SystemExit, "different seed"):
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


class DuplicateGuardTest(unittest.TestCase):
    def test_resumed_hashes_guard_rows_sampled_before_their_own_index(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0), row("q00000002", 0)]

            # Accept only the later zero row, so the sidecar holds a replacement
            # whose slot the loop does not reach until after the earlier row has
            # been sampled.
            later, _, _ = replace([source[1]], FakeExecutor())
            later_row = later[0]
            later_row["query_id"] = "q00000002"
            later_row["source_row_index"] = 1
            later_row["nonzero_replacement"].update(
                {
                    "source_row_index": 1,
                    "replaced_query_id": "q00000002",
                    "replaced_sql_sha256": canonical_sql_hash(source[1]),
                    "replacement_seed": resample.replacement_seed(123, "q00000002", 1),
                }
            )
            header = {
                "progress_header": resample.centers_binding(
                    config=config(),
                    source_workload_sha256="source-sha",
                    sample_cache_size=3,
                    seed=123,
                )
            }
            progress.write_text(
                json.dumps(header, sort_keys=True)
                + "\n"
                + json.dumps(later_row, sort_keys=True, default=str)
                + "\n",
                encoding="utf-8",
            )

            seen = {}
            original = resample.replacement_record

            def capture(target, **kwargs):
                seen.setdefault("hashes", set(kwargs["used_sql_hashes"]))
                return original(target, **kwargs)

            resample.replacement_record = capture
            try:
                replace(source, FakeExecutor(), progress=progress)
            finally:
                resample.replacement_record = original

            self.assertIn(canonical_sql_hash(later_row), seen["hashes"])

    def test_duplicate_sql_inside_the_progress_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0), row("q00000002", 0)]
            first, _, _ = replace(source, FakeExecutor(), progress=progress)
            lines = progress.read_text().splitlines()
            self.assertEqual(len(lines), 3)
            clone = json.loads(lines[1])
            clone["sql"] = json.loads(lines[2])["sql"]
            progress.write_text(
                "\n".join([lines[0], json.dumps(clone, sort_keys=True), lines[2]]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "duplicate replacement SQL"):
                replace(source, FakeExecutor(), progress=progress)

    def test_progress_from_another_source_workload_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]
            replace(source, FakeExecutor(), progress=progress)
            with self.assertRaisesRegex(SystemExit, "different source_workload_sha256"):
                replace(source, FakeExecutor(), progress=progress, source_sha="other-sha")

    def test_progress_without_a_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]
            first, _, _ = replace(source, FakeExecutor())
            progress.write_text(
                json.dumps(first[0], sort_keys=True, default=str) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SystemExit, "no progress header"):
                replace(source, FakeExecutor(), progress=progress)

    def test_progress_from_another_generator_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]
            replace(source, FakeExecutor(), progress=progress)
            changed = config()
            changed["tables"]["items"]["attributes"][0]["domain"]["max"] = 99
            with self.assertRaisesRegex(SystemExit, "different config_hash"):
                replace(source, FakeExecutor(), progress=progress, generator_config=changed)


class CenterPoolTest(unittest.TestCase):
    def test_cache_callback_sees_the_newly_fetched_pool(self):
        executor = FakeExecutor()
        captured = []
        cache = LiveCenterCache(
            config(),
            executor,
            3,
            on_pool_cached=lambda: captured.append(cache.snapshot()),
        )

        values = cache.values(
            "items", config()["tables"]["items"]["attributes"][0]
        )

        self.assertEqual(values, [2.0, 4.0, 6.0])
        self.assertEqual(captured, [{"items.value": [2.0, 4.0, 6.0]}])

    def test_pools_round_trip_through_the_cache_file(self):
        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            source = [row("q00000001", 0)]

            first_executor = FakeExecutor()
            replace(source, first_executor, centers=centers)
            self.assertGreater(first_executor.rows_calls, 0)
            payload = json.loads(centers.read_text())
            self.assertEqual(payload["pools"]["items.value"], [2.0, 4.0, 6.0])
            self.assertEqual(payload["binding"]["source_workload_sha256"], "source-sha")
            self.assertEqual(payload["binding"]["seed"], 123)

            second_executor = FakeExecutor()
            replace(source, second_executor, centers=centers)
            self.assertEqual(second_executor.rows_calls, 0)

    def test_pools_are_persisted_before_the_replacement_that_used_them(self):
        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            progress = Path(directory) / "progress.jsonl"
            source = [row("q00000001", 0)]

            original_append = resample.append_jsonl

            def exploding_append(path, entry):
                if "progress_header" in entry:
                    return original_append(path, entry)
                raise RuntimeError("interrupted")

            resample.append_jsonl = exploding_append
            try:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    replace(source, FakeExecutor(), progress=progress, centers=centers)
            finally:
                resample.append_jsonl = original_append

            self.assertEqual(len(progress.read_text().splitlines()), 1)
            self.assertEqual(
                json.loads(centers.read_text())["pools"]["items.value"], [2.0, 4.0, 6.0]
            )

    def test_pools_from_a_rejected_candidate_are_persisted(self):
        class RejectingExecutor(FakeExecutor):
            """Returns zero until the pools have been fetched and a retry happens."""

            def scalar(self, sql):
                super().scalar(sql)
                return 0 if self.scalar_calls == 1 else 3

        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            executor = RejectingExecutor()
            captured = {}
            original = resample.write_json_atomic

            def capture(path, payload):
                captured.setdefault("scalar_calls_at_first_write", executor.scalar_calls)
                return original(path, payload)

            resample.write_json_atomic = capture
            try:
                output, attempts, _ = replace(
                    [row("q00000001", 0)], executor, centers=centers
                )
            finally:
                resample.write_json_atomic = original

            # The pools reached disk during the first (rejected) candidate's
            # generation, before any COUNT(*) had run.
            self.assertEqual(captured["scalar_calls_at_first_write"], 0)
            self.assertEqual(attempts, [2])
            self.assertEqual(
                json.loads(centers.read_text())["pools"]["items.value"], [2.0, 4.0, 6.0]
            )

    def test_atomic_snapshot_fsyncs_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "centers.json"
            called = []
            original = resample.fsync_directory
            resample.fsync_directory = lambda parent: called.append(parent)
            try:
                resample.write_json_atomic(path, {"pool": []})
            finally:
                resample.fsync_directory = original

            self.assertEqual(called, [path.parent])

    def test_cache_bound_to_another_config_or_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            source = [row("q00000001", 0)]
            replace(source, FakeExecutor(), centers=centers)

            changed = config()
            changed["tables"]["items"]["attributes"][0]["domain"]["max"] = 99
            with self.assertRaisesRegex(SystemExit, "config_hash"):
                replace(source, FakeExecutor(), centers=centers, generator_config=changed)
            with self.assertRaisesRegex(SystemExit, "seed"):
                replace(source, FakeExecutor(), centers=centers, base_seed=999)
            with self.assertRaisesRegex(SystemExit, "sample_cache_size"):
                replace(source, FakeExecutor(), centers=centers, sample_cache_size=4)

    def test_foreign_cache_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            centers = Path(directory) / "centers.json"
            centers.write_text(json.dumps({"items.value": [1.0]}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "not a center-pool snapshot"):
                replace([row("q00000001", 0)], FakeExecutor(), centers=centers)

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
