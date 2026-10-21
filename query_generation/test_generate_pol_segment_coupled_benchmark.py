import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from query_generation.generate_pol_segment_coupled_benchmark import (
    generate_records,
    generate_shard_incrementally,
    load_jsonl_recover_trailing_write,
)
from query_generation.query_generator import all_valid_categories, load_config


class FakeExecutor:
    def __init__(self, fail_scalar_call=None):
        self.fail_scalar_call = fail_scalar_call
        self.scalar_calls = 0

    def rows(self, sql):
        if "ST_X(sampled_point)" in sql:
            return [(741000.0, 3736000.0)]
        if "s.t_s, s.t_e" in sql or "t.start_time, t.end_time" in sql:
            return [(datetime(2020, 1, 1), datetime(2020, 1, 2))]
        return [(1.0,)]

    def scalar(self, sql):
        self.scalar_calls += 1
        if self.scalar_calls == self.fail_scalar_call:
            raise RuntimeError("simulated interruption")
        return 1


class SegmentCoupledBenchmarkGeneratorTest(unittest.TestCase):
    def test_preserves_16_buckets_and_prohibits_trip_geometry(self):
        config = load_config(Path(__file__).with_name("pol_query_config.json"))
        records = generate_records(
            config,
            categories=all_valid_categories(config),
            queries_per_category=2,
            seed=13,
            executor=FakeExecutor(),
            source_label="test-config",
        )

        self.assertEqual(len(records), 32)
        self.assertEqual(sum(row["category"]["relation"] == "single" for row in records), 16)
        self.assertEqual(sum(row["category"]["relation"] == "multi" for row in records), 16)
        self.assertFalse(
            any(
                predicate["table"] == "trips" and predicate["attribute"] == "trip_geom"
                for row in records
                for predicate in row["predicates"]
            )
        )
        self.assertTrue(
            all(
                row["semantic_correction"]["original"]["category"] == row["category"]
                for row in records
                if "semantic_correction" in row
            )
        )

    def test_incremental_shard_resumes_without_reevaluating_completed_rows(self):
        config_path = Path(__file__).with_name("pol_query_config.json")
        config = load_config(config_path)
        category = all_valid_categories(config)[0]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "shard.jsonl"
            state = Path(tmp) / "shard.state.json"
            interrupted = FakeExecutor(fail_scalar_call=2)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                generate_shard_incrementally(
                    config,
                    category=category,
                    queries_per_category=3,
                    start_index=0,
                    query_count=3,
                    base_seed=17,
                    sample_cache_size=4,
                    executor=interrupted,
                    source_label=str(config_path),
                    output_path=output,
                    state_path=state,
                    progress_interval=0,
                )

            self.assertEqual(len(load_jsonl_recover_trailing_write(output)), 1)
            resumed_executor = FakeExecutor()
            rows, resumed = generate_shard_incrementally(
                config,
                category=category,
                queries_per_category=3,
                start_index=0,
                query_count=3,
                base_seed=17,
                sample_cache_size=4,
                executor=resumed_executor,
                source_label=str(config_path),
                output_path=output,
                state_path=state,
                progress_interval=0,
            )

            self.assertEqual(resumed, 1)
            self.assertEqual(len(rows), 3)
            self.assertEqual(resumed_executor.scalar_calls, 2)
            self.assertEqual([row["category"]["category_index"] for row in rows], [0, 1, 2])

    def test_incomplete_trailing_json_is_discarded(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            path.write_bytes(b'{"query_id":"q1"}\n{"query_id":')
            rows = load_jsonl_recover_trailing_write(path)
            self.assertEqual(rows, [{"query_id": "q1"}])
            self.assertEqual(path.read_text(), '{"query_id":"q1"}\n')
