import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from query_generation.generate_pol_segment_coupled_benchmark import generate_records
from query_generation.merge_pol_segment_coupled_shards import merge_shards
from query_generation.query_generator import all_valid_categories, load_config
from query_generation.test_generate_pol_segment_coupled_benchmark import FakeExecutor


class PolSegmentCoupledShardMergeTest(unittest.TestCase):
    def test_merges_complete_bucket_shards_in_workload_order(self):
        config_path = Path(__file__).with_name("pol_query_config.json")
        config = load_config(config_path)
        categories = all_valid_categories(config)
        rows = generate_records(
            config,
            categories=categories,
            queries_per_category=1,
            seed=29,
            executor=FakeExecutor(),
            source_label=str(config_path),
        )
        with TemporaryDirectory() as tmp:
            shard_dir = Path(tmp)
            for row in rows:
                category = row["category"]["key"]
                path = shard_dir / f"{category}.chunk-00.jsonl"
                path.write_text(json.dumps(row, default=str) + "\n", encoding="utf-8")

            merged, sources = merge_shards(
                shard_dir, config=config, queries_per_category=1
            )

        self.assertEqual(len(merged), 16)
        self.assertEqual(len(sources), 16)
        self.assertEqual(
            [row["category"]["key"] for row in merged],
            [category.key for category in categories],
        )
        self.assertEqual(
            [row["query_id"] for row in merged],
            [f"q{index:08d}" for index in range(1, 17)],
        )

    def test_refuses_missing_category_index(self):
        config_path = Path(__file__).with_name("pol_query_config.json")
        config = load_config(config_path)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no shards found"):
                merge_shards(Path(tmp), config=config, queries_per_category=1)


if __name__ == "__main__":
    unittest.main()
