"""Pools: order-independent subsampling and tamper detection on load."""

import json
import tempfile
import unittest
from pathlib import Path

from query_generation.benchmark_v3.pools import Pools, subsample


class PoolTests(unittest.TestCase):
    def test_subsample_ignores_scan_order(self):
        rows = [[i, f"2019-07-0{i % 9 + 1} 00:00:00.000000", i * 0.5] for i in range(500)]
        self.assertEqual(subsample(rows, 50, seed=4), subsample(list(reversed(rows)), 50, seed=4))
        self.assertNotEqual(subsample(rows, 50, seed=4), subsample(rows, 50, seed=5))

    def test_load_detects_edited_pools(self):
        pools = Pools({"agents": {"columns": ["age"], "rows": [[20.0], [30.0]]}}, {"base_seed": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pools.json"
            pools.save(path)
            self.assertEqual(Pools.load(path).sha256, pools.sha256)
            payload = json.loads(path.read_text())
            payload["tables"]["agents"]["rows"][0][0] = 21.0
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                Pools.load(path)


if __name__ == "__main__":
    unittest.main()
