"""sample_coverage_map.py: the same samples as traj_sampling, rasterized correctly."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_pol_porto.scripts import sample_coverage_map as scm
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap
from evaluation.trajectory_pol_porto.traj_eval.adapters.sampling import draw_sample

from ._fixture import V3_CONFIG, synthetic_settings, write_synthetic_snapshot


class SampleCoverageMapTest(unittest.TestCase):
    def test_samples_equal_the_adapters_and_counts_add_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_snapshot(root / "snapshot")
            settings = synthetic_settings()
            data = snap.load_snapshot(root / "snapshot")
            trips = snap.load_trip_index(root / "snapshot")
            total = int(trips["trip_id"].size)
            for seed in (0, 1):
                positions = scm.sample_positions(total, 60, seed)
                drawn = draw_sample(settings, data, trips, 60, seed)
                self.assertTrue(np.array_equal(np.asarray(trips["trip_id"])[positions], drawn["trips"]["trip_id"]))

            output = root / "out" / "coverage.npz"
            self.assertEqual(scm.main(["--snapshot-root", str(root / "snapshot"), "--benchmark-config", str(V3_CONFIG),
                                       "--sizes", f"60,{total}", "--seed", "0", "--cell-m", "50",
                                       "--extent", "data", "--points-per-segment", "3",
                                       "--output", str(output)]), 0)
            arrays = np.load(output)
            summary = json.loads(output.with_suffix(".json").read_text())
            # the full sample is the full data; a sub-sample never exceeds it, cell by cell
            self.assertTrue(np.array_equal(arrays[f"n{total}"], arrays["full"]))
            self.assertTrue(np.all(arrays["n60"] <= arrays["full"]))
            self.assertEqual(summary["samples"][1]["cells_hit"], 1.0)
            self.assertLess(summary["samples"][0]["sampled_segments"], summary["population_segments"])
            # with the data extent nothing falls outside: every segment touches 1..3 cells
            segments = int(summary["population_segments"])
            self.assertGreaterEqual(int(arrays["full"].sum()), segments)
            self.assertLessEqual(int(arrays["full"].sum()), 3 * segments)
            with self.assertRaises(SystemExit):  # never overwrites
                scm.main(["--snapshot-root", str(root / "snapshot"), "--benchmark-config", str(V3_CONFIG),
                          "--cell-m", "50", "--output", str(output)])

    def test_rasterize_counts_a_segment_once_per_cell(self) -> None:
        segments = {"sx": np.array([0.5, 0.5]), "sy": np.array([0.5, 0.5]),
                    "ex": np.array([2.5, 0.6]), "ey": np.array([0.5, 0.6])}
        full, (sampled,) = scm.rasterize(segments, np.array([0, 1]), [np.array([True, False])],
                                         (0.0, 0.0, 3.0, 1.0), 1.0, 5)
        # segment 0 crosses cells 0, 1, 2 (5 points: x = 0.5, 1.0, 1.5, 2.0, 2.5); segment 1 stays in cell 0
        self.assertEqual(full.tolist(), [[2, 1, 1]])
        self.assertEqual(sampled.tolist(), [[1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
