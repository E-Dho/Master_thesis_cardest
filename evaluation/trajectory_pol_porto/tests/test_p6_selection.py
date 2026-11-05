"""P6 selection under the memory budget, and the own-model inference-only size."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.trajectory_pol_porto.traj_eval import select_tuned
from evaluation.trajectory_pol_porto.traj_eval.tune import COLUMNS


def _write(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for parameters, p50, p90, p99, size in rows:
            writer.writerow({"variant_id": "v", "parameters": json.dumps(parameters), "query_count": 1800,
                             "scored_count": 678, "coverage_fraction": 678 / 1800, "q_p50": p50, "q_p90": p90,
                             "q_p95": p90, "q_p99": p99, "q_max": p99, "zero_estimate_count": 0,
                             "serialized_bytes": size, "theoretical_bytes": size, "build_seconds": 1,
                             "run_directory": "r"})


class SelectionTest(unittest.TestCase):
    def test_budget_and_best_picks_across_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            _write(first, [({"unit_size_m": 25, "time_buckets": 400}, 1.03, 16.8, 174, 1_768_869_066),
                           ({"unit_size_m": 400, "time_buckets": 400}, 1.23, 20.9, 350, 6_534_850),
                           ({"unit_size_m": 400, "time_buckets": 100}, 1.28, 80.3, 1354, 1_626_850)])
            _write(second, [({"unit_size_m": 400, "time_buckets": 3200}, 1.20, 4.0, 60, 52_000_000),
                            ({"unit_size_m": 400, "time_buckets": 1600}, 1.21, 7.0, 90, 26_000_000)])
            rows = select_tuned.read_rows([first, second])
            result = select_tuned.select(rows, budget_bytes=30_000_000)
        self.assertEqual(result["settings"], 5)
        self.assertEqual(result["best"]["parameters"], {"unit_size_m": 400, "time_buckets": 3200})
        self.assertEqual(result["budget"]["parameters"], {"unit_size_m": 400, "time_buckets": 1600})
        self.assertEqual(result["smallest"]["parameters"], {"unit_size_m": 400, "time_buckets": 100})
        self.assertAlmostEqual(result["best_over_budget_factor"], 52 / 30)

    def test_nothing_under_the_budget_and_ties(self) -> None:
        rows = [{"parameters": {"u": 1}, "scored_count": 5, "q_p50": 1, "q_p90": 2, "q_p99": 3,
                 "serialized_bytes": 100, "variant_id": "a", "coverage_fraction": 1, "q_max": 3,
                 "zero_estimate_count": 0, "theoretical_bytes": 100, "source": "s"},
                {"parameters": {"u": 2}, "scored_count": 5, "q_p50": 1, "q_p90": 2, "q_p99": 3,
                 "serialized_bytes": 50, "variant_id": "b", "coverage_fraction": 1, "q_max": 3,
                 "zero_estimate_count": 0, "theoretical_bytes": 50, "source": "s"}]
        result = select_tuned.select(rows, budget_bytes=10)
        self.assertIsNone(result["budget"])
        self.assertEqual(result["best"]["parameters"], {"u": 2})  # tie -> smaller model
        with self.assertRaises(ValueError):  # different coverage would make accuracy incomparable
            select_tuned.select([rows[0], dict(rows[1], scored_count=4)], budget_bytes=10)


class OwnModelSizeTest(unittest.TestCase):
    def test_inference_only_size_drops_the_optimizer_state(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "own_model_size.py"
        spec = importlib.util.spec_from_file_location("own_model_size", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = torch.nn.Linear(200, 100)
        optimizer = torch.optim.Adam(model.parameters())
        model(torch.randn(4, 200)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "step": 7, "trajectory_distinct": {"enabled": True},
                        "trajectory_target_semantics_version": "pol_exact_st_v3", "metadata": {}}, path)
            before = path.read_bytes()
            row = module.measure(path)
            self.assertEqual(path.read_bytes(), before)  # read-only
        self.assertEqual(row["parameters"], 200 * 100 + 100)
        self.assertEqual(row["parameter_bytes"], 4 * (200 * 100 + 100))
        self.assertTrue(row["has_optimizer_state"])
        self.assertLess(row["inference_only_bytes"], row["checkpoint_bytes"] / 2)  # Adam keeps 2 moments
        self.assertGreater(row["inference_only_bytes"], row["parameter_bytes"])
        self.assertEqual(row["step"], 7)
        self.assertTrue(row["trajectory_distinct_enabled"])


if __name__ == "__main__":
    unittest.main()
