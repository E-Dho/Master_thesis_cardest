"""A whole run on a v3 benchmark: three seeds, aggregate, compare."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.report import aggregate_runs, compare_aggregates
from evaluation.eval_core.runner import run_seed

from ._fixture import MULTI, REPO, SINGLE, V3_CONFIG, write_benchmark

FAKE = Path(__file__).with_name("fake_traj_estimator.py")


def _config(root: Path, method: str) -> Path:
    path = root / f"{method}.yaml"
    path.write_text(f"""schema_version: 1
experiment:
  experiment_id: traj_fixture
  method_id: {method}
  variant_id: v
  display_name: Fake {method}
  seeds: [0, 1, 2]
source:
  url: https://example.invalid/r.git
  revision: abc
paths:
  results_root: {root / 'results'}
workloads:
  pol_v3_multi:
    format: traj_v3_jsonl
    queries_jsonl: {root / 'benchmark.jsonl'}
    select:
      relation: multi
    expected_query_count: {8 * MULTI}
    benchmark_config: {V3_CONFIG}
  pol_v3_single:
    format: traj_v3_jsonl
    queries_jsonl: {root / 'benchmark.jsonl'}
    select:
      relation: single
    expected_query_count: {8 * SINGLE}
timing:
  device: cpu
  warmup_passes: 0
  repetitions: 1
resources:
  profile: test
artifacts:
  checkpoint: none
adapter:
  type: external
  working_directory: {REPO}
  environment:
    PYTHONPATH: {REPO}
  evaluate_command: {sys.executable} {FAKE} {{queries_jsonl}} '{{workload_select_json}}' {{predictions_csv}} {{latency_csv}} {method}-{{seed}}
""", encoding="utf-8")
    return path


class EndToEndTest(unittest.TestCase):
    def test_runs_aggregate_and_compare_with_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_benchmark(root / "benchmark.jsonl")
            aggregates = []
            for method in ("fake_a", "fake_b"):
                config = load_experiment_config(_config(root, method))
                runs = [run_seed(config, seed, run_id="fixed") for seed in config.seeds]
                summary = json.loads((runs[0] / "summary.json").read_text())
                multi = summary["workloads"]["pol_v3_multi"]
                self.assertEqual(multi["workload"]["select"], {"relation": "multi"})
                self.assertEqual(multi["workload"]["label_kind_counts"], {"distinct_trips": 8 * MULTI})
                coverage = multi["extension"]["coverage"]
                self.assertEqual(coverage["query_count"], 8 * MULTI)
                self.assertEqual(coverage["unsupported_count"], len([i for i in range(8 * MULTI) if i % 11 == 5]))
                self.assertGreater(coverage["ok_approx_count"], 0)
                self.assertIn("window_area_fraction:[1e-1,1]", multi["extension"]["strata"])
                # the core's own accuracy block is still there and agrees on coverage
                self.assertEqual(multi["accuracy"]["scored_query_count"], coverage["scored_count"])
                aggregate = aggregate_runs(runs, root / "agg" / method)
                self.assertIn("extension", aggregate["workloads"]["pol_v3_multi"])
                self.assertTrue((root / "agg" / method / "trajectory_summary_v1.csv").exists())
                aggregates.append(root / "agg" / method / "comparison.json")
            compare_aggregates(aggregates, root / "compare")
            rows = list(csv.DictReader((root / "compare" / "trajectory_summary_v1.csv").open()))
            self.assertEqual({row["method_id"] for row in rows}, {"fake_a", "fake_b"})
            self.assertEqual({row["workload"] for row in rows}, {"pol_v3_multi", "pol_v3_single"})
            markdown = (root / "compare" / "trajectory_summary_v1.md").read_text()
            self.assertTrue((root / "compare" / "comparison.md").read_text().startswith(
                "# Method comparison (traj_"), "trajectory comparisons must not be titled JOB-light")
            self.assertIn("category:spatio_temporal.range.multi", markdown)

    def test_query_index_cli_maps_ids_back_to_the_benchmark(self) -> None:
        from evaluation.trajectory_pol_porto.traj_eval.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_benchmark(root / "benchmark.jsonl")
            self.assertEqual(main(["query-index", "--config", str(_config(root, "fake_a")),
                                   "--output-directory", str(root / "index")]), 0)
            rows = list(csv.DictReader((root / "index" / "pol_v3_single_query_index.csv").open()))
            self.assertEqual(len(rows), 8 * SINGLE)
            self.assertEqual(rows[0]["query_id"], "0")
            self.assertEqual(rows[0]["label_kind"], "row_count")


if __name__ == "__main__":
    unittest.main()
