"""Same-query comparison (traj_eval/same_query.py) on real pipeline runs of the fixture."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.report import aggregate_runs
from evaluation.eval_core.runner import run_seed
from evaluation.trajectory_pol_porto.traj_eval import same_query

from ._fixture import write_benchmark, write_synthetic_snapshot
from .test_p3_adapters import _config


def _errors(run: Path, workload: str) -> dict:
    with (run / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        return {row["query_id"]: float(row["smoothed_q_error"]) for row in csv.DictReader(handle)
                if row["workload"] == workload and row["status"] == "ok"}


class SameQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")
        cls.aggregates, cls.runs = {}, {}
        for method, adapter_type, variant, seeds, extra in (
            ("exact_mbr", "traj_exact_mbr", "boxes", "[0]", {}),
            ("traj_sampling", "traj_sampling", "n40", "[0, 1]", {"sample_trips": 40}),
            ("case_plus", "traj_case_plus", "base", "[0]", {"unit_size_m": 200, "plus": "true",
                                                            "temporal": "false"}),
        ):
            config = load_experiment_config(_config(cls.root, method, adapter_type, variant, seeds=seeds, **extra))
            runs = [run_seed(config, seed, run_id="sq") for seed in config.seeds]
            aggregate_runs(runs, cls.root / "agg" / method)
            cls.aggregates[f"{method}:{variant}"] = cls.root / "agg" / method / "comparison.json"
            cls.runs[f"{method}:{variant}"] = runs

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_intersection_pairs_and_metrics(self) -> None:
        result = same_query.compare(list(self.aggregates.values()), ["traj_sampling:n40", "own_model:x"])
        self.assertEqual(result["missing_references"], ["own_model:x"])
        multi = result["workloads"]["pol_v3_multi"]
        answered = {key: set.intersection(*(set(_errors(run, "pol_v3_multi")) for run in runs))
                    for key, runs in self.runs.items()}
        common = set.intersection(*answered.values())
        self.assertGreater(len(common), 5)
        self.assertLess(len(common), min(len(v) for v in answered.values()) + 1)
        self.assertEqual(multi["intersection"]["queries"], len(common))
        self.assertEqual(multi["coverage"], {k: len(v) for k, v in answered.items()})
        # the metrics are those of exactly the common queries, averaged over the seeds
        sampling = multi["intersection"]["strata"]["all"]["traj_sampling:n40"]
        p50 = [float(np.percentile([_errors(run, "pol_v3_multi")[q] for q in common], 50))
               for run in self.runs["traj_sampling:n40"]]
        self.assertAlmostEqual(sampling["p50_mean"], float(np.mean(p50)))
        self.assertAlmostEqual(sampling["p50_std"], float(np.std(p50, ddof=1)))
        self.assertEqual(sampling["seeds"], 2)
        wins = sum(cell["wins"] for cell in multi["intersection"]["strata"]["all"].values())
        self.assertAlmostEqual(wins, 1.0)
        # per query type: the categories partition the common queries
        categories = [name for name in multi["intersection"]["strata"] if name.startswith("category:")]
        self.assertEqual(sum(multi["intersection"]["strata"][c]["exact_mbr:boxes"]["queries"] for c in categories),
                         len(common))
        # Exact-MBR answers no single query: not compared there, the others are
        single = result["workloads"]["pol_v3_single"]
        self.assertIn("exact_mbr:boxes", single["not_compared"])
        self.assertEqual(single["intersection"]["methods"], ["traj_sampling:n40"])
        # pairs: the reference against each other method on the queries both answer
        pair = multi["pairs"]["traj_sampling:n40|exact_mbr:boxes"]
        self.assertEqual(pair["queries"], len(answered["traj_sampling:n40"] & answered["exact_mbr:boxes"]))
        self.assertGreater(pair["queries"], multi["intersection"]["queries"])
        exact = pair["strata"]["all"]["exact_mbr:boxes"]
        values = [_errors(self.runs["exact_mbr:boxes"][0], "pol_v3_multi")[q]
                  for q in answered["traj_sampling:n40"] & answered["exact_mbr:boxes"]]
        self.assertAlmostEqual(exact["gmean_mean"], math.exp(float(np.mean(np.log(values)))))

    def test_outputs_and_cli(self) -> None:
        output = self.root / "out"
        args = [x for path in self.aggregates.values() for x in ("--aggregate-json", str(path))]
        self.assertEqual(same_query.main(args + ["--reference", "traj_sampling:n40", "--output", str(output)]), 0)
        rows = list(csv.DictReader((output / "same_query.csv").open()))
        self.assertTrue(any(r["scope"] == "intersection" and r["stratum"] == "all" for r in rows))
        self.assertTrue(any(r["scope"].startswith("pair:") for r in rows))
        text = (output / "same_query.md").read_text()
        self.assertIn("## pol_v3_multi", text)
        self.assertIn("Not compared (answer no query): exact_mbr:boxes", text)
        self.assertEqual(json.loads((output / "same_query.json").read_text())["schema_version"], same_query.SCHEMA)

    def test_config_input_reads_the_current_aggregate(self) -> None:
        from evaluation.eval_core.cli import main as core_main

        configs = [self.root / "exact_mbr_boxes.yaml", self.root / "traj_sampling_n40.yaml"]
        with self.assertRaises(FileNotFoundError):  # never aggregated with `cli aggregate`
            same_query.aggregate_of(self.root / "case_plus_base.yaml")
        for config in configs:
            self.assertEqual(core_main(["aggregate", "--config", str(config)]), 0)
        output = self.root / "from_configs"
        args = [x for config in configs for x in ("--config", str(config))]
        self.assertEqual(same_query.main(args + ["--output", str(output)]), 0)
        result = json.loads((output / "same_query.json").read_text())
        self.assertEqual(sorted(result["methods"]), ["exact_mbr:boxes", "traj_sampling:n40"])

    def test_aggregate_compare_job_writes_the_same_query_tables(self) -> None:
        import os
        import subprocess
        import sys

        repo = Path(__file__).resolve().parents[3]
        configs = [self.root / "exact_mbr_boxes.yaml", self.root / "traj_sampling_n40.yaml"]
        env = {**os.environ, "REPO_ROOT": str(repo), "PYTHON": sys.executable,
               "CONFIGS": ":".join(str(c) for c in configs), "COMPARISON_DIR": str(self.root / "job_compare"),
               "SAME_QUERY_REFERENCES": "traj_sampling:n40"}
        result = subprocess.run(["bash", str(repo / "evaluation/trajectory_pol_porto/slurm/aggregate_compare.sbatch")],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        data = json.loads((self.root / "job_compare" / "same_query" / "same_query.json").read_text())
        self.assertIn("traj_sampling:n40|exact_mbr:boxes", data["workloads"]["pol_v3_multi"]["pairs"])

    def test_runs_of_different_benchmarks_are_refused(self) -> None:
        rows = [json.loads(line) for line in (self.root / "benchmark.jsonl").read_text().splitlines()]
        rows[0], rows[1] = rows[1], rows[0]  # same queries, another file (and query ids)
        other = self.root / "other"
        other.mkdir()
        write_benchmark(other / "benchmark.jsonl", rows)
        (other / "snapshot").symlink_to(self.root / "snapshot")
        config = load_experiment_config(_config(other, "exact_mbr", "traj_exact_mbr", "boxes"))
        aggregate_runs([run_seed(config, 0, run_id="sq")], other / "agg")
        with self.assertRaisesRegex(ValueError, "different benchmarks"):
            same_query.compare([self.aggregates["traj_sampling:n40"], other / "agg" / "comparison.json"])


if __name__ == "__main__":
    unittest.main()
