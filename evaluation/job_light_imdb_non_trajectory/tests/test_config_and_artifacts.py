from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.job_light_imdb_non_trajectory.joblight_eval.artifacts import (
    create_run_directory,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import (
    load_experiment_config,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.report import (
    aggregate_runs,
    compare_aggregates,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters.base import (
    AdapterEvaluation,
    StageMetrics,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters.external import (
    ExternalCommandAdapter,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
    LatencyRecord,
    PredictionRecord,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.runner import (
    REQUIRED_ARTIFACTS,
    run_seed,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.resources import run_logged_command


class ConfigAndArtifactsTest(unittest.TestCase):
    def test_inheritance_naming_and_collision_prevention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.txt"
            query_path.write_text("title t##t.id,=,1#1\n", encoding="utf-8")
            base = root / "base.yaml"
            base.write_text(
                _config_text(query_path, root / "results", "base", "Base"),
                encoding="utf-8",
            )
            child = root / "child.yaml"
            child.write_text(
                "extends: base.yaml\nexperiment:\n  variant_id: changed\n"
                "  display_name: Changed name\n",
                encoding="utf-8",
            )
            config = load_experiment_config(child)
            self.assertEqual(config.variant_id, "changed")
            self.assertEqual(config.display_name, "Changed name")
            run = create_run_directory(config, 0, "fixed")
            self.assertTrue(run.is_dir())
            with self.assertRaises(FileExistsError):
                create_run_directory(config, 0, "fixed")

    def test_aggregate_refuses_partial_and_mismatched_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write_complete_run(root / "one", seed=0, config_hash="abc")
            second = _write_complete_run(root / "two", seed=1, config_hash="abc")
            aggregate = aggregate_runs([first, second], root / "aggregate")
            metric = aggregate["workloads"]["job_light"]["metrics"]["raw_q_error.p50"]
            self.assertEqual(metric, {"mean": 2.0, "std": 0.0})
            comparison = compare_aggregates(
                [root / "aggregate" / "comparison.json"], root / "comparison"
            )
            self.assertEqual(comparison["aggregate_count"], 1)
            aggregate_markdown = (root / "aggregate" / "comparison.md").read_text()
            comparison_markdown = (root / "comparison" / "comparison.md").read_text()
            for label in (
                "Raw, all scored queries",
                "Raw, true cardinality > 0",
                "Smoothed, true cardinality = 0",
                "Smoothed, all scored queries",
            ):
                self.assertIn(label, aggregate_markdown)
                self.assertIn(label, comparison_markdown)
            partial = root / "partial"
            partial.mkdir()
            with self.assertRaises(ValueError):
                aggregate_runs([first, partial], root / "bad")
            mismatch = _write_complete_run(root / "mismatch", seed=2, config_hash="def")
            with self.assertRaises(ValueError):
                aggregate_runs([first, mismatch], root / "bad2")

    def test_complete_run_writes_contract_and_preserves_raw_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.txt"
            query_path.write_text("title t##t.id,=,1#10\n", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                _config_text(query_path, root / "results", "v", "Fixture"),
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            adapter = _FixtureAdapter(config, 0, root)
            with patch(
                "evaluation.job_light_imdb_non_trajectory.joblight_eval.runner.create_adapter",
                return_value=adapter,
            ):
                run_directory = run_seed(config, 0, run_id="fixed")
            for name in REQUIRED_ARTIFACTS:
                self.assertTrue((run_directory / name).exists(), name)
            prediction = (run_directory / "predictions.csv").read_text(encoding="utf-8")
            self.assertIn(",0.25,", prediction)
            summary = json.loads((run_directory / "summary.json").read_text())
            accuracy = summary["workloads"]["job_light"]["accuracy"]
            self.assertEqual(accuracy["estimate_lt_1_count"], 1)
            self.assertEqual(accuracy["smoothed_q_error"]["max"], 10.0)

    def test_reports_primary_and_supplementary_timing_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                _write_complete_run(root / f"seed-{seed}", seed, "abc")
                for seed in (0, 1)
            ]
            for run in runs:
                path = run / "summary.json"
                summary = json.loads(path.read_text(encoding="utf-8"))
                workload = summary["workloads"]["job_light"]
                workload["timing_protocol"] = {
                    "device": "cuda",
                    "primary_device": "cuda",
                    "synchronize_cuda": True,
                    "published_reference_hardware": "Reference GPU",
                }
                workload["inference_by_device"] = {
                    "cuda": {
                        **workload["inference"],
                        "device": "cuda",
                        "device_names": ["Test GPU"],
                        "scopes": ["inference"],
                    },
                    "cpu": {
                        **workload["inference"],
                        "mean_ms": 5.0,
                        "device": "cpu",
                        "device_names": ["Test CPU"],
                        "scopes": ["inference"],
                    },
                }
                path.write_text(json.dumps(summary), encoding="utf-8")
            aggregate = aggregate_runs(runs, root / "aggregate")
            profiles = aggregate["workloads"]["job_light"]["inference_profiles"]
            self.assertEqual(set(profiles), {"cpu", "cuda"})
            self.assertEqual(profiles["cuda"]["device_names"], ["Test GPU"])
            markdown = (root / "aggregate" / "comparison.md").read_text()
            self.assertIn("Primary timing device: **cuda**", markdown)
            self.assertIn("Test GPU", markdown)
            self.assertIn("Test CPU", markdown)

    def test_external_long_build_requires_smoke_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.txt"
            query_path.write_text("title t##t.id,=,1#1\n", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                _config_text(query_path, root / "results", "v", "Fixture")
                + "  require_smoke_before_build: true\n"
                + "  build_command: true\n",
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            run_directory = create_run_directory(config, 0, "gate")
            (run_directory / "logs").mkdir(exist_ok=True)
            adapter = ExternalCommandAdapter(config, 0, run_directory)
            with self.assertRaises(ValueError):
                adapter.build()

    def test_subprocess_environment_can_remove_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = os.environ.get("PYTHONPATH")
            os.environ["PYTHONPATH"] = "incompatible-package-layer"
            try:
                result = run_logged_command(
                    ["/usr/bin/env"],
                    cwd=root,
                    env={"PYTHONPATH": None},
                    stdout_path=root / "stdout",
                    stderr_path=root / "stderr",
                )
            finally:
                if previous is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous
            self.assertEqual(result.returncode, 0)
            self.assertIsNotNone(result.peak_rss_bytes)
            self.assertGreater(result.peak_rss_bytes, 0)
            self.assertNotIn("PYTHONPATH=", (root / "stdout").read_text(encoding="utf-8"))

    def test_rss_measurement_is_per_command_not_a_cumulative_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            larger = run_logged_command(
                [sys.executable, "-c", "x = bytearray(24_000_000)"],
                cwd=root,
                env={},
                stdout_path=root / "stdout",
                stderr_path=root / "stderr",
            )
            smaller = run_logged_command(
                [sys.executable, "-c", "x = bytearray(2_000_000)"],
                cwd=root,
                env={},
                stdout_path=root / "stdout",
                stderr_path=root / "stderr",
            )
            self.assertGreater(larger.peak_rss_bytes, smaller.peak_rss_bytes)
            self.assertGreater(smaller.peak_rss_bytes, 0)

    def test_dotted_minor_version_remains_a_string(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.txt"
            query_path.write_text("title t##t.id,=,1#1\n", encoding="utf-8")
            config_path = root / "config.yaml"
            text = _config_text(query_path, root / "results", "v", "Fixture")
            text = text.replace("revision: abcdef", 'installed_version: "16.10"')
            text = text.replace(
                "  type: external\n",
                '  type: external\n  postgres_version: "16.10"\n',
            )
            config_path.write_text(text, encoding="utf-8")
            config = load_experiment_config(config_path)
            self.assertEqual(config.source["installed_version"], "16.10")
            self.assertEqual(config.adapter["postgres_version"], "16.10")

    def test_cuda_timing_requires_explicit_synchronization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            query_path = root / "queries.txt"
            query_path.write_text("title t##t.id,=,1#1\n", encoding="utf-8")
            config_path = root / "config.yaml"
            text = _config_text(query_path, root / "results", "v", "Fixture")
            text = text.replace(
                "timing:\n  warmup_passes:",
                "timing:\n  device: cuda\n  warmup_passes:",
            )
            config_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "synchronize_cuda"):
                load_experiment_config(config_path)
            config_path.write_text(
                text.replace("  device: cuda\n", "  device: cuda\n  synchronize_cuda: true\n"),
                encoding="utf-8",
            )
            self.assertEqual(load_experiment_config(config_path).timing["device"], "cuda")


def _config_text(query_path: Path, result_path: Path, variant: str, display: str) -> str:
    return f"""schema_version: 1
experiment:
  experiment_id: fixture
  method_id: external
  variant_id: {variant}
  display_name: {display}
  seeds: [0, 1, 2]
source:
  url: https://example.invalid/repository.git
  revision: abcdef
paths:
  results_root: {result_path}
workloads:
  job_light:
    queries_csv: {query_path}
timing:
  warmup_passes: 1
  repetitions: 10
resources:
  profile: test
artifacts:
  checkpoint: none
adapter:
  type: external
"""


def _write_complete_run(path: Path, seed: int, config_hash: str) -> Path:
    path.mkdir()
    (path / "run_manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    accuracy = {
        "query_count": 1,
        "scored_query_count": 1,
        "true_zero_matching_count": 0,
        "coverage_fraction": 1.0,
        "estimate_lt_1_count": 0,
        "estimate_lt_0_1_count": 0,
        "estimate_lt_0_01_count": 0,
        "zero_estimate_count": 0,
        "raw_q_error": {"p50": 2.0, "p90": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
        "raw_q_error_true_positive": {"p50": 2.0, "p90": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
        "smoothed_q_error_true_zero": {"p50": None, "p90": None, "p95": None, "p99": None, "max": None},
        "smoothed_q_error": {"p50": 2.0, "p90": 2.0, "p95": 2.0, "p99": 2.0, "max": 2.0},
    }
    inference = {
        "mean_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "throughput_queries_per_second": 1000.0,
    }
    payload = {
        "experiment_id": "fixture",
        "method_id": "external",
        "variant_id": "v",
        "display_name": "Fixture",
        "config_hash": config_hash,
        "seed": seed,
        "workloads": {"job_light": {"accuracy": accuracy, "inference": inference}},
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


class _FixtureAdapter:
    def __init__(self, config, seed, run_directory):
        self.config = config
        self.seed = seed
        self.run_directory = run_directory

    def prepare(self):
        return StageMetrics(wall_seconds=1.0, peak_rss_bytes=10)

    def build(self):
        return StageMetrics(wall_seconds=2.0, peak_rss_bytes=20)

    def smoke(self, workloads, query_limit=2):
        return StageMetrics(wall_seconds=0.5, detail={"query_limit": query_limit})

    def evaluate(self, workloads):
        return AdapterEvaluation(
            predictions=(PredictionRecord("job_light", 0, "ok", 10, 0.25),),
            latencies=(LatencyRecord("job_light", 0, 0, 2.0),),
        )

    def artifact_metadata(self):
        return {"parameter_count": 5, "serialized_model_mb": 0.001}


if __name__ == "__main__":
    unittest.main()
