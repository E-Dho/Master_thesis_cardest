"""Profile-controlled timing of trajectory estimators: time stage, plan, launcher."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from evaluation.eval_core import timing_guard as guard
from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.report import aggregate_runs, load_timing_summaries
from evaluation.eval_core.runner import run_seed
from evaluation.eval_core.timing import run_timing_stage
from evaluation.trajectory_pol_porto.traj_eval.timing_plan import main as plan_main, plan_entries

from ._fixture import write_benchmark, write_synthetic_snapshot
from .test_p3_adapters import _config

REPO = Path(__file__).resolve().parents[3]
LAUNCH_SPEC = importlib.util.spec_from_file_location(
    "traj_timing_launch", REPO / "evaluation" / "job_light_imdb_non_trajectory" / "scripts" / "timing_launch.py")
LAUNCH = importlib.util.module_from_spec(LAUNCH_SPEC)
assert LAUNCH_SPEC.loader is not None
LAUNCH_SPEC.loader.exec_module(LAUNCH)
TIMING_KEYS = "  repetitions: 2\n  timing_warmup_passes: 1\n  timing_repetitions: 3\n"


def _timed_config(root: Path, name: str, adapter_type: str, seeds: str = "[0]", **adapter) -> Path:
    path = _config(root, name, adapter_type, "timed", seeds=seeds, **adapter)
    path.write_text(path.read_text().replace("  repetitions: 2\n", TIMING_KEYS), encoding="utf-8")
    return path


@unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs Linux CPU affinity")
class InProcessTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        self.original_affinity = os.sched_getaffinity(0)
        first = min(self.original_affinity)
        os.sched_setaffinity(0, set(guard.thread_siblings(first)) & self.original_affinity)
        spec = self.root / "timing_hardware.json"
        spec.write_text(json.dumps({"schema_version": 1, "profiles": {
            "cpu_1core": {"cpu_model": guard.cpu_model()["model_name"]}}}))
        # what the launcher exports to the measuring process (cuda hidden, thread limits)
        self.environment = mock.patch.dict(os.environ, {
            **guard.profile_environment("cpu_1core"),
            guard.ALLOW_MISSING_THREADPOOLCTL_ENV: "1", guard.HARDWARE_SPEC_ENV: str(spec)})
        self.environment.start()
        # Earlier tests of the suite may leave helper processes (multiprocessing's
        # resource tracker); they predate the measured code, so ignore exactly those.
        existing = {child["pid"] for child in guard.child_processes()}
        original_children = guard.child_processes
        self.children = mock.patch.object(
            guard, "child_processes",
            lambda *a, **k: [c for c in original_children(*a, **k) if c["pid"] not in existing])
        self.children.start()

    def tearDown(self) -> None:
        self.children.stop()
        self.environment.stop()
        os.sched_setaffinity(0, self.original_affinity)

    def test_time_stage_uses_its_own_passes_and_aggregates(self) -> None:
        path = _timed_config(self.root, "sampling", "traj_sampling", seeds="[0, 1]", sample_trips=60)
        config = load_experiment_config(path)
        runs = [run_seed(config, seed, run_id=f"acc{seed}") for seed in config.seeds]
        accuracy_rows = list(csv.DictReader((runs[0] / "latency.csv").open()))
        directories = [run_timing_stage(config, seed, run, "cpu_1core") for seed, run in zip(config.seeds, runs)]
        summary = json.loads((directories[0] / "timing_summary.json").read_text())
        self.assertTrue(summary["reportable"], summary.get("failures"))
        self.assertTrue(summary["estimate_consistency"]["passed"])
        timed = list(csv.DictReader((directories[0] / "latency.csv").open()))
        answered = {(row["workload"], row["query_id"]) for row in accuracy_rows}
        # accuracy: 2 repetitions; timing: 3 (after 1 warm-up pass); answered queries only
        self.assertEqual(len(accuracy_rows), 2 * len(answered))
        self.assertEqual(len(timed), 3 * len(answered))
        self.assertEqual({row["profile"] for row in timed}, {"cpu_1core"})
        self.assertTrue(summary["workloads"]["pol_v3_multi"]["guard"]["passed"])
        self.assertEqual(list(load_timing_summaries(runs[0])), ["cpu_1core"])
        aggregate = aggregate_runs(runs, self.root / "aggregate")
        timing = aggregate["workloads"]["pol_v3_multi"]["standardized_timing"]["cpu_1core"]
        self.assertTrue(timing["available"])
        self.assertIn("extension", aggregate["workloads"]["pol_v3_multi"])

    def test_plan_lists_the_latest_complete_run_of_every_seed(self) -> None:
        path = _timed_config(self.root, "exact", "traj_exact_mbr")
        config = load_experiment_config(path)
        first = run_seed(config, 0, run_id="a")
        latest = run_seed(config, 0, run_id="b")
        entries = plan_entries([path], "cpu_1core")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1], 0)
        self.assertEqual(entries[0][2], max(first, latest))
        with self.assertRaisesRegex(ValueError, "does not declare timing profile"):
            plan_entries([path], "cpu_postgres_2core")
        output = self.root / "plans" / "cpu_1core.txt"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(plan_main(["--profile", "cpu_1core", "--output", str(output), "--config", str(path)]), 0)
        parsed = LAUNCH._read_plan(output)
        self.assertEqual([(e["config"], e["seed"], e["run_directory"]) for e in parsed],
                         [(str(entries[0][0]), "0", str(entries[0][2]))])

    def test_launcher_runs_the_trajectory_cli(self) -> None:
        path = _timed_config(self.root, "exact_launch", "traj_exact_mbr")
        config = load_experiment_config(path)
        run = run_seed(config, 0, run_id="launch")
        plan = self.root / "launch_plan.txt"
        plan.write_text(f"{path} 0 {run}\n")
        out = io.StringIO()
        with redirect_stdout(out):
            code = LAUNCH.main(["--profile", "cpu_1core", "--plan", str(plan), "--dry-run",
                                "--cli-module", "evaluation.trajectory_pol_porto.traj_eval.cli",
                                "--report", str(self.root / "launch.json")])
        self.assertEqual(code, 0)
        self.assertIn("-m evaluation.trajectory_pol_porto.traj_eval.cli time --config", out.getvalue())
        report = json.loads((self.root / "launch.json").read_text())
        self.assertEqual(report["cli_module"], "evaluation.trajectory_pol_porto.traj_eval.cli")
        # the default stays the JOB-light CLI
        with redirect_stdout(io.StringIO()) as default:
            LAUNCH.main(["--profile", "cpu_1core", "--plan", str(plan), "--dry-run",
                         "--report", str(self.root / "launch_default.json")])
        self.assertIn("joblight_eval.cli time", default.getvalue())


class ShippedTimingConfigTest(unittest.TestCase):
    def test_shipped_configs_declare_their_profile_and_timing_passes(self) -> None:
        from evaluation.eval_core.config import load_simple_yaml

        configs = Path(__file__).resolve().parents[1] / "configs" / "pol_v3"
        base = load_simple_yaml(configs / "_base.yaml")["timing"]
        self.assertEqual((base["timing_warmup_passes"], base["timing_repetitions"]), (1, 3))
        self.assertEqual((base["warmup_passes"], base["repetitions"]), (0, 1))
        for path in configs.glob("*.yaml"):
            if path.name.startswith("_"):
                continue
            timing = load_simple_yaml(path).get("timing", {})
            expected = "cpu_postgres_2core" if path.name.startswith("mobilitydb_") else None
            if expected:
                self.assertEqual(timing["profiles"], [expected], path.name)
            self.assertNotIn("timing_repetitions", timing, f"{path.name} must inherit the timing passes")
        spec = json.loads((configs.parent / "timing_hardware.json").read_text())
        self.assertEqual({spec["profiles"][p]["cpu_model"] for p in ("cpu_1core", "cpu_postgres_2core")},
                         {"allocated"})


if __name__ == "__main__":
    unittest.main()
