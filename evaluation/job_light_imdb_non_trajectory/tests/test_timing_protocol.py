from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evaluation.job_light_imdb_non_trajectory.joblight_eval import timing_guard as guard
from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import load_experiment_config
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import PredictionRecord
from evaluation.job_light_imdb_non_trajectory.joblight_eval.report import aggregate_runs, compare_aggregates
from evaluation.job_light_imdb_non_trajectory.joblight_eval.timing import (
    TimingProtocolError,
    config_drift,
    estimate_consistency,
    load_timing_summaries,
    run_timing_stage,
    strip_paths,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LAUNCH_SPEC = importlib.util.spec_from_file_location("timing_launch", ROOT / "scripts" / "timing_launch.py")
LAUNCH = importlib.util.module_from_spec(LAUNCH_SPEC)
assert LAUNCH_SPEC.loader is not None
LAUNCH_SPEC.loader.exec_module(LAUNCH)


def _fake_sysfs(root: Path, siblings: dict[int, tuple[int, str]]) -> Path:
    cpu_root = root / "cpu"
    for cpu, (core, sibling_list) in siblings.items():
        topology = cpu_root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text("0\n")
        (topology / "core_id").write_text(f"{core}\n")
        (topology / "thread_siblings_list").write_text(sibling_list + "\n")
    (cpu_root / "online").write_text(f"0-{max(siblings)}\n")
    return cpu_root


class TopologyTests(unittest.TestCase):
    def test_cpu_list_round_trip(self):
        self.assertEqual(guard.parse_cpu_list("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(guard.format_cpu_list([11, 10, 8, 3, 2, 1, 0]), "0-3,8,10-11")

    def test_smt_siblings_group_into_physical_cores(self):
        with tempfile.TemporaryDirectory() as temporary:
            sysfs = _fake_sysfs(Path(temporary), {0: (0, "0,2"), 1: (1, "1,3"), 2: (0, "0,2"), 3: (1, "1,3")})
            self.assertEqual(guard.split_physical_cores([0, 1, 2, 3], sysfs), [[0, 2], [1, 3]])
            self.assertEqual(guard.launch_violations("cpu_1core", [0, 2], sysfs), [])
            self.assertTrue(guard.launch_violations("cpu_1core", [0, 1], sysfs))
            # the client of the PostgreSQL profile is pinned to one core
            self.assertEqual(guard.launch_violations("cpu_postgres_2core", [1, 3], sysfs), [])
            self.assertTrue(guard.launch_violations("cpu_postgres_2core", [0, 1], sysfs))
            environment = guard.profile_environment("cpu_fullnode_exclusive", [0, 1, 2, 3], sysfs)
            self.assertEqual(environment["OMP_NUM_THREADS"], "2")

    def test_job_level_cgroup_is_preferred(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc"
            (proc / "self").mkdir(parents=True)
            (proc / "self" / "cgroup").write_text("0::/system.slice/job_42/step_0/task_0\n")
            cgroup = root / "cgroup"
            for relative, cpus in (("system.slice/job_42", "0-1"), ("system.slice/job_42/step_0/task_0", "0")):
                (cgroup / relative).mkdir(parents=True, exist_ok=True)
                (cgroup / relative / "cpuset.cpus.effective").write_text(cpus + "\n")
            (cgroup / "system.slice/job_42/cpu.max").write_text("max 100000\n")
            result = guard.cgroup_cpuset(proc, cgroup)
            self.assertEqual(result["cpus"], [0, 1])
            self.assertEqual(result["level"], "slurm_job")

    def test_child_processes_follow_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            for pid, ppid in ((10, 1), (11, 10), (12, 11), (13, 1)):
                (proc / str(pid)).mkdir()
                (proc / str(pid) / "stat").write_text(f"{pid} (worker {pid}) S {ppid} 0 0\n")
            children = guard.child_processes(10, proc)
            self.assertEqual([item["pid"] for item in children], [11, 12])

    def test_profile_environment_hides_cuda_for_cpu_profiles(self):
        cpu = guard.profile_environment("cpu_1core", [0])
        gpu = guard.profile_environment("gpu_single_query", [0, 1, 2, 3])
        self.assertEqual({cpu[name] for name in guard.THREAD_LIMIT_VARIABLES}, {"1"})
        self.assertEqual(cpu["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", gpu)
        self.assertEqual(gpu[guard.PROFILE_ENV], "gpu_single_query")


class _FakeTorch:
    def __init__(self, threads=1, interop=1):
        self.threads, self.interop = threads, interop
        self.cuda = SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False,
                                    device_count=lambda: 0)
        self.version = SimpleNamespace(cuda=None)

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, value):
        self.threads = value

    def get_num_interop_threads(self):
        return self.interop

    def set_num_interop_threads(self, value):
        self.interop = value


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, guard.profile_environment("cpu_1core", [0]))
        self.environment.start()
        self.patches = [
            mock.patch.object(guard, "allowed_cpus", return_value=[0]),
            mock.patch.object(guard, "physical_cores", return_value={(0, 0): [0]}),
            mock.patch.object(guard, "thread_siblings", return_value=[0]),
            mock.patch.object(guard, "cgroup_cpuset", return_value={"available": True, "cpus": [0], "path": "/x"}),
            mock.patch.object(guard, "child_processes", return_value=[]),
            mock.patch.object(guard, "thread_names", return_value={"python": 1}),
            mock.patch.object(guard, "threadpool_snapshot",
                              return_value={"available": True, "pools": [{"internal_api": "openblas", "num_threads": 1}]}),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        self.environment.stop()

    def test_compliant_process_passes_and_records(self):
        session = guard.TimingSession("cpu_1core")
        session.configure_torch(_FakeTorch(threads=4, interop=4))
        self.assertTrue(session.verify("pre_timing")["passed"])
        with session.measure():
            sum(range(1000))
        report = session.report()
        self.assertTrue(report["passed"])
        self.assertIn("cpu_to_wall_ratio", report["measurements"][0])

    def test_oversized_blas_pool_is_a_hard_failure(self):
        with mock.patch.object(guard, "threadpool_snapshot",
                               return_value={"available": True, "pools": [{"internal_api": "mkl", "num_threads": 8}]}):
            with self.assertRaisesRegex(guard.TimingGuardError, "blas_thread_pools"):
                guard.TimingSession("cpu_1core").verify("pre_timing")

    def test_second_core_and_child_processes_are_hard_failures(self):
        with mock.patch.object(guard, "physical_cores", return_value={(0, 0): [0], (0, 1): [1]}):
            with self.assertRaisesRegex(guard.TimingGuardError, "affinity_physical_cores"):
                guard.TimingSession("cpu_1core").verify("pre_timing")
        with mock.patch.object(guard, "child_processes", return_value=[{"pid": 5, "ppid": 1, "name": "worker"}]):
            with self.assertRaisesRegex(guard.TimingGuardError, "no_child_processes"):
                guard.TimingSession("cpu_1core").verify("pre_timing")

    def test_foreign_smt_sibling_is_a_hard_failure(self):
        with mock.patch.object(guard, "thread_siblings", return_value=[0, 64]):
            with self.assertRaisesRegex(guard.TimingGuardError, "smt_siblings_reserved"):
                guard.TimingSession("cpu_1core").verify("pre_timing")

    def test_raw_thread_count_only_warns(self):
        with mock.patch.object(guard, "thread_names", return_value={"python": 1, "cuda-EvtHandlr": 3}):
            session = guard.TimingSession("cpu_1core")
            self.assertTrue(session.verify("pre_timing")["passed"])
            self.assertTrue(any("os_thread_count" in warning for warning in session.warnings))

    def test_torch_threads_and_cuda_are_checked(self):
        session = guard.TimingSession("cpu_1core")
        session._torch = _FakeTorch(threads=4)
        with self.assertRaisesRegex(guard.TimingGuardError, "torch_intra_op_threads"):
            session.verify("pre_timing")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(guard.TimingGuardError, "cuda_hidden"):
                guard.TimingSession("cpu_1core").verify("pre_timing")

    def test_missing_threadpoolctl_needs_explicit_allowance(self):
        with mock.patch.object(guard, "threadpool_snapshot", return_value={"available": False, "pools": []}):
            with self.assertRaisesRegex(guard.TimingGuardError, "threadpoolctl_available"):
                guard.TimingSession("cpu_1core").verify("pre_timing")
            with mock.patch.dict(os.environ, {guard.ALLOW_MISSING_THREADPOOLCTL_ENV: "1"}):
                self.assertTrue(guard.TimingSession("cpu_1core").verify("pre_timing")["passed"])


class TimingStageUnitTests(unittest.TestCase):
    def test_timing_only_keys_are_ignored_by_drift(self):
        before = {"timing": {"device": "cuda"}, "adapter": {"evaluate_command": "a", "timing_commands": {"x": 1}},
                  "experiment": {"display_name": "A", "variant_id": "v"}}
        after = {"timing": {"device": "cpu"}, "adapter": {"evaluate_command": "b"},
                 "experiment": {"display_name": "B", "variant_id": "v"}}
        self.assertEqual(config_drift(strip_paths(before), strip_paths(after)), ["adapter.evaluate_command"])

    def test_estimate_consistency_modes(self):
        accuracy = [PredictionRecord("w", 0, "ok", 10, 100.0), PredictionRecord("w", 1, "ok", 10, 3.0)]
        timing = [PredictionRecord("w", 0, "ok", 10, 100.0), PredictionRecord("w", 1, "ok", 10, 4.0)]
        strict = estimate_consistency(accuracy, timing, {"mode": "deterministic", "relative_tolerance": 1e-9,
                                                         "absolute_tolerance": 0.0})
        self.assertFalse(strict["passed"])
        self.assertEqual(strict["violation_count"], 1)
        rounded = estimate_consistency(accuracy, timing, {"mode": "deterministic", "relative_tolerance": 1e-4,
                                                          "absolute_tolerance": 1.0})
        self.assertTrue(rounded["passed"])
        stochastic = estimate_consistency(accuracy, timing, {"mode": "stochastic", "relative_tolerance": 0.0,
                                                             "absolute_tolerance": 0.0})
        self.assertTrue(stochastic["passed"])
        self.assertAlmostEqual(stochastic["max_relative_difference"], 0.25)

    def test_launch_plan_and_core_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "plan.tsv"
            plan.write_text("# comment\nconf.yaml 0 /runs/a --allow-estimate-drift\n\n")
            entries = LAUNCH._read_plan(plan)
            self.assertEqual(entries[0]["extra"], ["--allow-estimate-drift"])
        profile = guard.get_profile("cpu_postgres_2core")
        client, server = LAUNCH._assign_cores(profile, [0, 1, 2, 3], [[0, 2], [1, 3]], False)
        self.assertEqual((client, server), ([0, 2], [1, 3]))
        with self.assertRaises(SystemExit):
            LAUNCH._assign_cores(guard.get_profile("cpu_1core"), [0, 1], [[0], [1]], False)


FIXTURE_TIMER = textwrap.dedent('''
    import csv, sys, time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    import _timing
    predictions, latency, factor = Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
    session = _timing.session_from_environment("fixture", device="cpu")
    session.configure()
    session.verify("pre_timing")
    rows = []
    with _timing.measured(session):
        for repetition in range(2):
            for query_id in (0, 1):
                started = time.perf_counter()
                estimate = (query_id + 1) * 10.0 * factor
                rows.append([query_id, repetition, (time.perf_counter() - started) * 1000, "fixture", "cpu", "CPU", 0.001])
    session.verify("post_timing")
    session.write()
    with predictions.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "estimated_cardinality", "status", "diagnostic"])
        writer.writerows([[0, 10.0 * factor, "ok", ""], [1, 20.0 * factor, "ok", ""]])
    with latency.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "repetition", "latency_ms", "scope", "device", "device_name", "model_core_ms"])
        writer.writerows(rows)
''')


@unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs Linux CPU affinity")
class TimingStageIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original_affinity = os.sched_getaffinity(0)
        first = min(self.original_affinity)
        # One whole physical core: the first CPU and its SMT siblings.
        self.core = set(guard.thread_siblings(first)) & self.original_affinity
        os.sched_setaffinity(0, self.core)
        self.environment = mock.patch.dict(os.environ, {guard.ALLOW_MISSING_THREADPOOLCTL_ENV: "1"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        os.sched_setaffinity(0, self.original_affinity)
        self.temporary.cleanup()

    def _config(self, factor: float = 1.0, variant: str = "v") -> Path:
        queries = self.root / "queries.csv"
        queries.write_text("title t##t.id,=,1#10\ntitle t##t.id,=,2#20\n")
        script = self.root / "fixture_timer.py"
        script.write_text(FIXTURE_TIMER)
        command = (f"{sys.executable} {script} {ROOT / 'scripts'} {{predictions_csv}} "
                   f"{{latency_csv}} {factor}")
        path = self.root / f"config_{variant}.yaml"
        path.write_text(
            "schema_version: 1\nexperiment:\n  experiment_id: e\n  method_id: fixture\n"
            f"  variant_id: {variant}\n  display_name: Fixture\n  seeds: [0, 1]\n"
            "source:\n  url: u\n  revision: r\n"
            f"paths:\n  results_root: {self.root / 'results'}\n"
            f"workloads:\n  job_light:\n    queries_csv: {queries}\n"
            "timing:\n  device: cpu\n  warmup_passes: 0\n  repetitions: 2\n"
            "  primary_profile: cpu_1core\n  profiles: [cpu_1core]\n"
            "  estimate_consistency:\n    mode: deterministic\n    relative_tolerance: 1.0e-9\n"
            "    absolute_tolerance: 0.0\n"
            "resources:\n  profile: test\nartifacts:\n  checkpoint: none\n"
            f"adapter:\n  type: external\n  working_directory: {REPO}\n"
            f"  evaluate_command: echo unused\n  timing_commands:\n    cpu_1core: {command}\n"
        )
        return path

    def _accuracy_run(self, config_path: Path, seed: int) -> Path:
        config = load_experiment_config(config_path)
        run = self.root / "results" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "resolved_config.json").write_text(json.dumps(config.raw))
        (run / "run_manifest.json").write_text(json.dumps({
            "status": "complete", "experiment_id": "e", "method_id": "fixture", "variant_id": "v",
            "seed": seed, "config_hash": config.config_hash}))
        (run / "predictions.csv").write_text(
            "workload,query_id,status,true_cardinality,estimated_cardinality,raw_q_error,smoothed_q_error,diagnostic\n"
            "job_light,0,ok,10,10.0,1.0,1.0,\njob_light,1,ok,20,20.0,1.0,1.0,\n")
        stats = {"p50": 1.0, "p90": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
        inference = {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0,
                     "throughput_queries_per_second": 1.0}
        (run / "summary.json").write_text(json.dumps({
            "experiment_id": "e", "method_id": "fixture", "variant_id": "v", "display_name": "Fixture",
            "config_hash": config.config_hash, "seed": seed,
            "workloads": {"job_light": {"accuracy": {
                "query_count": 2, "scored_query_count": 2, "coverage_fraction": 1.0,
                "true_zero_matching_count": 0, "estimate_lt_1_count": 0, "estimate_lt_0_1_count": 0,
                "estimate_lt_0_01_count": 0, "zero_estimate_count": 0, "raw_q_error": stats,
                "raw_q_error_true_positive": stats, "smoothed_q_error_true_zero": stats,
                "smoothed_q_error": stats}, "inference": inference}}}))
        return run

    def test_timing_run_is_linked_verified_and_aggregated(self):
        config_path = self._config()
        runs = [self._accuracy_run(config_path, seed) for seed in (0, 1)]
        config = load_experiment_config(config_path)
        directories = [run_timing_stage(config, seed, run, "cpu_1core") for seed, run in zip((0, 1), runs)]
        summary = json.loads((directories[0] / "timing_summary.json").read_text())
        self.assertTrue(summary["reportable"])
        self.assertTrue(summary["estimate_consistency"]["passed"])
        workload = summary["workloads"]["job_light"]
        self.assertEqual(workload["inference"]["observation_count"], 4)
        self.assertIsNotNone(workload["model_core"])
        self.assertTrue(workload["guard"]["passed"])
        self.assertIn(",cpu_1core,", (directories[0] / "latency.csv").read_text())
        self.assertEqual(list(load_timing_summaries(runs[0])), ["cpu_1core"])
        aggregate = aggregate_runs(runs, self.root / "aggregate")
        timing = aggregate["workloads"]["job_light"]["standardized_timing"]["cpu_1core"]
        self.assertTrue(timing["available"])
        self.assertIn("Standardized latency", (self.root / "aggregate" / "comparison.md").read_text())
        compare_aggregates([self.root / "aggregate" / "comparison.json"], self.root / "compare")
        markdown = (self.root / "compare" / "comparison.md").read_text()
        self.assertIn("(`cpu_1core`)", markdown)
        self.assertIn("timing.cpu_1core.mean_ms.mean", (self.root / "compare" / "comparison.csv").read_text())

    def test_estimate_drift_and_config_drift_fail_explicitly(self):
        run = self._accuracy_run(self._config(), 0)
        drifted = load_experiment_config(self._config(factor=2.0, variant="v2").rename(self.root / "d.yaml"))
        with self.assertRaisesRegex(ValueError, "accuracy run identity"):
            run_timing_stage(drifted, 0, run, "cpu_1core")
        config_path = self._config(factor=2.0)
        config = load_experiment_config(config_path)
        with self.assertRaises(TimingProtocolError):
            run_timing_stage(config, 0, run, "cpu_1core")
        failed = next((run / "timing" / "cpu_1core").iterdir())
        self.assertEqual(json.loads((failed / "timing_manifest.json").read_text())["status"], "failed")
        self.assertEqual(load_timing_summaries(run), {})
        raw = json.loads((run / "resolved_config.json").read_text())
        raw["adapter"]["evaluate_command"] = "echo other"
        (run / "resolved_config.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "adapter.evaluate_command"):
            run_timing_stage(load_experiment_config(self._config()), 0, run, "cpu_1core")
        directory = run_timing_stage(
            load_experiment_config(self._config()), 0, run, "cpu_1core",
            allow_config_drift=["adapter.evaluate_command"],
        )
        self.assertEqual(json.loads((directory / "timing_summary.json").read_text())["config_drift"],
                         ["adapter.evaluate_command"])

    def test_unpinned_launch_is_refused(self):
        run = self._accuracy_run(self._config(), 0)
        os.sched_setaffinity(0, self.original_affinity)
        if len(guard.physical_cores(sorted(self.original_affinity))) < 2:
            self.skipTest("needs at least two physical cores")
        with self.assertRaises(TimingProtocolError):
            run_timing_stage(load_experiment_config(self._config()), 0, run, "cpu_1core")


class DeterminismTests(unittest.TestCase):
    def test_external_commands_fix_the_python_hash_seed_per_experiment_seed(self):
        from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters import external

        config = SimpleNamespace(
            seed=2, adapter={"environment": {}}, timing={}, method_id="m", variant_id="v",
            source_path=Path("/tmp/config.yaml"),
        )
        captured = {}

        def fake_run(rendered, *, cwd, env, stdout_path, stderr_path):
            captured.update(env)
            return SimpleNamespace(wall_seconds=0.0, peak_rss_bytes=0, command=("x",))

        adapter = external.ExternalCommandAdapter.__new__(external.ExternalCommandAdapter)
        adapter.config, adapter.seed = config, 2
        with tempfile.TemporaryDirectory() as temporary:
            adapter.run_directory = Path(temporary)
            with mock.patch.object(external, "run_logged_command", side_effect=fake_run):
                adapter._run_command("evaluate", "echo x")
                self.assertEqual(captured["PYTHONHASHSEED"], "2")
                config.adapter["environment"] = {"PYTHONHASHSEED": "7"}
                adapter._run_command("evaluate", "echo x")
                self.assertEqual(captured["PYTHONHASHSEED"], "7")


class ConfigProfileTests(unittest.TestCase):
    def _write(self, root: Path, timing: str, adapter: str = "") -> Path:
        queries = root / "q.csv"
        queries.write_text("title t##t.id,=,1#1\n")
        path = root / "c.yaml"
        path.write_text(
            "schema_version: 1\nexperiment:\n  experiment_id: e\n  method_id: m\n  variant_id: v\n"
            f"source:\n  url: u\n  revision: r\npaths:\n  results_root: {root}\n"
            f"workloads:\n  w:\n    queries_csv: {queries}\ntiming:\n  device: cpu\n{timing}"
            f"artifacts:\n  x: y\nadapter:\n  type: external\n{adapter}"
        )
        return path

    def test_profile_declarations_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            good = "  primary_profile: cpu_1core\n  profiles: [cpu_1core]\n"
            command = "  timing_commands:\n    cpu_1core: echo x\n"
            self.assertEqual(load_experiment_config(self._write(root, good, command)).timing["profiles"],
                             ["cpu_1core"])
            for timing, adapter, message in (
                (good, "", "timing_commands"),
                ("  primary_profile: cpu_1core\n  profiles: [cpu_2core]\n", command, "unknown timing profiles"),
                ("  primary_profile: gpu_single_query\n  profiles: [cpu_1core]\n", command, "primary_profile"),
                ("  primary_profile: cpu_fullnode_exclusive\n  profiles: [cpu_fullnode_exclusive]\n",
                 "  timing_commands:\n    cpu_fullnode_exclusive: echo x\n", "cannot be primary"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    load_experiment_config(self._write(root, timing, adapter))

    def test_every_shipped_config_declares_compliant_profiles(self):
        from model.src.config import load_simple_yaml
        from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import _deep_merge, _validate_timing_profiles

        checked = 0
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            raw = load_simple_yaml(path)
            if "extends" in raw:
                raw = _deep_merge(load_simple_yaml(path.parent / raw.pop("extends")), raw)
            if "timing" not in raw or "adapter" not in raw:
                continue
            _validate_timing_profiles(raw["timing"], raw["adapter"])
            if raw["timing"].get("profiles"):
                checked += 1
                for profile, command in (raw["adapter"].get("timing_commands") or {}).items():
                    device = guard.get_profile(profile).device
                    self.assertIn(f"--device {device}", command) if "--device" in command else None
                    self.assertNotRegex(command, r"--cpu-threads (?!1\b)\d+")
        self.assertGreaterEqual(checked, 9)


if __name__ == "__main__":
    unittest.main()
