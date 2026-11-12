"""Porto v3 (P9): the shipped configs, the grid extent of the histograms and every method on Porto data.

The Porto fixture is a real v3 benchmark (the generator's own Porto test data:
origin_stand NULL for most trips, GPS jumps 500 km outside the window domain,
no agents table). Methods that need a database (MobilityDB), the DeepDB
environment or the TraHist checkout run only when those are configured, as in
their POL tests.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_core.artifacts import read_predictions
from evaluation.eval_core.config import load_experiment_config, load_simple_yaml
from evaluation.eval_core.runner import run_seed
from evaluation.eval_core.workload_formats import load_queries
from evaluation.trajectory_pol_porto.traj_eval import deepdb_mbr
from evaluation.trajectory_pol_porto.traj_eval import own_model_translate as omt
from evaluation.trajectory_pol_porto.traj_eval.adapters.base import grid_extent
from evaluation.trajectory_pol_porto.traj_eval.adapters.case_plus import Grid, build_case

from query_generation.benchmark_v3.config import load_settings

from ._fixture import synthetic_settings
from ._porto_fixture import (
    MULTI, PORTO_CONFIG, SINGLE, porto_benchmark_rows, porto_settings, porto_synthetic_data,
    write_porto_benchmark, write_porto_snapshot, write_porto_training,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
P6 = ("trahist", "trahist_temporal", "case_plus", "case_plus_temporal", "case_plus_temporal_best")
OWN = ("own_model", "own_model_no_dedup")
# POL -> Porto: everything a Porto config may differ in besides these keys
SUBSTITUTIONS = (
    ("/mobilitydb/query_runs/qgen_pol_v3_exact_st_seed20261021/",
     "/mobilitydb_porto/query_runs/qgen_porto_v3_exact_st_seed20261102/"),
    ("/mobilitydb/query_runs/qgen_pol_v3_training_seed20261030/",
     "/mobilitydb_porto/query_runs/qgen_porto_v3_training_seed20261104/"),
    ("/mobilitydb/query_runs/qgen_pol_v3_validation_seed20261026/",
     "/mobilitydb_porto/query_runs/qgen_porto_v3_validation_seed20261103/"),
    ("traj_phase1_pol_v3", "traj_phase1_porto_v3"),
    ("pol_v3_multi", "porto_v3_multi"), ("pol_v3_single", "porto_v3_single"),
    ("pol_v3_config.json", "porto_v3_config.json"),
    ("traj_snapshots/pol_v3", "traj_snapshots/porto_v3"),
    ("p7_models/pol_v3/", "p7_models/porto_v3/"),
)
TRAHIST_SOURCE = os.environ.get("TRAJ_EVAL_TRAHIST_SOURCE")


def _substitute(value):
    if isinstance(value, str):
        for old, new in SUBSTITUTIONS:
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {_substitute(k): _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def _fixture_configs(root: Path, names) -> dict:
    """The shipped Porto configs with only the cluster paths and the query counts swapped."""
    target = root / "shipped"
    target.mkdir(exist_ok=True)
    out = {}
    for name in ("_base",) + tuple(names):
        text = (CONFIGS / "porto_v3" / f"{name}.yaml").read_text(encoding="utf-8")
        text = re.sub(r"/work_beegfs/\S+/benchmark\.jsonl", str(root / "benchmark.jsonl"), text)
        text = re.sub(r"/work_beegfs/\S+/traj_snapshots/porto_v3\b", str(root / "snapshot"), text)
        text = text.replace("../../../../query_generation/benchmark_v3/porto_v3_config.json", str(PORTO_CONFIG))
        text = text.replace("expected_query_count: 6000", f"expected_query_count: {8 * MULTI}")
        text = text.replace("expected_query_count: 2000", f"expected_query_count: {8 * SINGLE}")
        text = text.replace("results_root: ../../results", f"results_root: {root / 'results'}")
        text = re.sub(r"sample_trips: \d+", "sample_trips: 60", text)
        text = text.replace("/work_beegfs/sunip956/master_thesis_trajectories/external/trahist", str(TRAHIST_SOURCE))
        # MSCN: the published recipe, scaled down for the fixture
        text = re.sub(r"epochs: \d+", "epochs: 20", text)
        text = re.sub(r"batch_size: \d+", "batch_size: 64", text)
        text = re.sub(r"sample_bits: \d+", "sample_bits: 100", text)
        text = re.sub(r"hidden_size: \d+", "hidden_size: 32", text)
        # P6: a grid fine enough for the fixture's few kilometres
        text = re.sub(r"unit_size_m: \d+", "unit_size_m: 2000", text)
        text = re.sub(r"time_buckets: \d+", "time_buckets: 8", text)
        (target / f"{name}.yaml").write_text(text, encoding="utf-8")
        out[name] = target / f"{name}.yaml"
    return out


class ShippedPortoConfigTest(unittest.TestCase):
    """Every POL method has a Porto config that differs only where Porto must differ."""

    def test_same_methods_as_pol(self) -> None:
        self.assertEqual(sorted(p.name for p in (CONFIGS / "porto_v3").glob("*.yaml")),
                         sorted(p.name for p in (CONFIGS / "pol_v3").glob("*.yaml")))

    def test_no_drift_from_pol(self) -> None:
        for path in sorted((CONFIGS / "pol_v3").glob("*.yaml")):
            pol = _substitute(load_simple_yaml(path))
            porto = load_simple_yaml(CONFIGS / "porto_v3" / path.name)
            name = path.stem
            if name in P6:
                # tuned on Porto's own validation workload; the budget is the Porto own model's size
                for raw in (pol, porto):
                    for key in ("unit_size_m", "time_buckets", "budget_bytes"):
                        raw["adapter"].pop(key, None)
                self.assertEqual(porto["adapter"].pop("grid_extent"), "window_domain", name)
            if name in OWN:
                self.assertEqual(porto["experiment"]["seeds"], [0], name)  # one Porto model so far
                self.assertIn("model/porto-segments-first", porto["source"]["installed_version"])
                for raw in (pol, porto):
                    raw["experiment"].pop("seeds")
                    raw["source"].pop("installed_version")
            self.assertEqual(porto, pol, name)
        base = load_simple_yaml(CONFIGS / "porto_v3" / "_base.yaml")
        self.assertEqual(base["experiment"]["experiment_id"], "traj_phase1_porto_v3")
        self.assertEqual(sorted(base["workloads"]), ["porto_v3_multi", "porto_v3_single"])
        self.assertEqual(base["artifacts"]["snapshot_root"],
                         "/work_beegfs/sunip956/master_thesis_trajectories/traj_snapshots/porto_v3")

class GridExtentTest(unittest.TestCase):
    def test_window_domain_of_the_porto_config(self) -> None:
        settings = porto_settings()
        self.assertIsNone(grid_extent(settings, None))
        self.assertIsNone(grid_extent(settings, "data"))
        self.assertEqual(grid_extent(settings, "window_domain"),
                         (-52319.988053530826, 56518.77209420884, 29628.239768669046, 213325.0134066536))
        with self.assertRaises(ValueError):
            grid_extent(settings, "everything")
        with self.assertRaises(ValueError):  # POL declares no window domain
            grid_extent(synthetic_settings(), "window_domain")

    def test_case_grid_covers_the_window_domain_and_outliers_fall_into_the_border(self) -> None:
        data = porto_synthetic_data()
        segments = data["segments"]
        counts = np.bincount(np.searchsorted(np.unique(segments["trip_id"]), segments["trip_id"]))
        trips = {"offset": np.concatenate(([0], np.cumsum(counts)))}
        extent = grid_extent(porto_settings(), "window_domain")
        jumps = (segments["ex"] > extent[2]) | (segments["ex"] < extent[0])
        self.assertGreater(int(jumps.sum()), 0)  # the fixture has GPS jumps outside the domain
        arrays, meta = build_case(segments, trips, unit_size_m=2000.0, temporal=False, time_buckets=1, extent=extent)
        grid = Grid.covering(*extent, 2000.0)
        self.assertEqual(meta["grid_shape"], [grid.ny, grid.nx])
        _, data_meta = build_case(segments, trips, unit_size_m=2000.0, temporal=False, time_buckets=1)
        # the data extent reaches 500 km past the domain: a far larger grid
        self.assertGreater(data_meta["theoretical_bytes"], 10 * meta["theoretical_bytes"])


class _PortoRoot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_porto_benchmark(cls.root / "benchmark.jsonl")
        write_porto_snapshot(cls.root / "snapshot")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_shipped(self, name: str):
        paths = _fixture_configs(self.root, [name])
        config = load_experiment_config(paths[name])
        run = run_seed(config, config.seeds[0], run_id=name)
        predictions = {(p.workload, p.query_id): p for p in read_predictions(run / "predictions.csv")}
        records = {w.workload_id: list(load_queries(w)) for w in config.workloads}
        summary = json.loads((run / "summary.json").read_text())
        return run, predictions, records, summary


class PortoConfigLoadTest(_PortoRoot):
    def test_every_config_loads_on_the_porto_workloads(self) -> None:
        names = [p.stem for p in sorted((CONFIGS / "porto_v3").glob("[!_]*.yaml"))]
        for name, path in _fixture_configs(self.root, names).items():
            if name == "_base":
                continue
            config = load_experiment_config(path)
            self.assertEqual(config.experiment_id, "traj_phase1_porto_v3", name)
            self.assertEqual([w.workload_id for w in config.workloads], ["porto_v3_multi", "porto_v3_single"])
            shipped = (CONFIGS / "porto_v3" / f"{name}.yaml").read_text(encoding="utf-8")
            self.assertNotRegex(shipped.replace("trajectory_pol_porto", ""), r"(?m)^[^#]*\bpol_", name)


class PortoMethodsTest(_PortoRoot):
    """The shipped Porto configs run end to end on the Porto fixture."""

    def test_exact_mbr_never_below_truth(self) -> None:
        _, predictions, records, summary = self.run_shipped("exact_mbr")
        with_stand = 0
        for record in records["porto_v3_multi"]:
            prediction = predictions[("porto_v3_multi", record.query_id)]
            if prediction.status != "ok":
                continue
            # a box overlap is necessary for a match; NULL origin_stand never matches (else this could fail)
            self.assertGreaterEqual(prediction.estimated_cardinality, record.true_cardinality, record.query_id)
            with_stand += any(p["attribute"] == "origin_stand" for p in record.predicates)
        self.assertGreater(with_stand, 0)
        for record in records["porto_v3_single"]:
            self.assertEqual(predictions[("porto_v3_single", record.query_id)].status, "unsupported")
        self.assertIn("extension", summary["workloads"]["porto_v3_multi"])

    def test_sampling_ladder_answers_every_query(self) -> None:
        for name in ("traj_sampling_n1000", "traj_sampling_n10000", "traj_sampling_n100000", "traj_sampling_n1000000"):
            _, predictions, records, _ = self.run_shipped(name)
            # no agents on Porto: every single query (trips or segments) is answered
            self.assertTrue(all(p.status == "ok" for p in predictions.values()), name)
            self.assertEqual(len(predictions), sum(len(v) for v in records.values()))

    def test_case_plus_variants_on_the_window_domain_grid(self) -> None:
        extent = grid_extent(porto_settings(), "window_domain")
        for name in ("case_plus", "case_plus_temporal", "case_plus_temporal_best"):
            run, predictions, records, summary = self.run_shipped(name)
            meta = json.loads((run / "model" / "model_meta.json").read_text())
            self.assertEqual(meta["grid_extent"], "window_domain", name)
            self.assertEqual(meta["grid_extent_box"], list(extent), name)
            self.assertGreater(summary["workloads"]["porto_v3_multi"]["extension"]["coverage"]["scored_count"], 0)
            self.assertEqual(summary["workloads"]["porto_v3_single"]["extension"]["coverage"]["scored_count"], 0)
            for (workload, _), prediction in predictions.items():
                if prediction.status == "ok":
                    self.assertGreaterEqual(prediction.estimated_cardinality, 0.0)

    def test_mscn_trains_and_answers_every_query(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("torch is not installed")
        paths = _fixture_configs(self.root, ["mscn"])
        text = paths["mscn"].read_text(encoding="utf-8")
        training = write_porto_training(self.root, self.root / "benchmark.jsonl")
        text = re.sub(r"(training_jsonl|validation_jsonl): \S+", rf"\1: {training}", text)
        paths["mscn"].write_text(text, encoding="utf-8")
        config = load_experiment_config(paths["mscn"])
        run = run_seed(config, 0, run_id="mscn")
        predictions = read_predictions(run / "predictions.csv")
        self.assertEqual(len(predictions), len(porto_benchmark_rows()))
        self.assertTrue(all(p.status == "ok" and p.estimated_cardinality >= 1.0 for p in predictions))

    @unittest.skipUnless(TRAHIST_SOURCE, "set TRAJ_EVAL_TRAHIST_SOURCE to the TraHist checkout (870e8ce)")
    def test_trahist_variants_clip_the_outliers_onto_the_window_domain(self) -> None:
        extent = grid_extent(porto_settings(), "window_domain")
        for name in ("trahist", "trahist_temporal"):
            run, predictions, _, summary = self.run_shipped(name)
            meta = json.loads((run / "model" / "model_meta.json").read_text())
            self.assertEqual(meta["grid_extent_box"], list(extent), name)
            self.assertGreater(meta["points_outside_grid_extent"], 0, name)
            # the grid never reaches past the window domain
            self.assertLessEqual(meta["grid_shape"][0], int(np.ceil((extent[2] - extent[0]) / 2000)) + 1)
            self.assertLessEqual(meta["grid_shape"][1], int(np.ceil((extent[3] - extent[1]) / 2000)) + 1)
            self.assertGreater(summary["workloads"]["porto_v3_multi"]["extension"]["coverage"]["scored_count"], 0)


class PortoDeepDbExportTest(_PortoRoot):
    def test_null_origin_stand_is_exported_as_nan_and_translated_as_a_range(self) -> None:
        settings = load_settings(PORTO_CONFIG)  # the settings the snapshot was exported for
        out = self.root / "deepdb"
        manifest = deepdb_mbr.export(settings, self.root / "snapshot", out, log=lambda message: None)
        layout = deepdb_mbr.read_layout(out)
        names = [column["name"] for column in layout["columns"]]
        self.assertIn("t_origin_stand", names)
        self.assertNotIn("a_", " ".join(names))  # no agents table
        self.assertEqual(manifest["nominal_values"], {"t_call_type": ["A", "B", "C"]})
        column = names.index("t_origin_stand")
        lines = (out / deepdb_mbr.CSV_NAME).read_text().splitlines()
        stands = [line.split(",")[column] for line in lines]
        nulls = int(np.isnan(porto_synthetic_data()["trips"]["origin_stand"]).sum())
        self.assertEqual(stands.count("nan"), nulls)  # pandas reads it as NaN: DeepDB's NULL
        self.assertGreater(nulls, len(lines) // 2)
        translated = 0
        for row in porto_benchmark_rows():
            conditions, diagnostic = deepdb_mbr.translate(row, layout)
            if row["category"]["relation"] == "single":
                self.assertIsNone(conditions)
                continue
            if any(p["attribute"] == "segment_idx" for p in row["predicates"]):
                self.assertIsNone(conditions)
                continue
            self.assertIsNotNone(conditions, diagnostic)
            deepdb_mbr.to_sql(conditions)
            translated += any(c[0] == "t_origin_stand" for c in conditions)
        self.assertGreater(translated, 0)


DEEPDB_PYTHON = os.environ.get("TRAJ_EVAL_DEEPDB_PYTHON")
DEEPDB_SOURCE = os.environ.get("TRAJ_EVAL_DEEPDB_SOURCE")


@unittest.skipUnless(DEEPDB_PYTHON and DEEPDB_SOURCE, "set TRAJ_EVAL_DEEPDB_PYTHON and TRAJ_EVAL_DEEPDB_SOURCE")
class PortoRealDeepDbTest(_PortoRoot):
    """The shipped Porto DeepDB-MBR config with DeepDB itself: NULL origin_stand never matches."""

    def test_shipped_config_against_exact_mbr(self) -> None:
        import subprocess

        settings = load_settings(PORTO_CONFIG)
        shared = self.root / "porto_v3_deepdb_mbr"
        deepdb_mbr.export(settings, self.root / "snapshot", shared, log=lambda message: None)
        bridge = CONFIGS.parent / "scripts" / "deepdb_mbr_bridge.py"
        subprocess.run([DEEPDB_PYTHON, "-W", "ignore", str(bridge), "prepare-shared", "--source-root", DEEPDB_SOURCE,
                        "--revision", "655ada13a043a4226a239b0c0a65bcdefef87a02", "--shared-root", str(shared)],
                       check=True, capture_output=True, text=True)
        paths = _fixture_configs(self.root, ["deepdb_mbr", "exact_mbr"])
        text = paths["deepdb_mbr"].read_text(encoding="utf-8")
        text = text.replace("/work_beegfs/sunip956/master_thesis_trajectories/joblight_evaluation/envs/deepdb/bin/python",
                            f"{DEEPDB_PYTHON} -W ignore")
        text = text.replace("/work_beegfs/sunip956/master_thesis_trajectories/external/deepdb", DEEPDB_SOURCE)
        text = text.replace("/work_beegfs/sunip956/master_thesis_trajectories/traj_snapshots/porto_v3_deepdb_mbr",
                            str(shared))
        text = text.replace("working_directory: ../../../..", f"working_directory: {CONFIGS.parents[2]}")
        text = text.replace("seeds: [0, 1, 2]", "seeds: [0]")
        self.assertNotIn("/work_beegfs", text)
        paths["deepdb_mbr"].write_text(text, encoding="utf-8")
        runs = {}
        for name in ("deepdb_mbr", "exact_mbr"):
            config = load_experiment_config(paths[name])
            run = run_seed(config, 0, run_id=name)
            runs[name] = {(p.workload, p.query_id): p for p in read_predictions(run / "predictions.csv")}
        records = {r.query_id: r for w in config.workloads for r in load_queries(w)}
        q_all, q_stand = [], []
        for key, exact in runs["exact_mbr"].items():
            estimate = runs["deepdb_mbr"][key]
            self.assertEqual(estimate.status, exact.status, key)
            if exact.status != "ok":
                continue
            a, b = max(estimate.estimated_cardinality, 1.0), max(exact.estimated_cardinality, 1.0)
            q_all.append(max(a / b, b / a))
            if any(p["attribute"] == "origin_stand" for p in records[key[1]].predicates):
                q_stand.append(max(a / b, b / a))
        self.assertGreater(len(q_all), 30)
        self.assertGreater(len(q_stand), 3)
        # an SPN on all 240 trips; counting the NULL trips (more than half) would inflate these
        self.assertLess(float(np.median(q_all)), 2.0)
        self.assertLess(float(np.median(q_stand)), 2.0, q_stand)


class PortoOwnModelTranslationTest(unittest.TestCase):
    def test_every_porto_query_translates(self) -> None:
        routes = {}
        for row in porto_benchmark_rows():
            translation = omt.translate(copy.deepcopy(row), srid=3763)
            routes.setdefault(row["category"]["relation"], set()).add(translation.route)
            for predicate in translation.record["predicates"]:
                if predicate.get("attribute") == "origin_stand":
                    self.assertIn(predicate["mode"], ("range", "unbounded"))
        self.assertEqual(routes, {"multi": {"distinct"}, "single": {"rows"}})


if __name__ == "__main__":
    unittest.main()


class PortoTuningTest(_PortoRoot):
    """A setting the builder refuses as too large is skipped, not fatal (review of aa6b5e1)."""

    def _shipped(self, name: str) -> Path:
        return _fixture_configs(self.root, [name])[name]  # the fixture benchmark stands in for validation

    def test_oversized_porto_setting_is_skipped_and_the_sweep_goes_on(self) -> None:
        from evaluation.trajectory_pol_porto.traj_eval import tune
        from evaluation.trajectory_pol_porto.traj_eval.temporal import SettingTooLarge

        config = self._shipped("case_plus_temporal")
        output = self.root / "tuning" / "case_plus_temporal"
        messages = []
        # 25 m on the window domain: 20,562,894 cells x the keys exceed the dense-count guard
        rows = tune.sweep(config, self.root / "benchmark.jsonl",
                          tune.parse_grid(["unit_size_m=25,2000", "time_buckets=4"]), 8 * MULTI, output,
                          log=messages.append)
        self.assertEqual([json.loads(row["parameters"])["unit_size_m"] for row in rows], [2000])
        skipped = json.loads((output / "tuning_skipped.json").read_text())
        self.assertEqual([json.loads(item["parameters"]) for item in skipped], [{"unit_size_m": 25, "time_buckets": 4}])
        self.assertIn("20562894 cells is too large", skipped[0]["reason"])
        self.assertTrue(any(message.startswith("skipped ") for message in messages))
        self.assertEqual(len(list(csv_rows(output / "tuning.csv"))), 1)
        # nothing feasible at all: the sweep fails instead of writing an empty table
        with self.assertRaises(SettingTooLarge):
            tune.sweep(config, self.root / "benchmark.jsonl", tune.parse_grid(["unit_size_m=25", "time_buckets=4"]),
                       8 * MULTI, self.root / "tuning" / "none", log=lambda message: None)

    def test_other_errors_still_stop_the_sweep(self) -> None:
        from unittest import mock

        from evaluation.trajectory_pol_porto.traj_eval import tune

        config = self._shipped("case_plus")
        real = tune.run_seed
        calls = []

        def failing(config_, seed):
            calls.append(config_.variant_id)
            if len(calls) == 2:
                raise MemoryError("a real allocation failure")  # not a guard: must not be skipped
            return real(config_, seed)

        output = self.root / "tuning" / "stops"
        with mock.patch.object(tune, "run_seed", failing), self.assertRaises(MemoryError) as caught:
            tune.sweep(config, self.root / "benchmark.jsonl", tune.parse_grid(["unit_size_m=4000,2000,1000"]),
                       8 * MULTI, output, log=lambda message: None)
        self.assertEqual(str(caught.exception), "a real allocation failure")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list(csv_rows(output / "tuning.csv"))), 1)  # the finished setting is kept


def csv_rows(path: Path):
    import csv as _csv

    with path.open(encoding="utf-8", newline="") as handle:
        yield from _csv.DictReader(handle)


class SubmitScriptsDatasetTest(unittest.TestCase):
    """TRAJ_DATASET=porto_v3 points every submit script at the Porto configs, results and database job."""

    SCRIPTS = {
        "submit_p3.sh": {},
        "submit_p4.sh": {},
        "submit_p5.sh": {},
        "submit_p6.sh": {"STAGE": "tune", "CONFIG_NAMES": "case_plus_temporal"},
        "submit_p8b.sh": {},
        "submit_timing.sh": {"PROFILES": "cpu_1core cpu_postgres_2core"},
    }

    def _submit(self, script: str, env_extra: dict):
        import subprocess

        repo = CONFIGS.parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "sbatch"
            log = Path(tmp) / "sbatch.log"
            fake.write_text('#!/bin/bash\necho "$@" >> "$FAKE_SBATCH_LOG"\necho 4242\n')
            fake.chmod(0o755)
            env = {k: v for k, v in os.environ.items()
                   if k not in ("TRAJ_DATASET", "COMPARE_WITH", "CONFIG_NAMES", "UNIT_SIZES", "TIME_BUCKETS")}
            env.update({"PATH": f"{tmp}:{os.environ['PATH']}", "FAKE_SBATCH_LOG": str(log), "REPO_ROOT": str(repo),
                        **env_extra})
            result = subprocess.run(["bash", str(repo / "evaluation/trajectory_pol_porto/slurm" / script)],
                                    env=env, capture_output=True, text=True)
            submitted = log.read_text().splitlines() if log.exists() else []
        return result, submitted

    def test_porto(self) -> None:
        for script, extra in self.SCRIPTS.items():
            result, submitted = self._submit(script, {"TRAJ_DATASET": "porto_v3", **extra})
            self.assertEqual(result.returncode, 0, (script, result.stderr))
            self.assertTrue(submitted, script)
            text = "\n".join(submitted)
            self.assertNotIn("configs/pol_v3", text, script)
            self.assertNotIn("traj_phase1_pol_v3", text, script)
            self.assertIn("configs/porto_v3/", text, script)
            self.assertTrue(all("TRAJ_DATASET=porto_v3" in line for line in submitted), script)
            if script in ("submit_p4.sh", "submit_timing.sh"):
                self.assertIn("--job-name=porto_mobilitydb", text, script)
            if script == "submit_p6.sh":
                self.assertIn("qgen_porto_v3_validation_seed20261103", text)
                self.assertIn("traj_phase1_porto_v3_tuning/case_plus_temporal", text)
                # coarse first on Porto's 20.6 M-cell window domain (review of aa6b5e1)
                self.assertIn("GRIDS=unit_size_m=800/400/200/100/50/25+time_buckets=", text)
            if script not in ("submit_p6.sh",):
                self.assertIn("results/traj_phase1_porto_v3/", text, script)

    def test_pol_stays_the_default(self) -> None:
        for script, extra in self.SCRIPTS.items():
            result, submitted = self._submit(script, extra)
            self.assertEqual(result.returncode, 0, (script, result.stderr))
            text = "\n".join(submitted).replace("trajectory_pol_porto", "")
            self.assertNotIn("porto", text, script)
            self.assertIn("configs/pol_v3/", text, script)
            if script in ("submit_p4.sh", "submit_timing.sh"):
                self.assertIn("--job-name=pol_mobilitydb", text, script)
            if script == "submit_p6.sh":
                self.assertIn("GRIDS=unit_size_m=25/50/100/200/400+time_buckets=", text)

    def test_unknown_dataset_is_refused(self) -> None:
        result, submitted = self._submit("submit_p3.sh", {"TRAJ_DATASET": "porto"})
        self.assertEqual(result.returncode, 2)
        self.assertEqual(submitted, [])
