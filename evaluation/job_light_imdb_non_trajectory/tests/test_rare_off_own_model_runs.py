"""scripts/rare_off_own_model_runs.py and the rare-support-off rule configs."""

from __future__ import annotations

import importlib.util
import runpy
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rare_off_runs", HERE / "scripts" / "rare_off_own_model_runs.py")
RUNS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNS)


def _merged(path: Path) -> dict:
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import _deep_merge, load_simple_yaml

    raw = load_simple_yaml(path)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    base = load_simple_yaml(path.parent / extends)
    assert "extends" not in base, "nested inheritance"
    return _deep_merge(base, raw)


def _commands(config: dict) -> list[str]:
    adapter = config["adapter"]
    return [value for value in adapter.values() if isinstance(value, str) and "own_model_bridge" in value] + list(
        adapter["timing_commands"].values())


class RuleConfigTest(unittest.TestCase):
    def test_rule_configs_point_only_at_the_rare_off_runs(self) -> None:
        for workload, spec in RUNS.WORKLOADS.items():
            config = _merged(HERE / "configs" / spec["rule_config"])
            self.assertEqual(config["experiment"]["variant_id"], spec["rule_variant"])
            self.assertEqual(config["experiment"]["seeds"], [0, 1, 2])
            self.assertEqual(list(config["workloads"]), [workload])
            directory = f"joblight_evaluation/{spec['training_directory']}/seed_{{seed}}/"
            texts = _commands(config) + [config["artifacts"]["checkpoint_template"],
                                         config["artifacts"]["training_config_template"]]
            self.assertEqual(len(_commands(config)), 6)
            for text in texts:
                self.assertIn(directory, text)
                self.assertNotIn("rare_aux", text)
            for text in _commands(config) + [config["artifacts"]["checkpoint_template"]]:
                self.assertIn("checkpoint_best.pt", text)
                self.assertNotIn("checkpoint_step_", text)
            # same inference code and timing protocol as the earlier rule rows
            self.assertEqual(config["source"]["revision"], "7c156b247031d7f5dfbd96e812f111abe14a75f2")
            self.assertEqual(config["timing"]["profiles"], ["cpu_1core", "gpu_single_query"])
            self.assertTrue(config["adapter"]["timing_commands"]["cpu_1core"].endswith("--device cpu"))
            self.assertTrue(config["adapter"]["timing_commands"]["gpu_single_query"].endswith("--device cuda"))


class FinalConfigTest(unittest.TestCase):
    def test_final_step_is_the_last_checkpoint_of_a_finished_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for step in (500, 11000, 20000, 19500):
                (run / f"checkpoint_step_{step}.pt").write_bytes(b"")
            (run / "checkpoint_best.pt").write_bytes(b"")
            with self.assertRaises(SystemExit):
                RUNS.final_step(run)  # no training_summary.json yet
            (run / "training_summary.json").write_text("{}")
            self.assertEqual(RUNS.final_step(run), 20000)

    def test_generated_config_changes_only_step_variant_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            training, configs = Path(directory) / "training", Path(directory) / "configs"
            configs.mkdir()
            for spec in RUNS.WORKLOADS.values():
                for name in (spec["rule_config"],):
                    (configs / name).write_text((HERE / "configs" / name).read_text())
            for name in ("own_model_job_light_step_11000.yaml", "own_model_job_light_ranges_full_schema_step_11000.yaml"):
                (configs / name).write_text((HERE / "configs" / name).read_text())
            run = training / "own_model_job_light_rare_off_20k" / "seed_1"
            run.mkdir(parents=True)
            (run / "checkpoint_step_20000.pt").write_bytes(b"")
            (run / "checkpoint_best.pt").write_bytes(b"")
            (run / "training_summary.json").write_text("{}")
            path = RUNS.write_final_config("job_light", 1, training_root=training, configs=configs)
            final, rule = _merged(path), _merged(configs / RUNS.WORKLOADS["job_light"]["rule_config"])
            self.assertEqual(final["experiment"]["variant_id"], "rare_off_final_seed1")
            self.assertEqual(final["experiment"]["seeds"], [1])
            self.assertIn("20,000", final["experiment"]["display_name"])
            for text in _commands(final):
                self.assertIn("checkpoint_step_20000.pt", text)
                self.assertNotIn("checkpoint_best.pt", text)
            self.assertEqual(
                [text.replace("checkpoint_step_20000.pt", "checkpoint_best.pt") for text in _commands(final)],
                _commands(rule))
            final["experiment"].pop("seeds"), rule["experiment"].pop("seeds")
            for key in ("variant_id", "display_name"):
                final["experiment"].pop(key), rule["experiment"].pop(key)
            final.pop("adapter"), rule.pop("adapter")
            final["artifacts"] = {k: v.replace("checkpoint_step_20000.pt", "checkpoint_best.pt") if isinstance(v, str) else v
                                  for k, v in final["artifacts"].items()}
            self.assertEqual(final, rule)
        self.assertEqual(RUNS.final_variant("job_light_ranges", 2), "rare_off_content_final_seed2")


class RunLookupTest(unittest.TestCase):
    def _run(self, results: Path, variant: str, seed: int, name: str, complete: bool = True) -> Path:
        path = results / variant / f"seed_{seed}" / name
        path.mkdir(parents=True)
        if complete:
            (path / "summary.json").write_text("{}")
        return path

    def test_plans_and_sensitivity_arguments_cover_six_runs_each(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            for workload, spec in RUNS.WORKLOADS.items():
                for seed in RUNS.SEEDS:
                    self._run(results, spec["rule_variant"], seed, "best_val_rare_off-abc")
                    self._run(results, RUNS.final_variant(workload, seed), seed, "final_sensitivity_rare_off-def")
            lines = RUNS.timing_plan_lines("best_val_rare_off", results)
            self.assertEqual(len(lines), 7)
            self.assertEqual({line.split()[1] for line in lines[1:]}, {"0", "1", "2"})
            arguments = RUNS.sensitivity_arguments("best_val_rare_off", "final_sensitivity_rare_off", results)
            self.assertEqual(len(arguments), 24)
            self.assertEqual(arguments[0::2].count("--job-light-final"), 3)
            # a second or an incomplete run of the same id is refused
            self._run(results, "rare_off_best_val", 0, "best_val_rare_off-zzz")
            with self.assertRaises(SystemExit):
                RUNS.timing_plan_lines("best_val_rare_off", results)
            self._run(results, "rare_off_content_final_seed0", 0, "other-1", complete=False)
            with self.assertRaises(SystemExit):
                RUNS.completed_run("rare_off_content_final_seed0", 0, "other", results)


class TrainingHardwareTest(unittest.TestCase):
    def test_gpu_per_run_from_the_finished_logs_and_the_table_note(self) -> None:
        spec = importlib.util.spec_from_file_location("hardware", HERE / "scripts" / "own_model_training_hardware.py")
        hardware = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hardware)
        runs = [f"/x/own_model_job_light_rare_off_20k/seed_{seed}" for seed in RUNS.SEEDS]
        with tempfile.TemporaryDirectory() as directory:
            logs = []
            for index, (run, gpu, finished) in enumerate([
                    (runs[0], "NVIDIA L40S", True), (runs[1], "NVIDIA H100 80GB HBM3", True),
                    (runs[2], "NVIDIA L40S", True), (runs[1], "NVIDIA L40S", False)]):
                log = Path(directory) / f"own_main_rare_off_{index}.out"
                log.write_text(f"job_id={index}\noutput_directory={run}\nruntime_python=x cuda_device={gpu}\n"
                               + ("end_utc=2026-11-30T00:00:00Z\n" if finished else ""))
                logs.append(log)
            result = hardware.hardware(runs, logs)
            self.assertEqual([result[run]["gpu"] for run in runs],
                             ["NVIDIA L40S", "NVIDIA H100 80GB HBM3", "NVIDIA L40S"])
            with self.assertRaises(SystemExit):  # a run without a finished log
                hardware.hardware(runs + ["/x/other"], logs)

        MODULE = runpy.run_path(str(HERE / "scripts" / "make_latex_tables.py"))

        variant = MODULE["OWN_MODEL_VARIANTS"]["rare_off_best_val"]
        directories = [str(Path(path).parent) for path in variant["training_summaries"]]
        row = {"method_id": "own_model", "variant_id": "rare_off_best_val"}
        table_hardware = MODULE["OWN_TRAINING_HARDWARE"]
        try:
            table_hardware.update({directories[0]: {"gpu": "NVIDIA L40S"},
                                   directories[1]: {"gpu": "NVIDIA H100 80GB HBM3"},
                                   directories[2]: {"gpu": "NVIDIA L40S"}})
            self.assertEqual(MODULE["own_model_training_hardware_note"]([row]),
                             r"\OwnModelName{}: trained on one GPU per seed, "
                             "H100 80GB HBM3 (seed 1); L40S (seeds 0, 2).")
            for directory in directories:
                table_hardware[directory] = {"gpu": "NVIDIA L40S"}
            self.assertEqual(MODULE["own_model_training_hardware_note"]([row]),
                             r"\OwnModelName{}: trained on one L40S GPU per seed.")
            table_hardware.clear()
            self.assertIn("not recorded", MODULE["own_model_training_hardware_note"]([row]))
        finally:
            table_hardware.clear()


if __name__ == "__main__":
    unittest.main()
