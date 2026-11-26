"""IMDB trainings with rare support off (decided 2026-11-25), checkpoint by validation loss.

The two main templates are the evaluated models' templates with only rare support
off, prefetch, the learned-gpu runtime, the validation section and the output
directory changed; the three ablation templates have rare support off and the same
validation section. Since 2026-11-26 every IMDB training uses the recipe of the
POL/Porto mains (20,000 steps, no early stopping, checkpoint_best.pt by validation
loss on fresh sampler batches every 250 steps); the evaluation configs point at
checkpoint_best.pt of the new runs with the 7c156b2 inference code.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "model/configs"
EVALUATION = ROOT / "evaluation/job_light_imdb_non_trajectory/configs"
LEARNED_GPU = "/work_beegfs/sunip956/master_thesis_trajectories/joblight_evaluation/envs/learned-gpu/bin/python"


def _normalized(config: dict) -> dict:
    config = {**config}
    config["rare_support"] = {**config["rare_support"], "enabled": None}
    config["rare_auxiliary"] = {**config["rare_auxiliary"], "enabled": None}
    config["training"] = {k: v for k, v in config["training"].items() if k != "prefetch_workers"}
    config["logging"] = {k: v for k, v in config["logging"].items() if k != "output_directory"}
    config.pop("runtime", None)
    config.pop("validation", None)
    return config


VALIDATION = {"enabled": True, "prepared_directory": None, "interval_steps": 250, "fresh_sampler_batches": 4,
              "batch_size": 2048, "selection_metric": "validation_weighted_nll", "minimize": True}
SCRIPTS = ROOT / "model/slurm"


class ImdbRareOffConfigTest(unittest.TestCase):
    def test_main_templates_change_only_rare_support_and_runtime(self) -> None:
        for main, base in (
            ("imdb_main_job_light_ranges_rare_off.yaml", "job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml"),
            ("imdb_main_job_light_rare_off.yaml", "job_light_duet_binary_native_anpm_40k_rare_auxiliary.yaml"),
        ):
            config = load_simple_yaml(str(CONFIGS / main))
            validate_config(config)
            reference = load_simple_yaml(str(CONFIGS / base))
            self.assertFalse(config["rare_support"]["enabled"] or config["rare_auxiliary"]["enabled"], main)
            self.assertTrue(reference["rare_support"]["enabled"] and reference["rare_auxiliary"]["enabled"], base)
            self.assertEqual(config["training"]["prefetch_workers"], 3)
            self.assertEqual(config["runtime"], {"python": LEARNED_GPU, "require_cuda": True})
            self.assertEqual(_normalized(config), _normalized(reference), main)

    def test_ablation_templates_have_rare_support_off(self) -> None:
        ablations = sorted(CONFIGS.glob("imdb_ablation_*.yaml"))
        self.assertEqual(len(ablations), 3)
        for path in ablations:
            config = load_simple_yaml(str(path))
            validate_config(config)
            self.assertFalse(config["rare_support"]["enabled"] or config["rare_auxiliary"]["enabled"], path.name)

    def test_every_imdb_template_selects_by_validation_loss(self) -> None:
        names = ["imdb_main_job_light_ranges_rare_off.yaml", "imdb_main_job_light_rare_off.yaml"] + [
            path.name for path in sorted(CONFIGS.glob("imdb_ablation_*.yaml"))]
        self.assertEqual(len(names), 5)
        for name in names:
            config = load_simple_yaml(str(CONFIGS / name))
            self.assertEqual(config["validation"], VALIDATION, name)
            self.assertEqual(config["training"]["validation_interval_steps"], 250, name)

    def test_training_scripts_set_20000_steps_without_early_stopping(self) -> None:
        main = (SCRIPTS / "job_light_main_rare_off.sbatch").read_text()
        self.assertIn("STEPS=20000; PATIENCE=0; PLANNING=20000", main)
        self.assertNotIn("40000; PATIENCE", main)
        self.assertIn("own_model_job_light_rare_off_20k/seed_${SEED}", main)
        self.assertIn('validation["selection_metric"] == "validation_weighted_nll"', main)
        ablation = (SCRIPTS / "job_light_ranges_ablation_20k.sbatch").read_text()
        self.assertIn("early_stopping_patience_steps: 5000$/  early_stopping_patience_steps: 0/", ablation)
        self.assertIn("steps_per_epoch: 40000$/  steps_per_epoch: 20000/", ablation)
        self.assertIn('assert config["training"]["early_stopping_patience_steps"] == 0', ablation)

    def test_evaluation_configs_use_the_best_validation_checkpoint(self) -> None:
        cases = [
            ("own_model_job_light_ranges_rare_off_best_val.yaml", "own_model_job_light_ranges_rare_off_20k",
             "rare_off_content_best_val", "[0, 1, 2]"),
            ("own_model_job_light_rare_off_best_val.yaml", "own_model_job_light_rare_off_20k",
             "rare_off_best_val", "[0, 1, 2]"),
        ] + [(f"own_model_ablation_{name}_best_val.yaml", f"own_model_ablations/imdb_ablation_{name}",
              f"ablation_{name}_best_val", "[0]") for name in ("direct_io_off", "embed_input", "no_native_ranges")]
        for name, run, variant, seeds in cases:
            text = (EVALUATION / name).read_text()
            body = text.split("schema_version", 1)[1]
            self.assertEqual(body.count(f"/joblight_evaluation/{run}/seed_{{seed}}/checkpoint_best.pt"), 7, name)
            self.assertNotIn("checkpoint_step_", body)
            self.assertNotIn("rare_aux", body)
            self.assertIn("revision: 7c156b247031d7f5dfbd96e812f111abe14a75f2", text)
            self.assertIn(f"seeds: {seeds}", text)
            self.assertIn(f"variant_id: {variant}\n", text)


if __name__ == "__main__":
    unittest.main()
