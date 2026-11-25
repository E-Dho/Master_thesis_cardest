"""IMDB trainings with rare support off (decided 2026-11-25).

The two main templates are the evaluated models' templates with only rare support
off, prefetch, the learned-gpu runtime and the output directory changed; the three
ablation templates have rare support off as well; the evaluation configs of the new
main models point at their run directories with the 7c156b2 inference code.
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
    return config


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

    def test_evaluation_configs_point_at_the_new_runs(self) -> None:
        for name, run, variant in (
            ("own_model_job_light_ranges_rare_off_step_11000.yaml", "own_model_job_light_ranges_rare_off_20k",
             "rare_off_content_step_11000"),
            ("own_model_job_light_rare_off_step_11000.yaml", "own_model_rare_off_40k", "rare_off_step_11000"),
        ):
            text = (EVALUATION / name).read_text()
            self.assertIn(f"/joblight_evaluation/{run}/seed_{{seed}}/checkpoint_step_11000.pt", text)
            self.assertNotIn("rare_aux", text.split("schema_version", 1)[1])
            self.assertIn("revision: 7c156b247031d7f5dfbd96e812f111abe14a75f2", text)
            self.assertIn("seeds: [0, 1, 2]", text)
            self.assertIn(f"variant_id: {variant}\n", text)


if __name__ == "__main__":
    unittest.main()
