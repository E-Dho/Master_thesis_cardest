"""The gated Porto main config is the evaluated Porto config plus the log-width gate,
with rare support off (decided 2026-11-25), prefetch and the learned-gpu runtime.

The Porto ablation "rare support off" was dropped on 2026-11-25: rare support is off
for all own-model trainings, so it would equal the main model.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVALUATED = CONFIGS / "porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"
MAIN = CONFIGS / "porto_main_log_width_gate.yaml"
LEARNED_GPU = "/work_beegfs/sunip956/master_thesis_trajectories/joblight_evaluation/envs/learned-gpu/bin/python"


class PortoMainLogWidthGateConfigTest(unittest.TestCase):
    def test_main_is_the_evaluated_config_plus_gate_and_rare_off(self) -> None:
        main = load_simple_yaml(str(MAIN))
        validate_config(main)
        evaluated = load_simple_yaml(str(EVALUATED))
        gate = main["predicate_generation"].pop("log_width_gate")
        self.assertEqual(gate["probability"], 0.5)
        self.assertEqual(gate["span_quantile"], 0.01)
        self.assertEqual(main["training"].pop("prefetch_workers"), 3)
        self.assertEqual(main.pop("runtime"), {"python": LEARNED_GPU, "require_cuda": True})
        self.assertEqual(main["logging"].pop("output_directory"), "model/runs/porto_main_log_width_gate")
        evaluated["logging"].pop("output_directory")
        self.assertTrue(evaluated["rare_support"]["enabled"] and evaluated["rare_auxiliary"]["enabled"])
        self.assertFalse(main["rare_support"]["enabled"] or main["rare_auxiliary"]["enabled"])
        main["rare_support"]["enabled"] = main["rare_auxiliary"]["enabled"] = True
        self.assertEqual(main, evaluated)

    def test_no_porto_ablation_configs_left(self) -> None:
        self.assertEqual(sorted(CONFIGS.glob("porto_ablation_*.yaml")), [])


if __name__ == "__main__":
    unittest.main()
