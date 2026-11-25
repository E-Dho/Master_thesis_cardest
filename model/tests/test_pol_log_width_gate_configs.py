"""The POL ablations train with the same log-width gate, runtime and prefetch as the
gated main model they are compared with (pol_main_log_width_gate_seed0.yaml), and the
three gated main seeds differ only in their seeds and output directories."""

from __future__ import annotations

import unittest
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class PolLogWidthGateConfigTest(unittest.TestCase):
    def test_ablations_share_the_main_model_gate(self) -> None:
        main = load_simple_yaml(str(CONFIGS / "pol_main_log_width_gate_seed0.yaml"))
        validate_config(main)
        gate = main["predicate_generation"]["log_width_gate"]
        self.assertEqual(gate["probability"], 0.5)
        ablations = sorted(CONFIGS.glob("pol_ablation_*.yaml"))
        self.assertEqual(len(ablations), 5)
        for path in ablations:
            config = load_simple_yaml(str(path))
            validate_config(config)
            self.assertEqual(config["predicate_generation"]["log_width_gate"], gate, path.name)
            self.assertEqual(config["runtime"], main["runtime"], path.name)
            self.assertEqual(config["training"]["prefetch_workers"], main["training"]["prefetch_workers"])
            self.assertEqual(config["training"]["seed"], 0)

    def test_main_seeds_differ_only_in_seed_and_output(self) -> None:
        configs = [load_simple_yaml(str(CONFIGS / f"pol_main_log_width_gate_seed{seed}.yaml")) for seed in range(3)]
        for seed, config in enumerate(configs):
            self.assertEqual(config["training"]["seed"], seed)
            self.assertEqual(config["predicate_generation"]["seed"], seed)
            self.assertEqual(config["logging"]["output_directory"], f"model/runs/pol_main_log_width_gate_seed{seed}")
            normalized = {**config, "training": {**config["training"], "seed": 0},
                          "predicate_generation": {**config["predicate_generation"], "seed": 0},
                          "logging": {**config["logging"], "output_directory": None}}
            reference = {**configs[0], "logging": {**configs[0]["logging"], "output_directory": None}}
            self.assertEqual(normalized, reference, seed)


if __name__ == "__main__":
    unittest.main()
