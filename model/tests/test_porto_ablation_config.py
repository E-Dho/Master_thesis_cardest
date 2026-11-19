"""Porto ablation configs differ from the evaluated Porto model's config only in their switch."""

from __future__ import annotations

import unittest
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVALUATED = CONFIGS / "porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"


def _without_comments(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]


class PortoAblationConfigTest(unittest.TestCase):
    def test_rare_support_off_changes_only_rare_support_and_the_output(self) -> None:
        path = CONFIGS / "porto_ablation_rare_support_off.yaml"
        reference, ablation = _without_comments(EVALUATED), _without_comments(path)
        self.assertEqual(len(reference), len(ablation))
        changed = [(a, b) for a, b in zip(reference, ablation) if a != b]
        self.assertEqual(changed, [
            ("  enabled: true", "  enabled: false"),
            ("  enabled: true", "  enabled: false"),
            ("  output_directory: model/runs/porto_traj_dedup_single_anchor_mbr_segments_first_20k",
             "  output_directory: model/runs/porto_ablation_rare_support_off"),
        ])
        config = load_simple_yaml(str(path))
        validate_config(config)
        evaluated = load_simple_yaml(str(EVALUATED))
        self.assertFalse(config["rare_support"]["enabled"])
        self.assertFalse(config["rare_auxiliary"]["enabled"])
        self.assertTrue(evaluated["rare_support"]["enabled"] and evaluated["rare_auxiliary"]["enabled"])
        for key in ("dataset", "model", "training", "predicate_generation", "trajectory_distinct"):
            self.assertEqual(config.get(key), evaluated.get(key), key)


if __name__ == "__main__":
    unittest.main()
