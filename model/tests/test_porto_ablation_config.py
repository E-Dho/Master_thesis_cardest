"""Porto ablation configs differ from the evaluated Porto model's config only in their switch."""

from __future__ import annotations

import unittest
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EVALUATED = CONFIGS / "porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml"


LEARNED_GPU = "/work_beegfs/sunip956/master_thesis_trajectories/joblight_evaluation/envs/learned-gpu/bin/python"


def _without_comments(path: Path) -> list[str]:
    # training.prefetch_workers only sets how many processes prepare steps ahead;
    # the training is the same with any value (test_training_prefetch), so it is
    # not part of the ablation's switch. Neither is the runtime section: it selects
    # the interpreter the evaluated model was trained with (learned-gpu, PYTHON
    # override at its submission), see model.src.runtime_interpreter.
    lines, in_runtime = [], False
    for line in path.read_text().splitlines():
        if line == "runtime:":
            in_runtime = True
            continue
        if in_runtime and line.startswith("  "):
            continue
        in_runtime = False
        if line.lstrip().startswith("#") or line.startswith("  prefetch_workers:"):
            continue
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


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
        self.assertEqual(config["training"].pop("prefetch_workers"), 3)
        self.assertEqual(config.pop("runtime"), {"python": LEARNED_GPU, "require_cuda": True})
        for key in ("dataset", "model", "training", "predicate_generation", "trajectory_distinct"):
            self.assertEqual(config.get(key), evaluated.get(key), key)


if __name__ == "__main__":
    unittest.main()
