"""Data-shift ablation on JOB-light-ranges: the Slurm scripts.

- job_light_ranges_data_shift_20k.sbatch writes the evaluated model's 20k
  training config with only the data-shift changes (run with a stub ROOT,
  without nvidia-smi, sampler start-up and training);
- submit_imdb_data_shift.sh chains select -> subset -> relabel and subset -> prepare.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SLURM = REPO / "model/slurm"
SOURCE = REPO / "model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml"
ROOT = "/work_beegfs/sunip956/master_thesis_trajectories"


def _recipe(text: str, run_dir: str) -> str:
    """The evaluated model's 20k recipe (job_light_ranges_rare_aux_20k.sbatch, seed 0) without data paths."""

    text = re.sub(r"(?m)^    support_planning_steps: 40000$", "    support_planning_steps: 20000", text)
    text = re.sub(r"(?m)^  steps_per_epoch: 40000$", "  steps_per_epoch: 20000", text)
    text = re.sub(r"(?m)^  early_stopping_patience_steps: 5000$", "  early_stopping_patience_steps: 3000", text)
    return re.sub(r"(?m)^  output_directory:.*$", f"  output_directory: {run_dir}", text)


class TrainingScriptTest(unittest.TestCase):
    def _run(self, root: Path, kind: str, split: str = "temporal80") -> subprocess.CompletedProcess:
        script = (SLURM / "job_light_ranges_data_shift_20k.sbatch").read_text()
        script = script.replace(f"ROOT={ROOT}\n", f"ROOT={root}\n", 1)
        script = re.sub(r"(?m)^PYTHON=.*$", f"PYTHON={sys.executable}", script, count=1)
        script = re.sub(r"(?m)^nvidia-smi .*$", "true", script)
        tail = script.index('"$PYTHON" -m model.scripts.check_live_sampler_startup')
        script = script[:tail] + 'echo "stubbed start-up and training"\n'
        path = root / "train.sh"
        path.write_text(script)
        env = {**os.environ, "REPO": str(REPO), "KIND": kind, "SPLIT": split}
        return subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True)

    def test_configs_differ_from_the_evaluated_recipe_only_by_the_data_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data_shift/imdb/temporal80"
            for directory in (data / "csv", data / "prepared", data / "neurocard_r2/neurocard",
                              root / "Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content"):
                directory.mkdir(parents=True)
            for prepared in (data / "prepared", root / "Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content"):
                (prepared / "manifest.json").write_text(json.dumps({"join_cardinality": 80}))
            (data / "prepared/subset.json").write_text(json.dumps(
                {"neurocard_package": str(data / "neurocard_r2/neurocard"), "subset_join_cardinality": 80,
                 "full_join_cardinality": 100}))
            result = self._run(root, "finetune")
            self.assertNotEqual(result.returncode, 0)  # the stale checkpoint is missing
            self.assertIn("missing", result.stderr)
            result = self._run(root, "stale")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("config_checked kind=stale", result.stdout)
            run = root / "joblight_evaluation/own_model_data_shift/temporal80_stale/seed_0"
            stale = (run / "training_config.yaml").read_text()
            expected = _recipe(SOURCE.read_text(), str(run))
            expected = re.sub(r"(?m)^  csv_directory:.*$", f"  csv_directory: {data / 'csv'}", expected)
            expected = re.sub(r"(?m)^  prepared_directory:.*$", f"  prepared_directory: {data / 'prepared'}", expected)
            # the working directory the preparation built, read from prepared/subset.json
            expected = re.sub(r"(?m)^  neurocard_path:.*$", f"  neurocard_path: {data / 'neurocard_r2/neurocard'}", expected)
            self.assertEqual(stale, expected)
            (run / "checkpoint_step_11000.pt").write_bytes(b"stub")
            result = self._run(root, "finetune")
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            tuned_run = root / "joblight_evaluation/own_model_data_shift/temporal80_finetune/seed_0"
            tuned = (tuned_run / "training_config.yaml").read_text()
            full = root / "Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content"
            expected = _recipe(SOURCE.read_text(), str(tuned_run))
            expected = re.sub(r"(?m)^  csv_directory:.*$", f"  csv_directory: {root / 'datasets/job'}", expected)
            expected = re.sub(r"(?m)^  prepared_directory:.*$", f"  prepared_directory: {full}", expected)
            expected = re.sub(r"(?m)^  neurocard_path:.*$", f"  neurocard_path: {root / 'external/neurocard/neurocard'}", expected)
            expected = expected.replace("    support_planning_steps: 20000", "    support_planning_steps: 2000")
            expected = expected.replace("  steps_per_epoch: 20000", "  steps_per_epoch: 2000")
            expected = expected.replace("  learning_rate: 0.0001\n", "  learning_rate: 0.00001\n")
            expected = expected.replace("training:\n", f"training:\n  init_checkpoint: {run / 'checkpoint_step_11000.pt'}\n", 1)
            self.assertEqual(tuned, expected)
            result = self._run(root, "stale")
            self.assertNotEqual(result.returncode, 0)  # the run directory is not reused
            self.assertTrue((run / "training_config.yaml").exists())


class SubmitScriptTest(unittest.TestCase):
    def test_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake, log, counter = Path(tmp) / "sbatch", Path(tmp) / "log", Path(tmp) / "id"
            counter.write_text("200")
            fake.write_text('#!/bin/bash\necho "$@" >> "$FAKE_LOG"\nn=$(cat "$FAKE_ID"); n=$((n + 1)); '
                            'echo $n > "$FAKE_ID"; echo $n\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "FAKE_LOG": str(log), "FAKE_ID": str(counter),
                   "SPLIT": "random80", "REPO": "/r"}
            result = subprocess.run(["bash", str(SLURM / "submit_imdb_data_shift.sh")], env=env,
                                    capture_output=True, text=True)
            if "exists; refusing" in result.stderr:
                self.skipTest("the cluster data directory exists here")
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = log.read_text().splitlines()
            steps = [re.search(r"STEP=(\w+)", line).group(1) for line in lines]
            self.assertEqual(steps, ["select", "subset", "relabel", "prepare"])
            self.assertNotIn("--dependency", lines[0])
            self.assertIn("--dependency=afterok:201 ", lines[1])
            self.assertIn("--dependency=afterok:202 ", lines[2])
            self.assertIn("--dependency=afterok:202 ", lines[3])
            self.assertTrue(all("REPO=/r" in line and "SPLIT=random80" in line for line in lines))
            self.assertIn("PREPARE_JOB=204", result.stdout)
            env["SPLIT"] = "other"
            self.assertNotEqual(subprocess.run(["bash", str(SLURM / "submit_imdb_data_shift.sh")], env=env,
                                               capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
