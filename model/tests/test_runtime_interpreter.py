"""runtime.python / runtime.require_cuda in the training config (model.src.runtime_interpreter)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from model.src.config import load_simple_yaml
from model.src.runtime_interpreter import REEXEC_MARKER, reexec_target, runtime_settings

HAS_TORCH = importlib.util.find_spec("torch") is not None
REPOSITORY = Path(__file__).resolve().parents[2]


class ReexecTargetTest(unittest.TestCase):
    def test_no_runtime_section_continues(self) -> None:
        self.assertIsNone(reexec_target({}, executable=sys.executable, environ={}))
        self.assertEqual(runtime_settings({}), (None, False))

    def test_already_under_the_configured_interpreter_continues(self) -> None:
        config = {"runtime": {"python": sys.executable}}
        self.assertIsNone(reexec_target(config, executable=sys.executable, environ={}))

    def test_other_interpreter_is_the_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "python"
            link.symlink_to(sys.executable)
            config = {"runtime": {"python": str(link), "require_cuda": "true"}}
            self.assertEqual(reexec_target(config, executable="/usr/bin/other", environ={}), str(link))
            self.assertEqual(runtime_settings(config), (str(link), True))

    def test_missing_interpreter_stops(self) -> None:
        config = {"runtime": {"python": "/nonexistent/bin/python"}}
        with self.assertRaisesRegex(SystemExit, "not an executable file"):
            reexec_target(config, executable=sys.executable, environ={})

    def test_no_second_reexec(self) -> None:
        config = {"runtime": {"python": "/envs/gpu/bin/python"}}
        with self.assertRaisesRegex(SystemExit, "refusing to loop"):
            reexec_target(config, executable="/usr/bin/other", environ={REEXEC_MARKER: "/envs/gpu/bin/python"})


def _smoke_config(directory: Path, runtime: str) -> Path:
    text = (REPOSITORY / "model/configs/resmade_smoke.yaml").read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("  output_directory:"):
            line = f"  output_directory: {directory / 'run'}"
        elif line.startswith("  steps_per_epoch:"):
            line = "  steps_per_epoch: 2"
        lines.append(line)
    path = directory / "config.yaml"
    path.write_text("\n".join(lines) + "\n" + runtime)
    return path


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrainResmadeRuntimeTest(unittest.TestCase):
    def _run(self, directory: Path, runtime: str) -> subprocess.CompletedProcess:
        config = _smoke_config(directory, runtime)
        environment = dict(os.environ, PYTHONPATH=str(REPOSITORY))
        environment.pop(REEXEC_MARKER, None)
        return subprocess.run([sys.executable, "-m", "model.scripts.train_resmade", "--config", str(config)],
                              cwd=REPOSITORY, env=environment, capture_output=True, text=True)

    def test_reexecutes_under_the_configured_interpreter_and_trains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "python"
            link.symlink_to(sys.executable)
            result = self._run(Path(directory), f"runtime:\n  python: {link}\n")
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertIn(f"runtime_python_reexec={link}", result.stdout)
            self.assertIn(f"runtime_python={link} ", result.stdout)
            self.assertIn("checkpoint=", result.stdout)

    def test_require_cuda_stops_instead_of_training_on_the_cpu(self) -> None:
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA is available here")
        with tempfile.TemporaryDirectory() as directory:
            result = self._run(Path(directory), "runtime:\n  require_cuda: true\n")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to train on the CPU", result.stderr + result.stdout)
            self.assertFalse((Path(directory) / "run").exists())


if __name__ == "__main__":
    unittest.main()
