from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "own_model_bridge.py"
SPEC = importlib.util.spec_from_file_location("own_model_bridge_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


class OwnModelBridgeTest(unittest.TestCase):
    def test_activates_and_records_the_pinned_model_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "model_source"
            (source / "model").mkdir(parents=True)
            (source / "model" / "__init__.py").write_text("", encoding="utf-8")
            subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(
                [
                    "git", "-C", str(source),
                    "-c", "user.name=Evaluation Test",
                    "-c", "user.email=evaluation@example.invalid",
                    "commit", "-m", "source fixture",
                ],
                check=True,
                capture_output=True,
            )
            revision = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            resolved_source = str(source.resolve())
            original_path = list(sys.path)
            try:
                with mock.patch.dict(
                    os.environ,
                    {
                        "OWN_MODEL_SOURCE": str(source),
                        "OWN_MODEL_SOURCE_REVISION": revision,
                        "PYTHONPATH": "/existing/python/path",
                    },
                    clear=False,
                ):
                    observed = BRIDGE._activate_model_source()
                    self.assertEqual(
                        observed, {"path": resolved_source, "revision": revision}
                    )
                    self.assertEqual(sys.path[0], resolved_source)
                    self.assertTrue(
                        os.environ["PYTHONPATH"].startswith(f"{resolved_source}:")
                    )
            finally:
                sys.path[:] = original_path

