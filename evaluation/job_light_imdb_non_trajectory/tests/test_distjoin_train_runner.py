from __future__ import annotations

import argparse
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "distjoin_train_runner.py"
SPEC = importlib.util.spec_from_file_location("distjoin_train_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class DistJoinTrainRunnerTests(unittest.TestCase):
    def test_installs_missing_optional_transformer_sentinel(self):
        module = types.SimpleNamespace()
        RUNNER._install_missing_tesseract_transformer(module)
        sentinel = module.TesseractTransformer
        self.assertFalse(isinstance(object(), sentinel))
        RUNNER._install_missing_tesseract_transformer(module)
        self.assertIs(module.TesseractTransformer, sentinel)

    def test_initializes_upstream_main_only_globals(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "IMDB.yaml"
            config.write_text(
                "seed: 7\nexcludes: []\ntrain:\n  epochs: 1\n",
                encoding="utf-8",
            )
            module = types.SimpleNamespace()
            class FakeYaml:
                @staticmethod
                def safe_load(_text):
                    return {"seed": 7, "excludes": [], "train": {"epochs": 1}}

            original = RUNNER._yaml_module
            RUNNER._yaml_module = lambda: FakeYaml
            self.addCleanup(setattr, RUNNER, "_yaml_module", original)
            RUNNER._configure_train_module(module, config, "smoke")
            self.assertEqual(module.config_seed, 7)
            self.assertEqual(module.config, {"epochs": 1})
            self.assertEqual(
                module.args,
                argparse.Namespace(config="IMDB", exp_mark="smoke"),
            )


if __name__ == "__main__":
    unittest.main()
