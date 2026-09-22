from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_BRIDGE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "neurocard_bridge.py"
)
_SPEC = importlib.util.spec_from_file_location("neurocard_bridge", _BRIDGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BRIDGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BRIDGE)


class NeuroCardBridgeTest(unittest.TestCase):
    def test_resolves_singleton_upstream_grid_search_values(self) -> None:
        value = {
            "scheduler": {"grid_search": ["OneCycleLR-0.28"]},
            "nested": [{"grid_search": [11]}],
        }
        self.assertEqual(
            _BRIDGE._resolve_grid_values(value),
            {"scheduler": "OneCycleLR-0.28", "nested": [11]},
        )

    def test_rejects_multi_choice_upstream_grid_search(self) -> None:
        with self.assertRaises(ValueError):
            _BRIDGE._resolve_grid_values({"grid_search": [1, 2]})

    def test_model_variants_use_distinct_upstream_experiments(self) -> None:
        self.assertEqual(
            _BRIDGE.MODEL_CONFIGS["job_light"]["experiment"], "job-light"
        )
        self.assertEqual(
            _BRIDGE.MODEL_CONFIGS["job_light_ranges"]["experiment"],
            "job-light-ranges",
        )


if __name__ == "__main__":
    unittest.main()
