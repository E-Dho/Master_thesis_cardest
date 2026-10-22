"""v3 config: domain overrides apply to v3 only and are validated."""

import json
import tempfile
import unittest
from pathlib import Path

from query_generation.benchmark_v3.config import load_settings
from query_generation.query_generator import ConfigError

CONFIG = Path(__file__).with_name("pol_v3_config.json")
BASE = Path(__file__).parent.parent / "pol_query_config.json"


class ConfigTests(unittest.TestCase):
    def write(self, tmp, **changes):
        payload = json.loads(CONFIG.read_text())
        payload["base_config"] = str(BASE.resolve())
        payload.update(changes)
        path = Path(tmp) / "v3.json"
        path.write_text(json.dumps(payload))
        return path

    def test_overrides_change_v3_domains_but_not_the_base_file(self):
        before = BASE.read_text()
        settings = load_settings(CONFIG)
        self.assertEqual(settings.attr("segments", "segment_geom")["domain"]["min_x"], 739732.900565014)
        self.assertEqual(settings.attr("trips", "trip_time")["domain"]["min"], "2019-07-01 05:25:00")
        self.assertEqual(BASE.read_text(), before)
        base = json.loads(before)
        self.assertNotEqual(base["tables"]["segments"]["attributes"][2]["domain"]["min_x"], 739732.900565014)

    def test_overrides_change_the_config_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain = load_settings(self.write(tmp, domain_overrides={}))
        self.assertNotEqual(plain.sha256, load_settings(CONFIG).sha256)

    def test_invalid_overrides_are_rejected(self):
        bad = [
            {"agents.height": {"min": 1, "max": 2}},
            {"agents.age": {"min": 1}},
            {"agents.age": {"min": 5, "max": 1}},
        ]
        for overrides in bad:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ConfigError, msg=overrides):
                load_settings(self.write(tmp, domain_overrides=overrides))


if __name__ == "__main__":
    unittest.main()
