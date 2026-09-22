from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deepdb_bridge.py"
SPEC = importlib.util.spec_from_file_location("deepdb_bridge", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


class DeepDbBridgeTests(unittest.TestCase):
    def test_header_removal_preserves_escaped_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            target = Path(temporary) / "target.csv"
            source.write_bytes(b'id,text\n1,"a, b"\n2,"quoted \\"value\\""\n')
            BRIDGE.copy_without_header(source, target)
            self.assertEqual(target.read_bytes(), b'1,"a, b"\n2,"quoted \\"value\\""\n')

    def test_synthetic_fixture_is_headerless_and_has_expected_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            BRIDGE.write_fixture(root, rows=3)
            with (root / "title.csv").open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(rows[0]), 12)
            self.assertEqual(rows[0][0], "1")


if __name__ == "__main__":
    unittest.main()
