from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


_BRIDGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mscn_bridge.py"
_SPEC = importlib.util.spec_from_file_location("mscn_bridge", _BRIDGE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BRIDGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BRIDGE)


class _EncodingUtility:
    @staticmethod
    def get_set_encoding(values):
        ordered = sorted(values)
        return ({value: index for index, value in enumerate(ordered)}, ordered)


class MscnBridgeTest(unittest.TestCase):
    def test_dictionary_reconstruction_is_sorted_and_deterministic(self) -> None:
        metadata = {
            "table_keys": ["title t", "movie_info mi"],
            "column_keys": ["t.id", "mi.info_type_id"],
            "operator_keys": [">", "="],
            "join_keys": ["t.id=mi.movie_id"],
        }
        first = _BRIDGE._dictionaries(_EncodingUtility, metadata)
        second = _BRIDGE._dictionaries(_EncodingUtility, metadata)
        self.assertEqual(first, second)
        self.assertEqual(first[2], {"=": 0, ">": 1})

    def test_file_count_and_checksum_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            path.write_bytes(b"one\n\ntwo\n")
            self.assertEqual(_BRIDGE._line_count(path), 2)
            self.assertEqual(
                _BRIDGE._sha256(path),
                "ca48018cd69ec26f9206d26f99b5019ba6075d5482c587a072361987ae3179dd",
            )


if __name__ == "__main__":
    unittest.main()
