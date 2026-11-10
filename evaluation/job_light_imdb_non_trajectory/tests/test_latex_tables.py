from __future__ import annotations

import runpy
import unittest
from pathlib import Path


MODULE = runpy.run_path(
    Path(__file__).parents[1] / "scripts" / "make_latex_tables.py"
)


class LatexTablesTest(unittest.TestCase):
    def test_rounds_before_selecting_decimal_places(self) -> None:
        self.assertEqual(MODULE["fmt_number"](9.997), "10.0")
        self.assertEqual(MODULE["fmt_number"](0.9997), "1.00")

    def test_preprocessing_column_is_never_marked_best(self) -> None:
        column = MODULE["Column"]("Preprocessing", lambda _: 0.0, lambda _: "$0$", mark_best=False)
        rows = [{"_eligible": True}]
        self.assertEqual(MODULE["best_rows"](rows, column), set())

    def test_combines_single_seed_and_missing_timing_markers(self) -> None:
        self.assertEqual(
            MODULE["_append_footnote_marker"](r"Own$^{\S}$", r"\ddagger"),
            r"Own$^{\S\ddagger}$",
        )


if __name__ == "__main__":
    unittest.main()
