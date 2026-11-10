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

    def test_best_p99_selection_is_qerror_only(self) -> None:
        rows = MODULE["prepare_rows"]([
            {
                "workload": "job_light",
                "method_id": "own_model",
                "variant_id": "duet_neurocard_distjoin_rare_aux_best_median_step_18000",
                "seed_count": 1,
                "coverage_complete_all_seeds": True,
            },
            {
                "workload": "job_light",
                "method_id": "own_model",
                "variant_id": "duet_neurocard_distjoin_rare_aux_step_11000",
                "seed_count": 3,
                "coverage_complete_all_seeds": True,
            },
        ], "job_light")
        self.assertEqual([row["_qerror_only"] for row in rows], [False, True])


if __name__ == "__main__":
    unittest.main()
