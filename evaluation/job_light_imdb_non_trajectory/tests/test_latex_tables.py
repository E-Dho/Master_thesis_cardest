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
                "variant_id": "duet_neurocard_distjoin_rare_aux_step_11000_corrected_model_code",
                "seed_count": 3,
                "coverage_complete_all_seeds": True,
            },
        ], "job_light")
        self.assertEqual([row["_qerror_only"] for row in rows], [False, True])

    def test_adapted_mscn_uses_an_explicit_label(self) -> None:
        self.assertEqual(
            MODULE["row_label"]({
                "method_id": "mscn", "variant_id": "mscn_job_light_ranges_adapted",
                "protocol": "adapted",
            }),
            "MSCN (adapted)",
        )

    def test_printed_numbers_must_round_to_their_source(self) -> None:
        verify = MODULE["verify_printed"]
        # Correct rounding, including two significant digits on a deviation.
        verify(r"$4{,}257$\,{\scriptsize$\pm 240$}", [4257.2, 236.2])
        verify(r"$(1.33\,{\scriptstyle\pm 0.78})\cdot 10^{5}$", [1.327e5, 0.778e5])
        verify(r"$8.91$\,h", [32089.0], unit_scale=3600.0)
        # A value that does not round to what is printed is rejected.
        with self.assertRaises(ValueError):
            verify(r"$4{,}257$", [4300.0])
        with self.assertRaises(ValueError):
            verify(r"$8.91$\,h", [32089.0], unit_scale=60.0)  # wrong unit for the source
        with self.assertRaises(ValueError):
            verify(r"$1.24$", [1.24, 0.1])  # deviation missing from the printed text

    def test_formatters_verify_their_output(self) -> None:
        self.assertEqual(MODULE["fmt_cell"](23.4405, None), "$23.4$")
        self.assertEqual(MODULE["fmt_duration"](32089.0, None), r"$8.91$\,h")
        self.assertEqual(MODULE["fmt_int"](4205152), r"$4.21$\,M")

    def test_table_cells_are_keys_into_the_data_file(self) -> None:
        store = MODULE["DataStore"]()
        column = MODULE["Column"](
            "Median", lambda row: row["v"], lambda row: MODULE["fmt_cell"](row["v"]), key="p50"
        )
        rows = [
            {"method_id": "a", "variant_id": "x", "_eligible": True, "_label": "A", "v": 2.0},
            {"method_id": "b", "variant_id": "y", "_eligible": True, "_label": "B", "v": 1.5},
        ]
        table = MODULE["render_table"](
            rows, [column], caption="T", label="tab:t", data=store, key_prefix="w/t"
        )
        self.assertIn(r"\JLcell{w/t/a/x/p50}", table)
        self.assertNotIn("1.50", table)
        self.assertEqual(store.values["w/t/b/y/p50"], "$1.50$")
        self.assertEqual(store.best, {"w/t/b/y/p50"})
        with self.assertRaises(ValueError):
            store.add("w/t/b/y/p50", "$1.60$")

    def test_out_of_competition_rows_are_never_marked_best(self) -> None:
        store = MODULE["DataStore"]()
        base = {
            "workload": "job_light", "seed_count": 3, "coverage_complete_all_seeds": True,
            "timing.cpu_1core.mean_ms.mean": None,
        }
        rows = MODULE["prepare_rows"]([
            dict(base, method_id="mscn", variant_id="m", **{"timing.cpu_1core.mean_ms.mean": 0.7}),
            dict(base, method_id="deepdb", variant_id="d", **{"timing.cpu_1core.mean_ms.mean": 9.3}),
        ], "job_light")
        table = MODULE["inference_table"](rows, "job_light", True, store)
        self.assertIn(r"MSCN$^{\circ}$", table)
        self.assertIn("job_light/inference/deepdb/d/cpu_mean_ms", store.best)
        self.assertNotIn("job_light/inference/mscn/m/cpu_mean_ms", store.best)


if __name__ == "__main__":
    unittest.main()
