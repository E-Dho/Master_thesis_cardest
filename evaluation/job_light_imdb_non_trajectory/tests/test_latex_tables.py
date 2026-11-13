from __future__ import annotations

import json
import runpy
import tempfile
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
            "timing.gpu_single_query.mean_ms.mean": 2.0,
        }
        rows = MODULE["prepare_rows"]([
            dict(base, method_id="mscn", variant_id="m", **{"timing.cpu_1core.mean_ms.mean": 0.7}),
            dict(base, method_id="deepdb", variant_id="d", **{"timing.cpu_1core.mean_ms.mean": 9.3}),
        ], "job_light")
        table = MODULE["inference_table"](rows, "job_light", True, store)
        self.assertIn(r"MSCN$^{\circ}$", table)
        self.assertIn("job_light/inference/deepdb/d/cpu_mean_ms", store.best)
        self.assertNotIn("job_light/inference/mscn/m/cpu_mean_ms", store.best)

    def test_partial_standardized_timing_is_marked_and_explained(self) -> None:
        row = {
            "workload": "job_light", "method_id": "own_model", "variant_id": "v",
            "coverage_complete_all_seeds": True, "seed_count": 1,
            "timing.cpu_1core.mean_ms.mean": 1.0,
            "timing.gpu_single_query.mean_ms.mean": None,
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        table = MODULE["inference_table"](rows, "job_light", False, MODULE["DataStore"]())
        self.assertIn(r"$^{\P}$", table)
        self.assertIn("missing its standardized GPU timing profile", table)

    def test_missing_size_does_not_clear_selected_training_time(self) -> None:
        attach = MODULE["attach_build_metrics"]
        runtime = attach.__globals__
        variant = "test_selected_own_model"
        original_timing = runtime["_run_dirs_from_timing"]
        original_scan = runtime["_run_dirs_from_scan"]
        runtime["OWN_MODEL_VARIANTS"][variant] = {"label": "Test"}
        try:
            with tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                (run / "build_metrics.json").write_text(json.dumps({
                    "build": {"detail": {"bridge_metrics": {"status": "reused_checkpoint"}}},
                    "artifacts": {"selected_checkpoint_training_seconds": 123.0},
                }))
                runtime["_run_dirs_from_timing"] = lambda _row, _root: [run]
                runtime["_run_dirs_from_scan"] = lambda _row, _root: [run]
                row = {
                    "workload": "job_light", "method_id": "own_model", "variant_id": variant,
                }
                attach([row], run)
                self.assertEqual(row["_build.training_s.mean"], 123.0)
                self.assertIsNone(row["_build.size_mb.mean"])
        finally:
            runtime["_run_dirs_from_timing"] = original_timing
            runtime["_run_dirs_from_scan"] = original_scan
            del runtime["OWN_MODEL_VARIANTS"][variant]

    def test_selected_own_model_requires_size_and_training_records(self) -> None:
        validate = MODULE["validate_selected_own_model_build_metrics"]
        runtime = validate.__globals__
        variant = "test_selected_own_model"
        runtime["OWN_MODEL_VARIANTS"][variant] = {"label": "Test"}
        row = {"workload": "job_light", "method_id": "own_model", "variant_id": variant}
        try:
            with self.assertRaisesRegex(
                ValueError, "inference_state_dict_mb.*selected_checkpoint_training_seconds"
            ):
                validate([row])
            row["_build.has_inference_state_dict_mb"] = True
            with self.assertRaisesRegex(ValueError, "selected_checkpoint_training_seconds"):
                validate([row])
            row["_build.has_selected_checkpoint_training_seconds"] = True
            validate([row])
        finally:
            del runtime["OWN_MODEL_VARIANTS"][variant]


if __name__ == "__main__":
    unittest.main()
