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

    def test_timing_config_reader_resolves_one_level_of_inheritance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "experiment:\n  method_id: own_model\n  variant_id: base\n"
                "timing:\n  profiles:\n    - cpu_1core\n",
                encoding="utf-8",
            )
            child = root / "child.yaml"
            child.write_text(
                "extends: base.yaml\nexperiment:\n  variant_id: child\n"
                "timing:\n  profiles:\n    - cpu_1core\n    - gpu_single_query\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE["_timing_config_identity"](child),
                ("own_model", "child", ("cpu_1core", "gpu_single_query")),
            )

    def test_preprocessing_column_is_never_marked_best(self) -> None:
        column = MODULE["Column"]("Preprocessing", lambda _: 0.0, lambda _: "$0$", mark_best=False)
        rows = [{"_eligible": True}]
        self.assertEqual(MODULE["best_rows"](rows, column), set())

    def test_combines_single_seed_and_missing_timing_markers(self) -> None:
        self.assertEqual(
            MODULE["_append_footnote_marker"](r"Own$^{\S}$", r"\ddagger"),
            r"Own$^{\S\ddagger}$",
        )

    def test_rule_based_own_model_row_is_complete_across_three_seeds(self) -> None:
        rows = MODULE["prepare_rows"]([
            {
                "workload": "job_light",
                "method_id": "own_model",
                "variant_id": "duet_neurocard_distjoin_rare_aux_rule_step_11000_corrected_model_code",
                "seed_count": 3,
                "coverage_complete_all_seeds": True,
            },
        ], "job_light")
        self.assertEqual([row["_qerror_only"] for row in rows], [False])
        self.assertNotIn(r"$^{\S}$", rows[0]["_label"])

    def test_rule_rows_do_not_add_a_selection_note(self) -> None:
        row = {
            "workload": "job_light",
            "method_id": "own_model",
            "variant_id": "duet_neurocard_distjoin_rare_aux_rule_step_11000_corrected_model_code",
            "seed_count": 3,
            "coverage_complete_all_seeds": True,
            "smoothed_q_error.p50.mean": 1.0,
            "smoothed_q_error.p50.std": 0.0,
            "smoothed_q_error.p90.mean": 1.0,
            "smoothed_q_error.p90.std": 0.0,
            "smoothed_q_error.p95.mean": 1.0,
            "smoothed_q_error.p95.std": 0.0,
            "smoothed_q_error.p99.mean": 1.0,
            "smoothed_q_error.p99.std": 0.0,
            "smoothed_q_error.max.mean": 1.0,
            "smoothed_q_error.max.std": 0.0,
        }
        table = MODULE["qerror_table"](
            MODULE["prepare_rows"]([row], "job_light"), "job_light",
            "smoothed_q_error", False, MODULE["DataStore"](),
        )
        self.assertNotIn("step-11,000", table)

    def test_ranges_excludes_native_deepdb_but_keeps_adapted(self) -> None:
        rows = MODULE["prepare_rows"]([
            {"workload": "job_light_ranges", "method_id": "deepdb",
             "variant_id": "imdb_light_rdc_published", "coverage_complete_all_seeds": True},
            {"workload": "job_light_ranges", "method_id": "deepdb",
             "variant_id": "imdb_light_ranges_adapted", "protocol": "adapted",
             "coverage_complete_all_seeds": True},
        ], "job_light_ranges")
        self.assertEqual([row["variant_id"] for row in rows], ["imdb_light_ranges_adapted"])

    def test_interim_ranges_rule_uses_old_accuracy_and_current_timing(self) -> None:
        legacy = {
            "workload": "job_light_ranges", "method_id": "own_model",
            "variant_id": "duet_neurocard_distjoin_rare_aux_content_step_11000",
            "seed_count": 3, "coverage_complete_all_seeds": True,
            "smoothed_q_error.p50.mean": 2.0,
        }
        timing = {
            "workload": "job_light_ranges", "method_id": "own_model",
            "variant_id": "duet_neurocard_distjoin_rare_aux_best_median_step_40000_corrected_model_code",
            "seed_count": 1, "coverage_complete_all_seeds": True,
            "timing.cpu_1core.mean_ms.mean": 10.0,
            "timing.gpu_single_query.mean_ms.mean": 3.0,
            "_configured_timing_profiles": ("cpu_1core", "gpu_single_query"),
        }
        rows = MODULE["prepare_rows"](
            [legacy, timing], "job_light_ranges", interim_ranges_accuracy=legacy,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["variant_id"],
            "duet_neurocard_distjoin_rare_aux_content_rule_step_11000_corrected_model_code",
        )
        self.assertEqual(rows[0]["smoothed_q_error.p50.mean"], 2.0)
        self.assertEqual(rows[0]["timing.cpu_1core.mean_ms.mean"], 10.0)

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

    def test_cpu_only_declared_profile_is_not_marked_missing(self) -> None:
        row = {
            "workload": "job_light", "method_id": "postgresql", "variant_id": "v",
            "coverage_complete_all_seeds": True, "seed_count": 1,
            "_configured_timing_profiles": ("cpu_1core",),
            "timing.cpu_1core.mean_ms.mean": 1.0,
            "timing.gpu_single_query.mean_ms.mean": None,
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        table = MODULE["inference_table"](rows, "job_light", False, MODULE["DataStore"]())
        self.assertNotIn(r"$^{\P}$", table)
        self.assertIn("no implementation for this device", table)

    def test_declared_but_unmeasured_standardized_profile_is_marked(self) -> None:
        row = {
            "workload": "job_light", "method_id": "own_model", "variant_id": "v",
            "coverage_complete_all_seeds": True, "seed_count": 1,
            "_configured_timing_profiles": ("cpu_1core", "gpu_single_query"),
            "timing.cpu_1core.mean_ms.mean": None,
            "timing.gpu_single_query.mean_ms.mean": 1.0,
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        table = MODULE["inference_table"](rows, "job_light", False, MODULE["DataStore"]())
        self.assertIn(r"$^{\P}$", table)
        self.assertIn("missing its standardized CPU timing profile", table)

    def test_distjoin_cpu_consistency_note_uses_timing_summary_bound(self) -> None:
        row = {
            "workload": "job_light", "method_id": "distjoin", "variant_id": "published",
            "coverage_complete_all_seeds": True, "seed_count": 3,
            "_configured_timing_profiles": ("cpu_1core", "gpu_single_query"),
            "timing.cpu_1core.mean_ms.mean": 100.0,
            "timing.gpu_single_query.mean_ms.mean": 10.0,
            "standardized_timing": {
                "cpu_1core": {
                    "estimate_consistency": [
                        {"seed": 0, "by_workload": {"job_light": {
                            "median_relative_difference": 0.001,
                            "max_relative_difference": 0.015,
                        }}},
                        {"seed": 1, "by_workload": {"job_light": {
                            "median_relative_difference": 0.003,
                            "max_relative_difference": 0.5,
                        }}},
                    ],
                },
            },
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        table = MODULE["inference_table"](rows, "job_light", False, MODULE["DataStore"]())
        self.assertIn("GPU runs use mixed precision", table)
        self.assertIn("q-errors come from the GPU runs", table)
        self.assertIn(r"\JLval{job_light/inference/distjoin/published/cpu_accuracy_median_relative_difference}", table)
        self.assertIn(r"\JLval{job_light/inference/distjoin/published/cpu_accuracy_max_relative_difference}", table)

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

    def test_interim_build_record_aggregates_three_checkpoints_into_one_cpu_run(self) -> None:
        refresh = MODULE["refresh_selected_own_model_build_records"]
        attach = MODULE["attach_build_metrics"]
        runtime = refresh.__globals__
        variant = "test_interim_selected_own_model"
        originals = {
            "timing": runtime["_run_dirs_from_timing"],
            "scan": runtime["_run_dirs_from_scan"],
            "metadata": runtime["_selected_own_model_metadata"],
            "training": runtime["_selected_own_model_training_seconds"],
        }
        runtime["OWN_MODEL_VARIANTS"][variant] = {"label": "Test", "training_time_mode": "checkpoint"}
        try:
            with tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                (run / "build_metrics.json").write_text("{}", encoding="utf-8")
                runtime["_run_dirs_from_timing"] = lambda _row, _root: [run]
                runtime["_run_dirs_from_scan"] = lambda _row, _root: [run]
                runtime["_selected_own_model_metadata"] = lambda _row: [
                    {"inference_state_dict_mb": 10.0},
                    {"inference_state_dict_mb": 20.0},
                    {"inference_state_dict_mb": 30.0},
                ]
                runtime["_selected_own_model_training_seconds"] = lambda _row: (90.0, 9.0)
                row = {"workload": "job_light_ranges", "method_id": "own_model",
                       "variant_id": variant}
                self.assertEqual(refresh([row], run), 1)
                document = json.loads((run / "build_metrics.json").read_text())
                artifact = document["artifacts"]
                self.assertEqual(artifact["inference_state_dict_mb"], 20.0)
                self.assertEqual(artifact["inference_state_dict_mb_std"], 10.0)
                self.assertEqual(artifact["selected_checkpoint_training_seconds"], 90.0)
                self.assertEqual(artifact["selected_checkpoint_training_seconds_std"], 9.0)
                attach([row], run)
                self.assertEqual(row["_build.size_mb.std"], 10.0)
                self.assertEqual(row["_build.training_s.std"], 9.0)
        finally:
            runtime["_run_dirs_from_timing"] = originals["timing"]
            runtime["_run_dirs_from_scan"] = originals["scan"]
            runtime["_selected_own_model_metadata"] = originals["metadata"]
            runtime["_selected_own_model_training_seconds"] = originals["training"]
            del runtime["OWN_MODEL_VARIANTS"][variant]


if __name__ == "__main__":
    unittest.main()
