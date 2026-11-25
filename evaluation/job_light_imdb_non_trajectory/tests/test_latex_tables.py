from __future__ import annotations

import json
import io
import importlib.util
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

    def test_rounded_baselines_hide_sub_one_until_raw_diagnostic_exists(self) -> None:
        missing = {
            "method_id": "neurocard", "estimate_lt_1_count.mean": 0.0,
        }
        complete = dict(
            missing,
            **{
                "unrounded_estimate_count.mean": 70.0,
                "query_count.mean": 70.0,
                "estimate_lt_1_count.mean": 9.0,
            },
        )
        self.assertIsNone(MODULE["_sub_one_value"](missing))
        self.assertEqual(MODULE["_sub_one_value"](complete), 9.0)

    def test_estimates_below_one_column_is_never_bold(self) -> None:
        column = MODULE["Column"](
            "Est.", lambda _: 0.0, lambda _: "$0$", mark_best=False,
        )
        self.assertEqual(MODULE["best_rows"]([{"_eligible": True}], column), set())

    def test_qerror_notes_use_math_mode_for_estimates_below_one(self) -> None:
        row = {
            "workload": "job_light", "method_id": "own_model", "variant_id": "v",
            "seed_count": 3, "coverage_complete_all_seeds": True,
            "estimate_lt_1_count.mean": 1.0, "estimate_lt_1_count.std": 0.0,
        }
        for family in ("raw_q_error", "smoothed_q_error"):
            for metric in ("p50", "p90", "p95", "p99", "max"):
                row[f"{family}.{metric}.mean"] = 1.0
                row[f"{family}.{metric}.std"] = 0.0
        table = MODULE["qerror_table"](
            MODULE["prepare_rows"]([row], "job_light"), "job_light",
            "raw_q_error", False, MODULE["DataStore"](),
        )
        self.assertNotIn("Est. < 1", table)
        self.assertNotIn("Bold: best per column", table)
        self.assertIn(r"Est.\,$<1$ counts them before rounding.", table)
        self.assertNotIn("Est. < 1", table)

    def test_own_model_training_table_uses_the_sped_up_placeholder(self) -> None:
        row = {
            "workload": "job_light", "method_id": "own_model", "variant_id": "v",
            "seed_count": 3, "coverage_complete_all_seeds": True,
            "_build.preprocessing_s.mean": 0.27,
            "_build.preprocessing_s.std": 0.02,
            "_build.training_s.mean": 32089.0,
            "_build.training_s.std": 20.0,
            "_build.total_s.mean": 32089.27,
            "_build.total_s.std": 20.0,
        }
        store = MODULE["DataStore"]()
        MODULE["training_table"](
            MODULE["prepare_rows"]([row], "job_light"), "job_light", True, store,
        )
        prefix = "job_light/training/own_model/v"
        self.assertEqual(store.values[f"{prefix}/training"], r"$1.07$\,h")
        self.assertEqual(store.values[f"{prefix}/total"], r"$1.07$\,h")
        self.assertEqual(
            store.values[f"{prefix}/preprocessing"],
            r"$0.270$\,{\scriptsize$\pm 0.020$}\,s",
        )

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
            {"method_id": "deepdb", "variant_id": "x", "_eligible": True, "_label": "A", "v": 2.0},
            {"method_id": "neurocard", "variant_id": "y", "_eligible": True, "_label": "B", "v": 1.5},
        ]
        table = MODULE["render_table"](
            rows, [column], caption="T", label="tab:t", data=store, key_prefix="w/t"
        )
        self.assertIn(r"\JLcell{w/t/deepdb/x/p50}", table)
        self.assertNotIn("1.50", table)
        self.assertEqual(store.values["w/t/neurocard/y/p50"], "$1.50$")
        self.assertEqual(store.best, {"w/t/neurocard/y/p50"})
        self.assertEqual(store.under, {"w/t/neurocard/y/p50"})
        with self.assertRaises(ValueError):
            store.add("w/t/neurocard/y/p50", "$1.60$")

    def test_mscn_can_be_underlined_but_not_bold(self) -> None:
        store = MODULE["DataStore"]()
        base = {
            "workload": "job_light", "seed_count": 3, "coverage_complete_all_seeds": True,
            "timing.cpu_1core.mean_ms.mean": None,
            "timing.gpu_single_query.mean_ms.mean": 2.0,
        }
        rows = MODULE["prepare_rows"]([
            dict(base, method_id="mscn", variant_id="m", **{"timing.cpu_1core.mean_ms.mean": 0.7}),
            dict(base, method_id="deepdb", variant_id="d", **{"timing.cpu_1core.mean_ms.mean": 9.3}),
            dict(base, method_id="neurocard", variant_id="n", **{"timing.cpu_1core.mean_ms.mean": 12.0}),
        ], "job_light")
        table = MODULE["inference_table"](rows, "job_light", True, store)
        self.assertIn("MSCN &", table)
        self.assertIn("job_light/inference/deepdb/d/gpu_mean_ms", store.best)
        self.assertNotIn("job_light/inference/mscn/m/gpu_mean_ms", store.best)
        self.assertIn("job_light/inference/mscn/m/cpu_mean_ms", store.under)

    def test_cpu_only_declared_profile_is_not_marked_missing(self) -> None:
        row = {
            "workload": "job_light", "method_id": "postgresql", "variant_id": "v",
            "coverage_complete_all_seeds": True, "seed_count": 1,
            "_configured_timing_profiles": ("cpu_1core",),
            "timing.cpu_1core.mean_ms.mean": 1.0,
            "timing.gpu_single_query.mean_ms.mean": None,
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        store = MODULE["DataStore"]()
        table = MODULE["inference_table"](rows, "job_light", False, store)
        self.assertNotIn(r"$^{\P}$", table)
        self.assertNotIn("no implementation for this device", table)

    def test_declared_but_unmeasured_standardized_profile_is_marked(self) -> None:
        row = {
            "workload": "job_light", "method_id": "own_model", "variant_id": "v",
            "coverage_complete_all_seeds": True, "seed_count": 1,
            "_configured_timing_profiles": ("cpu_1core", "gpu_single_query"),
            "timing.cpu_1core.mean_ms.mean": None,
            "timing.gpu_single_query.mean_ms.mean": 1.0,
        }
        rows = MODULE["prepare_rows"]([row], "job_light")
        store = MODULE["DataStore"]()
        table = MODULE["inference_table"](rows, "job_light", False, store)
        self.assertIn(r"$^{\P}$", table)
        self.assertIn("missing its standardized CPU timing profile", table)

    def test_distjoin_cpu_consistency_note_is_not_rendered(self) -> None:
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
        store = MODULE["DataStore"]()
        table = MODULE["inference_table"](rows, "job_light", False, store)
        self.assertNotIn("GPU runs use mixed precision", table)
        self.assertNotIn("q-errors come from the GPU runs", table)
        self.assertFalse(any("cpu_accuracy_" in key for key in store.values))

    def test_marking_requires_two_distinct_data_driven_methods(self) -> None:
        column = MODULE["Column"]("Metric", lambda row: row["v"], lambda row: MODULE["fmt_cell"](row["v"]))
        rows = [{"method_id": "own_model", "_eligible": True, "v": 1.0}]
        self.assertEqual(MODULE["marked_rows"](rows, column, MODULE["DATA_DRIVEN_METHODS"]), set())

    def test_marking_normalizes_printed_duration_units(self) -> None:
        column = MODULE["Column"](
            "Training", lambda row: row["seconds"],
            lambda row: MODULE["fmt_duration"](row["seconds"], None),
        )
        rows = [
            {"method_id": "own_model", "_eligible": True, "seconds": 121.0},
            {"method_id": "deepdb", "_eligible": True, "seconds": 119.0},
        ]
        self.assertEqual(MODULE["marked_rows"](rows, column, MODULE["DATA_DRIVEN_METHODS"]), {1})

    def test_sampling_rows_are_never_marked(self) -> None:
        column = MODULE["Column"]("Metric", lambda row: row["v"], lambda row: MODULE["fmt_cell"](row["v"]))
        rows = [
            {"method_id": "own_model", "_eligible": True, "v": 2.0},
            {"method_id": "neurocard", "_eligible": True, "v": 3.0},
            {"method_id": "foj_sampling", "_eligible": True, "v": 0.1},
        ]
        self.assertEqual(MODULE["marked_rows"](rows, column, MODULE["DATA_DRIVEN_METHODS"]), {0})

    def test_inference_cpu_gpu_of_one_method_counts_once(self) -> None:
        columns = [
            MODULE["Column"](f"cpu_{index}", lambda row, i=index: row["cpu"][i],
                             lambda row, i=index: MODULE["fmt_cell"](row["cpu"][i]))
            for index in range(4)
        ] + [
            MODULE["Column"](f"gpu_{index}", lambda row, i=index: row["gpu"][i],
                             lambda row, i=index: MODULE["fmt_cell"](row["gpu"][i]))
            for index in range(4)
        ]
        rows = [{"method_id": "own_model", "_eligible": True,
                 "cpu": [10.0] * 4, "gpu": [1.0] * 4}]
        self.assertEqual(MODULE["inference_marks"](rows, columns), [(set(), set())] * 8)

    def test_ties_and_dagger_rows_follow_marking_rules(self) -> None:
        column = MODULE["Column"]("Metric", lambda row: row["v"], lambda row: MODULE["fmt_cell"](row["v"]))
        tied = [
            {"method_id": "own_model", "_eligible": True, "v": 1.0},
            {"method_id": "deepdb", "_eligible": True, "v": 1.0},
        ]
        self.assertEqual(MODULE["marked_rows"](tied, column, MODULE["DATA_DRIVEN_METHODS"]), {0, 1})
        partial = [
            {"method_id": "own_model", "_eligible": False, "v": 0.1},
            {"method_id": "deepdb", "_eligible": True, "v": 1.0},
            {"method_id": "neurocard", "_eligible": True, "v": 2.0},
        ]
        self.assertEqual(MODULE["marked_rows"](partial, column, MODULE["DATA_DRIVEN_METHODS"]), {1})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required for checkpoint measurement")
    def test_inference_checkpoint_retains_domains_and_never_changes_source(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            payload = {
                "model_state_dict": {"weight": torch.arange(6).reshape(2, 3),
                                     "mask": torch.ones(2, 3, dtype=torch.bool)},
                "metadata": {"columns": [{"name": "x", "kind": "data", "domain": [1, 2, 3]}]},
                "predicate_vocabularies": {"x": {"equal": [1, 2, 3]}},
                "optimizer_state_dict": {"state": {0: {"exp_avg": torch.ones(2, 3)}}},
                "step": 11000,
            }
            torch.save(payload, path)
            before, stat = path.read_bytes(), path.stat()
            expected = dict(payload, optimizer_state_dict=None)
            buffer = io.BytesIO()
            torch.save(expected, buffer)
            measured = MODULE["_inference_checkpoint_metadata"](path)
            self.assertEqual(measured["inference_checkpoint_mb"], buffer.getbuffer().nbytes / 1_000_000)
            self.assertGreater(measured["inference_checkpoint_mb"], measured["state_dict_mb"])
            self.assertEqual(measured["full_checkpoint_mb"], len(before) / 1_000_000)
            self.assertNotIn("parameter_count", measured)
            self.assertNotIn("optimizer_state_dict", measured["breakdown"]["fields_bytes"])
            self.assertEqual(measured["breakdown"]["largest_columns"][0]["domain_size"], 3)
            self.assertIn("metadata", measured["breakdown"]["fields_bytes"])
            self.assertIn("predicate_vocabularies", measured["breakdown"]["fields_bytes"])
            self.assertEqual(measured["model_size_source"],
                             "torch.save(checkpoint without optimizer_state_dict)")
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
            self.assertEqual(path.stat().st_mode, stat.st_mode)

    def test_model_size_note_uses_full_inference_checkpoint_definition(self) -> None:
        for workload in ("job_light", "job_light_ranges"):
            row = {"method_id": "own_model", "variant_id": "test", "_eligible": True,
                   "_build_runs": 1, "_build.size_is_inference_checkpoint": True, "_label": "Own model"}
            table = MODULE["model_size_table"]([row], workload, True, MODULE["DataStore"]())
            self.assertIn(r"Size: everything needed at inference except the base data (\OwnModelName{}: checkpoint without optimizer state, including the value domains).", table)
            self.assertIn(r"MB = $10^6$ bytes.", table)

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
                ValueError, "inference_checkpoint_mb.*selected_checkpoint_training_seconds"
            ):
                validate([row])
            row["_build.has_inference_checkpoint_mb"] = True
            with self.assertRaisesRegex(ValueError, "selected_checkpoint_training_seconds"):
                validate([row])
            row["_build.has_selected_checkpoint_training_seconds"] = True
            validate([row])
        finally:
            del runtime["OWN_MODEL_VARIANTS"][variant]

    def test_refresh_overrides_are_in_memory_and_preserve_existing_build_record(self) -> None:
        refresh = MODULE["refresh_selected_own_model_build_records"]
        runtime = refresh.__globals__
        variant = "test_readonly_size_refresh"
        originals = {k: runtime[k] for k in ["_run_dirs_from_timing", "_selected_own_model_metadata", "_selected_own_model_training_seconds"]}
        runtime["OWN_MODEL_VARIANTS"][variant] = {"training_time_mode": "checkpoint"}
        try:
            with tempfile.TemporaryDirectory() as directory:
                run = Path(directory); path = run / "build_metrics.json"
                path.write_text('{}\n');before = path.read_bytes();mtime = path.stat().st_mtime_ns
                runtime["_run_dirs_from_timing"] = lambda _r, _root: [run]
                runtime["_selected_own_model_metadata"] = lambda _r: [{"inference_checkpoint_mb": 32.0}]
                runtime["_selected_own_model_training_seconds"] = lambda _r: (90.0, 0.0)
                overrides = {}
                row = {"workload": "job_light", "method_id": "own_model", "variant_id": variant}
                self.assertEqual(refresh([row], run, overrides), 1)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(path.stat().st_mtime_ns, mtime)
                self.assertEqual(overrides[str(run)]["artifacts"]["inference_checkpoint_mb"], 32.0)
                MODULE["attach_build_metrics"]([row], run, overrides)
                self.assertEqual(row["_build.size_mb.mean"], 32.0)
        finally:
            runtime.update(originals)
            del runtime["OWN_MODEL_VARIANTS"][variant]

    def test_size_override_changes_only_size_and_distjoin_parameter_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run = root / "distjoin/v/seed_0/run";run.mkdir(parents=True)
            (run / "build_metrics.json").write_text('{}')
            row = {"method_id": "distjoin", "variant_id": "v", "workload": "job_light",
                   "_build.params.mean": 221.0, "_build.size_mb.mean": 1200.0,
                   "_build.training_s.mean": 17.0, "smoothed_q_error.p50.mean": 1.23}
            record = {"inference_artifact_mb": 60.0, "parameter_count": 10.0,
                      "inference_files": [], "inference_metadata_sources": []}
            MODULE["apply_inference_size_records"]([row], {"distjoin/v/seed_0": record}, root)
            self.assertEqual(row["_build.size_mb.mean"], 60.0)
            self.assertEqual(row["_build.params.mean"], 10.0)
            self.assertEqual(row["_build.training_s.mean"], 17.0)
            self.assertEqual(row["smoothed_q_error.p50.mean"], 1.23)
            self.assertEqual((run / "build_metrics.json").read_text(), '{}')

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
                    {"inference_checkpoint_mb": 10.0},
                    {"inference_checkpoint_mb": 20.0},
                    {"inference_checkpoint_mb": 30.0},
                ]
                runtime["_selected_own_model_training_seconds"] = lambda _row: (90.0, 9.0)
                row = {"workload": "job_light_ranges", "method_id": "own_model",
                       "variant_id": variant}
                overrides = {}
                self.assertEqual(refresh([row], run, overrides), 1)
                document = overrides[str(run)]
                self.assertEqual((run / "build_metrics.json").read_text(), "{}")
                artifact = document["artifacts"]
                self.assertEqual(artifact["inference_checkpoint_mb"], 20.0)
                self.assertEqual(artifact["inference_checkpoint_mb_std"], 10.0)
                self.assertEqual(len(artifact["selected_checkpoint_metadata"]), 3)
                self.assertEqual(artifact["selected_checkpoint_training_seconds"], 90.0)
                self.assertEqual(artifact["selected_checkpoint_training_seconds_std"], 9.0)
                attach([row], run, overrides)
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
