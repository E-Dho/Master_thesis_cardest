from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import (
    SyntheticFullJoinSampleSource,
    canonicalize_fanout_value,
)
from model.src.data.neurocard_schema import configured_neurocard_use_cols
from model.src.data.sample_sources import sample_source_from_config
from model.src.predicates.vocabulary import PredicateVocabularies
from model.scripts.prepare_neurocard_data import _bypass_neurocard_prepare_on_cache_hit


class SamplerMetadataTest(unittest.TestCase):
    def test_preparation_bypasses_ray_prepare_when_cache_is_complete(self) -> None:
        original_prepare_calls = []
        prepare_utils = SimpleNamespace(
            check_required_files=lambda _spec: True,
            prepare=lambda spec: original_prepare_calls.append(spec),
        )
        factorized_sampler = SimpleNamespace(prepare_utils=prepare_utils)
        spec = object()

        self.assertTrue(_bypass_neurocard_prepare_on_cache_hit(factorized_sampler, spec))
        factorized_sampler.prepare_utils.prepare(spec)
        self.assertEqual(original_prepare_calls, [])

    def test_preparation_preserves_cache_miss_rebuild_path(self) -> None:
        original_prepare_calls = []
        prepare_utils = SimpleNamespace(
            check_required_files=lambda _spec: False,
            prepare=lambda spec: original_prepare_calls.append(spec),
        )
        factorized_sampler = SimpleNamespace(prepare_utils=prepare_utils)
        spec = object()

        self.assertFalse(_bypass_neurocard_prepare_on_cache_hit(factorized_sampler, spec))
        factorized_sampler.prepare_utils.prepare(spec)
        self.assertEqual(original_prepare_calls, [spec])

    def test_metadata_records_separate_input_and_output_bins(self) -> None:
        source = SyntheticFullJoinSampleSource()
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        self.assertEqual(source.metadata.data_output_bins[-1], 3)
        self.assertEqual(vocabularies.input_bins[-1], 2)
        self.assertNotEqual(vocabularies.input_bins[-1], source.metadata.data_output_bins[-1])

    def test_sampler_ordering_and_reproducibility(self) -> None:
        source = SyntheticFullJoinSampleSource()
        batch_a = source.batches(4, seed=7)
        batch_b = source.batches(4, seed=7)
        self.assertTrue(np.array_equal(batch_a.encoded_values, batch_b.encoded_values))
        kinds = [column.kind.value for column in batch_a.column_metadata]
        self.assertEqual(kinds, ["data", "data", "data", "indicator", "indicator", "indicator", "fanout", "fanout"])

    def test_indicator_and_fanout_validation(self) -> None:
        source = SyntheticFullJoinSampleSource()
        inspection = source.inspect()
        self.assertGreater(inspection.join_cardinality, 0)
        self.assertIn("I_A", inspection.indicator_frequencies)
        for minimum, maximum in inspection.fanout_min_max.values():
            self.assertGreater(minimum, 0)
            self.assertGreaterEqual(maximum, minimum)

    def test_known_outer_padding_fanout_is_neutral_one(self) -> None:
        self.assertEqual(canonicalize_fanout_value(None, outer_padding=True), 1)
        with self.assertRaises(ValueError):
            canonicalize_fanout_value(0)

    def test_resmade_configs_validate(self) -> None:
        for path in (
            "model/configs/resmade_smoke.yaml",
            "model/configs/resmade_inv_fanout_example.yaml",
            "model/configs/job_light_resmade_inv_fanout.yaml",
            "model/configs/resmade_factorized_smoke.yaml",
            "model/configs/job_light_resmade_factorized_smoke.yaml",
            "model/configs/job_light_resmade_factorized_anpm.yaml",
            "model/configs/job_light_duet_binary_native_anpm_smoke.yaml",
            "model/configs/job_light_duet_binary_native_anpm.yaml",
            "model/configs/job_light_duet_binary_native_anpm_10k_early_stop.yaml",
            "model/configs/job_light_duet_binary_native_anpm_20k_patience_3000.yaml",
            "model/configs/job_light_ranges_duet_binary_native_anpm_rare_auxiliary_smoke.yaml",
            "model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml",
        ):
            validate_config(load_simple_yaml(path))

    def test_factorized_config_validates_direct_io_source_kinds_and_parses_block_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/config.yaml"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "\n".join(
                        [
                            "model:",
                            "  type: predicate_resmade",
                            "  direct_io_connections: true",
                            "factorization:",
                            "  enabled: true",
                            "  strategy: bitwise_lossless",
                            "  blacklist_kinds:",
                            "    - indicator",
                            "    - fanout",
                            "anpm:",
                            "  enabled: true",
                            "inference:",
                            "  progressive_sampling: false",
                            "  factorized_decoder: anpm",
                        ]
                    )
                )
            config = load_simple_yaml(path)
        self.assertEqual(config["factorization"]["blacklist_kinds"], ["indicator", "fanout"])
        validate_config(config)
        config["model"]["direct_io_source_kinds"] = ["data", "future"]
        with self.assertRaises(ValueError):
            validate_config(config)
        config["model"]["direct_io_source_kinds"] = ["data", "indicator", "fanout"]
        config["model"]["direct_io_destination_kinds"] = ["data", "future"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_strict_production_predicate_probabilities_must_sum_to_one(self) -> None:
        config = load_simple_yaml("model/configs/job_light_duet_binary_native_anpm_smoke.yaml")
        validate_config(config)
        config["predicate_generation"]["equality_probability"] = 0.4
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_neurocard_column_projection_defaults_and_content_schema(self) -> None:
        self.assertEqual(configured_neurocard_use_cols({}), "simple")
        self.assertEqual(
            configured_neurocard_use_cols({"use_cols": "content"}),
            "content",
        )
        self.assertIsNone(configured_neurocard_use_cols({"use_cols": None}))
        with self.assertRaisesRegex(ValueError, "dataset.use_cols"):
            configured_neurocard_use_cols({"use_cols": "job_light_ranges"})

        config = load_simple_yaml("model/configs/job_light_duet_binary_native_anpm_smoke.yaml")
        config["dataset"]["use_cols"] = "content"
        validate_config(config)
        config["dataset"]["use_cols"] = "invalid"
        with self.assertRaisesRegex(ValueError, "dataset.use_cols"):
            validate_config(config)

    def test_live_sampler_receives_configured_content_projection(self) -> None:
        config = {
            "dataset": {
                "type": "neurocard_full_join",
                "sampling_mode": "live",
                "prepared_directory": "prepared",
                "csv_directory": "csv",
                "use_cols": "content",
                "sampler_batch_size": 4096,
                "sampler_seed": 7,
            },
            "factorization": {"enabled": False},
            "rare_support": {"enabled": False},
            "importance_sampling": {"enabled": False},
        }
        sentinel = object()
        with patch(
            "model.src.data.sample_sources.LiveNeuroCardFullJoinSampleSource",
            return_value=sentinel,
        ) as constructor:
            self.assertIs(sample_source_from_config(config), sentinel)
        constructor.assert_called_once_with(
            Path("prepared"),
            csv_directory=Path("csv"),
            neurocard_path=None,
            sampler_batch_size=4096,
            seed=7,
            startup_callback=None,
            use_cols="content",
        )


if __name__ == "__main__":
    unittest.main()
