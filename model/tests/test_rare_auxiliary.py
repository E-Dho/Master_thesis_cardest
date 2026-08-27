from __future__ import annotations

import tempfile
import unittest
import importlib.util

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.importance_sampling import RareSupportSampleSource
from model.src.data.sample_sources import sample_source_from_config
from model.src.predicates.generation import (
    PredicateTrainingContextGenerator,
    forced_predicate_for_stratum,
)
from model.src.predicates.operators import PredicateOp

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch is not installed")

from model.src.training.resmade_trainer import (
    auxiliary_eligibility_mask,
    train_resmade_sample_source,
)


class RareAuxiliaryTest(unittest.TestCase):
    def _config(self) -> dict:
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        config["model"]["hidden_sizes"] = [16]
        config["model"]["embedding_size"] = 4
        config["training"]["device"] = "cpu"
        config["training"]["batch_size"] = 8
        config["training"]["steps_per_epoch"] = 2
        config["training"]["checkpoint_interval_steps"] = 0
        config["training"]["validation_interval_steps"] = 1
        config["training"]["head_loss_reduction"] = "sum"
        config["predicate_generation"].update(
            {
                "strategy": "duet_batch_bounds",
                "table_subset_sampling": "neurocard_table_dropout_rooted",
                "enable_native_range_tokens": True,
                "normalize_predicate_probabilities": False,
                "wildcard_probability": 0.2,
                "equality_probability": 0.2,
                "lower_bound_probability": 0.2,
                "upper_bound_probability": 0.2,
                "native_range_probability": 0.2,
                "per_row_contexts": 1,
            }
        )
        config["importance_sampling"] = {"enabled": False}
        config["rare_support"] = {
            "enabled": True,
            "discovery": {
                "enabled": True,
                "root_data_only": True,
                "minimum_expected_context_support": 20,
                "max_selected_strata": 4,
                "support_planning_steps": 20,
            },
            "allocation": {"strategy": "support_deficit"},
            "diagnostics": {"enabled": False},
        }
        config["rare_auxiliary"] = {
            "enabled": True,
            "batch_size": 4,
            "beta": 0.25,
        }
        return config

    def test_main_batch_is_uniform_and_rare_batch_is_additional_without_rho(self) -> None:
        source = sample_source_from_config(self._config())
        self.assertIsInstance(source, RareSupportSampleSource)
        main = source.batches(8, seed=1)
        rare = source.rare_batches(4, seed=2)
        self.assertEqual(main.encoded_values.shape[0], 8)
        self.assertEqual(rare.encoded_values.shape[0], 4)
        self.assertIsNone(main.importance_weights)
        self.assertIsNone(rare.importance_weights)
        self.assertIn("selected_strata", rare.importance_metadata)

    def test_suffix_eligibility_uses_original_column_indices_and_includes_later_data(self) -> None:
        source = sample_source_from_config(self._config())
        metadata = source.metadata
        selected = np.zeros(3, dtype=int)
        mask = auxiliary_eligibility_mask(metadata, selected, row_count=3)
        self.assertFalse(mask[:, 0].any())
        self.assertTrue(mask[:, 1].all())
        self.assertTrue(mask[:, 2].all())
        later_data_columns = [
            index
            for index, column in enumerate(metadata.columns)
            if index > 0 and column.kind.value == "data"
        ]
        self.assertTrue(later_data_columns)
        self.assertTrue(mask[:, later_data_columns].all())

    def test_forced_stratum_predicates_are_exact_and_satisfied(self) -> None:
        source = sample_source_from_config(self._config())
        rare = source.rare_batches(8, seed=3)
        generator = PredicateTrainingContextGenerator(self._config()["predicate_generation"])
        contexts, rows, stats = generator.generate_forced_stratum_batch(
            encoded_rows=rare.encoded_values,
            metadata=source.metadata,
            strata=rare.importance_metadata["selected_strata"],
            rng=np.random.default_rng(4),
        )
        self.assertEqual(rows.shape[0], 8)
        self.assertEqual(stats.rejected_unsatisfied_contexts, 0)
        self.assertEqual(stats.included_indicator_contradictions, 0)
        for context, row, stratum in zip(
            contexts,
            rows,
            rare.importance_metadata["selected_strata"],
        ):
            token = context.tokens[int(stratum.column_index)]
            expected = forced_predicate_for_stratum(stratum, encoded_row=row, metadata=source.metadata)
            self.assertEqual(token.stable_key(), expected.stable_key())
            if stratum.support_bottleneck == "native_range":
                self.assertEqual(token.op, PredicateOp.RANGE)
            elif stratum.region_type == "equality":
                self.assertEqual(token.op, PredicateOp.EQUAL)
            elif stratum.region_type == "lower_tail":
                self.assertEqual(token.op, PredicateOp.GREATER_EQUAL)
                self.assertEqual(token.value, stratum.lower)
            elif stratum.region_type == "upper_tail":
                self.assertEqual(token.op, PredicateOp.LESS_EQUAL)
                self.assertEqual(token.value, stratum.upper)

    def test_training_reports_independently_normalized_auxiliary_loss(self) -> None:
        config = self._config()
        with tempfile.TemporaryDirectory() as output_directory:
            config["logging"]["output_directory"] = output_directory
            validate_config(config)
            source = sample_source_from_config(config)
            result = train_resmade_sample_source(source, config)
            summary = result.summary_path.read_text(encoding="utf-8")
        self.assertIn('"rare_auxiliary"', summary)
        self.assertGreater(result.total_sampled_tuples, 0)

    def test_beta_zero_skips_rare_sampling_and_preserves_no_rho_main_path(self) -> None:
        config = self._config()
        config["rare_auxiliary"]["beta"] = 0.0
        source = sample_source_from_config(config)
        with tempfile.TemporaryDirectory() as output_directory:
            config["logging"]["output_directory"] = output_directory
            result = train_resmade_sample_source(source, config)
        self.assertEqual(source.rare_rows_drawn, 0)
        self.assertEqual(result.total_sampled_tuples, config["training"]["batch_size"] * 2)


if __name__ == "__main__":
    unittest.main()
