from __future__ import annotations

import json
import tempfile
import unittest
import importlib.util

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.importance_sampling import RareSupportSampleSource
from model.src.data.sample_sources import sample_source_from_config
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import RootDataStratum
from model.src.model.factorization import FactorizationConfig, apply_factorization_to_metadata
from model.src.predicates.generation import (
    PredicateTrainingContextGenerator,
    forced_predicate_for_stratum,
)
from model.src.predicates.operators import PredicateOp

if importlib.util.find_spec("torch") is not None:
    import torch

    from model.src.training.resmade_trainer import (
        auxiliary_eligibility_mask,
        main_predicate_rng_seed,
        rare_predicate_rng_seed,
        rare_row_rng_seed,
        train_resmade_sample_source,
    )
    from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy
else:
    torch = None
    auxiliary_eligibility_mask = None
    main_predicate_rng_seed = None
    rare_predicate_rng_seed = None
    rare_row_rng_seed = None
    train_resmade_sample_source = None
    torch_weighted_per_head_cross_entropy = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
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
            allow_row_dependent_native_range_tail=True,
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
            expected = forced_predicate_for_stratum(
                stratum,
                encoded_row=row,
                metadata=source.metadata,
                allow_row_dependent_native_range_tail=True,
            )
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
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertIn("rare_auxiliary", summary)
        self.assertIn(
            "whole_run_auxiliary_column_diagnostics",
            summary["rare_auxiliary"],
        )
        self.assertIn(
            "downstream_eligible_head_context_count_by_stratum",
            summary["rare_auxiliary"],
        )
        self.assertIn("total_rare_context_count", summary["rare_auxiliary"])
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

    def test_rng_streams_are_global_step_based_not_forward_call_based(self) -> None:
        generator_seed = 11
        training_seed = 23
        steps = range(4)
        main_draws = [
            np.random.default_rng(main_predicate_rng_seed(generator_seed, step)).integers(0, 1_000_000)
            for step in steps
        ]
        simulated_forward_calls_with_auxiliary = [step * 2 for step in steps]
        repeated_main_draws = [
            np.random.default_rng(main_predicate_rng_seed(generator_seed, step)).integers(0, 1_000_000)
            for step, _forward_calls in zip(steps, simulated_forward_calls_with_auxiliary)
        ]
        self.assertEqual(main_draws, repeated_main_draws)
        self.assertEqual(
            [main_predicate_rng_seed(generator_seed, step) for step in steps],
            [generator_seed + step for step in steps],
        )
        self.assertEqual(
            [rare_row_rng_seed(training_seed, step) for step in steps],
            [training_seed + 1_000_000 + step for step in steps],
        )
        self.assertEqual(
            [rare_predicate_rng_seed(generator_seed, step) for step in steps],
            [generator_seed + 2_000_000 + step for step in steps],
        )

    def test_independent_uniform_and_auxiliary_normalization(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("x", ColumnKind.DATA, (0, 1)),),
            full_join_cardinality=2,
        )
        uniform_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
        uniform_targets = torch.tensor([[0], [1]])
        uniform_weights = torch.tensor([[1.0], [3.0]])
        rare_logits = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
        rare_targets = torch.tensor([[0], [1]])
        rare_weights = torch.tensor([[5.0], [7.0]])
        beta = 0.25

        uniform = torch_weighted_per_head_cross_entropy(
            uniform_logits,
            uniform_targets,
            uniform_weights,
            metadata,
            head_loss_reduction="sum",
        )
        auxiliary = torch_weighted_per_head_cross_entropy(
            rare_logits,
            rare_targets,
            rare_weights,
            metadata,
            head_loss_reduction="sum",
        )
        total = uniform.total_loss + beta * auxiliary.total_loss

        uniform_ce = torch.nn.functional.cross_entropy(
            uniform_logits,
            uniform_targets[:, 0],
            reduction="none",
        )
        rare_ce = torch.nn.functional.cross_entropy(
            rare_logits,
            rare_targets[:, 0],
            reduction="none",
        )
        expected = (
            torch.sum(uniform_weights[:, 0] * uniform_ce) / torch.sum(uniform_weights[:, 0])
            + beta * torch.sum(rare_weights[:, 0] * rare_ce) / torch.sum(rare_weights[:, 0])
        )
        self.assertTrue(torch.allclose(total, expected))
        self.assertNotEqual(float(torch.sum(uniform_weights)), float(torch.sum(rare_weights)))

    def test_factorized_auxiliary_eligibility_uses_original_ar_index_for_all_factors(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("root", ColumnKind.DATA, tuple(range(4))),
                ColumnMetadata("large", ColumnKind.DATA, tuple(range(100))),
                ColumnMetadata("F_child", ColumnKind.FANOUT, (1, 2, 3)),
            ),
            full_join_cardinality=100,
        )
        metadata = apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=3,
                minimum_domain_size=8,
            ),
        )
        mask = auxiliary_eligibility_mask(metadata, np.array([0]), row_count=1)
        self.assertTrue(mask[0, 1])
        head_indices = metadata.factorization_plan.output_heads_for_column(1)
        self.assertGreater(len(head_indices), 1)
        for head_index in head_indices:
            self.assertEqual(
                metadata.factorization_plan.output_head_specs[head_index].source_column_index,
                1,
            )

    def test_native_range_tail_guard_requires_explicit_row_dependent_support(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("year", ColumnKind.DATA, (2015, 2016, 2017)),),
            full_join_cardinality=3,
        )
        row = np.array([2], dtype=int)
        lower_tail = RootDataStratum(
            stratum_id="year:ge:2015",
            column_index=0,
            column_name="year",
            region_type="lower_tail",
            lower=2015,
            support_bottleneck="native_range",
        )
        with self.assertRaisesRegex(ValueError, "row_dependent_native_range_tail"):
            forced_predicate_for_stratum(lower_tail, encoded_row=row, metadata=metadata)
        token = forced_predicate_for_stratum(
            lower_tail,
            encoded_row=row,
            metadata=metadata,
            allow_row_dependent_native_range_tail=True,
        )
        self.assertEqual(token.op, PredicateOp.RANGE)
        self.assertEqual((token.value, token.upper), (2015, 2017))

        singleton = RootDataStratum(
            stratum_id="year:eq:2016",
            column_index=0,
            column_name="year",
            region_type="equality",
            value=2016,
            support_bottleneck="native_range",
        )
        token = forced_predicate_for_stratum(
            singleton,
            encoded_row=np.array([1], dtype=int),
            metadata=metadata,
        )
        self.assertEqual(token.op, PredicateOp.RANGE)
        self.assertEqual((token.value, token.upper), (2016, 2016))


if __name__ == "__main__":
    unittest.main()
