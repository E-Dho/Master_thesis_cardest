from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import FullJoinBatch, SyntheticDataset
from model.src.data.importance_sampling import ImportanceSamplingSampleSource, _sampler_counters
from model.src.data.sample_sources import sample_source_from_config
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import (
    ExactRootStratumProvider,
    RootDataStratum,
    SQL_NULL,
    _expected_native_range_interval_counts,
    membership_matrix,
    rho_for_memberships,
)
from model.src.predicates.generation import PredicateTrainingContextGenerator
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.training.losses import (
    cumulative_inverse_fanout_weights,
    importance_weights_for_generated_contexts,
    stable_combine_importance_and_inverse_weights,
)


class _MaterializedSource:
    def __init__(self, dataset: SyntheticDataset) -> None:
        self.dataset = dataset
        self._metadata = dataset.metadata

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def join_cardinality(self) -> int:
        return int(self._metadata.full_join_cardinality)

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.dataset.encoded_rows), size=batch_size)
        return FullJoinBatch(
            encoded_values=self.dataset.encoded_rows[indices],
            column_metadata=self.metadata.columns,
            fixture_rows_reused=batch_size,
        )


class ImportanceSamplingTest(unittest.TestCase):
    def _numeric_source(self) -> _MaterializedSource:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("A.x", ColumnKind.DATA, (0, 1, 2, 3), table="A"),
                ColumnMetadata("I_A", ColumnKind.INDICATOR, (0, 1), table="A"),
                ColumnMetadata("F_A", ColumnKind.FANOUT, (1, 2, 10), table="A", fanout_source="A->A"),
            ),
            full_join_cardinality=10,
            join_root="A",
            join_tables=("A",),
            join_edges=(),
        )
        decoded = (
            (0, 1, 1),
            (0, 1, 1),
            (0, 1, 1),
            (0, 1, 1),
            (0, 1, 1),
            (1, 1, 2),
            (1, 1, 2),
            (2, 1, 10),
            (2, 1, 10),
            (3, 1, 10),
        )
        encoded = np.array(
            [[metadata.columns[i].encode_value(value) for i, value in enumerate(row)] for row in decoded],
            dtype=np.int64,
        )
        return _MaterializedSource(SyntheticDataset(metadata, decoded, encoded))

    def _config(self) -> dict:
        return {
            "training": {"batch_size": 1000, "steps_per_epoch": 1, "epochs": 1, "seed": 0},
            "predicate_generation": {
                "per_row_contexts": 1,
                "wildcard_probability": 0.2,
                "equality_probability": 0.2,
                "lower_bound_probability": 0.2,
                "upper_bound_probability": 0.2,
                "native_range_probability": 0.2,
            },
            "importance_sampling": {
                "enabled": True,
                "mixture_probability": 0.25,
                "discovery": {
                    "enabled": True,
                    "root_data_only": True,
                    "minimum_expected_context_support": 100,
                    "max_selected_strata": 4,
                    "root_column_semantics": {"A.x": "ordered"},
                },
                "allocation": {"strategy": "support_deficit"},
            },
        }

    def test_lambda_zero_yields_unit_rho(self) -> None:
        strata = (
            RootDataStratum(
                "A.x:eq:3",
                0,
                "A.x",
                "equality",
                value=3,
                foj_count=1,
                probability=0.1,
                alpha=1.0,
            ),
        )
        memberships = np.array([[False], [True], [True]])
        self.assertTrue(np.allclose(rho_for_memberships(memberships, strata, 0.0), 1.0))

    def test_overlapping_strata_use_membership_sum(self) -> None:
        strata = (
            RootDataStratum("eq", 0, "A.x", "equality", value=3, probability=0.1, alpha=0.4),
            RootDataStratum("ge", 0, "A.x", "lower_tail", lower=2, probability=0.3, alpha=0.6),
        )
        rho = rho_for_memberships(np.array([[True, True]]), strata, 0.2)[0]
        expected = 1.0 / (0.8 + 0.2 * (0.4 / 0.1 + 0.6 / 0.3))
        self.assertAlmostEqual(rho, expected)

    def test_discovery_selects_positive_mass_strata_and_alpha_sums_to_one(self) -> None:
        source = self._numeric_source()
        provider = ExactRootStratumProvider.from_encoded_rows(
            source.metadata,
            source.dataset.encoded_rows,
            root_column_semantics={"A.x": "ordered"},
        )
        strata = provider.discover(
            n_total=1000,
            predicate_probabilities={
                "equality": 0.2,
                "lower": 0.2,
                "upper": 0.2,
                "range": 0.2,
            },
            minimum_expected_context_support=100,
            max_selected_strata=4,
        )
        self.assertTrue(strata)
        self.assertAlmostEqual(sum(stratum.alpha for stratum in strata), 1.0)
        self.assertTrue(all(stratum.probability > 0.0 for stratum in strata))

    def test_proposal_frequencies_and_importance_correction(self) -> None:
        source = self._numeric_source()
        wrapped = ImportanceSamplingSampleSource(source, self._config())
        batch = wrapped.batches(20000, seed=9)
        x = batch.encoded_values[:, 0]
        values = np.array([source.metadata.columns[0].domain[int(index)] for index in x])
        rho = batch.importance_weights
        self.assertIsNotNone(rho)
        self.assertTrue(np.all(np.isfinite(rho)))
        self.assertAlmostEqual(float(np.mean(rho)), 1.0, delta=0.04)
        rare_indicator = (values == 3).astype(float)
        corrected = float(np.mean(rho * rare_indicator))  # type: ignore[operator]
        self.assertAlmostEqual(corrected, 0.1, delta=0.03)
        summary = wrapped.importance_sampling_summary()
        self.assertGreater(summary["rare_component_sample_count"], 0)
        self.assertGreater(summary["rho"]["ess"], 0)
        self.assertNotIn("rho_values", summary)

    def test_context_diagnostics_align_repeated_contexts_to_source_rows(self) -> None:
        source = self._numeric_source()
        wrapped = ImportanceSamplingSampleSource(source, self._config())
        batch = wrapped.batches(32, seed=12)
        tokens = [
            [
                PredicateToken(PredicateOp.GREATER_EQUAL, value=3),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ]
            for _ in range(64)
        ]
        generation_stats = SimpleNamespace(source_row_indices=tuple(np.repeat(np.arange(32), 2)))
        inv_only = np.ones((64, len(source.metadata.columns)), dtype=float)
        combined = inv_only * np.repeat(batch.importance_weights, 2)[:, None]
        wrapped.update_importance_context_statistics(
            generation_stats=generation_stats,
            token_rows=tokens,
            inv_only_weights=inv_only,
            combined_weights=combined,
            batch_metadata=batch.importance_metadata,
        )
        summary = wrapped.importance_sampling_summary()
        context_stats = summary["conditional_context_stats_by_stratum"]
        self.assertTrue(context_stats)
        first = next(iter(context_stats.values()))
        self.assertGreater(first["context_count"], 0)
        self.assertIn("F_A", first["fanout_effective_sample_size"])

    def test_sampler_counter_summary_unwraps_nested_sources(self) -> None:
        inner = SimpleNamespace(
            sampler_run_calls=3,
            conditional_sampler_batch_calls=2,
            conditional_rows_drawn=17,
        )
        outer = SimpleNamespace(base_source=inner)
        counters = _sampler_counters(outer)
        self.assertEqual(counters["sampler_run_calls"], 3)
        self.assertEqual(counters["conditional_sampler_batch_calls"], 2)
        self.assertEqual(counters["conditional_rows_drawn"], 17)

    def test_exact_conditional_sampling_within_synthetic_stratum(self) -> None:
        source = self._numeric_source()
        provider = ExactRootStratumProvider.from_encoded_rows(
            source.metadata,
            source.dataset.encoded_rows,
            root_column_semantics={"A.x": "ordered"},
        )
        stratum = RootDataStratum("ge2", 0, "A.x", "lower_tail", lower=2, probability=0.3)
        rows = provider.sample_conditional(stratum, 2000, np.random.default_rng(4))
        values = np.array([source.metadata.columns[0].domain[int(index)] for index in rows[:, 0]])
        self.assertTrue(np.all(values >= 2))
        self.assertAlmostEqual(float(np.mean(values == 3)), 1.0 / 3.0, delta=0.05)

    def test_live_root_jct_provider_honors_discovery_filters_and_full_jct_mass(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("title:id", ColumnKind.DATA, tuple(range(10000)), table="title"),
                ColumnMetadata("title:production_year", ColumnKind.DATA, (2014, 2015, SQL_NULL, "__OUTER_MISSING__"), table="title"),
                ColumnMetadata("title:name", ColumnKind.DATA, ("a", "b"), table="title"),
                ColumnMetadata("title:kind_id", ColumnKind.DATA, (1, 2), table="title"),
            ),
            full_join_cardinality=7,
            join_root="title",
            join_tables=("title",),
            join_edges=(),
        )
        table_actor = SimpleNamespace(
            table="title",
            join_keys=["id"],
            df=pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "title.id": [1, 2, 3],
                    "title.production_year": [2014, 2015, np.nan],
                    "title.name": ["a", "b", "a"],
                    "title.kind_id": [1, 2, 1],
                }
            ),
        )
        jct_actor = SimpleNamespace(
            jct=pd.DataFrame({"id": [1, 2, 3, 4], "title.weight": [1.0, 2.0, 3.0, 1.0]})
        )
        sampler = SimpleNamespace(
            join_spec=SimpleNamespace(join_root="title"),
            dt_actors=[table_actor],
            jct_actors={"title": jct_actor},
            join_card=7,
        )
        provider = ExactRootStratumProvider.from_neurocard_root_jct(
            metadata,
            sampler,
            include_categorical=False,
            max_domain_size=4096,
            root_column_semantics={
                "title": {
                    "production_year": "ordered",
                    "kind_id": "categorical",
                }
            },
        )
        self.assertEqual(set(provider.column_masses), {1})
        self.assertTrue(np.allclose(provider.column_masses[1].counts, [1.0, 2.0, 3.0, 1.0]))
        self.assertEqual(provider.column_masses[1].total_count, 7.0)
        self.assertEqual(provider.mass_diagnostics["root_jct_total_weight"], 7.0)

    def test_numeric_root_columns_default_to_categorical_without_semantics(self) -> None:
        source = self._numeric_source()
        provider = ExactRootStratumProvider.from_encoded_rows(
            source.metadata,
            source.dataset.encoded_rows,
        )
        strata = provider.discover(
            n_total=1000,
            predicate_probabilities={
                "equality": 0.2,
                "lower": 0.2,
                "upper": 0.2,
                "range": 0.2,
            },
            minimum_expected_context_support=100,
            max_selected_strata=10,
        )
        self.assertTrue(strata)
        self.assertEqual({stratum.region_type for stratum in strata}, {"equality"})
        self.assertTrue(all(stratum.semantic_type == "categorical" for stratum in strata))

    def test_native_range_support_matches_bruteforce_tiny_domain(self) -> None:
        probabilities = np.array([0.2, 0.3, 0.5])
        n_total = 1000
        p_range = 0.2
        equality, lower, upper = _expected_native_range_interval_counts(
            probabilities,
            n_total=n_total,
            p_range=p_range,
        )

        brute_equality = np.zeros(3)
        brute_lower = np.zeros(3)
        brute_upper = np.zeros(3)
        for x_rank, probability in enumerate(probabilities, start=1):
            for lower_rank in range(1, x_rank + 1):
                for upper_rank in range(x_rank, 4):
                    mass = n_total * p_range * probability * (1.0 / x_rank) * (1.0 / (4 - x_rank))
                    for interval_rank in range(1, 4):
                        if lower_rank == interval_rank and upper_rank == interval_rank:
                            brute_equality[interval_rank - 1] += mass
                        if lower_rank >= interval_rank:
                            brute_lower[interval_rank - 1] += mass
                        if upper_rank <= interval_rank:
                            brute_upper[interval_rank - 1] += mass
        self.assertTrue(np.allclose(equality, brute_equality))
        self.assertTrue(np.allclose(lower, brute_lower))
        self.assertTrue(np.allclose(upper, brute_upper))

    def test_row_specific_predicate_generation_probabilities_are_unchanged(self) -> None:
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "table_subset_sampling": "full",
                "wildcard_probability": 0.2,
                "equality_probability": 0.2,
                "lower_bound_probability": 0.2,
                "upper_bound_probability": 0.2,
                "native_range_probability": 0.2,
                "enable_native_range_tokens": True,
            }
        )
        self.assertEqual(generator.probability_diagnostics()["probability_total"], 1.0)

    def test_rho_multiplies_every_head_and_current_fanout_still_not_self_weighted(self) -> None:
        source = self._numeric_source()
        rows = source.dataset.encoded_rows[[7, 9]]
        tokens = [
            [PredicateToken.wildcard(), PredicateToken.wildcard(), PredicateToken.inv_fanout()],
            [PredicateToken.wildcard(), PredicateToken.wildcard(), PredicateToken.inv_fanout()],
        ]
        inv = cumulative_inverse_fanout_weights(rows, tokens, source.metadata)
        rho = np.array([0.25, 0.5])
        combined = inv * rho[:, None]
        self.assertTrue(np.allclose(inv[:, 2], [1.0, 1.0]))
        self.assertTrue(np.allclose(combined[:, 0], rho))
        self.assertTrue(np.allclose(combined[:, 1], rho))
        self.assertTrue(np.allclose(combined[:, 2], rho))

    def test_rho_is_gathered_for_generated_context_repeats(self) -> None:
        source = self._numeric_source()
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "table_subset_sampling": "full",
                "per_row_contexts": 3,
                "wildcard_probability": 1.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
            }
        )
        _, target_rows, stats = generator.generate_batch(
            encoded_rows=source.dataset.encoded_rows[[0, 9]],
            metadata=source.metadata,
            rng=np.random.default_rng(10),
        )
        gathered = importance_weights_for_generated_contexts(
            np.array([0.25, 0.5]),
            target_rows.shape[0],
            stats,
        )
        self.assertTrue(np.allclose(gathered, [0.25, 0.25, 0.25, 0.5, 0.5, 0.5]))

    def test_log_space_combined_weights_preserve_relative_direct_weights(self) -> None:
        inv = np.array([[1.0, 0.1], [0.5, 0.01], [0.25, 0.001]])
        rho = np.array([0.8, 0.4, 0.2])
        stable = stable_combine_importance_and_inverse_weights(inv, rho)
        direct = inv * rho[:, None]
        for column in range(inv.shape[1]):
            self.assertTrue(np.allclose(stable[:, column] / stable[0, column], direct[:, column] / direct[0, column]))

    def test_disabled_path_does_not_wrap_sample_source(self) -> None:
        config = {
            "dataset": {"type": "synthetic_full_join"},
            "importance_sampling": {"enabled": False},
            "factorization": {"enabled": False},
        }
        source = sample_source_from_config(config)
        self.assertNotIsInstance(source, ImportanceSamplingSampleSource)

    def test_new_job_light_importance_config_validates(self) -> None:
        validate_config(
            load_simple_yaml(
                "model/configs/job_light_duet_binary_native_anpm_20k_importance_sampling.yaml"
            )
        )
        validate_config(
            load_simple_yaml(
                "model/configs/job_light_duet_binary_native_anpm_importance_sampling_smoke.yaml"
            )
        )

    def test_no_held_out_constants_in_importance_modules(self) -> None:
        from pathlib import Path

        text = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in (
                "model/src/data/importance_sampling.py",
                "model/src/data/strata.py",
            )
        )
        for forbidden in ("query_id", "2015", "8200", "JOB-light q-error"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
