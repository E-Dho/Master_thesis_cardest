from __future__ import annotations

import copy
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import (
    FullJoinBatch,
    LiveNeuroCardFullJoinSampleSource,
    NeuroCardFullJoinSampleSource,
    SyntheticDataset,
)
from model.src.data.importance_sampling import (
    ImportanceSamplingRunningStats,
    ImportanceSamplingSampleSource,
    StreamingLogWeightStats,
    StreamingMomentStats,
    StratumPredicateContextStats,
    _assert_rho_defensive_bound,
    _context_amplification_by_stratum,
    _sampler_counters,
    _token_relevant_to_stratum,
)
from model.src.data.sample_sources import sample_source_from_config
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import (
    ExactRootStratumProvider,
    RootDataStratum,
    SQL_NULL,
    build_membership_lookup,
    _expected_native_range_interval_counts,
    _with_score,
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
                "diagnostics": {
                    "enabled": False,
                    "global_rho_reservoir_size": 5,
                    "per_stratum_rho_reservoir_size": 2,
                },
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
        self.assertLessEqual(summary["rho"]["percentile_reservoir_size"], 5)
        for stratum_summary in summary["rho_by_stratum"].values():
            self.assertLessEqual(stratum_summary["percentile_reservoir_size"], 2)

    def test_fixture_importance_conditional_rows_preserve_provenance(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("A.x", ColumnKind.DATA, tuple(range(10)), table="A"),
                ColumnMetadata("I_A", ColumnKind.INDICATOR, (0, 1), table="A"),
                ColumnMetadata("F_A", ColumnKind.FANOUT, (1,), table="A", fanout_source="A->A"),
            ),
            full_join_cardinality=10,
            join_root="A",
            join_tables=("A",),
            join_edges=(),
        )
        rows = np.asarray([[value, 1, 0] for value in range(10)], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "dataset_name": "fixture",
                "dataset_type": "neurocard_full_join",
                "join_cardinality": 10,
                "metadata": metadata.to_json_dict(),
                "domains_complete": True,
                "metadata_source": "complete_base_tables_and_join_metadata",
                "sample_rows": len(rows),
                "format_version": 2,
            }
            with open(f"{tmpdir}/manifest.json", "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            np.save(f"{tmpdir}/sample_rows.npy", rows)
            np.save(f"{tmpdir}/sample_trajectory_ids.npy", np.asarray([100 + i for i in range(10)]))
            np.save(
                f"{tmpdir}/sample_segment_ids.npy",
                np.asarray([(100 + i, i) for i in range(10)], dtype=object),
            )
            source = NeuroCardFullJoinSampleSource(tmpdir)
            config = self._config()
            config["training"]["batch_size"] = 64
            config["importance_sampling"]["mixture_probability"] = 0.5
            config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 1
            wrapped = ImportanceSamplingSampleSource(source, config)
            batch = wrapped.batches(64, seed=2)
        self.assertIsNotNone(batch.trajectory_ids)
        self.assertIsNotNone(batch.segment_ids)
        self.assertIsNotNone(batch.importance_weights)
        self.assertEqual(len(batch.trajectory_ids), len(batch.encoded_values))  # type: ignore[arg-type]
        self.assertEqual(len(batch.segment_ids), len(batch.encoded_values))  # type: ignore[arg-type]
        for encoded_row, trajectory_id, segment_id in zip(
            batch.encoded_values,
            batch.trajectory_ids,  # type: ignore[arg-type]
            batch.segment_ids,  # type: ignore[arg-type]
        ):
            x_value = int(encoded_row[0])
            self.assertEqual(trajectory_id, 100 + x_value)
            self.assertEqual(tuple(segment_id), (100 + x_value, x_value))

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
        wrapped.update_importance_context_statistics(
            generation_stats=generation_stats,
            token_rows=tokens,
            inv_only_weights=inv_only,
            rho=np.repeat(batch.importance_weights, 2),
            batch_metadata=batch.importance_metadata,
        )
        summary = wrapped.importance_sampling_summary()
        context_stats = summary["conditional_context_stats_by_stratum"]
        self.assertTrue(context_stats)
        first = next(iter(context_stats.values()))
        self.assertGreater(first["context_count"], 0)
        self.assertIn("F_A", first["fanout_effective_sample_size"])
        fanout_stats = first["fanout_effective_sample_size"]["F_A"]
        self.assertFalse(fanout_stats["inv_only"]["retains_sample_history"])
        self.assertFalse(fanout_stats["importance_times_inv"]["retains_sample_history"])
        self.assertIn("lower_threshold_counts", first)
        self.assertIn("relevant_fanout_token_signature_count", first)

    def test_fanout_conditional_stats_are_constant_size_after_many_updates(self) -> None:
        source = self._numeric_source()
        wrapped = ImportanceSamplingSampleSource(source, self._config())
        batch = wrapped.batches(16, seed=4)
        generation_stats = SimpleNamespace(source_row_indices=tuple(range(16)))
        tokens = [
            [
                PredicateToken(PredicateOp.RANGE, value=3, upper=3),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ]
            for _ in range(16)
        ]
        inv_only = np.ones((16, len(source.metadata.columns)), dtype=float)
        for _ in range(200):
            wrapped.update_importance_context_statistics(
                generation_stats=generation_stats,
                token_rows=tokens,
                inv_only_weights=inv_only,
                rho=batch.importance_weights,
                batch_metadata=batch.importance_metadata,
            )
        summary = wrapped.importance_sampling_summary()
        first = next(iter(summary["conditional_context_stats_by_stratum"].values()))
        fanout_stats = first["fanout_effective_sample_size"]["F_A"]
        self.assertFalse(hasattr(StreamingMomentStats(), "reservoir"))
        self.assertFalse(fanout_stats["inv_only"]["retains_sample_history"])
        self.assertFalse(fanout_stats["importance_times_inv"]["retains_sample_history"])
        self.assertLessEqual(summary["rho"]["percentile_reservoir_size"], 5)

    def test_vectorized_membership_lookup_matches_direct_contains_value(self) -> None:
        source = self._numeric_source()
        strata = (
            RootDataStratum("eq2", 0, "A.x", "equality", value=2),
            RootDataStratum("ge1", 0, "A.x", "lower_tail", lower=1),
            RootDataStratum("le1", 0, "A.x", "upper_tail", upper=1),
        )
        lookup = build_membership_lookup(source.metadata, strata)
        vectorized = membership_matrix(source.metadata, source.dataset.encoded_rows, strata, lookup)
        direct = membership_matrix(source.metadata, source.dataset.encoded_rows, strata)
        self.assertTrue(np.array_equal(vectorized, direct))
        self.assertTrue(np.any(vectorized[:, 1] & vectorized[:, 2]))

    def test_rho_defensive_bound_is_enforced(self) -> None:
        _assert_rho_defensive_bound(np.array([0.1, 1.25]), 0.2)
        with self.assertRaisesRegex(ValueError, "defensive mixture bound"):
            _assert_rho_defensive_bound(np.array([1.250001]), 0.2)

    def test_rare_selected_stratum_membership_assertion_raises(self) -> None:
        source = self._numeric_source()
        wrapped = ImportanceSamplingSampleSource(source, self._config())
        with self.assertRaisesRegex(ValueError, "selected-stratum membership"):
            wrapped._assert_rare_rows_match_selected_strata(
                rare_positions=np.array([0]),
                selected_ids=[wrapped.selected_strata[0].stratum_id],
                memberships=np.zeros((1, len(wrapped.selected_strata)), dtype=bool),
            )

    def test_global_true_rho_inv_ess_uses_unscaled_weights_across_batches(self) -> None:
        source = self._numeric_source()
        stratum = RootDataStratum("all", 0, "A.x", "lower_tail", lower=0, probability=1.0)
        stats = ImportanceSamplingRunningStats(
            enabled=True,
            mixture_probability=0.2,
            selected_strata=(stratum,),
        )
        memberships = np.ones((2, 1), dtype=bool)
        token_rows = [
            [PredicateToken.wildcard(), PredicateToken.wildcard(), PredicateToken.inv_fanout()],
            [PredicateToken.wildcard(), PredicateToken.wildcard(), PredicateToken.inv_fanout()],
        ]
        generation_stats = SimpleNamespace(source_row_indices=(0, 1))
        batch1_inv = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        batch2_inv = np.array([[1.0, 1.0, 1000.0], [1.0, 1.0, 1.0]])
        rho1 = np.array([1.0, 0.5])
        rho2 = np.array([0.25, 0.25])
        stats.update_context_batch(
            metadata=source.metadata,
            generation_stats=generation_stats,
            token_rows=token_rows,
            memberships=memberships,
            inv_only_weights=batch1_inv,
            combined_weights=np.empty((0, 0)),
            rho=rho1,
        )
        stats.update_context_batch(
            metadata=source.metadata,
            generation_stats=generation_stats,
            token_rows=token_rows,
            memberships=memberships,
            inv_only_weights=batch2_inv,
            combined_weights=np.empty((0, 0)),
            rho=rho2,
        )
        summary = stats.to_json_dict()["conditional_context_stats_by_stratum"]["all"]
        observed = summary["fanout_effective_sample_size"]["F_A"]["importance_times_inv"]["ess"]
        true_values = np.array([1.0, 0.5, 250.0, 0.25])
        expected = float(true_values.sum() ** 2 / np.dot(true_values, true_values))
        self.assertAlmostEqual(observed, expected)

        naive = np.concatenate(
            [
                stable_combine_importance_and_inverse_weights(batch1_inv, rho1)[:, 2],
                stable_combine_importance_and_inverse_weights(batch2_inv, rho2)[:, 2],
            ]
        )
        naive_ess = float(naive.sum() ** 2 / np.dot(naive, naive))
        self.assertNotAlmostEqual(naive_ess, expected)

    def test_range_relevance_matches_stratum_semantics(self) -> None:
        eq = RootDataStratum("eq", 0, "A.x", "equality", value=3)
        self.assertTrue(_token_relevant_to_stratum(PredicateToken(PredicateOp.RANGE, value=3, upper=3), eq))
        self.assertFalse(_token_relevant_to_stratum(PredicateToken(PredicateOp.RANGE, value=2, upper=3), eq))
        self.assertFalse(_token_relevant_to_stratum(PredicateToken(PredicateOp.RANGE, value=3, upper=4), eq))
        self.assertTrue(
            _token_relevant_to_stratum(
                PredicateToken(PredicateOp.RANGE, value=3, upper=5),
                RootDataStratum("ge", 0, "A.x", "lower_tail", lower=3),
            )
        )
        self.assertTrue(
            _token_relevant_to_stratum(
                PredicateToken(PredicateOp.RANGE, value=1, upper=3),
                RootDataStratum("le", 0, "A.x", "upper_tail", upper=3),
            )
        )
        self.assertTrue(
            _token_relevant_to_stratum(
                PredicateToken(PredicateOp.RANGE, value=3, upper=4),
                RootDataStratum("range", 0, "A.x", "range", lower=2, upper=5),
            )
        )

    def test_zero_expected_support_is_not_treated_as_non_applicable(self) -> None:
        scored = _with_score(
            RootDataStratum(
                "zero",
                0,
                "A.x",
                "equality",
                value=3,
                expected_target_rows=10.0,
                expected_equality_count=0.0,
                expected_range_support=None,
            ),
            threshold=5.0,
        )
        self.assertEqual(scored.support_score, 0.0)
        self.assertEqual(scored.support_deficit, 5.0)
        self.assertEqual(scored.support_bottleneck, "equality")

    def test_support_bottleneck_identifies_minimum_applicable_metric(self) -> None:
        cases = {
            "target_rows": RootDataStratum(
                "target",
                0,
                "A.x",
                "equality",
                expected_target_rows=1.0,
                expected_equality_count=3.0,
            ),
            "lower": RootDataStratum(
                "lower",
                0,
                "A.x",
                "lower_tail",
                expected_target_rows=10.0,
                expected_lower_count=2.0,
                expected_range_support=4.0,
            ),
            "upper": RootDataStratum(
                "upper",
                0,
                "A.x",
                "upper_tail",
                expected_target_rows=10.0,
                expected_upper_count=2.0,
                expected_range_support=4.0,
            ),
            "native_range": RootDataStratum(
                "range",
                0,
                "A.x",
                "lower_tail",
                expected_target_rows=10.0,
                expected_lower_count=6.0,
                expected_range_support=2.0,
            ),
        }
        for expected, stratum in cases.items():
            self.assertEqual(
                _with_score(stratum, threshold=5.0).support_bottleneck,
                expected,
            )

    def test_support_planning_defaults_to_nominal_optimizer_steps(self) -> None:
        config = self._config()
        config["training"]["steps_per_epoch"] = 10
        config["training"]["epochs"] = 2
        config["training"]["early_stopping_patience_steps"] = 3
        config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 1000
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), config)
        self.assertEqual(wrapped.maximum_configured_steps, 20)
        self.assertEqual(wrapped.support_planning_steps, 20)
        self.assertEqual(wrapped.planned_sample_count, 20_000)

    def test_explicit_support_planning_overrides_short_smoke_length(self) -> None:
        config = self._config()
        config["training"]["steps_per_epoch"] = 100
        config["importance_sampling"]["discovery"]["support_planning_steps"] = 20_000
        config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 4_000_000
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), config)
        self.assertEqual(wrapped.maximum_configured_steps, 100)
        self.assertEqual(wrapped.support_planning_steps, 20_000)
        self.assertEqual(wrapped.planned_sample_count, 20_000_000)

    def test_same_support_horizon_discovers_same_proposal_for_smoke_and_full_steps(self) -> None:
        smoke = self._config()
        full = copy.deepcopy(smoke)
        for config, steps in ((smoke, 100), (full, 20_000)):
            config["training"]["steps_per_epoch"] = steps
            config["importance_sampling"]["discovery"]["support_planning_steps"] = 20_000
            config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 4_000_000
        smoke_wrapped = ImportanceSamplingSampleSource(self._numeric_source(), smoke)
        full_wrapped = ImportanceSamplingSampleSource(self._numeric_source(), full)
        smoke_summary = smoke_wrapped.importance_sampling_summary()
        full_summary = full_wrapped.importance_sampling_summary()
        keys = [
            "stratum_id",
            "probability",
            "support_score",
            "support_deficit",
            "support_bottleneck",
            "alpha",
        ]
        self.assertEqual(
            [{key: item[key] for key in keys} for item in smoke_summary["selected_strata"]],
            [{key: item[key] for key in keys} for item in full_summary["selected_strata"]],
        )
        for column in smoke_wrapped._membership_lookup.by_column:
            smoke_masks = smoke_wrapped._membership_lookup.by_column[column]
            full_masks = full_wrapped._membership_lookup.by_column[column]
            self.assertEqual(len(smoke_masks), len(full_masks))
            for left, right in zip(smoke_masks, full_masks):
                self.assertTrue(np.array_equal(left, right))

    def test_realized_support_fraction_and_scaled_support_are_reported_only(self) -> None:
        config = self._config()
        config["training"]["steps_per_epoch"] = 100
        config["importance_sampling"]["discovery"]["support_planning_steps"] = 20_000
        config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 4_000_000
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), config)
        alpha_before = tuple(stratum.alpha for stratum in wrapped.selected_strata)
        summary = wrapped.importance_sampling_summary(
            actual_optimizer_steps=100,
            early_stopped=True,
            early_stopping_stop_step=100,
        )
        self.assertAlmostEqual(summary["realized_support_fraction"], 0.005)
        first = summary["selected_strata"][0]
        self.assertAlmostEqual(
            first["realized_expected_target_rows"],
            first["planned_expected_target_rows"] * 0.005,
        )
        self.assertEqual(alpha_before, tuple(stratum.alpha for stratum in wrapped.selected_strata))
        self.assertEqual(summary["early_stopping_stop_step"], 100)

    def test_proposal_composition_alpha_sums_to_one(self) -> None:
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), self._config())
        composition = wrapped.importance_sampling_summary()["proposal_composition"]
        self.assertAlmostEqual(composition["alpha_sum"], 1.0)
        self.assertAlmostEqual(
            sum(item["alpha_sum"] for item in composition["by_region_type"].values()),
            1.0,
        )
        self.assertAlmostEqual(
            sum(item["alpha_sum"] for item in composition["by_support_bottleneck"].values()),
            1.0,
        )

    def test_smallest_rho_patterns_are_bounded_and_explain_overlap(self) -> None:
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), self._config())
        for seed in range(10):
            wrapped.batches(128, seed=seed)
        patterns = wrapped.importance_sampling_summary()["smallest_rho_patterns"]
        self.assertLessEqual(len(patterns), 32)
        self.assertTrue(patterns)
        self.assertTrue(all("alpha_over_probability_sum" in item for item in patterns))
        self.assertEqual(patterns, sorted(patterns, key=lambda item: item["rho"]))

    def test_context_amplification_uses_actual_smoke_fraction(self) -> None:
        config = self._config()
        config["importance_sampling"]["discovery"]["support_planning_steps"] = 20_000
        config["importance_sampling"]["discovery"]["minimum_expected_context_support"] = 4_000_000
        wrapped = ImportanceSamplingSampleSource(self._numeric_source(), config)
        batch = wrapped.batches(64, seed=5)
        stratum = wrapped.selected_strata[0]
        if stratum.region_type == "equality":
            token = PredicateToken(PredicateOp.EQUAL, value=stratum.value)
            op_name = PredicateOp.EQUAL.value
            planned = stratum.expected_equality_count
        elif stratum.region_type == "lower_tail":
            token = PredicateToken(PredicateOp.GREATER_EQUAL, value=stratum.lower)
            op_name = PredicateOp.GREATER_EQUAL.value
            planned = stratum.expected_lower_count
        else:
            token = PredicateToken(PredicateOp.LESS_EQUAL, value=stratum.upper)
            op_name = PredicateOp.LESS_EQUAL.value
            planned = stratum.expected_upper_count
        tokens = [[token, PredicateToken.wildcard(), PredicateToken.inv_fanout()] for _ in range(64)]
        inv_only = np.ones((64, len(self._numeric_source().metadata.columns)), dtype=float)
        wrapped.update_importance_context_statistics(
            generation_stats=SimpleNamespace(source_row_indices=tuple(range(64))),
            token_rows=tokens,
            inv_only_weights=inv_only,
            rho=batch.importance_weights,
            batch_metadata=batch.importance_metadata,
        )
        summary = wrapped.importance_sampling_summary(actual_optimizer_steps=100)
        op_summary = summary["context_amplification_by_stratum"][stratum.stratum_id][
            "by_operator"
        ][op_name]
        self.assertAlmostEqual(
            op_summary["expected_uniform_count_at_actual_steps"],
            planned * 0.005,
        )
        self.assertIn("observed_exact_support_event_count", op_summary)

    def test_exact_lower_support_amplification_excludes_stricter_thresholds(self) -> None:
        stratum = RootDataStratum(
            "ge2",
            0,
            "A.x",
            "lower_tail",
            lower=2,
            expected_lower_count=100.0,
            expected_range_support=50.0,
        )
        stats = StratumPredicateContextStats()
        tokens = [
            PredicateToken(PredicateOp.GREATER_EQUAL, value=2),
            PredicateToken(PredicateOp.GREATER_EQUAL, value=3),
            PredicateToken(PredicateOp.GREATER_THAN, value=2),
            PredicateToken(PredicateOp.RANGE, value=2, upper=3),
            PredicateToken(PredicateOp.RANGE, value=3, upper=3),
            PredicateToken(PredicateOp.RANGE, value=1, upper=3),
        ]
        stats.update_tokens(tokens=tokens, stratum=stratum, fanout_signatures=[""] * len(tokens))
        payload = stats.to_json_dict()
        self.assertEqual(payload["stratum_relevant_context_count"], 5)
        self.assertEqual(payload["stratum_relevant_operator_count"][PredicateOp.GREATER_EQUAL.value], 2)
        self.assertEqual(payload["stratum_relevant_operator_count"][PredicateOp.GREATER_THAN.value], 1)
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.GREATER_EQUAL.value], 1)
        self.assertNotIn(PredicateOp.GREATER_THAN.value, payload["exact_support_event_operator_count"])
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.RANGE.value], 2)

        amplification = _context_amplification_by_stratum(
            (stratum,),
            {stratum.stratum_id: stats},
            realized_fraction=0.1,
        )[stratum.stratum_id]["by_operator"]
        lower = amplification[PredicateOp.GREATER_EQUAL.value]
        self.assertEqual(lower["expected_uniform_count_at_actual_steps"], 10.0)
        self.assertEqual(lower["observed_exact_support_event_count"], 1)
        self.assertEqual(lower["raw_context_amplification"], 0.1)
        native_range = amplification[PredicateOp.RANGE.value]
        self.assertEqual(native_range["expected_uniform_count_at_actual_steps"], 5.0)
        self.assertEqual(native_range["observed_exact_support_event_count"], 2)

    def test_exact_upper_support_amplification_excludes_stricter_thresholds(self) -> None:
        stratum = RootDataStratum(
            "le2",
            0,
            "A.x",
            "upper_tail",
            upper=2,
            expected_upper_count=80.0,
            expected_range_support=40.0,
        )
        stats = StratumPredicateContextStats()
        tokens = [
            PredicateToken(PredicateOp.LESS_EQUAL, value=2),
            PredicateToken(PredicateOp.LESS_EQUAL, value=1),
            PredicateToken(PredicateOp.LESS_THAN, value=2),
            PredicateToken(PredicateOp.RANGE, value=0, upper=2),
            PredicateToken(PredicateOp.RANGE, value=0, upper=1),
            PredicateToken(PredicateOp.RANGE, value=0, upper=3),
        ]
        stats.update_tokens(tokens=tokens, stratum=stratum, fanout_signatures=[""] * len(tokens))
        payload = stats.to_json_dict()
        self.assertEqual(payload["stratum_relevant_context_count"], 5)
        self.assertEqual(payload["stratum_relevant_operator_count"][PredicateOp.LESS_EQUAL.value], 2)
        self.assertEqual(payload["stratum_relevant_operator_count"][PredicateOp.LESS_THAN.value], 1)
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.LESS_EQUAL.value], 1)
        self.assertNotIn(PredicateOp.LESS_THAN.value, payload["exact_support_event_operator_count"])
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.RANGE.value], 2)

        amplification = _context_amplification_by_stratum(
            (stratum,),
            {stratum.stratum_id: stats},
            realized_fraction=0.25,
        )[stratum.stratum_id]["by_operator"]
        upper = amplification[PredicateOp.LESS_EQUAL.value]
        self.assertEqual(upper["expected_uniform_count_at_actual_steps"], 20.0)
        self.assertEqual(upper["observed_exact_support_event_count"], 1)
        self.assertEqual(upper["raw_context_amplification"], 0.05)
        native_range = amplification[PredicateOp.RANGE.value]
        self.assertEqual(native_range["expected_uniform_count_at_actual_steps"], 10.0)
        self.assertEqual(native_range["observed_exact_support_event_count"], 2)

    def test_exact_equality_support_amplification_counts_exact_literal_and_point_range(self) -> None:
        stratum = RootDataStratum(
            "eq2",
            0,
            "A.x",
            "equality",
            value=2,
            expected_equality_count=60.0,
            expected_range_support=30.0,
        )
        stats = StratumPredicateContextStats()
        tokens = [
            PredicateToken(PredicateOp.EQUAL, value=2),
            PredicateToken(PredicateOp.EQUAL, value=3),
            PredicateToken(PredicateOp.RANGE, value=2, upper=2),
            PredicateToken(PredicateOp.RANGE, value=2, upper=3),
            PredicateToken(PredicateOp.GREATER_EQUAL, value=2),
        ]
        stats.update_tokens(tokens=tokens, stratum=stratum, fanout_signatures=[""] * len(tokens))
        payload = stats.to_json_dict()
        self.assertEqual(payload["stratum_relevant_context_count"], 2)
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.EQUAL.value], 1)
        self.assertEqual(payload["exact_support_event_operator_count"][PredicateOp.RANGE.value], 1)

        amplification = _context_amplification_by_stratum(
            (stratum,),
            {stratum.stratum_id: stats},
            realized_fraction=0.5,
        )[stratum.stratum_id]["by_operator"]
        equality = amplification[PredicateOp.EQUAL.value]
        self.assertEqual(equality["expected_uniform_count_at_actual_steps"], 30.0)
        self.assertEqual(equality["observed_exact_support_event_count"], 1)
        native_range = amplification[PredicateOp.RANGE.value]
        self.assertEqual(native_range["expected_uniform_count_at_actual_steps"], 15.0)
        self.assertEqual(native_range["observed_exact_support_event_count"], 1)

    def test_log_weight_summary_is_strict_json_safe_for_extreme_weights(self) -> None:
        stats = StreamingLogWeightStats()
        stats.update_log_weights(np.array([-1000.0, 0.0, 1000.0]))
        payload = stats.to_json_dict()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["max"])
        self.assertIsNone(payload["sum"])

    def test_rng_reproducibility_and_step_variation(self) -> None:
        source = self._numeric_source()
        first = ImportanceSamplingSampleSource(source, self._config()).batches(128, seed=7)
        second = ImportanceSamplingSampleSource(source, self._config()).batches(128, seed=7)
        third = ImportanceSamplingSampleSource(source, self._config()).batches(128, seed=8)
        self.assertTrue(np.array_equal(first.encoded_values, second.encoded_values))
        self.assertTrue(np.array_equal(first.importance_weights, second.importance_weights))
        self.assertFalse(np.array_equal(first.encoded_values, third.encoded_values))

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

    def test_conditional_candidate_mass_matches_stratum_foj_count(self) -> None:
        source = self._fake_live_root_source()
        stratum = RootDataStratum(
            "title:production_year:ge:2015",
            0,
            "title:production_year",
            "lower_tail",
            lower=2015,
            foj_count=5.0,
        )
        _, weights = source._root_stratum_candidates(stratum)
        self.assertEqual(float(weights.sum()), 5.0)
        cached_values = source._root_jct_value_cache["title.production_year"]
        self.assertTrue(np.issubdtype(cached_values.dtype, np.number))

    def test_conditional_candidate_mass_mismatch_raises(self) -> None:
        source = self._fake_live_root_source()
        stratum = RootDataStratum(
            "title:production_year:ge:2015",
            0,
            "title:production_year",
            "lower_tail",
            lower=2015,
            foj_count=4.0,
        )
        with self.assertRaisesRegex(ValueError, "candidate mass does not match"):
            source._root_stratum_candidates(stratum)

    def _fake_live_root_source(self) -> LiveNeuroCardFullJoinSampleSource:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata(
                    "title:production_year",
                    ColumnKind.DATA,
                    (2014, 2015, 2016),
                    table="title",
                ),
            ),
            full_join_cardinality=6,
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
                    "title.production_year": [2014, 2015, 2016],
                }
            ),
        )
        jct_actor = SimpleNamespace(
            jct=pd.DataFrame(
                {"id": [1, 2, 3], "title.weight": [1.0, 2.0, 3.0]}
            )
        )
        source = LiveNeuroCardFullJoinSampleSource.__new__(LiveNeuroCardFullJoinSampleSource)
        source._metadata = metadata
        source._sampler = SimpleNamespace(
            join_spec=SimpleNamespace(join_root="title"),
            dt_actors=[table_actor],
            jct_actors={"title": jct_actor},
        )
        source._root_jct_value_cache = {}
        source._root_jct_weight_cache = {}
        source._root_stratum_candidate_cache = {}
        return source

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
        validate_config(
            load_simple_yaml(
                "model/configs/job_light_duet_binary_native_anpm_importance_sampling_performance_smoke.yaml"
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
