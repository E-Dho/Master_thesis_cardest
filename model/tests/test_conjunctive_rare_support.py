"""Conjunctive rare-support strata: representation, scoring, search, forcing.

These run without PyTorch so the discovery machinery stays testable on any
machine that can run the sampler.
"""

from __future__ import annotations

import ast
import json
import pathlib
import shutil
import sys
import time
import tempfile
import unittest

import numpy as np

from model.src.config import (
    load_simple_yaml,
    resolve_early_stopping_monitor,
    validate_config,
)
from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import (
    RareSupportDiscoveryError,
    DEFAULT_DEFICIT_OBJECTIVE,
    ExactRootStratumProvider,
    deficit_from_supports,
    SQL_NULL,
    _is_structural_value,
    RootColumnMass,
    _can_extend_any_seed,
    _expected_lower_threshold_counts,
    _expected_native_range_interval_counts,
    _expected_upper_threshold_counts,
    auxiliary_eligibility_mask,
    per_value_realization_factors,
    RootColumnMass,
    RootDataStratum,
    StratumTerm,
    _capped_allocation,
    build_membership_lookup,
    conditional_head_support,
    fixed_forced_predicate_is_supported,
    membership_matrix,
    TERM_OPERATOR_FAMILY,
    served_head_indices,
    table_subset_inclusion_probability,
    with_realization_probabilities,
    with_conditional_score,
)
from model.src.data.complete_domain_preparation import COMPLETE_METADATA_SOURCE
from model.src.data.full_join_sampler import (
    LiveNeuroCardFullJoinSampleSource,
    NeuroCardFullJoinSampleSource,
)
from model.src.data.schema import root_paths_closure
from model.src.predicates.operators import PredicateOp
from model.src.predicates.generation import (
    GeneratedTrainingContext,
    PredicateTrainingContextGenerator,
    forced_predicate_for_stratum,
    forced_prefix_is_complete,
)


PROBABILITIES = {
    "wildcard": 0.2,
    "equality": 0.2,
    "lower": 0.2,
    "upper": 0.2,
    "range": 0.2,
}


def metadata() -> ModelMetadata:
    """Two-table schema: root ``T`` plus child ``K``, with a suffix data head."""

    return ModelMetadata(
        columns=(
            ColumnMetadata("T.year", ColumnKind.DATA, (1950, 1951, 1952, 1953), table="T"),
            ColumnMetadata("K.keyword", ColumnKind.DATA, (0, 1, 2), table="K"),
            ColumnMetadata("T.phonetic", ColumnKind.DATA, (0, 1, 2, 3), table="T"),
            ColumnMetadata("K.extra", ColumnKind.DATA, (0, 1), table="K"),
            ColumnMetadata("I_K", ColumnKind.INDICATOR, (0, 1), table="K"),
            ColumnMetadata("F_K", ColumnKind.FANOUT, (1, 2), table="T", fanout_source="T->K"),
        ),
        full_join_cardinality=64.0,
        join_root="T",
        join_tables=("T", "K"),
        join_edges=(("T", "K"),),
    )


def encoded_rows() -> np.ndarray:
    """40 rows; year id 2 and keyword id 1 co-occur on exactly 4 of them."""

    rows = []
    for index in range(40):
        year = 2 if index < 10 else index % 2
        keyword = 1 if index < 4 else 0
        rows.append([year, keyword, index % 4, index % 2, 1, index % 2])
    return np.asarray(rows, dtype=np.int64)


def provider() -> ExactRootStratumProvider:
    meta = metadata()
    rows = encoded_rows()
    masses = {}
    for column_index in (0, 1, 2, 3):
        column = meta.columns[column_index]
        counts = np.bincount(rows[:, column_index], minlength=column.domain_size).astype(float)
        masses[column_index] = RootColumnMass(
            column_index=column_index,
            values=column.domain,
            counts=counts,
            total_count=float(counts.sum()),
            source="test",
            semantic_type="ordered" if column_index == 0 else "categorical",
        )
    return ExactRootStratumProvider(meta, masses, encoded_rows=rows, source="test")


def term(column_index: int, name: str, value) -> StratumTerm:
    return StratumTerm(
        column_index=column_index, column_name=name, region_type="equality", value=value
    )


def stratum(
    *terms: StratumTerm,
    probability: float = 0.1,
    realized_support_mass: float | None = None,
) -> RootDataStratum:
    """Build a stratum, defaulting the joint mass to the independent product.

    Real conjunctions get their mass from the provider, which averages the
    product of per-row factors over the joint rows.  A hand-built fixture has
    no rows, so tests that are not about correlation state the independent
    product explicitly instead of having it assumed for them.
    """

    if realized_support_mass is None and len(terms) > 1:
        factors = 1.0
        for item in terms:
            factors *= (
                item.realization_probability
                if item.realization_probability is not None
                else PROBABILITIES[TERM_OPERATOR_FAMILY[item.region_type]]
            )
        realized_support_mass = probability * factors
    return RootDataStratum(
        realized_support_mass=realized_support_mass,
        stratum_id="&".join(item.column_name for item in terms),
        column_index=terms[0].column_index,
        column_name=terms[0].column_name,
        region_type=terms[0].region_type,
        value=terms[0].value,
        terms=terms,
        probability=probability,
        foj_count=probability * 40.0,
    )


class StratumRepresentationTest(unittest.TestCase):
    def test_a_single_term_is_derived_from_the_legacy_fields(self):
        legacy = RootDataStratum(
            stratum_id="T.year:eq:1952",
            column_index=0,
            column_name="T.year",
            region_type="equality",
            value=1952,
        )
        self.assertEqual(legacy.depth, 1)
        self.assertFalse(legacy.is_conjunctive)
        self.assertEqual(legacy.terms[0].value, 1952)
        self.assertTrue(legacy.contains_value(1952))
        self.assertFalse(legacy.contains_value(1951))

    def test_terms_are_ordered_and_the_legacy_fields_track_the_last_one(self):
        # The last term gates which suffix heads the stratum can serve, so the
        # legacy scalars must describe it rather than an arbitrary term.
        conjunction = stratum(term(2, "T.phonetic", 3), term(0, "T.year", 1952))
        self.assertEqual(conjunction.column_indices, (0, 2))
        self.assertEqual(conjunction.column_index, 2)
        self.assertEqual(conjunction.column_name, "T.phonetic")
        self.assertEqual(conjunction.value, 3)
        self.assertTrue(conjunction.is_conjunctive)

    def test_a_repeated_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeats a column"):
            stratum(term(0, "T.year", 1952), term(0, "T.year", 1953))

    def test_contains_row_requires_every_term(self):
        meta = metadata()
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertTrue(conjunction.contains_row(np.array([2, 1, 0, 0, 1, 0]), meta))
        self.assertFalse(conjunction.contains_row(np.array([2, 0, 0, 0, 1, 0]), meta))
        self.assertFalse(conjunction.contains_row(np.array([1, 1, 0, 0, 1, 0]), meta))

    def test_tables_spans_every_term(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertEqual(conjunction.tables(metadata()), frozenset({"T", "K"}))

    def test_round_trip_through_json_keeps_the_conjunction(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        restored = RootDataStratum(**conjunction.to_json_dict())
        self.assertEqual(restored.column_indices, conjunction.column_indices)
        self.assertTrue(restored.is_conjunctive)


class MembershipTest(unittest.TestCase):
    def test_membership_ands_the_terms_with_and_without_the_lookup(self):
        meta = metadata()
        rows = encoded_rows()
        strata = (
            stratum(term(0, "T.year", 1952)),
            stratum(term(1, "K.keyword", 1)),
            stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1)),
        )
        direct = membership_matrix(meta, rows, strata)
        via_lookup = membership_matrix(
            meta, rows, strata, lookup=build_membership_lookup(meta, strata)
        )
        np.testing.assert_array_equal(direct, via_lookup)
        self.assertEqual(int(direct[:, 0].sum()), 10)
        self.assertEqual(int(direct[:, 1].sum()), 4)
        # The conjunction is the intersection, not either marginal.
        self.assertEqual(int(direct[:, 2].sum()), 4)
        np.testing.assert_array_equal(direct[:, 2], direct[:, 0] & direct[:, 1])

    def test_provider_row_membership_matches_the_matrix(self):
        source = provider()
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        mask = source.row_membership(encoded_rows(), conjunction)
        self.assertEqual(int(mask.sum()), 4)


class ConditionalSupportTest(unittest.TestCase):
    def test_a_head_at_or_before_the_last_term_is_not_served(self):
        meta = metadata()
        conjunction = stratum(term(0, "T.year", 1952), term(2, "T.phonetic", 3))
        for head in (0, 1, 2):
            self.assertEqual(
                conditional_head_support(
                    conjunction,
                    head,
                    metadata=meta,
                    n_total=1_000_000,
                    predicate_probabilities=PROBABILITIES,
                    root_table="T",
                ),
                0.0,
            )
        self.assertGreater(
            conditional_head_support(
                conjunction,
                3,
                metadata=meta,
                n_total=1_000_000,
                predicate_probabilities=PROBABILITIES,
                root_table="T",
            ),
            0.0,
        )

    def test_every_term_multiplies_its_operator_probability(self):
        # This is what the marginal bottleneck missed: a depth-2 stratum needs
        # both tokens emitted, so its support is p_op lower than depth-1.
        # Both terms sit on the root table and the scored head is the same, so
        # the table-subset factor cancels and only the operator factor is left.
        meta = metadata()
        single = stratum(term(0, "T.year", 1952), probability=0.25)
        double = stratum(
            term(0, "T.year", 1952), term(2, "T.phonetic", 1), probability=0.25
        )
        kwargs = dict(
            metadata=meta,
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            root_table="T",
        )
        single_support = conditional_head_support(single, 3, **kwargs)
        double_support = conditional_head_support(double, 3, **kwargs)
        self.assertGreater(single_support, 0.0)
        self.assertAlmostEqual(double_support, single_support * 0.2, places=6)

    def test_a_term_on_another_table_also_pays_the_table_subset_factor(self):
        # A conjunction that reaches a second table is rarer still: the subset
        # draw has to keep that table as well as the head's own.
        meta = metadata()
        single = stratum(term(0, "T.year", 1952), probability=0.25)
        crossing = stratum(
            term(0, "T.year", 1952), term(1, "K.keyword", 1), probability=0.25
        )
        kwargs = dict(
            metadata=meta,
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            root_table="T",
        )
        # Head 2 is on the root table, so the depth-1 stratum needs no non-root
        # table at all while the crossing one drags ``K`` into the subset.
        table_factor = table_subset_inclusion_probability(
            frozenset({"K"}), metadata=meta, root_table="T"
        )
        self.assertAlmostEqual(
            conditional_head_support(crossing, 2, **kwargs),
            conditional_head_support(single, 2, **kwargs) * 0.2 * table_factor,
            places=6,
        )

    def test_the_table_subset_factor_is_below_one_for_a_non_root_table(self):
        meta = metadata()
        self.assertEqual(
            table_subset_inclusion_probability(frozenset({"T"}), metadata=meta, root_table="T"),
            1.0,
        )
        child = table_subset_inclusion_probability(
            frozenset({"K"}), metadata=meta, root_table="T"
        )
        self.assertGreater(child, 0.0)
        self.assertLess(child, 1.0)

    def test_served_heads_are_every_column_after_the_last_term(self):
        # Indicator and fanout heads included: the fanout heads are the ones
        # whose effective sample size motivated rare-support training.
        meta = metadata()
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertEqual(served_head_indices(conjunction, meta), (2, 3, 4, 5))
        kinds = {meta.columns[index].kind for index in served_head_indices(conjunction, meta)}
        self.assertIn(ColumnKind.FANOUT, kinds)
        self.assertIn(ColumnKind.INDICATOR, kinds)

    def test_served_heads_agree_with_the_auxiliary_eligibility_mask(self):
        # These two must not drift apart: scoring fewer heads than the loss
        # actually trains is what left the fanout heads out of the deficit.
        meta = metadata()
        for selected in (stratum(term(0, "T.year", 1952)),
                         stratum(term(0, "T.year", 1952), term(2, "T.phonetic", 1))):
            mask = auxiliary_eligibility_mask(
                meta,
                np.array([max(selected.column_indices)], dtype=int),
                row_count=1,
            )
            self.assertEqual(
                served_head_indices(selected, meta),
                tuple(int(index) for index in np.flatnonzero(mask[0])),
            )

    def test_deficit_sums_over_served_heads_so_earlier_strata_rank_higher(self):
        meta = metadata()
        kwargs = dict(
            metadata=meta,
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=100.0,
            root_table="T",
        )
        early = with_conditional_score(stratum(term(0, "T.year", 1952)), **kwargs)
        late = with_conditional_score(stratum(term(2, "T.phonetic", 3)), **kwargs)
        self.assertEqual(len(served_head_indices(early, meta)), 5)
        self.assertEqual(len(served_head_indices(late, meta)), 3)
        self.assertGreater(early.support_deficit, late.support_deficit)
        self.assertIn("heads", str(early.support_bottleneck))


class BeamSearchTest(unittest.TestCase):
    def search(self, **overrides):
        options = dict(
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=1000.0,
            max_depth=2,
            beam_width=8,
            minimum_joint_rows=1.0,
            require_distinct_tables=True,
        )
        options.update(overrides)
        seeds = (
            stratum(term(0, "T.year", 1952), probability=0.25),
            stratum(term(1, "K.keyword", 1), probability=0.1),
            stratum(term(2, "T.phonetic", 3), probability=0.25),
        )
        return provider().extend_with_conjunctions(seeds, **options)

    def test_it_finds_the_cross_table_conjunction_with_its_exact_joint_count(self):
        found = {item.stratum_id: item for item in self.search()}
        self.assertTrue(found)
        pairs = [item for item in found.values() if item.column_indices == (0, 1)]
        self.assertEqual(len(pairs), 1)
        # 4 of 40 rows carry both, which is neither marginal.
        self.assertEqual(pairs[0].foj_count, 4.0)
        self.assertAlmostEqual(pairs[0].probability, 0.1)

    def test_a_region_with_too_few_joint_rows_is_pruned(self):
        found = self.search(minimum_joint_rows=5.0)
        self.assertFalse([item for item in found if item.column_indices == (0, 1)])

    def test_same_table_pairs_are_skipped_when_distinct_tables_are_required(self):
        found = self.search()
        self.assertFalse([item for item in found if item.column_indices == (0, 2)])
        relaxed = self.search(require_distinct_tables=False)
        self.assertTrue([item for item in relaxed if item.column_indices == (0, 2)])

    def test_depth_one_requests_produce_nothing(self):
        self.assertEqual(self.search(max_depth=1), ())

    def test_a_provider_without_rows_cannot_search(self):
        source = provider()
        rowless = ExactRootStratumProvider(
            source.metadata, source.column_masses, source="root_jct"
        )
        self.assertFalse(rowless.supports_conjunctive_strata)
        self.assertEqual(
            rowless.extend_with_conjunctions(
                (stratum(term(0, "T.year", 1952)),),
                n_total=1000,
                predicate_probabilities=PROBABILITIES,
                threshold=100.0,
                max_depth=2,
                beam_width=4,
                minimum_joint_rows=1.0,
                require_distinct_tables=True,
            ),
            (),
        )


class DiscoveryTest(unittest.TestCase):
    def discover(self, source, **conjunctive):
        return source.discover(
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1000.0,
            max_selected_strata=16,
            conjunctive=conjunctive,
        )

    def test_conjunctions_are_selected_when_enabled(self):
        source = provider()
        selected = self.discover(source, enabled=True, max_depth=2, minimum_joint_rows=1.0)
        self.assertTrue(selected)
        self.assertTrue(any(item.is_conjunctive for item in selected))
        self.assertTrue(source.conjunctive_status["applied"])
        self.assertGreater(source.conjunctive_status["conjunctive_selected"], 0)

    def test_disabled_discovery_stays_depth_one(self):
        source = provider()
        selected = self.discover(source)
        self.assertTrue(selected)
        self.assertFalse(any(item.is_conjunctive for item in selected))
        self.assertFalse(source.conjunctive_status["applied"])

    def test_a_rowless_provider_falls_back_and_records_why(self):
        source = provider()
        rowless = ExactRootStratumProvider(
            source.metadata, source.column_masses, source="root_jct"
        )
        selected = self.discover(rowless, enabled=True, max_depth=2)
        self.assertFalse(any(item.is_conjunctive for item in selected))
        self.assertTrue(rowless.conjunctive_status["requested"])
        self.assertFalse(rowless.conjunctive_status["applied"])
        self.assertIn("joint counts", rowless.conjunctive_status["reason"])

    def test_alpha_still_sums_to_one_under_the_share_cap(self):
        source = provider()
        selected = self.discover(
            source, enabled=True, max_depth=2, minimum_joint_rows=1.0, max_stratum_share=0.25
        )
        self.assertAlmostEqual(sum(item.alpha for item in selected), 1.0, places=9)
        self.assertLessEqual(max(item.alpha for item in selected), 0.25 + 1e-9)

    def test_discovery_excludes_unforceable_native_range_tails(self):
        meta = ModelMetadata(
            columns=(
                ColumnMetadata("T.x", ColumnKind.DATA, (1, 2, 3), table="T"),
            ),
            full_join_cardinality=10.0,
            join_root="T",
            join_tables=("T",),
            join_edges=(),
        )
        rows = np.asarray([[0]] * 5 + [[1]] * 3 + [[2]] * 2, dtype=np.int64)
        source = ExactRootStratumProvider.from_encoded_rows(
            meta,
            rows,
            root_column_semantics={"T.x": "ordered"},
        )
        range_rare_probabilities = {**PROBABILITIES, "range": 1.0e-6}
        selected = source.discover(
            n_total=1000,
            predicate_probabilities=range_rare_probabilities,
            minimum_expected_context_support=1e9,
            max_selected_strata=32,
        )

        self.assertTrue(selected)
        self.assertGreater(
            source.conjunctive_status[
                "unsupported_fixed_native_range_tail_candidates_excluded"
            ],
            0,
        )
        self.assertTrue(
            all(fixed_forced_predicate_is_supported(item, meta) for item in selected)
        )
        for item in selected:
            matching_row = next(
                row
                for row in rows
                if all(
                    term.contains_value(meta.columns[term.column_index].domain[row[term.column_index]])
                    for term in item.terms
                )
            )
            forced_predicate_for_stratum(
                item,
                encoded_row=matching_row,
                metadata=meta,
            )


class AllocationCapTest(unittest.TestCase):
    def test_the_cap_is_enforced_and_renormalized(self):
        capped = _capped_allocation(np.array([0.9, 0.05, 0.05]), 0.5)
        self.assertAlmostEqual(float(capped.sum()), 1.0, places=9)
        self.assertLessEqual(float(capped.max()), 0.5 + 1e-9)
        # The clipped stratum lands exactly on the cap and its surplus goes to
        # the others; conserving mass rather than merely renormalizing at the
        # end is what keeps the remaining shares meaningful.
        np.testing.assert_allclose(capped, [0.5, 0.25, 0.25], atol=1e-9)

    def test_the_surplus_is_water_filled_toward_the_cap_not_split_evenly(self):
        # Headroom-proportional redistribution: the stratum furthest below the
        # cap absorbs the most, so capping one conjunction does not silently
        # reorder the strata underneath it.
        capped = _capped_allocation(np.array([0.7, 0.2, 0.1]), 0.4)
        np.testing.assert_allclose(capped, [0.4, 0.32, 0.28], atol=1e-9)
        self.assertAlmostEqual(float(capped.sum()), 1.0, places=9)

    def test_an_unsatisfiable_cap_is_ignored_rather_than_deadlocking(self):
        alpha = np.array([0.5, 0.3, 0.2])
        np.testing.assert_allclose(_capped_allocation(alpha, 0.2), alpha)
        np.testing.assert_allclose(_capped_allocation(alpha, 0.0), alpha)


class ConfigValidationTest(unittest.TestCase):
    def config(self, **conjunctive):
        return {
            "dataset": {"type": "synthetic_full_join"},
            "model": {"type": "predicate_resmade", "hidden_sizes": [8]},
            "training": {"batch_size": 2, "learning_rate": 0.1, "epochs": 1, "steps_per_epoch": 1},
            "predicate_generation": {"enabled": True},
            "rare_support": {
                "enabled": True,
                "discovery": {"conjunctive": {"enabled": True, **conjunctive}},
            },
            "rare_auxiliary": {"enabled": True, "batch_size": 2, "beta": 0.25},
        }

    def test_a_well_formed_block_is_accepted(self):
        validate_config(self.config(max_depth=2, beam_width=8, minimum_joint_rows=4.0))

    def test_depth_one_conjunctions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_depth must be at least 2"):
            validate_config(self.config(max_depth=1))

    def test_a_non_positive_joint_row_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimum_joint_rows must be"):
            validate_config(self.config(minimum_joint_rows=0.0))

    def test_a_negative_materialized_row_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "minimum_materialized_rows"):
            validate_config(self.config(max_depth=2, minimum_materialized_rows=-1))

    def test_an_unknown_deficit_objective_is_rejected(self):
        config = self.config(max_depth=2)
        config["rare_support"]["discovery"]["deficit_objective"] = "mean"
        with self.assertRaisesRegex(ValueError, "deficit_objective"):
            validate_config(config)

    def test_both_deficit_objectives_are_accepted(self):
        for objective in ("clipped_sum", "log_ratio"):
            with self.subTest(deficit_objective=objective):
                config = self.config(max_depth=2)
                config["rare_support"]["discovery"]["deficit_objective"] = objective
                validate_config(config)

    def test_an_out_of_range_joint_row_fraction_is_rejected(self):
        for value in (-0.1, 1.5):
            with self.subTest(max_joint_row_fraction=value):
                with self.assertRaisesRegex(ValueError, "max_joint_row_fraction"):
                    validate_config(self.config(max_depth=2, max_joint_row_fraction=value))

    def test_a_dotted_column_name_is_rejected(self):
        # table.column is the NeuroCard root join-count dataframe spelling.  It
        # looks right, matches no model column, and silently empties the pool.
        for setting in ("required_extension_columns",):
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ValueError, "qualified model column name"):
                    validate_config(
                        self.config(max_depth=2, **{setting: ["movie_keyword.keyword_id"]})
                    )

    def test_a_dotted_candidate_column_name_is_rejected(self):
        config = self.config(max_depth=2)
        config["rare_support"]["discovery"]["candidate_column_names"] = [
            "title.production_year"
        ]
        with self.assertRaisesRegex(ValueError, "qualified model column name"):
            validate_config(config)

    def test_colon_qualified_candidate_names_are_accepted(self):
        config = self.config(max_depth=2)
        config["rare_support"]["discovery"]["candidate_column_names"] = [
            "title:production_year"
        ]
        validate_config(config)

    def test_required_extension_columns_must_be_a_list(self):
        with self.assertRaisesRegex(ValueError, "required_extension_columns"):
            validate_config(
                self.config(max_depth=2, required_extension_columns="title.kind_id")
            )

    def test_a_well_formed_precondition_block_is_accepted(self):
        validate_config(
            self.config(
                max_depth=2,
                minimum_materialized_rows=2_000_000,
                max_joint_row_fraction=0.01,
                require_conjunctions=True,
                required_extension_columns=["movie_keyword:keyword_id"],
            )
        )

    def test_an_out_of_range_share_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_stratum_share"):
            validate_config(self.config(max_stratum_share=1.5))

    def test_the_removed_term_subset_key_is_rejected_rather_than_ignored(self):
        # Silently ignoring it would leave a config that reads as if partial
        # conjunctions were still being generated.
        config = self.config(max_depth=2)
        config["rare_auxiliary"]["term_subset_probability"] = 0.5
        with self.assertRaisesRegex(ValueError, "was removed"):
            validate_config(config)


class LiveSourceCapabilityTest(unittest.TestCase):
    """The live NeuroCard path filters the root JCT one column at a time."""

    def source(self) -> LiveNeuroCardFullJoinSampleSource:
        # The guard is argument validation, so it must fire before the sampler
        # is touched; build the object without running __init__ to prove it.
        instance = object.__new__(LiveNeuroCardFullJoinSampleSource)
        instance._sampler = None
        instance._root_stratum_candidate_cache = {}
        instance._root_jct_weight_cache = {}
        return instance

    def test_it_reports_that_it_cannot_impose_a_conjunction(self):
        self.assertFalse(self.source().supports_conjunctive_strata)

    def test_a_conjunctive_stratum_is_refused_rather_than_silently_reduced(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        with self.assertRaisesRegex(ValueError, "is conjunctive"):
            self.source()._root_stratum_candidates(conjunction)

    def test_a_depth_one_stratum_gets_past_the_guard(self):
        # It fails later, on the uninitialized sampler, which is what proves the
        # conjunction guard did not swallow it.
        with self.assertRaises(RuntimeError):
            self.source()._root_stratum_candidates(stratum(term(0, "T.year", 1952)))


class RealizationProbabilityTest(unittest.TestCase):
    """A tail predicate also has to pick its exact threshold."""

    def tail(self, *, expected_lower, target_rows, probability=0.25):
        return RootDataStratum(
            stratum_id="T.year:ge:1951",
            column_index=0,
            column_name="T.year",
            region_type="lower_tail",
            lower=1951,
            probability=probability,
            expected_target_rows=target_rows,
            expected_lower_count=expected_lower,
        )

    def kwargs(self):
        return dict(
            metadata=metadata(),
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            root_table="T",
        )

    def test_a_tail_term_without_its_factor_is_refused_not_guessed(self):
        # Scoring it with the bare operator probability is the overestimate.
        bare = stratum(
            StratumTerm(
                column_index=0,
                column_name="T.year",
                region_type="lower_tail",
                lower=1951,
            )
        )
        with self.assertRaisesRegex(ValueError, "realization probability"):
            conditional_head_support(bare, 2, **self.kwargs())

    def test_the_factor_is_the_threshold_weighted_operator_probability(self):
        # 1000 rows in the region, but only 50 of the expected contexts pick
        # this exact threshold, so the per-row factor is 0.05, not p_lower=0.2.
        filled = with_realization_probabilities(
            self.tail(expected_lower=50.0, target_rows=1000.0), PROBABILITIES
        )
        self.assertAlmostEqual(filled.terms[0].realization_probability, 0.05, places=9)
        support = conditional_head_support(filled, 2, **self.kwargs())
        self.assertAlmostEqual(support, 1_000_000 * 0.25 * 0.05, places=3)

    def test_ignoring_threshold_selection_would_overestimate_fourfold_here(self):
        filled = with_realization_probabilities(
            self.tail(expected_lower=50.0, target_rows=1000.0), PROBABILITIES
        )
        honest = conditional_head_support(filled, 2, **self.kwargs())
        naive = 1_000_000 * 0.25 * PROBABILITIES["lower"]
        self.assertAlmostEqual(naive / honest, PROBABILITIES["lower"] / 0.05, places=6)
        self.assertLess(honest, naive)

    def test_the_gap_grows_with_column_cardinality(self):
        # The omitted factor is the mass-weighted mean of 1 / #valid thresholds,
        # so it shrinks as the ordered domain grows.  This is why the
        # overestimate reaches orders of magnitude on a real column.
        factors = []
        for size in (4, 64, 1024):
            probabilities = np.full(size, 1.0 / size)
            expected = _expected_lower_threshold_counts(
                probabilities, n_total=1_000_000, p_lower=0.2
            )
            # Lowest threshold: the whole column is in the region.
            factors.append(float(expected[0]) / 1_000_000.0)
        self.assertTrue(
            all(later < earlier for earlier, later in zip(factors, factors[1:])), factors
        )
        self.assertLess(factors[-1], 0.2)
        self.assertGreater(0.2 / factors[-1], 50.0)

    def test_an_equality_term_reduces_to_the_plain_operator_probability(self):
        filled = with_realization_probabilities(
            RootDataStratum(
                stratum_id="T.year:eq:1952",
                column_index=0,
                column_name="T.year",
                region_type="equality",
                value=1952,
                probability=0.25,
                expected_target_rows=1000.0,
                expected_equality_count=1000.0 * PROBABILITIES["equality"],
            ),
            PROBABILITIES,
        )
        self.assertAlmostEqual(
            filled.terms[0].realization_probability, PROBABILITIES["equality"], places=9
        )

    def test_a_conjunction_multiplies_both_terms_realized_factors(self):
        meta = metadata()
        tail = with_realization_probabilities(
            self.tail(expected_lower=50.0, target_rows=1000.0), PROBABILITIES
        )
        conjunction = RootDataStratum(
            stratum_id="tail&eq",
            column_index=0,
            column_name="T.year",
            region_type="lower_tail",
            terms=(tail.terms[0], term(2, "T.phonetic", 1)),
            probability=0.25,
            expected_target_rows=1000.0,
            realized_support_mass=0.25 * 0.05 * PROBABILITIES["equality"],
        )
        # Head 3 is on the child table, so the table-subset factor applies too.
        table_factor = table_subset_inclusion_probability(
            frozenset({"K"}), metadata=meta, root_table="T"
        )
        support = conditional_head_support(conjunction, 3, **self.kwargs())
        self.assertAlmostEqual(
            support,
            1_000_000 * 0.25 * 0.05 * PROBABILITIES["equality"] * table_factor,
            places=3,
        )


def chain_metadata() -> ModelMetadata:
    """Chain schema ``R -> P -> C``: forcing C needs P to stay rooted."""

    return ModelMetadata(
        columns=(
            ColumnMetadata("R.a", ColumnKind.DATA, (0, 1, OUTER_MISSING), table="R"),
            ColumnMetadata("P.b", ColumnKind.DATA, (0, 1, OUTER_MISSING), table="P"),
            ColumnMetadata("C.c", ColumnKind.DATA, (0, 1, OUTER_MISSING), table="C"),
            ColumnMetadata("I_R", ColumnKind.INDICATOR, (0, 1), table="R"),
            ColumnMetadata("I_P", ColumnKind.INDICATOR, (0, 1), table="P"),
            ColumnMetadata("I_C", ColumnKind.INDICATOR, (0, 1), table="C"),
        ),
        full_join_cardinality=8.0,
        join_root="R",
        join_tables=("R", "P", "C"),
        join_edges=(("R", "P"), ("P", "C")),
    )


class RootPathClosureTest(unittest.TestCase):
    EDGES = (("R", "P"), ("P", "C"))

    def test_forcing_a_grandchild_pulls_in_its_parent(self):
        self.assertEqual(
            root_paths_closure({"C"}, "R", self.EDGES), {"R", "P", "C"}
        )

    def test_the_root_alone_needs_nothing_added(self):
        self.assertEqual(root_paths_closure({"R"}, "R", self.EDGES), {"R"})

    def test_an_unreachable_table_is_an_error_not_a_silent_drop(self):
        with self.assertRaisesRegex(ValueError, "not reachable"):
            root_paths_closure({"Z"}, "R", self.EDGES + (("Z", "Y"),))

    def test_an_edgeless_graph_refuses_a_non_root_table(self):
        with self.assertRaisesRegex(ValueError, "no edges"):
            root_paths_closure({"P"}, "R", ())


class ForcedSubsetConnectivityTest(unittest.TestCase):
    """A forced term must never produce a subset disconnected from the root."""

    def generator(self) -> PredicateTrainingContextGenerator:
        return PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "table_subset_sampling": "neurocard_table_dropout_rooted",
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "seed": 0,
            }
        )

    def child_stratum(self) -> RootDataStratum:
        return RootDataStratum(
            stratum_id="C.c:eq:1",
            column_index=2,
            column_name="C.c",
            region_type="equality",
            value=1,
        )

    def contexts_for(self, rows, seed):
        generated, _, _ = self.generator().generate_forced_stratum_batch(
            encoded_rows=np.asarray(rows, dtype=np.int64),
            metadata=chain_metadata(),
            strata=tuple([self.child_stratum()] * len(rows)),
            rng=np.random.default_rng(seed),
        )
        return generated

    def test_forcing_the_grandchild_keeps_the_subset_rooted(self):
        # Table dropout will omit P on many of these rows.  Before the
        # root-path closure, forcing C back in afterwards produced {R, C},
        # with C hanging off nothing.
        rows = [[1, 1, 1, 1, 1, 1]] * 64
        forced_any = False
        for context in self.contexts_for(rows, seed=7):
            tables = set(context.included_tables)
            if not tables:
                continue
            forced_any = True
            self.assertEqual(context.forced_term_columns, (2,))
            self.assertEqual(tables, {"R", "P", "C"}, context.included_tables)
        self.assertTrue(forced_any, "no context forced the stratum")

    def test_a_row_without_the_forced_table_degrades_to_wildcards(self):
        # P and C are both outer-missing, so no rooted subset can carry the
        # forced term and the context must pin nothing at all.
        missing = 2  # encoded id of OUTER_MISSING in the data domains
        rows = [[1, missing, missing, 1, 0, 0]] * 16
        contexts = self.contexts_for(rows, seed=3)
        self.assertTrue(contexts)
        for context in contexts:
            self.assertEqual(context.forced_term_columns, ())
            self.assertEqual(set(context.included_tables), set())

    def test_a_path_table_missing_from_the_row_also_degrades_to_wildcards(self):
        # C present but its parent P outer-missing: there is no rooted path to
        # the forced table, so the context must not be emitted as if there were.
        missing = 2
        rows = [[1, missing, 1, 1, 0, 1]] * 16
        for context in self.contexts_for(rows, seed=11):
            self.assertEqual(context.forced_term_columns, ())
            self.assertEqual(set(context.included_tables), set())


class ExactSupportEventTest(unittest.TestCase):
    """Only a context that pinned every term counts as an exact event."""

    def context(self, forced):
        return GeneratedTrainingContext(
            tokens=(),
            included_tables=frozenset(),
            inverse_fanout_columns=frozenset(),
            ordinary_predicates={},
            encoded_two_slot_row=None,
            forced_term_columns=tuple(forced),
        )

    def test_the_complete_conjunction_counts(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertTrue(forced_prefix_is_complete(self.context((0, 1)), conjunction))

    def test_an_absent_table_wildcard_context_does_not_count(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertFalse(forced_prefix_is_complete(self.context(()), conjunction))

    def test_a_partial_prefix_does_not_count(self):
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertFalse(forced_prefix_is_complete(self.context((0,)), conjunction))

    def test_a_depth_one_stratum_still_counts(self):
        self.assertTrue(
            forced_prefix_is_complete(self.context((0,)), stratum(term(0, "T.year", 1952)))
        )


class RealizationRouteTest(unittest.TestCase):
    """The rarest route decides both the score and the forced predicate."""

    def equality_stratum(self, *, equality, native_range):
        return RootDataStratum(
            stratum_id="T.year:eq:1952",
            column_index=0,
            column_name="T.year",
            region_type="equality",
            value=1952,
            probability=0.25,
            expected_target_rows=250.0,
            expected_equality_count=equality,
            expected_range_support=native_range,
        )

    def test_a_native_range_bottleneck_is_recorded_on_the_term(self):
        filled = with_realization_probabilities(
            self.equality_stratum(equality=50.0, native_range=1.0), PROBABILITIES
        )
        self.assertEqual(filled.terms[0].realization_family, "range")
        self.assertAlmostEqual(filled.terms[0].realization_probability, 1.0 / 250.0)

    def test_the_equality_route_wins_when_it_is_the_rarer_one(self):
        filled = with_realization_probabilities(
            self.equality_stratum(equality=1.0, native_range=50.0), PROBABILITIES
        )
        self.assertEqual(filled.terms[0].realization_family, "equality")

    def test_a_native_range_bottleneck_is_forced_as_a_range_not_an_equality(self):
        # This is the regression: scoring used to overwrite support_bottleneck,
        # so the generator emitted "= v" for a stratum found rare as "[v, v]"
        # and trained a different event from the one discovery selected.
        meta = ModelMetadata(
            columns=(
                ColumnMetadata("T.year", ColumnKind.DATA, (1950, 1951, 1952), table="T"),
            ),
            full_join_cardinality=3.0,
            join_root="T",
            join_tables=("T",),
            join_edges=(),
        )
        filled = with_realization_probabilities(
            self.equality_stratum(equality=50.0, native_range=1.0), PROBABILITIES
        )
        row = np.array([2], dtype=np.int64)
        for source in (filled, filled.terms[-1]):
            token = forced_predicate_for_stratum(
                source, encoded_row=row, metadata=meta
            )
            self.assertEqual(token.op, PredicateOp.RANGE, source)
            self.assertEqual((token.value, token.upper), (1952, 1952))

    def test_the_route_survives_conditional_rescoring(self):
        # with_conditional_score rewrites support_bottleneck, which is exactly
        # why the route cannot live there.
        scored = with_conditional_score(
            self.equality_stratum(equality=50.0, native_range=1.0),
            metadata=metadata(),
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=100.0,
            root_table="T",
        )
        self.assertNotEqual(scored.support_bottleneck, "native_range")
        self.assertEqual(scored.terms[-1].realization_family, "range")

    def test_fixed_native_range_safety_keeps_equality_and_singleton_tails(self):
        meta = ModelMetadata(
            columns=(
                ColumnMetadata("T.year", ColumnKind.DATA, (1950, 1951, 1952), table="T"),
            ),
            full_join_cardinality=3.0,
            join_root="T",
            join_tables=("T",),
            join_edges=(),
        )

        def routed(term):
            return RootDataStratum(
                stratum_id=f"safe:{term.region_type}",
                column_index=term.column_index,
                column_name=term.column_name,
                region_type=term.region_type,
                value=term.value,
                lower=term.lower,
                upper=term.upper,
                terms=(term,),
                probability=1.0 / 3.0,
                expected_target_rows=1.0,
            )

        equality = routed(
            StratumTerm(
                0,
                "T.year",
                "equality",
                value=1951,
                realization_family="range",
            )
        )
        singleton_lower = routed(
            StratumTerm(
                0,
                "T.year",
                "lower_tail",
                lower=1952,
                realization_family="range",
            )
        )
        singleton_upper = routed(
            StratumTerm(
                0,
                "T.year",
                "upper_tail",
                upper=1950,
                realization_family="range",
            )
        )
        self.assertTrue(fixed_forced_predicate_is_supported(equality, meta))
        self.assertTrue(fixed_forced_predicate_is_supported(singleton_lower, meta))
        self.assertTrue(fixed_forced_predicate_is_supported(singleton_upper, meta))

    def test_fixed_native_range_safety_rejects_a_non_singleton_tail_term(self):
        meta = ModelMetadata(
            columns=(
                ColumnMetadata("T.year", ColumnKind.DATA, (1950, 1951, 1952), table="T"),
            ),
            full_join_cardinality=3.0,
            join_root="T",
            join_tables=("T",),
            join_edges=(),
        )
        tail = RootDataStratum(
            stratum_id="unsafe:tail",
            column_index=0,
            column_name="T.year",
            region_type="lower_tail",
            lower=1951,
            terms=(
                StratumTerm(
                    0,
                    "T.year",
                    "lower_tail",
                    lower=1951,
                    realization_family="range",
                ),
            ),
            probability=2.0 / 3.0,
            expected_target_rows=2.0,
        )
        self.assertFalse(fixed_forced_predicate_is_supported(tail, meta))


class PerRowFactorTest(unittest.TestCase):
    """Per-row factors must reproduce the depth-1 expectation vectors."""

    VALUES = (10.0, 20.0, 30.0, 40.0)
    COUNTS = np.array([4.0, 3.0, 2.0, 1.0])

    def probabilities(self):
        return self.COUNTS / self.COUNTS.sum()

    def factors(self, region_type, boundary, family):
        kwargs = {"value": boundary} if region_type == "equality" else (
            {"lower": boundary} if region_type == "lower_tail" else {"upper": boundary}
        )
        return per_value_realization_factors(
            StratumTerm(
                column_index=0,
                column_name="c",
                region_type=region_type,
                realization_family=family,
                **kwargs,
            ),
            values=self.VALUES,
            predicate_probabilities=PROBABILITIES,
        )

    def test_every_route_matches_its_discovery_vector(self):
        probabilities = self.probabilities()
        lower = _expected_lower_threshold_counts(probabilities, n_total=1, p_lower=0.2)
        upper = _expected_upper_threshold_counts(probabilities, n_total=1, p_upper=0.2)
        range_eq, range_lower, range_upper = _expected_native_range_interval_counts(
            probabilities, n_total=1, p_range=0.2
        )
        cases = [
            ("lower_tail", "lower", lower),
            ("upper_tail", "upper", upper),
            ("equality", "range", range_eq),
            ("lower_tail", "range", range_lower),
            ("upper_tail", "range", range_upper),
        ]
        for region_type, family, expected in cases:
            for rank, boundary in enumerate(self.VALUES):
                factors = self.factors(region_type, boundary, family)
                self.assertAlmostEqual(
                    float(factors @ probabilities),
                    float(expected[rank]),
                    places=12,
                    msg=f"{region_type}/{family} at rank {rank}",
                )

    def test_a_boundary_outside_the_domain_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "outside the column domain"):
            self.factors("lower_tail", 99.0, "lower")

    def test_the_factor_is_zero_outside_the_region(self):
        factors = self.factors("lower_tail", 30.0, "lower")
        self.assertEqual(list(factors[:2]), [0.0, 0.0])
        self.assertTrue(all(value > 0.0 for value in factors[2:]))


class JointRealizationMassTest(unittest.TestCase):
    """Correlated terms must be averaged jointly, not multiplied marginally."""

    def provider_and_terms(self):
        # 40 rows.  year id 2 covers rows 0..9; keyword id 1 covers rows 0..3.
        # The two overlap on 4 rows, far from independent.
        source = provider()
        filled = [
            with_realization_probabilities(seed, PROBABILITIES).terms[0]
            for seed in (
                RootDataStratum(
                    stratum_id="T.year:eq:2",
                    column_index=0,
                    column_name="T.year",
                    region_type="equality",
                    value=1952,
                    probability=10.0 / 40.0,
                    expected_target_rows=10.0,
                    expected_equality_count=10.0 * PROBABILITIES["equality"],
                ),
                RootDataStratum(
                    stratum_id="K.keyword:eq:1",
                    column_index=1,
                    column_name="K.keyword",
                    region_type="equality",
                    value=1,
                    probability=4.0 / 40.0,
                    expected_target_rows=4.0,
                    expected_equality_count=4.0 * PROBABILITIES["equality"],
                ),
            )
        ]
        return source, tuple(filled)

    def test_the_joint_mass_is_the_exact_overlap_not_the_product(self):
        source, terms = self.provider_and_terms()
        joint = source.realized_support_mass(terms, PROBABILITIES)
        # 4 of 40 rows satisfy both, each realizing at p_equality squared.
        self.assertAlmostEqual(
            joint, (4.0 / 40.0) * PROBABILITIES["equality"] ** 2, places=12
        )
        independent = (10.0 / 40.0) * (4.0 / 40.0) * PROBABILITIES["equality"] ** 2
        self.assertGreater(joint, independent)
        self.assertAlmostEqual(joint / independent, 4.0, places=9)

    def test_a_depth_one_mass_matches_the_discovery_statistic(self):
        source, terms = self.provider_and_terms()
        self.assertAlmostEqual(
            source.realized_support_mass(terms[:1], PROBABILITIES),
            (10.0 / 40.0) * PROBABILITIES["equality"],
            places=12,
        )

    def test_disjoint_terms_give_zero_rather_than_a_positive_product(self):
        source, terms = self.provider_and_terms()
        disjoint = StratumTerm(
            column_index=0,
            column_name="T.year",
            region_type="equality",
            value=1950,
            realization_probability=PROBABILITIES["equality"],
            realization_family="equality",
        )
        # year 1950 and year 1952 cannot both hold, and the terms are on the
        # same column, so the marginal product would be positive nonsense.
        self.assertEqual(
            source.realized_support_mass((terms[0], disjoint), PROBABILITIES), 0.0
        )

    def test_a_conjunction_scored_without_the_joint_mass_is_refused(self):
        conjunction = RootDataStratum(
            stratum_id="a&b",
            column_index=0,
            column_name="T.year",
            region_type="equality",
            terms=(term(0, "T.year", 1952), term(1, "K.keyword", 1)),
            probability=0.1,
        )
        with self.assertRaisesRegex(ValueError, "realized support mass"):
            conditional_head_support(
                conjunction,
                2,
                metadata=metadata(),
                n_total=1000,
                predicate_probabilities=PROBABILITIES,
                root_table="T",
            )

    def test_the_beam_search_attaches_the_exact_mass(self):
        seeds = (
            stratum(term(0, "T.year", 1952), probability=0.25),
            stratum(term(1, "K.keyword", 1), probability=0.1),
        )
        found = provider().extend_with_conjunctions(
            seeds,
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=1000.0,
            max_depth=2,
            beam_width=8,
            minimum_joint_rows=1.0,
            require_distinct_tables=True,
        )
        self.assertTrue(found)
        for candidate in found:
            self.assertIsNotNone(candidate.realized_support_mass)
            # Exactly the overlap, not the product of the two marginals.
            self.assertAlmostEqual(
                candidate.realized_support_mass,
                (4.0 / 40.0) * PROBABILITIES["equality"] ** 2,
                places=12,
            )

    def test_the_search_result_scores_from_the_joint_mass(self):
        seeds = (
            stratum(term(0, "T.year", 1952), probability=0.25),
            stratum(term(1, "K.keyword", 1), probability=0.1),
        )
        found = provider().extend_with_conjunctions(
            seeds,
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
            max_depth=2,
            beam_width=8,
            minimum_joint_rows=1.0,
            require_distinct_tables=True,
        )
        meta = metadata()
        for candidate in found:
            head = served_head_indices(candidate, meta)[0]
            expected = (
                1_000_000
                * candidate.realized_support_mass
                * table_subset_inclusion_probability(
                    candidate.tables(meta) | {meta.columns[head].table},
                    metadata=meta,
                    root_table="T",
                )
            )
            self.assertAlmostEqual(
                conditional_head_support(
                    candidate,
                    head,
                    metadata=meta,
                    n_total=1_000_000,
                    predicate_probabilities=PROBABILITIES,
                    root_table="T",
                ),
                expected,
                places=6,
            )


class MultiHopTableSurvivalTest(unittest.TestCase):
    """Keeping a grandchild also means keeping its parent."""

    def meta(self):
        return chain_metadata()

    def test_a_grandchild_pays_for_its_ancestor_too(self):
        factor = table_subset_inclusion_probability(
            frozenset({"C"}), metadata=self.meta(), root_table="R"
        )
        # Three tables, num_dropped ~ U{1,2}, P(drop|d) = d/3 per non-root
        # table, and both P and C must survive: ((2/3)^2 + (1/3)^2) / 2.
        self.assertAlmostEqual(factor, 5.0 / 18.0, places=12)

    def test_a_direct_child_is_unchanged(self):
        self.assertAlmostEqual(
            table_subset_inclusion_probability(
                frozenset({"P"}), metadata=self.meta(), root_table="R"
            ),
            0.5,
            places=12,
        )

    def test_the_grandchild_is_strictly_rarer_than_the_child(self):
        meta = self.meta()
        self.assertLess(
            table_subset_inclusion_probability(
                frozenset({"C"}), metadata=meta, root_table="R"
            ),
            table_subset_inclusion_probability(
                frozenset({"P"}), metadata=meta, root_table="R"
            ),
        )

    def test_requiring_both_hops_matches_requiring_the_grandchild(self):
        meta = self.meta()
        self.assertAlmostEqual(
            table_subset_inclusion_probability(
                frozenset({"P", "C"}), metadata=meta, root_table="R"
            ),
            table_subset_inclusion_probability(
                frozenset({"C"}), metadata=meta, root_table="R"
            ),
            places=12,
        )


def chain_rows() -> np.ndarray:
    """40 rows on the R -> P -> C chain, all three tables present."""

    return np.array(
        [[index % 3, index % 3, index % 2, 1, 1, 1] for index in range(40)],
        dtype=np.int64,
    )


def chain_data_metadata() -> ModelMetadata:
    """``chain_metadata`` with wider DATA domains so regions are non-trivial."""

    return ModelMetadata(
        columns=(
            ColumnMetadata("R.a", ColumnKind.DATA, (0, 1, 2), table="R"),
            ColumnMetadata("P.b", ColumnKind.DATA, (0, 1, 2), table="P"),
            ColumnMetadata("C.c", ColumnKind.DATA, (0, 1), table="C"),
            ColumnMetadata("I_R", ColumnKind.INDICATOR, (0, 1), table="R"),
            ColumnMetadata("I_P", ColumnKind.INDICATOR, (0, 1), table="P"),
            ColumnMetadata("I_C", ColumnKind.INDICATOR, (0, 1), table="C"),
        ),
        full_join_cardinality=40.0,
        join_root="R",
        join_tables=("R", "P", "C"),
        join_edges=(("R", "P"), ("P", "C")),
    )


class ProductionProviderDiscoveryTest(unittest.TestCase):
    """The provider the training pipeline actually builds must find conjunctions."""

    def provider(self):
        return ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows()
        )

    def discover(self, source=None, **conjunctive):
        options = dict(
            enabled=True,
            max_depth=2,
            beam_width=8,
            seed_width=8,
            minimum_joint_rows=1.0,
            require_distinct_tables=True,
        )
        options.update(conjunctive)
        source = source or self.provider()
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=8,
            conjunctive=options,
        )
        return source, selected

    def test_seed_masses_stay_root_only(self):
        source = self.provider()
        meta = chain_data_metadata()
        self.assertEqual(
            [meta.columns[index].table for index in source.column_masses], ["R"]
        )

    def test_extension_masses_cover_every_data_column(self):
        source = self.provider()
        meta = chain_data_metadata()
        self.assertEqual(
            sorted(meta.columns[index].name for index in source.extension_column_masses),
            ["C.c", "P.b", "R.a"],
        )

    def test_cross_table_conjunctions_are_actually_discovered(self):
        # The regression: with a root-only extension pool every extension was
        # rejected by require_distinct_tables and discovery was silently inert.
        source, selected = self.discover()
        self.assertGreater(source.conjunctive_status["conjunctive_candidates"], 0)
        self.assertGreater(source.conjunctive_status["conjunctive_selected"], 0)
        self.assertIn(2, [item.depth for item in selected])

    def test_the_conjunctions_span_tables(self):
        meta = chain_data_metadata()
        _, selected = self.discover()
        spanning = [
            item for item in selected if item.is_conjunctive and len(item.tables(meta)) > 1
        ]
        self.assertTrue(spanning, [item.stratum_id for item in selected])

    def test_depth_one_strata_remain_root_only(self):
        # Non-root columns may extend a conjunction but must never become a
        # depth-1 stratum, which the live sampler could not impose.
        meta = chain_data_metadata()
        _, selected = self.discover()
        for item in selected:
            if item.depth == 1:
                self.assertEqual(
                    meta.columns[item.column_index].table, "R", item.stratum_id
                )

    def test_applied_is_false_when_nothing_conjunctive_was_selected(self):
        # A root-only extension pool is exactly the old behaviour; it must now
        # report itself instead of claiming the feature was applied.
        source = self.provider()
        source.extension_column_masses = dict(source.column_masses)
        source, selected = self.discover(source=source)
        self.assertEqual(source.conjunctive_status["cross_table_extension_candidates"], 0)
        self.assertFalse(source.conjunctive_status["applied"])
        self.assertIn(
            "require_distinct_tables", str(source.conjunctive_status["reason"])
        )
        self.assertIn("root table", str(source.conjunctive_status["reason"]))
        self.assertEqual({item.depth for item in selected}, {1})

    def test_an_impossible_joint_row_floor_reports_the_search_not_the_pool(self):
        # max_joint_row_fraction off, or the floor would be refused outright
        # before the search ever runs.
        source, _ = self.discover(minimum_joint_rows=1e9, max_joint_row_fraction=0.0)
        self.assertEqual(source.conjunctive_status["conjunctive_candidates"], 0)
        self.assertFalse(source.conjunctive_status["applied"])
        self.assertIn("minimum_joint_rows", str(source.conjunctive_status["reason"]))

    def test_a_successful_run_reports_no_reason(self):
        source, _ = self.discover()
        self.assertTrue(source.conjunctive_status["applied"])
        self.assertIsNone(source.conjunctive_status["reason"])

    def test_a_too_wide_column_is_excluded_from_both_pools(self):
        # Widening the extension pool must honour the same cap the root-JCT
        # provider does, or a high-cardinality column allocates a mass vector.
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows(), max_domain_size=2
        )
        meta = chain_data_metadata()
        kept = {meta.columns[index].name for index in source.extension_column_masses}
        self.assertEqual(kept, {"C.c"})
        self.assertEqual(dict(source.column_masses), {})

    def test_the_extension_width_caps_the_pool(self):
        source, _ = self.discover(extension_width=1)
        self.assertEqual(source.conjunctive_status["extension_candidates"], 1)

    def test_the_narrowest_width_still_spends_itself_on_a_usable_extension(self):
        # The width is a budget, so it must not be spent on candidates the
        # search cannot use.  Ranking before filtering let high-scoring
        # root-table regions consume it and left discovery inert.
        meta = chain_data_metadata()
        source, selected = self.discover(extension_width=1)
        status = source.conjunctive_status
        self.assertEqual(status["cross_table_extension_candidates"], 1)
        self.assertGreater(status["conjunctive_candidates"], 0)
        self.assertTrue(status["applied"])
        self.assertTrue(
            [item for item in selected if item.is_conjunctive and len(item.tables(meta)) > 1]
        )

    def test_every_width_from_one_upward_produces_a_conjunction(self):
        # The defect showed as a threshold: widths 1..3 were inert and only 4
        # happened to reach a usable candidate.
        for width in range(1, 6):
            with self.subTest(extension_width=width):
                source, _ = self.discover(extension_width=width)
                self.assertGreater(
                    source.conjunctive_status["conjunctive_candidates"], 0
                )
                self.assertTrue(source.conjunctive_status["applied"])

    def test_the_marginal_pool_alone_still_filters_before_truncating(self):
        # With the conditional route switched off the marginal pool is the only
        # one, so its filter-then-truncate order is what decides whether a
        # usable cross-table extension survives a single slot.
        source, _ = self.discover(extension_width=1, conditional_extension_width=0)
        status = source.conjunctive_status
        self.assertEqual(status["conditional_extension_candidates"], 0)
        self.assertEqual(status["extension_candidates"], 1)
        self.assertEqual(status["cross_table_extension_candidates"], 1)
        self.assertGreater(status["conjunctive_candidates"], 0)

    def test_ineligible_candidates_are_reported_but_not_pooled(self):
        source, _ = self.discover(extension_width=8)
        status = source.conjunctive_status
        # Root-table regions are still scored; they just cannot be spent.
        self.assertGreater(
            status["scored_extension_candidates"], status["eligible_extension_candidates"]
        )
        self.assertEqual(
            status["eligible_extension_candidates"],
            status["cross_table_extension_candidates"],
        )

    def test_no_eligible_candidate_is_reported_as_such(self):
        source = self.provider()
        source.extension_column_masses = dict(source.column_masses)
        source, _ = self.discover(source=source)
        self.assertEqual(source.conjunctive_status["eligible_extension_candidates"], 0)
        self.assertFalse(source.conjunctive_status["applied"])
        self.assertIn("extend any seed", str(source.conjunctive_status["reason"]))


class ExtensionEligibilityTest(unittest.TestCase):
    """What may spend a slot of the extension width."""

    def meta(self):
        return metadata()

    def can_extend(self, candidate, seeds, *, require_distinct_tables=True):
        return _can_extend_any_seed(
            candidate,
            seeds,
            metadata=self.meta(),
            require_distinct_tables=require_distinct_tables,
        )

    def test_a_candidate_on_a_seeds_own_column_cannot_extend_it(self):
        # A stratum pins one region per column, so a second region on the same
        # column is not an extension.  This holds with or without the distinct
        # table rule, which would otherwise mask it.
        seed = stratum(term(0, "T.year", 1952))
        other_region = stratum(term(0, "T.year", 1951))
        self.assertFalse(self.can_extend(other_region, (seed,)))
        self.assertFalse(
            self.can_extend(other_region, (seed,), require_distinct_tables=False)
        )

    def test_a_candidate_on_the_seeds_table_is_refused_under_the_distinct_rule(self):
        seed = stratum(term(0, "T.year", 1952))
        same_table = stratum(term(2, "T.phonetic", 1))
        self.assertFalse(self.can_extend(same_table, (seed,)))
        self.assertTrue(
            self.can_extend(same_table, (seed,), require_distinct_tables=False)
        )

    def test_a_candidate_on_another_table_can_extend(self):
        seed = stratum(term(0, "T.year", 1952))
        self.assertTrue(self.can_extend(stratum(term(1, "K.keyword", 1)), (seed,)))

    def test_extending_any_one_seed_is_enough(self):
        seeds = (
            stratum(term(1, "K.keyword", 1)),
            stratum(term(0, "T.year", 1952)),
        )
        # Blocked by the first seed, usable with the second.
        self.assertTrue(self.can_extend(stratum(term(3, "K.extra", 1)), seeds))

    def test_no_seeds_means_nothing_is_eligible(self):
        self.assertFalse(self.can_extend(stratum(term(1, "K.keyword", 1)), ()))


class EarlyStoppingMonitorConfigTest(unittest.TestCase):
    """Early stopping must say what it watches instead of inferring it."""

    def config(self, *, validation=None, monitor=None, patience=3000):
        config = {
            "dataset": {"type": "synthetic_full_join"},
            "model": {"type": "predicate_resmade", "hidden_sizes": [8]},
            "training": {
                "batch_size": 2,
                "learning_rate": 0.1,
                "epochs": 1,
                "steps_per_epoch": 1,
                "early_stopping_patience_steps": patience,
                "validation_interval_steps": 250,
            },
            "predicate_generation": {"enabled": True},
        }
        if monitor is not None:
            config["training"]["early_stopping_monitor"] = monitor
        if validation is not None:
            config["validation"] = validation
        return config

    def enabled_validation(self):
        return {"enabled": True, "interval_steps": 250, "fresh_sampler_batches": 2}

    def test_early_stopping_without_validation_is_refused(self):
        # The run that motivated this asked for early stopping every 250 steps,
        # got no validation source, and silently selected a checkpoint off the
        # per-step training loss.
        with self.assertRaisesRegex(ValueError, "per-step training loss"):
            validate_config(self.config())

    def test_watching_the_training_loss_is_allowed_when_declared(self):
        validate_config(self.config(monitor="train_loss"))

    def test_asking_for_validation_without_enabling_it_is_refused(self):
        with self.assertRaisesRegex(ValueError, "requires validation.enabled"):
            validate_config(self.config(monitor="validation"))

    def test_enabled_validation_needs_no_declaration(self):
        validate_config(self.config(validation=self.enabled_validation()))

    def test_an_unknown_monitor_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be validation or train_loss"):
            validate_config(self.config(monitor="loss"))

    def test_the_interval_requirement_still_applies(self):
        config = self.config(monitor="train_loss")
        config["training"]["validation_interval_steps"] = 0
        with self.assertRaisesRegex(ValueError, "validation_interval_steps"):
            validate_config(config)

    def test_no_early_stopping_needs_no_monitor(self):
        validate_config(self.config(patience=0))

    def test_the_resolver_is_what_both_the_validator_and_trainer_read(self):
        # The trainer used to infer the mode independently, which is how the
        # two came to disagree in the first place.
        self.assertEqual(
            resolve_early_stopping_monitor(self.config(validation=self.enabled_validation())),
            "validation",
        )
        self.assertEqual(resolve_early_stopping_monitor(self.config()), "train_loss")

    def test_a_declared_monitor_overrides_enabled_validation(self):
        # Declared always wins: a run may deliberately watch the training loss
        # while still recording validation history.
        config = self.config(
            validation=self.enabled_validation(), monitor="train_loss"
        )
        validate_config(config)
        self.assertEqual(resolve_early_stopping_monitor(config), "train_loss")


class ShippedConfigTest(unittest.TestCase):
    """The checked-in configs must stay consistent with these rules."""

    ROOT = pathlib.Path(__file__).resolve().parents[1] / "configs"

    def configs(self):
        # load_simple_yaml, not PyYAML: PyYAML is not a declared dependency
        # and this suite has to run wherever the project installs.
        for path in sorted(self.ROOT.glob("*.yaml")):
            yield path, load_simple_yaml(path)

    def test_every_early_stopping_config_declares_its_monitor_or_validates(self):
        for path, config in self.configs():
            training = config.get("training", {}) or {}
            patience = int(
                training.get(
                    "early_stopping_patience_steps",
                    training.get("early_stopping_patience", 0) or 0,
                )
                or 0
            )
            if patience <= 0:
                continue
            with self.subTest(config=path.name):
                validation_on = bool((config.get("validation", {}) or {}).get("enabled", False))
                self.assertTrue(
                    validation_on or training.get("early_stopping_monitor") is not None,
                    f"{path.name} early-stops without declaring what it monitors",
                )

    def test_the_package_does_not_import_undeclared_dependencies(self):
        # This suite itself imported PyYAML, which is not in pyproject, so it
        # failed on any machine that had not installed it incidentally.
        # load_simple_yaml exists precisely so it does not have to.
        # Scoped to the test suite: model/src reaches the vendored NeuroCard
        # modules through sys.path at runtime, so those are legitimately not
        # pip dependencies, but a test has nothing to load them from.
        declared = {"numpy", "pandas", "psycopg", "torch", "pytest", "model"}
        package = pathlib.Path(__file__).resolve().parent
        offenders = []
        for path in sorted(package.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    root = name.split(".")[0]
                    if not root or root in declared:
                        continue
                    if root in sys.stdlib_module_names:
                        continue
                    offenders.append(f"{path.relative_to(package)}: {name}")
        self.assertEqual(
            offenders, [], f"tests import undeclared dependencies: {offenders}"
        )

    def test_a_conjunctive_config_declares_what_its_fixture_must_hold(self):
        # A conjunctive run on a fixture estimates rarity from the row count,
        # so a config that enables it has to state the scale it needs or state
        # that it is deliberately a smoke run.
        for path, config in self.configs():
            discovery = ((config.get("rare_support", {}) or {}).get("discovery", {}) or {})
            conjunctive = discovery.get("conjunctive", {}) or {}
            if not bool(conjunctive.get("enabled", False)):
                continue
            dataset = config.get("dataset", {}) or {}
            if str(dataset.get("sampling_mode", "fixture")) == "live":
                continue
            with self.subTest(config=path.name):
                self.assertIn(
                    "minimum_materialized_rows",
                    conjunctive,
                    f"{path.name} enables conjunctive discovery on a fixture without "
                    "declaring the row count it needs",
                )

    def test_every_configured_column_name_is_colon_qualified(self):
        # The dotted spelling matches no model column, so a config using it
        # silently ends up with an empty candidate pool.
        for path, config in self.configs():
            discovery = ((config.get("rare_support", {}) or {}).get("discovery", {}) or {})
            conjunctive = discovery.get("conjunctive", {}) or {}
            names = list(discovery.get("candidate_column_names", ()) or ())
            names += list(conjunctive.get("required_extension_columns", ()) or ())
            for name in names:
                with self.subTest(config=path.name, column=name):
                    self.assertIn(":", str(name))
                    self.assertNotRegex(str(name), r"^[^:]*\.[^:]*$")

    def test_a_fixture_config_can_build_the_fixture_it_requires(self):
        # --rebuild-domains reads dataset.csv_directory unconditionally, so a
        # config that documents a rebuild without it cannot create its fixture.
        for path, config in self.configs():
            conjunctive = (
                ((config.get("rare_support", {}) or {}).get("discovery", {}) or {})
                .get("conjunctive", {})
                or {}
            )
            if int(conjunctive.get("minimum_materialized_rows", 0) or 0) <= 0:
                continue
            dataset = config.get("dataset", {}) or {}
            with self.subTest(config=path.name):
                self.assertIn(
                    "csv_directory",
                    dataset,
                    f"{path.name} requires a rebuilt fixture but has no csv_directory "
                    "for prepare_neurocard_data --rebuild-domains",
                )

    def test_a_rebuilt_fixture_does_not_overwrite_a_shared_one(self):
        owners = {}
        consumers = {}
        for path, config in self.configs():
            dataset = config.get("dataset", {}) or {}
            directory = dataset.get("prepared_directory")
            if directory is None:
                continue
            consumers.setdefault(str(directory), []).append(path.name)
            if not bool(dataset.get("prepared_fixture_read_only", False)):
                owners.setdefault(str(directory), []).append(path.name)
        for path, config in self.configs():
            conjunctive = (
                ((config.get("rare_support", {}) or {}).get("discovery", {}) or {})
                .get("conjunctive", {})
                or {}
            )
            if int(conjunctive.get("minimum_materialized_rows", 0) or 0) <= 0:
                continue
            dataset = config.get("dataset", {}) or {}
            directory = str(dataset.get("prepared_directory"))
            with self.subTest(config=path.name):
                directory_owners = owners.get(directory, [])
                if bool(dataset.get("prepared_fixture_read_only", False)):
                    self.assertEqual(
                        len(directory_owners),
                        1,
                        f"{path.name} reads {directory}, but that fixture must have "
                        f"exactly one rebuild owner; consumers={consumers.get(directory, [])}",
                    )
                else:
                    self.assertEqual(
                        directory_owners,
                        [path.name],
                        f"{path.name} rebuilds {directory}, which another config also owns",
                    )

    def test_the_conjunctive_job_light_config_can_actually_use_conjunctions(self):
        # Live sampling cannot impose a cross-table conjunction, so a config
        # that enables conjunctive discovery on sampling_mode: live validates
        # nothing.  At least one JOB-light config must be able to.
        usable = []
        for path, config in self.configs():
            discovery = ((config.get("rare_support", {}) or {}).get("discovery", {}) or {})
            if not bool((discovery.get("conjunctive", {}) or {}).get("enabled", False)):
                continue
            dataset = config.get("dataset", {}) or {}
            if str(dataset.get("sampling_mode", "fixture")) != "live":
                usable.append(path.name)
        self.assertTrue(
            any("job_light" in name for name in usable),
            f"no JOB-light config can exercise conjunctive discovery; usable: {usable}",
        )


class FixtureEndToEndTest(unittest.TestCase):
    """Discovery through conditional sampling on the real fixture source.

    This is the path the JOB-light fixture config takes.  Everything above
    tests a piece of it; this asserts the pieces connect, so a cluster run is
    not the first thing to find out that they do not.
    """

    PROBABILITIES = PROBABILITIES
    CONJUNCTIVE = {
        "enabled": True,
        "max_depth": 2,
        "beam_width": 64,
        "seed_width": 64,
        "extension_width": 64,
        "minimum_joint_rows": 10,
        "require_distinct_tables": True,
        "max_stratum_share": 0.1,
    }

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            columns=(
                ColumnMetadata(
                    "title:production_year", ColumnKind.DATA, (1950, 1960, 1970, 1980),
                    table="title",
                ),
                ColumnMetadata(
                    "movie_keyword:keyword_id", ColumnKind.DATA, (1, 2, 3, OUTER_MISSING),
                    table="movie_keyword",
                ),
                ColumnMetadata(
                    "I_movie_keyword", ColumnKind.INDICATOR, (0, 1), table="movie_keyword"
                ),
                ColumnMetadata(
                    "F_title_to_mk", ColumnKind.FANOUT, (1, 2), table="title",
                    fanout_source="title->movie_keyword",
                ),
            ),
            full_join_cardinality=200.0,
            join_root="title",
            join_tables=("title", "movie_keyword"),
            join_edges=(("title", "movie_keyword"),),
        )

    def rows(self) -> np.ndarray:
        return np.array([[index % 4, index % 4, 1, index % 2] for index in range(200)],
                        dtype=np.int64)

    def prepared_source(self, meta, rows):
        directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "metadata": meta.to_json_dict(),
                    "domains_complete": True,
                    "metadata_source": COMPLETE_METADATA_SOURCE,
                }
            ),
            encoding="utf-8",
        )
        np.save(directory / "sample_rows.npy", rows)
        return NeuroCardFullJoinSampleSource(directory, sampling_mode="fixture")

    def discovered(self, meta, rows, **overrides):
        options = dict(self.CONJUNCTIVE)
        options.update(overrides)
        provider = ExactRootStratumProvider.from_encoded_rows(
            meta, rows, max_domain_size=4096
        )
        selected = provider.discover(
            n_total=10_000_000,
            predicate_probabilities=self.PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=8,
            conjunctive=options,
        )
        return provider, selected

    def test_a_cross_table_conjunction_is_discovered_and_sampled(self):
        meta, rows = self.metadata(), self.rows()
        provider, selected = self.discovered(meta, rows)
        self.assertTrue(provider.conjunctive_status["applied"])
        self.assertIsNone(provider.conjunctive_status["reason"])
        conjunctions = [item for item in selected if item.is_conjunctive]
        self.assertTrue(conjunctions, [item.stratum_id for item in selected])
        spanning = [item for item in conjunctions if len(item.tables(meta)) > 1]
        self.assertTrue(spanning, [item.stratum_id for item in conjunctions])

        source = self.prepared_source(meta, rows)
        drawn = source.sample_root_strata_rows(
            [spanning[0]] * 16, rng=np.random.default_rng(0)
        )
        encoded = np.asarray(getattr(drawn, "encoded_values", drawn))
        self.assertEqual(encoded.shape, (16, len(meta.columns)))
        for row in encoded:
            self.assertTrue(
                spanning[0].contains_row(row, meta),
                f"{spanning[0].stratum_id} not satisfied by {row.tolist()}",
            )

    def test_the_sampled_rows_satisfy_every_term_not_just_the_last(self):
        meta, rows = self.metadata(), self.rows()
        _, selected = self.discovered(meta, rows)
        spanning = next(
            item for item in selected if item.is_conjunctive and len(item.tables(meta)) > 1
        )
        source = self.prepared_source(meta, rows)
        encoded = np.asarray(
            source.sample_root_strata_rows([spanning] * 32, rng=np.random.default_rng(1))
            .encoded_values
        )
        for term in spanning.terms:
            column = meta.columns[term.column_index]
            values = [column.domain[int(value)] for value in encoded[:, term.column_index]]
            self.assertTrue(
                all(term.contains_value(value) for value in values),
                f"term on {term.column_name} violated",
            )

    def test_the_fixture_provider_reports_the_capability(self):
        meta, rows = self.metadata(), self.rows()
        provider = ExactRootStratumProvider.from_encoded_rows(meta, rows)
        self.assertTrue(provider.supports_conjunctive_strata)


class DiscoveryPreconditionTest(unittest.TestCase):
    """A run that cannot estimate rare support must refuse, not produce numbers."""

    def provider(self, *, rows=None, cap=None, names=None):
        return ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(),
            chain_rows() if rows is None else rows,
            max_domain_size=cap,
            column_names=names,
        )

    def discover(self, source, **conjunctive):
        options = dict(
            enabled=True,
            max_depth=2,
            beam_width=8,
            seed_width=8,
            minimum_joint_rows=1.0,
            require_distinct_tables=True,
        )
        options.update(conjunctive)
        return source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=8,
            conjunctive=options,
        )

    def test_a_fixture_below_the_row_floor_is_refused(self):
        # The shipped JOB-light fixture held 512 rows against a 2.1e12-row join,
        # which cannot tell a rare joint region from an absent one.
        with self.assertRaisesRegex(ValueError, "materialized rows"):
            self.discover(self.provider(), minimum_materialized_rows=100_000)

    def test_the_refusal_names_the_rebuild_command(self):
        with self.assertRaisesRegex(ValueError, "prepare_neurocard_data"):
            self.discover(self.provider(), minimum_materialized_rows=100_000)

    def test_a_fixture_at_the_floor_is_accepted(self):
        self.discover(self.provider(), minimum_materialized_rows=40)

    def test_no_floor_keeps_the_previous_behaviour(self):
        self.discover(self.provider())

    def test_a_joint_floor_that_defines_away_rarity_is_refused(self):
        # minimum_joint_rows=100 on 512 rows demanded that a "rare" conjunction
        # cover a fifth of the sample.
        with self.assertRaisesRegex(ValueError, "not rare"):
            self.discover(
                self.provider(), minimum_joint_rows=100.0, max_joint_row_fraction=0.05
            )

    def test_the_fraction_guard_reports_both_sides(self):
        with self.assertRaisesRegex(ValueError, r"250\.0% of the 40 materialized rows"):
            self.discover(
                self.provider(), minimum_joint_rows=100.0, max_joint_row_fraction=0.05
            )

    def test_a_small_joint_floor_passes_the_fraction_guard(self):
        self.discover(self.provider(), minimum_joint_rows=1.0, max_joint_row_fraction=0.05)

    def test_the_fraction_guard_can_be_switched_off_for_a_smoke_fixture(self):
        self.discover(
            self.provider(), minimum_joint_rows=20.0, max_joint_row_fraction=1.0
        )

    def test_require_conjunctions_turns_an_empty_result_into_a_failure(self):
        with self.assertRaisesRegex(ValueError, "require_conjunctions"):
            self.discover(
                self.provider(),
                minimum_joint_rows=1e9,
                max_joint_row_fraction=0.0,
                require_conjunctions=True,
            )

    def test_the_refusal_carries_the_status_that_explains_it(self):
        # This is the refusal a correctly configured conjunctive run actually
        # hits, and it fires before anything can read the provider's status,
        # so the status has to travel on the exception.  Without it the audit
        # script can only report a traceback for the one failure it exists to
        # diagnose.
        with self.assertRaises(RareSupportDiscoveryError) as caught:
            self.discover(
                self.provider(),
                minimum_joint_rows=1e9,
                max_joint_row_fraction=0.0,
                require_conjunctions=True,
            )
        status = caught.exception.conjunctive_status
        self.assertTrue(status["requested"])
        self.assertFalse(status["applied"])
        self.assertIsNotNone(status["reason"])
        self.assertIn("row_floor_applied", status)

    def test_without_require_conjunctions_an_empty_result_only_reports(self):
        source = self.provider()
        self.discover(source, minimum_joint_rows=1e9, max_joint_row_fraction=0.0)
        self.assertFalse(source.conjunctive_status["applied"])
        self.assertIsNotNone(source.conjunctive_status["reason"])

    def test_the_status_records_what_the_allowlist_excluded(self):
        # candidate_column_names hides a column as effectively as the domain cap
        # does and leaves no other trace.  Omitting the columns that carried the
        # JOB-light tail from that list is what made three rounds of work aim at
        # the wrong column, with nothing in the output saying so.
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows(), column_names=["R.a", "C.c"]
        )
        self.discover(source=source)
        status = source.conjunctive_status
        self.assertEqual(status["columns_excluded_by_name"], ["P.b"])
        self.assertEqual(status["candidate_columns"], ["C.c", "R.a"])

    def test_an_unrestricted_pool_reports_no_exclusions(self):
        source = self.provider()
        self.discover(source=source)
        status = source.conjunctive_status
        self.assertEqual(status["columns_excluded_by_name"], [])
        self.assertEqual(status["candidate_columns"], ["C.c", "P.b", "R.a"])

    def test_the_status_records_the_row_count_and_the_capped_columns(self):
        source = self.provider(cap=2)
        self.discover(source)
        status = source.conjunctive_status
        self.assertEqual(status["materialized_rows"], 40)
        self.assertEqual(status["columns_skipped_by_domain_cap"], {"R.a": 3, "P.b": 3})


class RequiredExtensionColumnTest(unittest.TestCase):
    """The column carrying the tail has to be in the pool, or say why not."""

    def discover(self, *, cap=None, names=None, required=()):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows(), max_domain_size=cap, column_names=names
        )
        return source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=8,
            conjunctive={
                "enabled": True,
                "max_depth": 2,
                "beam_width": 8,
                "seed_width": 8,
                "minimum_joint_rows": 1.0,
                "require_distinct_tables": True,
                "required_extension_columns": list(required),
            },
        )

    def test_a_column_excluded_by_the_domain_cap_is_named_as_such(self):
        # This is the JOB-light case: max_candidate_domain_size=4096 silently
        # excluded keyword_id at ~134k, so the tail conjunction was
        # undiscoverable and nothing said so.
        with self.assertRaisesRegex(ValueError, "excluded .* by domain size"):
            self.discover(cap=2, required=["P.b"])

    def test_a_column_left_out_by_name_is_named_as_such(self):
        with self.assertRaisesRegex(ValueError, "does not list"):
            self.discover(names=["R.a", "C.c"], required=["P.b"])

    def test_a_column_absent_from_the_schema_is_named_as_such(self):
        with self.assertRaisesRegex(ValueError, "not DATA columns"):
            self.discover(required=["nope.col"])

    def test_a_satisfied_requirement_passes(self):
        self.discover(names=["R.a", "C.c"], required=["C.c"])

    def test_naming_columns_keeps_the_others_out_of_the_pool(self):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows(), column_names=["R.a", "C.c"]
        )
        meta = chain_data_metadata()
        self.assertEqual(
            sorted(meta.columns[index].name for index in source.extension_column_masses),
            ["C.c", "R.a"],
        )
        self.assertEqual(source.columns_excluded_by_name, ("P.b",))


class RowlessProviderFailClosedTest(unittest.TestCase):
    """require_conjunctions must also cover the oldest fallback of all."""

    def provider(self):
        meta = ModelMetadata(
            columns=(ColumnMetadata("t:a", ColumnKind.DATA, (0, 1, 2), table="t"),),
            full_join_cardinality=9.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )
        masses = {
            0: RootColumnMass(
                column_index=0,
                values=(0, 1, 2),
                counts=np.array([3.0, 3.0, 3.0]),
                total_count=9.0,
                source="test",
                semantic_type="categorical",
            )
        }
        # No encoded rows: this is the shape a live NeuroCard root-JCT provider
        # has, and the one every JOB-light live run actually builds.
        return ExactRootStratumProvider(meta, masses)

    def discover(self, source, **conjunctive):
        options = {"enabled": True, "max_depth": 2}
        options.update(conjunctive)
        return source.discover(
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e9,
            max_selected_strata=8,
            conjunctive=options,
        )

    def test_a_rowless_provider_cannot_serve_conjunctions(self):
        self.assertFalse(self.provider().supports_conjunctive_strata)

    def test_require_conjunctions_raises_instead_of_falling_back(self):
        # Setting conjunctive_requested=False for a rowless provider used to
        # skip the later require_conjunctions check entirely, so the run
        # returned depth-1 strata with no error at all.
        with self.assertRaisesRegex(ValueError, "no materialized rows"):
            self.discover(self.provider(), require_conjunctions=True)

    def test_the_error_points_at_the_fixture_path(self):
        with self.assertRaisesRegex(ValueError, "fixture or prepared-directory"):
            self.discover(self.provider(), require_conjunctions=True)

    def test_without_the_flag_it_still_falls_back_and_says_why(self):
        source = self.provider()
        selected = self.discover(source)
        self.assertEqual({item.depth for item in selected}, {1})
        self.assertFalse(source.conjunctive_status["applied"])
        self.assertIn("materialized rows", str(source.conjunctive_status["reason"]))


class ScaleTest(unittest.TestCase):
    """Discovery has to survive a fixture large enough to mean something.

    These sizes are small next to the 2M-row fixture the JOB-light config
    requires, but large enough that the per-row Python decode this replaced
    took orders of magnitude longer than the vectorized form.
    """

    ROWS = 200_000
    DOMAIN = 20_000
    BUDGET_SECONDS = 20.0

    def setUp(self):
        self.meta = ModelMetadata(
            columns=(
                ColumnMetadata(
                    "title:production_year", ColumnKind.DATA,
                    tuple(range(1900, 1900 + 200)), table="title",
                ),
                ColumnMetadata(
                    "movie_keyword:keyword_id", ColumnKind.DATA,
                    tuple(range(self.DOMAIN)), table="movie_keyword",
                ),
            ),
            full_join_cardinality=float(self.ROWS),
            join_root="title",
            join_tables=("title", "movie_keyword"),
            join_edges=(("title", "movie_keyword"),),
        )
        rng = np.random.default_rng(0)
        self.rows = np.stack(
            [rng.integers(0, 200, self.ROWS), rng.integers(0, self.DOMAIN, self.ROWS)],
            axis=1,
        ).astype(np.int64)
        self.provider = ExactRootStratumProvider.from_encoded_rows(self.meta, self.rows)

    def term(self, column_index, name, value):
        return StratumTerm(
            column_index=column_index,
            column_name=name,
            region_type="equality",
            value=value,
            realization_probability=PROBABILITIES["equality"],
            realization_family="equality",
        )

    def pairs(self, count):
        return [
            (
                self.term(0, "title:production_year", 1900 + index % 200),
                self.term(1, "movie_keyword:keyword_id", index % self.DOMAIN),
            )
            for index in range(count)
        ]

    def test_joint_count_stays_within_budget_over_many_pairs(self):
        started = time.perf_counter()
        for terms in self.pairs(200):
            self.provider.joint_count(terms)
        elapsed = time.perf_counter() - started
        self.assertLess(
            elapsed,
            self.BUDGET_SECONDS,
            f"200 joint_count calls over {self.ROWS} rows took {elapsed:.1f}s; the "
            "beam evaluates beam_width * extension_width of them",
        )

    def test_joint_count_agrees_with_a_direct_row_scan(self):
        # The vectorized path must give the same answer as decoding each row.
        terms = self.pairs(1)[0]
        direct = sum(
            1
            for row in self.rows
            if all(
                term.contains_value(self.meta.columns[term.column_index].domain[int(row[term.column_index])])
                for term in terms
            )
        )
        self.assertEqual(self.provider.joint_count(terms), float(direct))

    def test_realized_support_mass_stays_within_budget(self):
        started = time.perf_counter()
        for terms in self.pairs(100):
            self.provider.realized_support_mass(terms, PROBABILITIES)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, self.BUDGET_SECONDS, f"took {elapsed:.1f}s")

    def test_the_candidate_cap_bounds_a_high_cardinality_column(self):
        uncapped = self.provider._candidates_for_masses(
            self.provider.extension_column_masses,
            n_total=self.ROWS,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
        )
        capped = self.provider._candidates_for_masses(
            self.provider.extension_column_masses,
            n_total=self.ROWS,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
            max_candidates_per_column=256,
        )
        # One candidate per observed value: the cap is what keeps that bounded.
        self.assertGreater(len(uncapped), 10_000)
        self.assertLessEqual(len(capped), 256 + 200)

    def test_the_cap_keeps_the_rarest_regions(self):
        capped = self.provider._candidates_for_masses(
            {1: self.provider.extension_column_masses[1]},
            n_total=self.ROWS,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
            max_candidates_per_column=64,
        )
        counts = np.asarray(self.provider.extension_column_masses[1].counts)
        kept = sorted(float(item.foj_count) for item in capped)
        rarest = sorted(float(value) for value in np.sort(counts[counts > 0])[:64])
        self.assertEqual(kept, rarest)

    def test_an_ordered_column_is_not_trimmed(self):
        # Tail regions are cumulative, so dropping values would move thresholds.
        masses = {0: self.provider.extension_column_masses[0]}
        masses[0] = RootColumnMass(
            column_index=0,
            values=masses[0].values,
            counts=masses[0].counts,
            total_count=masses[0].total_count,
            source=masses[0].source,
            semantic_type="ordered",
        )
        kwargs = dict(
            n_total=self.ROWS,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
        )
        uncapped = self.provider._candidates_for_masses(masses, **kwargs)
        capped = self.provider._candidates_for_masses(
            masses, max_candidates_per_column=8, **kwargs
        )
        # The cap must be a no-op on an ordered column: trimming values would
        # move every tail threshold, not just shorten the candidate list.
        self.assertGreater(len(uncapped), 8)
        self.assertEqual(
            [item.stratum_id for item in capped],
            [item.stratum_id for item in uncapped],
        )
        self.assertEqual(
            [item.support_deficit for item in capped],
            [item.support_deficit for item in uncapped],
        )


class ConditionalExtensionRankingTest(unittest.TestCase):
    """A marginally common value can still be jointly rare, and must survive.

    The fixture is the case the marginal cap destroys: keyword 0 is the most
    common keyword overall, but almost never appears with the latest production
    year.  Ranking extensions by marginal count prunes it before the beam sees
    it, which is precisely the dependence a conjunctive stratum exists to find.
    """

    LATE_YEAR = 3

    def metadata(self):
        return ModelMetadata(
            columns=(
                ColumnMetadata(
                    "title:production_year", ColumnKind.DATA, (1990, 2000, 2010, 2020),
                    table="title",
                ),
                ColumnMetadata(
                    "movie_keyword:keyword_id", ColumnKind.DATA, (0, 1, 2, 3, 4),
                    table="movie_keyword",
                ),
                ColumnMetadata("I_movie_keyword", ColumnKind.INDICATOR, (0, 1),
                               table="movie_keyword"),
            ),
            full_join_cardinality=1000.0,
            join_root="title",
            join_tables=("title", "movie_keyword"),
            join_edges=(("title", "movie_keyword"),),
        )

    def rows(self):
        rows = []
        # keyword 0 dominates the early years: 600 rows, so it is by far the
        # most common keyword marginally.
        for index in range(600):
            rows.append([index % 3, 0, 1])
        # keywords 1..4 are each much rarer overall, spread over every year.
        for index in range(380):
            rows.append([index % 4, 1 + index % 4, 1])
        # keyword 0 with the latest year is the joint tail: 20 rows only.
        for _ in range(20):
            rows.append([self.LATE_YEAR, 0, 1])
        return np.asarray(rows, dtype=np.int64)

    def provider(self):
        return ExactRootStratumProvider.from_encoded_rows(self.metadata(), self.rows())

    def late_year_seed(self):
        meta = self.metadata()
        return with_realization_probabilities(
            RootDataStratum(
                stratum_id="title:production_year:eq:2020",
                column_index=0,
                column_name="title:production_year",
                region_type="equality",
                value=meta.columns[0].domain[self.LATE_YEAR],
                probability=0.02,
                expected_target_rows=20.0,
                expected_equality_count=20.0 * PROBABILITIES["equality"],
            ),
            PROBABILITIES,
        )

    def test_keyword_zero_is_the_most_common_marginally(self):
        source = self.provider()
        counts = np.asarray(source.extension_column_masses[1].counts)
        self.assertEqual(int(np.argmax(counts)), 0)

    def test_the_marginal_cap_prunes_it(self):
        # One slot for the keyword column: marginal ranking keeps the rarest
        # keyword overall, which is never keyword 0.
        source = self.provider()
        capped = source._candidates_for_masses(
            {1: source.extension_column_masses[1]},
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=1e9,
            max_candidates_per_column=1,
        )
        self.assertEqual(len(capped), 1)
        self.assertNotEqual(capped[0].value, 0)

    def test_conditional_counting_surfaces_the_joint_tail(self):
        source = self.provider()
        counts = source.conditional_extension_counts(
            (self.late_year_seed(),),
            source.extension_column_masses,
            minimum_joint_rows=1.0,
            per_seed_width=1,
            require_distinct_tables=True,
        )
        self.assertIn((1, 0), counts, counts)
        self.assertEqual(counts[(1, 0)], 20.0)

    def test_the_conditional_count_is_the_exact_joint_count(self):
        source = self.provider()
        seed = self.late_year_seed()
        counts = source.conditional_extension_counts(
            (seed,), source.extension_column_masses,
            minimum_joint_rows=1.0, per_seed_width=5, require_distinct_tables=True,
        )
        meta = self.metadata()
        for (column_index, encoded_id), joint in counts.items():
            terms = seed.terms + (
                StratumTerm(
                    column_index=column_index,
                    column_name=meta.columns[column_index].name,
                    region_type="equality",
                    value=meta.columns[column_index].domain[encoded_id],
                ),
            )
            self.assertEqual(source.joint_count(terms), joint)

    def test_a_column_the_seed_already_pins_is_not_counted(self):
        source = self.provider()
        counts = source.conditional_extension_counts(
            (self.late_year_seed(),), source.extension_column_masses,
            minimum_joint_rows=1.0, per_seed_width=5, require_distinct_tables=False,
        )
        self.assertFalse([key for key in counts if key[0] == 0], counts)

    def test_the_joint_row_floor_excludes_absent_combinations(self):
        source = self.provider()
        counts = source.conditional_extension_counts(
            (self.late_year_seed(),), source.extension_column_masses,
            minimum_joint_rows=1000.0, per_seed_width=5, require_distinct_tables=True,
        )
        self.assertEqual(counts, {})

    def test_discovery_reports_the_conditional_pool(self):
        source = self.provider()
        source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=16,
            conjunctive={
                "enabled": True,
                "max_depth": 2,
                "beam_width": 8,
                "seed_width": 8,
                "extension_width": 4,
                "conditional_extension_width": 2,
                "minimum_joint_rows": 1.0,
                "max_joint_row_fraction": 1.0,
                "require_distinct_tables": True,
            },
        )
        self.assertGreater(
            source.conjunctive_status["conditional_extension_candidates"], 0
        )

    def test_the_pool_fills_with_the_rarest_conditional_candidates_first(self):
        # More conditional candidates than slots, so the ordering decides which
        # ones survive.  Ranking by descending joint count would keep the
        # common pairings and drop the tail.
        source = self.provider()
        seed = self.late_year_seed()
        candidates = source._conditional_extension_candidates(
            (seed,),
            source.extension_column_masses,
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            threshold=1e12,
            minimum_joint_rows=1.0,
            per_seed_width=5,
            require_distinct_tables=True,
        )
        self.assertGreater(len(candidates), 1)
        counts = sorted(float(item.foj_count) for item in candidates)
        rarest_value = min(candidates, key=lambda item: item.foj_count).value
        self.assertEqual(rarest_value, 0)
        self.assertEqual(counts[0], 20.0)

    def test_a_value_rare_with_one_seed_keeps_that_seed_s_count(self):
        # Across seeds the smallest joint count is the one that matters: a
        # value common with one seed and rare with another is still a tail.
        source = self.provider()
        meta = self.metadata()
        early_seed = with_realization_probabilities(
            RootDataStratum(
                stratum_id="title:production_year:eq:1990",
                column_index=0,
                column_name="title:production_year",
                region_type="equality",
                value=meta.columns[0].domain[0],
                probability=0.2,
                expected_target_rows=200.0,
                expected_equality_count=200.0 * PROBABILITIES["equality"],
            ),
            PROBABILITIES,
        )
        late_only = source.conditional_extension_counts(
            (self.late_year_seed(),), source.extension_column_masses,
            minimum_joint_rows=1.0, per_seed_width=5, require_distinct_tables=True,
        )
        both = source.conditional_extension_counts(
            (early_seed, self.late_year_seed()), source.extension_column_masses,
            minimum_joint_rows=1.0, per_seed_width=5, require_distinct_tables=True,
        )
        early_only = source.conditional_extension_counts(
            (early_seed,), source.extension_column_masses,
            minimum_joint_rows=1.0, per_seed_width=5, require_distinct_tables=True,
        )
        self.assertGreater(early_only[(1, 0)], late_only[(1, 0)])
        self.assertEqual(both[(1, 0)], late_only[(1, 0)])

    def test_one_extension_slot_goes_to_the_rarest_conditional_pairing(self):
        # Five conditional candidates compete for a single slot, so the pool's
        # fill order decides the outcome: rarest-first keeps the joint tail,
        # commonest-first keeps a pairing the model already sees.
        source = self.provider()
        meta = self.metadata()
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=32,
            conjunctive={
                "enabled": True,
                "max_depth": 2,
                "beam_width": 16,
                "seed_width": 16,
                "extension_width": 1,
                "conditional_extension_width": 5,
                "minimum_joint_rows": 1.0,
                "max_joint_row_fraction": 1.0,
                "require_distinct_tables": True,
            },
            discovery_limits={"max_candidates_per_column": 1},
        )
        keyword_terms = {
            term.value
            for item in selected
            if item.is_conjunctive
            for term in item.terms
            if meta.columns[term.column_index].name == "movie_keyword:keyword_id"
        }
        self.assertEqual(
            keyword_terms,
            {0},
            [item.stratum_id for item in selected if item.is_conjunctive],
        )

    def test_the_jointly_rare_value_reaches_a_selected_conjunction(self):
        # End to end: with the marginal cap squeezed down to one slot per
        # column, the conditional route is the only way keyword 0 can appear.
        source = self.provider()
        meta = self.metadata()
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=32,
            conjunctive={
                "enabled": True,
                "max_depth": 2,
                "beam_width": 16,
                "seed_width": 16,
                "extension_width": 4,
                "conditional_extension_width": 2,
                "minimum_joint_rows": 1.0,
                "max_joint_row_fraction": 1.0,
                "require_distinct_tables": True,
            },
            discovery_limits={"max_candidates_per_column": 1},
        )
        keyword_zero = [
            item
            for item in selected
            if item.is_conjunctive
            and any(
                meta.columns[term.column_index].name == "movie_keyword:keyword_id"
                and term.value == 0
                for term in item.terms
            )
        ]
        self.assertTrue(
            keyword_zero,
            [item.stratum_id for item in selected if item.is_conjunctive],
        )


class FactorCacheTest(unittest.TestCase):
    """A cached factor vector must not outlive the probabilities it used."""

    def provider(self):
        meta = ModelMetadata(
            columns=(ColumnMetadata("t:a", ColumnKind.DATA, (0, 1, 2), table="t"),),
            full_join_cardinality=9.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )
        rows = np.array([[index % 3] for index in range(9)], dtype=np.int64)
        return ExactRootStratumProvider.from_encoded_rows(meta, rows), meta

    def probabilities(self, equality):
        return {
            "wildcard": 0.2,
            "equality": equality,
            "lower": 0.2,
            "upper": 0.2,
            "range": 0.2,
        }

    def term(self):
        return StratumTerm(
            column_index=0,
            column_name="t:a",
            region_type="equality",
            value=1,
            realization_family="equality",
        )

    def test_changing_the_probabilities_changes_the_answer(self):
        source, _ = self.provider()
        first = source.realized_support_mass((self.term(),), self.probabilities(0.2))
        second = source.realized_support_mass((self.term(),), self.probabilities(0.6))
        self.assertAlmostEqual(second / first, 3.0, places=9)

    def test_a_reused_provider_agrees_with_a_fresh_one(self):
        # The cache was keyed on the term alone, so a provider reused across two
        # discovery calls answered the second with the first call's factors.
        reused, _ = self.provider()
        reused.realized_support_mass((self.term(),), self.probabilities(0.2))
        warm = reused.realized_support_mass((self.term(),), self.probabilities(0.6))
        fresh, _ = self.provider()
        cold = fresh.realized_support_mass((self.term(),), self.probabilities(0.6))
        self.assertAlmostEqual(warm, cold, places=12)

    def test_the_cache_still_serves_a_repeated_call(self):
        source, _ = self.provider()
        probabilities = self.probabilities(0.2)
        first = source.realized_support_mass((self.term(),), probabilities)
        self.assertEqual(len(source._term_factor_cache), 1)
        again = source.realized_support_mass((self.term(),), probabilities)
        self.assertEqual(first, again)
        self.assertEqual(len(source._term_factor_cache), 1)


class RequiredColumnCoverageTest(unittest.TestCase):
    """Being in the pool is not the same as being in the output."""

    def discover(self, **conjunctive):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows()
        )
        options = {
            "enabled": True,
            "max_depth": 2,
            "beam_width": 8,
            "seed_width": 8,
            "minimum_joint_rows": 1.0,
            "max_joint_row_fraction": 1.0,
            "require_distinct_tables": True,
        }
        options.update(conjunctive)
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=options.pop("max_selected", 8),
            conjunctive=options,
        )
        return source, selected

    def test_coverage_is_reported_for_the_selected_conjunctions(self):
        source, _ = self.discover(required_extension_columns=["C.c"])
        status = source.conjunctive_status
        self.assertIn("required_columns_covered", status)
        self.assertIn("required_columns_uncovered", status)

    def test_a_required_column_absent_from_the_output_is_a_failure(self):
        # P.b is in the pool, so the pool-level guard passes, but one selected
        # stratum cannot carry both it and anything else.
        with self.assertRaisesRegex(ValueError, "appear in no selected"):
            self.discover(
                required_extension_columns=["P.b", "C.c"],
                require_conjunctions=True,
                max_selected=1,
                extension_width=1,
            )

    def test_the_failure_names_the_uncovered_column(self):
        try:
            self.discover(
                required_extension_columns=["P.b", "C.c"],
                require_conjunctions=True,
                max_selected=1,
                extension_width=1,
            )
        except ValueError as error:
            self.assertRegex(str(error), r"P\.b|C\.c")
        else:
            self.fail("expected a coverage failure")

    def test_without_the_flag_uncovered_columns_are_only_reported(self):
        source, _ = self.discover(
            required_extension_columns=["P.b", "C.c"], max_selected=1, extension_width=1
        )
        self.assertTrue(source.conjunctive_status["required_columns_uncovered"])

    def test_a_depth_one_stratum_does_not_count_as_coverage(self):
        # Coverage means a conjunction uses the column.  A depth-1 stratum on
        # the same column says nothing about the joint region.
        source, selected = self.discover(
            required_extension_columns=["R.a"],
            minimum_joint_rows=1e9,
            max_joint_row_fraction=0.0,
        )
        self.assertEqual(source.conjunctive_status["conjunctive_selected"], 0)
        self.assertTrue(
            any(
                chain_data_metadata().columns[item.column_index].name == "R.a"
                for item in selected
                if item.depth == 1
            )
        )
        self.assertEqual(source.conjunctive_status["required_columns_uncovered"], ["R.a"])

    def test_a_covered_column_passes(self):
        source, _ = self.discover(
            required_extension_columns=["C.c"], require_conjunctions=True
        )
        self.assertEqual(source.conjunctive_status["required_columns_uncovered"], [])
        self.assertIn("C.c", source.conjunctive_status["required_columns_covered"])


class StructuralPaddingTest(unittest.TestCase):
    """Join padding must not consume a slot of the candidate budget.

    A full-outer-join child column is full of OUTER_MISSING, and on many seeds
    that padding is the rarest encoded id.  Candidate builders discard it, so
    ranking and truncating before the discard spends budget on nothing.
    """

    def metadata(self):
        return ModelMetadata(
            columns=(
                ColumnMetadata("title:production_year", ColumnKind.DATA, (1990, 2000),
                               table="title"),
                ColumnMetadata(
                    "movie_keyword:keyword_id", ColumnKind.DATA,
                    (7, 8, OUTER_MISSING, SQL_NULL), table="movie_keyword",
                ),
                ColumnMetadata("I_movie_keyword", ColumnKind.INDICATOR, (0, 1),
                               table="movie_keyword"),
            ),
            full_join_cardinality=100.0,
            join_root="title",
            join_tables=("title", "movie_keyword"),
            join_edges=(("title", "movie_keyword"),),
        )

    def rows(self):
        # With the late year: keyword 7 x5, keyword 8 x9, OUTER_MISSING x1.
        # The padding is the rarest id, so it wins any count-based ranking.
        return np.asarray(
            [[1, 0, 1]] * 5 + [[1, 1, 1]] * 9 + [[1, 2, 0]] * 1 + [[0, 0, 1]] * 40,
            dtype=np.int64,
        )

    def provider(self):
        return ExactRootStratumProvider.from_encoded_rows(self.metadata(), self.rows())

    def seed(self):
        return with_realization_probabilities(
            RootDataStratum(
                stratum_id="title:production_year:eq:2000",
                column_index=0,
                column_name="title:production_year",
                region_type="equality",
                value=2000,
                probability=0.15,
                expected_target_rows=15.0,
                expected_equality_count=15.0 * PROBABILITIES["equality"],
            ),
            PROBABILITIES,
        )

    def test_the_padding_really_is_the_rarest_encoded_id(self):
        source = self.provider()
        seed_rows = self.rows()[self.rows()[:, 0] == 1]
        counts = np.bincount(seed_rows[:, 1], minlength=4)
        self.assertEqual(int(np.argmin(np.where(counts > 0, counts, 1 << 30))), 2)
        self.assertTrue(_is_structural_value(self.metadata().columns[1].domain[2]))

    def test_one_slot_goes_to_a_real_value_not_the_padding(self):
        source = self.provider()
        counts = source.conditional_extension_counts(
            (self.seed(),),
            source.extension_column_masses,
            minimum_joint_rows=1.0,
            per_seed_width=1,
            require_distinct_tables=True,
        )
        self.assertEqual(counts, {(1, 0): 5.0})

    def test_the_candidate_survives_instead_of_vanishing(self):
        source = self.provider()
        candidates = source._conditional_extension_candidates(
            (self.seed(),),
            source.extension_column_masses,
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=1e9,
            minimum_joint_rows=1.0,
            per_seed_width=1,
            require_distinct_tables=True,
        )
        self.assertEqual([item.value for item in candidates], [7])

    def test_no_structural_value_ever_reaches_the_counts(self):
        source = self.provider()
        meta = self.metadata()
        counts = source.conditional_extension_counts(
            (self.seed(),),
            source.extension_column_masses,
            minimum_joint_rows=1.0,
            per_seed_width=4,
            require_distinct_tables=True,
        )
        for column_index, encoded_id in counts:
            self.assertFalse(
                _is_structural_value(meta.columns[column_index].domain[encoded_id])
            )

    def test_the_marginal_cap_also_skips_padding(self):
        # Same ordering bug: the builder drops padding afterwards, so a capped
        # column would come back one candidate short.
        source = self.provider()
        capped = source._candidates_for_masses(
            {1: source.extension_column_masses[1]},
            n_total=1000,
            predicate_probabilities=PROBABILITIES,
            threshold=1e9,
            max_candidates_per_column=1,
        )
        self.assertEqual(len(capped), 1)
        self.assertFalse(_is_structural_value(capped[0].value))


class UncoveredColumnDiagnosisTest(unittest.TestCase):
    """A failed coverage check must say which stage lost the column.

    A column can be lost by never reaching the pool, by the joint-row floor
    rejecting every pairing, or by being out-ranked in the final sort. Those
    need opposite fixes, so a message that only says "not covered" costs a
    cluster round to narrow down -- and a message that names the wrong stage
    costs one too.
    """

    def discover(self, **conjunctive):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows()
        )
        options = {
            "enabled": True,
            "max_depth": 2,
            "beam_width": 8,
            "seed_width": 8,
            "minimum_joint_rows": 1.0,
            "max_joint_row_fraction": 0.0,
            "require_distinct_tables": True,
        }
        options.update(conjunctive)
        max_selected = options.pop("max_selected", 8)
        source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=max_selected,
            conjunctive=options,
        )
        return source

    def diagnosis(self, source, column):
        return source.conjunctive_status["required_column_diagnosis"][column]

    def test_a_floor_that_rejects_every_pairing_says_so(self):
        # The JOB-light failure: no pairing of the required column could clear
        # minimum_joint_rows, so widening the search cannot help.
        source = self.discover(
            required_extension_columns=["C.c"],
            minimum_joint_rows=1000.0,
        )
        entry = self.diagnosis(source, "C.c")
        self.assertEqual(entry["conjunctions_discovered"], 0)
        self.assertIsNotNone(entry["best_pruned_joint_count"])
        self.assertIn("minimum_joint_rows", entry["explanation"])
        self.assertIn("extension_width", entry["explanation"])
        self.assertIn("can change that", entry["explanation"])

    def test_the_beam_stage_floor_is_recorded_too(self):
        # With conditional counting off, the floor prunes inside the beam
        # instead, and that path has its own recording.
        source = self.discover(
            required_extension_columns=["C.c"],
            minimum_joint_rows=1000.0,
            conditional_extension_width=0,
        )
        entry = self.diagnosis(source, "C.c")
        self.assertIsNotNone(entry["best_pruned_joint_count"])
        self.assertGreater(entry["best_pruned_joint_count"], 0.0)
        self.assertIn("largest joint count any pairing reached", entry["explanation"])

    def test_the_recorded_count_is_the_largest_rejected_not_the_smallest(self):
        # Reporting the smallest would tell the user to lower the floor further
        # than necessary and overstate the fixture size needed.  Both recording
        # paths -- conditional counting and the beam -- must agree, and the
        # value must be the largest rejected count, which on this fixture is
        # strictly greater than the smallest.
        recorded = {}
        for width in (0, 8):
            source = self.discover(
                required_extension_columns=["C.c"],
                minimum_joint_rows=1000.0,
                conditional_extension_width=width,
            )
            recorded[width] = self.diagnosis(source, "C.c")["best_pruned_joint_count"]
        self.assertEqual(recorded[0], recorded[8])
        self.assertEqual(recorded[0], 7.0)

    def test_the_conditional_stage_records_on_its_own(self):
        # Read the conditional recording directly, with no beam run at all, so
        # the beam cannot supply the value this stage is supposed to.
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows()
        )
        source.pruned_joint_counts = {}
        source.conditional_extension_counts(
            (stratum(term(0, "R.a", 1), probability=0.25),),
            source.extension_column_masses,
            minimum_joint_rows=1000.0,
            per_seed_width=8,
            require_distinct_tables=True,
        )
        self.assertTrue(source.pruned_joint_counts)
        self.assertTrue(
            all(value > 0.0 for value in source.pruned_joint_counts.values()),
            source.pruned_joint_counts,
        )

    def test_running_discovery_twice_does_not_accumulate_counters(self):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows()
        )
        options = {
            "enabled": True,
            "max_depth": 2,
            "beam_width": 8,
            "seed_width": 8,
            "minimum_joint_rows": 1.0,
            "max_joint_row_fraction": 0.0,
            "require_distinct_tables": True,
        }
        source.discover(
            n_total=1_000_000, predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12, max_selected_strata=8,
            conjunctive=dict(options, minimum_joint_rows=1000.0),
        )
        first = dict(source.pruned_joint_counts)
        self.assertTrue(first)
        source.discover(
            n_total=1_000_000, predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12, max_selected_strata=8,
            conjunctive=options,
        )
        # The second call clears everything the first recorded.
        self.assertNotEqual(source.pruned_joint_counts, first)

    def test_the_floor_diagnosis_reports_the_rows_that_would_be_needed(self):
        source = self.discover(
            required_extension_columns=["C.c"], minimum_joint_rows=1000.0
        )
        entry = self.diagnosis(source, "C.c")
        rows = source.conjunctive_status["materialized_rows"]
        best = entry["best_pruned_joint_count"]
        self.assertEqual(
            entry["fixture_rows_for_this_floor"], int(rows * 1000.0 / best)
        )
        self.assertGreater(entry["fixture_rows_for_this_floor"], rows)

    def test_the_floor_diagnosis_sizes_a_fixture_for_several_conjunctions(self):
        # The maximum alone says whether ONE pairing could be admitted by a
        # bigger fixture.  A fixture-growth decision turns on how much bigger it
        # must be for a usable number of them, so the top of the rejected
        # distribution is kept, not only its peak.
        source = self.discover(
            required_extension_columns=["C.c"], minimum_joint_rows=1000.0
        )
        entry = self.diagnosis(source, "C.c")
        top = entry["pruned_joint_counts_top"]
        self.assertTrue(top)
        self.assertEqual(top, sorted(top, reverse=True))
        needed = entry["fixture_rows_for_n_conjunctions"]
        self.assertEqual(needed["1"], entry["fixture_rows_for_this_floor"])
        ranks = [needed[key] for key in ("1", "5", "10") if needed[key] is not None]
        # Rarer pairings need a bigger fixture, so the series is non-decreasing.
        self.assertEqual(ranks, sorted(ranks))

    def test_a_rank_beyond_the_rejected_sample_reports_nothing(self):
        source = self.discover(
            required_extension_columns=["C.c"], minimum_joint_rows=1000.0
        )
        entry = self.diagnosis(source, "C.c")
        sample = source.pruned_joint_counts_sample
        column_index = next(
            index
            for index, column in enumerate(chain_data_metadata().columns)
            if column.name == "C.c"
        )
        if len(sample[column_index]) < 50:
            self.assertIsNone(entry["fixture_rows_for_n_conjunctions"]["50"])

    def test_the_recorded_sample_is_bounded(self):
        source = self.discover(
            required_extension_columns=["C.c"], minimum_joint_rows=1000.0
        )
        for counts in source.pruned_joint_counts_sample.values():
            self.assertLessEqual(len(counts), ExactRootStratumProvider.PRUNED_SAMPLE_SIZE)

    def test_the_conditional_stage_floor_is_attributed_to_the_floor(self):
        # The floor filters in conditional counting too, before the beam sees a
        # pairing.  Attributing that to extension_width sends the fix the wrong
        # way, which is what this asserts against.
        source = self.discover(
            required_extension_columns=["C.c"],
            minimum_joint_rows=1000.0,
            conditional_extension_width=8,
        )
        entry = self.diagnosis(source, "C.c")
        self.assertIsNotNone(entry["best_pruned_joint_count"])
        self.assertNotIn("widen extension_width", entry["explanation"])

    def test_a_column_outside_the_pool_says_so(self):
        source = ExactRootStratumProvider.from_encoded_rows(
            chain_data_metadata(), chain_rows(), column_names=["R.a", "P.b"]
        )
        source.conjunctive_status = {}
        with self.assertRaises(ValueError):
            source.discover(
                n_total=1_000_000,
                predicate_probabilities=PROBABILITIES,
                minimum_expected_context_support=1e12,
                max_selected_strata=8,
                conjunctive={
                    "enabled": True,
                    "max_depth": 2,
                    "minimum_joint_rows": 1.0,
                    "max_joint_row_fraction": 0.0,
                    "require_distinct_tables": True,
                    "required_extension_columns": ["C.c"],
                },
            )

    def test_an_out_ranked_column_is_attributed_to_selection(self):
        source = self.discover(
            required_extension_columns=["P.b", "C.c"],
            max_selected=1,
            extension_width=1,
            conditional_extension_width=0,
        )
        uncovered = source.conjunctive_status["required_columns_uncovered"]
        self.assertTrue(uncovered)
        for name in uncovered:
            entry = self.diagnosis(source, name)
            if entry["conjunctions_discovered"]:
                self.assertIn("out-ranked", entry["explanation"])
                self.assertIn("max_selected_strata", entry["explanation"])

    def test_the_diagnosis_is_recorded_even_without_require_conjunctions(self):
        source = self.discover(
            required_extension_columns=["C.c"], minimum_joint_rows=1000.0
        )
        self.assertIn("required_column_diagnosis", source.conjunctive_status)

    def test_a_covered_column_needs_no_diagnosis(self):
        source = self.discover(required_extension_columns=["C.c"])
        self.assertEqual(source.conjunctive_status["required_columns_uncovered"], [])
        self.assertEqual(source.conjunctive_status["required_column_diagnosis"], {})


class DeficitObjectiveTest(unittest.TestCase):
    """The deficit must keep measuring rarity once every candidate is rare.

    ``clipped_sum`` contributes at most ``threshold`` per head, so once support
    is far below the threshold the sum is ``heads * threshold`` whatever the
    support is.  Ranking then collapses onto the served-head count, which is the
    column's position in the schema, and a conjunction on a later column cannot
    win however much rarer it is.  That is what put 42 info_type_id and 11
    role_id conjunctions in a JOB-light selection and no keyword_id ones.
    """

    THRESHOLD = 100.0

    def test_the_clipped_sum_saturates(self):
        far = deficit_from_supports([0.1] * 6, threshold=self.THRESHOLD)
        further = deficit_from_supports([0.0001] * 6, threshold=self.THRESHOLD)
        self.assertAlmostEqual(far, further, delta=1.0)
        self.assertLess(further, 6 * self.THRESHOLD)
        self.assertGreater(further, 6 * self.THRESHOLD - 1.0)

    def test_the_log_ratio_keeps_growing(self):
        far = deficit_from_supports(
            [0.1] * 6, threshold=self.THRESHOLD, objective="log_ratio"
        )
        further = deficit_from_supports(
            [0.0001] * 6, threshold=self.THRESHOLD, objective="log_ratio"
        )
        self.assertGreater(further, far * 1.5)

    def test_head_count_outranks_rarity_under_the_clipped_sum(self):
        # Six heads at support 0.03 against five heads a thousand times rarer.
        many = deficit_from_supports([0.03] * 6, threshold=self.THRESHOLD)
        rare = deficit_from_supports([0.00003] * 5, threshold=self.THRESHOLD)
        self.assertGreater(many, rare)

    def test_rarity_outranks_head_count_under_the_log_ratio(self):
        many = deficit_from_supports(
            [0.03] * 6, threshold=self.THRESHOLD, objective="log_ratio"
        )
        rare = deficit_from_supports(
            [0.00003] * 5, threshold=self.THRESHOLD, objective="log_ratio"
        )
        self.assertGreater(rare, many)

    def test_a_head_at_or_above_the_threshold_contributes_nothing(self):
        for objective in ("clipped_sum", "log_ratio"):
            with self.subTest(objective=objective):
                self.assertEqual(
                    deficit_from_supports(
                        [self.THRESHOLD, self.THRESHOLD * 2],
                        threshold=self.THRESHOLD,
                        objective=objective,
                    ),
                    0.0,
                )

    def test_zero_support_is_bounded_by_epsilon_not_infinite(self):
        value = deficit_from_supports(
            [0.0], threshold=self.THRESHOLD, objective="log_ratio", epsilon=1e-9
        )
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(value, float(np.log(self.THRESHOLD / 1e-9)), places=9)

    def test_an_unknown_objective_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported deficit objective"):
            deficit_from_supports([1.0], threshold=self.THRESHOLD, objective="mean")

    def test_the_default_is_unchanged(self):
        self.assertEqual(DEFAULT_DEFICIT_OBJECTIVE, "clipped_sum")
        self.assertEqual(
            deficit_from_supports([1.0], threshold=self.THRESHOLD),
            deficit_from_supports(
                [1.0], threshold=self.THRESHOLD, objective="clipped_sum"
            ),
        )

    def test_discovery_records_which_objective_ran(self):
        for objective in ("clipped_sum", "log_ratio"):
            with self.subTest(objective=objective):
                source = ExactRootStratumProvider.from_encoded_rows(
                    chain_data_metadata(), chain_rows()
                )
                source.discover(
                    n_total=1_000_000,
                    predicate_probabilities=PROBABILITIES,
                    minimum_expected_context_support=1e12,
                    max_selected_strata=8,
                    conjunctive={
                        "enabled": True,
                        "max_depth": 2,
                        "beam_width": 8,
                        "seed_width": 8,
                        "minimum_joint_rows": 1.0,
                        "max_joint_row_fraction": 1.0,
                        "require_distinct_tables": True,
                    },
                    discovery_limits={"deficit_objective": objective},
                )
                self.assertEqual(
                    source.conjunctive_status["deficit_objective"], objective
                )


class PrunedDistributionTest(unittest.TestCase):
    """Sizing a fixture needs the spread of rejected counts, not just the peak.

    The fixture here gives the child column 120 distinct values with joint
    counts that differ, so rank 1 and rank 10 genuinely disagree and the
    recorded sample has to be ordered, bounded and free of zeros.
    """

    VALUES = 120
    FLOOR = 10_000.0

    def metadata(self):
        return ModelMetadata(
            columns=(
                ColumnMetadata("t:seed", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("c:value", ColumnKind.DATA, tuple(range(self.VALUES)),
                               table="c"),
                ColumnMetadata("I_c", ColumnKind.INDICATOR, (0, 1), table="c"),
            ),
            full_join_cardinality=10_000.0,
            join_root="t",
            join_tables=("t", "c"),
            join_edges=(("t", "c"),),
        )

    def rows(self):
        # Value v appears v+1 times, so every rejected count is distinct.
        rows = []
        for value in range(self.VALUES):
            rows.extend([[0, value, 1]] * (value + 1))
        return np.asarray(rows, dtype=np.int64)

    def source(self):
        provider = ExactRootStratumProvider.from_encoded_rows(
            self.metadata(), self.rows()
        )
        provider.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e12,
            max_selected_strata=8,
            conjunctive={
                "enabled": True,
                "max_depth": 2,
                "beam_width": 8,
                "seed_width": 8,
                "conditional_extension_width": 8,
                "minimum_joint_rows": self.FLOOR,
                "max_joint_row_fraction": 0.0,
                "require_distinct_tables": True,
                "required_extension_columns": ["c:value"],
            },
        )
        return provider

    def entry(self, provider):
        return provider.conjunctive_status["required_column_diagnosis"]["c:value"]

    def test_the_sample_is_capped_at_the_declared_size(self):
        provider = self.source()
        column_index = 1
        recorded = provider.pruned_joint_counts_sample[column_index]
        # More distinct rejected counts exist than the cap allows.
        self.assertGreater(self.VALUES, ExactRootStratumProvider.PRUNED_SAMPLE_SIZE)
        self.assertEqual(len(recorded), ExactRootStratumProvider.PRUNED_SAMPLE_SIZE)

    def test_the_ranks_disagree_and_rise_with_rarity(self):
        entry = self.entry(self.source())
        needed = entry["fixture_rows_for_n_conjunctions"]
        self.assertIsNotNone(needed["1"])
        self.assertIsNotNone(needed["10"])
        # A rarer pairing needs a strictly larger fixture, so the ranks must not
        # collapse onto the single best count.
        self.assertGreater(needed["10"], needed["1"])
        self.assertGreater(needed["50"], needed["10"])

    def test_zero_counts_never_enter_the_sample(self):
        provider = self.source()
        for counts in provider.pruned_joint_counts_sample.values():
            self.assertTrue(all(value > 0 for value in counts), counts)

    def test_a_zero_count_is_not_recorded_as_a_rejected_pairing(self):
        # joint_count returns 0 for a combination that does not occur, and the
        # beam records what it rejected, so zeros can arrive here.  A zero is an
        # absent region, not a rejected one: recording it would divide by zero
        # when sizing a fixture, or suggest one of infinite size.
        provider = ExactRootStratumProvider.from_encoded_rows(
            self.metadata(), self.rows()
        )
        provider.pruned_joint_counts_sample = {}
        provider.pruned_joint_counts = {}
        provider._record_pruned(1, np.array([0.0, 0.0, 3.0, 0.0]))
        self.assertEqual(provider.pruned_joint_counts_sample[1], [3.0])
        self.assertEqual(provider.pruned_joint_counts[1], 3.0)

    def test_only_zero_counts_record_nothing_at_all(self):
        provider = ExactRootStratumProvider.from_encoded_rows(
            self.metadata(), self.rows()
        )
        provider.pruned_joint_counts_sample = {}
        provider.pruned_joint_counts = {}
        provider._record_pruned(1, np.array([0.0, 0.0]))
        self.assertEqual(provider.pruned_joint_counts_sample[1], [])
        self.assertNotIn(1, provider.pruned_joint_counts)

    def test_the_peak_matches_the_largest_observed_pairing(self):
        provider = self.source()
        entry = self.entry(provider)
        # Value 119 appears 120 times and the seed covers every row.
        self.assertEqual(entry["best_pruned_joint_count"], float(self.VALUES))


class AuditScriptInvocationTest(unittest.TestCase):
    """The audit script must run both ways it is plausibly invoked.

    model/scripts has no __init__.py and nothing puts the repo root on
    sys.path, so running the file by path raises ModuleNotFoundError on
    "import model" before it reads a single argument.  The repo convention is
    "python3 -m model.scripts.<name>" from the root, but the path form is what
    anyone copies off a README line or out of shell history, and on a cluster
    that import error costs a round trip to diagnose.
    """

    SCRIPT = "model/scripts/inspect_rare_support.py"

    def _run(self, argv):
        import subprocess

        return subprocess.run(
            [sys.executable, *argv, "--config", "model/configs/resmade_smoke.yaml"],
            capture_output=True,
            text=True,
            cwd=pathlib.Path.cwd(),
        )

    def test_running_the_file_by_path_gets_past_the_imports(self):
        result = self._run([self.SCRIPT])
        self.assertNotIn("No module named 'model'", result.stderr)

    def test_running_it_as_a_module_gets_past_the_imports(self):
        result = self._run(["-m", "model.scripts.inspect_rare_support"])
        self.assertNotIn("No module named 'model'", result.stderr)

    RARE_BLOCKS = """
rare_support:
  enabled: true
  discovery:
    enabled: true
    root_data_only: true
    minimum_expected_context_support: 20
    max_selected_strata: 4
    support_planning_steps: 20
    minimum_stratum_rows: 1000000000
  allocation:
    strategy: support_deficit

rare_auxiliary:
  enabled: true
  batch_size: 4
  beta: 0.25
"""

    def test_per_row_exposure_measures_size_against_allocation(self):
        # The numbers are the JOB-light ranges geometry: 512 auxiliary rows and
        # 2048 main rows over 20,000 steps against a 2M-row fixture, so the
        # main stream shows a typical row 20.48 times.  A stratum holding 1032
        # rows at an even 1/64 allocation is drawn 160,000 times, which is
        # 155.0 draws per row, or 7.6x the main stream.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "inspect_rare_support_under_test", self.SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = {
            "training": {"batch_size": 2048, "steps_per_epoch": 20000, "epochs": 1},
            "rare_auxiliary": {"batch_size": 512},
        }
        status = {"materialized_rows": 2_000_000}
        strata = [
            {"stratum_id": "wide", "foj_count": 5114.0, "alpha": 1 / 64},
            {"stratum_id": "narrow", "foj_count": 1032.0, "alpha": 1 / 64},
        ]
        exposure = module.per_row_exposure(config, status, strata)

        self.assertAlmostEqual(exposure["main_stream_draws_per_row"], 20.48)
        self.assertEqual(exposure["worst_stratum"], "narrow")
        self.assertAlmostEqual(exposure["worst_multiple_of_main_stream"], 7.57, places=2)

    def test_per_row_exposure_is_absent_rather_than_wrong_without_rows(self):
        # The live path reports no materialized_rows, and the baseline is
        # meaningless without them.  Reporting a number anyway would be worse
        # than reporting none.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "inspect_rare_support_under_test", self.SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        config = {
            "training": {"batch_size": 2048, "steps_per_epoch": 20000, "epochs": 1},
            "rare_auxiliary": {"batch_size": 512},
        }
        strata = [{"stratum_id": "a", "foj_count": 10.0, "alpha": 1.0}]
        self.assertIsNone(
            module.per_row_exposure(config, {"materialized_rows": None}, strata)
        )
        self.assertIsNone(
            module.per_row_exposure(config, {"materialized_rows": 2_000_000}, [])
        )

    def test_a_refusal_still_writes_the_report(self):
        # The report is needed most when discovery refuses, and that was the
        # one case producing none: the error came out of the constructor, so
        # the source never existed and the script died with a traceback and no
        # file.  An impossible row floor is the cheapest way to force a
        # refusal with a fully populated status behind it.
        import subprocess

        directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        config = directory / "config.yaml"
        config.write_text(
            pathlib.Path("model/configs/resmade_smoke.yaml").read_text() + self.RARE_BLOCKS
        )
        output = directory / "report.json"
        result = subprocess.run(
            [sys.executable, self.SCRIPT, "--config", str(config), "--output", str(output)],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("DISCOVERY REFUSED", result.stdout)
        self.assertTrue(output.exists(), result.stderr)
        report = json.loads(output.read_text())
        self.assertIn("no strata", report["error"])
        self.assertEqual(report["selected_strata"], [])
        # The status that explains the refusal has to survive the raise, which
        # is the whole point: a traceback alone says a run failed, not why.
        self.assertEqual(report["conjunctive"]["minimum_stratum_rows"], 1e9)
        self.assertTrue(report["conjunctive"]["row_floor_applied"])

    def test_both_forms_reach_the_same_decision(self):
        # Not merely "no import error": both must get far enough to judge the
        # config, so a bootstrap that imports a different copy of the package
        # would show up here rather than passing quietly.
        by_path = self._run([self.SCRIPT])
        as_module = self._run(["-m", "model.scripts.inspect_rare_support"])
        self.assertEqual(by_path.returncode, as_module.returncode)
        self.assertEqual(by_path.stdout, as_module.stdout)


class ReadOnlyFixtureGuardTest(unittest.TestCase):
    """A fixture consumer must not be able to rebuild the fixture it reads.

    The 2M-row JOB-light fixture takes a separate prepare job to build.  A
    training config that merely consumes it shares the prepared directory with
    the builder, so running the prepare script against the consumer would
    silently overwrite a two-million-row fixture with this config's own much
    smaller sample -- the failure that produced a 512-row "2M" fixture once
    already.
    """

    def _run(self, config_path, rebuild):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "prepare_neurocard_data_under_test",
            "model/scripts/prepare_neurocard_data.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        argv = ["prepare_neurocard_data.py", "--config", str(config_path)]
        if rebuild:
            argv.append("--rebuild-domains")
        original = sys.argv
        sys.argv = argv
        try:
            module.main()
        finally:
            sys.argv = original

    def _config_path(self, read_only):
        config = pathlib.Path(
            "model/configs/"
            "job_light_duet_binary_native_anpm_20k_conjunctive_rare_auxiliary_fixture.yaml"
        ).read_text()
        directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, True)
        lines = []
        for line in config.splitlines():
            if line.startswith("  prepared_directory:"):
                line = f"  prepared_directory: {directory / 'prepared'}"
            lines.append(line)
            if line.startswith("dataset:") and read_only is not None:
                flag = "true" if read_only else "false"
                lines.append(f"  prepared_fixture_read_only: {flag}")
        path = directory / "config.yaml"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_a_config_that_omits_the_key_can_still_rebuild(self):
        # The guard must default to permissive: every prepare job written
        # before the key existed omits it, and refusing those would make the
        # guard a breaking change rather than a safety rail.
        try:
            self._run(self._config_path(read_only=None), rebuild=True)
        except SystemExit as exc:  # pragma: no cover - depends on local data
            self.assertNotIn("read-only", str(exc))
        except Exception:
            pass

    def test_rebuilding_a_read_only_fixture_consumer_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            self._run(self._config_path(read_only=True), rebuild=True)
        self.assertIn("read-only", str(caught.exception))

    def test_the_guard_only_fires_on_rebuild(self):
        # Without --rebuild-domains the same config must get past the guard.
        # A guard that refuses the config outright would block the training
        # run itself, so the test has to distinguish the two.
        try:
            self._run(self._config_path(read_only=True), rebuild=False)
        except SystemExit as exc:  # pragma: no cover - depends on local data
            self.assertNotIn("read-only", str(exc))
        except Exception:
            pass


class StratumRowFloorTest(unittest.TestCase):
    """A stratum must cover enough rows that oversampling is not memorisation.

    ``minimum_joint_rows`` guards only conjunctions.  The depth-1 path had no
    row floor, which was safe only while the candidate columns were dense.
    Opening up a sparse high-cardinality column produced 53 of 64 strata
    covering one fixture row apiece and taking 83% of the auxiliary batch.
    """

    VALUES = 60

    def metadata(self):
        return ModelMetadata(
            columns=(
                ColumnMetadata("t:sparse", ColumnKind.DATA, tuple(range(self.VALUES)),
                               table="t"),
                ColumnMetadata("t:dense", ColumnKind.DATA, (0, 1), table="t"),
            ),
            full_join_cardinality=400.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )

    def rows(self):
        # Values 0..39 appear once each; 40..59 appear many times.
        rows = [[value, value % 2] for value in range(40)]
        for value in range(40, self.VALUES):
            rows.extend([[value, value % 2]] * 18)
        return np.asarray(rows, dtype=np.int64)

    def discover(self, floor=None):
        source = ExactRootStratumProvider.from_encoded_rows(self.metadata(), self.rows())
        limits = {"deficit_objective": "log_ratio"}
        if floor is not None:
            limits["minimum_stratum_rows"] = floor
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e9,
            max_selected_strata=32,
            conjunctive={"enabled": True, "max_depth": 2, "minimum_joint_rows": 1.0,
                         "max_joint_row_fraction": 1.0, "require_distinct_tables": False},
            discovery_limits=limits,
        )
        return source, selected

    def test_without_a_floor_single_row_strata_dominate(self):
        # log_ratio ranks an unboundedly rare region top, and a one-row region
        # is the rarest there is.
        _, selected = self.discover()
        singletons = [item for item in selected if item.foj_count <= 1]
        self.assertGreater(len(singletons), len(selected) // 2)

    def test_the_floor_removes_them(self):
        source, selected = self.discover(floor=10.0)
        self.assertTrue(selected)
        self.assertTrue(all(item.foj_count >= 10.0 for item in selected))
        self.assertGreater(source.conjunctive_status["candidates_below_row_floor"], 0)

    def test_the_status_reports_the_row_distribution_either_way(self):
        for floor in (None, 10.0):
            with self.subTest(floor=floor):
                source, _ = self.discover(floor=floor)
                rows = source.conjunctive_status["selected_stratum_rows"]
                self.assertIn("min", rows)
                self.assertIn("at_or_below_one", rows)

    def test_the_report_names_how_many_single_row_strata_were_selected(self):
        source, _ = self.discover()
        self.assertGreater(
            source.conjunctive_status["selected_stratum_rows"]["at_or_below_one"], 0
        )
        source, _ = self.discover(floor=10.0)
        self.assertEqual(
            source.conjunctive_status["selected_stratum_rows"]["at_or_below_one"], 0
        )

    def test_a_zero_floor_changes_nothing(self):
        _, without = self.discover()
        _, zero = self.discover(floor=0.0)
        self.assertEqual(
            [item.stratum_id for item in without], [item.stratum_id for item in zero]
        )

    def test_a_candidate_exactly_at_the_floor_survives(self):
        # "minimum_stratum_rows" names a minimum, so a region holding exactly
        # that many rows clears it.  An off-by-one here silently raises the
        # floor by one row for every run that sets it.
        source, selected = self.discover(floor=18.0)
        self.assertTrue(selected)
        self.assertEqual(min(item.foj_count for item in selected), 18.0)

    def test_the_floor_does_not_apply_without_materialized_rows(self):
        # The live NeuroCard provider carries root-table counts, not
        # full-outer-join row counts, so the floor must not prune there -- and
        # must say so, because a floor that quietly does nothing would
        # otherwise look exactly like a floor that found nothing to prune.
        meta = self.metadata()
        rows = self.rows()
        counts = np.bincount(rows[:, 0], minlength=self.VALUES).astype(float)
        masses = {
            0: RootColumnMass(
                column_index=0,
                values=meta.columns[0].domain,
                counts=counts,
                total_count=float(counts.sum()),
                source="test",
                semantic_type="categorical",
            )
        }
        source = ExactRootStratumProvider(meta, masses)
        selected = source.discover(
            n_total=1_000_000,
            predicate_probabilities=PROBABILITIES,
            minimum_expected_context_support=1e9,
            max_selected_strata=32,
            discovery_limits={"deficit_objective": "log_ratio",
                              "minimum_stratum_rows": 10.0},
        )
        self.assertTrue(selected)
        self.assertFalse(source.conjunctive_status["row_floor_applied"])
        self.assertEqual(source.conjunctive_status["candidates_below_row_floor"], 0)
        self.assertTrue(any(item.foj_count < 10.0 for item in selected))

    def test_the_status_says_the_floor_was_applied_when_it_was(self):
        source, _ = self.discover(floor=10.0)
        self.assertTrue(source.conjunctive_status["row_floor_applied"])
        source, _ = self.discover()
        self.assertFalse(source.conjunctive_status["row_floor_applied"])

    def test_the_floor_is_rejected_when_negative(self):
        config = {
            "dataset": {"type": "synthetic_full_join"},
            "model": {"type": "predicate_resmade", "hidden_sizes": [8]},
            "training": {"batch_size": 2, "learning_rate": 0.1, "epochs": 1,
                         "steps_per_epoch": 1},
            "predicate_generation": {"enabled": True},
            "rare_support": {"enabled": True, "discovery": {"minimum_stratum_rows": -1}},
            "rare_auxiliary": {"enabled": True, "batch_size": 2, "beta": 0.25},
        }
        with self.assertRaisesRegex(ValueError, "minimum_stratum_rows"):
            validate_config(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
