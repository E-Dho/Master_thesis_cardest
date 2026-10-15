"""Conjunctive rare-support strata: representation, scoring, search, forcing.

These run without PyTorch so the discovery machinery stays testable on any
machine that can run the sampler.
"""

from __future__ import annotations

import unittest

import numpy as np

from model.src.config import validate_config
from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import (
    ExactRootStratumProvider,
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
    membership_matrix,
    TERM_OPERATOR_FAMILY,
    served_head_indices,
    table_subset_inclusion_probability,
    with_realization_probabilities,
    with_conditional_score,
)
from model.src.data.full_join_sampler import LiveNeuroCardFullJoinSampleSource
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
