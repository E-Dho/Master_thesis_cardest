"""Conjunctive rare-support strata: representation, scoring, search, forcing.

These run without PyTorch so the discovery machinery stays testable on any
machine that can run the sampler.
"""

from __future__ import annotations

import unittest

import numpy as np

from model.src.config import validate_config
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.strata import (
    ExactRootStratumProvider,
    RootColumnMass,
    RootDataStratum,
    StratumTerm,
    _capped_allocation,
    build_membership_lookup,
    conditional_head_support,
    membership_matrix,
    served_head_indices,
    table_subset_inclusion_probability,
    with_conditional_score,
)
from model.src.data.full_join_sampler import LiveNeuroCardFullJoinSampleSource
from model.src.predicates.generation import _forced_terms_for_stratum


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


def stratum(*terms: StratumTerm, probability: float = 0.1) -> RootDataStratum:
    return RootDataStratum(
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

    def test_served_heads_are_the_data_columns_after_the_last_term(self):
        meta = metadata()
        conjunction = stratum(term(0, "T.year", 1952), term(1, "K.keyword", 1))
        self.assertEqual(served_head_indices(conjunction, meta), (2, 3))

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
        self.assertEqual(len(served_head_indices(early, meta)), 3)
        self.assertEqual(len(served_head_indices(late, meta)), 1)
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


class TermSubsetTest(unittest.TestCase):
    def terms(self, probability, seed):
        conjunction = stratum(
            term(0, "T.year", 1952), term(1, "K.keyword", 1), term(2, "T.phonetic", 3)
        )
        return _forced_terms_for_stratum(
            conjunction, np.random.default_rng(seed), probability
        )

    def test_zero_probability_always_forces_the_whole_conjunction(self):
        for seed in range(20):
            self.assertEqual(len(self.terms(0.0, seed)), 3)

    def test_certain_subsetting_always_forces_a_non_empty_proper_subset(self):
        sizes = {len(self.terms(1.0, seed)) for seed in range(40)}
        self.assertTrue(sizes)
        self.assertTrue(all(1 <= size <= 2 for size in sizes), sizes)

    def test_a_depth_one_stratum_is_never_subsetted(self):
        single = stratum(term(0, "T.year", 1952))
        self.assertEqual(
            len(_forced_terms_for_stratum(single, np.random.default_rng(0), 1.0)), 1
        )

    def test_terms_stay_ordered_by_column(self):
        for seed in range(20):
            chosen = self.terms(1.0, seed)
            indices = [item.column_index for item in chosen]
            self.assertEqual(indices, sorted(indices))


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

    def test_an_out_of_range_term_subset_probability_is_rejected(self):
        config = self.config(max_depth=2)
        config["rare_auxiliary"]["term_subset_probability"] = 1.5
        with self.assertRaisesRegex(ValueError, "term_subset_probability"):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
