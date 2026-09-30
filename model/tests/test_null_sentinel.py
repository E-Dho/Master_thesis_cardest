from __future__ import annotations

import unittest

import numpy as np

from model.src.data.null_sentinel import (
    NULL_SENTINEL_TOKEN,
    NullSentinelConfig,
    apply_null_sentinel_to_metadata,
    build_void_pair_catalog,
    column_has_null_sentinel,
    eligible_catalog_columns,
    null_sentinel_index,
    sample_chain_extension_count,
)
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
)
from model.src.predicates.encoding import column_factor, predicate_mask
from model.src.predicates.generation import (
    GeneratedTrainingContext,
    PredicateTrainingContextGenerator,
    context_satisfies_row,
    void_target_row,
)
from model.src.predicates.operators import PredicateOp, PredicateToken


def _metadata() -> ModelMetadata:
    """Three small categorical DATA columns plus an indicator and a fanout.

    The third DATA column matters: because ResMADE masks head j from its own
    token, a void on columns (0, 1) can only be supervised at a head that comes
    after both, so a schema with just the pair has nowhere to put the sentinel.
    """

    return ModelMetadata(
        columns=(
            ColumnMetadata(
                name="t:colour",
                kind=ColumnKind.DATA,
                domain=(1, 2, 3, 4),
                table="t",
            ),
            ColumnMetadata(
                name="t:size",
                kind=ColumnKind.DATA,
                domain=(10, 20, 30),
                table="t",
            ),
            ColumnMetadata(
                name="t:grade",
                kind=ColumnKind.DATA,
                domain=(100, 200, 300),
                table="t",
            ),
            ColumnMetadata(name="I_t", kind=ColumnKind.INDICATOR, domain=(0, 1), table="t"),
            ColumnMetadata(name="F_t", kind=ColumnKind.FANOUT, domain=(1, 2), table="t"),
        ),
        full_join_cardinality=1000.0,
    )


def _enabled_config(**overrides: object) -> NullSentinelConfig:
    base = {
        "enabled": True,
        "min_marginal_count": 1,
        "min_anchor_partners": 1,
    }
    base.update(overrides)
    return NullSentinelConfig(**base)  # type: ignore[arg-type]


class NullSentinelMetadataTest(unittest.TestCase):
    def test_disabled_config_returns_metadata_unchanged(self) -> None:
        metadata = _metadata()
        self.assertIs(
            apply_null_sentinel_to_metadata(metadata, NullSentinelConfig()),
            metadata,
        )

    def test_sentinel_is_appended_only_to_data_columns(self) -> None:
        extended = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        self.assertEqual(extended.columns[0].domain, (1, 2, 3, 4, NULL_SENTINEL_TOKEN))
        self.assertEqual(extended.columns[1].domain, (10, 20, 30, NULL_SENTINEL_TOKEN))
        self.assertEqual(extended.columns[2].domain, (100, 200, 300, NULL_SENTINEL_TOKEN))
        self.assertEqual(extended.columns[3].domain, (0, 1))
        self.assertEqual(extended.columns[4].domain, (1, 2))
        self.assertTrue(column_has_null_sentinel(extended.columns[0]))
        self.assertFalse(column_has_null_sentinel(extended.columns[3]))
        self.assertIsNone(null_sentinel_index(extended.columns[4]))

    def test_existing_encoded_ids_keep_their_meaning(self) -> None:
        original = _metadata()
        extended = apply_null_sentinel_to_metadata(original, _enabled_config())
        for index, column in enumerate(original.columns):
            for encoded_id, value in enumerate(column.domain):
                self.assertEqual(extended.columns[index].domain[encoded_id], value)

    def test_applying_twice_does_not_stack_sentinels(self) -> None:
        config = _enabled_config()
        once = apply_null_sentinel_to_metadata(_metadata(), config)
        twice = apply_null_sentinel_to_metadata(once, config)
        self.assertEqual(twice.columns[0].domain, once.columns[0].domain)

    def test_output_width_grows_by_one_per_data_column(self) -> None:
        original = _metadata()
        extended = apply_null_sentinel_to_metadata(original, _enabled_config())
        self.assertEqual(
            sum(extended.data_output_bins) - sum(original.data_output_bins), 3
        )

    def test_factorization_plan_is_built_on_the_extended_domain(self) -> None:
        extended = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        plan = apply_factorization_to_metadata(
            extended,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=1,
                minimum_domain_size=3,
            ),
        ).factorization_plan
        factorization = plan.factorization_for_column(0)
        self.assertIsNotNone(factorization)
        # The sentinel must be a *valid* encoded id, not masked-out slack.
        self.assertEqual(factorization.original_domain_size, 5)


class NullSentinelMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        self.column = self.metadata.columns[0]
        self.sentinel = null_sentinel_index(self.column)

    def test_sentinel_never_satisfies_any_predicate(self) -> None:
        tokens = (
            PredicateToken.equal(2),
            PredicateToken.range(1, 4),
            PredicateToken(PredicateOp.GREATER_EQUAL, value=1),
            PredicateToken(PredicateOp.LESS_EQUAL, value=4),
            PredicateToken(PredicateOp.GREATER_THAN, value=0),
            PredicateToken(PredicateOp.LESS_THAN, value=9),
            PredicateToken.wildcard(),
        )
        for token in tokens:
            with self.subTest(op=token.op.value):
                self.assertEqual(predicate_mask(self.column, token)[self.sentinel], 0.0)

    def test_wildcard_factor_is_one_however_much_sentinel_mass_there_is(self) -> None:
        # A wildcard column must contribute exactly 1.0. Returning
        # 1 - q(sentinel) here lets unpredicated heads vote on impossibility,
        # and on the 50M POL benchmark two mostly-wildcarded heads settled on
        # constant sentinel mass that multiplied a fixed ~1/16 into every
        # estimate -- shrinking true zeros and true positives alike.
        for sentinel_mass in (0.0, 0.1, 0.6, 0.99):
            with self.subTest(sentinel_mass=sentinel_mass):
                rest = (1.0 - sentinel_mass) / 4
                distribution = np.array([rest] * 4 + [sentinel_mass])
                self.assertAlmostEqual(
                    column_factor(distribution, self.column, PredicateToken.wildcard()),
                    1.0,
                )

    def test_wildcard_factor_stays_one_without_a_sentinel(self) -> None:
        plain = _metadata().columns[0]
        distribution = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(
            column_factor(distribution, plain, PredicateToken.wildcard()), 1.0
        )

    def test_equality_factor_ignores_sentinel_mass(self) -> None:
        distribution = np.array([0.0, 0.3, 0.0, 0.0, 0.7])
        factor = column_factor(distribution, self.column, PredicateToken.equal(2))
        self.assertAlmostEqual(factor, 0.3)

    def test_full_sentinel_mass_collapses_every_predicated_factor(self) -> None:
        distribution = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        for token in (
            PredicateToken.equal(2),
            PredicateToken.range(1, 4),
            PredicateToken(PredicateOp.GREATER_EQUAL, value=1),
        ):
            with self.subTest(op=token.op.value):
                self.assertAlmostEqual(
                    column_factor(distribution, self.column, token), 0.0
                )
        # ... but a column the context does not constrain still contributes 1.0.
        self.assertAlmostEqual(
            column_factor(distribution, self.column, PredicateToken.wildcard()), 1.0
        )


class VoidPairCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        # colour 1 pairs with sizes 10/20/30; colour 2 pairs only with 10.
        # (colour=1, size=*) is therefore never void, while (colour=2, size=20)
        # and (colour=2, size=30) are.
        self.rows = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 2, 1, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 1, 0],
            ],
            dtype=int,
        )

    def test_eligible_columns_exclude_indicators_and_fanouts(self) -> None:
        self.assertEqual(
            eligible_catalog_columns(self.metadata, _enabled_config()), (0, 1, 2)
        )

    def test_large_domains_are_excluded_from_the_catalog(self) -> None:
        config = _enabled_config(max_catalog_domain_size=3)
        # colour has four real values; size and grade have three.
        self.assertEqual(eligible_catalog_columns(self.metadata, config), (1, 2))

    def test_absent_pairs_are_reported_as_void(self) -> None:
        catalog = build_void_pair_catalog(self.rows, self.metadata, _enabled_config())
        self.assertIsNotNone(catalog)
        self.assertEqual(sorted(catalog.absent_values_for(0, 1, 1)), [1, 2])
        self.assertEqual(list(catalog.absent_values_for(0, 0, 1)), [])

    def test_marginal_floor_suppresses_poorly_supported_values(self) -> None:
        catalog = build_void_pair_catalog(
            self.rows, self.metadata, _enabled_config(min_marginal_count=3)
        )
        # size=20 and size=30 occur once each, below the floor, so they can no
        # longer be claimed impossible.
        self.assertIsNone(catalog)

    def test_anchor_partner_guard_rejects_narrow_branches(self) -> None:
        catalog = build_void_pair_catalog(
            self.rows, self.metadata, _enabled_config(min_anchor_partners=2)
        )
        # colour=1 co-occurs with exactly one size, so voiding one of its
        # partners would teach "colour=1 implies void" rather than a pair fact.
        # With the guard no anchor value on that pair qualifies, so the pair is
        # not merely unsampleable -- it never enters the catalog, and therefore
        # cannot depress the share cap's 1/len(pairs) floor either.
        if catalog is not None:
            self.assertNotIn((0, 1), dict.fromkeys(catalog._present))
            self.assertEqual(catalog.present_partner_count(0, 1, 1), 0)
            self.assertEqual(list(catalog.absent_values_for(0, 1, 1)), [])

    def test_sampled_void_pair_is_anchored_on_the_row_value(self) -> None:
        catalog = build_void_pair_catalog(self.rows, self.metadata, _enabled_config())
        rng = np.random.default_rng(7)
        for _ in range(20):
            pair = catalog.sample_void_pair(self.rows[3], rng)
            if pair is None:
                continue
            self.assertEqual(
                pair.anchor_value_id, int(self.rows[3][pair.anchor_column_index])
            )
            self.assertIn(
                pair.void_value_id,
                list(
                    catalog.absent_values_for(
                        pair.anchor_column_index,
                        pair.anchor_value_id,
                        pair.void_column_index,
                    )
                ),
            )

    def test_catalog_is_none_when_no_void_exists(self) -> None:
        dense = np.array(
            [
                [colour, size, grade, 1, 0]
                for colour in range(4)
                for size in range(3)
                for grade in range(3)
            ],
            dtype=int,
        )
        self.assertIsNone(
            build_void_pair_catalog(dense, self.metadata, _enabled_config())
        )


class PairShareCapTest(unittest.TestCase):
    """A functionally dependent pair must not consume the whole void budget."""

    def setUp(self) -> None:
        # Column 2 is functionally dependent on column 0 (value a only ever
        # occurs with fd <= a), so it offers a void for almost every anchor
        # value. Column 1 offers one only when a == 0. A pair is proposed
        # uniformly but accepted only when it can produce a void for this row,
        # so the dependent pair wins the rejection round far more often --
        # which is exactly how segment_idx / num_of_segments crowded out every
        # other relationship on the 50M run.
        self.metadata = apply_null_sentinel_to_metadata(
            ModelMetadata(
                columns=(
                    ColumnMetadata("t:a", ColumnKind.DATA, tuple(range(4)), table="t"),
                    ColumnMetadata("t:b", ColumnKind.DATA, tuple(range(4)), table="t"),
                    ColumnMetadata("t:fd", ColumnKind.DATA, tuple(range(4)), table="t"),
                    ColumnMetadata("t:tail", ColumnKind.DATA, tuple(range(3)), table="t"),
                    ColumnMetadata("I_t", ColumnKind.INDICATOR, (0, 1), table="t"),
                ),
                full_join_cardinality=1000.0,
            ),
            _enabled_config(),
        )
        rows = []
        for a in range(4):
            for b in ([0, 1] if a == 0 else [0, 1, 2, 3]):
                for fd in range(a + 1):
                    for tail in range(3):
                        rows.append([a, b, fd, tail, 1])
        self.rows = np.array(rows, dtype=int)

    def _draw(self, share: float, draws: int = 900):
        catalog = build_void_pair_catalog(
            self.rows, self.metadata, _enabled_config(max_pair_share=share)
        )
        self.assertIsNotNone(catalog)
        rng = np.random.default_rng(0)
        counts: dict[tuple[int, int], int] = {}
        for index in range(draws):
            pair = catalog.sample_void_pair(self.rows[index % len(self.rows)], rng)
            if pair is None:
                continue
            # The generator charges a pair only once its context materialized,
            # so the cap sees committed usage rather than raw draws.
            catalog.commit_void_pair(pair)
            key = tuple(sorted((pair.anchor_column_index, pair.void_column_index)))
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        self.assertTrue(total, "no void pair could be drawn at all")
        return catalog, counts, counts.get((0, 2), 0) / total

    def test_dependency_pair_dominates_without_a_cap(self) -> None:
        _, counts, share = self._draw(1.0)
        self.assertGreater(share, 0.5, f"expected FD dominance, got {counts}")

    def test_cap_reduces_the_dependency_pair_share(self) -> None:
        _, _, uncapped = self._draw(1.0)
        _, counts, capped = self._draw(0.34)
        self.assertLess(capped, uncapped)
        self.assertGreaterEqual(
            len(counts), 2, "the freed budget must reach another pair"
        )

    def test_configured_share_is_actually_respected(self) -> None:
        # The cap is enforced on every attempt, so no pair may finish above it.
        # The floor at 1/len(pairs) keeps it satisfiable.
        catalog, counts, _ = self._draw(0.5)
        total = sum(counts.values())
        allowed = max(0.5, 1.0 / catalog.pair_count)
        for pair, count in counts.items():
            # The projected-share test permits at most one draw of overshoot,
            # which decays as the run goes on.
            self.assertLessEqual(
                count / total,
                allowed + 1.0 / total + 1e-9,
                f"pair {pair} took {count / total:.3f} of a {allowed:.3f} budget",
            )

    def test_single_viable_pair_is_never_starved(self) -> None:
        # Only pairs that can produce a void enter the catalog, so the cap's
        # 1/len(pairs) floor is computed over viable pairs alone. With one such
        # pair the floor is 1.0 and nothing is throttled -- no unsatisfiable
        # cap, and so no need to fall back to ordinary contexts.
        rows = np.array(
            [
                [a, 0, fd, tail, 1]
                for a in range(4)
                for fd in range(a + 1)
                for tail in range(3)
            ],
            dtype=int,
        )
        catalog = build_void_pair_catalog(
            rows, self.metadata, _enabled_config(max_pair_share=0.01)
        )
        self.assertIsNotNone(catalog)
        self.assertEqual(catalog.pair_count, 1)
        rng = np.random.default_rng(0)
        drawn = 0
        for index in range(300):
            pair = catalog.sample_void_pair(rows[index % len(rows)], rng)
            if pair is None:
                continue
            catalog.commit_void_pair(pair)
            drawn += 1
        # The invariant is that the CAP never refuses a draw; some rows still
        # fail the sampler's own eligibility rules, which is unrelated.
        self.assertGreater(drawn, 0)
        self.assertEqual(catalog.pair_usage_summary()["share_cap_skips"], 0)

    def test_catalog_drops_pairs_that_can_never_produce_a_void(self) -> None:
        # Columns 0 and 1 co-occur densely here, so that pair is unusable and
        # must not inflate the denominator of the share cap's floor.
        rows = np.array(
            [
                [a, b, fd, tail, 1]
                for a in range(4)
                for b in range(4)
                for fd in range(a + 1)
                for tail in range(3)
            ],
            dtype=int,
        )
        catalog = build_void_pair_catalog(rows, self.metadata, _enabled_config())
        self.assertIsNotNone(catalog)
        self.assertEqual(catalog.pair_count, 1)
        self.assertEqual(list(catalog.absent_values_for(0, 0, 1)), [])

    def test_usage_summary_reports_the_spread(self) -> None:
        catalog, _, _ = self._draw(0.34)
        summary = catalog.pair_usage_summary()
        self.assertGreater(summary["void_pairs_drawn"], 0)
        self.assertGreaterEqual(summary["distinct_pairs_used"], 2)
        self.assertLessEqual(summary["top_pairs"][0]["share"], 1.0)


class VoidTargetRowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        self.row = np.array([2, 1, 0, 1, 0], dtype=int)

    def test_sentinel_is_written_at_the_void_column(self) -> None:
        targets = void_target_row(self.row, self.metadata, 1, cascade=False)
        self.assertEqual(targets[0], 2)
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))

    def test_cascade_propagates_to_later_data_columns_only(self) -> None:
        targets = void_target_row(self.row, self.metadata, 0, cascade=True)
        self.assertEqual(targets[0], null_sentinel_index(self.metadata.columns[0]))
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))
        self.assertEqual(targets[2], null_sentinel_index(self.metadata.columns[2]))
        # Indicator and fanout heads keep their real values: table presence and
        # the inverse fanout weights have no meaningful void state.
        self.assertEqual(targets[3], 1)
        self.assertEqual(targets[4], 0)

    def test_cascade_ignores_whether_a_later_column_is_predicated(self) -> None:
        # A head cannot observe its own predicate token, so its target must not
        # depend on it. Skipping unpredicated heads would hand head k the
        # sentinel in one void context and a real value in another with the
        # same visible prefix, and the head could only fit how often predicate
        # generation happened to constrain it.
        targets = void_target_row(self.row, self.metadata, 1, cascade=True)
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))
        self.assertEqual(targets[2], null_sentinel_index(self.metadata.columns[2]))

    def test_cascade_can_be_disabled(self) -> None:
        targets = void_target_row(self.row, self.metadata, 1, cascade=False)
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))
        self.assertEqual(targets[2], 0)

    def test_source_row_is_not_mutated(self) -> None:
        void_target_row(self.row, self.metadata, 0, cascade=True)
        self.assertEqual(list(self.row), [2, 1, 0, 1, 0])

    def test_column_without_sentinel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            void_target_row(self.row, self.metadata, 4, cascade=False)


class ChainExtensionTest(unittest.TestCase):
    def test_zero_probability_never_extends(self) -> None:
        config = _enabled_config(chain_extension_probability=0.0)
        rng = np.random.default_rng(0)
        self.assertEqual(
            {sample_chain_extension_count(rng, config) for _ in range(50)}, {0}
        )

    def test_extension_count_is_geometric_and_capped(self) -> None:
        config = _enabled_config(
            chain_extension_probability=0.2, max_chain_extensions=5
        )
        rng = np.random.default_rng(11)
        draws = [sample_chain_extension_count(rng, config) for _ in range(20_000)]
        self.assertLessEqual(max(draws), 5)
        share_zero = draws.count(0) / len(draws)
        share_one = draws.count(1) / len(draws)
        # P(0) = 0.8 and P(1) = 0.16 for a geometric chain at p = 0.2.
        self.assertAlmostEqual(share_zero, 0.8, delta=0.02)
        self.assertAlmostEqual(share_one, 0.16, delta=0.02)
        self.assertGreater(draws.count(0), draws.count(1))
        self.assertGreater(draws.count(1), draws.count(2))


class VoidContextGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _enabled_config(void_probability=1.0)
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), self.config)
        self.rows = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 2, 1, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 2, 1, 0],
            ],
            dtype=int,
        )
        self.catalog = build_void_pair_catalog(self.rows, self.metadata, self.config)
        # No wildcards: the sentinel head must carry a predicate, so a fixture
        # that frequently leaves the trailing column unconstrained would mostly
        # produce no void at all and hide regressions.
        self.generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "seed": 0,
            }
        )

    def test_injection_is_inert_until_a_catalog_is_attached(self) -> None:
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(all(c.void_column_index is None for c in contexts))
        np.testing.assert_array_equal(targets, self.rows)

    def test_void_contexts_target_the_sentinel(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(3),
        )
        void_contexts = [
            (context, target)
            for context, target in zip(contexts, targets)
            if context.void_column_index is not None
        ]
        self.assertTrue(void_contexts, "void_probability=1.0 produced no void context")
        for context, target in void_contexts:
            index = context.void_column_index
            self.assertEqual(
                target[index], null_sentinel_index(self.metadata.columns[index])
            )
            # The void predicate must actually reach the token row, otherwise
            # the model is asked to predict the sentinel from nothing.
            self.assertEqual(context.tokens[index].op, PredicateOp.EQUAL)

    def test_void_predicate_is_unsatisfied_by_the_source_row(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        rng = np.random.default_rng(5)
        compiled = self.generator._compiled(self.metadata)
        for row in self.rows:
            base = self.generator._generate_one(row, self.metadata, rng)
            for context in self.generator._void_and_hard_negative(
                row, self.metadata, rng, compiled, base
            ):
                if context.void_column_index is None:
                    continue
                # The supervised head's own predicate is row-satisfied; it is
                # the pair's void column that the row fails.
                anchor, void = context.void_pair_columns
                self.assertLess(void, context.void_column_index)
                anchor_value = self.metadata.columns[anchor].domain[int(row[anchor])]
                self.assertTrue(context.tokens[anchor].satisfies(anchor_value))
                void_value = self.metadata.columns[void].domain[int(row[void])]
                self.assertFalse(context.tokens[void].satisfies(void_value))

    def test_row_satisfaction_validation_exempts_void_contexts(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        rng = np.random.default_rng(1)
        compiled = self.generator._compiled(self.metadata)
        contexts = []
        rows = []
        for _ in range(20):
            for row in self.rows:
                base = self.generator._generate_one(row, self.metadata, rng)
                for context in self.generator._void_and_hard_negative(
                    row, self.metadata, rng, compiled, base
                ):
                    if context.void_column_index is not None:
                        contexts.append(context)
                        rows.append(row)
        self.assertTrue(contexts, "no void context could be built for any row")
        for context, row in zip(contexts, rows):
            self.assertTrue(context_satisfies_row(context, row, self.metadata))

    def test_sentinel_targets_are_positional_not_predicate_dependent(self) -> None:
        # Every sentinel-bearing DATA column from the supervised head onward is
        # targeted, whether or not it carries a predicate. Making that depend on
        # a head's own token would give two void contexts with the same visible
        # prefix different targets at that head, which it cannot tell apart.
        self.generator.attach_void_catalog(self.catalog, self.config)
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(3),
        )
        checked = 0
        for context, target in zip(contexts, targets):
            if context.void_column_index is None:
                continue
            checked += 1
            for index, column in enumerate(self.metadata.columns):
                sentinel = null_sentinel_index(column)
                if sentinel is None:
                    continue
                if index >= context.void_column_index:
                    self.assertEqual(
                        target[index],
                        sentinel,
                        f"column {index} at or after the break must be sentinel",
                    )
                else:
                    self.assertNotEqual(
                        target[index],
                        sentinel,
                        f"column {index} before the break must keep its value",
                    )
        self.assertTrue(checked, "no void context was produced")

    def test_every_void_comes_with_a_matched_hard_negative(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        rng = np.random.default_rng(5)
        compiled = self.generator._compiled(self.metadata)
        pairs = 0
        for _ in range(20):
          for row in self.rows:
            base = self.generator._generate_one(row, self.metadata, rng)
            emitted = self.generator._void_and_hard_negative(
                row, self.metadata, rng, compiled, base
            )
            if not emitted:
                continue
            pairs += 1
            self.assertEqual(len(emitted), 2, "hard_negative_probability=1 must pair")
            void, negative = emitted
            self.assertIsNotNone(void.void_column_index)
            self.assertIsNone(negative.void_column_index)
            # Same two columns constrained, so the model cannot tell the pair
            # apart by the shape of the query -- only by the value on the void
            # column, which the supervised head sits after and can therefore
            # actually observe.
            _anchor, void_column = void.void_pair_columns
            self.assertLess(void_column, void.void_column_index)
            self.assertEqual(negative.tokens[void_column].op, PredicateOp.EQUAL)
            self.assertNotEqual(
                negative.tokens[void_column].value, void.tokens[void_column].value
            )
            # The negative's value is the one this very row carries, so the
            # combination is valid by construction.
            self.assertEqual(
                negative.tokens[void_column].value,
                self.metadata.columns[void_column].domain[int(row[void_column])],
            )
        self.assertTrue(pairs, "no void/negative pair was produced")

    def test_pair_is_token_identical_except_the_void_value(self) -> None:
        # The whole point of the hard negative is that the model cannot separate
        # it from the void by the shape of the query. Sampling the extension
        # predicates independently for each side would break that, so assert
        # token equality on every column but the void one.
        self.generator.attach_void_catalog(self.catalog, self.config)
        rng = np.random.default_rng(5)
        compiled = self.generator._compiled(self.metadata)
        checked = 0
        for _ in range(40):
            for row in self.rows:
                base = self.generator._generate_one(row, self.metadata, rng)
                emitted = self.generator._void_and_hard_negative(
                    row, self.metadata, rng, compiled, base
                )
                if len(emitted) != 2:
                    continue
                void, negative = emitted
                checked += 1
                self.assertEqual(
                    set(void.ordinary_predicates), set(negative.ordinary_predicates)
                )
                self.assertEqual(void.included_tables, negative.included_tables)
                self.assertEqual(
                    void.inverse_fanout_columns, negative.inverse_fanout_columns
                )
                differing = [
                    index
                    for index in range(len(self.metadata.columns))
                    if void.tokens[index] != negative.tokens[index]
                ]
                # Exactly one token differs: the constrained void column. And it
                # must precede the supervised head, or that head could not see
                # the difference it is being asked to act on.
                self.assertEqual(
                    differing,
                    [differing[0]] if differing else [],
                    f"more than one token differs: {differing}",
                )
                self.assertEqual(len(differing), 1)
                self.assertLess(differing[0], void.void_column_index)
        self.assertGreater(checked, 5, "too few pairs to be meaningful")

    def test_hard_negative_targets_the_real_row_value(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(3),
        )
        negatives = [
            (c, t)
            for c, t in zip(contexts, targets)
            if c.void_column_index is None
            and len(c.ordinary_predicates) >= 2
        ]
        self.assertTrue(negatives)
        for context, target in negatives:
            for index, column in enumerate(self.metadata.columns):
                sentinel = null_sentinel_index(column)
                if sentinel is not None:
                    self.assertNotEqual(
                        target[index],
                        sentinel,
                        "a hard negative must never target the sentinel",
                    )

    def test_hard_negative_can_be_disabled(self) -> None:
        config = _enabled_config(void_probability=1.0, hard_negative_probability=0.0)
        self.generator.attach_void_catalog(self.catalog, config)
        rng = np.random.default_rng(5)
        compiled = self.generator._compiled(self.metadata)
        for row in self.rows:
            base = self.generator._generate_one(row, self.metadata, rng)
            emitted = self.generator._void_and_hard_negative(
                row, self.metadata, rng, compiled, base
            )
            self.assertLessEqual(len(emitted), 1)

    def test_wildcard_between_pair_and_carrier_does_not_drop_the_context(self) -> None:
        # Four DATA columns so the positional sentinel head (2) can be
        # wildcarded while a later head (3) still carries the collapse. Only the
        # carrier needs a predicate; requiring it of the positional head made
        # void coverage depend on that head's own predicate state, which it
        # cannot observe.
        metadata = apply_null_sentinel_to_metadata(
            ModelMetadata(
                columns=(
                    ColumnMetadata("t:a", ColumnKind.DATA, tuple(range(4)), table="t"),
                    ColumnMetadata("t:b", ColumnKind.DATA, tuple(range(4)), table="t"),
                    ColumnMetadata("t:c", ColumnKind.DATA, tuple(range(3)), table="t"),
                    ColumnMetadata("t:d", ColumnKind.DATA, tuple(range(3)), table="t"),
                    ColumnMetadata("I_t", ColumnKind.INDICATOR, (0, 1), table="t"),
                ),
                full_join_cardinality=1000.0,
            ),
            self.config,
        )
        rows = np.array(
            [[a, b, a % 3, b % 3, 1] for a in range(4) for b in range(a + 1)],
            dtype=int,
        )
        catalog = build_void_pair_catalog(rows, metadata, self.config)
        self.assertIsNotNone(catalog)
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "seed": 0,
            }
        )
        generator.attach_void_catalog(catalog, self.config)
        rng = np.random.default_rng(0)
        compiled = generator._compiled(metadata)
        produced = wildcard_positional = 0
        for _ in range(60):
            for row in rows:
                base = generator._generate_one(row, metadata, rng)
                # Drop the predicate on column 2 so it is wildcarded, keeping 3.
                thinned = {
                    name: token
                    for name, token in base.ordinary_predicates.items()
                    if name != "t:c"
                }
                if "t:d" not in thinned:
                    continue
                stripped = GeneratedTrainingContext(
                    tokens=base.tokens,
                    included_tables=base.included_tables,
                    inverse_fanout_columns=base.inverse_fanout_columns,
                    ordinary_predicates=thinned,
                )
                for context in generator._void_and_hard_negative(
                    row, metadata, rng, compiled, stripped
                ):
                    if context.void_column_index is None:
                        continue
                    produced += 1
                    if context.tokens[context.void_column_index].op is PredicateOp.WILDCARD:
                        wildcard_positional += 1
        self.assertGreater(produced, 0, "no void context was produced at all")
        self.assertGreater(
            wildcard_positional,
            0,
            "a wildcarded positional head must not drop the context",
        )

    def test_diagnostics_count_generated_voids(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(3),
        )
        diagnostics = self.generator.void_injection_diagnostics()
        self.assertTrue(diagnostics["enabled"])
        self.assertGreater(diagnostics["void_contexts_generated"], 0)

    def test_zero_probability_leaves_every_context_ordinary(self) -> None:
        config = _enabled_config(void_probability=0.0)
        self.generator.attach_void_catalog(self.catalog, config)
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(all(c.void_column_index is None for c in contexts))
        np.testing.assert_array_equal(targets, self.rows)


class TorchSentinelExclusionTest(unittest.TestCase):
    """The torch decode path must exclude the sentinel exactly as numpy does."""

    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - torch is an optional extra
            self.skipTest("torch is not installed")
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        self.column = self.metadata.columns[0]
        self.sentinel = null_sentinel_index(self.column)

    def _outputs(self, probabilities: list[float]) -> object:
        import torch

        from model.src.model.output_adapter import TorchBackboneOutputs

        logits = torch.log(torch.tensor([probabilities], dtype=torch.float64))
        return TorchBackboneOutputs(
            logits=logits,
            split_logits=[logits] + [logits] * (len(self.metadata.columns) - 1),
            output_embeddings=None,
        )

    def test_identity_wildcard_factor_is_one_despite_sentinel_mass(self) -> None:
        from model.src.model.output_adapter import TorchIdentityOutputAdapter

        outputs = self._outputs([0.1, 0.1, 0.1, 0.1, 0.6])
        factor = TorchIdentityOutputAdapter().column_factor(
            original_column_index=0,
            backbone_outputs=outputs,
            metadata=self.metadata,
            predicate_token=PredicateToken.wildcard(),
        )
        self.assertAlmostEqual(float(factor[0]), 1.0, places=6)

    def test_identity_equality_factor_ignores_sentinel_mass(self) -> None:
        from model.src.model.output_adapter import TorchIdentityOutputAdapter

        outputs = self._outputs([0.05, 0.25, 0.05, 0.05, 0.6])
        factor = TorchIdentityOutputAdapter().column_factor(
            original_column_index=0,
            backbone_outputs=outputs,
            metadata=self.metadata,
            predicate_token=PredicateToken.equal(2),
        )
        self.assertAlmostEqual(float(factor[0]), 0.25, places=6)

    def test_identity_factor_matches_numpy_reference(self) -> None:
        from model.src.model.output_adapter import TorchIdentityOutputAdapter

        probabilities = [0.05, 0.25, 0.10, 0.10, 0.50]
        outputs = self._outputs(probabilities)
        adapter = TorchIdentityOutputAdapter()
        for token in (
            PredicateToken.wildcard(),
            PredicateToken.equal(2),
            PredicateToken.range(1, 3),
            PredicateToken(PredicateOp.GREATER_EQUAL, value=2),
        ):
            with self.subTest(op=token.op.value):
                torch_factor = adapter.column_factor(
                    original_column_index=0,
                    backbone_outputs=outputs,
                    metadata=self.metadata,
                    predicate_token=token,
                )
                reference = column_factor(
                    np.array(probabilities), self.column, token
                )
                self.assertAlmostEqual(float(torch_factor[0]), reference, places=6)


class TableSubsetConsistencyTest(unittest.TestCase):
    """Adding the pair's tables must not leave stale fanouts or a hole."""

    def setUp(self) -> None:
        from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource

        self.source = SyntheticFullJoinSampleSource()
        self.config = _enabled_config(void_probability=1.0)
        self.metadata = apply_null_sentinel_to_metadata(
            self.source.metadata, self.config
        )
        self.catalog = build_void_pair_catalog(
            self.source.dataset.encoded_rows, self.metadata, self.config
        )
        self.generator = PredicateTrainingContextGenerator(
            {"enabled": True, "wildcard_probability": 0.0, "seed": 0}
        )
        self.generator.attach_void_catalog(self.catalog, self.config)

    def _void_contexts(self, draws: int = 60):
        if self.catalog is None:
            self.skipTest("synthetic fixture yields no void pair")
        rng = np.random.default_rng(0)
        compiled = self.generator._compiled(self.metadata)
        rows = np.asarray(self.source.dataset.encoded_rows)
        out = []
        for _ in range(draws):
            for row in rows:
                base = self.generator._generate_one(row, self.metadata, rng)
                out.extend(
                    context
                    for context in self.generator._void_and_hard_negative(
                        row, self.metadata, rng, compiled, base
                    )
                )
        return out

    def test_fanouts_are_derived_from_the_final_table_subset(self) -> None:
        contexts = self._void_contexts()
        self.assertTrue(contexts)
        for context in contexts:
            for index, column in enumerate(self.metadata.columns):
                if column.kind != ColumnKind.FANOUT:
                    continue
                active = context.tokens[index].op == PredicateOp.INV_FANOUT
                self.assertEqual(
                    active,
                    column.name in context.inverse_fanout_columns,
                    "emitted fanout tokens must match inverse_fanout_columns",
                )
                # An inverse fanout points at a table the subset excludes; if
                # that table is included the two statements contradict.
                if active:
                    child = self.generator._compiled(
                        self.metadata
                    ).fanout_child_by_name.get(column.name)
                    if child is not None:
                        self.assertNotIn(child, context.included_tables)

    def test_table_subset_stays_connected_to_the_root(self) -> None:
        from model.src.predicates.generation import (
            _root_connected_component_mask,
            _table_mask_from_tables,
        )

        compiled = self.generator._compiled(self.metadata)
        for context in self._void_contexts():
            mask = _table_mask_from_tables(set(context.included_tables), compiled)
            closed = _root_connected_component_mask(mask, compiled, compiled.graph)
            self.assertEqual(
                mask,
                closed,
                f"subset {sorted(context.included_tables)} is not root-connected",
            )


class JoinPathClosureTest(unittest.TestCase):
    """A pair spanning a gap must complete the path, not be discarded."""

    def _compiled(self):
        from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource

        source = SyntheticFullJoinSampleSource()
        generator = PredicateTrainingContextGenerator({"enabled": True, "seed": 0})
        metadata = apply_null_sentinel_to_metadata(source.metadata, _enabled_config())
        return generator._compiled(metadata), metadata

    def test_closure_adds_the_intermediate_table(self) -> None:
        from model.src.predicates.generation import (
            _root_connected_component_mask,
            _root_path_closure_mask,
            _table_mask_from_tables,
        )

        compiled, _ = self._compiled()
        graph = compiled.graph
        leaf = None
        for table in graph.tables:
            if table != graph.root_table and table not in {
                left for left, _ in graph.edges if left == graph.root_table
            }:
                leaf = table
        if leaf is None or len(graph.tables) < 3:
            self.skipTest("fixture has no chain to close over")

        # The pruning helper alone drops the far table, because it can only
        # remove: it never adds the intermediate that would connect it.
        sparse = _table_mask_from_tables({graph.root_table, leaf}, compiled)
        pruned = _root_connected_component_mask(sparse, compiled, graph)
        self.assertNotEqual(
            pruned & compiled.table_to_bit[leaf],
            compiled.table_to_bit[leaf],
            "the pruning helper unexpectedly kept a disconnected table",
        )

        # Closure completes the path instead.
        closed = _root_path_closure_mask({leaf}, compiled, graph)
        self.assertEqual(
            closed & compiled.table_to_bit[leaf], compiled.table_to_bit[leaf]
        )
        self.assertEqual(
            closed & compiled.table_to_bit[graph.root_table],
            compiled.table_to_bit[graph.root_table],
        )
        # And the result survives pruning, i.e. it really is connected.
        self.assertEqual(
            _root_connected_component_mask(closed, compiled, graph), closed
        )

    def test_unreachable_table_yields_no_subset(self) -> None:
        from model.src.predicates.generation import _root_path_closure_mask

        compiled, _ = self._compiled()
        self.assertEqual(
            _root_path_closure_mask({"__not_a_table__"}, compiled, compiled.graph), 0
        )


class AlternatePathClosureTest(unittest.TestCase):
    """Closure must search the row-present graph, not a precomputed tree."""

    def _diamond(self):
        from model.src.predicates.generation import (
            JoinGraphMetadata,
            _CompiledGenerationMetadata,
        )

        # A-B-D and A-C-D: two routes from the root to D.
        tables = ("A", "B", "C", "D")
        graph = JoinGraphMetadata(
            root_table="A",
            tables=tables,
            edges=(("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")),
        )
        table_to_bit = {table: 1 << index for index, table in enumerate(tables)}
        compiled = _CompiledGenerationMetadata.__new__(_CompiledGenerationMetadata)
        object.__setattr__(compiled, "graph", graph)
        object.__setattr__(compiled, "table_to_bit", table_to_bit)
        object.__setattr__(
            compiled, "bit_to_table", {bit: t for t, bit in table_to_bit.items()}
        )
        return compiled, graph, table_to_bit

    def test_closure_uses_a_route_the_row_can_supply(self) -> None:
        from model.src.predicates.generation import _root_path_closure_mask

        compiled, graph, bits = self._diamond()
        available = bits["A"] | bits["C"] | bits["D"]
        closed = _root_path_closure_mask(
            {"D"}, compiled, graph, available_mask=available
        )
        self.assertTrue(closed, "a usable route exists and must be found")
        # It must pick A-C-D, never pull in the absent B.
        self.assertEqual(closed & bits["B"], 0)
        for table in ("A", "C", "D"):
            self.assertEqual(closed & bits[table], bits[table])

    def test_unavailable_route_yields_no_subset(self) -> None:
        from model.src.predicates.generation import _root_path_closure_mask

        compiled, graph, bits = self._diamond()
        # D is present but neither intermediate is, so no route exists.
        available = bits["A"] | bits["D"]
        self.assertEqual(
            _root_path_closure_mask({"D"}, compiled, graph, available_mask=available), 0
        )

    def test_without_a_mask_any_route_is_acceptable(self) -> None:
        from model.src.predicates.generation import _root_path_closure_mask

        compiled, graph, bits = self._diamond()
        closed = _root_path_closure_mask({"D"}, compiled, graph)
        self.assertEqual(closed & bits["A"], bits["A"])
        self.assertEqual(closed & bits["D"], bits["D"])


class CausalVisibilityTest(unittest.TestCase):
    """The supervised head must be able to see what makes the context void.

    ResMADE masks head j to inputs from columns before j, so a head never
    observes its own predicate token. Supervising the void column would give the
    void and its twin identical visible input with opposite targets, and the
    only function the head could fit is the marginal P(sentinel | prefix) --
    which is exactly the constant sentinel mass the 50M run produced.
    """

    def setUp(self) -> None:
        self.config = _enabled_config(void_probability=1.0)
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), self.config)
        self.rows = np.array(
            [
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 2, 1, 0],
                [1, 0, 1, 1, 0],
                [1, 1, 2, 1, 0],
            ],
            dtype=int,
        )
        self.catalog = build_void_pair_catalog(self.rows, self.metadata, self.config)
        self.generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "seed": 0,
            }
        )
        self.generator.attach_void_catalog(self.catalog, self.config)

    @staticmethod
    def _visible_columns(head_index: int) -> set[int]:
        """Columns a ResMADE head can read.

        Stated directly rather than through torch so this invariant is checked
        even where the optional dependency is absent;
        ``test_matches_the_models_own_mask`` pins it to the real rule wherever
        torch is installed.
        """

        return set(range(head_index))

    def test_head_cannot_see_its_own_token(self) -> None:
        for head in range(6):
            self.assertNotIn(head, self._visible_columns(head))

    def test_matches_the_models_own_mask(self) -> None:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is an optional extra
            self.skipTest("torch is not installed")
        from model.src.model.masked_layers import mask_from_degrees

        count = 8
        degrees = torch.arange(count)
        # Input degree of column i is i, output degree of head j is j, and the
        # output layer and direct-IO both use strict=True.
        mask = mask_from_degrees(degrees, degrees, strict=True)
        for head in range(count):
            from_mask = {index for index in range(count) if mask[head, index] > 0}
            self.assertEqual(from_mask, self._visible_columns(head))

    def test_supervised_head_sees_both_constrained_columns(self) -> None:
        rng = np.random.default_rng(0)
        compiled = self.generator._compiled(self.metadata)
        checked = 0
        for _ in range(40):
            for row in self.rows:
                base = self.generator._generate_one(row, self.metadata, rng)
                for context in self.generator._void_and_hard_negative(
                    row, self.metadata, rng, compiled, base
                ):
                    if context.void_column_index is None:
                        continue
                    checked += 1
                    head = context.void_column_index
                    anchor, void = context.void_pair_columns
                    visible = self._visible_columns(head)
                    self.assertIn(anchor, visible, "anchor invisible to the head")
                    self.assertIn(void, visible, "void column invisible to the head")
        self.assertGreater(checked, 5, "too few void contexts to be meaningful")

    def test_supervised_head_carries_a_predicate(self) -> None:
        # A wildcard column contributes exactly 1.0, so supervising an
        # unpredicated head could never move the estimate at inference.
        rng = np.random.default_rng(1)
        compiled = self.generator._compiled(self.metadata)
        checked = 0
        for _ in range(40):
            for row in self.rows:
                base = self.generator._generate_one(row, self.metadata, rng)
                for context in self.generator._void_and_hard_negative(
                    row, self.metadata, rng, compiled, base
                ):
                    if context.void_column_index is None:
                        continue
                    checked += 1
                    head = context.void_column_index
                    self.assertNotEqual(context.tokens[head].op, PredicateOp.WILDCARD)
        self.assertGreater(checked, 5)


class NullSentinelConfigTest(unittest.TestCase):
    def test_defaults_are_disabled(self) -> None:
        self.assertFalse(NullSentinelConfig.from_dict(None).enabled)
        self.assertFalse(NullSentinelConfig.from_dict({}).enabled)

    def test_invalid_values_are_rejected(self) -> None:
        cases = (
            {"enabled": True, "void_probability": 1.5},
            {"enabled": True, "chain_extension_probability": 1.0},
            {"enabled": True, "max_chain_extensions": -1},
            {"enabled": True, "max_catalog_domain_size": 1},
            {"enabled": True, "max_catalog_columns": 1},
            {"enabled": True, "min_marginal_count": 0},
            {"enabled": True, "min_anchor_partners": 0},
            {"enabled": True, "max_catalog_rows": 0},
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    NullSentinelConfig.from_dict(case).validate()


if __name__ == "__main__":
    unittest.main()
