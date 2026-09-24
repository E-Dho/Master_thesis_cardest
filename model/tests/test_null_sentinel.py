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
    PredicateTrainingContextGenerator,
    context_satisfies_row,
    void_target_row,
)
from model.src.predicates.operators import PredicateOp, PredicateToken


def _metadata() -> ModelMetadata:
    """Two small categorical DATA columns plus an indicator and a fanout."""

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
        self.assertEqual(extended.columns[2].domain, (0, 1))
        self.assertEqual(extended.columns[3].domain, (1, 2))
        self.assertTrue(column_has_null_sentinel(extended.columns[0]))
        self.assertFalse(column_has_null_sentinel(extended.columns[2]))
        self.assertIsNone(null_sentinel_index(extended.columns[3]))

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
            sum(extended.data_output_bins) - sum(original.data_output_bins), 2
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
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
            dtype=int,
        )

    def test_eligible_columns_exclude_indicators_and_fanouts(self) -> None:
        self.assertEqual(
            eligible_catalog_columns(self.metadata, _enabled_config()), (0, 1)
        )

    def test_large_domains_are_excluded_from_the_catalog(self) -> None:
        config = _enabled_config(max_catalog_domain_size=3)
        # colour has four real values, size has three.
        self.assertEqual(eligible_catalog_columns(self.metadata, config), (1,))

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
        self.assertIsNotNone(catalog)
        # colour=2 co-occurs with exactly one size, so voiding one of its
        # partners would teach "colour=2 implies void" rather than a pair fact.
        self.assertEqual(catalog.present_partner_count(0, 1, 1), 1)
        pair = catalog.sample_void_pair(
            self.rows[3], np.random.default_rng(0), max_attempts=64
        )
        self.assertIsNone(pair)

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
            [[colour, size, 1, 0] for colour in range(4) for size in range(3)],
            dtype=int,
        )
        self.assertIsNone(
            build_void_pair_catalog(dense, self.metadata, _enabled_config())
        )


class VoidTargetRowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = apply_null_sentinel_to_metadata(_metadata(), _enabled_config())
        self.row = np.array([2, 1, 1, 0], dtype=int)

    def test_sentinel_is_written_at_the_void_column(self) -> None:
        targets = void_target_row(self.row, self.metadata, 1, cascade=False)
        self.assertEqual(targets[0], 2)
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))

    def test_cascade_propagates_to_later_data_columns_only(self) -> None:
        targets = void_target_row(self.row, self.metadata, 0, cascade=True)
        self.assertEqual(targets[0], null_sentinel_index(self.metadata.columns[0]))
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))
        # Indicator and fanout heads keep their real values: table presence and
        # the inverse fanout weights have no meaningful void state.
        self.assertEqual(targets[2], 1)
        self.assertEqual(targets[3], 0)

    def test_cascade_skips_columns_the_context_does_not_constrain(self) -> None:
        # Column 1 carries no predicate here, so supervising it toward the
        # sentinel would buy nothing at inference (a wildcard factor is 1.0)
        # while pulling its marginal away from the data.
        targets = void_target_row(
            self.row, self.metadata, 0, cascade=True, predicated_columns={0}
        )
        self.assertEqual(targets[0], null_sentinel_index(self.metadata.columns[0]))
        self.assertEqual(targets[1], 1)

    def test_cascade_reaches_constrained_columns(self) -> None:
        targets = void_target_row(
            self.row, self.metadata, 0, cascade=True, predicated_columns={0, 1}
        )
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))

    def test_void_column_is_always_written_even_if_not_listed(self) -> None:
        targets = void_target_row(
            self.row, self.metadata, 1, cascade=True, predicated_columns=set()
        )
        self.assertEqual(targets[1], null_sentinel_index(self.metadata.columns[1]))

    def test_source_row_is_not_mutated(self) -> None:
        void_target_row(self.row, self.metadata, 0, cascade=True)
        self.assertEqual(list(self.row), [2, 1, 1, 0])

    def test_column_without_sentinel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            void_target_row(self.row, self.metadata, 3, cascade=False)


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
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]],
            dtype=int,
        )
        self.catalog = build_void_pair_catalog(self.rows, self.metadata, self.config)
        self.generator = PredicateTrainingContextGenerator(
            {"enabled": True, "wildcard_probability": 0.3, "seed": 0}
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
        for row in self.rows:
            base = self.generator._generate_one(row, self.metadata, rng)
            context = self.generator._generate_void_context(
                row, self.metadata, rng, self.generator._compiled(self.metadata), base
            )
            if context is None:
                continue
            index = context.void_column_index
            row_value = self.metadata.columns[index].domain[int(row[index])]
            self.assertFalse(context.tokens[index].satisfies(row_value))

    def test_row_satisfaction_validation_exempts_void_contexts(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        rng = np.random.default_rng(1)
        compiled = self.generator._compiled(self.metadata)
        # colour=1 never co-occurs with size=30, so this row can host a void.
        row = self.rows[2]
        contexts = []
        for _ in range(20):
            base = self.generator._generate_one(row, self.metadata, rng)
            context = self.generator._generate_void_context(
                row, self.metadata, rng, compiled, base
            )
            if context is not None:
                contexts.append(context)
        self.assertTrue(contexts, "no void context could be built for this row")
        for context in contexts:
            self.assertTrue(context_satisfies_row(context, row, self.metadata))

    def test_cascade_never_targets_an_unpredicated_column(self) -> None:
        self.generator.attach_void_catalog(self.catalog, self.config)
        contexts, targets, _ = self.generator.generate_batch(
            encoded_rows=self.rows,
            metadata=self.metadata,
            rng=np.random.default_rng(3),
        )
        checked = 0
        for context, target, row in zip(contexts, targets, self.rows):
            if context.void_column_index is None:
                continue
            checked += 1
            for index, column in enumerate(self.metadata.columns):
                sentinel = null_sentinel_index(column)
                if sentinel is None or index == context.void_column_index:
                    continue
                if target[index] == sentinel:
                    self.assertNotEqual(
                        context.tokens[index].op,
                        PredicateOp.WILDCARD,
                        f"column {index} got a sentinel target while unpredicated",
                    )
        self.assertTrue(checked, "no void context was produced")

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
