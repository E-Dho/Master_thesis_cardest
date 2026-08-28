from __future__ import annotations

import tempfile
import unittest

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
    decode_factors,
    factorize_rows,
    factorize_value,
)
from model.src.model.predicate_made import PredicateConditionedTableModel
from model.src.predicates.operators import PredicateToken


class FactorizationCheckpointTest(unittest.TestCase):
    def test_factorization_defaults_to_disabled(self) -> None:
        config = FactorizationConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.strategy, "none")
        config.validate()

    def test_factorization_requires_lossless_strategy(self) -> None:
        with self.assertRaises(ValueError):
            FactorizationConfig(enabled=True, strategy="anpm").validate()

    def test_lossless_bit_factorization_round_trips_valid_ids(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("x", ColumnKind.DATA, tuple(range(20))),),
            full_join_cardinality=20,
        )
        metadata = apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=3,
                minimum_domain_size=2,
            ),
        )
        plan = metadata.factorization_plan.factorization_for_column(0)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.factor_domains, (4, 8))
        self.assertEqual(factorize_value(0, plan), (0, 0))
        for value in range(20):
            self.assertEqual(decode_factors(factorize_value(value, plan), plan), value)
        with self.assertRaises(ValueError):
            decode_factors((3, 7), plan)

    def test_factorize_rows_keeps_original_rows_external(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("x", ColumnKind.DATA, tuple(range(20))),
                ColumnMetadata("I_T", ColumnKind.INDICATOR, (0, 1), table="T"),
            ),
            full_join_cardinality=2,
        )
        metadata = apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=3,
                minimum_domain_size=2,
            ),
        )
        rows = np.array([[0, 1], [19, 0]])
        factor_rows = factorize_rows(rows, metadata)
        self.assertEqual(factor_rows.tolist(), [[0, 0, 1], [2, 3, 0]])
        self.assertEqual(rows.tolist(), [[0, 1], [19, 0]])

    def test_factorization_reduces_large_domain_output_width(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("large", ColumnKind.DATA, tuple(range(5000))),),
            full_join_cardinality=5000,
        )
        metadata = apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=8,
                minimum_domain_size=2,
            ),
        )
        plan = metadata.factorization_plan
        self.assertLess(plan.factorized_output_width, plan.original_output_width)
        self.assertEqual(metadata.model_output_bins, (32, 256))

    def test_checkpoint_preserves_order_and_domains(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("x", ColumnKind.DATA, ("a", "b")),
                ColumnMetadata("I_T", ColumnKind.INDICATOR, (0, 1), table="T"),
                ColumnMetadata("F", ColumnKind.FANOUT, (1, 5), table="T"),
            ),
            full_join_cardinality=10,
        )
        model = PredicateConditionedTableModel(metadata)
        rows = np.array([[0, 1, 0], [1, 1, 1]])
        tokens = [[PredicateToken.wildcard()] * 3, [PredicateToken.wildcard()] * 3]
        weights = np.ones_like(rows, dtype=float)
        model.fit_weighted_counts(rows, tokens, weights)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/checkpoint.json"
            model.save(path)
            loaded = PredicateConditionedTableModel.load(path)
        self.assertEqual([column.name for column in loaded.metadata.columns], ["x", "I_T", "F"])
        self.assertEqual(loaded.metadata.columns[2].domain, (1, 5))
        loaded_distribution = loaded.predict_distributions([PredicateToken.wildcard()] * 3)[2]
        self.assertTrue(np.isclose(np.sum(loaded_distribution), 1.0))


if __name__ == "__main__":
    unittest.main()
