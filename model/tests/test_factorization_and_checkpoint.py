from __future__ import annotations

import tempfile
import unittest

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.factorization import FactorizationConfig
from model.src.model.predicate_made import PredicateConditionedTableModel
from model.src.predicates.operators import PredicateToken


class FactorizationCheckpointTest(unittest.TestCase):
    def test_factorization_defaults_to_disabled(self) -> None:
        config = FactorizationConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.strategy, "none")
        config.validate()

    def test_factorization_enabled_fails_explicitly(self) -> None:
        with self.assertRaises(NotImplementedError):
            FactorizationConfig(enabled=True, strategy="anpm").validate()

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

