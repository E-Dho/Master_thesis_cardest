from __future__ import annotations

import math
import unittest

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.output_adapter import IdentityOutputAdapter
from model.src.predicates.encoding import (
    column_factor,
    predicate_mask,
    reciprocal_fanout_mask,
)
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.training.losses import (
    cumulative_inverse_fanout_weights,
    effective_sample_size,
    weighted_cross_entropy,
)


class TokenMaskLossTest(unittest.TestCase):
    def test_logit_slicing_and_softmax_normalization(self) -> None:
        columns = (
            ColumnMetadata("x", ColumnKind.DATA, (0, 1)),
            ColumnMetadata("f", ColumnKind.FANOUT, (1, 10)),
        )
        logits = np.array([[1.0, 2.0, 4.0, 4.0]])
        distributions = IdentityOutputAdapter().distributions_from_logits(logits, columns)
        self.assertEqual(len(distributions), 2)
        self.assertTrue(np.allclose(np.sum(distributions[0], axis=1), 1.0))
        self.assertTrue(np.allclose(distributions[1], [[0.5, 0.5]]))

    def test_ordinary_and_indicator_masks(self) -> None:
        data_column = ColumnMetadata("x", ColumnKind.DATA, (1, 2, 3))
        indicator = ColumnMetadata("I_T", ColumnKind.INDICATOR, (0, 1), table="T")
        self.assertTrue(
            np.array_equal(
                predicate_mask(data_column, PredicateToken(PredicateOp.LESS_EQUAL, 2)),
                np.array([1.0, 1.0, 0.0]),
            )
        )
        self.assertTrue(
            np.array_equal(predicate_mask(indicator, PredicateToken.equal(1)), [0.0, 1.0])
        )

    def test_range_masks_treat_incomparable_sentinels_as_false(self) -> None:
        data_column = ColumnMetadata("x", ColumnKind.DATA, ("__SQL_NULL__", 1990, 2000))
        self.assertTrue(
            np.array_equal(
                predicate_mask(data_column, PredicateToken(PredicateOp.GREATER_EQUAL, 1995)),
                np.array([0.0, 0.0, 1.0]),
            )
        )

    def test_inv_fanout_reciprocal_and_wildcard_factor(self) -> None:
        fanout = ColumnMetadata("F", ColumnKind.FANOUT, (1, 2, 10))
        distribution = np.array([0.2, 0.3, 0.5])
        self.assertTrue(np.allclose(reciprocal_fanout_mask(fanout), [1.0, 0.5, 0.1]))
        self.assertAlmostEqual(column_factor(distribution, fanout, PredicateToken.wildcard()), 1.0)
        self.assertAlmostEqual(
            column_factor(distribution, fanout, PredicateToken.inv_fanout()),
            0.2 + 0.15 + 0.05,
        )

    def test_inverse_expectation_is_not_inverse_of_expectation(self) -> None:
        fanout = ColumnMetadata("F", ColumnKind.FANOUT, (1, 10))
        distribution = np.array([0.5, 0.5])
        expected_inverse = column_factor(distribution, fanout, PredicateToken.inv_fanout())
        inverse_expected = 1.0 / float(np.dot(distribution, np.array([1.0, 10.0])))
        self.assertAlmostEqual(expected_inverse, 0.55)
        self.assertNotAlmostEqual(expected_inverse, inverse_expected)

    def test_cumulative_weights_and_wildcard_exclusion(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("x", ColumnKind.DATA, (0, 1)),
                ColumnMetadata("F1", ColumnKind.FANOUT, (1, 10)),
                ColumnMetadata("F2", ColumnKind.FANOUT, (1, 10)),
                ColumnMetadata("F3", ColumnKind.FANOUT, (1, 10)),
            ),
            full_join_cardinality=2,
        )
        encoded_rows = np.array([[0, 0, 1, 1], [1, 1, 1, 0]])
        tokens = [
            [
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
            [
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
                PredicateToken.wildcard(),
                PredicateToken.inv_fanout(),
            ],
        ]
        weights = cumulative_inverse_fanout_weights(encoded_rows, tokens, metadata)
        self.assertTrue(np.allclose(weights[:, 1], [1.0, 1.0]))
        self.assertTrue(np.allclose(weights[:, 2], [1.0, 0.1]))
        self.assertTrue(np.allclose(weights[:, 3], [1.0, 0.1]))

    def test_weighted_cross_entropy_manual_value(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75]])
        targets = np.array([0, 1])
        weights = np.array([1.0, 3.0])
        expected = (-math.log(0.8) + 3.0 * -math.log(0.75)) / 4.0
        self.assertAlmostEqual(weighted_cross_entropy(probabilities, targets, weights), expected)

    def test_effective_sample_size(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.array([1.0, 1.0, 1.0])), 3.0)


if __name__ == "__main__":
    unittest.main()
