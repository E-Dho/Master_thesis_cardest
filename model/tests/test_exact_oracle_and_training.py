from __future__ import annotations

import unittest

import numpy as np

from model.src.data.full_join_sampler import build_synthetic_chain_dataset
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.inference.estimator import OnePassEstimator
from model.src.model.predicate_made import PredicateConditionedTableModel
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.operators import PredicateToken
from model.src.training.losses import cumulative_inverse_fanout_weights


class ExactOracleTrainingTest(unittest.TestCase):
    def test_exact_two_fanout_reweighted_marginal(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("F1", ColumnKind.FANOUT, (1, 10)),
                ColumnMetadata("F2", ColumnKind.FANOUT, (1, 10)),
            ),
            full_join_cardinality=2,
        )
        encoded_rows = np.array([[0, 0], [1, 1]])
        oracle = ExactOracle(metadata, encoded_rows)
        exact = oracle.exact_weighted_product_for_fanouts(("F1", "F2"))
        self.assertAlmostEqual(exact, 0.505)

        f1 = oracle.factor_from_oracle_distribution(0, PredicateToken.inv_fanout())
        f2_wrong = oracle.factor_from_oracle_distribution(1, PredicateToken.inv_fanout())
        self.assertAlmostEqual(f1 * f2_wrong, 0.3025)

        f1_values = np.array([1.0, 10.0])
        row_weights_for_f2 = 1.0 / f1_values
        f2_reweighted = oracle.factor_from_oracle_distribution(
            1, PredicateToken.inv_fanout(), row_weights=row_weights_for_f2
        )
        self.assertAlmostEqual(f1 * f2_reweighted, 0.505)

    def test_synthetic_schema_exact_cases(self) -> None:
        dataset = build_synthetic_chain_dataset()
        oracle = ExactOracle(dataset.metadata, dataset.encoded_rows)
        tokens = tokens_for_query_tables(dataset.metadata, {"A", "B"}, {"F_A_to_B"})
        estimate = oracle.independent_factor_estimate(tokens)
        self.assertGreaterEqual(estimate, 0.0)

        indicator_index = dataset.metadata.column_index("I_A")
        indicator_factor = oracle.factor_from_oracle_distribution(
            indicator_index, PredicateToken.equal(1)
        )
        self.assertAlmostEqual(indicator_factor, 4.0 / 5.0)
        self.assertAlmostEqual(
            oracle.exact_weighted_product_for_fanouts(("F_A_to_B", "F_B_to_C")),
            (1 / 20 + 1 / 20 + 1 / 2 + 1 + 1) / 5,
        )

    def test_training_smoke_checkpoint_free_estimation(self) -> None:
        dataset = build_synthetic_chain_dataset()
        token_rows = [
            tokens_for_query_tables(
                dataset.metadata,
                {"A", "B", "C"},
                {"F_A_to_B", "F_B_to_C"},
            )
            for _ in dataset.decoded_rows
        ]
        weights = cumulative_inverse_fanout_weights(
            dataset.encoded_rows, token_rows, dataset.metadata
        )
        self.assertTrue(np.all(weights > 0))
        self.assertTrue(np.all(np.isfinite(weights)))

        model = PredicateConditionedTableModel(dataset.metadata, smoothing=1.0e-6)
        pre_loss, post_loss = model.fit_weighted_counts(
            dataset.encoded_rows, token_rows, weights
        )
        self.assertLess(post_loss, pre_loss)
        distributions = model.predict_distributions(token_rows[0])
        self.assertTrue(all(np.all(np.isfinite(dist)) for dist in distributions))

        estimator = OnePassEstimator(model, dataset.metadata)
        result = estimator.estimate(token_rows[0])
        self.assertGreaterEqual(result.estimated_cardinality, 0.0)
        self.assertTrue(np.isfinite(result.estimated_cardinality))


if __name__ == "__main__":
    unittest.main()

