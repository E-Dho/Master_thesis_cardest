from __future__ import annotations

import unittest

import numpy as np

from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.scripts.inspect_predicate_coverage import unseen_required_token_types
from model.src.predicates.generation import (
    PredicateTrainingContextGenerator,
    connected_table_subsets,
    context_satisfies_row,
    inverse_fanouts_for_table_subset,
    tokens_for_query_tables,
    token_coverage,
)
from model.src.predicates.operators import PredicateOp
from model.src.training.losses import cumulative_inverse_fanout_weights


class PredicateTrainingContextTest(unittest.TestCase):
    def _generator(self) -> PredicateTrainingContextGenerator:
        return PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "wildcard_probability": 0.1,
                "equality_probability": 0.3,
                "lower_bound_probability": 0.3,
                "upper_bound_probability": 0.3,
                "table_subset_sampling": "connected",
                "per_row_contexts": 2,
                "seed": 0,
            }
        )

    def test_generated_contexts_satisfy_sampled_rows(self) -> None:
        source = SyntheticFullJoinSampleSource()
        batch = source.batches(12, seed=11)
        contexts, repeated_rows, stats = self._generator().generate_batch(
            encoded_rows=batch.encoded_values,
            metadata=source.metadata,
            rng=np.random.default_rng(123),
        )
        self.assertEqual(stats.generated_contexts, 24)
        self.assertEqual(repeated_rows.shape[0], 24)
        for context, row in zip(contexts, repeated_rows):
            self.assertTrue(context_satisfies_row(context, row, source.metadata))
            for table in context.included_tables:
                indicator_index = source.metadata.column_index(f"I_{table}")
                indicator_column = source.metadata.columns[indicator_index]
                self.assertEqual(indicator_column.domain[row[indicator_index]], 1)

    def test_connected_subsets_respect_join_tree(self) -> None:
        source = SyntheticFullJoinSampleSource()
        subsets = connected_table_subsets(source.metadata)
        self.assertIn(frozenset({"A", "B"}), subsets)
        self.assertIn(frozenset({"B", "C"}), subsets)
        self.assertNotIn(frozenset({"A", "C"}), subsets)

    def test_fanout_tokens_follow_child_table_semantics(self) -> None:
        source = SyntheticFullJoinSampleSource()
        self.assertEqual(
            inverse_fanouts_for_table_subset(source.metadata, frozenset({"A", "B", "C"})),
            frozenset(),
        )
        self.assertEqual(
            inverse_fanouts_for_table_subset(source.metadata, frozenset({"A"})),
            frozenset({"F_A_to_B", "F_B_to_C"}),
        )
        self.assertEqual(
            inverse_fanouts_for_table_subset(source.metadata, frozenset({"A", "B"})),
            frozenset({"F_B_to_C"}),
        )

    def test_current_fanout_does_not_weight_its_own_head(self) -> None:
        source = SyntheticFullJoinSampleSource()
        row = source.dataset.encoded_rows[[0]]
        inverse_fanouts = inverse_fanouts_for_table_subset(
            source.metadata,
            frozenset({"A"}),
        )
        tokens = [tokens_for_query_tables(source.metadata, {"A"}, set(inverse_fanouts))]
        weights = cumulative_inverse_fanout_weights(row, tokens, source.metadata)
        first_fanout = source.metadata.column_index("F_A_to_B")
        second_fanout = source.metadata.column_index("F_B_to_C")
        self.assertAlmostEqual(weights[0, first_fanout], 1.0)
        self.assertAlmostEqual(weights[0, second_fanout], 0.5)

    def test_token_coverage_records_non_wildcard_operator_types(self) -> None:
        source = SyntheticFullJoinSampleSource()
        batch = source.batches(200, seed=21)
        contexts, _, _ = self._generator().generate_batch(
            encoded_rows=batch.encoded_values,
            metadata=source.metadata,
            rng=np.random.default_rng(4),
        )
        coverage = token_coverage([context.tokens for context in contexts], source.metadata)
        data_counts = coverage["A.value"]
        self.assertGreater(data_counts[PredicateOp.EQUAL.value], 0)
        self.assertGreater(data_counts[PredicateOp.LESS_EQUAL.value], 0)
        self.assertGreater(data_counts[PredicateOp.GREATER_EQUAL.value], 0)
        self.assertGreater(coverage["I_A"]["indicator_wildcard"], 0)
        self.assertGreater(coverage["F_A_to_B"]["fanout_inv"], 0)

    def test_unseen_required_token_types_are_reported(self) -> None:
        training = {"x": {"equal": 3, "less_equal": 0}}
        requirements = {"x": {"equal": 1, "less_equal": 2, "wildcard": 0}}
        self.assertEqual(
            unseen_required_token_types(training, requirements),
            ["x:less_equal"],
        )


if __name__ == "__main__":
    unittest.main()
