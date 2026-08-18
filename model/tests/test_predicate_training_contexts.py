from __future__ import annotations

import unittest

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.src.data.full_join_sampler import OUTER_MISSING
from model.scripts.inspect_predicate_coverage import unseen_required_token_types
from model.src.predicates.generation import (
    PredicateTrainingContextGenerator,
    connected_table_subsets,
    context_satisfies_row,
    inverse_fanouts_for_table_subset,
    tokens_for_query_tables,
    token_coverage,
)
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.vocabulary import PredicateVocabularies
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

    def test_duet_style_batches_contain_distinct_row_contexts(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "full",
                "per_row_contexts": 1,
            }
        )
        contexts, repeated_rows, stats = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(7),
        )

        self.assertEqual(stats.generated_contexts, len(rows))
        self.assertEqual(repeated_rows.shape[0], len(rows))
        self.assertGreater(len({context.tokens for context in contexts}), 1)
        for context, row in zip(contexts, repeated_rows):
            self.assertTrue(context_satisfies_row(context, row, source.metadata))

    def test_duet_equality_uses_individual_row_value(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows[:3]
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "full",
            }
        )
        contexts, repeated_rows, _ = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(2),
        )
        for context, row in zip(contexts, repeated_rows):
            token = context.ordinary_predicates["B.value"]
            self.assertEqual(token.op, PredicateOp.EQUAL)
            self.assertEqual(token.value, source.metadata.columns[1].domain[int(row[1])])
        self.assertGreater(
            len({context.ordinary_predicates["B.value"].value for context in contexts}),
            1,
        )

    def test_duet_row_lower_bounds_are_no_greater_than_row_value(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows[:3]
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 0.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 1.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "full",
            }
        )
        contexts, repeated_rows, _ = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(3),
        )
        for context, row in zip(contexts, repeated_rows):
            token = context.ordinary_predicates["B.value"]
            row_value = source.metadata.columns[1].domain[int(row[1])]
            self.assertEqual(token.op, PredicateOp.GREATER_EQUAL)
            self.assertLessEqual(token.value, row_value)

    def test_duet_row_upper_bounds_are_no_less_than_row_value(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows[:3]
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 0.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 1.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "full",
            }
        )
        contexts, repeated_rows, _ = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(4),
        )
        for context, row in zip(contexts, repeated_rows):
            token = context.ordinary_predicates["B.value"]
            row_value = source.metadata.columns[1].domain[int(row[1])]
            self.assertEqual(token.op, PredicateOp.LESS_EQUAL)
            self.assertGreaterEqual(token.value, row_value)

    def test_duet_batch_native_range_toggle_controls_range_tokens(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows[:3]
        base_config = {
            "enabled": True,
            "strategy": "duet_batch_bounds",
            "wildcard_probability": 0.0,
            "equality_probability": 0.0,
            "lower_bound_probability": 0.0,
            "upper_bound_probability": 0.0,
            "native_range_probability": 1.0,
            "table_subset_sampling": "full",
        }
        off_contexts, _, _ = PredicateTrainingContextGenerator(
            {**base_config, "enable_native_range_tokens": False}
        ).generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(5),
        )
        self.assertNotIn(
            PredicateOp.RANGE,
            {token.op for token in off_contexts[0].ordinary_predicates.values()},
        )

        on_contexts, _, _ = PredicateTrainingContextGenerator(
            {**base_config, "enable_native_range_tokens": True}
        ).generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(5),
        )
        self.assertIn(
            PredicateOp.RANGE,
            {token.op for token in on_contexts[0].ordinary_predicates.values()},
        )
        coverage = token_coverage([on_contexts[0].tokens], source.metadata)
        self.assertGreater(
            sum(column_counts[PredicateOp.RANGE.value] for column_counts in coverage.values()),
            0,
        )
        for context, row in zip(on_contexts, rows):
            token = context.ordinary_predicates["B.value"]
            row_value = source.metadata.columns[1].domain[int(row[1])]
            self.assertTrue(token.satisfies(row_value))

    def test_duet_batch_context_has_positive_fanout_ess_and_no_indicator_contradictions(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows[:3]
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 1.0,
                "equality_probability": 0.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "connected",
            }
        )
        contexts, repeated_rows, stats = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(8),
        )
        self.assertEqual(stats.included_indicator_contradictions, 0)
        tokens = [list(context.tokens) for context in contexts]
        weights = cumulative_inverse_fanout_weights(repeated_rows, tokens, source.metadata)
        for fanout_index in source.metadata.fanout_indices():
            self.assertGreater(float(weights[:, fanout_index].sum()), 0.0)

    def test_outer_padding_fallback_is_row_local(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rows = source.dataset.encoded_rows
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "strategy": "duet_batch_bounds",
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
                "table_subset_sampling": "full",
            }
        )
        contexts, repeated_rows, _ = generator.generate_batch(
            encoded_rows=rows,
            metadata=source.metadata,
            rng=np.random.default_rng(9),
        )
        b_index = source.metadata.column_index("B.value")
        padded_rows = 0
        normal_predicates = 0
        for context, row in zip(contexts, repeated_rows):
            row_value = source.metadata.columns[b_index].domain[int(row[b_index])]
            token = context.tokens[b_index]
            if row_value == OUTER_MISSING:
                padded_rows += 1
                self.assertEqual(token.op, PredicateOp.WILDCARD)
            else:
                normal_predicates += int(token.op == PredicateOp.EQUAL)
        self.assertGreater(padded_rows, 0)
        self.assertGreater(normal_predicates, 0)

    def test_job_light_like_metadata_uses_explicit_join_topology(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("title:kind_id", ColumnKind.DATA, (1, 2), table="title"),
                ColumnMetadata("movie_info:info_type_id", ColumnKind.DATA, (1, 2), table="movie_info"),
                ColumnMetadata("movie_keyword:keyword_id", ColumnKind.DATA, (1, 2), table="movie_keyword"),
                ColumnMetadata("__in_title", ColumnKind.INDICATOR, (0, 1), table="title"),
                ColumnMetadata("__in_movie_info", ColumnKind.INDICATOR, (0, 1), table="movie_info"),
                ColumnMetadata("__in_movie_keyword", ColumnKind.INDICATOR, (0, 1), table="movie_keyword"),
                ColumnMetadata("__fanout_movie_info", ColumnKind.FANOUT, (1, 2), table="movie_info", fanout_source="movie_info:movie_id"),
                ColumnMetadata("__fanout_movie_keyword", ColumnKind.FANOUT, (1, 2), table="movie_keyword", fanout_source="movie_keyword:movie_id"),
            ),
            full_join_cardinality=10,
            join_root="title",
            join_tables=("title", "movie_info", "movie_keyword"),
            join_edges=(("title", "movie_info"), ("title", "movie_keyword")),
        )
        subsets = connected_table_subsets(metadata)
        self.assertIn(frozenset({"title", "movie_info"}), subsets)
        self.assertIn(frozenset({"title", "movie_keyword"}), subsets)
        self.assertNotIn(frozenset({"movie_info", "movie_keyword"}), subsets)

    def test_job_light_legacy_manifest_uses_title_star_fallback(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("title:kind_id", ColumnKind.DATA, (1, 2), table="title"),
                ColumnMetadata("movie_info:info_type_id", ColumnKind.DATA, (1, 2), table="movie_info"),
                ColumnMetadata("movie_keyword:keyword_id", ColumnKind.DATA, (1, 2), table="movie_keyword"),
                ColumnMetadata("__in_title", ColumnKind.INDICATOR, (0, 1), table="title"),
                ColumnMetadata("__in_movie_info", ColumnKind.INDICATOR, (0, 1), table="movie_info"),
                ColumnMetadata("__in_movie_keyword", ColumnKind.INDICATOR, (0, 1), table="movie_keyword"),
                ColumnMetadata("__fanout_movie_info", ColumnKind.FANOUT, (1, 2), table="movie_info", fanout_source="movie_info:movie_id"),
                ColumnMetadata("__fanout_movie_keyword", ColumnKind.FANOUT, (1, 2), table="movie_keyword", fanout_source="movie_keyword:movie_id"),
            ),
            full_join_cardinality=10,
        )
        subsets = connected_table_subsets(metadata)
        self.assertIn(frozenset({"title", "movie_info"}), subsets)
        self.assertIn(frozenset({"title", "movie_keyword"}), subsets)
        self.assertIn(frozenset({"title", "movie_info", "movie_keyword"}), subsets)
        self.assertNotIn(frozenset({"movie_info", "movie_keyword"}), subsets)

    def test_two_slot_range_encoding_does_not_allocate_interval_vocab(self) -> None:
        source = SyntheticFullJoinSampleSource()
        vocab = PredicateVocabularies.from_metadata(
            source.metadata,
            encoding_mode="two_slot",
        )
        self.assertEqual(vocab.input_bins[1], 1)
        encoded = vocab.encode_token_two_slot(1, PredicateToken.range("b1", "b2"))
        self.assertEqual(len(encoded), 4)


if __name__ == "__main__":
    unittest.main()
