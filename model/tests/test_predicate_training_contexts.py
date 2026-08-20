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
    neurocard_table_dropout_rooted_subset,
    predicate_context_diagnostics,
    tokens_for_query_tables,
    token_coverage,
)
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.predicates.vocabulary import binary_bits_lsb, binary_literal_width
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

    def test_binary_literal_encoding_is_lsb_first(self) -> None:
        self.assertEqual(binary_literal_width(1), 1)
        self.assertEqual(binary_literal_width(16), 4)
        self.assertEqual(binary_bits_lsb(13, 4), (1, 0, 1, 1))

    def test_rooted_connected_subsets_include_root(self) -> None:
        source = SyntheticFullJoinSampleSource()
        subsets = connected_table_subsets(
            source.metadata,
            required_root="A",
        )
        self.assertIn(frozenset({"A"}), subsets)
        self.assertIn(frozenset({"A", "B"}), subsets)
        self.assertNotIn(frozenset({"B"}), subsets)
        self.assertNotIn(frozenset({"B", "C"}), subsets)

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

    def test_neurocard_rooted_generation_wildcards_omitted_table_data(self) -> None:
        source = SyntheticFullJoinSampleSource()
        generator = PredicateTrainingContextGenerator(
            {
                "enabled": True,
                "table_subset_sampling": "neurocard_rooted_connected",
                "wildcard_probability": 0.0,
                "equality_probability": 1.0,
                "lower_bound_probability": 0.0,
                "upper_bound_probability": 0.0,
                "native_range_probability": 0.0,
            }
        )
        batch = source.batches(80, seed=13)
        contexts, repeated_rows, _ = generator.generate_batch(
            encoded_rows=batch.encoded_values,
            metadata=source.metadata,
            rng=np.random.default_rng(23),
        )
        self.assertTrue(contexts)
        self.assertTrue(all("A" in context.included_tables for context in contexts))
        self.assertTrue(any(context.included_tables == frozenset({"A"}) for context in contexts))
        self.assertTrue(any(len(context.included_tables) > 1 for context in contexts))
        for context, row in zip(contexts, repeated_rows):
            self.assertTrue(context_satisfies_row(context, row, source.metadata))
            for column, token in zip(source.metadata.columns, context.tokens):
                if column.kind == ColumnKind.DATA and column.table not in context.included_tables:
                    self.assertEqual(token.op, PredicateOp.WILDCARD)
                if column.kind == ColumnKind.FANOUT:
                    child = column.fanout_source.split("->", 1)[1]
                    if child in context.included_tables:
                        self.assertEqual(token.op, PredicateOp.WILDCARD)
                    else:
                        self.assertEqual(token.op, PredicateOp.INV_FANOUT)

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

    def test_duet_style_batches_share_one_batch_context(self) -> None:
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
        self.assertEqual(len({context.tokens for context in contexts}), 1)
        for context, row in zip(contexts, repeated_rows):
            self.assertTrue(context_satisfies_row(context, row, source.metadata))

    def test_duet_batch_equality_requires_identical_batch_values(self) -> None:
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
        self.assertEqual(len({context.tokens for context in contexts}), 1)
        token = contexts[0].ordinary_predicates["B.value"]
        self.assertNotEqual(token.op, PredicateOp.EQUAL)
        for context, row in zip(contexts, repeated_rows):
            self.assertTrue(token.satisfies(source.metadata.columns[1].domain[int(row[1])]))

        repeated_same_row = np.repeat(source.dataset.encoded_rows[[0]], 5, axis=0)
        same_contexts, _, _ = generator.generate_batch(
            encoded_rows=repeated_same_row,
            metadata=source.metadata,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(same_contexts[0].ordinary_predicates["B.value"].op, PredicateOp.EQUAL)
        self.assertEqual(same_contexts[0].ordinary_predicates["B.value"].value, "b1")

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
            self.assertEqual(token.op, PredicateOp.GREATER_EQUAL)
            batch_min = min(source.metadata.columns[1].domain[int(row[1])] for row in rows)
            self.assertLessEqual(token.value, batch_min)

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
            self.assertEqual(token.op, PredicateOp.LESS_EQUAL)
            batch_max = max(source.metadata.columns[1].domain[int(row[1])] for row in rows)
            self.assertGreaterEqual(token.value, batch_max)

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

    def test_batch_outer_padding_fallback_wildcards_column_for_all_rows(self) -> None:
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
        for context, row in zip(contexts, repeated_rows):
            row_value = source.metadata.columns[b_index].domain[int(row[b_index])]
            token = context.tokens[b_index]
            if row_value == OUTER_MISSING:
                self.assertEqual(token.op, PredicateOp.WILDCARD)
            else:
                self.assertEqual(token.op, PredicateOp.WILDCARD)

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

    def test_strict_predicate_probabilities_and_empirical_twenty_percent_mix(self) -> None:
        source = SyntheticFullJoinSampleSource()
        bad_config = {
            "enabled": True,
            "strategy": "duet_batch_bounds",
            "normalize_predicate_probabilities": False,
            "wildcard_probability": 0.2,
            "equality_probability": 0.4,
            "lower_bound_probability": 0.2,
            "upper_bound_probability": 0.2,
            "native_range_probability": 0.2,
        }
        with self.assertRaises(ValueError):
            PredicateTrainingContextGenerator(bad_config)

        config = {
            **bad_config,
            "equality_probability": 0.2,
            "enable_native_range_tokens": True,
            "table_subset_sampling": "full",
        }
        generator = PredicateTrainingContextGenerator(config)
        rng = np.random.default_rng(55)
        rows = np.repeat(source.dataset.encoded_rows[[0]], 4, axis=0)
        contexts = []
        for _ in range(3000):
            batch_contexts, _, _ = generator.generate_batch(
                encoded_rows=rows,
                metadata=source.metadata,
                rng=rng,
            )
            contexts.append(batch_contexts[0])
        diagnostics = predicate_context_diagnostics(contexts, source.metadata)
        frequencies = diagnostics["empirical_predicate_choice_frequencies"]
        for key in ("wildcard", "equality", "lower", "upper", "two_sided_range"):
            self.assertAlmostEqual(frequencies[key], 0.2, delta=0.04)

    def test_neurocard_table_dropout_rooted_matches_expected_chain_law(self) -> None:
        source = SyntheticFullJoinSampleSource()
        rng = np.random.default_rng(42)
        counts: dict[frozenset[str], int] = {}
        present = frozenset({"A", "B", "C"})
        samples = 20000
        for _ in range(samples):
            subset = neurocard_table_dropout_rooted_subset(source.metadata, present, rng)
            counts[subset] = counts.get(subset, 0) + 1
            self.assertIn("A", subset)
        self.assertAlmostEqual(counts.get(frozenset({"A"}), 0) / samples, 0.5, delta=0.04)
        self.assertAlmostEqual(counts.get(frozenset({"A", "B"}), 0) / samples, 2 / 9, delta=0.04)
        self.assertAlmostEqual(counts.get(frozenset({"A", "B", "C"}), 0) / samples, 5 / 18, delta=0.04)

    def test_binary_predicate_vocab_is_compact_and_uses_exact_value_lookup(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("big", ColumnKind.DATA, tuple(range(100_000))),
                ColumnMetadata("small", ColumnKind.DATA, ("x", "y")),
            ),
            full_join_cardinality=100_000,
        )
        vocab = PredicateVocabularies.from_metadata(
            metadata,
            encoding_mode="two_slot_binary_duet",
        )
        self.assertEqual(vocab.structural_entry_count(), len(metadata.columns))
        self.assertEqual(vocab.input_bins, (1, 1))
        encoded = vocab.encode_token_two_slot(0, PredicateToken.equal(99_999))
        self.assertEqual(encoded[1], 99_999)
        range_encoded = vocab.encode_token_two_slot(0, PredicateToken.range(123, 45_678))
        self.assertEqual(range_encoded[1], 123)
        self.assertEqual(range_encoded[3], 45_678)
        payload = vocab.to_json_dict()
        self.assertNotIn("domains_by_column", payload)
        roundtrip = PredicateVocabularies.from_json_dict(payload, metadata)
        self.assertEqual(
            roundtrip.encode_token_two_slot(0, PredicateToken.equal(98_765))[1],
            98_765,
        )
        diagnostics = vocab.metadata_size_diagnostics()
        self.assertEqual(diagnostics["structural_predicate_entries"], 2)
        self.assertGreater(diagnostics["compression_ratio"], 100.0)


if __name__ == "__main__":
    unittest.main()
