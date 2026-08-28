from __future__ import annotations

import unittest

from model.src.evaluation.metrics import q_error, q_error_floor_one
from model.scripts.evaluate_job_light_queries import RawPredicate, build_normalized_token
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.sets import ColumnPredicateSet
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata


class NativeRangeNormalizationTest(unittest.TestCase):
    def test_open_and_closed_interval_normalization(self) -> None:
        normalized = ColumnPredicateSet(
            (
                PredicateToken(PredicateOp.GREATER_EQUAL, value=10),
                PredicateToken(PredicateOp.LESS_THAN, value=20),
            )
        ).normalize()
        self.assertEqual(normalized.lower, 10)
        self.assertTrue(normalized.lower_inclusive)
        self.assertEqual(normalized.upper, 20)
        self.assertFalse(normalized.upper_inclusive)
        token = normalized.output_token()
        self.assertTrue(token.satisfies(10))
        self.assertTrue(token.satisfies(19))
        self.assertFalse(token.satisfies(20))

    def test_equality_reduces_compatible_bounds(self) -> None:
        normalized = ColumnPredicateSet(
            (
                PredicateToken(PredicateOp.GREATER_EQUAL, value=10),
                PredicateToken.equal(15),
            )
        ).normalize()
        self.assertFalse(normalized.contradiction)
        self.assertEqual(normalized.output_token(), PredicateToken.equal(15))

    def test_equality_contradiction_is_explicit(self) -> None:
        normalized = ColumnPredicateSet(
            (PredicateToken.equal(15), PredicateToken.equal(16))
        ).normalize()
        self.assertTrue(normalized.contradiction)

    def test_empty_open_interval_is_contradiction(self) -> None:
        normalized = ColumnPredicateSet(
            (
                PredicateToken(PredicateOp.GREATER_THAN, value=10),
                PredicateToken(PredicateOp.LESS_THAN, value=10),
            )
        ).normalize()
        self.assertTrue(normalized.contradiction)

    def test_job_light_query20_open_year_range_shape(self) -> None:
        normalized = ColumnPredicateSet(
            (
                PredicateToken(PredicateOp.GREATER_THAN, value=1980),
                PredicateToken(PredicateOp.LESS_THAN, value=1984),
            )
        ).normalize()
        self.assertFalse(normalized.contradiction)
        token = normalized.output_token()
        self.assertEqual(token.op, PredicateOp.RANGE)
        self.assertEqual(token.value, 1980)
        self.assertEqual(token.upper, 1984)
        self.assertFalse(token.lower_inclusive)
        self.assertFalse(token.upper_inclusive)
        self.assertFalse(token.satisfies(1980))
        self.assertTrue(token.satisfies(1981))
        self.assertTrue(token.satisfies(1983))
        self.assertFalse(token.satisfies(1984))

        legacy_upper_estimate = 144842.47
        legacy_lower_estimate = 211711.12
        legacy_subtracted = max(0.0, legacy_upper_estimate - legacy_lower_estimate)
        self.assertEqual(legacy_subtracted, 0.0)
        self.assertGreater(q_error(legacy_subtracted, 695701.0), 1.0e12)

    def test_too_many_predicates_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ColumnPredicateSet(
                (
                    PredicateToken(PredicateOp.GREATER_EQUAL, value=1),
                    PredicateToken(PredicateOp.LESS_EQUAL, value=2),
                    PredicateToken.equal(1),
                )
            ).normalize(max_predicates=2)

    def test_q_error_floor_one_supplements_epsilon_metric(self) -> None:
        self.assertGreater(q_error(0.0, 100.0), q_error_floor_one(0.0, 100.0))
        self.assertEqual(q_error_floor_one(0.0, 100.0), 100.0)

    def test_old_predicate_vocab_keys_still_encode(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("x", ColumnKind.DATA, (1, 2)),),
            full_join_cardinality=2,
        )
        legacy = PredicateVocabularies((('["wildcard", null, null]', '["equal", 1, null]'),))
        self.assertEqual(legacy.encode_token(0, PredicateToken.equal(1)), 1)

    def test_evaluator_uses_normalized_predicate_for_conditioning(self) -> None:
        metadata = ModelMetadata(
            columns=(ColumnMetadata("x", ColumnKind.DATA, (10, 20, 50, 60)),),
            full_join_cardinality=4,
        )
        built = build_normalized_token(
            metadata,
            "x",
            [
                RawPredicate("x", ">=", 10, "x>=10"),
                RawPredicate("x", ">=", 50, "x>=50"),
            ],
        )
        self.assertEqual(built.token, PredicateToken(PredicateOp.GREATER_EQUAL, 50))

        ranged = build_normalized_token(
            metadata,
            "x",
            [
                RawPredicate("x", ">", 10, "x>10"),
                RawPredicate("x", "<=", 50, "x<=50"),
            ],
        )
        self.assertEqual(ranged.token, PredicateToken.range(20, 50))


if __name__ == "__main__":
    unittest.main()
