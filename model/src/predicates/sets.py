from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.src.predicates.operators import PredicateOp, PredicateToken


@dataclass(frozen=True)
class NormalizedColumnPredicate:
    """Canonical conjunctive predicate over one original database column."""

    equality: Any | None = None
    lower: Any | None = None
    lower_inclusive: bool = True
    upper: Any | None = None
    upper_inclusive: bool = True
    contradiction: bool = False

    def output_token(self) -> PredicateToken:
        if self.contradiction:
            raise ValueError("contradictory predicates have no output token")
        if self.equality is not None:
            return PredicateToken.equal(self.equality)
        if self.lower is not None and self.upper is not None:
            return PredicateToken(
                PredicateOp.RANGE,
                value=self.lower,
                upper=self.upper,
                lower_inclusive=self.lower_inclusive,
                upper_inclusive=self.upper_inclusive,
            )
        if self.lower is not None:
            return PredicateToken(
                PredicateOp.GREATER_EQUAL if self.lower_inclusive else PredicateOp.GREATER_THAN,
                value=self.lower,
            )
        if self.upper is not None:
            return PredicateToken(
                PredicateOp.LESS_EQUAL if self.upper_inclusive else PredicateOp.LESS_THAN,
                value=self.upper,
            )
        return PredicateToken.wildcard()

    @property
    def is_native_range(self) -> bool:
        return self.lower is not None and self.upper is not None and self.equality is None


@dataclass(frozen=True)
class ColumnPredicateSet:
    """Conjunctive predicates applied to one original database column."""

    predicates: tuple[PredicateToken, ...] = ()

    def validate(self, *, max_predicates: int = 2) -> None:
        if len(self.predicates) > max_predicates:
            raise ValueError(
                f"column has {len(self.predicates)} predicates, max supported is {max_predicates}"
            )
        for token in self.predicates:
            if token.op in {PredicateOp.WILDCARD, PredicateOp.INV_FANOUT}:
                raise ValueError(f"{token.op.value} is not an ordinary column predicate")

    def normalize(self, *, max_predicates: int = 2) -> NormalizedColumnPredicate:
        """Normalize equality/bounds and detect contradictions explicitly."""

        self.validate(max_predicates=max_predicates)
        equality = None
        lower = None
        lower_inclusive = True
        upper = None
        upper_inclusive = True
        contradiction = False
        for token in self.predicates:
            if token.op == PredicateOp.EQUAL:
                if equality is not None and equality != token.value:
                    contradiction = True
                equality = token.value
            elif token.op in {PredicateOp.GREATER_THAN, PredicateOp.GREATER_EQUAL}:
                inclusive = token.op == PredicateOp.GREATER_EQUAL
                lower, lower_inclusive = _select_stronger_lower(
                    lower,
                    lower_inclusive,
                    token.value,
                    inclusive,
                )
            elif token.op in {PredicateOp.LESS_THAN, PredicateOp.LESS_EQUAL}:
                inclusive = token.op == PredicateOp.LESS_EQUAL
                upper, upper_inclusive = _select_stronger_upper(
                    upper,
                    upper_inclusive,
                    token.value,
                    inclusive,
                )
            elif token.op == PredicateOp.RANGE:
                lower, lower_inclusive = _select_stronger_lower(
                    lower,
                    lower_inclusive,
                    token.value,
                    token.lower_inclusive,
                )
                upper, upper_inclusive = _select_stronger_upper(
                    upper,
                    upper_inclusive,
                    token.upper,
                    token.upper_inclusive,
                )
            else:
                raise ValueError(f"unsupported predicate op {token.op!r}")
        if equality is not None:
            if lower is not None and not _lower_satisfied(equality, lower, lower_inclusive):
                contradiction = True
            if upper is not None and not _upper_satisfied(equality, upper, upper_inclusive):
                contradiction = True
            return NormalizedColumnPredicate(
                equality=equality,
                contradiction=contradiction,
            )
        if lower is not None and upper is not None:
            try:
                if lower > upper or (lower == upper and not (lower_inclusive and upper_inclusive)):
                    contradiction = True
            except TypeError:
                contradiction = True
        return NormalizedColumnPredicate(
            lower=lower,
            lower_inclusive=lower_inclusive,
            upper=upper,
            upper_inclusive=upper_inclusive,
            contradiction=contradiction,
        )


def canonical_predicate_tokens(
    normalized: NormalizedColumnPredicate,
) -> tuple[PredicateToken, ...]:
    """Return deterministic input-token order for a normalized predicate."""

    if normalized.contradiction:
        return ()
    if normalized.equality is not None:
        return (PredicateToken.equal(normalized.equality),)
    tokens = []
    if normalized.lower is not None:
        tokens.append(
            PredicateToken(
                PredicateOp.GREATER_EQUAL if normalized.lower_inclusive else PredicateOp.GREATER_THAN,
                value=normalized.lower,
            )
        )
    if normalized.upper is not None:
        tokens.append(
            PredicateToken(
                PredicateOp.LESS_EQUAL if normalized.upper_inclusive else PredicateOp.LESS_THAN,
                value=normalized.upper,
            )
        )
    return tuple(tokens)


def _select_stronger_lower(
    current: Any | None,
    current_inclusive: bool,
    candidate: Any,
    candidate_inclusive: bool,
) -> tuple[Any, bool]:
    if current is None:
        return candidate, candidate_inclusive
    if candidate > current:
        return candidate, candidate_inclusive
    if candidate == current:
        return current, current_inclusive and candidate_inclusive
    return current, current_inclusive


def _select_stronger_upper(
    current: Any | None,
    current_inclusive: bool,
    candidate: Any,
    candidate_inclusive: bool,
) -> tuple[Any, bool]:
    if current is None:
        return candidate, candidate_inclusive
    if candidate < current:
        return candidate, candidate_inclusive
    if candidate == current:
        return current, current_inclusive and candidate_inclusive
    return current, current_inclusive


def _lower_satisfied(value: Any, lower: Any, inclusive: bool) -> bool:
    return value >= lower if inclusive else value > lower


def _upper_satisfied(value: Any, upper: Any, inclusive: bool) -> bool:
    return value <= upper if inclusive else value < upper
