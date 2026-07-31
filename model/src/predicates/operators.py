from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PredicateOp(str, Enum):
    EQUAL = "equal"
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    RANGE = "range"
    WILDCARD = "wildcard"
    INV_FANOUT = "inv_fanout"


@dataclass(frozen=True)
class PredicateToken:
    """Virtual token consumed by autoregressive predicate-conditioned heads."""

    op: PredicateOp
    value: Any = None
    upper: Any = None

    @classmethod
    def equal(cls, value: Any) -> "PredicateToken":
        return cls(PredicateOp.EQUAL, value=value)

    @classmethod
    def wildcard(cls) -> "PredicateToken":
        return cls(PredicateOp.WILDCARD)

    @classmethod
    def inv_fanout(cls) -> "PredicateToken":
        return cls(PredicateOp.INV_FANOUT)

    @classmethod
    def range(cls, lower: Any, upper: Any) -> "PredicateToken":
        return cls(PredicateOp.RANGE, value=lower, upper=upper)

    def satisfies(self, candidate: Any) -> bool:
        if self.op == PredicateOp.WILDCARD:
            return True
        if self.op == PredicateOp.EQUAL:
            return candidate == self.value
        try:
            if self.op == PredicateOp.LESS_THAN:
                return candidate < self.value
            if self.op == PredicateOp.LESS_EQUAL:
                return candidate <= self.value
            if self.op == PredicateOp.GREATER_THAN:
                return candidate > self.value
            if self.op == PredicateOp.GREATER_EQUAL:
                return candidate >= self.value
            if self.op == PredicateOp.RANGE:
                return self.value <= candidate <= self.upper
        except TypeError:
            return False
        if self.op == PredicateOp.INV_FANOUT:
            raise ValueError("INV_FANOUT is a potential token, not a Boolean predicate")
        raise ValueError(f"unsupported predicate op {self.op!r}")

    def stable_key(self) -> tuple[Any, ...]:
        return (self.op.value, self.value, self.upper)
