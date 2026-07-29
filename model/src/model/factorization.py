from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorizationConfig:
    """Runtime switch for future lossless factorization support."""

    enabled: bool = False
    strategy: str = "none"

    def validate(self) -> None:
        if not self.enabled and self.strategy != "none":
            raise ValueError("factorization.strategy must be 'none' when disabled")
        if self.enabled:
            raise NotImplementedError(
                "factorization.enabled=true requires the future ANPM integration "
                "milestone; it is intentionally disabled here."
            )

