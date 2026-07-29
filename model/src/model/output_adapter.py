from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from model.src.data.schema import ColumnMetadata
from model.src.predicates.encoding import softmax


class OutputDistributionAdapter(Protocol):
    """Boundary for converting raw logits into original-column distributions."""

    def distributions_from_logits(
        self, logits: np.ndarray, columns: tuple[ColumnMetadata, ...]
    ) -> list[np.ndarray]:
        ...


class IdentityOutputAdapter:
    """Unfactorized adapter: each output slice is already one original column."""

    def distributions_from_logits(
        self, logits: np.ndarray, columns: tuple[ColumnMetadata, ...]
    ) -> list[np.ndarray]:
        distributions: list[np.ndarray] = []
        start = 0
        for column in columns:
            stop = start + column.domain_size
            distributions.append(softmax(logits[..., start:stop]))
            start = stop
        if start != logits.shape[-1]:
            raise ValueError(
                f"logit width {logits.shape[-1]} does not match column domains {start}"
            )
        return distributions


class TorchIdentityOutputAdapter:
    """Torch unfactorized adapter with one softmax per output slice."""

    def distributions_from_logits(
        self, logits: Any, columns: tuple[ColumnMetadata, ...]
    ) -> list[Any]:
        import torch

        distributions: list[Any] = []
        start = 0
        for column in columns:
            stop = start + column.domain_size
            distributions.append(torch.softmax(logits[:, start:stop], dim=1))
            start = stop
        if start != logits.shape[-1]:
            raise ValueError(
                f"logit width {logits.shape[-1]} does not match column domains {start}"
            )
        return distributions


class ANPMFactorizedOutputAdapter:
    """Placeholder for DistJoin-style ANPM reconstruction in the next milestone."""

    def distributions_from_logits(
        self, logits: np.ndarray, columns: tuple[ColumnMetadata, ...]
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "ANPM factorized output reconstruction is not implemented in this "
            "milestone. Keep factorization.enabled=false."
        )
