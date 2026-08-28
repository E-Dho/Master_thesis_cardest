from __future__ import annotations

from typing import TYPE_CHECKING

from model.src.predicates.operators import PredicateToken
from model.src.predicates.vocabulary import PredicateVocabularies

if TYPE_CHECKING:
    import torch


def encode_tokens_tensor(
    token_rows: list[list[PredicateToken]],
    vocabularies: PredicateVocabularies,
    *,
    device: str | "torch.device" = "cpu",
) -> "torch.Tensor":
    """Encode token rows as categorical IDs or two-slot predicate tensors."""

    import torch

    return torch.tensor(vocabularies.encode_rows(token_rows), dtype=torch.long, device=device)
