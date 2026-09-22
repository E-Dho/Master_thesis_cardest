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


def encode_contexts_tensor(
    contexts: list[object],
    vocabularies: PredicateVocabularies,
    *,
    device: str | "torch.device" = "cpu",
) -> "torch.Tensor":
    """Encode generated contexts, using prebuilt two-slot rows when present."""

    import numpy as np
    import torch

    if vocabularies.encoding_mode == "two_slot_binary_duet":
        encoded_rows = [getattr(context, "encoded_two_slot_row", None) for context in contexts]
        if encoded_rows and all(row is not None for row in encoded_rows):
            return torch.as_tensor(np.asarray(encoded_rows, dtype=np.int64), device=device)
    token_rows = [list(getattr(context, "tokens")) for context in contexts]
    return encode_tokens_tensor(token_rows, vocabularies, device=device)
