from __future__ import annotations

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.predicates.operators import PredicateToken
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.vocabulary import PredicateVocabularies


class TorchDistributionModel:
    """Adapter exposing ResMADE through the existing one-pass estimator API."""

    def __init__(
        self,
        resmade: object,
        metadata: ModelMetadata,
        predicate_vocabularies: PredicateVocabularies,
        *,
        device: str = "cpu",
    ) -> None:
        self.resmade = resmade
        self.metadata = metadata
        self.predicate_vocabularies = predicate_vocabularies
        self.device = device

    def predict_distributions(self, tokens: list[PredicateToken]) -> list[np.ndarray]:
        """Run exactly one ResMADE forward pass and return NumPy distributions."""

        import torch

        token_ids = encode_tokens_tensor([tokens], self.predicate_vocabularies, device=self.device)
        self.resmade.eval()
        with torch.no_grad():
            distributions = self.resmade.predict_distributions(token_ids)
        return [distribution[0].detach().cpu().numpy() for distribution in distributions]

