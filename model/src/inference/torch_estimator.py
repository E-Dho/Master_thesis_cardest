from __future__ import annotations

from time import perf_counter

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.model.output_adapter import (
    TorchANPMFactorizedOutputAdapter,
    TorchBackboneOutputs,
    TorchIdentityOutputAdapter,
)
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
        self.last_backbone_seconds = 0.0
        self.last_decode_seconds = 0.0
        if metadata.factorization_plan.enabled:
            self.output_adapter = TorchANPMFactorizedOutputAdapter(
                metadata=metadata,
                anpm_decoders=resmade.anpm_decoders,
                anpm_config=resmade.anpm_config,
            )
        else:
            self.output_adapter = TorchIdentityOutputAdapter()

    def predict_distributions(self, tokens: list[PredicateToken]) -> list[np.ndarray]:
        """Run one ResMADE pass and return original-column distributions."""

        import torch

        token_ids = encode_tokens_tensor([tokens], self.predicate_vocabularies, device=self.device)
        self.resmade.eval()
        with torch.no_grad():
            backbone_start = perf_counter()
            logits = self.resmade(token_ids)
            self.last_backbone_seconds = perf_counter() - backbone_start
            outputs = TorchBackboneOutputs(
                logits=logits,
                split_logits=self.resmade.split_head_outputs(logits),
                output_embeddings=(
                    [embedding.weight for embedding in self.resmade.output_embeddings]
                    if getattr(self.resmade.config, "output_encoding", "one_hot") == "embed"
                    else None
                ),
            )
            decode_start = perf_counter()
            distributions = [
                self.output_adapter.original_distribution(
                    original_column_index=column_index,
                    backbone_outputs=outputs,
                )
                if self.metadata.factorization_plan.enabled
                else self.output_adapter.original_distribution(
                    original_column_index=column_index,
                    backbone_outputs=outputs,
                    metadata=self.metadata,
                )
                for column_index in range(len(self.metadata.columns))
            ]
            self.last_decode_seconds = perf_counter() - decode_start
        return [distribution[0].detach().cpu().numpy() for distribution in distributions]

    def predict_column_factors(self, tokens: list[PredicateToken]) -> np.ndarray:
        """Run one ResMADE pass and return original-column scalar factors."""

        import torch

        token_ids = encode_tokens_tensor([tokens], self.predicate_vocabularies, device=self.device)
        self.resmade.eval()
        with torch.no_grad():
            backbone_start = perf_counter()
            logits = self.resmade(token_ids)
            self.last_backbone_seconds = perf_counter() - backbone_start
            outputs = TorchBackboneOutputs(
                logits=logits,
                split_logits=self.resmade.split_head_outputs(logits),
                output_embeddings=(
                    [embedding.weight for embedding in self.resmade.output_embeddings]
                    if getattr(self.resmade.config, "output_encoding", "one_hot") == "embed"
                    else None
                ),
            )
            values = []
            decode_start = perf_counter()
            for column_index, token in enumerate(tokens):
                if self.metadata.factorization_plan.enabled:
                    factor = self.output_adapter.column_factor(
                        original_column_index=column_index,
                        backbone_outputs=outputs,
                        predicate_token=token,
                    )
                else:
                    factor = self.output_adapter.column_factor(
                        original_column_index=column_index,
                        backbone_outputs=outputs,
                        metadata=self.metadata,
                        predicate_token=token,
                    )
                values.append(factor[0].detach().cpu())
            self.last_decode_seconds = perf_counter() - decode_start
        return np.array([float(value) for value in values], dtype=float)
