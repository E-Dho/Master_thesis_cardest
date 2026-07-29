from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from model.src.data.schema import ModelMetadata
from model.src.predicates.operators import PredicateToken
from model.src.training.losses import per_head_weighted_cross_entropy


def _token_key(token: PredicateToken) -> tuple[Any, ...]:
    return token.stable_key()


class PredicateConditionedTableModel:
    """Autoregressive conditional tables with Duet-style token ordering.

    This is a correctness-first stand-in for a masked MADE: head i is keyed only
    by virtual tokens T_<i, and each head emits one categorical distribution.
    """

    def __init__(self, metadata: ModelMetadata, *, smoothing: float = 1.0e-3) -> None:
        self.metadata = metadata
        self.smoothing = float(smoothing)
        self.default_distributions = [
            np.full(column.domain_size, 1.0 / column.domain_size, dtype=float)
            for column in metadata.columns
        ]
        self.conditional_distributions: list[dict[tuple[Any, ...], np.ndarray]] = [
            {} for _ in metadata.columns
        ]

    def _context_key(self, tokens: list[PredicateToken], column_index: int) -> tuple[Any, ...]:
        return tuple(_token_key(token) for token in tokens[:column_index])

    def predict_distributions(self, tokens: list[PredicateToken]) -> list[np.ndarray]:
        """Return q_i(X_i|T_<i) for all heads in one predicate-conditioned pass."""

        if len(tokens) != len(self.metadata.columns):
            raise ValueError("token length does not match model metadata")
        distributions = []
        for column_index in range(len(self.metadata.columns)):
            context_key = self._context_key(tokens, column_index)
            distribution = self.conditional_distributions[column_index].get(
                context_key, self.default_distributions[column_index]
            )
            distributions.append(distribution.copy())
        return distributions

    def fit_weighted_counts(
        self,
        encoded_rows: np.ndarray,
        token_rows: list[list[PredicateToken]],
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """Fit empirical weighted conditionals and return pre/post CE sums."""

        encoded_rows = np.asarray(encoded_rows, dtype=int)
        weights = np.asarray(weights, dtype=float)
        pre_loss = float(
            np.sum(
                per_head_weighted_cross_entropy(
                    self.predict_batch(token_rows), encoded_rows, weights
                )
            )
        )
        for column_index, column in enumerate(self.metadata.columns):
            counts: dict[tuple[Any, ...], np.ndarray] = defaultdict(
                lambda: np.full(column.domain_size, self.smoothing, dtype=float)
            )
            default_counts = np.full(column.domain_size, self.smoothing, dtype=float)
            for row_index, tokens in enumerate(token_rows):
                encoded_value = encoded_rows[row_index, column_index]
                row_weight = weights[row_index, column_index]
                counts[self._context_key(tokens, column_index)][encoded_value] += row_weight
                default_counts[encoded_value] += row_weight
            self.conditional_distributions[column_index] = {
                context: values / np.sum(values) for context, values in counts.items()
            }
            self.default_distributions[column_index] = default_counts / np.sum(default_counts)
        post_loss = float(
            np.sum(
                per_head_weighted_cross_entropy(
                    self.predict_batch(token_rows), encoded_rows, weights
                )
            )
        )
        return pre_loss, post_loss

    def predict_batch(self, token_rows: list[list[PredicateToken]]) -> list[np.ndarray]:
        """Return per-head arrays with shape [batch_size, domain_size]."""

        by_head: list[list[np.ndarray]] = [[] for _ in self.metadata.columns]
        for tokens in token_rows:
            distributions = self.predict_distributions(tokens)
            for column_index, distribution in enumerate(distributions):
                by_head[column_index].append(distribution)
        return [np.vstack(distributions) for distributions in by_head]

    def to_json_dict(self) -> dict[str, Any]:
        encoded_conditionals: list[dict[str, list[float]]] = []
        for head in self.conditional_distributions:
            encoded_conditionals.append(
                {json.dumps(context): distribution.tolist() for context, distribution in head.items()}
            )
        return {
            "metadata": self.metadata.to_json_dict(),
            "smoothing": self.smoothing,
            "default_distributions": [
                distribution.tolist() for distribution in self.default_distributions
            ],
            "conditional_distributions": encoded_conditionals,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "PredicateConditionedTableModel":
        metadata = ModelMetadata.from_json_dict(data["metadata"])
        model = cls(metadata, smoothing=float(data["smoothing"]))
        model.default_distributions = [
            np.array(distribution, dtype=float)
            for distribution in data["default_distributions"]
        ]
        conditionals = []
        for raw_head in data["conditional_distributions"]:
            conditionals.append(
                {
                    tuple(tuple(item) for item in json.loads(context)): np.array(
                        distribution, dtype=float
                    )
                    for context, distribution in raw_head.items()
                }
            )
        model.conditional_distributions = conditionals
        return model

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PredicateConditionedTableModel":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

