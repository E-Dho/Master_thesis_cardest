from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from model.src.model.masked_layers import MaskedLinear, MaskedResidualBlock, mask_from_degrees

import torch
from torch import nn


@dataclass(frozen=True)
class PredicateResMADEConfig:
    """Configuration for grouped predicate-input ResMADE."""

    predicate_input_bins: tuple[int, ...]
    data_output_bins: tuple[int, ...]
    hidden_sizes: tuple[int, ...] = (512, 512, 512, 512)
    residual_connections: bool = True
    direct_io_connections: bool = True
    activation: str = "relu"
    input_encoding: str = "embed"
    embedding_size: int = 32
    residual_dropout: float = 0.0
    fixed_ordering: bool = True

    def validate(self) -> None:
        if len(self.predicate_input_bins) != len(self.data_output_bins):
            raise ValueError("predicate_input_bins and data_output_bins must align by column")
        if len(self.predicate_input_bins) == 0:
            raise ValueError("at least one modeled column is required")
        if not self.hidden_sizes:
            raise ValueError("at least one hidden layer is required")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        if self.residual_connections and len(set(self.hidden_sizes)) != 1:
            raise ValueError("residual hidden layers must have compatible dimensions")
        if self.input_encoding not in {"embed", "one_hot"}:
            raise ValueError("input_encoding must be 'embed' or 'one_hot'")
        if not self.fixed_ordering:
            raise ValueError("this milestone requires fixed_ordering=true")


class PredicateResMADE(nn.Module):
    """ResMADE over virtual tokens with real-value categorical output slices."""

    def __init__(self, config: PredicateResMADEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.num_columns = len(config.predicate_input_bins)
        self.forward_calls = 0
        self.column_input_widths = self._column_input_widths(config)
        self.input_width = sum(self.column_input_widths)
        self.output_width = sum(config.data_output_bins)
        self.output_slices = self._output_slices(config.data_output_bins)
        self.input_degrees = self._expanded_input_degrees()
        hidden_degrees = self._hidden_degrees(config.hidden_sizes[0], self.num_columns)
        self.hidden_degrees = hidden_degrees

        if config.input_encoding == "embed":
            self.embeddings = nn.ModuleList(
                [nn.Embedding(num_embeddings=bins, embedding_dim=config.embedding_size)
                 for bins in config.predicate_input_bins]
            )
        else:
            self.embeddings = nn.ModuleList()

        self.input_layer = MaskedLinear(self.input_width, config.hidden_sizes[0])
        self.input_layer.set_mask(
            mask_from_degrees(self.input_degrees, hidden_degrees, strict=False)
        )
        self.activation = self._make_activation(config.activation)

        blocks: list[nn.Module] = []
        if config.residual_connections:
            hidden_mask = mask_from_degrees(hidden_degrees, hidden_degrees, strict=False)
            for _ in config.hidden_sizes[1:]:
                blocks.append(
                    MaskedResidualBlock(
                        config.hidden_sizes[0],
                        hidden_mask,
                        activation=self._make_activation(config.activation),
                        dropout=config.residual_dropout,
                    )
                )
        else:
            previous_degrees = hidden_degrees
            previous_size = config.hidden_sizes[0]
            for hidden_size in config.hidden_sizes[1:]:
                next_degrees = self._hidden_degrees(hidden_size, self.num_columns)
                layer = MaskedLinear(previous_size, hidden_size)
                layer.set_mask(mask_from_degrees(previous_degrees, next_degrees, strict=False))
                blocks.extend([layer, self._make_activation(config.activation)])
                previous_degrees = next_degrees
                previous_size = hidden_size
            hidden_degrees = previous_degrees
        self.hidden_layers = nn.Sequential(*blocks)
        self.final_hidden_degrees = hidden_degrees

        output_degrees = self._expanded_output_degrees()
        self.output_layer = MaskedLinear(config.hidden_sizes[-1], self.output_width)
        self.output_layer.set_mask(
            mask_from_degrees(self.final_hidden_degrees, output_degrees, strict=True)
        )
        if config.direct_io_connections:
            self.direct_io_layer = MaskedLinear(self.input_width, self.output_width, bias=False)
            self.direct_io_layer.set_mask(
                mask_from_degrees(self.input_degrees, output_degrees, strict=True)
            )
        else:
            self.direct_io_layer = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, sum_i data_output_bins[i]]."""

        self.forward_calls += 1
        encoded_inputs = self.encode_inputs(token_ids)
        hidden = self.input_layer(encoded_inputs)
        hidden = self.activation(hidden)
        hidden = self.hidden_layers(hidden)
        logits = self.output_layer(hidden)
        if self.direct_io_layer is not None:
            logits = logits + self.direct_io_layer(encoded_inputs)
        return logits

    def encode_inputs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode predicate-token IDs using per-column embeddings or one-hot blocks."""

        if token_ids.ndim != 2 or token_ids.shape[1] != self.num_columns:
            raise ValueError("token_ids must have shape [batch, number_of_columns]")
        pieces = []
        if self.config.input_encoding == "embed":
            for column_index, embedding in enumerate(self.embeddings):
                pieces.append(embedding(token_ids[:, column_index]))
        else:
            for column_index, bins in enumerate(self.config.predicate_input_bins):
                pieces.append(
                    torch.nn.functional.one_hot(
                        token_ids[:, column_index], num_classes=bins
                    ).to(dtype=torch.float32)
                )
        return torch.cat(pieces, dim=1)

    def split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        return [logits[:, start:stop] for start, stop in self.output_slices]

    def predict_distributions(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return one separately normalized softmax distribution per column."""

        logits = self(token_ids)
        return [torch.softmax(slice_logits, dim=1) for slice_logits in self.split_logits(logits)]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"unsupported activation {name!r}")

    @staticmethod
    def _column_input_widths(config: PredicateResMADEConfig) -> tuple[int, ...]:
        if config.input_encoding == "embed":
            return tuple(config.embedding_size for _ in config.predicate_input_bins)
        return tuple(config.predicate_input_bins)

    def _expanded_input_degrees(self) -> torch.Tensor:
        degrees = []
        for column_index, width in enumerate(self.column_input_widths):
            degrees.extend([column_index] * width)
        return torch.tensor(degrees, dtype=torch.long)

    def _expanded_output_degrees(self) -> torch.Tensor:
        degrees = []
        for column_index, width in enumerate(self.config.data_output_bins):
            degrees.extend([column_index] * width)
        return torch.tensor(degrees, dtype=torch.long)

    @staticmethod
    def _hidden_degrees(hidden_size: int, num_columns: int) -> torch.Tensor:
        if num_columns == 1:
            return torch.full((hidden_size,), -1, dtype=torch.long)
        return torch.tensor(
            [index % (num_columns - 1) for index in range(hidden_size)],
            dtype=torch.long,
        )

    @staticmethod
    def _output_slices(output_bins: Sequence[int]) -> tuple[tuple[int, int], ...]:
        slices = []
        start = 0
        for width in output_bins:
            stop = start + int(width)
            slices.append((start, stop))
            start = stop
        return tuple(slices)

