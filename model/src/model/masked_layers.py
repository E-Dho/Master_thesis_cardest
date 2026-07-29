from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TorchMissingError(ImportError):
    """Raised when a PyTorch-only path is used without torch installed."""


def require_torch() -> object:
    try:
        import torch

        return torch
    except ModuleNotFoundError as exc:
        raise TorchMissingError(
            "PyTorch is required for predicate_resmade. Install torch on the "
            "training machine or use model.type=prototype_table."
        ) from exc


def require_torch_nn() -> tuple[object, object]:
    torch = require_torch()
    return torch, torch.nn


torch, nn = require_torch_nn()


class MaskedLinear(nn.Linear):
    """Linear layer whose weights are multiplied by a fixed autoregressive mask."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: "torch.Tensor") -> None:
        if tuple(mask.shape) != tuple(self.mask.shape):
            raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(self.mask.shape)}")
        self.mask.data.copy_(mask.to(dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        return torch.nn.functional.linear(inputs, self.weight * self.mask, self.bias)


class MaskedResidualBlock(nn.Module):
    """Two masked linear layers plus residual connection preserving hidden degree."""

    def __init__(
        self,
        hidden_size: int,
        mask: "torch.Tensor",
        *,
        activation: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.linear1 = MaskedLinear(hidden_size, hidden_size)
        self.linear2 = MaskedLinear(hidden_size, hidden_size)
        self.linear1.set_mask(mask)
        self.linear2.set_mask(mask)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        residual = inputs
        outputs = self.linear1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        outputs = self.dropout(outputs)
        return self.activation(outputs + residual)


def mask_from_degrees(
    input_degrees: "torch.Tensor",
    output_degrees: "torch.Tensor",
    *,
    strict: bool,
) -> "torch.Tensor":
    """Return mask[o,i]=1 when degree(input_i) <= or < degree(output_o)."""

    if strict:
        mask = input_degrees.unsqueeze(0) < output_degrees.unsqueeze(1)
    else:
        mask = input_degrees.unsqueeze(0) <= output_degrees.unsqueeze(1)
    return mask.to(dtype=torch.float32)

