from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from model.src.data.schema import ColumnKind, ModelMetadata

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TorchLossBreakdown:
    total_loss: "torch.Tensor"
    per_head_losses: list["torch.Tensor"]
    ordinary_loss: float
    indicator_loss: float
    fanout_losses: dict[str, float]


def torch_weighted_per_head_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    head_weights: "torch.Tensor",
    metadata: ModelMetadata,
    *,
    head_loss_reduction: str = "mean",
    epsilon: float = 1.0e-12,
) -> TorchLossBreakdown:
    """Torch WCE: L_i=sum_b w_bi CE(z_bi,y_bi)/(sum_b w_bi+eps)."""

    import torch
    import torch.nn.functional as functional

    if targets.shape != head_weights.shape:
        raise ValueError("targets and head_weights must both be [batch, columns]")
    if targets.shape[1] != len(metadata.columns):
        raise ValueError("target column count does not match metadata")

    per_head_losses = []
    ordinary_losses = []
    indicator_losses = []
    fanout_losses: dict[str, float] = {}
    for column_index, (start, stop) in enumerate(metadata.output_slices):
        column_logits = logits[:, start:stop]
        unweighted = functional.cross_entropy(
            column_logits, targets[:, column_index].long(), reduction="none"
        )
        weights = head_weights[:, column_index].to(dtype=unweighted.dtype)
        denominator = torch.sum(weights) + epsilon
        head_loss = torch.sum(weights * unweighted) / denominator
        per_head_losses.append(head_loss)
        column = metadata.columns[column_index]
        if column.kind == ColumnKind.DATA:
            ordinary_losses.append(head_loss)
        elif column.kind == ColumnKind.INDICATOR:
            indicator_losses.append(head_loss)
        elif column.kind == ColumnKind.FANOUT:
            fanout_losses[column.name] = float(head_loss.detach().cpu())
    if head_loss_reduction == "mean":
        total_loss = torch.stack(per_head_losses).mean()
    elif head_loss_reduction == "sum":
        total_loss = torch.stack(per_head_losses).sum()
    else:
        raise ValueError(f"unsupported head_loss_reduction {head_loss_reduction!r}")
    return TorchLossBreakdown(
        total_loss=total_loss,
        per_head_losses=per_head_losses,
        ordinary_loss=_mean_detached(ordinary_losses),
        indicator_loss=_mean_detached(indicator_losses),
        fanout_losses=fanout_losses,
    )


def _mean_detached(values: list["torch.Tensor"]) -> float:
    if not values:
        return 0.0
    import torch

    return float(torch.stack([value.detach().cpu() for value in values]).mean())

