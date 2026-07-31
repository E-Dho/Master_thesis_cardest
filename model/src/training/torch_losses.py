from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.model.factorization import factorize_rows, valid_factor_class_mask

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TorchLossBreakdown:
    total_loss: "torch.Tensor"
    per_head_losses: list["torch.Tensor"]
    ordinary_loss: float
    indicator_loss: float
    fanout_losses: dict[str, float]
    original_column_losses: dict[str, float] = field(default_factory=dict)
    factor_losses: dict[str, float] = field(default_factory=dict)


def torch_weighted_per_head_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    head_weights: "torch.Tensor",
    metadata: ModelMetadata,
    *,
    anpm_decoders: Mapping[str, Any] | None = None,
    head_loss_reduction: str = "mean",
    mask_invalid_factor_combinations: bool = True,
    epsilon: float = 1.0e-12,
) -> TorchLossBreakdown:
    """Torch WCE: L_i=sum_b w_bi CE(z_bi,y_bi)/(sum_b w_bi+eps)."""

    import torch
    import torch.nn.functional as functional

    if targets.shape != head_weights.shape:
        raise ValueError("targets and head_weights must both be [batch, columns]")
    if targets.shape[1] != len(metadata.columns):
        raise ValueError("target column count does not match metadata")
    if metadata.factorization_plan.enabled:
        return torch_factorized_grouped_cross_entropy(
            logits,
            targets,
            head_weights,
            metadata,
            anpm_decoders=anpm_decoders,
            head_loss_reduction=head_loss_reduction,
            mask_invalid_factor_combinations=mask_invalid_factor_combinations,
            epsilon=epsilon,
        )

    per_head_losses = []
    ordinary_losses = []
    indicator_losses = []
    fanout_losses: dict[str, float] = {}
    original_column_losses: dict[str, float] = {}
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
        original_column_losses[column.name] = float(head_loss.detach().cpu())
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
        original_column_losses=original_column_losses,
    )


def torch_factorized_grouped_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    head_weights: "torch.Tensor",
    metadata: ModelMetadata,
    *,
    anpm_decoders: Mapping[str, Any] | None,
    head_loss_reduction: str = "mean",
    mask_invalid_factor_combinations: bool = True,
    epsilon: float = 1.0e-12,
) -> TorchLossBreakdown:
    """Group teacher-forced factor CE before applying original-column weights."""

    import torch
    import torch.nn.functional as functional

    if anpm_decoders is None:
        raise ValueError("factorized loss requires ANPM decoders")
    plan = metadata.factorization_plan
    target_heads = factorize_rows(targets, metadata)
    slices = metadata.model_output_slices
    split_logits = [logits[:, start:stop] for start, stop in slices]
    per_column_losses: list[torch.Tensor] = []
    ordinary_losses: list[torch.Tensor] = []
    indicator_losses: list[torch.Tensor] = []
    fanout_losses: dict[str, float] = {}
    original_column_losses: dict[str, float] = {}
    factor_losses: dict[str, float] = {}

    for column_index, column in enumerate(metadata.columns):
        factorization = plan.factorization_for_column(column_index)
        head_indices = plan.output_heads_for_column(column_index)
        if factorization is None:
            if len(head_indices) != 1:
                raise ValueError(f"column {column.name!r} must have one atomic head")
            head_index = head_indices[0]
            unweighted = functional.cross_entropy(
                split_logits[head_index],
                targets[:, column_index].long(),
                reduction="none",
            )
            factor_losses[column.name] = float(unweighted.detach().mean().cpu())
            row_loss = unweighted
        else:
            decoder_key = str(column_index)
            if decoder_key not in anpm_decoders:
                raise ValueError(f"missing ANPM decoder for column {column.name!r}")
            decoder = anpm_decoders[decoder_key]
            true_factors = torch.stack(
                [target_heads[:, head_index].long() for head_index in head_indices],
                dim=1,
            )
            base_logits = [split_logits[head_index] for head_index in head_indices]
            valid_mask_provider = None
            if mask_invalid_factor_combinations:
                valid_mask_provider = lambda factor_index, prefix, f=factorization: (
                    valid_factor_class_mask(f, factor_index, prefix)
                )
            decoded_logits = decoder.training_logits(
                base_logits,
                true_factors,
                valid_mask_provider=valid_mask_provider,
            )
            factor_terms = []
            for local_factor_index, factor_logits in enumerate(decoded_logits):
                unweighted = functional.cross_entropy(
                    factor_logits,
                    true_factors[:, local_factor_index],
                    reduction="none",
                )
                factor_name = (
                    f"{column.name}__fact_{local_factor_index}"
                )
                factor_losses[factor_name] = float(unweighted.detach().mean().cpu())
                factor_terms.append(unweighted)
            row_loss = torch.stack(factor_terms, dim=1).sum(dim=1)
        weights = head_weights[:, column_index].to(dtype=row_loss.dtype)
        column_loss = torch.sum(weights * row_loss) / (torch.sum(weights) + epsilon)
        per_column_losses.append(column_loss)
        original_column_losses[column.name] = float(column_loss.detach().cpu())
        if column.kind == ColumnKind.DATA:
            ordinary_losses.append(column_loss)
        elif column.kind == ColumnKind.INDICATOR:
            indicator_losses.append(column_loss)
        elif column.kind == ColumnKind.FANOUT:
            fanout_losses[column.name] = float(column_loss.detach().cpu())

    if head_loss_reduction == "mean":
        total_loss = torch.stack(per_column_losses).mean()
    elif head_loss_reduction == "sum":
        total_loss = torch.stack(per_column_losses).sum()
    else:
        raise ValueError(f"unsupported head_loss_reduction {head_loss_reduction!r}")
    return TorchLossBreakdown(
        total_loss=total_loss,
        per_head_losses=per_column_losses,
        ordinary_loss=_mean_detached(ordinary_losses),
        indicator_loss=_mean_detached(indicator_losses),
        fanout_losses=fanout_losses,
        original_column_losses=original_column_losses,
        factor_losses=factor_losses,
    )


def _mean_detached(values: list["torch.Tensor"]) -> float:
    if not values:
        return 0.0
    import torch

    return float(torch.stack([value.detach().cpu() for value in values]).mean())
