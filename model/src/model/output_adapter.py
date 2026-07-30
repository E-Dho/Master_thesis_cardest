from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata, OriginalColumnFactorization
from model.src.model.anpm import ANPMConfig
from model.src.predicates.encoding import reciprocal_fanout_mask, softmax
from model.src.predicates.operators import PredicateOp, PredicateToken


class OutputDistributionAdapter(Protocol):
    """Boundary for converting raw logits into original-column distributions."""

    def distributions_from_logits(
        self, logits: np.ndarray, columns: tuple[ColumnMetadata, ...]
    ) -> list[np.ndarray]:
        ...

    def original_distribution(self, **kwargs: Any) -> Any:
        ...

    def column_factor(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class TorchBackboneOutputs:
    """Raw model output plus pre-split model-head logits."""

    logits: Any
    split_logits: list[Any]


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

    def original_distribution(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
        metadata: ModelMetadata,
    ) -> Any:
        """Return q_i over the original column domain for an atomic head."""

        import torch

        plan = metadata.factorization_plan
        if plan.enabled:
            head_indices = plan.output_heads_for_column(original_column_index)
            if len(head_indices) != 1:
                raise ValueError("identity adapter cannot decode factorized columns")
            head_index = head_indices[0]
        else:
            head_index = original_column_index
        return torch.softmax(backbone_outputs.split_logits[head_index], dim=1)

    def column_factor(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
        metadata: ModelMetadata,
        predicate_token: PredicateToken,
    ) -> Any:
        """Compute sum_x q_i(x) phi_i(x) for an unfactorized output head."""

        distribution = self.original_distribution(
            original_column_index=original_column_index,
            backbone_outputs=backbone_outputs,
            metadata=metadata,
        )
        return _torch_column_factor_from_distribution(
            distribution,
            metadata.columns[original_column_index],
            predicate_token,
        )


class TorchANPMFactorizedOutputAdapter:
    """Decode original-column factors from ANPM-conditioned factor heads."""

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        anpm_decoders: Mapping[str, Any],
        anpm_config: ANPMConfig,
    ) -> None:
        self.metadata = metadata
        self.anpm_decoders = anpm_decoders
        self.anpm_config = anpm_config
        self.identity_adapter = TorchIdentityOutputAdapter()

    def original_distribution(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
    ) -> Any:
        """Materialize q_i(x) by chunking valid original IDs and renormalizing."""

        import torch

        factorization = self.metadata.factorization_plan.factorization_for_column(
            original_column_index
        )
        if factorization is None:
            return self.identity_adapter.original_distribution(
                original_column_index=original_column_index,
                backbone_outputs=backbone_outputs,
                metadata=self.metadata,
            )
        batch_size = backbone_outputs.logits.shape[0]
        device = backbone_outputs.logits.device
        dtype = backbone_outputs.logits.dtype
        distribution = torch.zeros(
            batch_size,
            factorization.original_domain_size,
            dtype=dtype,
            device=device,
        )
        valid_mass = torch.zeros(batch_size, dtype=dtype, device=device)
        for start in range(
            0, factorization.original_domain_size, self.anpm_config.decode_chunk_size
        ):
            stop = min(
                start + self.anpm_config.decode_chunk_size,
                factorization.original_domain_size,
            )
            factor_values = _factor_tensor_for_id_chunk(
                factorization, start, stop, device=device
            )
            probabilities = self._probabilities_for_factor_values(
                factorization,
                factor_values,
                backbone_outputs,
            )
            distribution[:, start:stop] = probabilities
            valid_mass = valid_mass + probabilities.sum(dim=1)
        return distribution / valid_mass.clamp_min(1.0e-12).unsqueeze(1)

    def column_factor(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
        predicate_token: PredicateToken,
    ) -> Any:
        """Accumulate sum_x q_i(x) phi_i(x) without full-domain materialization."""

        import torch

        column = self.metadata.columns[original_column_index]
        factorization = self.metadata.factorization_plan.factorization_for_column(
            original_column_index
        )
        if factorization is None:
            return self.identity_adapter.column_factor(
                original_column_index=original_column_index,
                backbone_outputs=backbone_outputs,
                metadata=self.metadata,
                predicate_token=predicate_token,
            )
        batch_size = backbone_outputs.logits.shape[0]
        device = backbone_outputs.logits.device
        dtype = backbone_outputs.logits.dtype
        if predicate_token.op == PredicateOp.WILDCARD:
            return torch.ones(batch_size, dtype=dtype, device=device)
        numerator = torch.zeros(batch_size, dtype=dtype, device=device)
        valid_mass = torch.zeros(batch_size, dtype=dtype, device=device)
        for start in range(
            0, factorization.original_domain_size, self.anpm_config.decode_chunk_size
        ):
            stop = min(
                start + self.anpm_config.decode_chunk_size,
                factorization.original_domain_size,
            )
            factor_values = _factor_tensor_for_id_chunk(
                factorization, start, stop, device=device
            )
            probabilities = self._probabilities_for_factor_values(
                factorization,
                factor_values,
                backbone_outputs,
            )
            mask = _torch_predicate_mask_for_id_chunk(
                column,
                predicate_token,
                start,
                stop,
                dtype=dtype,
                device=device,
            )
            numerator = numerator + torch.matmul(probabilities, mask)
            valid_mass = valid_mass + probabilities.sum(dim=1)
        if self.anpm_config.mask_invalid_combinations:
            return numerator / valid_mass.clamp_min(1.0e-12)
        return numerator

    def _probabilities_for_factor_values(
        self,
        factorization: OriginalColumnFactorization,
        factor_values: Any,
        backbone_outputs: TorchBackboneOutputs,
    ) -> Any:
        """Evaluate product_k q(z_k | T_<i, z_<k) for one valid-ID chunk."""

        import torch

        decoder_key = str(factorization.original_column_index)
        if decoder_key not in self.anpm_decoders:
            raise ValueError(
                "missing ANPM decoder for original column "
                f"{factorization.original_column_index}"
            )
        decoder = self.anpm_decoders[decoder_key]
        batch_size = backbone_outputs.logits.shape[0]
        chunk_size = factor_values.shape[0]
        probabilities = torch.ones(
            batch_size,
            chunk_size,
            dtype=backbone_outputs.logits.dtype,
            device=backbone_outputs.logits.device,
        )
        for local_factor_index, head_index in enumerate(
            factorization.factor_column_indices
        ):
            base_logits = backbone_outputs.split_logits[head_index]
            if local_factor_index == 0:
                factor_probabilities = torch.softmax(base_logits, dim=1)[
                    :, factor_values[:, local_factor_index]
                ]
            else:
                domain = factorization.factor_domains[local_factor_index]
                expanded_base = (
                    base_logits.unsqueeze(1)
                    .expand(batch_size, chunk_size, domain)
                    .reshape(batch_size * chunk_size, domain)
                )
                prefix = (
                    factor_values[:, :local_factor_index]
                    .unsqueeze(0)
                    .expand(batch_size, chunk_size, local_factor_index)
                    .reshape(batch_size * chunk_size, local_factor_index)
                )
                conditional_logits = decoder.conditional_logits(
                    local_factor_index,
                    expanded_base,
                    prefix,
                )
                targets = (
                    factor_values[:, local_factor_index]
                    .unsqueeze(0)
                    .expand(batch_size, chunk_size)
                    .reshape(batch_size * chunk_size, 1)
                )
                factor_probabilities = (
                    torch.softmax(conditional_logits, dim=1)
                    .gather(1, targets)
                    .reshape(batch_size, chunk_size)
                )
            probabilities = probabilities * factor_probabilities
        return probabilities


class ANPMFactorizedOutputAdapter(TorchANPMFactorizedOutputAdapter):
    """Public factorized adapter name for ANPM-backed torch inference."""


def _factor_tensor_for_id_chunk(
    factorization: OriginalColumnFactorization,
    start: int,
    stop: int,
    *,
    device: Any,
) -> Any:
    """Vectorize original-ID factorization for one contiguous valid-ID chunk."""

    import torch

    values = torch.arange(start, stop, dtype=torch.long, device=device)
    pieces = []
    for width, offset in zip(factorization.bit_widths, factorization.bit_offsets):
        pieces.append((values >> int(offset)) & ((1 << int(width)) - 1))
    return torch.stack(pieces, dim=1)


def _torch_predicate_mask_for_id_chunk(
    column: ColumnMetadata,
    token: PredicateToken,
    start: int,
    stop: int,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Evaluate an original-domain predicate over encoded IDs in one chunk."""

    import torch

    if token.op == PredicateOp.INV_FANOUT:
        if column.kind != ColumnKind.FANOUT:
            raise ValueError("INV_FANOUT can only be applied to fanout columns")
        mask = reciprocal_fanout_mask(column)[start:stop]
    else:
        mask = np.array(
            [float(token.satisfies(column.domain[index])) for index in range(start, stop)],
            dtype=float,
        )
    return torch.tensor(mask, dtype=dtype, device=device)


def _torch_column_factor_from_distribution(
    distribution: Any,
    column: ColumnMetadata,
    token: PredicateToken,
) -> Any:
    """Apply the existing original-domain mask or fanout potential in torch."""

    import torch

    if token.op == PredicateOp.WILDCARD:
        return torch.ones(
            distribution.shape[0],
            dtype=distribution.dtype,
            device=distribution.device,
        )
    mask = _torch_predicate_mask_for_id_chunk(
        column,
        token,
        0,
        column.domain_size,
        dtype=distribution.dtype,
        device=distribution.device,
    )
    return torch.matmul(distribution, mask)
