from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata, OriginalColumnFactorization
from model.src.model.anpm import ANPMConfig
from model.src.model.factorization import factorize_value, valid_factor_class_mask
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

    def interval_mass(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
        metadata: ModelMetadata,
        lower_literal: Any | None,
        upper_literal: Any | None,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Any:
        """Return exact interval mass from one atomic original-column distribution."""

        distribution = self.original_distribution(
            original_column_index=original_column_index,
            backbone_outputs=backbone_outputs,
            metadata=metadata,
        )
        token = _interval_token(
            lower_literal=lower_literal,
            upper_literal=upper_literal,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )
        return _checked_probability_mass(
            _torch_column_factor_from_distribution(
                distribution,
                metadata.columns[original_column_index],
                token,
            )
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
        self.last_factorized_profile: dict[str, Any] = {}

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
        evaluator = self._evaluator(factorization, backbone_outputs)
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
            probabilities = evaluator.probabilities_for_factor_values(factor_values)
            distribution[:, start:stop] = probabilities
            valid_mass = valid_mass + probabilities.sum(dim=1)
        self.last_factorized_profile = evaluator.profile
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
        evaluator = self._evaluator(factorization, backbone_outputs)
        optimized = evaluator.predicate_mass(column, predicate_token)
        if optimized is not None:
            self.last_factorized_profile = evaluator.profile
            return optimized
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
            probabilities = evaluator.probabilities_for_factor_values(factor_values)
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
        evaluator.profile["fallback_enumeration_used"] = True
        self.last_factorized_profile = evaluator.profile
        if self.anpm_config.mask_invalid_combinations:
            return numerator / valid_mass.clamp_min(1.0e-12)
        return numerator

    def interval_mass(
        self,
        *,
        original_column_index: int,
        backbone_outputs: TorchBackboneOutputs,
        lower_literal: Any | None,
        upper_literal: Any | None,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Any:
        """Return exact interval mass from one conditioned original-column state."""

        column = self.metadata.columns[original_column_index]
        factorization = self.metadata.factorization_plan.factorization_for_column(
            original_column_index
        )
        if factorization is None:
            return self.identity_adapter.interval_mass(
                original_column_index=original_column_index,
                backbone_outputs=backbone_outputs,
                metadata=self.metadata,
                lower_literal=lower_literal,
                upper_literal=upper_literal,
                lower_inclusive=lower_inclusive,
                upper_inclusive=upper_inclusive,
            )
        token = _interval_token(
            lower_literal=lower_literal,
            upper_literal=upper_literal,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )
        evaluator = self._evaluator(factorization, backbone_outputs)
        optimized = evaluator.predicate_mass(column, token)
        if optimized is not None:
            self.last_factorized_profile = evaluator.profile
            return _checked_probability_mass(optimized)

        import torch

        batch_size = backbone_outputs.logits.shape[0]
        device = backbone_outputs.logits.device
        dtype = backbone_outputs.logits.dtype
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
            probabilities = evaluator.probabilities_for_factor_values(factor_values)
            mask = _torch_predicate_mask_for_id_chunk(
                column,
                token,
                start,
                stop,
                dtype=dtype,
                device=device,
            )
            numerator = numerator + torch.matmul(probabilities, mask)
            valid_mass = valid_mass + probabilities.sum(dim=1)
        evaluator.profile["fallback_enumeration_used"] = True
        self.last_factorized_profile = evaluator.profile
        if self.anpm_config.mask_invalid_combinations:
            return _checked_probability_mass(numerator / valid_mass.clamp_min(1.0e-12))
        return _checked_probability_mass(numerator)

    def _evaluator(
        self,
        factorization: OriginalColumnFactorization,
        backbone_outputs: TorchBackboneOutputs,
    ) -> "FactorizedColumnProbabilityEvaluator":
        decoder_key = str(factorization.original_column_index)
        if decoder_key not in self.anpm_decoders:
            raise ValueError(
                "missing ANPM decoder for original column "
                f"{factorization.original_column_index}"
            )
        return FactorizedColumnProbabilityEvaluator(
            factorization=factorization,
            decoder=self.anpm_decoders[decoder_key],
            anpm_config=self.anpm_config,
            backbone_outputs=backbone_outputs,
        )


class ANPMFactorizedOutputAdapter(TorchANPMFactorizedOutputAdapter):
    """Public factorized adapter name for ANPM-backed torch inference."""


class FactorizedColumnProbabilityEvaluator:
    """Query-local exact probability evaluator for one factorized column.

    The evaluator hides factor digits from the estimator. It reuses conditional
    distributions for repeated prefixes within one query and exposes exact
    original-column predicate masses for wildcard, equality, and one-sided range
    predicates. Fallback enumeration remains available for arbitrary masks.
    """

    def __init__(
        self,
        *,
        factorization: OriginalColumnFactorization,
        decoder: Any,
        anpm_config: ANPMConfig,
        backbone_outputs: TorchBackboneOutputs,
    ) -> None:
        self.factorization = factorization
        self.decoder = decoder
        self.anpm_config = anpm_config
        self.backbone_outputs = backbone_outputs
        self._distribution_cache: dict[tuple[Any, ...], Any] = {}
        self.profile: dict[str, Any] = {
            "anpm_calls": 0,
            "largest_anpm_batch": 0,
            "fallback_enumeration_used": False,
            "unique_prefixes_by_factor": {},
        }

    @property
    def batch_size(self) -> int:
        return int(self.backbone_outputs.logits.shape[0])

    @property
    def device(self) -> Any:
        return self.backbone_outputs.logits.device

    @property
    def dtype(self) -> Any:
        return self.backbone_outputs.logits.dtype

    def predicate_mass(
        self,
        column: ColumnMetadata,
        token: PredicateToken,
    ) -> Any | None:
        """Return optimized exact mass when the token has a prefix algorithm."""

        import torch

        if token.op == PredicateOp.WILDCARD:
            return torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        if token.op == PredicateOp.EQUAL:
            encoded = _encoded_id_or_none(column, token.value)
            if encoded is None:
                return torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
            return self.encoded_id_mass(encoded)
        if token.op == PredicateOp.RANGE:
            lower = _encoded_id_or_none(column, token.value)
            upper = _encoded_id_or_none(column, token.upper)
            if lower is None or upper is None:
                return None
            if not _encoded_interval_matches_predicate(column, token):
                return None
            lower_cdf = lower if not token.lower_inclusive else lower - 1
            upper_cdf = upper if token.upper_inclusive else upper - 1
            return self.less_equal_mass(upper_cdf) - self.less_equal_mass(lower_cdf)
        if token.op == PredicateOp.LESS_EQUAL:
            encoded = _encoded_id_or_none(column, token.value)
            if encoded is None:
                return None
            if not _encoded_interval_matches_predicate(column, token):
                return None
            return self.less_equal_mass(encoded)
        if token.op == PredicateOp.LESS_THAN:
            encoded = _encoded_id_or_none(column, token.value)
            if encoded is None:
                return None
            if not _encoded_interval_matches_predicate(column, token):
                return None
            return self.less_equal_mass(encoded - 1)
        if token.op == PredicateOp.GREATER_EQUAL:
            encoded = _encoded_id_or_none(column, token.value)
            if encoded is None:
                return None
            if not _encoded_interval_matches_predicate(column, token):
                return None
            return 1.0 - self.less_equal_mass(encoded - 1)
        if token.op == PredicateOp.GREATER_THAN:
            encoded = _encoded_id_or_none(column, token.value)
            if encoded is None:
                return None
            if not _encoded_interval_matches_predicate(column, token):
                return None
            return 1.0 - self.less_equal_mass(encoded)
        return None

    def encoded_id_mass(self, encoded_id: int) -> Any:
        """Evaluate P(X=id) by following exactly one factor path."""

        import torch

        if encoded_id < 0 or encoded_id >= self.factorization.original_domain_size:
            return torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
        digits = factorize_value(encoded_id, self.factorization)
        probability = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        prefix_digits: list[int] = []
        for factor_index, digit in enumerate(digits):
            prefix = torch.tensor(
                [prefix_digits],
                dtype=torch.long,
                device=self.device,
            )
            distribution = self.conditional_distribution(factor_index, prefix)[:, 0, :]
            probability = probability * distribution[:, int(digit)]
            prefix_digits.append(int(digit))
        return probability

    def less_equal_mass(self, encoded_id: int) -> Any:
        """Evaluate P(X <= encoded_id) by lexicographic factor-prefix CDF.

        Factorization is most-significant-factor first, so an encoded-ID prefix
        corresponds to a lexicographic prefix. At factor k, all classes smaller
        than the threshold digit can be accumulated with one cumulative sum, and
        only the equal digit needs to continue to the next factor.
        """

        import torch

        if encoded_id < 0:
            return torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
        if encoded_id >= self.factorization.original_domain_size - 1:
            return torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        digits = factorize_value(encoded_id, self.factorization)
        mass = torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
        equal_prefix_mass = torch.ones(self.batch_size, dtype=self.dtype, device=self.device)
        prefix_digits: list[int] = []
        for factor_index, digit in enumerate(digits):
            prefix = torch.tensor(
                [prefix_digits],
                dtype=torch.long,
                device=self.device,
            )
            distribution = self.conditional_distribution(factor_index, prefix)[:, 0, :]
            digit = int(digit)
            if digit > 0:
                mass = mass + equal_prefix_mass * distribution[:, :digit].sum(dim=1)
            equal_prefix_mass = equal_prefix_mass * distribution[:, digit]
            prefix_digits.append(digit)
        return mass + equal_prefix_mass

    def probabilities_for_factor_values(self, factor_values: Any) -> Any:
        """Evaluate products for original-ID factor rows using unique prefixes."""

        import torch

        factor_values = factor_values.long()
        chunk_size = int(factor_values.shape[0])
        probabilities = torch.ones(
            self.batch_size,
            chunk_size,
            dtype=self.dtype,
            device=self.device,
        )
        for factor_index in range(len(self.factorization.factor_domains)):
            if factor_index == 0:
                prefixes = torch.empty(1, 0, dtype=torch.long, device=self.device)
                distribution = self.conditional_distribution(factor_index, prefixes)
                selected = distribution[:, 0, :][:, factor_values[:, factor_index]]
            else:
                prefixes = factor_values[:, :factor_index]
                unique_prefixes, inverse_indices = torch.unique(
                    prefixes,
                    dim=0,
                    return_inverse=True,
                )
                self.profile["unique_prefixes_by_factor"][factor_index] = max(
                    int(unique_prefixes.shape[0]),
                    int(self.profile["unique_prefixes_by_factor"].get(factor_index, 0)),
                )
                distribution = self.conditional_distribution(factor_index, unique_prefixes)
                gathered = distribution[:, inverse_indices, :]
                targets = (
                    factor_values[:, factor_index]
                    .view(1, chunk_size, 1)
                    .expand(self.batch_size, chunk_size, 1)
                )
                selected = gathered.gather(2, targets).squeeze(2)
            probabilities = probabilities * selected
        return probabilities

    def conditional_distribution(self, factor_index: int, prefixes: Any) -> Any:
        """Return q(z_k | z_<k) for unique prefixes, cached per query."""

        import torch

        prefixes = prefixes.to(device=self.device, dtype=torch.long)
        key = _prefix_cache_key(factor_index, prefixes)
        cached = self._distribution_cache.get(key)
        if cached is not None:
            return cached
        head_index = self.factorization.factor_column_indices[factor_index]
        base_logits = self.backbone_outputs.split_logits[head_index]
        if factor_index == 0:
            prefix_for_batch = torch.empty(
                self.batch_size,
                0,
                dtype=torch.long,
                device=self.device,
            )
            valid_class_mask = (
                valid_factor_class_mask(self.factorization, factor_index, prefix_for_batch)
                if self.anpm_config.mask_invalid_combinations
                else None
            )
            logits = self.decoder.conditional_logits(
                factor_index,
                base_logits,
                prefix_for_batch,
                valid_class_mask=valid_class_mask,
            )
            distribution = torch.softmax(logits, dim=1).unsqueeze(1)
        else:
            prefix_count = int(prefixes.shape[0])
            domain = self.factorization.factor_domains[factor_index]
            expanded_base = (
                base_logits.unsqueeze(1)
                .expand(self.batch_size, prefix_count, domain)
                .reshape(self.batch_size * prefix_count, domain)
            )
            expanded_prefixes = (
                prefixes.unsqueeze(0)
                .expand(self.batch_size, prefix_count, factor_index)
                .reshape(self.batch_size * prefix_count, factor_index)
            )
            valid_class_mask = (
                valid_factor_class_mask(self.factorization, factor_index, expanded_prefixes)
                if self.anpm_config.mask_invalid_combinations
                else None
            )
            logits = self.decoder.conditional_logits(
                factor_index,
                expanded_base,
                expanded_prefixes,
                valid_class_mask=valid_class_mask,
            )
            distribution = torch.softmax(logits, dim=1).reshape(
                self.batch_size,
                prefix_count,
                domain,
            )
            self.profile["anpm_calls"] += 1
            self.profile["largest_anpm_batch"] = max(
                int(self.profile["largest_anpm_batch"]),
                self.batch_size * prefix_count,
            )
        self._distribution_cache[key] = distribution.detach()
        return self._distribution_cache[key]


def _encoded_id_or_none(column: ColumnMetadata, value: Any) -> int | None:
    try:
        return column.encode_value(value)
    except ValueError:
        return None


def _encoded_interval_matches_predicate(
    column: ColumnMetadata,
    token: PredicateToken,
) -> bool:
    """Check whether encoded-ID interval math preserves predicate semantics."""

    return all(
        token.satisfies(value) == _encoded_interval_contains(index, column, token)
        for index, value in enumerate(column.domain)
    )


def _encoded_interval_contains(
    index: int,
    column: ColumnMetadata,
    token: PredicateToken,
) -> bool:
    if token.op == PredicateOp.LESS_EQUAL:
        encoded = _encoded_id_or_none(column, token.value)
        return encoded is not None and index <= encoded
    if token.op == PredicateOp.LESS_THAN:
        encoded = _encoded_id_or_none(column, token.value)
        return encoded is not None and index < encoded
    if token.op == PredicateOp.GREATER_EQUAL:
        encoded = _encoded_id_or_none(column, token.value)
        return encoded is not None and index >= encoded
    if token.op == PredicateOp.GREATER_THAN:
        encoded = _encoded_id_or_none(column, token.value)
        return encoded is not None and index > encoded
    if token.op == PredicateOp.RANGE:
        lower = _encoded_id_or_none(column, token.value)
        upper = _encoded_id_or_none(column, token.upper)
        if lower is None or upper is None:
            return False
        lower_ok = index >= lower if token.lower_inclusive else index > lower
        upper_ok = index <= upper if token.upper_inclusive else index < upper
        return lower_ok and upper_ok
    return False


def _prefix_cache_key(factor_index: int, prefixes: Any) -> tuple[Any, ...]:
    """Build a query-local cache key for immutable prefix tensors."""

    detached = prefixes.detach().to(device="cpu", dtype=prefixes.dtype).contiguous()
    return (
        int(factor_index),
        tuple(int(size) for size in detached.shape),
        str(detached.dtype),
        detached.numpy().tobytes(),
    )


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


def _interval_token(
    *,
    lower_literal: Any | None,
    upper_literal: Any | None,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> PredicateToken:
    if lower_literal is not None and upper_literal is not None:
        return PredicateToken(
            PredicateOp.RANGE,
            value=lower_literal,
            upper=upper_literal,
            lower_inclusive=lower_inclusive,
            upper_inclusive=upper_inclusive,
        )
    if lower_literal is not None:
        return PredicateToken(
            PredicateOp.GREATER_EQUAL if lower_inclusive else PredicateOp.GREATER_THAN,
            value=lower_literal,
        )
    if upper_literal is not None:
        return PredicateToken(
            PredicateOp.LESS_EQUAL if upper_inclusive else PredicateOp.LESS_THAN,
            value=upper_literal,
        )
    return PredicateToken.wildcard()


def _checked_probability_mass(mass: Any, *, tolerance: float = 1.0e-7) -> Any:
    """Clamp only tiny probability-mass violations and reject larger ones."""

    import torch

    if torch.any(mass < -tolerance) or torch.any(mass > 1.0 + tolerance):
        raise ValueError(
            "interval probability mass outside [0,1]: "
            f"min={float(mass.min().detach().cpu())}, "
            f"max={float(mass.max().detach().cpu())}"
        )
    return mass.clamp(0.0, 1.0)
