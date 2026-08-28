from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ANPMConfig:
    """Configuration for local factor-prefix decoders on factorized columns."""

    enabled: bool = False
    previous_factor_embedding_size: int = 64
    hidden_size: int = 64
    teacher_force_during_training: bool = True
    mask_invalid_combinations: bool = True
    decode_chunk_size: int = 4096
    final_activation: str = "relu"
    materialize_original_distribution_for_debug: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ANPMConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            previous_factor_embedding_size=int(
                data.get("previous_factor_embedding_size", 64)
            ),
            hidden_size=int(data.get("hidden_size", 64)),
            teacher_force_during_training=bool(
                data.get("teacher_force_during_training", True)
            ),
            mask_invalid_combinations=bool(data.get("mask_invalid_combinations", True)),
            decode_chunk_size=int(data.get("decode_chunk_size", 4096)),
            final_activation=str(data.get("final_activation", "relu")),
            materialize_original_distribution_for_debug=bool(
                data.get("materialize_original_distribution_for_debug", False)
            ),
        )

    def validate(self) -> None:
        if self.previous_factor_embedding_size <= 0:
            raise ValueError("anpm.previous_factor_embedding_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("anpm.hidden_size must be positive")
        if self.decode_chunk_size <= 0:
            raise ValueError("anpm.decode_chunk_size must be positive")
        if self.final_activation not in {"relu", "identity"}:
            raise ValueError("anpm.final_activation must be 'relu' or 'identity'")
        if self.enabled and not self.teacher_force_during_training:
            raise ValueError("factorized ANPM training requires teacher forcing")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "previous_factor_embedding_size": self.previous_factor_embedding_size,
            "hidden_size": self.hidden_size,
            "teacher_force_during_training": self.teacher_force_during_training,
            "mask_invalid_combinations": self.mask_invalid_combinations,
            "decode_chunk_size": self.decode_chunk_size,
            "final_activation": self.final_activation,
            "materialize_original_distribution_for_debug": (
                self.materialize_original_distribution_for_debug
            ),
        }


try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch.
    torch = None
    nn = None


if nn is None:  # pragma: no cover - importable config on torch-free machines.

    class ANPMColumnDecoder:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("PyTorch is required for ANPMColumnDecoder")


else:

    @dataclass(frozen=True)
    class GeneratedANPMParameters:
        """Prefix-generated vectors used to construct DistJoin ANPM transforms.

        For a non-initial factor with domain size D and decoder hidden size H,
        the hypernetwork emits six tensors:

        - first_left: [batch, D], left side of W1
        - first_right: [batch, H], right side of W1
        - hidden_bias: [batch, H], first transform bias
        - second_left: [batch, H], left side of W2
        - second_right: [batch, D], right side of W2
        - logit_bias: [batch, D], second transform bias
        """

        first_left: "torch.Tensor"
        first_right: "torch.Tensor"
        hidden_bias: "torch.Tensor"
        second_left: "torch.Tensor"
        second_right: "torch.Tensor"
        logit_bias: "torch.Tensor"


    class FactorHypernetwork(nn.Module):  # type: ignore[no-redef]
        """Generate one factor's low-rank DistJoin ANPM parameters.

        Adapted from the Adaptive Neural Predicate Modulation structure in
        GIS-PuppetMaster/DistJoin `model/made.py` (`modulation_offset_layers_*`
        and `logits_for_col`). This local version keeps the same two-stage
        generated low-rank transform but scopes it to one factorized original
        column rather than copying DistJoin's full MADE implementation.
        """

        def __init__(
            self,
            *,
            prefix_embedding_size: int,
            current_domain_size: int,
            hidden_size: int,
        ) -> None:
            super().__init__()
            if prefix_embedding_size <= 0:
                raise ValueError("prefix_embedding_size must be positive")
            if current_domain_size <= 1:
                raise ValueError("current_domain_size must exceed one")
            if hidden_size <= 0:
                raise ValueError("hidden_size must be positive")
            self.prefix_embedding_size = int(prefix_embedding_size)
            self.current_domain_size = int(current_domain_size)
            self.hidden_size = int(hidden_size)
            self.first_left = self._generator(self.current_domain_size)
            self.first_right = self._generator(self.hidden_size)
            self.hidden_bias = self._generator(self.hidden_size)
            self.second_left = self._generator(self.hidden_size)
            self.second_right = self._generator(self.current_domain_size)
            self.logit_bias = self._generator(self.current_domain_size)

        def forward(self, prefix_embedding: "torch.Tensor") -> GeneratedANPMParameters:
            """Generate vectors for W1(p)=a1(p)b1(p)^T and W2(p)=a2(p)b2(p)^T.

            Args:
                prefix_embedding: Tensor with shape [batch, prefix_embedding_size],
                    built by concatenating embeddings of the same original
                    column's preceding factor values.
            """

            if prefix_embedding.ndim != 2:
                raise ValueError("prefix_embedding must be [batch, features]")
            if prefix_embedding.shape[1] != self.prefix_embedding_size:
                raise ValueError(
                    "prefix embedding width does not match this factor hypernetwork"
                )
            return GeneratedANPMParameters(
                # [batch, D]
                first_left=self.first_left(prefix_embedding),
                # [batch, H]
                first_right=self.first_right(prefix_embedding),
                # [batch, H]
                hidden_bias=self.hidden_bias(prefix_embedding),
                # [batch, H]
                second_left=self.second_left(prefix_embedding),
                # [batch, D]
                second_right=self.second_right(prefix_embedding),
                # [batch, D]
                logit_bias=self.logit_bias(prefix_embedding),
            )

        def _generator(self, output_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.prefix_embedding_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, int(output_size)),
            )


    class ANPMColumnDecoder(nn.Module):  # type: ignore[no-redef]
        """DistJoin-style generated-weight decoder for one factorized column.

        The previous implementation learned an additive prefix offset,
        `base_logits + h(prefix)`. This branch adapts DistJoin's ANPM mechanism:
        preceding factor embeddings generate low-rank weights and biases that
        transform the current factor's predicate-conditioned base logits.
        """

        def __init__(
            self,
            *,
            factor_domains: Sequence[int],
            embedding_size: int,
            hidden_size: int,
            final_activation: str = "relu",
            representation_size: int | None = None,
        ) -> None:
            super().__init__()
            if len(factor_domains) < 2:
                raise ValueError("ANPMColumnDecoder requires at least two factors")
            if any(int(domain) <= 1 for domain in factor_domains):
                raise ValueError("factor domains must exceed one")
            if embedding_size <= 0:
                raise ValueError("embedding_size must be positive")
            if hidden_size <= 0:
                raise ValueError("hidden_size must be positive")
            if final_activation not in {"relu", "identity"}:
                raise ValueError("final_activation must be 'relu' or 'identity'")
            self.factor_domains = tuple(int(domain) for domain in factor_domains)
            self.embedding_size = int(embedding_size)
            self.hidden_size = int(hidden_size)
            self.final_activation = final_activation
            self.representation_size = (
                None if representation_size is None else int(representation_size)
            )
            if self.representation_size is not None and self.representation_size <= 0:
                raise ValueError("representation_size must be positive")
            self.previous_factor_embeddings = nn.ModuleList(
                nn.Embedding(domain, self.embedding_size)
                for domain in self.factor_domains[:-1]
            )
            self.factor_hypernetworks = nn.ModuleList(
                FactorHypernetwork(
                    prefix_embedding_size=self.embedding_size * factor_index,
                    current_domain_size=(
                        self.representation_size
                        if self.representation_size is not None
                        else domain
                    ),
                    hidden_size=self.hidden_size,
                )
                for factor_index, domain in enumerate(self.factor_domains[1:], start=1)
            )

        def training_logits(
            self,
            base_logits: Sequence["torch.Tensor"],
            true_factors: "torch.Tensor",
            *,
            valid_mask_provider: Any | None = None,
            output_embeddings: Sequence["torch.Tensor"] | None = None,
        ) -> list["torch.Tensor"]:
            """Return factor logits using teacher-forced previous factor labels."""

            if len(base_logits) != len(self.factor_domains):
                raise ValueError("base_logits length must match factor count")
            if true_factors.ndim != 2 or true_factors.shape[1] != len(self.factor_domains):
                raise ValueError("true_factors must be [batch, factor_count]")
            decoded: list[torch.Tensor] = []
            for factor_index, logits in enumerate(base_logits):
                prefix = true_factors[:, :factor_index]
                valid_class_mask = (
                    None
                    if valid_mask_provider is None
                    else valid_mask_provider(factor_index, prefix)
                )
                decoded.append(
                    self.conditional_logits(
                        factor_index,
                        logits,
                        prefix,
                        valid_class_mask=valid_class_mask,
                        output_embedding=(
                            None if output_embeddings is None else output_embeddings[factor_index]
                        ),
                    )
                )
            return decoded

        def conditional_logits(
            self,
            factor_index: int,
            base_logits: "torch.Tensor",
            prefix_factors: "torch.Tensor",
            *,
            valid_class_mask: "torch.Tensor | None" = None,
            output_embedding: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Apply DistJoin's prefix-generated transform to one factor head.

            For factor k>0, this computes

                h = ReLU(base_logits @ W1(prefix) + beta1(prefix))
                logits = ReLU(h @ W2(prefix) + beta2(prefix))

            where W1 is [batch, D, H] and W2 is [batch, H, D]. The matrices are
            constructed transiently from low-rank outer products generated from
            embeddings of factors `0..k-1`. Factor zero returns the base logits
            unchanged, apart from optional validity masking.
            """

            factor_index = int(factor_index)
            if factor_index < 0 or factor_index >= len(self.factor_domains):
                raise ValueError("factor_index outside decoder factor range")
            if base_logits.ndim != 2:
                raise ValueError("base_logits must be [batch, factor_domain_or_repr]")
            expected_domain = self.factor_domains[factor_index]
            expected_width = (
                int(output_embedding.shape[1])
                if output_embedding is not None
                else expected_domain
            )
            if base_logits.shape[1] != expected_width:
                raise ValueError(
                    f"base width {base_logits.shape[1]} != expected width "
                    f"{expected_width}"
                )
            if prefix_factors.ndim != 2 or prefix_factors.shape[1] != factor_index:
                raise ValueError("prefix_factors must be [batch, factor_index]")
            if base_logits.shape[0] not in {1, prefix_factors.shape[0]}:
                raise ValueError("base logits batch must be one or match prefix batch")
            if base_logits.shape[0] == 1 and prefix_factors.shape[0] != 1:
                base_logits = base_logits.expand(prefix_factors.shape[0], -1)
            if factor_index == 0:
                logits = self._project_output(base_logits, output_embedding)
                return self._apply_valid_class_mask(logits, valid_class_mask)
            prefix_embedding = self._encode_prefix(factor_index, prefix_factors)
            parameters = self.generated_parameters(factor_index, prefix_embedding)
            transformed = self.distjoin_transform(base_logits, parameters)
            logits = self._project_output(transformed, output_embedding)
            return self._apply_valid_class_mask(logits, valid_class_mask)

        def _encode_prefix(
            self,
            factor_index: int,
            prefix_factors: "torch.Tensor",
        ) -> "torch.Tensor":
            """Embed and concatenate factors preceding `factor_index`.

            The prefix must come from the same original column and have shape
            [batch, factor_index]. Each prefix value is range-checked against its
            corresponding factor domain before being embedded as `torch.long`.
            """

            if prefix_factors.ndim != 2 or prefix_factors.shape[1] != factor_index:
                raise ValueError("prefix_factors must be [batch, factor_index]")
            if factor_index <= 0:
                raise ValueError("factor zero has no prefix to encode")
            embeddings = []
            for prefix_index in range(factor_index):
                values = prefix_factors[:, prefix_index].long()
                domain = self.factor_domains[prefix_index]
                if torch.any((values < 0) | (values >= domain)):
                    raise ValueError("prefix factor outside its domain")
                embeddings.append(self.previous_factor_embeddings[prefix_index](values))
            return torch.cat(embeddings, dim=1)

        def generated_parameters(
            self,
            factor_index: int,
            prefix_embedding: "torch.Tensor",
        ) -> GeneratedANPMParameters:
            """Generate the six vectors defining one prefix-conditioned transform."""

            if factor_index <= 0 or factor_index >= len(self.factor_domains):
                raise ValueError("only non-initial factors have hypernetworks")
            return self.factor_hypernetworks[factor_index - 1](prefix_embedding)

        def distjoin_transform(
            self,
            base_logits: "torch.Tensor",
            parameters: GeneratedANPMParameters,
        ) -> "torch.Tensor":
            """Transform base logits without materializing generated matrices.

            DistJoin's generated matrices are rank-one:

                W1(p) = a1(p) b1(p)^T
                W2(p) = a2(p) b2(p)^T

            The identity `x (a b^T) = (x dot a) b^T` lets us apply the same
            transform with vector contractions. This is exactly equivalent to
            constructing `[batch,D,H]` and `[batch,H,D]` matrices, but it avoids
            allocating those large tensors during inference and preserves the
            existing checkpoint parameterization.

            Args:
                base_logits: [batch, D] predicate-conditioned factor logits.
                parameters: generated vectors for the same batch/prefixes.

            Returns:
                [batch, D] transformed logits. The final ReLU is faithful to
                DistJoin by default; `final_activation='identity'` is available
                only for explicit diagnostics.
            """

            first_scale = (base_logits * parameters.first_left).sum(
                dim=-1,
                keepdim=True,
            )
            hidden = torch.relu(
                first_scale * parameters.first_right + parameters.hidden_bias
            )
            second_scale = (hidden * parameters.second_left).sum(
                dim=-1,
                keepdim=True,
            )
            logits = second_scale * parameters.second_right + parameters.logit_bias
            if self.final_activation == "relu":
                return torch.relu(logits)
            return logits

        @staticmethod
        def _project_output(
            representation: "torch.Tensor",
            output_embedding: "torch.Tensor | None",
        ) -> "torch.Tensor":
            if output_embedding is None:
                return representation
            return representation @ output_embedding.t()

        @staticmethod
        def _apply_valid_class_mask(
            logits: "torch.Tensor",
            valid_class_mask: "torch.Tensor | None",
        ) -> "torch.Tensor":
            """Set invalid factor classes to zero probability before softmax."""

            if valid_class_mask is None:
                return logits
            if valid_class_mask.dtype != torch.bool:
                raise ValueError("valid_class_mask must be Boolean")
            if valid_class_mask.shape != logits.shape:
                raise ValueError("valid_class_mask must match logits shape")
            if torch.any(~torch.any(valid_class_mask, dim=1)):
                raise ValueError("valid_class_mask contains an all-invalid row")
            return logits.masked_fill(~valid_class_mask, float("-inf"))
