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

    class ANPMColumnDecoder(nn.Module):  # type: ignore[no-redef]
        """DistJoin-style local decoder conditioned on previous factor values.

        DistJoin's `model/made.py` uses embeddings of preceding factor values to
        modulate later factor logits. This local adaptation keeps that idea but
        confines the modulation to factors of a single original column.
        """

        def __init__(
            self,
            *,
            factor_domains: Sequence[int],
            embedding_size: int,
            hidden_size: int,
        ) -> None:
            super().__init__()
            if len(factor_domains) < 2:
                raise ValueError("ANPMColumnDecoder requires at least two factors")
            if any(int(domain) <= 1 for domain in factor_domains):
                raise ValueError("factor domains must exceed one")
            self.factor_domains = tuple(int(domain) for domain in factor_domains)
            self.embedding_size = int(embedding_size)
            self.hidden_size = int(hidden_size)
            self.previous_factor_embeddings = nn.ModuleList(
                nn.Embedding(domain, self.embedding_size)
                for domain in self.factor_domains[:-1]
            )
            self.offset_networks = nn.ModuleList()
            for factor_index, domain in enumerate(self.factor_domains):
                if factor_index == 0:
                    self.offset_networks.append(nn.Identity())
                    continue
                self.offset_networks.append(
                    nn.Sequential(
                        nn.Linear(self.embedding_size * factor_index, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, domain),
                    )
                )

        def training_logits(
            self,
            base_logits: Sequence["torch.Tensor"],
            true_factors: "torch.Tensor",
        ) -> list["torch.Tensor"]:
            """Return factor logits using teacher-forced previous factor labels."""

            if len(base_logits) != len(self.factor_domains):
                raise ValueError("base_logits length must match factor count")
            if true_factors.ndim != 2 or true_factors.shape[1] != len(self.factor_domains):
                raise ValueError("true_factors must be [batch, factor_count]")
            decoded: list[torch.Tensor] = []
            for factor_index, logits in enumerate(base_logits):
                decoded.append(
                    self.conditional_logits(
                        factor_index,
                        logits,
                        true_factors[:, :factor_index],
                    )
                )
            return decoded

        def conditional_logits(
            self,
            factor_index: int,
            base_logits: "torch.Tensor",
            prefix_factors: "torch.Tensor",
        ) -> "torch.Tensor":
            """Add a prefix-dependent ANPM offset to one factor's base logits."""

            factor_index = int(factor_index)
            if factor_index < 0 or factor_index >= len(self.factor_domains):
                raise ValueError("factor_index outside decoder factor range")
            if base_logits.ndim != 2:
                raise ValueError("base_logits must be [batch, factor_domain]")
            expected_domain = self.factor_domains[factor_index]
            if base_logits.shape[1] != expected_domain:
                raise ValueError(
                    f"base logits width {base_logits.shape[1]} != factor domain "
                    f"{expected_domain}"
                )
            if factor_index == 0:
                return base_logits
            if prefix_factors.ndim != 2 or prefix_factors.shape[1] != factor_index:
                raise ValueError("prefix_factors must be [batch, factor_index]")
            if base_logits.shape[0] not in {1, prefix_factors.shape[0]}:
                raise ValueError("base logits batch must be one or match prefix batch")
            embeddings = []
            for prefix_index in range(factor_index):
                values = prefix_factors[:, prefix_index].long()
                domain = self.factor_domains[prefix_index]
                if torch.any((values < 0) | (values >= domain)):
                    raise ValueError("prefix factor outside its domain")
                embeddings.append(self.previous_factor_embeddings[prefix_index](values))
            prefix_embedding = torch.cat(embeddings, dim=1)
            offset = self.offset_networks[factor_index](prefix_embedding)
            if base_logits.shape[0] == 1 and offset.shape[0] != 1:
                base_logits = base_logits.expand(offset.shape[0], -1)
            return base_logits + offset
