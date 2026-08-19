from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from model.src.data.schema import FactorizationPlan, OutputHeadSpec
from model.src.model.anpm import ANPMColumnDecoder, ANPMConfig
from model.src.model.masked_layers import MaskedLinear, MaskedResidualBlock, mask_from_degrees
from model.src.predicates.operators import PredicateOp
from model.src.predicates.vocabulary import TWO_SLOT_OP_TO_ID

import torch
from torch import nn


@dataclass(frozen=True)
class PredicateResMADEConfig:
    """Configuration for grouped predicate-input ResMADE."""

    predicate_input_bins: tuple[int, ...]
    data_output_bins: tuple[int, ...]
    column_kinds: tuple[str, ...] = ()
    hidden_sizes: tuple[int, ...] = (512, 512, 512, 512)
    residual_connections: bool = True
    direct_io_connections: bool = True
    direct_io_source_kinds: tuple[str, ...] = ("data", "indicator", "fanout")
    activation: str = "relu"
    input_encoding: str = "embed"
    output_encoding: str = "one_hot"
    output_embedding_size: int = 64
    output_embeddings_tied: bool = False
    embedding_size: int = 32
    predicate_encoding_mode: str = "categorical_legacy"
    operator_embedding_size: int = 8
    value_embedding_size: int = 32
    special_embedding_size: int = 8
    merge_hidden_size: int = 64
    token_operator_ids_by_column: tuple[tuple[int, ...], ...] | None = None
    token_value_ids_by_column: tuple[tuple[int, ...], ...] | None = None
    token_upper_ids_by_column: tuple[tuple[int, ...], ...] | None = None
    token_special_ids_by_column: tuple[tuple[int, ...], ...] | None = None
    value_bins_by_column: tuple[int, ...] | None = None
    binary_value_widths_by_column: tuple[int, ...] | None = None
    operator_bins: int = 8
    special_bins: int = 4
    multi_predicate_merge: str = "sum"
    residual_dropout: float = 0.0
    fixed_ordering: bool = True
    output_head_specs: tuple[OutputHeadSpec, ...] | None = None
    factorization_plan: FactorizationPlan | None = None
    anpm_config: ANPMConfig | None = None

    def validate(self) -> None:
        if len(self.predicate_input_bins) != len(self.data_output_bins):
            raise ValueError("predicate_input_bins and data_output_bins must align by column")
        if len(self.predicate_input_bins) == 0:
            raise ValueError("at least one modeled column is required")
        if self.column_kinds and len(self.column_kinds) != len(self.predicate_input_bins):
            raise ValueError("column_kinds must align with predicate_input_bins")
        if not self.hidden_sizes:
            raise ValueError("at least one hidden layer is required")
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive")
        if self.residual_connections and len(set(self.hidden_sizes)) != 1:
            raise ValueError("residual hidden layers must have compatible dimensions")
        if self.input_encoding not in {"embed", "one_hot", "duet_binary"}:
            raise ValueError("input_encoding must be 'embed', 'one_hot', or 'duet_binary'")
        if self.output_encoding not in {"one_hot", "embed"}:
            raise ValueError("output_encoding must be 'one_hot' or 'embed'")
        if self.output_encoding == "embed" and self.output_embedding_size <= 0:
            raise ValueError("output_embedding_size must be positive")
        if self.output_embeddings_tied:
            raise ValueError("output_embeddings_tied=true is not supported yet")
        if self.predicate_encoding_mode not in {
            "categorical_legacy",
            "compositional",
            "hybrid",
            "two_slot",
            "two_slot_categorical_legacy",
            "two_slot_binary_duet",
        }:
            raise ValueError(
                "predicate_encoding_mode must be categorical_legacy, compositional, "
                "hybrid, two_slot, two_slot_categorical_legacy, or two_slot_binary_duet"
            )
        if self.predicate_encoding_mode in {
            "compositional",
            "hybrid",
            "two_slot",
            "two_slot_categorical_legacy",
            "two_slot_binary_duet",
        }:
            if self.input_encoding not in {"embed", "duet_binary"}:
                raise ValueError(
                    "compositional/hybrid/two-slot predicate encoding requires "
                    "input_encoding=embed or duet_binary"
                )
        if self.predicate_encoding_mode in {"two_slot", "two_slot_categorical_legacy"}:
            if self.value_bins_by_column is None:
                raise ValueError("two_slot predicate encoding requires value_bins_by_column")
            if len(self.value_bins_by_column) != len(self.predicate_input_bins):
                raise ValueError("value_bins_by_column must align with predicate_input_bins")
            for size_name, size in [
                ("operator_embedding_size", self.operator_embedding_size),
                ("value_embedding_size", self.value_embedding_size),
                ("merge_hidden_size", self.merge_hidden_size),
            ]:
                if size <= 0:
                    raise ValueError(f"{size_name} must be positive")
        if self.predicate_encoding_mode == "two_slot_binary_duet":
            if self.value_bins_by_column is None:
                raise ValueError("two_slot_binary_duet requires value_bins_by_column")
            if self.binary_value_widths_by_column is None:
                raise ValueError(
                    "two_slot_binary_duet requires binary_value_widths_by_column"
                )
            if len(self.binary_value_widths_by_column) != len(self.predicate_input_bins):
                raise ValueError(
                    "binary_value_widths_by_column must align with predicate_input_bins"
                )
            if self.multi_predicate_merge != "sum":
                raise ValueError(
                    "two_slot_binary_duet currently supports multi_predicate_merge=sum"
                )
            for size_name, size in [
                ("merge_hidden_size", self.merge_hidden_size),
                ("embedding_size", self.embedding_size),
            ]:
                if size <= 0:
                    raise ValueError(f"{size_name} must be positive")
        if self.predicate_encoding_mode in {"compositional", "hybrid"}:
            feature_tables = (
                self.token_operator_ids_by_column,
                self.token_value_ids_by_column,
                self.token_upper_ids_by_column,
                self.token_special_ids_by_column,
            )
            if any(table is None for table in feature_tables):
                raise ValueError(
                    "compositional/hybrid predicate encoding requires token feature tables"
                )
            if self.value_bins_by_column is None:
                raise ValueError(
                    "compositional/hybrid predicate encoding requires value_bins_by_column"
                )
            if len(self.value_bins_by_column) != len(self.predicate_input_bins):
                raise ValueError("value_bins_by_column must align with predicate_input_bins")
            for table in feature_tables:
                assert table is not None
                if len(table) != len(self.predicate_input_bins):
                    raise ValueError("token feature tables must align with predicate_input_bins")
                for column_index, values in enumerate(table):
                    if len(values) != self.predicate_input_bins[column_index]:
                        raise ValueError(
                            "token feature table width must match predicate_input_bins"
                        )
            for size_name, size in [
                ("operator_embedding_size", self.operator_embedding_size),
                ("value_embedding_size", self.value_embedding_size),
                ("special_embedding_size", self.special_embedding_size),
                ("merge_hidden_size", self.merge_hidden_size),
            ]:
                if size <= 0:
                    raise ValueError(f"{size_name} must be positive")
        if not self.fixed_ordering:
            raise ValueError("this milestone requires fixed_ordering=true")
        specs = self.resolved_output_head_specs()
        if not specs:
            raise ValueError("at least one output head is required")
        for spec in specs:
            if spec.domain_size <= 0:
                raise ValueError(f"output head {spec.name!r} has non-positive domain")
            if spec.source_column_index < 0 or spec.source_column_index >= len(self.data_output_bins):
                raise ValueError(
                    f"output head {spec.name!r} references invalid original column "
                    f"{spec.source_column_index}"
                )
        if any(kind not in {"data", "indicator", "fanout"} for kind in self.direct_io_source_kinds):
            raise ValueError(
                "direct_io_source_kinds must contain only data, indicator, or fanout"
            )
        plan = self.factorization_plan
        if plan is not None and plan.enabled and specs != plan.output_head_specs:
            raise ValueError("factorized ResMADE output heads must match metadata plan")
        anpm = self.anpm_config
        if plan is not None and plan.enabled:
            if anpm is None or not anpm.enabled:
                raise ValueError("factorization.enabled=true requires anpm.enabled=true")
            anpm.validate()

    def resolved_output_head_specs(self) -> tuple[OutputHeadSpec, ...]:
        """Return configured output heads or legacy one-head-per-column specs."""

        if self.output_head_specs is not None:
            return self.output_head_specs
        return tuple(
            OutputHeadSpec(f"column_{index}", index, None, domain_size)
            for index, domain_size in enumerate(self.data_output_bins)
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialize constructor settings without relying on dataclass pickling."""

        return {
            "predicate_input_bins": self.predicate_input_bins,
            "data_output_bins": self.data_output_bins,
            "column_kinds": self.column_kinds,
            "hidden_sizes": self.hidden_sizes,
            "residual_connections": self.residual_connections,
            "direct_io_connections": self.direct_io_connections,
            "direct_io_source_kinds": self.direct_io_source_kinds,
            "activation": self.activation,
            "input_encoding": self.input_encoding,
            "output_encoding": self.output_encoding,
            "output_embedding_size": self.output_embedding_size,
            "output_embeddings_tied": self.output_embeddings_tied,
            "embedding_size": self.embedding_size,
            "predicate_encoding_mode": self.predicate_encoding_mode,
            "operator_embedding_size": self.operator_embedding_size,
            "value_embedding_size": self.value_embedding_size,
            "special_embedding_size": self.special_embedding_size,
            "merge_hidden_size": self.merge_hidden_size,
            "token_operator_ids_by_column": self.token_operator_ids_by_column,
            "token_value_ids_by_column": self.token_value_ids_by_column,
            "token_upper_ids_by_column": self.token_upper_ids_by_column,
            "token_special_ids_by_column": self.token_special_ids_by_column,
            "value_bins_by_column": self.value_bins_by_column,
            "binary_value_widths_by_column": self.binary_value_widths_by_column,
            "operator_bins": self.operator_bins,
            "special_bins": self.special_bins,
            "multi_predicate_merge": self.multi_predicate_merge,
            "residual_dropout": self.residual_dropout,
            "fixed_ordering": self.fixed_ordering,
            "output_head_specs": (
                None
                if self.output_head_specs is None
                else [spec.__dict__ for spec in self.output_head_specs]
            ),
            "factorization_plan": (
                None
                if self.factorization_plan is None
                else self.factorization_plan.to_json_dict()
            ),
            "anpm_config": (
                None if self.anpm_config is None else self.anpm_config.to_json_dict()
            ),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> "PredicateResMADEConfig":
        """Load both new factorized configs and legacy unfactorized checkpoints."""

        raw_specs = data.get("output_head_specs")
        output_head_specs = None
        if raw_specs is not None:
            output_head_specs = tuple(
                OutputHeadSpec(
                    name=str(item["name"]),
                    source_column_index=int(item["source_column_index"]),
                    factor_index=(
                        None if item.get("factor_index") is None else int(item["factor_index"])
                    ),
                    domain_size=int(item["domain_size"]),
                )
                for item in raw_specs  # type: ignore[union-attr]
            )
        raw_plan = data.get("factorization_plan")
        raw_anpm = data.get("anpm_config")
        return cls(
            predicate_input_bins=tuple(data["predicate_input_bins"]),  # type: ignore[arg-type,index]
            data_output_bins=tuple(data["data_output_bins"]),  # type: ignore[arg-type,index]
            column_kinds=tuple(str(value) for value in data.get("column_kinds", ())),
            hidden_sizes=tuple(data["hidden_sizes"]),  # type: ignore[arg-type,index]
            residual_connections=bool(data["residual_connections"]),  # type: ignore[index]
            direct_io_connections=bool(data["direct_io_connections"]),  # type: ignore[index]
            direct_io_source_kinds=tuple(
                str(value)
                for value in data.get(
                    "direct_io_source_kinds",
                    ("data", "indicator", "fanout"),
                )
            ),
            activation=str(data["activation"]),  # type: ignore[index]
            input_encoding=str(data["input_encoding"]),  # type: ignore[index]
            output_encoding=str(data.get("output_encoding", "one_hot")),
            output_embedding_size=int(data.get("output_embedding_size", 64)),
            output_embeddings_tied=bool(data.get("output_embeddings_tied", False)),
            embedding_size=int(data["embedding_size"]),  # type: ignore[index]
            predicate_encoding_mode=str(data.get("predicate_encoding_mode", "categorical_legacy")),
            operator_embedding_size=int(data.get("operator_embedding_size", 8)),
            value_embedding_size=int(data.get("value_embedding_size", 32)),
            special_embedding_size=int(data.get("special_embedding_size", 8)),
            merge_hidden_size=int(data.get("merge_hidden_size", 64)),
            token_operator_ids_by_column=_tuple_table_or_none(
                data.get("token_operator_ids_by_column")
            ),
            token_value_ids_by_column=_tuple_table_or_none(
                data.get("token_value_ids_by_column")
            ),
            token_upper_ids_by_column=_tuple_table_or_none(
                data.get("token_upper_ids_by_column")
            ),
            token_special_ids_by_column=_tuple_table_or_none(
                data.get("token_special_ids_by_column")
            ),
            value_bins_by_column=(
                None
                if data.get("value_bins_by_column") is None
                else tuple(int(value) for value in data["value_bins_by_column"])  # type: ignore[index]
            ),
            binary_value_widths_by_column=(
                None
                if data.get("binary_value_widths_by_column") is None
                else tuple(int(value) for value in data["binary_value_widths_by_column"])  # type: ignore[index]
            ),
            operator_bins=int(data.get("operator_bins", 8)),
            special_bins=int(data.get("special_bins", 4)),
            multi_predicate_merge=str(data.get("multi_predicate_merge", "sum")),
            residual_dropout=float(data["residual_dropout"]),  # type: ignore[index]
            fixed_ordering=bool(data["fixed_ordering"]),  # type: ignore[index]
            output_head_specs=output_head_specs,
            factorization_plan=(
                None
                if raw_plan is None
                else FactorizationPlan.from_json_dict(raw_plan)  # type: ignore[arg-type]
            ),
            anpm_config=(
                None if raw_anpm is None else ANPMConfig.from_dict(raw_anpm)  # type: ignore[arg-type]
            ),
        )


class PredicateResMADE(nn.Module):
    """ResMADE over virtual tokens with real-value categorical output slices."""

    def __init__(self, config: PredicateResMADEConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.num_columns = len(config.predicate_input_bins)
        self.forward_calls = 0
        self.output_head_specs = config.resolved_output_head_specs()
        self.output_head_to_original_column = tuple(
            spec.source_column_index for spec in self.output_head_specs
        )
        self.column_kinds = (
            tuple(config.column_kinds)
            if config.column_kinds
            else tuple("data" for _ in range(self.num_columns))
        )
        self.factorization_plan = config.factorization_plan or FactorizationPlan()
        self.anpm_config = config.anpm_config or ANPMConfig()
        self.column_input_widths = self._column_input_widths(config)
        self.input_width = sum(self.column_input_widths)
        self.output_width = sum(spec.domain_size for spec in self.output_head_specs)
        self.output_head_widths = tuple(
            spec.domain_size
            if config.output_encoding == "one_hot"
            else int(config.output_embedding_size)
            for spec in self.output_head_specs
        )
        self.output_width = sum(self.output_head_widths)
        self.output_slices = self._output_slices(self.output_head_widths)
        if config.output_encoding == "embed":
            self.output_embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        num_embeddings=spec.domain_size,
                        embedding_dim=config.output_embedding_size,
                    )
                    for spec in self.output_head_specs
                ]
            )
        else:
            self.output_embeddings = nn.ModuleList()
        self.input_degrees = self._expanded_input_degrees()
        hidden_degrees = self._hidden_degrees(config.hidden_sizes[0], self.num_columns)
        self.hidden_degrees = hidden_degrees

        if self._uses_embedding_input(config) and config.predicate_encoding_mode == "categorical_legacy":
            self.embeddings = nn.ModuleList(
                [nn.Embedding(num_embeddings=bins, embedding_dim=config.embedding_size)
                 for bins in config.predicate_input_bins]
            )
        elif (
            self._uses_embedding_input(config)
            and config.predicate_encoding_mode in {"two_slot", "two_slot_categorical_legacy"}
        ):
            self.embeddings = nn.ModuleList()
            self.literal_embeddings = nn.ModuleList()
            self.operator_embedding = nn.Embedding(
                config.operator_bins, config.operator_embedding_size
            )
            assert config.value_bins_by_column is not None
            self.value_embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=bins, embedding_dim=config.value_embedding_size)
                    for bins in config.value_bins_by_column
                ]
            )
            merge_input = 2 * (
                config.operator_embedding_size + config.value_embedding_size
            )
            self.predicate_merge_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(merge_input, config.merge_hidden_size),
                        self._make_activation(config.activation),
                        nn.Linear(config.merge_hidden_size, config.embedding_size),
                    )
                    for _ in config.predicate_input_bins
                ]
            )
            self.special_embedding = nn.Embedding(1, config.special_embedding_size)
        elif (
            self._uses_embedding_input(config)
            and config.predicate_encoding_mode == "two_slot_binary_duet"
        ):
            self.embeddings = nn.ModuleList()
            self.literal_embeddings = nn.ModuleList()
            self.value_embeddings = nn.ModuleList()
            self.special_token_embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=2, embedding_dim=config.embedding_size)
                    for _ in config.predicate_input_bins
                ]
            )
            assert config.binary_value_widths_by_column is not None
            self.binary_value_widths_by_column = tuple(
                int(width) for width in config.binary_value_widths_by_column
            )
            self.predicate_slot_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            int(config.operator_bins) + int(binary_width),
                            config.merge_hidden_size,
                        ),
                        self._make_activation(config.activation),
                        nn.Linear(config.merge_hidden_size, config.embedding_size),
                    )
                    for binary_width in self.binary_value_widths_by_column
                ]
            )
            self.inv_fanout_operator_id = int(TWO_SLOT_OP_TO_ID[PredicateOp.INV_FANOUT])
        elif self._uses_embedding_input(config):
            self.embeddings = nn.ModuleList()
            if config.predicate_encoding_mode == "hybrid":
                self.literal_embeddings = nn.ModuleList(
                    [
                        nn.Embedding(
                            num_embeddings=bins,
                            embedding_dim=config.embedding_size,
                        )
                        for bins in config.predicate_input_bins
                    ]
                )
            else:
                self.literal_embeddings = nn.ModuleList()
            self.operator_embedding = nn.Embedding(
                config.operator_bins, config.operator_embedding_size
            )
            assert config.value_bins_by_column is not None
            self.value_embeddings = nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=bins, embedding_dim=config.value_embedding_size)
                    for bins in config.value_bins_by_column
                ]
            )
            self.special_embedding = nn.Embedding(
                config.special_bins, config.special_embedding_size
            )
            merge_input = (
                config.operator_embedding_size
                + 2 * config.value_embedding_size
                + config.special_embedding_size
            )
            self.predicate_merge_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(merge_input, config.merge_hidden_size),
                        self._make_activation(config.activation),
                        nn.Linear(config.merge_hidden_size, config.embedding_size),
                    )
                    for _ in config.predicate_input_bins
                ]
            )
            assert config.token_operator_ids_by_column is not None
            assert config.token_value_ids_by_column is not None
            assert config.token_upper_ids_by_column is not None
            assert config.token_special_ids_by_column is not None
            for name, table in [
                ("token_operator_ids", config.token_operator_ids_by_column),
                ("token_value_ids", config.token_value_ids_by_column),
                ("token_upper_ids", config.token_upper_ids_by_column),
                ("token_special_ids", config.token_special_ids_by_column),
            ]:
                padded = [torch.tensor(values, dtype=torch.long) for values in table]
                for column_index, values in enumerate(padded):
                    self.register_buffer(f"{name}_{column_index}", values)
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
            self.direct_io_layer.set_mask(self._direct_io_mask(output_degrees))
        else:
            self.direct_io_layer = None
        self.anpm_decoders = self._build_anpm_decoders()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch, sum_j model_output_bins[j]]."""

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

        pieces = []
        if self._uses_embedding_input(self.config) and self.config.predicate_encoding_mode == "categorical_legacy":
            if token_ids.ndim != 2 or token_ids.shape[1] != self.num_columns:
                raise ValueError("token_ids must have shape [batch, number_of_columns]")
            for column_index, embedding in enumerate(self.embeddings):
                pieces.append(embedding(token_ids[:, column_index]))
        elif (
            self._uses_embedding_input(self.config)
            and self.config.predicate_encoding_mode in {"two_slot", "two_slot_categorical_legacy"}
        ):
            if token_ids.ndim != 3 or token_ids.shape[1] != self.num_columns or token_ids.shape[2] != 4:
                raise ValueError("two_slot token_ids must have shape [batch, columns, 4]")
            for column_index, value_embedding in enumerate(self.value_embeddings):
                slot_values = token_ids[:, column_index, :]
                merged = torch.cat(
                    [
                        self.operator_embedding(slot_values[:, 0]),
                        value_embedding(slot_values[:, 1]),
                        self.operator_embedding(slot_values[:, 2]),
                        value_embedding(slot_values[:, 3]),
                    ],
                    dim=1,
                )
                pieces.append(self.predicate_merge_networks[column_index](merged))
        elif (
            self._uses_embedding_input(self.config)
            and self.config.predicate_encoding_mode == "two_slot_binary_duet"
        ):
            if token_ids.ndim != 3 or token_ids.shape[1] != self.num_columns or token_ids.shape[2] != 4:
                raise ValueError(
                    "two_slot_binary_duet token_ids must have shape [batch, columns, 4]"
                )
            for column_index, slot_network in enumerate(self.predicate_slot_networks):
                slot_values = token_ids[:, column_index, :]
                first_op_ids = slot_values[:, 0]
                second_op_ids = slot_values[:, 2]
                wildcard_mask = (first_op_ids == 0) & (second_op_ids == 0)
                inv_mask = first_op_ids == self.inv_fanout_operator_id
                encoded = self._encode_binary_duet_slot(
                    column_index,
                    first_op_ids,
                    slot_values[:, 1],
                    slot_network,
                )
                second_encoded = self._encode_binary_duet_slot(
                    column_index,
                    second_op_ids,
                    slot_values[:, 3],
                    slot_network,
                )
                encoded = encoded + second_encoded
                special_ids = torch.zeros_like(first_op_ids)
                special_ids = torch.where(inv_mask, torch.ones_like(special_ids), special_ids)
                special_encoded = self.special_token_embeddings[column_index](special_ids)
                use_special = wildcard_mask | inv_mask
                encoded = torch.where(use_special.unsqueeze(1), special_encoded, encoded)
                pieces.append(encoded)
        elif self._uses_embedding_input(self.config):
            if token_ids.ndim != 2 or token_ids.shape[1] != self.num_columns:
                raise ValueError("token_ids must have shape [batch, number_of_columns]")
            for column_index, value_embedding in enumerate(self.value_embeddings):
                column_token_ids = token_ids[:, column_index]
                operator_ids = getattr(self, f"token_operator_ids_{column_index}")[
                    column_token_ids
                ]
                value_ids = getattr(self, f"token_value_ids_{column_index}")[
                    column_token_ids
                ]
                upper_ids = getattr(self, f"token_upper_ids_{column_index}")[
                    column_token_ids
                ]
                special_ids = getattr(self, f"token_special_ids_{column_index}")[
                    column_token_ids
                ]
                merged = torch.cat(
                    [
                        self.operator_embedding(operator_ids),
                        value_embedding(value_ids),
                        value_embedding(upper_ids),
                        self.special_embedding(special_ids),
                    ],
                    dim=1,
                )
                encoded = self.predicate_merge_networks[column_index](merged)
                if self.config.predicate_encoding_mode == "hybrid":
                    encoded = encoded + self.literal_embeddings[column_index](
                        column_token_ids
                    )
                pieces.append(encoded)
        else:
            if token_ids.ndim != 2 or token_ids.shape[1] != self.num_columns:
                raise ValueError("token_ids must have shape [batch, number_of_columns]")
            for column_index, bins in enumerate(self.config.predicate_input_bins):
                pieces.append(
                    torch.nn.functional.one_hot(
                        token_ids[:, column_index], num_classes=bins
                    ).to(dtype=torch.float32)
                )
        return torch.cat(pieces, dim=1)

    def split_logits(self, logits: torch.Tensor) -> list[torch.Tensor]:
        head_outputs = self.split_head_outputs(logits)
        return [
            self.project_head_output(head_index, head_output)
            for head_index, head_output in enumerate(head_outputs)
        ]

    def split_head_outputs(self, outputs: torch.Tensor) -> list[torch.Tensor]:
        return [outputs[:, start:stop] for start, stop in self.output_slices]

    def project_head_output(self, head_index: int, head_output: torch.Tensor) -> torch.Tensor:
        if self.config.output_encoding == "one_hot":
            return head_output
        embedding = self.output_embeddings[int(head_index)].weight
        return head_output @ embedding.t()

    def project_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                self.project_head_output(head_index, head_output)
                for head_index, head_output in enumerate(self.split_head_outputs(outputs))
            ],
            dim=1,
        )

    def predict_distributions(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Return one normalized distribution per model output head."""

        logits = self(token_ids)
        return [torch.softmax(slice_logits, dim=1) for slice_logits in self.split_logits(logits)]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _uses_embedding_input(config: PredicateResMADEConfig) -> bool:
        return config.input_encoding in {"embed", "duet_binary"}

    def _encode_binary_duet_slot(
        self,
        column_index: int,
        operator_ids: torch.Tensor,
        value_ids: torch.Tensor,
        slot_network: nn.Module,
    ) -> torch.Tensor:
        """Encode one Duet predicate slot as f_i(binary(value), one_hot(op))."""

        import torch.nn.functional as functional

        width = int(self.binary_value_widths_by_column[column_index])
        op_features = functional.one_hot(
            operator_ids.long(),
            num_classes=int(self.config.operator_bins),
        ).to(dtype=torch.float32)
        bit_positions = torch.arange(width, device=value_ids.device, dtype=torch.long)
        raw_bits = (
            (value_ids.long().unsqueeze(1) >> bit_positions.unsqueeze(0)) & 1
        ).to(dtype=torch.float32)
        missing_value_id = int(self.config.value_bins_by_column[column_index] - 1)  # type: ignore[index]
        empty_mask = (operator_ids == 0) | (value_ids == missing_value_id)
        raw_bits = torch.where(empty_mask.unsqueeze(1), torch.zeros_like(raw_bits), raw_bits)
        features = torch.cat([op_features, raw_bits], dim=1)
        encoded = slot_network(features)
        return torch.where(empty_mask.unsqueeze(1), torch.zeros_like(encoded), encoded)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"unsupported activation {name!r}")

    @staticmethod
    def _column_input_widths(config: PredicateResMADEConfig) -> tuple[int, ...]:
        if config.input_encoding in {"embed", "duet_binary"}:
            return tuple(config.embedding_size for _ in config.predicate_input_bins)
        return tuple(config.predicate_input_bins)

    def _expanded_input_degrees(self) -> torch.Tensor:
        degrees = []
        for column_index, width in enumerate(self.column_input_widths):
            degrees.extend([column_index] * width)
        return torch.tensor(degrees, dtype=torch.long)

    def _expanded_output_degrees(self) -> torch.Tensor:
        degrees = []
        for spec, width in zip(self.output_head_specs, self.output_head_widths):
            degrees.extend([spec.source_column_index] * int(width))
        return torch.tensor(degrees, dtype=torch.long)

    def _direct_io_mask(self, output_degrees: torch.Tensor) -> torch.Tensor:
        mask = mask_from_degrees(self.input_degrees, output_degrees, strict=True)
        allowed_kinds = set(self.config.direct_io_source_kinds)
        input_allowed: list[float] = []
        for column_index, width in enumerate(self.column_input_widths):
            input_allowed.extend(
                [1.0 if self._column_kind_name(column_index) in allowed_kinds else 0.0]
                * width
            )
        source_mask = torch.tensor(input_allowed, dtype=mask.dtype).unsqueeze(0)
        return mask * source_mask

    def _column_kind_name(self, column_index: int) -> str:
        kinds = getattr(self, "column_kinds", None)
        if kinds is None:
            return "data"
        return str(kinds[column_index])

    def _build_anpm_decoders(self) -> nn.ModuleDict:
        decoders = nn.ModuleDict()
        plan = self.factorization_plan
        if not plan.enabled:
            return decoders
        if not self.anpm_config.enabled:
            return decoders
        for factorization in plan.original_column_factorizations:
            decoders[str(factorization.original_column_index)] = ANPMColumnDecoder(
                factor_domains=factorization.factor_domains,
                embedding_size=self.anpm_config.previous_factor_embedding_size,
                hidden_size=self.anpm_config.hidden_size,
                final_activation=self.anpm_config.final_activation,
                representation_size=(
                    self.config.output_embedding_size
                    if self.config.output_encoding == "embed"
                    else None
                ),
            )
        return decoders

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


def _tuple_table_or_none(data: object) -> tuple[tuple[int, ...], ...] | None:
    if data is None:
        return None
    return tuple(tuple(int(value) for value in values) for values in data)  # type: ignore[union-attr]
