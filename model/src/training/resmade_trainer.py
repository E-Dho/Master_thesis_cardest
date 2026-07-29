from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import resolve_device
from model.src.data.full_join_sampler import FullJoinBatch
from model.src.model.checkpoint import save_resmade_checkpoint
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.training.losses import cumulative_inverse_fanout_weights, effective_sample_size
from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    parameter_count: int
    first_loss: float
    last_loss: float


def build_resmade_from_config(metadata: object, config: dict[str, Any]) -> PredicateResMADE:
    model_config = config["model"]
    vocabularies = PredicateVocabularies.from_metadata(metadata)
    return PredicateResMADE(
        PredicateResMADEConfig(
            predicate_input_bins=vocabularies.input_bins,
            data_output_bins=metadata.data_output_bins,
            hidden_sizes=tuple(model_config.get("hidden_sizes", [128, 128])),
            residual_connections=bool(model_config.get("residual_connections", True)),
            direct_io_connections=bool(model_config.get("direct_io_connections", True)),
            activation=str(model_config.get("activation", "relu")),
            input_encoding=str(model_config.get("input_encoding", "embed")),
            embedding_size=int(model_config.get("embedding_size", 16)),
            residual_dropout=float(model_config.get("residual_dropout", 0.0)),
            fixed_ordering=bool(model_config.get("fixed_ordering", True)),
        )
    )


def train_resmade_sample_source(sample_source: object, config: dict[str, Any]) -> TrainingResult:
    """Run the ResMADE training loop over full-join batches and save a checkpoint."""

    import torch

    training = config["training"]
    logging = config.get("logging", {})
    seed = int(training.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(config)
    metadata = sample_source.metadata
    vocabularies = PredicateVocabularies.from_metadata(metadata)
    model = build_resmade_from_config(metadata, config).to(device)
    optimizer_name = str(training.get("optimizer", "adam")).lower()
    optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=float(training["learning_rate"]))
    first_loss: float | None = None
    last_loss = float("nan")
    global_step = 0
    for epoch in range(int(training["epochs"])):
        for step_in_epoch in range(int(training["steps_per_epoch"])):
            batch = sample_source.batches(int(training["batch_size"]), seed=seed + global_step)
            loss = _train_one_batch(
                model,
                optimizer,
                batch,
                metadata,
                vocabularies,
                config,
                device,
            )
            last_loss = loss
            if first_loss is None:
                first_loss = loss
            global_step += 1
            interval = int(training.get("checkpoint_interval_steps", 0) or 0)
            if interval and global_step % interval == 0:
                _save_checkpoint(model, optimizer, epoch, global_step, metadata, vocabularies, config)
        if int(training["steps_per_epoch"]) == 0:
            break
    checkpoint_path = _save_checkpoint(
        model, optimizer, int(training["epochs"]) - 1, global_step, metadata, vocabularies, config
    )
    output_directory = Path(logging.get("output_directory", checkpoint_path.parent))
    output_directory.mkdir(parents=True, exist_ok=True)
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        parameter_count=model.parameter_count(),
        first_loss=float(first_loss if first_loss is not None else float("nan")),
        last_loss=last_loss,
    )


def _train_one_batch(
    model: PredicateResMADE,
    optimizer: object,
    batch: FullJoinBatch,
    metadata: object,
    vocabularies: PredicateVocabularies,
    config: dict[str, Any],
    device: str,
) -> float:
    import torch

    token_rows = [
        tokens_for_query_tables(
            metadata,
            {column.table for column in metadata.columns if column.table is not None},
            {column.name for column in metadata.columns if column.kind.value == "fanout"},
        )
        for _ in range(len(batch.encoded_values))
    ]
    weights = cumulative_inverse_fanout_weights(
        batch.encoded_values,
        token_rows,
        metadata,
        compute_in_log_space=bool(config["fanout"].get("compute_weights_in_log_space", True)),
    )
    token_ids = encode_tokens_tensor(token_rows, vocabularies, device=device)
    targets = torch.tensor(batch.encoded_values, dtype=torch.long, device=device)
    head_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(token_ids)
    breakdown = torch_weighted_per_head_cross_entropy(
        logits,
        targets,
        head_weights,
        metadata,
        head_loss_reduction=str(config["training"].get("head_loss_reduction", "mean")),
    )
    breakdown.total_loss.backward()
    clip_norm = config["training"].get("gradient_clip_norm")
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
    optimizer.step()
    if not torch.isfinite(breakdown.total_loss):
        raise ValueError("training loss became non-finite")
    for fanout_index in metadata.fanout_indices():
        fanout_ess = effective_sample_size(weights[:, fanout_index])
        if fanout_ess <= 0:
            raise ValueError("fanout effective sample size is non-positive")
    return float(breakdown.total_loss.detach().cpu())


def _save_checkpoint(
    model: PredicateResMADE,
    optimizer: object,
    epoch: int,
    step: int,
    metadata: object,
    vocabularies: PredicateVocabularies,
    config: dict[str, Any],
) -> Path:
    output_directory = Path(config.get("logging", {}).get("output_directory", "model/runs/resmade"))
    checkpoint_path = output_directory / f"checkpoint_step_{step}.pt"
    save_resmade_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=epoch,
        step=step,
        metadata=metadata,
        predicate_vocabularies=vocabularies,
        config=config,
        preparation_manifest_id=config.get("dataset", {}).get("name"),
    )
    return checkpoint_path

