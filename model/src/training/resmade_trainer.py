from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from model.src.config import resolve_device
from model.src.data.full_join_sampler import FullJoinBatch
from model.src.model.anpm import ANPMConfig
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
    parameter_size_bytes: int
    backbone_parameter_count: int
    anpm_parameter_count: int
    first_loss: float
    last_loss: float
    total_sampled_tuples: int
    nominal_rows_seen: int
    training_seconds: float
    metrics_path: Path
    summary_path: Path
    fanout_effective_sample_size: dict[str, dict[str, float]]
    output_width_original: int
    output_width_factorized: int
    peak_gpu_memory_bytes: int | None
    last_original_column_losses: dict[str, float]
    last_factor_losses: dict[str, float]


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    fanout_effective_sample_size: dict[str, float]
    original_column_losses: dict[str, float]
    factor_losses: dict[str, float]


def build_resmade_from_config(metadata: object, config: dict[str, Any]) -> PredicateResMADE:
    model_config = config["model"]
    vocabularies = PredicateVocabularies.from_metadata(metadata)
    plan = getattr(metadata, "factorization_plan", None)
    output_head_specs = None
    if plan is not None and plan.enabled:
        output_head_specs = plan.output_head_specs
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
            output_head_specs=output_head_specs,
            factorization_plan=plan,
            anpm_config=ANPMConfig.from_dict(config.get("anpm", {})),
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
    batch_size = int(training["batch_size"])
    output_directory = Path(logging.get("output_directory", "model/runs/resmade"))
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "training_metrics.jsonl"
    summary_path = output_directory / "training_summary.json"
    metrics_path.write_text("", encoding="utf-8")
    fanout_ess_values: dict[str, list[float]] = {
        metadata.columns[index].name: [] for index in metadata.fanout_indices()
    }
    last_original_column_losses: dict[str, float] = {}
    last_factor_losses: dict[str, float] = {}
    metrics_interval = int(training.get("validation_interval_steps", 0) or 0)
    training_start = perf_counter()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(training["epochs"])):
        for step_in_epoch in range(int(training["steps_per_epoch"])):
            batch = sample_source.batches(batch_size, seed=seed + global_step)
            step_result = _train_one_batch(
                model,
                optimizer,
                batch,
                metadata,
                vocabularies,
                config,
                device,
            )
            loss = step_result.loss
            last_loss = loss
            if first_loss is None:
                first_loss = loss
            global_step += 1
            for fanout_name, fanout_ess in step_result.fanout_effective_sample_size.items():
                fanout_ess_values[fanout_name].append(float(fanout_ess))
            last_original_column_losses = step_result.original_column_losses
            last_factor_losses = step_result.factor_losses
            interval = int(training.get("checkpoint_interval_steps", 0) or 0)
            if interval and global_step % interval == 0:
                _save_checkpoint(model, optimizer, epoch, global_step, metadata, vocabularies, config)
            if metrics_interval and global_step % metrics_interval == 0:
                _append_metrics(
                    metrics_path,
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "nominal_rows_seen": global_step * batch_size,
                        "total_sampled_tuples": global_step * batch_size,
                        "loss": loss,
                        "fanout_effective_sample_size": step_result.fanout_effective_sample_size,
                        "original_column_losses": step_result.original_column_losses,
                        "factor_losses": step_result.factor_losses,
                    },
                )
        if int(training["steps_per_epoch"]) == 0:
            break
    checkpoint_path = _save_checkpoint(
        model, optimizer, int(training["epochs"]) - 1, global_step, metadata, vocabularies, config
    )
    training_seconds = perf_counter() - training_start
    parameter_size_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()))
    anpm_parameter_count = int(
        sum(parameter.numel() for parameter in model.anpm_decoders.parameters())
    )
    backbone_parameter_count = int(model.parameter_count() - anpm_parameter_count)
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated())
        if device == "cuda" and torch.cuda.is_available()
        else None
    )
    plan = getattr(metadata, "factorization_plan", None)
    output_width_original = (
        int(plan.original_output_width)
        if plan is not None and plan.original_output_width
        else int(sum(metadata.data_output_bins))
    )
    output_width_factorized = (
        int(plan.factorized_output_width)
        if plan is not None and plan.factorized_output_width
        else int(sum(metadata.data_output_bins))
    )
    fanout_ess_summary = {
        fanout_name: {
            "last": values[-1],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for fanout_name, values in fanout_ess_values.items()
        if values
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "parameter_count": model.parameter_count(),
        "parameter_size_bytes": parameter_size_bytes,
        "backbone_parameter_count": backbone_parameter_count,
        "anpm_parameter_count": anpm_parameter_count,
        "first_loss": float(first_loss if first_loss is not None else float("nan")),
        "last_loss": last_loss,
        "optimizer_steps": global_step,
        "batch_size": batch_size,
        "total_sampled_tuples": global_step * batch_size,
        "nominal_rows_seen": global_step * batch_size,
        "training_seconds": training_seconds,
        "fanout_effective_sample_size": fanout_ess_summary,
        "output_width_original": output_width_original,
        "output_width_factorized": output_width_factorized,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "last_original_column_losses": last_original_column_losses,
        "last_factor_losses": last_factor_losses,
        "metrics_path": str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        parameter_count=model.parameter_count(),
        parameter_size_bytes=parameter_size_bytes,
        backbone_parameter_count=backbone_parameter_count,
        anpm_parameter_count=anpm_parameter_count,
        first_loss=float(first_loss if first_loss is not None else float("nan")),
        last_loss=last_loss,
        total_sampled_tuples=global_step * batch_size,
        nominal_rows_seen=global_step * batch_size,
        training_seconds=training_seconds,
        metrics_path=metrics_path,
        summary_path=summary_path,
        fanout_effective_sample_size=fanout_ess_summary,
        output_width_original=output_width_original,
        output_width_factorized=output_width_factorized,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        last_original_column_losses=last_original_column_losses,
        last_factor_losses=last_factor_losses,
    )


def _train_one_batch(
    model: PredicateResMADE,
    optimizer: object,
    batch: FullJoinBatch,
    metadata: object,
    vocabularies: PredicateVocabularies,
    config: dict[str, Any],
    device: str,
) -> TrainingStepResult:
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
        anpm_decoders=getattr(model, "anpm_decoders", None),
        head_loss_reduction=str(config["training"].get("head_loss_reduction", "mean")),
        mask_invalid_factor_combinations=bool(
            config.get("anpm", {}).get("mask_invalid_combinations", True)
        ),
    )
    breakdown.total_loss.backward()
    clip_norm = config["training"].get("gradient_clip_norm")
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
    optimizer.step()
    if not torch.isfinite(breakdown.total_loss):
        raise ValueError("training loss became non-finite")
    fanout_effective_sample_sizes = {}
    for fanout_index in metadata.fanout_indices():
        fanout_ess = effective_sample_size(weights[:, fanout_index])
        if fanout_ess <= 0:
            raise ValueError("fanout effective sample size is non-positive")
        fanout_effective_sample_sizes[metadata.columns[fanout_index].name] = float(fanout_ess)
    return TrainingStepResult(
        loss=float(breakdown.total_loss.detach().cpu()),
        fanout_effective_sample_size=fanout_effective_sample_sizes,
        original_column_losses=breakdown.original_column_losses,
        factor_losses=breakdown.factor_losses,
    )


def _append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


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
