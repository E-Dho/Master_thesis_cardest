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
from model.src.predicates.generation import (
    PredicateTrainingContextGenerator,
    predicate_context_diagnostics,
    literal_token_occurrences,
    literal_token_stats,
    token_coverage,
)
from model.src.predicates.operators import PredicateOp
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.vocabulary import PredicateVocabularies, key_to_token
from model.src.predicates.vocabulary import (
    TWO_SLOT_OPERATOR_BINS,
    two_slot_value_bins_by_column,
)
from model.src.training.losses import cumulative_inverse_fanout_weights, effective_sample_size
from model.src.training.torch_losses import torch_weighted_per_head_cross_entropy


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    best_checkpoint_path: Path | None
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
    generated_predicate_contexts: int
    rejected_unsatisfied_contexts: int
    included_indicator_contradictions: int
    predicate_token_coverage: dict[str, dict[str, int]]
    predicate_literal_token_stats: dict[str, dict[str, dict[str, int | float | None]]]
    predicate_context_diagnostics: dict[str, Any]
    last_predicate_embedding_gradient_coverage: dict[str, Any]
    fresh_sampler_rows: int
    fixture_rows_reused: int
    validation_summary: dict[str, Any]


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    fanout_effective_sample_size: dict[str, float]
    original_column_losses: dict[str, float]
    factor_losses: dict[str, float]
    generated_contexts: int
    rejected_unsatisfied_contexts: int
    included_indicator_contradictions: int
    predicate_token_coverage: dict[str, dict[str, int]]
    literal_token_occurrences: dict[str, dict[str, dict[str, int]]]
    predicate_embedding_gradient_coverage: dict[str, Any]
    predicate_context_diagnostics: dict[str, Any]


def build_resmade_from_config(metadata: object, config: dict[str, Any]) -> PredicateResMADE:
    model_config = config["model"]
    predicate_encoding = config.get("predicate_encoding", {})
    vocabularies = predicate_vocabularies_from_config(metadata, config)
    plan = getattr(metadata, "factorization_plan", None)
    output_head_specs = None
    if plan is not None and plan.enabled:
        output_head_specs = plan.output_head_specs
    encoding_mode = str(predicate_encoding.get("mode", "categorical_legacy"))
    compositional_features = _predicate_encoding_feature_tables(
        metadata,
        vocabularies,
        mode=encoding_mode,
    )
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
            predicate_encoding_mode=encoding_mode,
            operator_embedding_size=int(predicate_encoding.get("operator_embedding_size", 8)),
            value_embedding_size=int(predicate_encoding.get("value_embedding_size", 32)),
            special_embedding_size=int(predicate_encoding.get("special_embedding_size", 8)),
            merge_hidden_size=int(predicate_encoding.get("merge_hidden_size", 64)),
            **compositional_features,
            residual_dropout=float(model_config.get("residual_dropout", 0.0)),
            fixed_ordering=bool(model_config.get("fixed_ordering", True)),
            output_head_specs=output_head_specs,
            factorization_plan=plan,
            anpm_config=ANPMConfig.from_dict(config.get("anpm", {})),
        )
    )


def predicate_vocabularies_from_config(
    metadata: object,
    config: dict[str, Any],
) -> PredicateVocabularies:
    """Build predicate vocabularies, including optional native training ranges."""

    predicate_generation = config.get("predicate_generation", {})
    predicate_encoding = config.get("predicate_encoding", {})
    encoding_mode = str(predicate_encoding.get("mode", "categorical_legacy"))
    include_native_ranges = (
        encoding_mode != "two_slot"
        and bool(predicate_generation.get("enable_native_range_tokens", False))
    )
    native_range_max_domain_size = int(
        predicate_generation.get("native_range_max_domain_size", 512)
    )
    return PredicateVocabularies.from_metadata(
        metadata,
        include_native_ranges=include_native_ranges,
        native_range_max_domain_size=native_range_max_domain_size,
        encoding_mode="two_slot" if encoding_mode == "two_slot" else "categorical",
    )


def _predicate_encoding_feature_tables(
    metadata: object,
    vocabularies: PredicateVocabularies,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode == "two_slot":
        return {
            "value_bins_by_column": two_slot_value_bins_by_column(metadata),
            "operator_bins": TWO_SLOT_OPERATOR_BINS,
        }
    if mode not in {"compositional", "hybrid"}:
        return {}
    op_to_id = {op: index for index, op in enumerate(PredicateOp)}
    operator_ids = []
    value_ids = []
    upper_ids = []
    special_ids = []
    value_bins = []
    for column_index, column in enumerate(metadata.columns):
        missing_value_id = int(column.domain_size)
        domain_index = {value: index for index, value in enumerate(column.domain)}
        column_operator_ids = []
        column_value_ids = []
        column_upper_ids = []
        column_special_ids = []
        for key in vocabularies.token_keys_by_column[column_index]:
            token = key_to_token(key)
            column_operator_ids.append(op_to_id[token.op])
            value_id = domain_index.get(token.value, missing_value_id)
            upper_id = domain_index.get(token.upper, missing_value_id)
            column_value_ids.append(value_id)
            column_upper_ids.append(upper_id)
            has_value = token.value in domain_index
            has_upper = token.upper in domain_index
            column_special_ids.append((1 if has_value else 0) + (2 if has_upper else 0))
        operator_ids.append(tuple(column_operator_ids))
        value_ids.append(tuple(column_value_ids))
        upper_ids.append(tuple(column_upper_ids))
        special_ids.append(tuple(column_special_ids))
        value_bins.append(missing_value_id + 1)
    return {
        "token_operator_ids_by_column": tuple(operator_ids),
        "token_value_ids_by_column": tuple(value_ids),
        "token_upper_ids_by_column": tuple(upper_ids),
        "token_special_ids_by_column": tuple(special_ids),
        "value_bins_by_column": tuple(value_bins),
        "operator_bins": len(PredicateOp),
        "special_bins": 4,
    }


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
    vocabularies = predicate_vocabularies_from_config(metadata, config)
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
    total_generated_contexts = 0
    total_rejected_contexts = 0
    total_indicator_contradictions = 0
    fresh_sampler_rows = 0
    fixture_rows_reused = 0
    aggregate_literal_occurrences: dict[str, dict[str, dict[str, int]]] = {}
    last_gradient_coverage: dict[str, Any] = {}
    last_context_diagnostics: dict[str, Any] = {}
    aggregate_token_coverage = {
        column.name: {
            "wildcard": 0,
            "equal": 0,
            "less_than": 0,
            "less_equal": 0,
            "greater_than": 0,
            "greater_equal": 0,
            "range": 0,
            "indicator_equal_1": 0,
            "indicator_wildcard": 0,
            "fanout_inv": 0,
            "fanout_wildcard": 0,
        }
        for column in metadata.columns
    }
    last_original_column_losses: dict[str, float] = {}
    last_factor_losses: dict[str, float] = {}
    metrics_interval = int(training.get("validation_interval_steps", 0) or 0)
    validation_config = config.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    validation_interval = int(
        validation_config.get("interval_steps", metrics_interval or 0) or 0
    )
    validation_batches = int(validation_config.get("fresh_sampler_batches", 0) or 0)
    validation_batch_size = int(validation_config.get("batch_size", batch_size) or batch_size)
    selection_metric = str(
        validation_config.get("selection_metric", "validation_weighted_nll")
    )
    minimize_selection = bool(validation_config.get("minimize", True))
    best_metric: float | None = None
    best_step: int | None = None
    best_checkpoint_path: Path | None = None
    validation_history: list[dict[str, Any]] = []
    validation_fresh_sampler_rows = 0
    validation_fixture_rows_reused = 0
    if validation_enabled:
        if validation_interval <= 0:
            raise ValueError("validation.enabled=true requires validation.interval_steps > 0")
        if validation_batches <= 0:
            raise ValueError("validation.enabled=true requires validation.fresh_sampler_batches > 0")
    training_start = perf_counter()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(training["epochs"])):
        for step_in_epoch in range(int(training["steps_per_epoch"])):
            batch = sample_source.batches(batch_size, seed=seed + global_step)
            fresh_sampler_rows += int(getattr(batch, "fresh_rows_drawn", 0))
            fixture_rows_reused += int(getattr(batch, "fixture_rows_reused", 0))
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
            total_generated_contexts += step_result.generated_contexts
            total_rejected_contexts += step_result.rejected_unsatisfied_contexts
            total_indicator_contradictions += step_result.included_indicator_contradictions
            _merge_coverage(aggregate_token_coverage, step_result.predicate_token_coverage)
            _merge_literal_occurrences(aggregate_literal_occurrences, step_result.literal_token_occurrences)
            last_gradient_coverage = step_result.predicate_embedding_gradient_coverage
            last_context_diagnostics = step_result.predicate_context_diagnostics
            interval = int(training.get("checkpoint_interval_steps", 0) or 0)
            if interval and global_step % interval == 0:
                _save_checkpoint(model, optimizer, epoch, global_step, metadata, vocabularies, config)
            if metrics_interval and global_step % metrics_interval == 0:
                metrics_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "nominal_rows_seen": global_step * batch_size,
                    "total_sampled_tuples": global_step * batch_size,
                    "generated_predicate_contexts": total_generated_contexts,
                    "rejected_unsatisfied_contexts": total_rejected_contexts,
                    "included_indicator_contradictions": total_indicator_contradictions,
                    "fresh_sampler_rows": fresh_sampler_rows,
                    "fixture_rows_reused": fixture_rows_reused,
                    "loss": loss,
                    "fanout_effective_sample_size": step_result.fanout_effective_sample_size,
                    "original_column_losses": step_result.original_column_losses,
                    "factor_losses": step_result.factor_losses,
                    "predicate_token_coverage": aggregate_token_coverage,
                    "predicate_literal_token_stats": literal_token_stats(
                        aggregate_literal_occurrences,
                        metadata,
                    ),
                    "predicate_context_diagnostics": last_context_diagnostics,
                    "predicate_embedding_gradient_coverage": last_gradient_coverage,
                }
                if validation_enabled and global_step % validation_interval == 0:
                    validation_metrics = _run_validation(
                        model,
                        sample_source,
                        metadata,
                        vocabularies,
                        config,
                        device,
                        batch_size=validation_batch_size,
                        batches=validation_batches,
                        seed=seed + 1_000_000 + global_step,
                    )
                    validation_fresh_sampler_rows += int(
                        validation_metrics["validation_fresh_sampler_rows"]
                    )
                    validation_fixture_rows_reused += int(
                        validation_metrics["validation_fixture_rows_reused"]
                    )
                    selected_value = float(validation_metrics[selection_metric])
                    improved = (
                        best_metric is None
                        or (selected_value < best_metric if minimize_selection else selected_value > best_metric)
                    )
                    validation_metrics.update(
                        {
                            "validation_selection_metric": selection_metric,
                            "validation_selection_metric_value": selected_value,
                            "validation_improved_best": improved,
                        }
                    )
                    if improved:
                        best_metric = selected_value
                        best_step = global_step
                        best_checkpoint_path = _save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            global_step,
                            metadata,
                            vocabularies,
                            config,
                            filename="checkpoint_best.pt",
                        )
                    validation_history.append({"step": global_step, **validation_metrics})
                    metrics_payload.update(validation_metrics)
                _append_metrics(metrics_path, metrics_payload)
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
        "generated_predicate_contexts": total_generated_contexts,
        "rejected_unsatisfied_contexts": total_rejected_contexts,
        "included_indicator_contradictions": total_indicator_contradictions,
        "fresh_sampler_rows": fresh_sampler_rows,
        "validation_fresh_sampler_rows": validation_fresh_sampler_rows,
        "sampler_run_calls": getattr(sample_source, "sampler_run_calls", None),
        "distinct_original_rows_seen_estimate": getattr(
            sample_source, "distinct_original_rows_seen_estimate", None
        ),
        "fixture_rows_reused": fixture_rows_reused,
        "validation_fixture_rows_reused": validation_fixture_rows_reused,
        "predicate_token_coverage": aggregate_token_coverage,
        "predicate_literal_token_stats": literal_token_stats(
            aggregate_literal_occurrences,
            metadata,
        ),
        "predicate_context_diagnostics": last_context_diagnostics,
        "last_predicate_embedding_gradient_coverage": last_gradient_coverage,
        "columns_with_unseen_evaluation_token_types": [],
        "training_seconds": training_seconds,
        "fanout_effective_sample_size": fanout_ess_summary,
        "output_width_original": output_width_original,
        "output_width_factorized": output_width_factorized,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "last_original_column_losses": last_original_column_losses,
        "last_factor_losses": last_factor_losses,
        "metrics_path": str(metrics_path),
        "validation": {
            "enabled": validation_enabled,
            "interval_steps": validation_interval if validation_enabled else None,
            "fresh_sampler_batches": validation_batches if validation_enabled else None,
            "batch_size": validation_batch_size if validation_enabled else None,
            "selection_metric": selection_metric if validation_enabled else None,
            "minimize": minimize_selection if validation_enabled else None,
            "best_metric": best_metric,
            "best_step": best_step,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
            "history": validation_history,
            "validation_fresh_sampler_rows": validation_fresh_sampler_rows,
            "validation_fixture_rows_reused": validation_fixture_rows_reused,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
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
        generated_predicate_contexts=total_generated_contexts,
        rejected_unsatisfied_contexts=total_rejected_contexts,
        included_indicator_contradictions=total_indicator_contradictions,
        predicate_token_coverage=aggregate_token_coverage,
        predicate_literal_token_stats=literal_token_stats(
            aggregate_literal_occurrences,
            metadata,
        ),
        predicate_context_diagnostics=last_context_diagnostics,
        last_predicate_embedding_gradient_coverage=last_gradient_coverage,
        fresh_sampler_rows=fresh_sampler_rows,
        fixture_rows_reused=fixture_rows_reused,
        validation_summary=summary["validation"],
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

    predicate_config = config.get("predicate_generation", {})
    generator_seed = int(predicate_config.get("seed", config["training"].get("seed", 0)))
    rng = np.random.default_rng(generator_seed + int(getattr(model, "forward_calls", 0)))
    context_generator = PredicateTrainingContextGenerator(predicate_config)
    contexts, target_rows, generation_stats = context_generator.generate_batch(
        encoded_rows=batch.encoded_values,
        metadata=metadata,
        rng=rng,
    )
    token_rows = [list(context.tokens) for context in contexts]
    coverage = token_coverage(
        [context.tokens for context in contexts],
        metadata,
    )
    literal_occurrences = literal_token_occurrences(
        [context.tokens for context in contexts],
        metadata,
    )
    context_diagnostics = predicate_context_diagnostics(contexts, metadata)
    weights = cumulative_inverse_fanout_weights(
        target_rows,
        token_rows,
        metadata,
        compute_in_log_space=bool(config["fanout"].get("compute_weights_in_log_space", True)),
    )
    token_ids = encode_tokens_tensor(token_rows, vocabularies, device=device)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
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
        )
    )
    breakdown.total_loss.backward()
    gradient_coverage = _predicate_embedding_gradient_coverage(
        model,
        token_rows,
        token_ids,
        metadata,
    )
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
        generated_contexts=generation_stats.generated_contexts,
        rejected_unsatisfied_contexts=generation_stats.rejected_unsatisfied_contexts,
        included_indicator_contradictions=generation_stats.included_indicator_contradictions,
        predicate_token_coverage=coverage,
        literal_token_occurrences=literal_occurrences,
        predicate_context_diagnostics=context_diagnostics,
        predicate_embedding_gradient_coverage=gradient_coverage,
    )


def _run_validation(
    model: PredicateResMADE,
    sample_source: object,
    metadata: object,
    vocabularies: PredicateVocabularies,
    config: dict[str, Any],
    device: str,
    *,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    if batches <= 0:
        raise ValueError("validation batches must be positive")
    if batch_size <= 0:
        raise ValueError("validation batch_size must be positive")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    ordinary_losses: list[float] = []
    indicator_losses: list[float] = []
    fanout_ess_values: dict[str, list[float]] = {
        metadata.columns[index].name: [] for index in metadata.fanout_indices()
    }
    fresh_rows = 0
    fixture_rows = 0
    predicate_config = config.get("predicate_generation", {})
    generator_seed = int(predicate_config.get("seed", config["training"].get("seed", 0)))
    context_generator = PredicateTrainingContextGenerator(predicate_config)
    discard_buffer = getattr(sample_source, "discard_buffer", None)
    if discard_buffer is not None:
        discard_buffer()
    with torch.no_grad():
        for batch_index in range(batches):
            batch = sample_source.batches(batch_size, seed=seed + batch_index)
            fresh_rows += int(getattr(batch, "fresh_rows_drawn", 0))
            fixture_rows += int(getattr(batch, "fixture_rows_reused", 0))
            rng = np.random.default_rng(generator_seed + seed + batch_index)
            contexts, target_rows, _ = context_generator.generate_batch(
                encoded_rows=batch.encoded_values,
                metadata=metadata,
                rng=rng,
            )
            token_rows = [list(context.tokens) for context in contexts]
            weights = cumulative_inverse_fanout_weights(
                target_rows,
                token_rows,
                metadata,
                compute_in_log_space=bool(
                    config["fanout"].get("compute_weights_in_log_space", True)
                ),
            )
            token_ids = encode_tokens_tensor(token_rows, vocabularies, device=device)
            targets = torch.tensor(target_rows, dtype=torch.long, device=device)
            head_weights = torch.tensor(weights, dtype=torch.float32, device=device)
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
            if not torch.isfinite(breakdown.total_loss):
                raise ValueError("validation loss became non-finite")
            losses.append(float(breakdown.total_loss.detach().cpu()))
            ordinary_losses.append(float(breakdown.ordinary_loss))
            indicator_losses.append(float(breakdown.indicator_loss))
            for fanout_index in metadata.fanout_indices():
                fanout_ess = effective_sample_size(weights[:, fanout_index])
                if fanout_ess <= 0:
                    raise ValueError("validation fanout effective sample size is non-positive")
                fanout_ess_values[metadata.columns[fanout_index].name].append(float(fanout_ess))
    if was_training:
        model.train()
    fanout_summary = {
        fanout_name: {
            "last": values[-1],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for fanout_name, values in fanout_ess_values.items()
        if values
    }
    weighted_nll = float(np.mean(losses))
    return {
        "validation_nll": weighted_nll,
        "validation_weighted_nll": weighted_nll,
        "validation_ordinary_nll": float(np.mean(ordinary_losses)) if ordinary_losses else 0.0,
        "validation_indicator_nll": float(np.mean(indicator_losses)) if indicator_losses else 0.0,
        "validation_batches": int(batches),
        "validation_batch_size": int(batch_size),
        "validation_rows": int(batch_size * batches),
        "validation_fresh_sampler_rows": int(fresh_rows),
        "validation_fixture_rows_reused": int(fixture_rows),
        "validation_fanout_effective_sample_size": fanout_summary,
    }


def _merge_coverage(
    aggregate: dict[str, dict[str, int]],
    update: dict[str, dict[str, int]],
) -> None:
    for column_name, counts in update.items():
        target = aggregate.setdefault(column_name, {})
        for key, value in counts.items():
            target[key] = int(target.get(key, 0)) + int(value)


def _merge_literal_occurrences(
    aggregate: dict[str, dict[str, dict[str, int]]],
    update: dict[str, dict[str, dict[str, int]]],
) -> None:
    for column_name, by_op in update.items():
        aggregate_column = aggregate.setdefault(column_name, {})
        for op_name, by_literal in by_op.items():
            aggregate_op = aggregate_column.setdefault(op_name, {})
            for literal_key, count in by_literal.items():
                aggregate_op[literal_key] = int(aggregate_op.get(literal_key, 0)) + int(count)


def _predicate_embedding_gradient_coverage(
    model: PredicateResMADE,
    token_rows: list[list[Any]],
    token_ids: Any,
    metadata: object,
) -> dict[str, Any]:
    """Report whether observed non-wildcard predicate embeddings received gradients."""

    if getattr(model.config, "input_encoding", "") != "embed":
        return {"mode": "not_applicable", "reason": "input_encoding is not embed"}
    if getattr(model.config, "predicate_encoding_mode", "") == "two_slot":
        return _two_slot_gradient_coverage(model, token_rows, token_ids, metadata)
    if getattr(model.config, "predicate_encoding_mode", "") in {"compositional", "hybrid"}:
        return _compositional_gradient_coverage(model, token_rows, token_ids, metadata)
    observed = 0
    nonzero = 0
    ordinary_observed = 0
    ordinary_nonzero = 0
    by_column: dict[str, dict[str, int]] = {}
    for column_index, column in enumerate(metadata.columns):
        gradient = model.embeddings[column_index].weight.grad
        if gradient is None:
            continue
        token_id_values = token_ids[:, column_index].detach().cpu().numpy()
        seen_pairs = {
            (int(token_id), token_rows[row_index][column_index])
            for row_index, token_id in enumerate(token_id_values)
        }
        column_counts = by_column.setdefault(
            column.name,
            {
                "observed_non_wildcard_tokens": 0,
                "nonzero_gradient_non_wildcard_tokens": 0,
            },
        )
        for token_id, token in seen_pairs:
            if token.op == PredicateOp.WILDCARD:
                continue
            observed += 1
            column_counts["observed_non_wildcard_tokens"] += 1
            is_nonzero = bool(float(gradient[token_id].norm().detach().cpu()) > 0.0)
            if is_nonzero:
                nonzero += 1
                column_counts["nonzero_gradient_non_wildcard_tokens"] += 1
            if token.op != PredicateOp.INV_FANOUT:
                ordinary_observed += 1
                if is_nonzero:
                    ordinary_nonzero += 1
    return {
        "mode": "embed",
        "observed_non_wildcard_tokens": observed,
        "nonzero_gradient_non_wildcard_tokens": nonzero,
        "ordinary_observed_non_wildcard_tokens": ordinary_observed,
        "ordinary_nonzero_gradient_non_wildcard_tokens": ordinary_nonzero,
        "by_column": by_column,
    }


def _two_slot_gradient_coverage(
    model: PredicateResMADE,
    token_rows: list[list[Any]],
    token_ids: Any,
    metadata: object,
) -> dict[str, Any]:
    operator_gradient = model.operator_embedding.weight.grad
    observed = 0
    nonzero = 0
    ordinary_observed = 0
    ordinary_nonzero = 0
    by_column: dict[str, dict[str, int]] = {}
    for column_index, column in enumerate(metadata.columns):
        value_gradient = model.value_embeddings[column_index].weight.grad
        column_counts = by_column.setdefault(
            column.name,
            {
                "observed_non_wildcard_tokens": 0,
                "nonzero_gradient_non_wildcard_tokens": 0,
                "observed_components": 0,
                "nonzero_gradient_components": 0,
            },
        )
        seen = {
            (
                tuple(int(value) for value in token_ids[row_index, column_index].detach().cpu().tolist()),
                token_rows[row_index][column_index],
            )
            for row_index in range(len(token_rows))
        }
        missing_value_id = int(model.config.value_bins_by_column[column_index] - 1)  # type: ignore[index]
        for encoded, token in seen:
            if token.op == PredicateOp.WILDCARD:
                continue
            observed += 1
            column_counts["observed_non_wildcard_tokens"] += 1
            token_nonzero = False
            op1, value1, op2, value2 = encoded
            components = [
                (operator_gradient, op1, False),
                (value_gradient, value1, value1 == missing_value_id),
                (operator_gradient, op2, False),
                (value_gradient, value2, value2 == missing_value_id),
            ]
            for gradient, component_id, skip in components:
                if skip or gradient is None:
                    continue
                column_counts["observed_components"] += 1
                is_nonzero = bool(float(gradient[component_id].norm().detach().cpu()) > 0.0)
                if is_nonzero:
                    token_nonzero = True
                    column_counts["nonzero_gradient_components"] += 1
            if token_nonzero:
                nonzero += 1
                column_counts["nonzero_gradient_non_wildcard_tokens"] += 1
            if token.op != PredicateOp.INV_FANOUT:
                ordinary_observed += 1
                if token_nonzero:
                    ordinary_nonzero += 1
    return {
        "mode": "two_slot",
        "observed_non_wildcard_tokens": observed,
        "nonzero_gradient_non_wildcard_tokens": nonzero,
        "ordinary_observed_non_wildcard_tokens": ordinary_observed,
        "ordinary_nonzero_gradient_non_wildcard_tokens": ordinary_nonzero,
        "by_column": by_column,
    }


def _compositional_gradient_coverage(
    model: PredicateResMADE,
    token_rows: list[list[Any]],
    token_ids: Any,
    metadata: object,
) -> dict[str, Any]:
    operator_gradient = model.operator_embedding.weight.grad
    special_gradient = model.special_embedding.weight.grad
    observed = 0
    component_observed = 0
    component_nonzero = 0
    literal_observed = 0
    literal_nonzero = 0
    ordinary_observed = 0
    ordinary_nonzero = 0
    by_column: dict[str, dict[str, int]] = {}
    for column_index, column in enumerate(metadata.columns):
        value_gradient = model.value_embeddings[column_index].weight.grad
        literal_gradient = None
        if getattr(model.config, "predicate_encoding_mode", "") == "hybrid":
            literal_gradient = model.literal_embeddings[column_index].weight.grad
        token_id_values = token_ids[:, column_index].detach().cpu().numpy()
        seen_pairs = {
            (int(token_id), token_rows[row_index][column_index])
            for row_index, token_id in enumerate(token_id_values)
        }
        column_counts = by_column.setdefault(
            column.name,
            {
                "observed_non_wildcard_tokens": 0,
                "nonzero_gradient_non_wildcard_tokens": 0,
                "observed_components": 0,
                "nonzero_gradient_components": 0,
                "observed_literal_specific_tokens": 0,
                "nonzero_gradient_literal_specific_tokens": 0,
            },
        )
        op_lookup = getattr(model, f"token_operator_ids_{column_index}").detach().cpu().numpy()
        value_lookup = getattr(model, f"token_value_ids_{column_index}").detach().cpu().numpy()
        upper_lookup = getattr(model, f"token_upper_ids_{column_index}").detach().cpu().numpy()
        special_lookup = getattr(model, f"token_special_ids_{column_index}").detach().cpu().numpy()
        missing_value_id = int(model.config.value_bins_by_column[column_index] - 1)  # type: ignore[index]
        for token_id, token in seen_pairs:
            if token.op == PredicateOp.WILDCARD:
                continue
            observed += 1
            column_counts["observed_non_wildcard_tokens"] += 1
            token_components = [
                ("operator", operator_gradient, int(op_lookup[token_id])),
                ("special", special_gradient, int(special_lookup[token_id])),
                ("value", value_gradient, int(value_lookup[token_id])),
                ("upper", value_gradient, int(upper_lookup[token_id])),
            ]
            nonzero_for_token = False
            if literal_gradient is not None:
                literal_observed += 1
                column_counts["observed_literal_specific_tokens"] += 1
                if bool(float(literal_gradient[token_id].norm().detach().cpu()) > 0.0):
                    literal_nonzero += 1
                    column_counts["nonzero_gradient_literal_specific_tokens"] += 1
                    nonzero_for_token = True
            for name, gradient, component_id in token_components:
                if gradient is None:
                    continue
                if name in {"value", "upper"} and component_id == missing_value_id:
                    continue
                component_observed += 1
                column_counts["observed_components"] += 1
                is_nonzero = bool(float(gradient[component_id].norm().detach().cpu()) > 0.0)
                if is_nonzero:
                    component_nonzero += 1
                    column_counts["nonzero_gradient_components"] += 1
                    nonzero_for_token = True
            if nonzero_for_token:
                column_counts["nonzero_gradient_non_wildcard_tokens"] += 1
            if token.op != PredicateOp.INV_FANOUT:
                ordinary_observed += 1
                if nonzero_for_token:
                    ordinary_nonzero += 1
    return {
        "mode": getattr(model.config, "predicate_encoding_mode", "compositional"),
        "observed_non_wildcard_tokens": observed,
        "nonzero_gradient_non_wildcard_tokens": sum(
            counts["nonzero_gradient_non_wildcard_tokens"]
            for counts in by_column.values()
        ),
        "ordinary_observed_non_wildcard_tokens": ordinary_observed,
        "ordinary_nonzero_gradient_non_wildcard_tokens": ordinary_nonzero,
        "observed_components": component_observed,
        "nonzero_gradient_components": component_nonzero,
        "observed_literal_specific_tokens": literal_observed,
        "nonzero_gradient_literal_specific_tokens": literal_nonzero,
        "by_column": by_column,
    }


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
    *,
    filename: str | None = None,
) -> Path:
    output_directory = Path(config.get("logging", {}).get("output_directory", "model/runs/resmade"))
    checkpoint_path = output_directory / (filename or f"checkpoint_step_{step}.pt")
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
