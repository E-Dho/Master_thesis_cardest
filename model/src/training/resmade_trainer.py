from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from model.src.config import resolve_device
from model.src.data.full_join_sampler import FullJoinBatch
from model.src.data.schema import ColumnKind
from model.src.data.trajectory_distinct import (
    TRAJECTORY_TARGET_SEMANTICS_VERSION,
    trajectory_base_measure_support,
)
from model.src.model.anpm import ANPMConfig
from model.src.model.checkpoint import save_resmade_checkpoint
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.model.resmade import TrajectoryDistinctConfig
from model.src.predicates.generation import (
    GeneratedTrainingContext,
    PredicateTrainingContextGenerator,
    predicate_context_diagnostics,
    literal_token_occurrences,
    literal_token_stats,
    token_coverage,
)
from model.src.predicates.operators import PredicateOp, PredicateToken
from model.src.predicates.torch_encoding import encode_tokens_tensor
from model.src.predicates.vocabulary import PredicateVocabularies, key_to_token
from model.src.predicates.vocabulary import (
    TWO_SLOT_OPERATOR_BINS,
    two_slot_binary_widths_by_column,
    two_slot_value_bins_by_column,
)
from model.src.training.losses import (
    cumulative_inverse_fanout_weights,
    effective_sample_size,
    importance_weights_for_generated_contexts,
    stable_combine_importance_and_inverse_weights,
    terminal_log_weights,
)
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
    early_stopping_summary: dict[str, Any]


@dataclass(frozen=True)
class TrainingStepResult:
    loss: float
    uniform_loss: float
    auxiliary_loss_unscaled: float
    auxiliary_loss_scaled: float
    fanout_effective_sample_size: dict[str, float]
    fanout_inv_only_effective_sample_size: dict[str, float]
    importance_weight_stats: dict[str, Any]
    original_column_losses: dict[str, float]
    factor_losses: dict[str, float]
    generated_contexts: int
    rejected_unsatisfied_contexts: int
    included_indicator_contradictions: int
    predicate_token_coverage: dict[str, dict[str, int]]
    literal_token_occurrences: dict[str, dict[str, dict[str, int]]]
    predicate_embedding_gradient_coverage: dict[str, Any]
    predicate_context_diagnostics: dict[str, Any]
    rare_auxiliary: dict[str, Any] | None = None
    trajectory_distinct: dict[str, Any] | None = None


@dataclass
class _RunningScalarStats:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    last: float | None = None

    def update(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.last = value

    def to_json_dict(self) -> dict[str, float | int | None]:
        return {
            "count": int(self.count),
            "mean": float(self.total / self.count) if self.count else None,
            "min": self.minimum,
            "max": self.maximum,
            "last": self.last,
        }


@dataclass
class _RunningAuxiliaryColumnStats:
    eligible_examples: int = 0
    inv_weight_sum: float = 0.0
    inv_weight_squared_sum: float = 0.0
    column_loss: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    scaled_column_loss: _RunningScalarStats = field(default_factory=_RunningScalarStats)

    def update(self, diagnostics: dict[str, Any]) -> None:
        self.eligible_examples += int(diagnostics.get("auxiliary_eligible_examples", 0))
        self.inv_weight_sum += float(diagnostics.get("auxiliary_inv_weight_sum", 0.0))
        self.inv_weight_squared_sum += float(
            diagnostics.get("auxiliary_inv_weight_squared_sum", 0.0)
        )
        self.column_loss.update(float(diagnostics.get("auxiliary_column_loss", 0.0)))
        self.scaled_column_loss.update(
            float(diagnostics.get("auxiliary_scaled_column_loss", 0.0))
        )

    def to_json_dict(self) -> dict[str, Any]:
        ess = (
            (self.inv_weight_sum * self.inv_weight_sum) / self.inv_weight_squared_sum
            if self.inv_weight_squared_sum > 0.0
            else 0.0
        )
        return {
            "auxiliary_eligible_examples": int(self.eligible_examples),
            "auxiliary_inv_weight_sum": float(self.inv_weight_sum),
            "auxiliary_inv_weight_squared_sum": float(self.inv_weight_squared_sum),
            "auxiliary_inv_ess": float(ess),
            "auxiliary_column_loss": self.column_loss.to_json_dict(),
            "auxiliary_scaled_column_loss": self.scaled_column_loss.to_json_dict(),
        }


@dataclass
class _RunningTrajectoryDistinctStats:
    attempted: int = 0
    targets_generated: int = 0
    targets_skipped: int = 0
    skipped_non_segment_measure: int = 0
    skipped_base_spatial_measure: int = 0
    skipped_unsupported_semantics: int = 0
    skipped_missing_provenance: int = 0
    target_failures: int = 0
    multiplicity_sum: float = 0.0
    multiplicity_min: int | None = None
    multiplicity_max: int | None = None
    multiplicity_eq_1: int = 0
    multiplicity_gt_1: int = 0
    multiplicity_ge_5: int = 0
    multiplicity_ge_10: int = 0
    multiplicity_reservoir: list[int] = field(default_factory=list)
    reservoir_limit: int = 4096
    targets: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    predictions: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    weighted_mse: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    unweighted_mse: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    weighted_ess: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    provider_seconds: _RunningScalarStats = field(default_factory=_RunningScalarStats)
    weighted_squared_error_sum: float = 0.0
    unweighted_squared_error_sum: float = 0.0
    traj_loss_weight_sum: float = 0.0
    traj_loss_weight_sq_sum: float = 0.0
    log_weight_sum: float = float("-inf")
    log_weight_sq_sum: float = float("-inf")
    log_weighted_squared_error_sum: float = float("-inf")
    target_sum: float = 0.0
    prediction_sum: float = 0.0

    def update(self, diagnostics: dict[str, Any]) -> None:
        if not diagnostics.get("enabled", False):
            return
        self.attempted += int(diagnostics.get("trajectory_targets_attempted", 0))
        self.targets_generated += int(diagnostics.get("trajectory_targets_generated", 0))
        self.targets_skipped += int(diagnostics.get("trajectory_targets_skipped", 0))
        self.skipped_non_segment_measure += int(
            diagnostics.get("trajectory_targets_skipped_non_segment_measure", 0)
        )
        self.skipped_base_spatial_measure += int(
            diagnostics.get("trajectory_targets_skipped_base_spatial_measure", 0)
        )
        self.skipped_unsupported_semantics += int(
            diagnostics.get("trajectory_targets_skipped_unsupported_semantics", 0)
        )
        self.skipped_missing_provenance += int(
            diagnostics.get("trajectory_targets_skipped_missing_provenance", 0)
        )
        self.target_failures += int(diagnostics.get("trajectory_target_failures", 0))
        for value in diagnostics.get("multiplicity_reservoir", ()):
            if len(self.multiplicity_reservoir) < self.reservoir_limit:
                self.multiplicity_reservoir.append(int(value))
        m_count = int(diagnostics.get("trajectory_targets_generated", 0))
        if m_count:
            self.multiplicity_sum += float(diagnostics.get("multiplicity_sum", 0.0))
            self.weighted_squared_error_sum += float(
                diagnostics.get("weighted_squared_error_sum", 0.0)
            )
            self.unweighted_squared_error_sum += float(
                diagnostics.get("unweighted_squared_error_sum", 0.0)
            )
            self.traj_loss_weight_sum += float(diagnostics.get("traj_loss_weight_sum", 0.0))
            self.traj_loss_weight_sq_sum += float(
                diagnostics.get("traj_loss_weight_sq_sum", 0.0)
            )
            for attr, key in [
                ("log_weight_sum", "traj_log_weight_sum"),
                ("log_weight_sq_sum", "traj_log_weight_sq_sum"),
                (
                    "log_weighted_squared_error_sum",
                    "traj_log_weighted_squared_error_sum",
                ),
            ]:
                value = diagnostics.get(key)
                if value is None:
                    continue
                log_value = float(value)
                if np.isnan(log_value) or np.isposinf(log_value):
                    raise ValueError(f"{key} must be finite or -inf")
                if np.isneginf(log_value) and attr != "log_weighted_squared_error_sum":
                    continue
                if np.isneginf(log_value) and np.isneginf(getattr(self, attr)):
                    setattr(self, attr, float("-inf"))
                else:
                    setattr(
                        self,
                        attr,
                        float(np.logaddexp(getattr(self, attr), log_value)),
                    )
            self.target_sum += float(diagnostics.get("target_sum", 0.0))
            self.prediction_sum += float(diagnostics.get("prediction_sum", 0.0))
            m_min = int(diagnostics.get("m_min", 0))
            m_max = int(diagnostics.get("m_max", 0))
            self.multiplicity_min = (
                m_min if self.multiplicity_min is None else min(self.multiplicity_min, m_min)
            )
            self.multiplicity_max = (
                m_max if self.multiplicity_max is None else max(self.multiplicity_max, m_max)
            )
            self.multiplicity_eq_1 += int(diagnostics.get("multiplicity_eq_1", 0))
            self.multiplicity_gt_1 += int(diagnostics.get("multiplicity_gt_1", 0))
            self.multiplicity_ge_5 += int(diagnostics.get("multiplicity_ge_5", 0))
            self.multiplicity_ge_10 += int(diagnostics.get("multiplicity_ge_10", 0))
        for source, key in [
            (self.targets, "target_mean"),
            (self.predictions, "prediction_mean"),
            (self.weighted_mse, "weighted_mse"),
            (self.unweighted_mse, "unweighted_mse"),
            (self.weighted_ess, "weighted_ess"),
            (self.provider_seconds, "traj_provider_seconds"),
        ]:
            value = diagnostics.get(key)
            if value is not None:
                source.update(float(value))

    def to_json_dict(self) -> dict[str, Any]:
        sample = np.asarray(self.multiplicity_reservoir, dtype=float)
        count = int(self.targets_generated)
        if count:
            m_stats = {
                "min": self.multiplicity_min,
                "mean": float(self.multiplicity_sum / count),
                "median": float(np.percentile(sample, 50)) if sample.size else None,
                "p90": float(np.percentile(sample, 90)) if sample.size else None,
                "p95": float(np.percentile(sample, 95)) if sample.size else None,
                "max": self.multiplicity_max,
                "fraction_m_equals_1": float(self.multiplicity_eq_1 / count),
                "fraction_m_gt_1": float(self.multiplicity_gt_1 / count),
                "fraction_m_ge_5": float(self.multiplicity_ge_5 / count),
                "fraction_m_ge_10": float(self.multiplicity_ge_10 / count),
                "percentiles_source": "bounded_reservoir",
            }
        else:
            m_stats = {
                "min": None,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "max": None,
                "fraction_m_equals_1": None,
                "fraction_m_gt_1": None,
                "fraction_m_ge_5": None,
                "fraction_m_ge_10": None,
            }
        warning = (
            "almost every trajectory target has m_t(Q)=1; the auxiliary head "
            "mostly learns traj_dedup_factor≈1"
            if m_stats["fraction_m_equals_1"] is not None
            and float(m_stats["fraction_m_equals_1"]) > 0.95
            else None
        )
        return {
            "enabled": self.targets_generated > 0 or self.targets_skipped > 0,
            "trajectory_targets_attempted": int(self.attempted),
            "trajectory_targets_generated": int(self.targets_generated),
            "trajectory_targets_skipped": int(self.targets_skipped),
            "trajectory_targets_skipped_non_segment_measure": int(
                self.skipped_non_segment_measure
            ),
            "trajectory_targets_skipped_base_spatial_measure": int(
                self.skipped_base_spatial_measure
            ),
            "trajectory_targets_skipped_unsupported_semantics": int(
                self.skipped_unsupported_semantics
            ),
            "trajectory_targets_skipped_missing_provenance": int(
                self.skipped_missing_provenance
            ),
            "trajectory_target_failures": int(self.target_failures),
            "fraction_targets_eligible": (
                float(self.targets_generated / self.attempted) if self.attempted else None
            ),
            "multiplicity": m_stats,
            "target": self.targets.to_json_dict(),
            "prediction": self.predictions.to_json_dict(),
            "weighted_mse": self.weighted_mse.to_json_dict(),
            "unweighted_mse": self.unweighted_mse.to_json_dict(),
            "global_weighted_mse": (
                float(np.exp(self.log_weighted_squared_error_sum - self.log_weight_sum))
                if np.isfinite(self.log_weight_sum)
                and (
                    np.isfinite(self.log_weighted_squared_error_sum)
                    or np.isneginf(self.log_weighted_squared_error_sum)
                )
                else None
            ),
            "global_unweighted_mse": (
                float(self.unweighted_squared_error_sum / self.targets_generated)
                if self.targets_generated
                else None
            ),
            "global_target_mean": (
                float(self.target_sum / self.targets_generated)
                if self.targets_generated
                else None
            ),
            "global_prediction_mean": (
                float(self.prediction_sum / self.targets_generated)
                if self.targets_generated
                else None
            ),
            "global_weighted_ess": (
                float(np.exp(2.0 * self.log_weight_sum - self.log_weight_sq_sum))
                if np.isfinite(self.log_weight_sum) and np.isfinite(self.log_weight_sq_sum)
                else None
            ),
            "traj_loss_weight_sum": float(self.traj_loss_weight_sum),
            "traj_loss_weight_sq_sum": float(self.traj_loss_weight_sq_sum),
            "traj_log_weight_sum": (
                float(self.log_weight_sum) if np.isfinite(self.log_weight_sum) else None
            ),
            "traj_log_weight_sq_sum": (
                float(self.log_weight_sq_sum)
                if np.isfinite(self.log_weight_sq_sum)
                else None
            ),
            "traj_log_weighted_squared_error_sum": (
                float(self.log_weighted_squared_error_sum)
                if np.isfinite(self.log_weighted_squared_error_sum)
                else None
            ),
            "weighted_ess": self.weighted_ess.to_json_dict(),
            "provider_seconds": self.provider_seconds.to_json_dict(),
            "warning": warning,
        }


def main_predicate_rng_seed(generator_seed: int, global_step: int) -> int:
    return int(generator_seed) + int(global_step)


def rare_row_rng_seed(training_seed: int, global_step: int) -> int:
    return int(training_seed) + 1_000_000 + int(global_step)


def rare_predicate_rng_seed(generator_seed: int, global_step: int) -> int:
    return int(generator_seed) + 2_000_000 + int(global_step)


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
            column_kinds=tuple(str(column.kind.value) for column in metadata.columns),
            hidden_sizes=tuple(model_config.get("hidden_sizes", [128, 128])),
            residual_connections=bool(model_config.get("residual_connections", True)),
            direct_io_connections=bool(model_config.get("direct_io_connections", True)),
            direct_io_source_kinds=tuple(
                str(kind)
                for kind in model_config.get(
                    "direct_io_source_kinds",
                    ["data", "indicator", "fanout"],
                )
            ),
            direct_io_destination_kinds=tuple(
                str(kind)
                for kind in model_config.get(
                    "direct_io_destination_kinds",
                    ["data", "indicator", "fanout"],
                )
            ),
            activation=str(model_config.get("activation", "relu")),
            input_encoding=str(model_config.get("input_encoding", "embed")),
            output_encoding=str(model_config.get("output_encoding", "one_hot")),
            output_embedding_size=int(model_config.get("output_embedding_size", 64)),
            output_embeddings_tied=bool(model_config.get("output_embeddings_tied", False)),
            embedding_size=int(model_config.get("embedding_size", 16)),
            predicate_encoding_mode=encoding_mode,
            operator_embedding_size=int(predicate_encoding.get("operator_embedding_size", 8)),
            value_embedding_size=int(predicate_encoding.get("value_embedding_size", 32)),
            special_embedding_size=int(predicate_encoding.get("special_embedding_size", 8)),
            merge_hidden_size=int(predicate_encoding.get("merge_hidden_size", 64)),
            multi_predicate_merge=str(predicate_encoding.get("multi_predicate_merge", "sum")),
            **compositional_features,
            residual_dropout=float(model_config.get("residual_dropout", 0.0)),
            fixed_ordering=bool(model_config.get("fixed_ordering", True)),
            output_head_specs=output_head_specs,
            factorization_plan=plan,
            anpm_config=ANPMConfig.from_dict(config.get("anpm", {})),
            trajectory_distinct_config=TrajectoryDistinctConfig.from_dict(
                config.get("trajectory_distinct", {})
            ),
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
    include_native_ranges = encoding_mode not in {
        "two_slot",
        "two_slot_categorical_legacy",
        "two_slot_binary_duet",
    } and bool(predicate_generation.get("enable_native_range_tokens", False))
    native_range_max_domain_size = int(
        predicate_generation.get("native_range_max_domain_size", 512)
    )
    return PredicateVocabularies.from_metadata(
        metadata,
        include_native_ranges=include_native_ranges,
        native_range_max_domain_size=native_range_max_domain_size,
        encoding_mode=(
            "two_slot_binary_duet"
            if encoding_mode == "two_slot_binary_duet"
            else (
                "two_slot"
                if encoding_mode in {"two_slot", "two_slot_categorical_legacy"}
                else "categorical"
            )
        ),
    )


def _predicate_encoding_feature_tables(
    metadata: object,
    vocabularies: PredicateVocabularies,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode in {"two_slot", "two_slot_categorical_legacy"}:
        return {
            "value_bins_by_column": two_slot_value_bins_by_column(metadata),
            "operator_bins": TWO_SLOT_OPERATOR_BINS,
        }
    if mode == "two_slot_binary_duet":
        return {
            "value_bins_by_column": two_slot_value_bins_by_column(metadata),
            "binary_value_widths_by_column": two_slot_binary_widths_by_column(metadata),
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
    first_uniform_loss: float | None = None
    first_auxiliary_loss_unscaled: float | None = None
    first_auxiliary_loss_scaled: float | None = None
    last_loss = float("nan")
    last_uniform_loss = float("nan")
    last_auxiliary_loss_unscaled = 0.0
    last_auxiliary_loss_scaled = 0.0
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
    fanout_inv_only_ess_values: dict[str, list[float]] = {
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
    last_rare_auxiliary_stats: dict[str, Any] = {"enabled": False}
    last_trajectory_distinct_stats: dict[str, Any] = {"enabled": False}
    aggregate_trajectory_distinct_stats = _RunningTrajectoryDistinctStats()
    total_rare_auxiliary_rows = 0
    total_rare_auxiliary_rejected_contexts = 0
    total_rare_auxiliary_indicator_contradictions = 0
    aggregate_rare_auxiliary_selected_strata: dict[str, int] = {}
    aggregate_rare_auxiliary_forced_ops: dict[str, int] = {}
    aggregate_rare_auxiliary_exact_support_events: dict[str, int] = {}
    aggregate_rare_auxiliary_downstream_eligible: dict[str, int] = {}
    aggregate_rare_auxiliary_contexts_by_stratum: dict[str, int] = {}
    total_rare_auxiliary_contexts = 0
    aggregate_rare_auxiliary_column_stats: dict[str, _RunningAuxiliaryColumnStats] = {}
    aggregate_rare_auxiliary_fanout_signatures: dict[str, int] = {}
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
    early_stopping_patience_steps = int(
        training.get(
            "early_stopping_patience_steps",
            training.get("early_stopping_patience", 0),
        )
        or 0
    )
    early_stopping_min_delta = float(training.get("early_stopping_min_delta", 0.0) or 0.0)
    early_stopping_enabled = early_stopping_patience_steps > 0
    early_stopping_monitor = (
        selection_metric if validation_enabled else "loss"
    )
    early_stopping_best_metric: float | None = None
    early_stopping_best_step: int | None = None
    early_stopped = False
    early_stopping_stop_step: int | None = None
    early_stopping_reason: str | None = None
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
                sample_source=sample_source,
                global_step=global_step,
            )
            loss = step_result.loss
            last_loss = loss
            last_uniform_loss = step_result.uniform_loss
            last_auxiliary_loss_unscaled = step_result.auxiliary_loss_unscaled
            last_auxiliary_loss_scaled = step_result.auxiliary_loss_scaled
            if first_loss is None:
                first_loss = loss
                first_uniform_loss = step_result.uniform_loss
                first_auxiliary_loss_unscaled = step_result.auxiliary_loss_unscaled
                first_auxiliary_loss_scaled = step_result.auxiliary_loss_scaled
            global_step += 1
            for fanout_name, fanout_ess in step_result.fanout_effective_sample_size.items():
                fanout_ess_values[fanout_name].append(float(fanout_ess))
            for fanout_name, fanout_ess in step_result.fanout_inv_only_effective_sample_size.items():
                fanout_inv_only_ess_values[fanout_name].append(float(fanout_ess))
            last_original_column_losses = step_result.original_column_losses
            last_factor_losses = step_result.factor_losses
            total_generated_contexts += step_result.generated_contexts
            total_rejected_contexts += step_result.rejected_unsatisfied_contexts
            total_indicator_contradictions += step_result.included_indicator_contradictions
            _merge_coverage(aggregate_token_coverage, step_result.predicate_token_coverage)
            _merge_literal_occurrences(aggregate_literal_occurrences, step_result.literal_token_occurrences)
            last_gradient_coverage = step_result.predicate_embedding_gradient_coverage
            last_context_diagnostics = step_result.predicate_context_diagnostics
            last_rare_auxiliary_stats = step_result.rare_auxiliary or {"enabled": False}
            last_trajectory_distinct_stats = (
                step_result.trajectory_distinct or {"enabled": False}
            )
            aggregate_trajectory_distinct_stats.update(last_trajectory_distinct_stats)
            if last_rare_auxiliary_stats.get("enabled"):
                total_rare_auxiliary_rows += int(
                    last_rare_auxiliary_stats.get("rare_rows_sampled", 0)
                )
                total_rare_auxiliary_rejected_contexts += int(
                    last_rare_auxiliary_stats.get("auxiliary_contexts_rejected", 0)
                )
                total_rare_auxiliary_indicator_contradictions += int(
                    last_rare_auxiliary_stats.get("included_indicator_contradictions", 0)
                )
                _merge_flat_counts(
                    aggregate_rare_auxiliary_selected_strata,
                    last_rare_auxiliary_stats.get("selected_stratum_counts", {}),
                )
                _merge_flat_counts(
                    aggregate_rare_auxiliary_forced_ops,
                    last_rare_auxiliary_stats.get(
                        "forced_auxiliary_predicate_operator_counts", {}
                    ),
                )
                for stratum_id, stats in last_rare_auxiliary_stats.get("per_stratum", {}).items():
                    aggregate_rare_auxiliary_contexts_by_stratum[stratum_id] = (
                        int(aggregate_rare_auxiliary_contexts_by_stratum.get(stratum_id, 0))
                        + int(stats.get("rare_context_count", 0))
                    )
                    aggregate_rare_auxiliary_exact_support_events[stratum_id] = (
                        int(aggregate_rare_auxiliary_exact_support_events.get(stratum_id, 0))
                        + int(stats.get("generated_exact_support_event_count", 0))
                    )
                    aggregate_rare_auxiliary_downstream_eligible[stratum_id] = (
                        int(aggregate_rare_auxiliary_downstream_eligible.get(stratum_id, 0))
                        + int(stats.get("downstream_eligible_head_context_count", 0))
                    )
                total_rare_auxiliary_contexts += int(
                    last_rare_auxiliary_stats.get("rare_context_count", 0)
                )
                for column_name, diagnostics in last_rare_auxiliary_stats.get(
                    "auxiliary_column_diagnostics", {}
                ).items():
                    aggregate_rare_auxiliary_column_stats.setdefault(
                        column_name,
                        _RunningAuxiliaryColumnStats(),
                    ).update(diagnostics)
                _merge_flat_counts(
                    aggregate_rare_auxiliary_fanout_signatures,
                    last_rare_auxiliary_stats.get(
                        "common_auxiliary_fanout_token_signatures", {}
                    ),
                )
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
                    "uniform_loss": step_result.uniform_loss,
                    "auxiliary_loss_unscaled": step_result.auxiliary_loss_unscaled,
                    "auxiliary_loss_scaled": step_result.auxiliary_loss_scaled,
                    "fanout_effective_sample_size": step_result.fanout_effective_sample_size,
                    "fanout_inv_only_effective_sample_size": (
                        step_result.fanout_inv_only_effective_sample_size
                    ),
                    "importance_weight_stats": step_result.importance_weight_stats,
                    "original_column_losses": step_result.original_column_losses,
                    "factor_losses": step_result.factor_losses,
                    "predicate_token_coverage": aggregate_token_coverage,
                    "predicate_literal_token_stats": literal_token_stats(
                        aggregate_literal_occurrences,
                        metadata,
                    ),
                    "predicate_context_diagnostics": last_context_diagnostics,
                    "predicate_embedding_gradient_coverage": last_gradient_coverage,
                    "rare_auxiliary": last_rare_auxiliary_stats,
                    "trajectory_distinct": last_trajectory_distinct_stats,
                }
                early_stopping_monitor_value: float | None = None
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
                    raw_selected_value = validation_metrics.get(selection_metric)
                    if raw_selected_value is None:
                        if (
                            selection_metric == "validation_traj_weighted_mse"
                            and bool(config.get("trajectory_distinct", {}).get("enabled", False))
                        ):
                            raise ValueError(
                                "validation selection metric validation_traj_weighted_mse "
                                "is unavailable because no eligible trajectory targets were generated"
                            )
                        raise ValueError(
                            f"validation selection metric {selection_metric} is unavailable"
                        )
                    selected_value = float(raw_selected_value)
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
                    early_stopping_monitor_value = selected_value
                elif not validation_enabled:
                    early_stopping_monitor_value = float(loss)
                if early_stopping_enabled and early_stopping_monitor_value is not None:
                    if early_stopping_best_metric is None:
                        improved_for_stop = True
                    elif minimize_selection:
                        improved_for_stop = (
                            early_stopping_monitor_value
                            < early_stopping_best_metric - early_stopping_min_delta
                        )
                    else:
                        improved_for_stop = (
                            early_stopping_monitor_value
                            > early_stopping_best_metric + early_stopping_min_delta
                        )
                    if improved_for_stop:
                        early_stopping_best_metric = early_stopping_monitor_value
                        early_stopping_best_step = global_step
                    steps_since_best = (
                        0
                        if early_stopping_best_step is None
                        else global_step - early_stopping_best_step
                    )
                    early_stopped = steps_since_best >= early_stopping_patience_steps
                    if early_stopped:
                        early_stopping_stop_step = global_step
                        early_stopping_reason = (
                            f"{early_stopping_monitor} did not improve for "
                            f"{steps_since_best} optimizer steps"
                        )
                    metrics_payload.update(
                        {
                            "early_stopping_enabled": early_stopping_enabled,
                            "early_stopping_monitor": early_stopping_monitor,
                            "early_stopping_monitor_value": early_stopping_monitor_value,
                            "early_stopping_best_metric": early_stopping_best_metric,
                            "early_stopping_best_step": early_stopping_best_step,
                            "early_stopping_steps_since_best": steps_since_best,
                            "early_stopping_patience_steps": early_stopping_patience_steps,
                            "early_stopping_min_delta": early_stopping_min_delta,
                            "early_stopping_should_stop": early_stopped,
                        }
                    )
                _append_metrics(metrics_path, metrics_payload)
                if early_stopped:
                    break
        if early_stopped:
            break
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
    fanout_inv_only_ess_summary = {
        fanout_name: {
            "last": values[-1],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for fanout_name, values in fanout_inv_only_ess_values.items()
        if values
    }
    importance_summary = (
        sample_source.importance_sampling_summary(
            actual_optimizer_steps=global_step,
            early_stopped=early_stopped,
            early_stopping_stop_step=early_stopping_stop_step,
        )
        if hasattr(sample_source, "importance_sampling_summary")
        else {"enabled": False}
    )
    rare_support_summary = (
        sample_source.rare_support_summary(actual_optimizer_steps=global_step)
        if hasattr(sample_source, "rare_support_summary")
        else {"enabled": False}
    )
    output_embedding_parameter_count = int(
        sum(parameter.numel() for parameter in getattr(model, "output_embeddings").parameters())
    )
    direct_io_parameter_count = int(
        0
        if getattr(model, "direct_io_layer", None) is None
        else sum(parameter.numel() for parameter in model.direct_io_layer.parameters())
    )
    binary_widths = tuple(
        int(width)
        for width in getattr(model.config, "binary_value_widths_by_column", ()) or ()
    )
    binary_literal_input_dimensions = int(sum(binary_widths))
    old_categorical_literal_embedding_params = int(
        sum(column.domain_size * int(model.config.value_embedding_size) for column in metadata.columns)
    )
    high_cardinality_columns = []
    for column_index, column in enumerate(metadata.columns):
        if column.kind.value != "data" or column.domain_size < 2048:
            continue
        factorization = metadata.factorization_plan.factorization_for_column(column_index)
        high_cardinality_columns.append(
            {
                "column": column.name,
                "domain_size": int(column.domain_size),
                "binary_width": (
                    int(binary_widths[column_index]) if binary_widths else None
                ),
                "old_categorical_input_embedding_params": int(
                    column.domain_size * int(model.config.value_embedding_size)
                ),
                "new_binary_input_dimensions": (
                    int(binary_widths[column_index]) if binary_widths else None
                ),
                "factor_domains": (
                    None if factorization is None else tuple(factorization.factor_domains)
                ),
                "output_embedding_dimensions": (
                    None
                    if getattr(model.config, "output_encoding", "one_hot") != "embed"
                    else [
                        [
                            int(metadata.factorization_plan.output_head_specs[head_index].domain_size),
                            int(model.config.output_embedding_size),
                        ]
                        for head_index in metadata.factorization_plan.output_heads_for_column(
                            column_index
                        )
                    ]
                ),
            }
        )
    summary = {
        "checkpoint": str(checkpoint_path),
        "parameter_count": model.parameter_count(),
        "parameter_size_bytes": parameter_size_bytes,
        "backbone_parameter_count": backbone_parameter_count,
        "anpm_parameter_count": anpm_parameter_count,
        "first_loss": float(first_loss if first_loss is not None else float("nan")),
        "last_loss": last_loss,
        "first_uniform_loss": float(
            first_uniform_loss if first_uniform_loss is not None else float("nan")
        ),
        "last_uniform_loss": last_uniform_loss,
        "first_auxiliary_loss_unscaled": float(first_auxiliary_loss_unscaled or 0.0),
        "last_auxiliary_loss_unscaled": last_auxiliary_loss_unscaled,
        "first_auxiliary_loss_scaled": float(first_auxiliary_loss_scaled or 0.0),
        "last_auxiliary_loss_scaled": last_auxiliary_loss_scaled,
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
        "fanout_inv_only_effective_sample_size": fanout_inv_only_ess_summary,
        "importance_sampling": importance_summary,
        "rare_support": rare_support_summary,
        "rare_auxiliary": {
            "enabled": bool(config.get("rare_auxiliary", {}).get("enabled", False)),
            "batch_size": int(config.get("rare_auxiliary", {}).get("batch_size", 0) or 0),
            "beta": float(config.get("rare_auxiliary", {}).get("beta", 0.0) or 0.0),
            "total_rare_rows_sampled": int(total_rare_auxiliary_rows),
            "total_rare_context_count": int(total_rare_auxiliary_contexts),
            "auxiliary_contexts_rejected": int(total_rare_auxiliary_rejected_contexts),
            "included_indicator_contradictions": int(
                total_rare_auxiliary_indicator_contradictions
            ),
            "selected_stratum_counts": aggregate_rare_auxiliary_selected_strata,
            "forced_auxiliary_predicate_operator_counts": (
                aggregate_rare_auxiliary_forced_ops
            ),
            "exact_support_event_counts_by_stratum": (
                aggregate_rare_auxiliary_exact_support_events
            ),
            "rare_context_count_by_stratum": aggregate_rare_auxiliary_contexts_by_stratum,
            "downstream_eligible_head_context_count_by_stratum": (
                aggregate_rare_auxiliary_downstream_eligible
            ),
            "whole_run_auxiliary_column_diagnostics": {
                column_name: stats.to_json_dict()
                for column_name, stats in aggregate_rare_auxiliary_column_stats.items()
            },
            "whole_run_auxiliary_fanout_token_signatures": dict(
                sorted(
                    aggregate_rare_auxiliary_fanout_signatures.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "last_step": last_rare_auxiliary_stats,
        },
        "trajectory_distinct": {
            **aggregate_trajectory_distinct_stats.to_json_dict(),
            "configuration": dict(config.get("trajectory_distinct", {})),
            "tuple_measure_correction": "applied",
            "static_fanout_correction": "applied",
            "query_generator_G_Q_given_s_correction": "not_applied_single_anchor_ablation",
            "target_semantics_version": TRAJECTORY_TARGET_SEMANTICS_VERSION,
            "last_step": last_trajectory_distinct_stats,
        },
        "output_width_original": output_width_original,
        "output_width_factorized": output_width_factorized,
        "predicate_encoding_mode": getattr(
            model.config,
            "predicate_encoding_mode",
            "categorical_legacy",
        ),
        "total_binary_literal_input_dimensions": binary_literal_input_dimensions,
        "parameters_saved_vs_categorical_literal_embeddings": (
            old_categorical_literal_embedding_params
            - _module_parameter_count(getattr(model, "value_embeddings", None))
        ),
        "direct_io_parameter_count": direct_io_parameter_count,
        "direct_io_source_kinds": tuple(getattr(model.config, "direct_io_source_kinds", ())),
        "direct_io_destination_kinds": tuple(
            getattr(model.config, "direct_io_destination_kinds", ())
        ),
        "output_encoding": getattr(model.config, "output_encoding", "one_hot"),
        "output_embedding_parameter_count": output_embedding_parameter_count,
        "backbone_output_width": int(getattr(model, "output_width", 0)),
        "anpm_latent_dimension": (
            int(getattr(model.config, "output_embedding_size", 0))
            if getattr(model.config, "output_encoding", "one_hot") == "embed"
            else None
        ),
        "high_cardinality_binary_columns": high_cardinality_columns,
        "predicate_vocabulary_metadata": vocabularies.metadata_size_diagnostics(),
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
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "maximum_configured_steps": int(training["steps_per_epoch"]) * int(training["epochs"]),
            "actual_optimizer_steps": global_step,
            "monitor": early_stopping_monitor if early_stopping_enabled else None,
            "patience_steps": (
                early_stopping_patience_steps if early_stopping_enabled else None
            ),
            "min_delta": early_stopping_min_delta if early_stopping_enabled else None,
            "minimize": minimize_selection if early_stopping_enabled else None,
            "best_metric": early_stopping_best_metric,
            "best_step": early_stopping_best_step,
            "stopped": early_stopped,
            "stop_step": early_stopping_stop_step,
            "reason": early_stopping_reason,
        },
    }
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
        early_stopping_summary=summary["early_stopping"],
    )


def _train_one_batch(
    model: PredicateResMADE,
    optimizer: object,
    batch: FullJoinBatch,
    metadata: object,
    vocabularies: PredicateVocabularies,
    config: dict[str, Any],
    device: str,
    *,
    sample_source: object | None = None,
    global_step: int,
) -> TrainingStepResult:
    import torch

    predicate_config = config.get("predicate_generation", {})
    generator_seed = int(predicate_config.get("seed", config["training"].get("seed", 0)))
    rng = np.random.default_rng(main_predicate_rng_seed(generator_seed, global_step))
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
    context_diagnostics["predicate_probability_configuration"] = (
        context_generator.probability_diagnostics()
    )
    context_diagnostics["table_subset_sampling_mode"] = (
        context_generator.table_subset_sampling
    )
    context_diagnostics["upstream_neurocard_table_dropout_probability_law"] = (
        "num_dropped_tables ~ Uniform{1,...,num_tables-1}; "
        "P(drop table | num_dropped_tables)=num_dropped_tables/num_tables; "
        "primary/root table forced included"
    )
    context_diagnostics["rooted_adjustment_applied"] = (
        context_generator.table_subset_sampling == "neurocard_table_dropout_rooted"
    )
    inv_only_weights = cumulative_inverse_fanout_weights(
        target_rows,
        token_rows,
        metadata,
        compute_in_log_space=bool(config["fanout"].get("compute_weights_in_log_space", True)),
    )
    weights = inv_only_weights
    importance_stats: dict[str, Any] = {"enabled": False}
    if batch.importance_weights is not None:
        rho = importance_weights_for_generated_contexts(
            batch.importance_weights,
            target_rows.shape[0],
            generation_stats,
        )
        if np.any(rho <= 0.0) or not np.all(np.isfinite(rho)):
            raise ValueError("importance weights rho must be finite and positive")
        # p/q corrects tuple-sampling bias; INV products below correct
        # query-measure fanout potentials. They are intentionally multiplied.
        # We combine them in log space and subtract a per-head constant before
        # exponentiation; normalized WCE is invariant to that common scaling.
        weights = stable_combine_importance_and_inverse_weights(inv_only_weights, rho)
        importance_stats = {
            "enabled": True,
            "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)),
            "rho_mean": float(np.mean(rho)),
            "rho_sum": float(np.sum(rho)),
            "rho_sum_squared": float(np.dot(rho, rho)),
            "rho_ess": effective_sample_size(rho),
        }
        update_importance_context_statistics = getattr(
            sample_source,
            "update_importance_context_statistics",
            None,
        )
        if update_importance_context_statistics is not None:
            # ``weights`` above are batch/head-rescaled for stable normalized WCE.
            # Long-running ESS diagnostics need true cross-batch relative weights,
            # so pass rho and INV separately and let the diagnostic layer combine
            # log(rho)+log(INV) without the per-batch rescaling constant.
            update_importance_context_statistics(
                generation_stats=generation_stats,
                token_rows=token_rows,
                inv_only_weights=inv_only_weights,
                rho=rho,
                batch_metadata=batch.importance_metadata,
            )
    token_ids = encode_tokens_tensor(token_rows, vocabularies, device=device)
    targets = torch.tensor(target_rows, dtype=torch.long, device=device)
    head_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    model_outputs = model.forward_with_auxiliary(token_ids)
    logits = model_outputs.ar_outputs
    split_head_outputs = model.split_head_outputs(logits)
    output_embeddings = (
        [embedding.weight for embedding in model.output_embeddings]
        if getattr(model.config, "output_encoding", "one_hot") == "embed"
        else None
    )
    breakdown = torch_weighted_per_head_cross_entropy(
        logits,
        targets,
        head_weights,
        metadata,
        anpm_decoders=getattr(model, "anpm_decoders", None),
        split_head_outputs=split_head_outputs,
        output_embeddings=output_embeddings,
        head_loss_reduction=str(config["training"].get("head_loss_reduction", "mean")),
        mask_invalid_factor_combinations=bool(
            config.get("anpm", {}).get("mask_invalid_combinations", True)
        )
    )
    total_loss = breakdown.total_loss
    trajectory_distinct_stats: dict[str, Any] = {"enabled": False}
    trajectory_config = TrajectoryDistinctConfig.from_dict(
        config.get("trajectory_distinct", {})
    )
    if trajectory_config.enabled and trajectory_config.loss_weight > 0.0:
        if model_outputs.traj_dedup_factor is None:
            raise ValueError("trajectory_distinct.enabled requires model trajectory head")
        (
            traj_loss,
            trajectory_distinct_stats,
        ) = trajectory_dedup_loss_for_batch(
            predictions=model_outputs.traj_dedup_factor,
            batch=batch,
            contexts=contexts,
            target_rows=target_rows,
            token_rows=token_rows,
            generation_stats=generation_stats,
            metadata=metadata,
            config=config,
            sample_source=sample_source,
            device=device,
        )
        total_loss = total_loss + trajectory_config.loss_weight * traj_loss
    rare_auxiliary_stats: dict[str, Any] = {"enabled": False}
    rare_token_rows: list[list[PredicateToken]] = []
    rare_token_ids = None
    rare_config = dict(config.get("rare_auxiliary", {}))
    rare_enabled = bool(rare_config.get("enabled", False))
    rare_beta = float(rare_config.get("beta", 0.0) or 0.0)
    if rare_enabled and rare_beta > 0.0:
        rare_batch_size = int(rare_config.get("batch_size", 0) or 0)
        if rare_batch_size <= 0:
            raise ValueError("rare_auxiliary.batch_size must be positive when enabled")
        rare_batches = getattr(sample_source, "rare_batches", None)
        if rare_batches is None:
            raise ValueError("rare_auxiliary.enabled requires a rare_support sample source")
        rare_batch = rare_batches(
            rare_batch_size,
            seed=rare_row_rng_seed(int(config["training"].get("seed", 0)), global_step),
        )
        selected_strata = tuple(rare_batch.importance_metadata["selected_strata"])  # type: ignore[index]
        rare_rng = np.random.default_rng(rare_predicate_rng_seed(generator_seed, global_step))
        rare_contexts, rare_target_rows, rare_generation_stats = (
            context_generator.generate_forced_stratum_batch(
                encoded_rows=rare_batch.encoded_values,
                metadata=metadata,
                strata=selected_strata,
                rng=rare_rng,
                debug_allow_row_dependent_native_range_tail=bool(
                    rare_config.get("debug_allow_row_dependent_native_range_tail", False)
                ),
            )
        )
        rare_token_rows = [list(context.tokens) for context in rare_contexts]
        rare_inv_only_weights = cumulative_inverse_fanout_weights(
            rare_target_rows,
            rare_token_rows,
            metadata,
            compute_in_log_space=bool(config["fanout"].get("compute_weights_in_log_space", True)),
        )
        rare_source_indices = np.asarray(
            rare_generation_stats.source_row_indices,
            dtype=int,
        )
        rare_selected_column_indices = np.asarray(
            rare_batch.importance_metadata["selected_stratum_column_index"],  # type: ignore[index]
            dtype=int,
        )[rare_source_indices]
        eligibility = auxiliary_eligibility_mask(
            metadata,
            rare_selected_column_indices,
            row_count=rare_target_rows.shape[0],
        )
        rare_weights = rare_inv_only_weights * eligibility.astype(float)
        rare_token_ids = encode_tokens_tensor(rare_token_rows, vocabularies, device=device)
        rare_targets = torch.tensor(rare_target_rows, dtype=torch.long, device=device)
        rare_head_weights = torch.tensor(rare_weights, dtype=torch.float32, device=device)
        rare_logits = model(rare_token_ids)
        rare_split_head_outputs = model.split_head_outputs(rare_logits)
        rare_breakdown = torch_weighted_per_head_cross_entropy(
            rare_logits,
            rare_targets,
            rare_head_weights,
            metadata,
            anpm_decoders=getattr(model, "anpm_decoders", None),
            split_head_outputs=rare_split_head_outputs,
            output_embeddings=output_embeddings,
            head_loss_reduction=str(config["training"].get("head_loss_reduction", "mean")),
            mask_invalid_factor_combinations=bool(
                config.get("anpm", {}).get("mask_invalid_combinations", True)
            )
        )
        total_loss = total_loss + rare_beta * rare_breakdown.total_loss
        rare_auxiliary_stats = rare_auxiliary_diagnostics(
            metadata=metadata,
            beta=rare_beta,
            rare_batch=rare_batch,
            contexts=rare_contexts,
            generation_stats=rare_generation_stats,
            selected_strata=selected_strata,
            selected_column_indices=rare_selected_column_indices,
            eligibility=eligibility,
            inv_only_weights=rare_inv_only_weights,
            rare_breakdown=rare_breakdown,
        )
    total_loss.backward()
    all_token_rows = token_rows + rare_token_rows
    all_token_ids = (
        torch.cat([token_ids, rare_token_ids], dim=0)
        if rare_token_ids is not None
        else token_ids
    )
    gradient_coverage = _predicate_embedding_gradient_coverage(
        model,
        all_token_rows,
        all_token_ids,
        metadata,
    )
    clip_norm = config["training"].get("gradient_clip_norm")
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
    optimizer.step()
    if not torch.isfinite(total_loss):
        raise ValueError("training loss became non-finite")
    fanout_effective_sample_sizes = {}
    fanout_inv_only_effective_sample_sizes = {}
    for fanout_index in metadata.fanout_indices():
        fanout_ess = effective_sample_size(weights[:, fanout_index])
        if fanout_ess <= 0:
            raise ValueError("fanout effective sample size is non-positive")
        fanout_effective_sample_sizes[metadata.columns[fanout_index].name] = float(fanout_ess)
        fanout_inv_only_effective_sample_sizes[metadata.columns[fanout_index].name] = float(
            effective_sample_size(inv_only_weights[:, fanout_index])
        )
    return TrainingStepResult(
        loss=float(total_loss.detach().cpu()),
        uniform_loss=float(breakdown.total_loss.detach().cpu()),
        auxiliary_loss_unscaled=float(
            rare_auxiliary_stats.get("auxiliary_loss_unscaled", 0.0)
        ),
        auxiliary_loss_scaled=float(
            rare_auxiliary_stats.get("auxiliary_loss_scaled", 0.0)
        ),
        fanout_effective_sample_size=fanout_effective_sample_sizes,
        fanout_inv_only_effective_sample_size=fanout_inv_only_effective_sample_sizes,
        importance_weight_stats=importance_stats,
        original_column_losses=breakdown.original_column_losses,
        factor_losses=breakdown.factor_losses,
        generated_contexts=generation_stats.generated_contexts,
        rejected_unsatisfied_contexts=generation_stats.rejected_unsatisfied_contexts,
        included_indicator_contradictions=generation_stats.included_indicator_contradictions,
        predicate_token_coverage=coverage,
        literal_token_occurrences=literal_occurrences,
        predicate_context_diagnostics=context_diagnostics,
        predicate_embedding_gradient_coverage=gradient_coverage,
        rare_auxiliary=rare_auxiliary_stats,
        trajectory_distinct=trajectory_distinct_stats,
    )


def trajectory_dedup_loss_for_batch(
    *,
    predictions: object,
    batch: FullJoinBatch,
    contexts: list[GeneratedTrainingContext],
    target_rows: np.ndarray,
    token_rows: list[list[PredicateToken]],
    generation_stats: object,
    metadata: object,
    config: dict[str, Any],
    sample_source: object | None,
    device: str,
) -> tuple[object, dict[str, Any]]:
    """Compute weighted MSE for the query-only trajectory deduplication factor."""

    import torch

    provider = getattr(sample_source, "trajectory_multiplicity_provider", None)
    if provider is None:
        raise ValueError(
            "trajectory_distinct.enabled requires sample_source.trajectory_multiplicity_provider"
        )
    if batch.trajectory_ids is None:
        zero_loss = predictions.sum() * 0.0
        return zero_loss, {
            "enabled": True,
            "trajectory_targets_attempted": int(len(contexts)),
            "trajectory_targets_generated": 0,
            "trajectory_targets_skipped": int(len(contexts)),
            "trajectory_targets_skipped_missing_provenance": int(len(contexts)),
            "trajectory_targets_skipped_non_segment_measure": 0,
            "trajectory_targets_skipped_base_spatial_measure": 0,
            "trajectory_targets_skipped_unsupported_semantics": 0,
            "trajectory_target_failures": 0,
            "fraction_targets_eligible": 0.0,
            "warning": "missing trajectory provenance; skipped trajectory loss",
        }
    source_indices = np.asarray(generation_stats.source_row_indices, dtype=int)
    if source_indices.shape != (len(contexts),):
        raise ValueError("trajectory source_row_indices must align with contexts")
    anchor_ids = [batch.trajectory_ids[int(index)] for index in source_indices]
    base_supported_indices: list[int] = []
    skip_counts: dict[str, int] = {}
    for context_index, context in enumerate(contexts):
        support = trajectory_base_measure_support(context)
        if support.eligible:
            base_supported_indices.append(context_index)
        else:
            reason = support.reason or "unsupported_base_segment_measure"
            skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1
    candidate_indices = np.asarray(base_supported_indices, dtype=int)
    result = None
    provider_seconds = 0.0
    lookup_seconds = 0.0
    predicate_eval_seconds = 0.0
    if candidate_indices.size:
        result = provider.evaluate_batch(
            anchor_trajectory_ids=[anchor_ids[int(index)] for index in candidate_indices],
            contexts=[contexts[int(index)] for index in candidate_indices],
        )
        provider_seconds = float(result.provider_seconds)
        lookup_seconds = float(result.lookup_seconds)
        predicate_eval_seconds = float(result.predicate_eval_seconds)
        for reason in result.skip_reasons:
            if reason is not None:
                skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1
        eligible_local_indices = np.flatnonzero(result.eligible_mask)
        eligible_indices = candidate_indices[eligible_local_indices]
    else:
        eligible_local_indices = np.empty(0, dtype=int)
        eligible_indices = np.empty(0, dtype=int)
    skipped = int(len(contexts) - len(eligible_indices))
    if eligible_indices.size == 0:
        zero_loss = predictions.sum() * 0.0
        return zero_loss, {
            "enabled": True,
            "trajectory_targets_attempted": int(len(contexts)),
            "trajectory_targets_generated": 0,
            "trajectory_targets_skipped": skipped,
            "trajectory_targets_skipped_non_segment_measure": int(
                skip_counts.get("non_segment_measure", 0)
            ),
            "trajectory_targets_skipped_base_spatial_measure": int(
                skip_counts.get("unsupported_base_segment_spatial_measure", 0)
            ),
            "trajectory_targets_skipped_unsupported_semantics": int(
                skip_counts.get("unsupported_semantics", 0)
            ),
            "trajectory_targets_skipped_missing_provenance": int(
                skip_counts.get("missing_provenance", 0)
            ),
            "trajectory_target_failures": 0,
            "fraction_targets_eligible": 0.0,
            "traj_provider_seconds": provider_seconds,
            "traj_target_lookup_seconds": lookup_seconds,
            "traj_target_predicate_eval_seconds": predicate_eval_seconds,
            "warning": "no segment-measure trajectory targets in this batch",
        }
    assert result is not None
    multiplicities = result.multiplicities[eligible_local_indices].astype(int)
    segments_scanned = result.segments_scanned[eligible_local_indices]
    wildcard_group_counts: dict[str, int] = {}
    operator_pattern_counts: dict[str, int] = {}
    table_subset_pattern_counts: dict[str, int] = {}
    fanout_inv_signature_counts: dict[str, int] = {}
    query_type_counts: dict[str, int] = {}
    segment_predicate_counts: dict[str, int] = {}
    trajectory_predicate_columns = set(
        str(value)
        for value in config.get("trajectory_distinct", {}).get("predicate_columns", ())
    )
    segment_varying_columns = set(
        str(value)
        for value in config.get("trajectory_distinct", {}).get(
            "segment_varying_columns",
            tuple(trajectory_predicate_columns),
        )
    )
    for context_index, multiplicity in zip(eligible_indices, multiplicities):
        context = contexts[int(context_index)]
        multiplicity = int(multiplicity)
        if multiplicity <= 0:
            raise ValueError("trajectory multiplicity must be positive for accepted targets")
        wildcarded = sum(
            1
            for column, token in zip(metadata.columns, context.tokens)
            if column.name in segment_varying_columns
            and token.op == PredicateOp.WILDCARD
        )
        wildcard_key = str(wildcarded)
        wildcard_group_counts[wildcard_key] = int(wildcard_group_counts.get(wildcard_key, 0)) + 1
        operator_pattern = ",".join(
            f"{column.name}:{token.op.value}"
            for column, token in zip(metadata.columns, context.tokens)
            if column.name in segment_varying_columns
            and token.op != PredicateOp.WILDCARD
        ) or "all_wildcard"
        operator_pattern_counts[operator_pattern] = int(
            operator_pattern_counts.get(operator_pattern, 0)
        ) + 1
        table_pattern = ",".join(sorted(context.included_tables)) or "none"
        table_subset_pattern_counts[table_pattern] = int(
            table_subset_pattern_counts.get(table_pattern, 0)
        ) + 1
        fanout_pattern = ",".join(
            column.name
            for column, token in zip(metadata.columns, context.tokens)
            if column.kind == ColumnKind.FANOUT and token.op == PredicateOp.INV_FANOUT
        ) or "none"
        fanout_inv_signature_counts[fanout_pattern] = int(
            fanout_inv_signature_counts.get(fanout_pattern, 0)
        ) + 1
        trajectory_query = getattr(context, "trajectory_query", None)
        query_type = getattr(trajectory_query, "query_type", "scalar_only")
        query_type_counts[query_type] = int(query_type_counts.get(query_type, 0)) + 1
        count_key = str(
            sum(
                1
                for column, token in zip(metadata.columns, context.tokens)
                if column.name in segment_varying_columns
                and token.op != PredicateOp.WILDCARD
            )
        )
        segment_predicate_counts[count_key] = int(
            segment_predicate_counts.get(count_key, 0)
        ) + 1
    indices = eligible_indices.astype(int)
    targets = 1.0 / multiplicities.astype(float)
    terminal_log_inv = terminal_log_weights(
        target_rows[indices],
        [token_rows[index] for index in indices],
        metadata,
    )
    terminal_inv = np.exp(terminal_log_inv)
    inv_only_for_diagnostics = terminal_inv.copy()
    rho = None
    importance_stats: dict[str, Any] = {"enabled": False}
    if batch.importance_weights is not None:
        all_rho = importance_weights_for_generated_contexts(
            batch.importance_weights,
            target_rows.shape[0],
            generation_stats,
        )
        rho = all_rho[indices]
        importance_stats = {
            "enabled": True,
            "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)),
            "rho_mean": float(np.mean(rho)),
            "rho_ess": effective_sample_size(rho),
        }
    log_weights = terminal_log_weights(
        target_rows[indices],
        [token_rows[index] for index in indices],
        metadata,
        rho=rho,
    )
    shift = float(np.max(log_weights))
    weights = np.exp(log_weights - shift)
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("trajectory weights must be finite and positive")
    pred = predictions[torch.tensor(indices, dtype=torch.long, device=device), 0]
    target_tensor = torch.tensor(targets, dtype=pred.dtype, device=device)
    weight_tensor = torch.tensor(weights, dtype=pred.dtype, device=device)
    squared = (pred - target_tensor) ** 2
    loss = torch.sum(weight_tensor * squared) / (torch.sum(weight_tensor) + 1.0e-12)
    detached_pred = pred.detach().cpu().numpy().astype(float)
    squared_np = (detached_pred - targets) ** 2
    unweighted_squared_error_sum = float(np.sum(squared_np))
    weighted_squared_error_sum = float(np.dot(weights, squared_np))
    positive_error = squared_np > 0.0
    traj_log_weight_sum = float(np.logaddexp.reduce(log_weights))
    traj_log_weight_sq_sum = float(np.logaddexp.reduce(2.0 * log_weights))
    traj_log_weighted_squared_error_sum = (
        float(np.logaddexp.reduce(log_weights[positive_error] + np.log(squared_np[positive_error])))
        if np.any(positive_error)
        else float("-inf")
    )
    unweighted_mse = float(unweighted_squared_error_sum / max(len(targets), 1))
    weighted_mse = float(loss.detach().cpu())
    stats = {
        "enabled": True,
        "trajectory_targets_attempted": int(len(contexts)),
        "trajectory_targets_generated": int(len(eligible_indices)),
        "trajectory_targets_skipped": int(skipped),
        "trajectory_targets_skipped_non_segment_measure": int(
            skip_counts.get("non_segment_measure", 0)
        ),
        "trajectory_targets_skipped_base_spatial_measure": int(
            skip_counts.get("unsupported_base_segment_spatial_measure", 0)
        ),
        "trajectory_targets_skipped_unsupported_semantics": int(
            skip_counts.get("unsupported_semantics", 0)
        ),
        "trajectory_targets_skipped_missing_provenance": int(
            skip_counts.get("missing_provenance", 0)
        ),
        "trajectory_target_failures": 0,
        "fraction_targets_eligible": float(len(eligible_indices) / max(len(contexts), 1)),
        "multiplicity_reservoir": [int(value) for value in multiplicities[:4096]],
        "multiplicity_sum": float(np.sum(multiplicities)),
        "multiplicity_eq_1": int(np.sum(multiplicities == 1)),
        "multiplicity_gt_1": int(np.sum(multiplicities > 1)),
        "multiplicity_ge_5": int(np.sum(multiplicities >= 5)),
        "multiplicity_ge_10": int(np.sum(multiplicities >= 10)),
        "m_min": int(np.min(multiplicities)),
        "m_mean": float(np.mean(multiplicities)),
        "m_median": float(np.percentile(multiplicities, 50)),
        "m_p90": float(np.percentile(multiplicities, 90)),
        "m_p95": float(np.percentile(multiplicities, 95)),
        "m_max": int(np.max(multiplicities)),
        "fraction_m_equals_1": float(np.mean(np.asarray(multiplicities) == 1)),
        "fraction_m_gt_1": float(np.mean(np.asarray(multiplicities) > 1)),
        "fraction_m_ge_5": float(np.mean(np.asarray(multiplicities) >= 5)),
        "fraction_m_ge_10": float(np.mean(np.asarray(multiplicities) >= 10)),
        "target_mean": float(np.mean(targets)),
        "target_sum": float(np.sum(targets)),
        "target_median": float(np.percentile(targets, 50)),
        "prediction_mean": float(np.mean(detached_pred)),
        "prediction_sum": float(np.sum(detached_pred)),
        "prediction_min": float(np.min(detached_pred)),
        "prediction_max": float(np.max(detached_pred)),
        "weighted_mse": weighted_mse,
        "unweighted_mse": unweighted_mse,
        "weighted_squared_error_sum": weighted_squared_error_sum,
        "unweighted_squared_error_sum": unweighted_squared_error_sum,
        "weighted_ess": effective_sample_size(weights),
        "traj_dedup_importance_weight_min": float(np.min(weights)),
        "traj_dedup_importance_weight_max": float(np.max(weights)),
        "traj_dedup_importance_weight_mean": float(np.mean(weights)),
        "traj_dedup_weight_ess": effective_sample_size(weights),
        "traj_dedup_inv_only_ess": effective_sample_size(inv_only_for_diagnostics),
        "traj_loss_weight_sum": float(np.sum(weights)),
        "traj_loss_weight_sq_sum": float(np.dot(weights, weights)),
        "traj_log_weight_sum": traj_log_weight_sum,
        "traj_log_weight_sq_sum": traj_log_weight_sq_sum,
        "traj_log_weighted_squared_error_sum": traj_log_weighted_squared_error_sum,
        "traj_weight_ess": effective_sample_size(weights),
        "traj_inv_only_ess": effective_sample_size(inv_only_for_diagnostics),
        "traj_provider_seconds": provider_seconds,
        "traj_target_lookup_seconds": lookup_seconds,
        "traj_target_predicate_eval_seconds": predicate_eval_seconds,
        "traj_targets_per_second": float(
            len(eligible_indices) / result.provider_seconds
            if result.provider_seconds > 0.0
            else 0.0
        ),
        "average_segments_scanned_per_target": float(
            np.mean(segments_scanned)
        ),
        "p95_segments_scanned_per_target": float(
            np.percentile(segments_scanned, 95)
        ),
        "tuple_measure_correction": "applied" if rho is not None else "not_applicable",
        "static_fanout_correction": "applied",
        "query_generator_G_Q_given_s_correction": "not_applied_single_anchor_ablation",
        "importance": importance_stats,
        "grouped_multiplicity_context_counts": {
            "number_of_wildcarded_segment_predicates": wildcard_group_counts,
            "predicate_operator_pattern": dict(
                sorted(
                    operator_pattern_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:128]
            ),
            "table_subset_pattern": table_subset_pattern_counts,
            "fanout_inv_signature": fanout_inv_signature_counts,
            "query_type": query_type_counts,
            "number_of_segment_predicates": segment_predicate_counts,
        },
    }
    if stats["fraction_m_equals_1"] > 0.95:
        stats["warning"] = (
            "almost every trajectory target has m_t(Q)=1; "
            "the auxiliary head mostly learns traj_dedup_factor≈1"
        )
    return loss, stats


def auxiliary_eligibility_mask(
    metadata: object,
    selected_column_indices: np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    """Return [rare_rows, original_columns] mask for strict AR suffix heads."""

    selected = np.asarray(selected_column_indices, dtype=int).reshape(-1)
    if selected.shape != (row_count,):
        raise ValueError("selected_column_indices must have one entry per rare row")
    column_indices = np.arange(len(metadata.columns), dtype=int)  # type: ignore[attr-defined]
    return column_indices[None, :] > selected[:, None]


def rare_auxiliary_diagnostics(
    *,
    metadata: object,
    beta: float,
    rare_batch: FullJoinBatch,
    contexts: list[GeneratedTrainingContext],
    generation_stats: object,
    selected_strata: tuple[object, ...],
    selected_column_indices: np.ndarray,
    eligibility: np.ndarray,
    inv_only_weights: np.ndarray,
    rare_breakdown: object,
) -> dict[str, Any]:
    selected_source_indices = np.asarray(generation_stats.source_row_indices, dtype=int)
    selected_for_context = tuple(selected_strata[int(index)] for index in selected_source_indices)
    forced_operator_counts: dict[str, int] = {}
    per_stratum: dict[str, dict[str, Any]] = {}
    for context, stratum, row_eligible in zip(contexts, selected_for_context, eligibility):
        token = context.tokens[int(stratum.column_index)]
        forced_operator_counts[token.op.value] = int(forced_operator_counts.get(token.op.value, 0)) + 1
        entry = per_stratum.setdefault(
            stratum.stratum_id,
            {
                "rare_context_count": 0,
                "raw_rare_row_count": 0,
                "generated_exact_support_event_count": 0,
                "downstream_eligible_head_context_count": 0,
                "forced_operator_count": {},
            },
        )
        entry["rare_context_count"] = int(entry["rare_context_count"]) + 1
        entry["raw_rare_row_count"] = int(entry["raw_rare_row_count"]) + 1
        entry["generated_exact_support_event_count"] = int(
            entry["generated_exact_support_event_count"]
        ) + 1
        entry["downstream_eligible_head_context_count"] = int(
            entry["downstream_eligible_head_context_count"]
        ) + int(np.sum(row_eligible))
        op_counts = entry["forced_operator_count"]
        op_counts[token.op.value] = int(op_counts.get(token.op.value, 0)) + 1

    column_diagnostics: dict[str, Any] = {}
    auxiliary_inv_ess_by_fanout: dict[str, float] = {}
    for column_index, column in enumerate(metadata.columns):  # type: ignore[attr-defined]
        eligible = eligibility[:, column_index].astype(bool)
        weights = inv_only_weights[:, column_index] * eligible.astype(float)
        eligible_count = int(np.sum(eligible))
        weight_sum = float(np.sum(weights))
        weight_squared_sum = float(np.dot(weights, weights))
        if eligible_count:
            ess = effective_sample_size(weights[eligible])
            column_diagnostics[column.name] = {
                "auxiliary_column_loss": float(
                    rare_breakdown.original_column_losses.get(column.name, 0.0)
                ),
                "auxiliary_scaled_column_loss": float(
                    beta * rare_breakdown.original_column_losses.get(column.name, 0.0)
                ),
                "auxiliary_eligible_examples": eligible_count,
                "auxiliary_inv_weight_sum": weight_sum,
                "auxiliary_inv_weight_squared_sum": weight_squared_sum,
                "auxiliary_inv_ess": float(ess),
            }
            if column.kind.value == "fanout":
                auxiliary_inv_ess_by_fanout[column.name] = float(ess)
        else:
            column_diagnostics[column.name] = {
                "auxiliary_column_loss": 0.0,
                "auxiliary_scaled_column_loss": 0.0,
                "auxiliary_eligible_examples": 0,
                "auxiliary_inv_weight_sum": 0.0,
                "auxiliary_inv_weight_squared_sum": 0.0,
                "auxiliary_inv_ess": 0.0,
            }
    selected_counts: dict[str, int] = {}
    for stratum in selected_strata:
        selected_counts[stratum.stratum_id] = int(selected_counts.get(stratum.stratum_id, 0)) + 1
    fanout_signatures: dict[str, int] = {}
    for context in contexts:
        parts = [
            f"{column.name}:{token.op.value}"
            for column, token in zip(metadata.columns, context.tokens)  # type: ignore[attr-defined]
            if column.kind.value == "fanout"
        ]
        key = "|".join(parts)
        fanout_signatures[key] = int(fanout_signatures.get(key, 0)) + 1
    return {
        "enabled": True,
        "beta": float(beta),
        "rare_batch_size": int(rare_batch.encoded_values.shape[0]),
        "rare_rows_sampled": int(rare_batch.encoded_values.shape[0]),
        "rare_context_count": int(len(contexts)),
        "auxiliary_loss_unscaled": float(rare_breakdown.total_loss.detach().cpu()),
        "auxiliary_loss_scaled": float(beta * rare_breakdown.total_loss.detach().cpu()),
        "auxiliary_column_diagnostics": column_diagnostics,
        "selected_stratum_counts": selected_counts,
        "forced_auxiliary_predicate_operator_counts": forced_operator_counts,
        "auxiliary_contexts_rejected": int(generation_stats.rejected_unsatisfied_contexts),
        "included_indicator_contradictions": int(
            generation_stats.included_indicator_contradictions
        ),
        "per_stratum": per_stratum,
        "auxiliary_inv_only_ess_by_fanout": auxiliary_inv_ess_by_fanout,
        "common_auxiliary_fanout_token_signatures": dict(
            sorted(fanout_signatures.items(), key=lambda item: (-item[1], item[0]))[:32]
        ),
        "selected_predicate_column_min": int(np.min(selected_column_indices))
        if selected_column_indices.size
        else None,
        "selected_predicate_column_max": int(np.max(selected_column_indices))
        if selected_column_indices.size
        else None,
    }


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
    traj_validation_stats = _RunningTrajectoryDistinctStats()
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
            contexts, target_rows, generation_stats = context_generator.generate_batch(
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
            if batch.importance_weights is not None:
                rho = importance_weights_for_generated_contexts(
                    batch.importance_weights,
                    target_rows.shape[0],
                    generation_stats,
                )
                if np.any(rho <= 0.0) or not np.all(np.isfinite(rho)):
                    raise ValueError("importance weights rho must be finite and positive")
                weights = stable_combine_importance_and_inverse_weights(weights, rho)
            token_ids = encode_tokens_tensor(token_rows, vocabularies, device=device)
            targets = torch.tensor(target_rows, dtype=torch.long, device=device)
            head_weights = torch.tensor(weights, dtype=torch.float32, device=device)
            trajectory_config = TrajectoryDistinctConfig.from_dict(
                config.get("trajectory_distinct", {})
            )
            if trajectory_config.enabled:
                model_outputs = model.forward_with_auxiliary(token_ids)
                logits = model_outputs.ar_outputs
            else:
                model_outputs = None
                logits = model(token_ids)
            split_head_outputs = model.split_head_outputs(logits)
            output_embeddings = (
                [embedding.weight for embedding in model.output_embeddings]
                if getattr(model.config, "output_encoding", "one_hot") == "embed"
                else None
            )
            breakdown = torch_weighted_per_head_cross_entropy(
                logits,
                targets,
                head_weights,
                metadata,
                anpm_decoders=getattr(model, "anpm_decoders", None),
                split_head_outputs=split_head_outputs,
                output_embeddings=output_embeddings,
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
            if trajectory_config.enabled:
                if model_outputs is None or model_outputs.traj_dedup_factor is None:
                    raise ValueError("trajectory validation requires traj_dedup_factor")
                _, traj_stats = trajectory_dedup_loss_for_batch(
                    predictions=model_outputs.traj_dedup_factor,
                    batch=batch,
                    contexts=contexts,
                    target_rows=target_rows,
                    token_rows=token_rows,
                    generation_stats=generation_stats,
                    metadata=metadata,
                    config=config,
                    sample_source=sample_source,
                    device=device,
                )
                traj_validation_stats.update(traj_stats)
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
    trajectory_summary = traj_validation_stats.to_json_dict()
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
        "validation_traj_weighted_mse": (
            trajectory_summary["global_weighted_mse"]
            if trajectory_summary.get("enabled")
            else None
        ),
        "validation_traj_unweighted_mse": (
            trajectory_summary["global_unweighted_mse"]
            if trajectory_summary.get("enabled")
            else None
        ),
        "validation_traj_target_count": trajectory_summary.get(
            "trajectory_targets_generated",
            0,
        ),
        "validation_traj_skipped_count": trajectory_summary.get(
            "trajectory_targets_skipped",
            0,
        ),
        "validation_traj_skipped_non_segment_measure": trajectory_summary.get(
            "trajectory_targets_skipped_non_segment_measure",
            0,
        ),
        "validation_traj_skipped_base_spatial_measure": trajectory_summary.get(
            "trajectory_targets_skipped_base_spatial_measure",
            0,
        ),
        "validation_traj_skipped_unsupported_semantics": trajectory_summary.get(
            "trajectory_targets_skipped_unsupported_semantics",
            0,
        ),
        "validation_traj_m_mean": trajectory_summary.get("multiplicity", {}).get("mean"),
        "validation_traj_fraction_m_eq_1": trajectory_summary.get(
            "multiplicity",
            {},
        ).get("fraction_m_equals_1"),
        "validation_traj_weighted_ess": (
            trajectory_summary["global_weighted_ess"]
            if trajectory_summary.get("enabled")
            else None
        ),
        "validation_traj_prediction_mean": (
            trajectory_summary["global_prediction_mean"]
            if trajectory_summary.get("enabled")
            else None
        ),
        "validation_traj_target_mean": (
            trajectory_summary["global_target_mean"]
            if trajectory_summary.get("enabled")
            else None
        ),
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


def _merge_flat_counts(target: dict[str, int], update: dict[str, Any]) -> None:
    for key, value in update.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def _predicate_embedding_gradient_coverage(
    model: PredicateResMADE,
    token_rows: list[list[Any]],
    token_ids: Any,
    metadata: object,
) -> dict[str, Any]:
    """Report whether observed non-wildcard predicate embeddings received gradients."""

    if getattr(model.config, "input_encoding", "") not in {"embed", "duet_binary"}:
        return {"mode": "not_applicable", "reason": "input_encoding is not embed"}
    if getattr(model.config, "predicate_encoding_mode", "") == "two_slot_binary_duet":
        return _binary_duet_gradient_coverage(model, token_rows, token_ids, metadata)
    if getattr(model.config, "predicate_encoding_mode", "") in {
        "two_slot",
        "two_slot_categorical_legacy",
    }:
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


def _binary_duet_gradient_coverage(
    model: PredicateResMADE,
    token_rows: list[list[Any]],
    token_ids: Any,
    metadata: object,
) -> dict[str, Any]:
    observed = 0
    nonzero = 0
    ordinary_observed = 0
    ordinary_nonzero = 0
    by_column: dict[str, dict[str, int]] = {}
    for column_index, column in enumerate(metadata.columns):
        network = model.predicate_slot_networks[column_index]
        network_grad = network[0].weight.grad
        special_grad = model.special_token_embeddings[column_index].weight.grad
        has_network_grad = network_grad is not None and bool(network_grad.abs().sum() > 0)
        has_special_grad = special_grad is not None and bool(special_grad.abs().sum() > 0)
        seen_tokens = {
            token_rows[row_index][column_index]
            for row_index in range(len(token_rows))
        }
        column_counts = by_column.setdefault(
            column.name,
            {
                "observed_non_wildcard_tokens": 0,
                "nonzero_gradient_non_wildcard_tokens": 0,
                "ordinary_observed_non_wildcard_tokens": 0,
                "ordinary_nonzero_gradient_non_wildcard_tokens": 0,
                "special_embedding_gradient_nonzero": int(has_special_grad),
            },
        )
        for token in seen_tokens:
            if token.op == PredicateOp.WILDCARD:
                continue
            observed += 1
            column_counts["observed_non_wildcard_tokens"] += 1
            token_has_grad = has_special_grad if token.op == PredicateOp.INV_FANOUT else has_network_grad
            if token_has_grad:
                nonzero += 1
                column_counts["nonzero_gradient_non_wildcard_tokens"] += 1
            if token.op != PredicateOp.INV_FANOUT:
                ordinary_observed += 1
                column_counts["ordinary_observed_non_wildcard_tokens"] += 1
                if token_has_grad:
                    ordinary_nonzero += 1
                    column_counts["ordinary_nonzero_gradient_non_wildcard_tokens"] += 1
    return {
        "mode": "two_slot_binary_duet",
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


def _module_parameter_count(module: Any | None) -> int:
    if module is None or not hasattr(module, "parameters"):
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


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
