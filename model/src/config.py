from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.model.anpm import ANPMConfig
from model.src.model.factorization import FactorizationConfig


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse the small indentation-only YAML subset used by baseline configs."""

    root: dict[str, Any] = {}
    parsed_lines: list[tuple[int, str]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        parsed_lines.append((indent, line.strip()))
    stack: list[tuple[int, Any]] = [(-1, root)]
    for line_index, (indent, stripped) in enumerate(parsed_lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("YAML list item has no list parent")
            parent.append(_parse_scalar(stripped[2:].strip()))
            continue
        key, raw_value = stripped.split(":", 1)
        value_text = raw_value.strip()
        if value_text == "":
            value: dict[str, Any] | list[Any]
            value = [] if _next_child_is_list(parsed_lines, line_index, indent) else {}
            if not isinstance(parent, dict):
                raise ValueError("YAML mapping item has no mapping parent")
            parent[key] = value
            stack.append((indent, value))
        else:
            if not isinstance(parent, dict):
                raise ValueError("YAML scalar item has no mapping parent")
            parent[key] = _parse_scalar(value_text)
    return root


def _next_child_is_list(
    parsed_lines: list[tuple[int, str]], line_index: int, parent_indent: int
) -> bool:
    for next_indent, next_text in parsed_lines[line_index + 1 :]:
        if next_indent <= parent_indent:
            return False
        return next_text.startswith("- ")
    return False


def _parse_scalar(value_text: str) -> Any:
    if value_text.startswith("[") and value_text.endswith("]"):
        inner = value_text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value_text == "false":
        return False
    if value_text == "true":
        return True
    if value_text == "null":
        return None
    try:
        if "." in value_text or "e" in value_text.lower():
            return float(value_text)
        return int(value_text)
    except ValueError:
        return value_text.strip("\"'")


def validate_config(config: dict[str, Any]) -> None:
    """Validate incompatible milestone settings at startup."""

    factorization_config = FactorizationConfig.from_dict(config.get("factorization", {}))
    factorization_config.validate()
    anpm_config = ANPMConfig.from_dict(config.get("anpm", {}))
    anpm_config.validate()
    trajectory_distinct = config.get("trajectory_distinct", {})
    trajectory_distinct_enabled = bool(trajectory_distinct.get("enabled", False))
    if str(trajectory_distinct.get("loss", "mse")) != "mse":
        raise ValueError("trajectory_distinct.loss currently supports only mse")
    if str(trajectory_distinct.get("output_activation", "sigmoid")) != "sigmoid":
        raise ValueError(
            "trajectory_distinct.output_activation currently supports only sigmoid"
        )
    if float(trajectory_distinct.get("loss_weight", 1.0) or 0.0) < 0.0:
        raise ValueError("trajectory_distinct.loss_weight must be nonnegative")
    if int(trajectory_distinct.get("anchor_samples_per_query", 1) or 0) != 1:
        raise ValueError(
            "trajectory_distinct.anchor_samples_per_query currently supports only 1"
        )
    inference = config.get("inference", {})
    if inference.get("progressive_sampling", False):
        raise ValueError("this milestone requires inference.progressive_sampling=false")
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    sampling_mode = str(dataset.get("sampling_mode", "fixture"))
    if sampling_mode not in {"fixture", "live", "materialized_large_sample"}:
        raise ValueError(
            "dataset.sampling_mode must be fixture, live, or materialized_large_sample"
        )
    if sampling_mode == "live":
        if "csv_directory" not in dataset:
            raise ValueError("dataset.sampling_mode=live requires dataset.csv_directory")
        sampler_batch_size = int(
            dataset.get("sampler_batch_size", dataset.get("sample_batch_size", 0))
        )
        if sampler_batch_size <= 0:
            raise ValueError("dataset.sampler_batch_size must be positive in live mode")
    if trajectory_distinct_enabled:
        if not str(trajectory_distinct.get("trajectory_key", "")).strip():
            raise ValueError(
                "trajectory_distinct.enabled=true requires trajectory_distinct.trajectory_key"
            )
        if not str(trajectory_distinct.get("entity_table", "")).strip():
            raise ValueError(
                "trajectory_distinct.enabled=true requires trajectory_distinct.entity_table"
            )
        if not str(trajectory_distinct.get("segment_table", "")).strip():
            raise ValueError(
                "trajectory_distinct.enabled=true requires trajectory_distinct.segment_table"
            )
        if not str(trajectory_distinct.get("segment_key", "")).strip():
            raise ValueError(
                "trajectory_distinct.enabled=true requires trajectory_distinct.segment_key"
            )
        if str(trajectory_distinct.get("predicate_scope", "")) != "segment_query":
            raise ValueError(
                "trajectory_distinct.predicate_scope must be segment_query"
            )
        segment_varying_columns = trajectory_distinct.get("segment_varying_columns", ())
        if not segment_varying_columns:
            raise ValueError(
                "trajectory_distinct.enabled=true requires segment_varying_columns"
            )
    if model.get("type") == "predicate_resmade" and not model.get("fixed_ordering", True):
        raise ValueError("predicate_resmade requires model.fixed_ordering=true")
    output_encoding = str(model.get("output_encoding", "one_hot"))
    if output_encoding not in {"one_hot", "embed"}:
        raise ValueError("model.output_encoding must be one_hot or embed")
    if output_encoding == "embed":
        if int(model.get("output_embedding_size", 64) or 0) <= 0:
            raise ValueError("model.output_embedding_size must be positive")
        if bool(model.get("output_embeddings_tied", False)):
            raise ValueError("model.output_embeddings_tied=true is not supported yet")
    direct_io_source_kinds = tuple(
        str(kind)
        for kind in model.get("direct_io_source_kinds", ["data", "indicator", "fanout"])
    )
    if any(kind not in {"data", "indicator", "fanout"} for kind in direct_io_source_kinds):
        raise ValueError(
            "model.direct_io_source_kinds must contain only data, indicator, or fanout"
        )
    direct_io_destination_kinds = tuple(
        str(kind)
        for kind in model.get(
            "direct_io_destination_kinds",
            ["data", "indicator", "fanout"],
        )
    )
    if any(kind not in {"data", "indicator", "fanout"} for kind in direct_io_destination_kinds):
        raise ValueError(
            "model.direct_io_destination_kinds must contain only data, indicator, or fanout"
        )
    if factorization_config.enabled:
        if not anpm_config.enabled:
            raise ValueError("factorization.enabled=true requires anpm.enabled=true")
        decoder = str(inference.get("factorized_decoder", "anpm"))
        if decoder != "anpm":
            raise ValueError("only inference.factorized_decoder=anpm is supported")
    predicate_generation = config.get("predicate_generation", {})
    probabilities = (
        float(predicate_generation.get("wildcard_probability", 0.2)),
        float(predicate_generation.get("equality_probability", 0.4)),
        float(predicate_generation.get("lower_bound_probability", 0.2)),
        float(predicate_generation.get("upper_bound_probability", 0.2)),
        float(predicate_generation.get("native_range_probability", 0.0)),
    )
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("predicate_generation probabilities must be nonnegative")
    if sum(probabilities) <= 0.0:
        raise ValueError("predicate_generation probabilities must have positive total")
    normalize_probabilities = bool(
        predicate_generation.get("normalize_predicate_probabilities", True)
    )
    if not normalize_probabilities and abs(sum(probabilities) - 1.0) > 1.0e-8:
        raise ValueError(
            "predicate_generation probabilities must sum to 1.0 when "
            "normalize_predicate_probabilities=false"
        )
    strategy = str(predicate_generation.get("strategy", "row_satisfied"))
    if strategy not in {"row_satisfied", "duet_batch_bounds"}:
        raise ValueError(
            "predicate_generation.strategy must be row_satisfied or duet_batch_bounds"
        )
    native_range_max_domain_size = int(
        predicate_generation.get("native_range_max_domain_size", 512)
    )
    if native_range_max_domain_size <= 0:
        raise ValueError(
            "predicate_generation.native_range_max_domain_size must be positive"
        )
    per_row_contexts = int(predicate_generation.get("per_row_contexts", 1))
    if per_row_contexts <= 0:
        raise ValueError("predicate_generation.per_row_contexts must be positive")
    table_subset_sampling = str(predicate_generation.get("table_subset_sampling", "full"))
    if table_subset_sampling not in {
        "full",
        "connected",
        "neurocard_rooted_connected",
        "rooted_connected_uniform_legacy",
        "neurocard_table_dropout_rooted",
    }:
        raise ValueError(
            "predicate_generation.table_subset_sampling must be 'full', 'connected', "
            "'rooted_connected_uniform_legacy', 'neurocard_rooted_connected', "
            "or 'neurocard_table_dropout_rooted'"
        )
    predicate_encoding = config.get("predicate_encoding", {})
    encoding_mode = str(predicate_encoding.get("mode", "categorical_legacy"))
    if encoding_mode not in {
        "categorical_legacy",
        "compositional",
        "hybrid",
        "two_slot",
        "two_slot_categorical_legacy",
        "two_slot_binary_duet",
    }:
        raise ValueError(
            "predicate_encoding.mode must be categorical_legacy, compositional, "
            "hybrid, two_slot, two_slot_categorical_legacy, or two_slot_binary_duet"
        )
    if str(model.get("input_encoding", "embed")) == "duet_binary":
        if encoding_mode != "two_slot_binary_duet":
            raise ValueError(
                "model.input_encoding=duet_binary requires "
                "predicate_encoding.mode=two_slot_binary_duet"
            )
    if encoding_mode in {
        "compositional",
        "hybrid",
        "two_slot",
        "two_slot_categorical_legacy",
        "two_slot_binary_duet",
    }:
        model = config.get("model", {})
        if str(model.get("input_encoding", "embed")) not in {"embed", "duet_binary"}:
            raise ValueError(
                "predicate_encoding.mode=compositional/hybrid/two_slot requires "
                "model.input_encoding=embed or duet_binary"
            )
    if encoding_mode in {"compositional", "hybrid"}:
        for key in [
            "operator_embedding_size",
            "value_embedding_size",
            "special_embedding_size",
            "merge_hidden_size",
        ]:
            if int(predicate_encoding.get(key, 1) or 0) <= 0:
                raise ValueError(f"predicate_encoding.{key} must be positive")
    training = config.get("training", {})
    early_stopping_patience_steps = int(
        training.get(
            "early_stopping_patience_steps",
            training.get("early_stopping_patience", 0),
        )
        or 0
    )
    if early_stopping_patience_steps < 0:
        raise ValueError("training early stopping patience must be nonnegative")
    if early_stopping_patience_steps > 0:
        metrics_interval = int(training.get("validation_interval_steps", 0) or 0)
        if metrics_interval <= 0:
            raise ValueError(
                "training early stopping requires training.validation_interval_steps > 0"
            )
    validation = config.get("validation", {})
    if validation.get("enabled", False):
        interval_steps = int(
            validation.get(
                "interval_steps",
                config.get("training", {}).get("validation_interval_steps", 0),
            )
            or 0
        )
        if interval_steps <= 0:
            raise ValueError("validation.enabled=true requires validation.interval_steps > 0")
        if int(validation.get("fresh_sampler_batches", 0) or 0) <= 0:
            raise ValueError(
                "validation.enabled=true requires validation.fresh_sampler_batches > 0"
            )
        metric = str(validation.get("selection_metric", "validation_weighted_nll"))
        if metric not in {
            "validation_nll",
            "validation_weighted_nll",
            "validation_traj_weighted_mse",
        }:
            raise ValueError(
                "validation.selection_metric must be validation_nll or "
                "validation_weighted_nll or validation_traj_weighted_mse"
            )
    importance = config.get("importance_sampling", {})
    if bool(importance.get("enabled", False)):
        mixture = float(importance.get("mixture_probability", 0.2))
        if not (0.0 <= mixture < 1.0):
            raise ValueError("importance_sampling.mixture_probability must be in [0, 1)")
        discovery = importance.get("discovery", {})
        if not bool(discovery.get("root_data_only", True)):
            raise ValueError(
                "importance_sampling.discovery.root_data_only=false is not supported yet"
            )
        if float(discovery.get("minimum_expected_context_support", 100.0)) <= 0.0:
            raise ValueError(
                "importance_sampling.discovery.minimum_expected_context_support must be positive"
            )
        if int(discovery.get("max_selected_strata", 64)) <= 0:
            raise ValueError("importance_sampling.discovery.max_selected_strata must be positive")
        if "support_planning_steps" in discovery and int(discovery["support_planning_steps"]) <= 0:
            raise ValueError(
                "importance_sampling.discovery.support_planning_steps must be positive"
            )
        root_column_semantics = discovery.get("root_column_semantics", {})
        if root_column_semantics is not None:
            if not isinstance(root_column_semantics, dict):
                raise ValueError(
                    "importance_sampling.discovery.root_column_semantics must be a mapping"
                )
            for column_name, semantic_type in root_column_semantics.items():
                nested_items = (
                    semantic_type.items()
                    if isinstance(semantic_type, dict)
                    else ((column_name, semantic_type),)
                )
                for nested_column_name, nested_semantic_type in nested_items:
                    if str(nested_semantic_type) not in {"ordered", "categorical"}:
                        raise ValueError(
                            "importance_sampling.discovery.root_column_semantics values "
                            f"must be ordered or categorical, got {nested_semantic_type!r} "
                            f"for {nested_column_name!r}"
                        )
        allocation = str(importance.get("allocation", {}).get("strategy", "support_deficit"))
        if allocation != "support_deficit":
            raise ValueError(
                "importance_sampling.allocation.strategy must be support_deficit"
            )
        diagnostics = importance.get("diagnostics", {})
        if int(diagnostics.get("global_rho_reservoir_size", 100_000)) < 0:
            raise ValueError(
                "importance_sampling.diagnostics.global_rho_reservoir_size must be nonnegative"
            )
        if int(diagnostics.get("per_stratum_rho_reservoir_size", 1_000)) < 0:
            raise ValueError(
                "importance_sampling.diagnostics.per_stratum_rho_reservoir_size must be nonnegative"
            )
    rare_support = config.get("rare_support", {})
    rare_auxiliary = config.get("rare_auxiliary", {})
    if bool(rare_auxiliary.get("enabled", False)):
        if bool(importance.get("enabled", False)):
            raise ValueError(
                "rare_auxiliary.enabled=true must not be combined with "
                "importance_sampling.enabled=true"
            )
        if not bool(rare_support.get("enabled", False)):
            raise ValueError("rare_auxiliary.enabled=true requires rare_support.enabled=true")
        if int(rare_auxiliary.get("batch_size", 0) or 0) <= 0:
            raise ValueError("rare_auxiliary.batch_size must be positive")
        if float(rare_auxiliary.get("beta", 0.0) or 0.0) < 0.0:
            raise ValueError("rare_auxiliary.beta must be nonnegative")
    if bool(rare_support.get("enabled", False)):
        discovery = rare_support.get("discovery", {})
        if not bool(discovery.get("root_data_only", True)):
            raise ValueError("rare_support.discovery.root_data_only=false is not supported yet")
        if float(discovery.get("minimum_expected_context_support", 100.0)) <= 0.0:
            raise ValueError(
                "rare_support.discovery.minimum_expected_context_support must be positive"
            )
        if int(discovery.get("max_selected_strata", 64)) <= 0:
            raise ValueError("rare_support.discovery.max_selected_strata must be positive")
        if "support_planning_steps" in discovery and int(discovery["support_planning_steps"]) <= 0:
            raise ValueError("rare_support.discovery.support_planning_steps must be positive")
        allocation = str(rare_support.get("allocation", {}).get("strategy", "support_deficit"))
        if allocation != "support_deficit":
            raise ValueError("rare_support.allocation.strategy must be support_deficit")


def resolve_device(config: dict[str, Any]) -> str:
    device = str(config.get("training", {}).get("device", "cpu"))
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"
