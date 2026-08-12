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
    if model.get("type") == "predicate_resmade" and not model.get("fixed_ordering", True):
        raise ValueError("predicate_resmade requires model.fixed_ordering=true")
    if factorization_config.enabled:
        if model.get("direct_io_connections", False):
            raise ValueError(
                "factorization.enabled=true requires model.direct_io_connections=false"
            )
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
    if table_subset_sampling not in {"full", "connected"}:
        raise ValueError(
            "predicate_generation.table_subset_sampling must be 'full' or 'connected'"
        )
    predicate_encoding = config.get("predicate_encoding", {})
    encoding_mode = str(predicate_encoding.get("mode", "categorical_legacy"))
    if encoding_mode not in {"categorical_legacy", "compositional", "hybrid", "two_slot"}:
        raise ValueError(
            "predicate_encoding.mode must be categorical_legacy, compositional, hybrid, or two_slot"
        )
    if encoding_mode in {"compositional", "hybrid", "two_slot"}:
        model = config.get("model", {})
        if str(model.get("input_encoding", "embed")) != "embed":
            raise ValueError(
                "predicate_encoding.mode=compositional/hybrid/two_slot requires "
                "model.input_encoding=embed"
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
        if metric not in {"validation_nll", "validation_weighted_nll"}:
            raise ValueError(
                "validation.selection_metric must be validation_nll or "
                "validation_weighted_nll in this milestone"
            )


def resolve_device(config: dict[str, Any]) -> str:
    device = str(config.get("training", {}).get("device", "cpu"))
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"
