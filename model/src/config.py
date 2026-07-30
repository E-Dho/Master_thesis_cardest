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


def resolve_device(config: dict[str, Any]) -> str:
    device = str(config.get("training", {}).get("device", "cpu"))
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"
