from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.model.factorization import FactorizationConfig


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Parse the small indentation-only YAML subset used by baseline configs."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value_text = raw_value.strip()
        if value_text == "":
            value: dict[str, Any] = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = _parse_scalar(value_text)
    return root


def _parse_scalar(value_text: str) -> Any:
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

    factorization = config.get("factorization", {})
    FactorizationConfig(
        enabled=bool(factorization.get("enabled", False)),
        strategy=str(factorization.get("strategy", "none")),
    ).validate()
    inference = config.get("inference", {})
    if inference.get("progressive_sampling", False):
        raise ValueError("this milestone requires inference.progressive_sampling=false")

