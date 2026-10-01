from __future__ import annotations

from typing import Any, Mapping


NEUROCARD_USE_COLS_MODES = frozenset({"simple", "content", "multi", "full"})


def configured_neurocard_use_cols(dataset: Mapping[str, Any]) -> str | None:
    """Return the upstream NeuroCard IMDB column projection for this dataset."""

    value = dataset.get("use_cols", "simple")
    if value is None:
        return None
    mode = str(value)
    if mode not in NEUROCARD_USE_COLS_MODES:
        choices = ", ".join(sorted(NEUROCARD_USE_COLS_MODES))
        raise ValueError(f"dataset.use_cols must be one of {choices}, or null")
    return mode
