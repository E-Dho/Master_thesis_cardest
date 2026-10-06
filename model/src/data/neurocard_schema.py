from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping


# Matches LoadImdb at https://github.com/neurocard/neurocard/blob/
# a98f509e8d3c522d52ce4b4db47b894b7fafc153/neurocard/datasets.py#L242-L294.
# `full` selects JOB-full predicate columns; null asks upstream to load every CSV column.
NEUROCARD_USE_COLS_MODES = frozenset({"simple", "content", "multi", "full"})
# Before manifests tracked this field, JOB_LIGHT_BASE used `simple` at the
# pinned upstream revision above. This is the sole legacy inference we permit.
LEGACY_JOB_LIGHT_USE_COLS = "simple"


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


def validate_neurocard_manifest_projection(
    manifest: Mapping[str, Any],
    configured_use_cols: str | None,
) -> None:
    """Reject a prepared manifest built for a different NeuroCard projection."""

    if "neurocard_use_cols" not in manifest:
        if configured_use_cols == LEGACY_JOB_LIGHT_USE_COLS:
            return
        raise ValueError(
            "prepared NeuroCard manifest predates projection tracking and cannot be "
            f"used with dataset.use_cols={configured_use_cols!r}; rebuild the manifest"
        )
    prepared_use_cols = manifest["neurocard_use_cols"]
    if prepared_use_cols != configured_use_cols:
        raise ValueError(
            "prepared NeuroCard projection does not match the configured projection: "
            f"manifest={prepared_use_cols!r}, config={configured_use_cols!r}. "
            "Rebuild the manifest for this dataset.use_cols value."
        )


@contextmanager
def scoped_neurocard_prepare_cache_bypass(
    factorized_sampler: Any,
    join_spec: Any,
) -> Iterator[bool]:
    """Skip NeuroCard's Ray-backed prepare call only while a cached sampler is built."""

    prepare_utils = factorized_sampler.prepare_utils
    if not prepare_utils.check_required_files(join_spec):
        yield False
        return
    original_prepare = prepare_utils.prepare
    prepare_utils.prepare = lambda _join_spec: None
    try:
        yield True
    finally:
        prepare_utils.prepare = original_prepare
