"""Workload formats: how a workload file becomes scored queries.

The runner and the subprocess adapter only need, per query, an integer
``query_id`` and its ``true_cardinality``. Everything else about a workload --
its file layout, its statistics, the placeholders handed to external
commands, extra summary blocks -- is format-specific and lives behind
:class:`WorkloadFormat`. Formats are provided by pipeline plugins
(:mod:`evaluation.eval_core.plugins`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .config import WorkloadConfig

DEFAULT_WORKLOAD_FORMAT = "joblight_csv"


class ScoredQuery(Protocol):
    query_id: int
    true_cardinality: int


class WorkloadFormat(Protocol):
    name: str

    def parse_config(
        self,
        workload_id: str,
        value: Mapping[str, Any],
        resolve_path: Callable[[Any], Path],
    ) -> WorkloadConfig:
        """Build the workload entry from its YAML mapping."""

    def validate(self, workload: WorkloadConfig) -> None:
        """Check files and ``expected_query_count`` at config-load time."""

    def load(self, workload: WorkloadConfig) -> Sequence[ScoredQuery]:
        """All scored queries in their canonical order."""

    def metadata(self, workload: WorkloadConfig, queries: Sequence[ScoredQuery]) -> dict[str, Any]:
        """Workload identity and statistics recorded in manifests and summaries."""

    def placeholders(self, workload: WorkloadConfig) -> dict[str, str]:
        """Workload-specific placeholders for external commands."""

    def summary_extension(
        self,
        workload: WorkloadConfig,
        queries: Sequence[ScoredQuery],
        predictions: Sequence[Any],
    ) -> dict[str, Any] | None:
        """Optional extra summary block (``None`` adds nothing)."""


def format_of(workload: WorkloadConfig) -> WorkloadFormat:
    from .plugins import workload_format

    return workload_format(workload.format)


def load_queries(workload: WorkloadConfig) -> Sequence[ScoredQuery]:
    return format_of(workload).load(workload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
