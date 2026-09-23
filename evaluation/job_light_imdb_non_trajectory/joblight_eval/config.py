from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model.src.config import load_simple_yaml


_SLUG = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
# "native": the method's published configuration.  "adapted": an explicitly
# project-modified variant (e.g. extended schema); reports label it as such.
PROTOCOLS = ("native", "adapted")


@dataclass(frozen=True)
class WorkloadConfig:
    workload_id: str
    queries_csv: Path
    queries_sql: Path | None = None
    expected_query_count: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    experiment_id: str
    method_id: str
    variant_id: str
    display_name: str
    seeds: tuple[int, ...]
    results_root: Path
    workloads: tuple[WorkloadConfig, ...]
    adapter_type: str
    adapter: dict[str, Any]
    timing: dict[str, Any]
    resources: dict[str, Any]
    source: dict[str, Any]
    artifacts: dict[str, Any]
    raw: dict[str, Any]
    config_hash: str
    protocol: str = "native"
    adaptation: str = ""


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path).resolve()
    raw = load_simple_yaml(source_path)
    extends = raw.pop("extends", None)
    if extends:
        base_path = (source_path.parent / str(extends)).resolve()
        base = load_simple_yaml(base_path)
        if "extends" in base:
            raise ValueError("nested config inheritance is not supported")
        raw = _deep_merge(base, raw)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("evaluation config schema_version must be 1")
    identity = _mapping(raw, "experiment")
    experiment_id = _slug(identity.get("experiment_id"), "experiment_id")
    method_id = _slug(identity.get("method_id"), "method_id")
    variant_id = _slug(identity.get("variant_id"), "variant_id")
    display_name = str(identity.get("display_name") or variant_id)
    protocol = str(identity.get("protocol") or "native")
    if protocol not in PROTOCOLS:
        raise ValueError(f"experiment.protocol must be one of {PROTOCOLS}; received {protocol!r}")
    adaptation = str(identity.get("adaptation") or "")
    if protocol == "adapted" and not adaptation:
        raise ValueError("adapted experiments must describe the adaptation in experiment.adaptation")
    seeds_value = identity.get("seeds", [0])
    if not isinstance(seeds_value, list) or not seeds_value:
        raise ValueError("experiment.seeds must be a non-empty list")
    seeds = tuple(int(value) for value in seeds_value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds must not contain duplicates")

    path_config = _mapping(raw, "paths")
    results_root = _resolve_path(source_path, path_config.get("results_root", "results"))
    workload_map = _mapping(raw, "workloads")
    workloads: list[WorkloadConfig] = []
    for workload_id, value in workload_map.items():
        workload_id = _slug(workload_id, "workload id")
        if not isinstance(value, dict):
            raise ValueError(f"workloads.{workload_id} must be a mapping")
        workloads.append(
            WorkloadConfig(
                workload_id=workload_id,
                queries_csv=_resolve_path(source_path, value.get("queries_csv")),
                queries_sql=(
                    None
                    if value.get("queries_sql") in (None, "")
                    else _resolve_path(source_path, value["queries_sql"])
                ),
                expected_query_count=(
                    None
                    if value.get("expected_query_count") in (None, "")
                    else int(value["expected_query_count"])
                ),
            )
        )
    if not workloads:
        raise ValueError("at least one workload must be configured")
    for workload in workloads:
        if not workload.queries_csv.exists():
            raise FileNotFoundError(workload.queries_csv)
        if workload.expected_query_count is not None:
            observed = sum(
                bool(line.strip())
                for line in workload.queries_csv.read_text(encoding="utf-8").splitlines()
            )
            if observed != workload.expected_query_count:
                raise ValueError(
                    f"{workload.workload_id} expected {workload.expected_query_count} "
                    f"queries, observed {observed}"
                )

    adapter = _mapping(raw, "adapter")
    adapter_type = _slug(adapter.get("type"), "adapter.type")
    source = _mapping(raw, "source")
    if not source.get("url"):
        raise ValueError("source.url is required")
    if not (source.get("revision") or source.get("installed_version")):
        raise ValueError("source.revision or source.installed_version is required")
    artifacts = _mapping(raw, "artifacts")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ExperimentConfig(
        source_path=source_path,
        experiment_id=experiment_id,
        method_id=method_id,
        variant_id=variant_id,
        display_name=display_name,
        seeds=seeds,
        results_root=results_root,
        workloads=tuple(workloads),
        adapter_type=adapter_type,
        adapter=adapter,
        timing=dict(raw.get("timing", {})),
        resources=dict(raw.get("resources", {})),
        source=source,
        artifacts=artifacts,
        raw=raw,
        config_hash=config_hash,
        protocol=protocol,
        adaptation=adaptation,
    )


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _slug(value: Any, label: str) -> str:
    text = str(value or "")
    if not _SLUG.fullmatch(text):
        raise ValueError(
            f"{label} must match {_SLUG.pattern!r}; received {text!r}"
        )
    return text


def _resolve_path(config_path: Path, value: Any) -> Path:
    if value in (None, ""):
        raise ValueError("required path is empty")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* on top of *base*.

    Mappings are merged recursively; all other types — including **lists** — are
    replaced entirely by the override value.  This means that a list in a child
    config (e.g. ``seeds``, ``sample_sizes``) replaces the base list rather than
    extending it.  When writing child configs, always specify the complete list
    even if only one element differs from the base.
    """
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
