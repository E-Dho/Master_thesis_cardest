"""Read-only accounting of inference artifacts; base relation rows are excluded."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

SIZE_RULE = "everything needed at inference except the base data"
POSTGRES_STATISTICS_SOURCE = "SELECT pg_total_relation_size('pg_catalog.pg_statistic')"


def measurement_version(method: str) -> int:
    # DistJoin v3 / NeuroCard v2 retain mapping subclasses as loaded;
    # DistJoin also excludes cached per-base-row null masks.
    return {"distjoin": 3, "neurocard": 2}.get(method, 1)


def files_size(paths: Iterable[Path], metadata: Any = None) -> dict[str, Any]:
    files = list(dict.fromkeys(Path(p).resolve() for p in paths))
    bindings = [{"path": str(p), "bytes": p.stat().st_size,
                 "mtime_ns": p.stat().st_mtime_ns} for p in files]
    metadata_bytes = len(pickle.dumps(metadata, protocol=4)) if metadata is not None else 0
    return {"inference_artifact_mb": (sum(p["bytes"] for p in bindings) + metadata_bytes) / 1_000_000,
            "inference_files": bindings, "inference_metadata_bytes": metadata_bytes,
            "model_size_source": SIZE_RULE}


def mscn_files(checkpoint: Path, metadata: Path, source: Path, runtime: Path | None = None) -> list[Path]:
    sidecar = (runtime / "domain_and_samples.pkl" if runtime is not None
               else source / "workloads/job-light.bitmaps")
    return [checkpoint, metadata, sidecar]


def deepdb_files(checkpoint: Path, rank_domains: Path | None = None) -> list[Path]:
    paths = [checkpoint, checkpoint.parent / "pairwise_rdc.pkl"]
    return paths + ([rank_domains] if rank_domains is not None else [])


def distjoin_checkpoints(directory: Path, pattern: str, datasets: Iterable[str], mark: str = "production") -> list[Path]:
    # This is the same template and set of dataset names _load_checkpoints uses.
    return [directory / pattern.format(dataset, mark) for dataset in datasets]


def distjoin_cardinality_files(source: Path, base_cardinalities: Path, tag: str = "") -> list[Path]:
    # JoinOrderBenchmark.LoadTrueBaseCard warns and does not open a missing
    # optional pickle. The wrapper always opens its explicit normalization JSON.
    optional = source / ("queries/basecard" + tag + ".pkl")
    return ([optional] if optional.is_file() else []) + [base_cardinalities]


def schema_payload(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Copy the loaded schema objects, retaining their types, domains and buffers.

    Base relation frames/tuples are replaced with None. Pickle therefore stores
    the actual inference schema classes, not an estimate or a JSON proxy.
    """
    import types
    if memo is None:
        memo = {}
    if value is None or isinstance(value, (str, int, float, bool, bytes, type)):
        return value
    if id(value) in memo:
        return memo[id(value)]
    if type(value).__module__.startswith("pandas.") and type(value).__name__ in ("DataFrame", "Series"):
        return None
    if type(value).__module__ == "torch" and type(value).__name__ == "device":
        return type(value)("cpu")
    if type(value).__module__.startswith(("numpy", "torch")):
        return value
    if isinstance(value, types.MethodType):
        return types.MethodType(value.__func__, schema_payload(value.__self__, memo))
    if isinstance(value, dict):
        result = value.copy()
        result.clear()
        memo[id(value)] = result
        result.update((k, schema_payload(v, memo)) for k, v in value.items())
        return result
    if isinstance(value, list):
        result = []; memo[id(value)] = result
        result.extend(schema_payload(v, memo) for v in value)
        return result
    if isinstance(value, tuple):
        values = [schema_payload(v, memo) for v in value]
        result = type(value)(*values) if hasattr(value, "_fields") else tuple(values)
        memo[id(value)] = result
        return result
    if isinstance(value, (set, frozenset)):
        result = type(value)(schema_payload(v, memo) for v in value); memo[id(value)] = result
        return result
    if hasattr(value, "__dict__") and not callable(value):
        result = type(value).__new__(type(value)); memo[id(value)] = result
        excluded = {"data", "tuples", "tuples_np", "raw_tuples", "factorized_tuples",
                    "factorized_tuples_np", "new_tuples", "inp", "queries", "preds",
                    "true_cards", "valid_i_list", "table_dataset", "nan_ind",
                    "loader", "model", "models", "optimizer"}
        result.__dict__.update((k, None if k in excluded else schema_payload(v, memo))
                               for k, v in vars(value).items())
        return result
    return value


def foj_size(reservoir: Path, metadata: Path, sample_size: int) -> dict[str, Any]:
    import numpy as np
    rows = np.load(reservoir, mmap_mode="r")
    if sample_size > len(rows):
        raise ValueError("sample larger than reservoir")
    # Inference uses N sampled rows, not every unused row of the shared reservoir.
    header_bytes = reservoir.stat().st_size - rows.nbytes
    sample_bytes = sample_size * rows.shape[1] * rows.dtype.itemsize + header_bytes
    report = files_size([metadata])
    report["inference_files"].append({"path": str(reservoir.resolve()),
                                     "bytes": reservoir.stat().st_size,
                                     "mtime_ns": reservoir.stat().st_mtime_ns,
                                     "counted_bytes": int(sample_bytes)})
    report["inference_artifact_mb"] += sample_bytes / 1_000_000
    return report


def postgres_size(artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = artifact.get("database_manifest") or {}
    # Legacy manifests measured the entire catalog, including system statistics.
    # Never silently relabel that as benchmark-table-specific storage.
    eligible = manifest.get("statistics_storage_scope") == "benchmark_tables"
    value = manifest.get("statistics_storage_bytes") if eligible else None
    return {"inference_artifact_mb": value / 1_000_000 if value is not None else None,
            "model_size_source": manifest.get("statistics_storage_source", POSTGRES_STATISTICS_SOURCE),
            "statistics_storage_scope": manifest.get("statistics_storage_scope", "whole_database_catalog")}
