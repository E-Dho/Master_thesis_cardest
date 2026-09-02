from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from model.src.data.full_join_sampler import OUTER_MISSING, canonicalize_fanout_value
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
    factorization_plan_hash,
)
from model.src.predicates.operators import PredicateOp

SQL_NULL = "__SQL_NULL__"
MANIFEST_FORMAT_VERSION = 2
COMPLETE_METADATA_SOURCE = "complete_base_tables_and_join_metadata"


@dataclass(frozen=True)
class SampleEncodingIssue:
    """One sampled value that could not be encoded against fixed metadata."""

    row_index: int
    column_name: str
    value: Any
    reason: str


@dataclass(frozen=True)
class EncodedSampleRows:
    """Encoded validation rows plus any out-of-domain conversion issues."""

    encoded_rows: np.ndarray
    issues: tuple[SampleEncodingIssue, ...] = ()


@dataclass(frozen=True)
class QueryLiteralClassification:
    """Classification of a query literal against an original-column domain."""

    column_name: str
    operator: str
    literal: Any
    category: str
    reason: str


@dataclass(frozen=True)
class PreparedArtifacts:
    """Paths and metadata produced by a complete-domain preparation run."""

    manifest_path: Path
    sample_rows_path: Path
    stats_path: Path
    metadata: ModelMetadata
    encoded_rows: np.ndarray
    stats: dict[str, Any]


@dataclass(frozen=True)
class CompleteDomainSpec:
    """Schema inputs needed to build metadata independent of sampled rows."""

    join_tables: tuple[str, ...]
    join_root: str
    join_keys: Mapping[str, tuple[str, ...]]
    join_cardinality: float
    dataset_name: str = "job_light"
    dataset_type: str = "neurocard_full_join"
    source: str = "neurocard_factorized_sampler"
    sample_source: str = "neurocard_factorized_sampler"
    upstream_attribution: Mapping[str, str] = field(
        default_factory=lambda: {
            "NeuroCard": (
                "complete base-table domains, full-outer indicators, "
                "fanout metadata, and Exact Weight validation samples"
            )
        }
    )


def build_complete_metadata(
    table_frames: Mapping[str, Any],
    spec: CompleteDomainSpec,
) -> ModelMetadata:
    """Construct ModelMetadata from complete base tables and join metadata.

    Ordinary domains are read from complete source-table columns, indicators are
    explicit `(0, 1)`, and fanouts are dense positive ranges derived from
    complete join-key frequencies. Sample rows are intentionally not consulted.
    """

    columns: list[ColumnMetadata] = []
    for table_name in spec.join_tables:
        frame = _as_dataframe(table_frames[table_name])
        join_key_set = set(spec.join_keys[table_name])
        for source_column in frame.columns:
            if source_column in join_key_set:
                continue
            domain = complete_ordinary_domain(frame[source_column])
            columns.append(
                ColumnMetadata(
                    _data_column_name(table_name, str(source_column)),
                    ColumnKind.DATA,
                    domain,
                    table=table_name,
                )
            )

    for table_name in spec.join_tables:
        columns.append(
            ColumnMetadata(
                _indicator_column_name(table_name),
                ColumnKind.INDICATOR,
                (0, 1),
                table=table_name,
            )
        )

    for table_name in spec.join_tables:
        if table_name == spec.join_root:
            continue
        frame = _as_dataframe(table_frames[table_name])
        keys = spec.join_keys[table_name]
        for fanout_name, domain in complete_fanout_domains_for_table(
            table_name,
            frame,
            keys,
        ).items():
            columns.append(
                ColumnMetadata(
                    fanout_name,
                    ColumnKind.FANOUT,
                    domain,
                    table=table_name,
                    fanout_source=f"{table_name}:{','.join(keys)}",
                )
            )

    return ModelMetadata(
        columns=tuple(columns),
        full_join_cardinality=float(spec.join_cardinality),
        upstream_attribution=dict(spec.upstream_attribution),
        join_root=spec.join_root,
        join_tables=tuple(spec.join_tables),
        join_edges=tuple(
            (spec.join_root, table_name)
            for table_name in spec.join_tables
            if table_name != spec.join_root
        ),
    )


def complete_ordinary_domain(values: Sequence[Any]) -> tuple[Any, ...]:
    """Return a complete ordinary domain plus a distinct outer-padding token."""

    canonical_values = [canonicalize_base_value(value) for value in values]
    canonical_values.append(OUTER_MISSING)
    return sorted_unique(canonical_values)


def complete_fanout_domains_for_table(
    table_name: str,
    frame: Any,
    join_keys: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    """Build dense positive fanout domains from complete join-key frequencies."""

    if not join_keys:
        raise ValueError(f"table {table_name!r} has no join keys")
    domains: dict[str, tuple[int, ...]] = {}
    if len(join_keys) == 1:
        key = join_keys[0]
        maximum = _max_group_frequency(frame, (key,))
        domains[_fanout_column_name(table_name)] = tuple(range(1, maximum + 1))
        return domains
    for key in join_keys:
        maximum = _max_group_frequency(frame, (key,))
        domains[_fanout_column_name(table_name, key)] = tuple(range(1, maximum + 1))
    return domains


def encode_sample_dataframe(
    sample_frame: Any,
    metadata: ModelMetadata,
    *,
    strict: bool = True,
) -> EncodedSampleRows:
    """Encode sampler rows against fixed complete metadata and report OODs."""

    frame = _as_dataframe(sample_frame)
    expected_columns = tuple(column.name for column in metadata.columns)
    observed_columns = tuple(str(column) for column in frame.columns)
    if observed_columns != expected_columns:
        raise ValueError(
            "sampled column order differs from metadata order: "
            f"expected {expected_columns}, observed {observed_columns}"
        )
    indicator_values_by_table = _canonical_indicator_values_by_table(frame, metadata)
    encoded = np.zeros((len(frame), len(metadata.columns)), dtype=np.int64)
    issues: list[SampleEncodingIssue] = []
    domain_maps = [
        {value: index for index, value in enumerate(column.domain)}
        for column in metadata.columns
    ]
    for row_index in range(len(frame)):
        for column_index, column in enumerate(metadata.columns):
            raw_value = frame.iloc[row_index, column_index]
            try:
                value = _canonicalize_sample_value(
                    raw_value,
                    column,
                    row_index,
                    indicator_values_by_table,
                )
            except ValueError as exc:
                issue = SampleEncodingIssue(
                    row_index,
                    column.name,
                    _jsonable(raw_value),
                    str(exc),
                )
                issues.append(issue)
                continue
            encoded_index = domain_maps[column_index].get(value)
            if encoded_index is None:
                issues.append(
                    SampleEncodingIssue(
                        row_index,
                        column.name,
                        value,
                        "sampled value is outside the complete manifest domain",
                    )
                )
                continue
            encoded[row_index, column_index] = encoded_index
    if strict and issues:
        first = issues[0]
        raise ValueError(
            "sample rows contain values outside complete metadata; first issue: "
            f"row={first.row_index}, column={first.column_name}, "
            f"value={first.value!r}, reason={first.reason}. "
            "Rebuild the manifest from complete base tables and join metadata."
        )
    return EncodedSampleRows(encoded, tuple(issues))


def build_manifest_payload(
    *,
    metadata: ModelMetadata,
    spec: CompleteDomainSpec,
    sample_rows: int,
    source_csv_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a versioned manifest that marks domains as complete."""

    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "dataset_name": spec.dataset_name,
        "dataset_type": spec.dataset_type,
        "join_cardinality": int(spec.join_cardinality),
        "metadata": metadata.to_json_dict(),
        "source": spec.source,
        "metadata_source": COMPLETE_METADATA_SOURCE,
        "sample_source": spec.sample_source,
        "sample_rows": int(sample_rows),
        "domains_complete": True,
        "source_csv_fingerprints": dict(source_csv_fingerprints or {}),
    }


def preparation_stats(
    *,
    metadata: ModelMetadata,
    encoded_sample: EncodedSampleRows,
    factorization_config: FactorizationConfig,
    source_csv_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
    ood_literal_classifications: Sequence[QueryLiteralClassification] = (),
) -> dict[str, Any]:
    """Summarize complete-domain preparation and factorization activation."""

    factorized_metadata = apply_factorization_to_metadata(metadata, factorization_config)
    plan = factorized_metadata.factorization_plan
    factorized_columns = []
    for factorization in plan.original_column_factorizations:
        column = metadata.columns[factorization.original_column_index]
        factorized_columns.append(
            {
                "column": column.name,
                "original_domain_size": column.domain_size,
                "factor_count": len(factorization.factor_column_indices),
                "factor_domains": factorization.factor_domains,
                "invalid_combinations": factorization.invalid_combination_count,
            }
        )
    fanout_min_max = {
        column.name: (int(min(column.domain)), int(max(column.domain)))
        for column in metadata.columns
        if column.kind == ColumnKind.FANOUT
    }
    original_width = sum(metadata.data_output_bins)
    factorized_width = sum(factorized_metadata.model_output_bins)
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "domains_complete": True,
        "number_of_original_columns": sum(
            1 for column in metadata.columns if column.kind == ColumnKind.DATA
        ),
        "number_of_indicator_columns": sum(
            1 for column in metadata.columns if column.kind == ColumnKind.INDICATOR
        ),
        "number_of_fanout_columns": sum(
            1 for column in metadata.columns if column.kind == ColumnKind.FANOUT
        ),
        "domain_size_per_column": {
            column.name: column.domain_size for column in metadata.columns
        },
        "total_original_output_width": original_width,
        "total_factorized_output_width": factorized_width,
        "factorized_output_reduction_ratio": (
            factorized_width / max(original_width, 1)
        ),
        "factorized_columns": factorized_columns,
        "factor_count_per_column": {
            item["column"]: item["factor_count"] for item in factorized_columns
        },
        "fanout_minimum_and_maximum": fanout_min_max,
        "sample_row_count": int(encoded_sample.encoded_rows.shape[0]),
        "sample_encoding_ood_values": len(encoded_sample.issues),
        "ood_evaluation_literals": len(ood_literal_classifications),
        "ood_literal_classifications": [
            classification.__dict__ for classification in ood_literal_classifications
        ],
        "source_csv_fingerprints": dict(source_csv_fingerprints or {}),
        "schema_hash": metadata.stable_schema_hash(),
        "factorization_hash": factorization_plan_hash(plan),
    }


def write_prepared_artifacts(
    *,
    prepared_directory: str | Path,
    manifest_payload: Mapping[str, Any],
    encoded_rows: np.ndarray,
    stats: Mapping[str, Any],
) -> PreparedArtifacts:
    """Atomically write manifest, encoded rows, and preparation statistics."""

    directory = Path(prepared_directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    sample_rows_path = directory / "sample_rows.npy"
    stats_path = directory / "preparation_stats.json"

    _atomic_write_text(manifest_path, json.dumps(manifest_payload, indent=2, sort_keys=True))
    _atomic_write_npy(sample_rows_path, encoded_rows)
    _atomic_write_text(stats_path, json.dumps(stats, indent=2, sort_keys=True))

    metadata = validate_prepared_manifest(directory)
    loaded_rows = np.load(sample_rows_path)
    return PreparedArtifacts(
        manifest_path=manifest_path,
        sample_rows_path=sample_rows_path,
        stats_path=stats_path,
        metadata=metadata,
        encoded_rows=loaded_rows,
        stats=dict(stats),
    )


def validate_prepared_manifest(prepared_directory: str | Path) -> ModelMetadata:
    """Validate that a prepared manifest has complete fixed domains."""

    directory = Path(prepared_directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing preparation manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("domains_complete") is not True:
        raise ValueError(
            "NeuroCard manifest domains are not marked complete. This usually "
            "means the manifest was inferred from sample_rows.npy. Rebuild with "
            "`python3 -m model.scripts.prepare_neurocard_data --config <cfg> "
            "--rebuild-domains`."
        )
    if manifest.get("metadata_source") != COMPLETE_METADATA_SOURCE:
        raise ValueError(
            "NeuroCard manifest metadata_source is not complete base-table "
            "metadata; rebuild the manifest with --rebuild-domains."
        )
    metadata = ModelMetadata.from_json_dict(manifest["metadata"])
    _validate_output_slices(metadata)
    sample_path = directory / "sample_rows.npy"
    if sample_path.exists():
        rows = np.load(sample_path, mmap_mode="r")
        if rows.ndim != 2 or rows.shape[1] != len(metadata.columns):
            raise ValueError(
                f"sample_rows.npy shape {rows.shape} does not match "
                f"{len(metadata.columns)} metadata columns"
            )
        for column_index, column in enumerate(metadata.columns):
            values = rows[:, column_index]
            if np.any(values < 0) or np.any(values >= column.domain_size):
                raise ValueError(
                    f"sample_rows.npy contains encoded values outside domain "
                    f"for column {column.name!r}"
                )
    return metadata


def classify_query_literal(
    metadata: ModelMetadata,
    column_name: str,
    operator: PredicateOp,
    literal: Any,
) -> QueryLiteralClassification:
    """Classify equality and range literals without requiring range membership."""

    column = metadata.columns[metadata.column_index(column_name)]
    canonical_literal = canonicalize_base_value(literal)
    if operator == PredicateOp.EQUAL:
        if canonical_literal in column.domain:
            return QueryLiteralClassification(
                column.name,
                operator.value,
                canonical_literal,
                "present_in_complete_domain",
                "equality literal is a modeled category",
            )
        return QueryLiteralClassification(
            column.name,
            operator.value,
            canonical_literal,
            "genuinely_absent_from_complete_dataset",
            "equality predicate has zero factor because the value is absent",
        )
    if operator in {
        PredicateOp.LESS_THAN,
        PredicateOp.LESS_EQUAL,
        PredicateOp.GREATER_THAN,
        PredicateOp.GREATER_EQUAL,
        PredicateOp.RANGE,
    }:
        try:
            _ = [value for value in column.domain if _is_comparable(value, canonical_literal)]
        except TypeError:
            return QueryLiteralClassification(
                column.name,
                operator.value,
                canonical_literal,
                "invalid_schema_or_type_mismatch",
                "range literal cannot be compared with decoded domain values",
            )
        return QueryLiteralClassification(
            column.name,
            operator.value,
            canonical_literal,
            "range_threshold_not_required_to_be_domain_member",
            "range thresholds are compared against decoded domain values",
        )
    return QueryLiteralClassification(
        column.name,
        operator.value,
        canonical_literal,
        "invalid_schema_or_type_mismatch",
        f"operator {operator.value!r} is not a literal membership check",
    )


def source_csv_fingerprints(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Return quick source fingerprints using size, mtime, and edge hashes."""

    return {str(path): _fingerprint_path(Path(path)) for path in paths}


def sorted_unique(values: Sequence[Any]) -> tuple[Any, ...]:
    """Deduplicate canonical values and sort them deterministically."""

    unique = {}
    for value in values:
        unique[_json_key(value)] = value
    return tuple(sorted(unique.values(), key=_domain_sort_key))


def canonicalize_base_value(value: Any) -> Any:
    """Canonicalize source-table values while preserving SQL NULL separately."""

    if _is_missing(value):
        return SQL_NULL
    return _python_scalar(value)


def _canonicalize_sample_value(
    raw_value: Any,
    column: ColumnMetadata,
    row_index: int,
    indicator_values_by_table: Mapping[str, np.ndarray],
) -> Any:
    if column.kind == ColumnKind.INDICATOR:
        return canonicalize_indicator_value(raw_value)
    if column.kind == ColumnKind.FANOUT:
        if _is_missing(raw_value):
            return 1
        return canonicalize_fanout_value(raw_value)
    if column.kind == ColumnKind.DATA:
        if not _is_missing(raw_value):
            return canonicalize_base_value(raw_value)
        present = True
        if column.table is not None and column.table in indicator_values_by_table:
            present = bool(indicator_values_by_table[column.table][row_index])
        return SQL_NULL if present else OUTER_MISSING
    raise ValueError(f"unsupported column kind {column.kind!r}")


def canonicalize_indicator_value(value: Any) -> int:
    """Convert NeuroCard indicator values, including NaN padding, to 0/1."""

    if _is_missing(value):
        return 0
    indicator = int(value)
    if indicator not in {0, 1}:
        raise ValueError(f"indicator value must be 0 or 1, got {value!r}")
    return indicator


def _canonical_indicator_values_by_table(
    frame: Any, metadata: ModelMetadata
) -> dict[str, np.ndarray]:
    values = {}
    for column in metadata.columns:
        if column.kind != ColumnKind.INDICATOR or column.table is None:
            continue
        raw = frame[column.name].tolist()
        values[column.table] = np.array(
            [canonicalize_indicator_value(value) for value in raw], dtype=np.int64
        )
    return values


def _validate_output_slices(metadata: ModelMetadata) -> None:
    expected = sum(column.domain_size for column in metadata.columns)
    observed = metadata.output_slices[-1][1] if metadata.output_slices else 0
    if observed != expected:
        raise ValueError(
            f"metadata output slices end at {observed}, expected {expected}"
        )
    for column in metadata.columns:
        if column.kind == ColumnKind.FANOUT:
            for value in column.domain:
                if int(value) <= 0:
                    raise ValueError(
                        f"fanout column {column.name!r} contains non-positive "
                        f"value {value!r}"
                    )


def _max_group_frequency(frame: Any, keys: Sequence[str]) -> int:
    if len(frame) == 0:
        return 1
    grouped = frame.groupby(list(keys), dropna=False).size()
    maximum = int(grouped.max()) if len(grouped) else 1
    return max(1, maximum)


def _as_dataframe(table_or_frame: Any) -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ImportError("pandas is required for complete-domain preparation") from exc

    if isinstance(table_or_frame, pd.DataFrame):
        return table_or_frame
    if hasattr(table_or_frame, "data") and isinstance(table_or_frame.data, pd.DataFrame):
        return table_or_frame.data
    raise TypeError(f"expected pandas DataFrame or NeuroCard CsvTable, got {type(table_or_frame)}")


def _is_missing(value: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ModuleNotFoundError, TypeError, ValueError):
        return False


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _data_column_name(table_name: str, column_name: str) -> str:
    return f"{table_name}:{column_name}"


def _indicator_column_name(table_name: str) -> str:
    return f"__in_{table_name}"


def _fanout_column_name(table_name: str, key: str | None = None) -> str:
    if key is None:
        return f"__fanout_{table_name}"
    return f"__fanout_{table_name}__{key}"


def _domain_sort_key(value: Any) -> tuple[Any, ...]:
    if value == SQL_NULL:
        return (0, "")
    if value == OUTER_MISSING:
        return (4, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, np.integer)):
        return (1, int(value))
    if isinstance(value, (float, np.floating)):
        return (1, float(value))
    return (2, str(value))


def _json_key(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if _is_missing(value):
        return SQL_NULL
    return value


def _is_comparable(left: Any, right: Any) -> bool:
    if left in {SQL_NULL, OUTER_MISSING}:
        return False
    _ = left < right or left == right or left > right
    return True


def _fingerprint_path(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        digest.update(head)
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "edge_sha256": digest.hexdigest(),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, rows: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, rows)
    os.replace(tmp, path)
