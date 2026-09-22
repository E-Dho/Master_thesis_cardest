from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .base import Adapter, AdapterEvaluation, StageMetrics
from ..config import WorkloadConfig
from ..records import LatencyRecord, PredictionRecord, QueryRecord
from ..workloads import load_workload


class FojSamplingAdapter(Adapter):
    """Unbiased full-outer-join reservoir baseline with fanout correction."""

    def prepare(self) -> StageMetrics:
        metadata_path = self._metadata_path()
        if metadata_path.exists():
            return StageMetrics(detail={"status": "cache_hit", "metadata": str(metadata_path)})
        return StageMetrics(detail={"status": "metadata_created_during_build"})

    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        start = time.perf_counter()
        metadata, sampler = self._create_live_sampler()
        validated_query_count = self._validate_workload_columns(metadata, workloads)
        rows = self._sample_encoded_batch(sampler, metadata)
        rows = rows[: min(len(rows), int(self.config.adapter.get("smoke_sample_size", 256)))]
        if len(rows) < 2:
            raise ValueError("FOJ smoke requires at least two sampled rows")
        statuses: list[dict[str, Any]] = []
        for workload in workloads:
            cache: dict[tuple[str, ...], np.ndarray] = {}
            for query in load_workload(workload.queries_csv, workload.workload_id)[:query_limit]:
                status, estimate, diagnostic = self._estimate(query, rows, metadata, cache)
                if status == "unsupported":
                    raise ValueError(
                        f"FOJ smoke rejected {workload.workload_id} query "
                        f"{query.query_id}: {diagnostic}"
                    )
                statuses.append(
                    {
                        "workload": workload.workload_id,
                        "query_id": query.query_id,
                        "status": status,
                        "estimate": estimate,
                    }
                )
        return StageMetrics(
            wall_seconds=time.perf_counter() - start,
            detail={
                "sample_rows": len(rows),
                "validated_query_count": validated_query_count,
                "queries": statuses,
                "sampler_cache": getattr(self, "_sampler_cache_detail", {}),
            },
        )

    def build(self) -> StageMetrics:
        reservoir_path = self._reservoir_path()
        metadata_path = self._metadata_path()
        required_rows = int(self.config.adapter.get("max_sample_size", 7_168_000))
        if reservoir_path.exists() and metadata_path.exists():
            rows = np.load(reservoir_path, mmap_mode="r")
            if len(rows) < required_rows:
                raise ValueError(
                    f"reservoir has {len(rows)} rows but {required_rows} are required"
                )
            return StageMetrics(
                detail={"status": "cache_hit", "rows": len(rows), "path": str(reservoir_path)}
            )
        start = time.perf_counter()
        metadata, sampler = self._create_live_sampler()
        encoded = self._sample_encoded_batch(sampler, metadata)
        smoke_rows = encoded[: min(len(encoded), int(self.config.adapter.get("smoke_sample_size", 256)))]
        if len(smoke_rows) < 2:
            raise ValueError("FOJ pre-build smoke requires at least two sampled rows")
        smoke_query_count = 0
        for workload in self.config.workloads:
            queries = load_workload(workload.queries_csv, workload.workload_id)[:2]
            cache: dict[tuple[str, ...], np.ndarray] = {}
            for query in queries:
                status, _, diagnostic = self._estimate(query, smoke_rows, metadata, cache)
                if status == "unsupported":
                    raise ValueError(
                        f"FOJ pre-build smoke rejected {workload.workload_id} "
                        f"query {query.query_id}: {diagnostic}"
                    )
                smoke_query_count += 1
        reservoir_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.lib.format.open_memmap(
            reservoir_path,
            mode="w+",
            dtype=np.int64,
            shape=(required_rows, len(metadata.columns)),
        )
        batch_size = int(self.config.adapter.get("sampler_batch_size", 16_384))
        thresholds = sorted(
            set(int(value) for value in self.config.adapter.get(
                "sample_sizes", [1_000, 10_000, 100_000, 1_000_000, 7_168_000]
            ))
        )
        cumulative: dict[str, float] = {}
        cursor = 0
        pending: np.ndarray | None = encoded
        while cursor < required_rows:
            take = min(batch_size, required_rows - cursor)
            encoded = (
                pending
                if pending is not None
                else self._sample_encoded_batch(sampler, metadata)
            )
            pending = None
            if len(encoded) < take:
                raise ValueError("upstream sampler returned fewer rows than requested")
            previous = cursor
            target[cursor : cursor + take] = encoded[:take]
            cursor += take
            for threshold in thresholds:
                if previous < threshold <= cursor:
                    cumulative[str(threshold)] = time.perf_counter() - start
        target.flush()
        metadata_path.write_text(
            json.dumps(
                {
                    "metadata": metadata.to_json_dict(),
                    "seed": self.seed,
                    "rows": required_rows,
                    "cumulative_sampling_seconds": cumulative,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return StageMetrics(
            wall_seconds=time.perf_counter() - start,
            detail={
                "rows": required_rows,
                "reservoir": str(reservoir_path),
                "cumulative_sampling_seconds": cumulative,
                "prebuild_smoke_query_count": smoke_query_count,
            },
        )

    def evaluate(self, workloads: tuple[WorkloadConfig, ...]) -> AdapterEvaluation:
        from model.src.data.schema import ModelMetadata

        payload = json.loads(self._metadata_path().read_text(encoding="utf-8"))
        metadata = ModelMetadata.from_json_dict(payload["metadata"])
        sample_size = int(self.config.adapter.get("sample_size", 1_000))
        all_rows = np.load(self._reservoir_path(), mmap_mode="r")
        if sample_size > len(all_rows):
            raise ValueError(f"sample_size {sample_size} exceeds reservoir size {len(all_rows)}")
        # Use a seeded random subsample rather than a positional head-slice so
        # that sub-max sample sizes are unbiased random subsets of the reservoir
        # instead of biased prefixes of the sampling order.
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(all_rows), size=sample_size, replace=False)
        rows = all_rows[indices]
        warmup_passes = int(self.config.timing.get("warmup_passes", 1))
        repetitions = int(self.config.timing.get("repetitions", 10))
        predictions: list[PredictionRecord] = []
        latencies: list[LatencyRecord] = []
        cache: dict[tuple[str, ...], np.ndarray] = {}
        for workload in workloads:
            queries = load_workload(workload.queries_csv, workload.workload_id)
            for _ in range(warmup_passes):
                for query in queries:
                    self._estimate(query, rows, metadata, cache)
            first_estimates: dict[int, tuple[str, float | None, str]] = {}
            for repetition in range(repetitions):
                for query in queries:
                    start = time.perf_counter()
                    status, estimate, diagnostic = self._estimate(query, rows, metadata, cache)
                    elapsed = (time.perf_counter() - start) * 1000.0
                    latencies.append(
                        LatencyRecord(workload.workload_id, query.query_id, repetition, elapsed)
                    )
                    if repetition == 0:
                        first_estimates[query.query_id] = (status, estimate, diagnostic)
            for query in queries:
                status, estimate, diagnostic = first_estimates[query.query_id]
                predictions.append(
                    PredictionRecord(
                        workload.workload_id,
                        query.query_id,
                        status,
                        query.true_cardinality,
                        estimate,
                        diagnostic=diagnostic,
                    )
                )
        return AdapterEvaluation(
            tuple(predictions),
            tuple(latencies),
            detail={"sample_size": sample_size, "full_join_cardinality": metadata.full_join_cardinality},
        )

    def artifact_metadata(self) -> dict[str, Any]:
        reservoir = self._reservoir_path()
        sample_size = int(self.config.adapter.get("sample_size", 1_000))
        if not reservoir.exists():
            return {"parameter_count": None, "serialized_model_mb": None}
        rows = np.load(reservoir, mmap_mode="r")
        bytes_per_row = int(rows.shape[1] * rows.dtype.itemsize)
        payload = json.loads(self._metadata_path().read_text(encoding="utf-8"))
        cumulative = payload.get("cumulative_sampling_seconds", {})
        return {
            "parameter_count": None,
            "parameter_count_reason": "sampling baseline has no learned parameters",
            "serialized_model_mb": sample_size * bytes_per_row / 1_000_000.0,
            "reservoir_path": str(reservoir),
            "sample_size": sample_size,
            "cumulative_sampling_seconds": cumulative.get(str(sample_size)),
            "full_reservoir_serialized_mb": reservoir.stat().st_size / 1_000_000.0,
        }

    def _estimate(self, query: QueryRecord, rows: np.ndarray, metadata: Any, cache):
        from model.src.data.schema import ColumnKind
        from model.src.predicates.generation import inverse_fanouts_for_table_subset

        alias_to_table = {table.alias: table.name for table in query.tables}
        included = frozenset(alias_to_table.values())
        subset_key = tuple(sorted(included))
        base_weights = cache.get(subset_key)
        if base_weights is None:
            base_weights = np.ones(len(rows), dtype=np.float64)
            for index, column in enumerate(metadata.columns):
                if column.kind == ColumnKind.INDICATOR and column.table in included:
                    values = np.asarray(column.domain, dtype=object)[rows[:, index]]
                    base_weights *= values == 1
            inverse = inverse_fanouts_for_table_subset(metadata, included)
            for name in inverse:
                index = metadata.column_index(name)
                column = metadata.columns[index]
                domain = np.asarray(column.domain, dtype=np.float64)
                base_weights *= 1.0 / domain[rows[:, index]]
            cache[subset_key] = base_weights
        mask = np.ones(len(rows), dtype=bool)
        for predicate in query.filters:
            try:
                alias, source_column = predicate.column.split(".", 1)
                table = alias_to_table[alias]
                column_index = metadata.column_index(f"{table}:{source_column}")
            except (ValueError, KeyError):
                return "unsupported", None, f"unknown predicate column {predicate.column}"
            column = metadata.columns[column_index]
            allowed = _allowed_domain_ids(column.domain, predicate.operator, predicate.value)
            if allowed is None:
                return "unsupported", None, f"incomparable literal for {predicate.column}"
            if not allowed:
                return "ok", 0.0, "predicate has empty domain support"
            mask &= np.isin(rows[:, column_index], np.fromiter(allowed, dtype=np.int64))
            if not np.any(mask):
                return "ok", 0.0, "no sampled row satisfies predicates"
        estimate = estimate_from_foj(
            metadata.full_join_cardinality,
            mask,
            inverse_weight_product=base_weights,
        )
        return "ok", estimate, ""

    def _create_live_sampler(self):
        root = Path(str(self.config.adapter["neurocard_path"])).resolve()
        workdir = root.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        with _pushd(workdir):
            import datasets  # type: ignore
            import experiments  # type: ignore
            import factorized_sampler  # type: ignore
            import join_utils  # type: ignore
            from factorized_sampler_lib import data_utils, prepare_utils  # type: ignore

            from model.src.data.complete_domain_preparation import CompleteDomainSpec, build_complete_metadata

            config = dict(experiments.JOB_LIGHT_BASE)
            spec = join_utils.get_join_spec(config)
            cache_hit = bool(prepare_utils.check_required_files(spec))
            cache_directory = (
                workdir / Path(data_utils.CACHE_DIR) / str(spec.join_name)
            ).resolve()
            missing_cache_files = [
                str(cache_directory / filename)
                for table in spec.join_tables
                for filename in (f"{table}.jct", f"{table}.pk.indices")
                if not (cache_directory / filename).exists()
            ]
            self._sampler_cache_detail = {
                "cache_hit": cache_hit,
                "cache_directory": str(cache_directory),
                "join_name": str(spec.join_name),
                "missing_file_count": len(missing_cache_files),
            }
            if cache_hit:
                # Upstream prepare() calls ray.init() before checking its cache.
                # Avoid Ray entirely when the immutable Exact Weight artifacts exist.
                factorized_sampler.prepare_utils.prepare = lambda join_spec: None
            elif bool(self.config.adapter.get("require_prepared_sampler_cache", True)):
                preview = ", ".join(missing_cache_files[:4])
                raise FileNotFoundError(
                    "required NeuroCard Exact Weight sampler cache is incomplete; "
                    "refusing to start upstream Ray preparation. "
                    f"cache={cache_directory}, missing={len(missing_cache_files)}, "
                    f"first_missing=[{preview}]"
                )
            csv_directory = Path(str(self.config.adapter["csv_directory"])).resolve()
            use_cols = str(self.config.adapter.get("use_cols", config["use_cols"]))
            if use_cols == "content":
                content_columns = datasets.JoinOrderBenchmark.ContentColumns()
                required_columns = self._required_predicate_columns()
                for table, required in required_columns.items():
                    filename = f"{table}.csv"
                    configured = list(content_columns.get(filename, ()))
                    missing = sorted(required.difference(configured))
                    if missing:
                        # Prefixing keeps NeuroCard's existing parsed-table naming
                        # for JOB-light's movie_companies.company_id projection.
                        content_columns[filename] = missing + configured
                self._sampler_cache_detail["predicate_projection"] = {
                    table: list(content_columns.get(f"{table}.csv", ()))
                    for table in spec.join_tables
                }
            tables = [
                datasets.LoadImdb(
                    table,
                    data_dir=str(csv_directory) + "/",
                    use_cols=use_cols,
                    try_load_parsed=True,
                )
                for table in spec.join_tables
            ]
            table_by_name = {table.name: table for table in tables}
            join_cardinality = float(
                datasets.JoinOrderBenchmark.GetFullOuterCardinalityOrFail(spec.join_tables)
            )
            complete_spec = CompleteDomainSpec(
                join_tables=tuple(spec.join_tables),
                join_root=spec.join_root,
                join_keys={table: tuple(keys) for table, keys in spec.join_keys.items()},
                join_cardinality=join_cardinality,
                dataset_name="job_light_foj_sampling",
                dataset_type="neurocard_full_join",
            )
            metadata = build_complete_metadata(table_by_name, complete_spec)
            sampler = factorized_sampler.FactorizedSampler(
                tables,
                spec,
                int(self.config.adapter.get("sampler_batch_size", 16_384)),
                rng=np.random.default_rng(self.seed),
                disambiguate_column_names=True,
            )
        return metadata, sampler

    def _required_predicate_columns(self) -> dict[str, set[str]]:
        required: dict[str, set[str]] = {}
        for workload in self.config.workloads:
            for query in load_workload(workload.queries_csv, workload.workload_id):
                alias_to_table = {table.alias: table.name for table in query.tables}
                for predicate in query.filters:
                    alias, source_column = predicate.column.split(".", 1)
                    required.setdefault(alias_to_table[alias], set()).add(source_column)
        return required

    def _validate_workload_columns(
        self,
        metadata: Any,
        workloads: tuple[WorkloadConfig, ...],
    ) -> int:
        available = {column.name for column in metadata.columns}
        query_count = 0
        missing: set[str] = set()
        for workload in workloads:
            for query in load_workload(workload.queries_csv, workload.workload_id):
                query_count += 1
                alias_to_table = {table.alias: table.name for table in query.tables}
                for predicate in query.filters:
                    alias, source_column = predicate.column.split(".", 1)
                    name = f"{alias_to_table[alias]}:{source_column}"
                    if name not in available:
                        missing.add(name)
        if missing:
            raise ValueError(
                "FOJ metadata does not cover workload predicate columns: "
                + ", ".join(sorted(missing))
            )
        return query_count

    def _sample_encoded_batch(self, sampler: Any, metadata: Any) -> np.ndarray:
        from model.src.data.complete_domain_preparation import encode_sample_dataframe

        root = Path(str(self.config.adapter["neurocard_path"])).resolve()
        with _pushd(root.parent):
            frame = sampler.run()
        return encode_sample_dataframe(frame, metadata, strict=True).encoded_rows

    def _reservoir_path(self) -> Path:
        value = str(self.config.adapter.get("reservoir_path", "foj_reservoir_seed_{seed}.npy"))
        return self._resolve(value.format(seed=self.seed))

    def _metadata_path(self) -> Path:
        value = str(self.config.adapter.get("metadata_path", "foj_reservoir_seed_{seed}.json"))
        return self._resolve(value.format(seed=self.seed))

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.config.source_path.parent / path
        return path.resolve()


def _allowed_domain_ids(domain: tuple[Any, ...], operator: str, literal: Any) -> set[int] | None:
    result: set[int] = set()
    try:
        for index, value in enumerate(domain):
            if isinstance(value, str) and value.startswith("__"):
                continue
            if operator == "=" and value == literal:
                result.add(index)
            elif operator == "<" and value < literal:
                result.add(index)
            elif operator == "<=" and value <= literal:
                result.add(index)
            elif operator == ">" and value > literal:
                result.add(index)
            elif operator == ">=" and value >= literal:
                result.add(index)
            elif operator not in {"=", "<", "<=", ">", ">="}:
                return None
    except TypeError:
        return None
    return result


def estimate_from_foj(
    full_join_cardinality: float,
    predicate_match: np.ndarray,
    *,
    included_indicator_product: np.ndarray | None = None,
    inverse_weight_product: np.ndarray | None = None,
) -> float:
    predicate = np.asarray(predicate_match, dtype=np.float64)
    weights = np.ones(len(predicate), dtype=np.float64)
    if included_indicator_product is not None:
        weights *= np.asarray(included_indicator_product, dtype=np.float64)
    if inverse_weight_product is not None:
        weights *= np.asarray(inverse_weight_product, dtype=np.float64)
    return float(full_join_cardinality * np.mean(predicate * weights))


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
