from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata

OUTER_MISSING = "__OUTER_MISSING__"


@dataclass(frozen=True)
class SyntheticDataset:
    """Tiny three-table chain with duplicates, unmatched rows, and fanout skew."""

    metadata: ModelMetadata
    decoded_rows: tuple[tuple[object, ...], ...]
    encoded_rows: np.ndarray


@dataclass(frozen=True)
class FullJoinBatch:
    """One batch from a full-outer-join sample source in canonical ordering."""

    encoded_values: np.ndarray
    column_metadata: tuple[ColumnMetadata, ...]
    raw_values: tuple[tuple[object, ...], ...] | None = None
    fresh_rows_drawn: int = 0
    fixture_rows_reused: int = 0
    importance_weights: np.ndarray | None = None
    importance_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SamplerInspection:
    join_cardinality: int
    column_order: tuple[str, ...]
    column_types: tuple[str, ...]
    domain_sizes: tuple[int, ...]
    indicator_frequencies: dict[str, float]
    fanout_domains: dict[str, tuple[object, ...]]
    fanout_min_max: dict[str, tuple[float, float]]
    padded_percentages: dict[str, float]
    decoded_sample_rows: tuple[tuple[object, ...], ...]


class FullJoinSampler(Protocol):
    """Interface expected from a NeuroCard-style uniform full-join sampler."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        ...


class FullJoinSampleSource(Protocol):
    @property
    def join_cardinality(self) -> int:
        ...

    @property
    def metadata(self) -> ModelMetadata:
        ...

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        ...


class SyntheticFullJoinSampleSource:
    """Deterministic sample source over the materialized synthetic oracle rows."""

    def __init__(self, dataset: SyntheticDataset | None = None) -> None:
        self.dataset = dataset or build_synthetic_chain_dataset()

    @property
    def join_cardinality(self) -> int:
        return int(self.dataset.metadata.full_join_cardinality)

    @property
    def metadata(self) -> ModelMetadata:
        return self.dataset.metadata

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.dataset.encoded_rows), size=batch_size)
        raw_rows = tuple(self.dataset.decoded_rows[index] for index in indices)
        return FullJoinBatch(
            encoded_values=self.dataset.encoded_rows[indices],
            column_metadata=self.dataset.metadata.columns,
            raw_values=raw_rows,
            fixture_rows_reused=batch_size,
        )

    def inspect(self, *, sample_rows: int = 5) -> SamplerInspection:
        return inspect_encoded_rows(self.dataset.metadata, self.dataset.encoded_rows, self.dataset.decoded_rows[:sample_rows])


class NeuroCardFullJoinSamplerAdapter:
    """Boundary for wiring NeuroCard's sampler without vendoring it here."""

    def sample_encoded_rows(self, num_rows: int, seed: int) -> np.ndarray:
        raise NotImplementedError(
            "Production NeuroCard sampler integration is the next milestone. "
            "This milestone validates the tuple contract with a synthetic oracle."
        )


class NeuroCardFullJoinSampleSource:
    """Manifest-backed fixture source for deterministic smoke-test sampling."""

    def __init__(self, prepared_directory: str | Path, *, sampling_mode: str = "fixture") -> None:
        self.prepared_directory = Path(prepared_directory)
        self.sampling_mode = sampling_mode
        manifest_path = self.prepared_directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"missing NeuroCard preparation manifest: {manifest_path}. "
                "Run python3 -m model.scripts.prepare_neurocard_data --config <config>."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from model.src.data.complete_domain_preparation import validate_prepared_manifest

        validate_prepared_manifest(self.prepared_directory)
        self._metadata = ModelMetadata.from_json_dict(self.manifest["metadata"])
        self.sample_batches_generated = 0
        self.fixture_rows_reused = 0
        self.fresh_rows_drawn = 0

    @property
    def join_cardinality(self) -> int:
        return int(self.manifest["join_cardinality"])

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        sample_path = self.prepared_directory / "sample_rows.npy"
        if self.sampling_mode == "live":
            raise NotImplementedError(
                "dataset.sampling_mode=live requires wiring NeuroCard's "
                "FactorizedSampler/FactorizedSamplerIterDataset against the fixed "
                "complete-domain manifest. Use sampling_mode=fixture for smoke tests."
            )
        if self.sampling_mode not in {"fixture", "materialized_large_sample"}:
            raise ValueError(f"unsupported NeuroCard sampling mode {self.sampling_mode!r}")
        if sample_path.exists():
            rows = np.load(sample_path)
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(rows), size=batch_size)
            self.sample_batches_generated += 1
            self.fixture_rows_reused += int(batch_size)
            return FullJoinBatch(
                rows[indices],
                self.metadata.columns,
                fresh_rows_drawn=0,
                fixture_rows_reused=batch_size,
            )
        raise NotImplementedError(
            "No local sample_rows.npy fixture is available. Wire NeuroCard's "
            "FactorizedSampler/FactorizedSamplerIterDataset here after its Rust "
            "join-count/index artifacts exist in the prepared directory."
        )


class LiveNeuroCardFullJoinSampleSource(NeuroCardFullJoinSampleSource):
    """Draw fresh Exact Weight full-join batches from an external NeuroCard checkout.

    The model metadata and domains remain loaded from the complete preparation
    manifest. Sampled rows are encoded against that fixed manifest; out-of-domain
    sampled values fail instead of mutating metadata.
    """

    def __init__(
        self,
        prepared_directory: str | Path,
        *,
        csv_directory: str | Path,
        neurocard_path: str | Path | None = None,
        sampler_batch_size: int = 16384,
        seed: int = 0,
        startup_callback: Any | None = None,
    ) -> None:
        super().__init__(prepared_directory, sampling_mode="live")
        self.csv_directory = Path(csv_directory).resolve()
        if not self.csv_directory.exists():
            raise FileNotFoundError(f"missing JOB-light CSV directory {self.csv_directory}")
        self.neurocard_path = _resolve_neurocard_package(neurocard_path)
        self.neurocard_workdir = self.neurocard_path.parent
        self.sampler_batch_size = int(sampler_batch_size)
        if self.sampler_batch_size <= 0:
            raise ValueError("sampler_batch_size must be positive")
        self.seed = int(seed)
        self.sampler_run_calls = 0
        self.conditional_sampler_batch_calls = 0
        self.conditional_rows_drawn = 0
        self._buffer: np.ndarray | None = None
        self._buffer_cursor = 0
        self._sampler: Any | None = None
        self._factorized_sampler_module: Any | None = None
        self._root_jct_value_cache: dict[str, np.ndarray] = {}
        self._root_jct_weight_cache: dict[str, np.ndarray] = {}
        self._root_stratum_candidate_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._startup_callback = startup_callback
        self._startup_event(
            "neurocard_path_resolved",
            {
                "csv_directory": str(self.csv_directory),
                "neurocard_path": str(self.neurocard_path),
                "neurocard_workdir": str(self.neurocard_workdir),
            },
        )
        self._load_neurocard_sampler()

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pieces: list[np.ndarray] = []
        fresh_rows_drawn = 0
        remaining = int(batch_size)
        while remaining > 0:
            available = 0 if self._buffer is None else len(self._buffer) - self._buffer_cursor
            if available <= 0:
                fresh_rows_drawn += self._draw_sampler_batch()
                available = len(self._buffer) - self._buffer_cursor  # type: ignore[arg-type]
            take = min(remaining, available)
            assert self._buffer is not None
            pieces.append(self._buffer[self._buffer_cursor : self._buffer_cursor + take])
            self._buffer_cursor += take
            remaining -= take
        encoded = np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
        self.fresh_rows_drawn += int(fresh_rows_drawn)
        return FullJoinBatch(
            encoded_values=encoded,
            column_metadata=self.metadata.columns,
            fresh_rows_drawn=fresh_rows_drawn,
            fixture_rows_reused=0,
        )

    @property
    def distinct_original_rows_seen_estimate(self) -> None:
        return None

    def discard_buffer(self) -> None:
        self._buffer = None
        self._buffer_cursor = 0

    def _load_neurocard_sampler(self) -> None:
        neurocard_path = str(self.neurocard_path)
        if neurocard_path not in sys.path:
            sys.path.insert(0, neurocard_path)
        with _pushd(self.neurocard_workdir):
            import datasets  # type: ignore
            import experiments  # type: ignore
            import factorized_sampler  # type: ignore
            import join_utils  # type: ignore
            from factorized_sampler_lib import prepare_utils  # type: ignore

            self._startup_event("neurocard_imports_loaded")
            cfg = experiments.JOB_LIGHT_BASE
            spec = join_utils.get_join_spec(cfg)
            self._startup_event(
                "join_spec_loaded",
                {
                    "join_root": getattr(spec, "join_root", None),
                    "join_tables": list(getattr(spec, "join_tables", ())),
                },
            )
            if prepare_utils.check_required_files(spec):
                self._startup_event(
                    "prepare_cache_hit",
                    {
                        "join_name": getattr(spec, "join_name", None),
                    },
                )
                factorized_sampler.prepare_utils.prepare = lambda join_spec: None
            tables = [
                datasets.LoadImdb(
                    table,
                    data_dir=str(self.csv_directory) + "/",
                    use_cols=cfg["use_cols"],
                    try_load_parsed=True,
                )
                for table in spec.join_tables
            ]
            self._startup_event(
                "tables_loaded",
                {
                    "table_count": len(tables),
                    "tables": list(getattr(spec, "join_tables", ())),
                },
            )
            self._startup_event(
                "factorized_sampler_construction_started",
                {
                    "sampler_batch_size": max(self.sampler_batch_size, 1),
                },
            )
            self._sampler = factorized_sampler.FactorizedSampler(
                tables,
                spec,
                max(self.sampler_batch_size, 1),
                rng=np.random.default_rng(self.seed),
                disambiguate_column_names=True,
            )
            self._startup_event(
                "factorized_sampler_constructed",
                {
                    "live_join_cardinality": int(self._sampler.join_card),
                },
            )
            if int(self._sampler.join_card) != int(self.join_cardinality):
                raise ValueError(
                    f"live sampler join cardinality {self._sampler.join_card} does not "
                    f"match manifest join cardinality {self.join_cardinality}"
                )
            self._factorized_sampler_module = factorized_sampler
            self._startup_event(
                "live_sampler_ready",
                {
                    "manifest_join_cardinality": int(self.join_cardinality),
                },
            )

    def _draw_sampler_batch(self) -> int:
        from model.src.data.complete_domain_preparation import encode_sample_dataframe

        if self._sampler is None:
            raise RuntimeError("live NeuroCard sampler was not initialized")
        with _pushd(self.neurocard_workdir):
            sample_frame = self._sampler.run()
        encoded_sample = encode_sample_dataframe(sample_frame, self.metadata, strict=True)
        self._buffer = encoded_sample.encoded_rows
        self._buffer_cursor = 0
        self.sampler_run_calls += 1
        self.sample_batches_generated += 1
        return int(self._buffer.shape[0])

    def sample_root_stratum_rows(
        self,
        stratum: Any,
        num_rows: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw exact NeuroCard rows conditional on a supported root-table stratum.

        NeuroCard's Exact Weight sampler first samples a row from the root join
        count table with probability proportional to ``{root}.weight`` and then
        samples child join-count rows using ``IndexProvider.sample_indices``.
        For root-table strata we can condition exactly by replacing only the
        root distribution with the root-weight distribution restricted to the
        stratum, then reusing the existing child JCT and data-table actors.
        """

        from model.src.data.complete_domain_preparation import encode_sample_dataframe

        if self._sampler is None:
            raise RuntimeError("live NeuroCard sampler was not initialized")
        if num_rows <= 0:
            return np.empty((0, len(self.metadata.columns)), dtype=np.int64)
        candidate_indices, weights = self._root_stratum_candidates(stratum)
        selected = rng.choice(
            candidate_indices,
            size=int(num_rows),
            replace=True,
            p=weights / float(weights.sum()),
        )
        root = self._sampler.join_spec.join_root
        root_jct = self._sampler.jct_actors[root].jct
        with _pushd(self.neurocard_workdir):
            sample = root_jct.iloc[selected].reset_index(drop=True)
            for table in self._sampler.sampling_tables_ordering[1:]:
                sample = self._sampler.jct_actors[table].take_sample(
                    sample,
                    int(num_rows),
                    rng,
                )
            frame = self._sampler._construct_complete_sample(sample)
            frame = self._sampler._rearrange_columns(frame)
            frame.replace(-1, np.nan, inplace=True)
        encoded_sample = encode_sample_dataframe(frame, self.metadata, strict=True)
        self.conditional_sampler_batch_calls += 1
        self.conditional_rows_drawn += int(num_rows)
        self.sample_batches_generated += 1
        self.fresh_rows_drawn += int(num_rows)
        return encoded_sample.encoded_rows

    def sample_root_strata_rows(
        self,
        strata: Any,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw one conditional row for each stratum in a single child-sampling pass."""

        from model.src.data.complete_domain_preparation import encode_sample_dataframe

        if self._sampler is None:
            raise RuntimeError("live NeuroCard sampler was not initialized")
        strata = tuple(strata)
        if not strata:
            return np.empty((0, len(self.metadata.columns)), dtype=np.int64)
        root = self._sampler.join_spec.join_root
        root_jct = self._sampler.jct_actors[root].jct
        selected_indices = np.empty(len(strata), dtype=int)
        for row_index, stratum in enumerate(strata):
            candidate_indices, weights = self._root_stratum_candidates(stratum)
            selected_indices[row_index] = int(
                rng.choice(
                    candidate_indices,
                    replace=True,
                    p=weights / float(weights.sum()),
                )
            )
        with _pushd(self.neurocard_workdir):
            sample = root_jct.iloc[selected_indices].reset_index(drop=True)
            for table in self._sampler.sampling_tables_ordering[1:]:
                sample = self._sampler.jct_actors[table].take_sample(
                    sample,
                    len(strata),
                    rng,
                )
            frame = self._sampler._construct_complete_sample(sample)
            frame = self._sampler._rearrange_columns(frame)
            frame.replace(-1, np.nan, inplace=True)
        encoded_sample = encode_sample_dataframe(frame, self.metadata, strict=True)
        self.conditional_sampler_batch_calls += 1
        self.conditional_rows_drawn += len(strata)
        self.sample_batches_generated += 1
        self.fresh_rows_drawn += len(strata)
        return encoded_sample.encoded_rows

    def prepare_root_strata(self, strata: Any) -> None:
        """Precompute root-JCT candidate distributions for selected strata."""

        for stratum in strata:
            self._root_stratum_candidates(stratum)

    def _root_stratum_candidates(self, stratum: Any) -> tuple[np.ndarray, np.ndarray]:
        if self._sampler is None:
            raise RuntimeError("live NeuroCard sampler was not initialized")
        cached = self._root_stratum_candidate_cache.get(stratum.stratum_id)
        if cached is not None:
            return cached
        root = self._sampler.join_spec.join_root
        root_jct = self._sampler.jct_actors[root].jct
        root_weight_column = f"{root}.weight"
        values = self._root_jct_values_for_stratum(stratum)
        mask = _vectorized_root_stratum_mask(values, stratum)
        root_weights = self._root_jct_weight_cache.get(root)
        if root_weights is None:
            root_weights = root_jct[root_weight_column].to_numpy(dtype=float)
            self._root_jct_weight_cache[root] = root_weights
        weights = root_weights[mask]
        if weights.size == 0 or float(weights.sum()) <= 0.0:
            raise ValueError(f"stratum {stratum.stratum_id!r} has no positive root JCT mass")
        restricted_root_mass = float(weights.sum())
        expected_mass = float(getattr(stratum, "foj_count", 0.0))
        tolerance = max(1.0e-6, max(abs(restricted_root_mass), abs(expected_mass)) * 1.0e-10)
        if abs(restricted_root_mass - expected_mass) > tolerance:
            absolute_difference = abs(restricted_root_mass - expected_mass)
            relative_difference = (
                absolute_difference / abs(expected_mass)
                if expected_mass
                else float("inf")
            )
            raise ValueError(
                "conditional root stratum candidate mass does not match discovery mass: "
                f"stratum_id={stratum.stratum_id!r}, "
                f"stratum_foj_count={expected_mass}, "
                f"restricted_root_mass={restricted_root_mass}, "
                f"absolute_difference={absolute_difference}, "
                f"relative_difference={relative_difference}"
            )
        candidate_indices = np.flatnonzero(mask)
        cached = (candidate_indices, weights)
        self._root_stratum_candidate_cache[stratum.stratum_id] = cached
        return cached

    def _root_jct_values_for_stratum(self, stratum: Any) -> np.ndarray:
        if self._sampler is None:
            raise RuntimeError("live NeuroCard sampler was not initialized")
        root = self._sampler.join_spec.join_root
        source_column = stratum.column_name.split(":", 1)[1]
        data_column = f"{root}.{source_column}"
        cached = self._root_jct_value_cache.get(data_column)
        if cached is not None:
            return cached
        root_jct = self._sampler.jct_actors[root].jct
        table_actor = next(actor for actor in self._sampler.dt_actors if actor.table == root)
        if data_column not in table_actor.df.columns:
            raise ValueError(f"root data column {data_column!r} is unavailable in NeuroCard table")
        if len(table_actor.join_keys) != 1:
            raise NotImplementedError("root stratum sampling currently supports one root join key")
        root_key = table_actor.join_keys[0]
        if table_actor.df[root_key].duplicated().any():
            raise NotImplementedError(
                "root stratum sampling requires a unique root join key so root JCT "
                "weights can be restricted by root DATA values exactly"
            )
        from model.src.data.complete_domain_preparation import canonicalize_base_value

        root_values = table_actor.df[[root_key, data_column]]
        merged = root_jct[[root_key]].merge(root_values, how="left", on=root_key, indicator=True)
        raw_values = merged[data_column].to_numpy(dtype=object)
        merge_states = merged["_merge"].to_numpy(dtype=object)
        if np.all(merge_states == "both"):
            try:
                cached_values = raw_values.astype(float)
                self._root_jct_value_cache[data_column] = cached_values
                return cached_values
            except (TypeError, ValueError):
                pass
        values = np.array(
            [
                (
                    canonicalize_base_value(raw_value)
                    if matched == "both"
                    else OUTER_MISSING
                )
                for raw_value, matched in zip(raw_values, merge_states)
            ],
            dtype=object,
        )
        try:
            cached_values = values.astype(float)
        except (TypeError, ValueError):
            cached_values = values
        self._root_jct_value_cache[data_column] = cached_values
        return cached_values

    def _startup_event(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._startup_callback is not None:
            self._startup_callback(name, payload or {})


def _resolve_neurocard_package(explicit_path: str | Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if os.environ.get("NEUROCARD_PATH"):
        candidates.append(Path(os.environ["NEUROCARD_PATH"]))
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "external/neurocard/neurocard",
            cwd.parent / "external/neurocard/neurocard",
            Path("/work_beegfs/sunip956/master_thesis_trajectories/external/neurocard/neurocard"),
        ]
    )
    for candidate in candidates:
        if (candidate / "datasets.py").exists() and (candidate / "factorized_sampler.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not locate NeuroCard package. Set dataset.neurocard_path or "
        "NEUROCARD_PATH to the directory containing datasets.py."
    )


@contextmanager
def _pushd(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _root_jct_mask_for_stratum(sampler: Any, root_jct: Any, stratum: Any) -> np.ndarray:
    root = sampler.join_spec.join_root
    table_actor = next(actor for actor in sampler.dt_actors if actor.table == root)
    source_column = stratum.column_name.split(":", 1)[1]
    data_column = f"{root}.{source_column}"
    if data_column not in table_actor.df.columns:
        raise ValueError(f"root data column {data_column!r} is unavailable in NeuroCard table")
    if len(table_actor.join_keys) != 1:
        raise NotImplementedError("root stratum sampling currently supports one root join key")
    root_key = table_actor.join_keys[0]
    if table_actor.df[root_key].duplicated().any():
        raise NotImplementedError(
            "root stratum sampling requires a unique root join key so root JCT "
            "weights can be restricted by root DATA values exactly"
        )
    from model.src.data.complete_domain_preparation import canonicalize_base_value

    values = table_actor.df[[root_key, data_column]]
    merged = root_jct[[root_key]].merge(values, how="left", on=root_key, indicator=True)
    column_values = np.array(
        [
            canonicalize_base_value(raw_value) if matched == "both" else OUTER_MISSING
            for raw_value, matched in zip(
                merged[data_column].to_numpy(dtype=object),
                merged["_merge"].to_numpy(dtype=object),
            )
        ],
        dtype=object,
    )
    return np.array([stratum.contains_value(value) for value in column_values], dtype=bool)


def _vectorized_root_stratum_mask(values: np.ndarray, stratum: Any) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        numeric = values.astype(float, copy=False)
        if stratum.region_type == "equality":
            return numeric == float(stratum.value)
        if stratum.region_type == "lower_tail":
            return numeric >= float(stratum.lower)
        if stratum.region_type == "upper_tail":
            return numeric <= float(stratum.upper)
        if stratum.region_type == "range":
            return (numeric >= float(stratum.lower)) & (numeric <= float(stratum.upper))
    return np.array([stratum.contains_value(value) for value in values], dtype=bool)


def _encode_rows(metadata: ModelMetadata, rows: tuple[tuple[object, ...], ...]) -> np.ndarray:
    encoded = np.zeros((len(rows), len(metadata.columns)), dtype=int)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            encoded[row_index, column_index] = metadata.columns[column_index].encode_value(value)
    return encoded


def canonicalize_fanout_value(value: object, *, outer_padding: bool = False) -> int:
    """Convert known neutral outer padding to fanout 1 and reject malformed values."""

    if value is None and outer_padding:
        return 1
    fanout_value = int(value)
    if fanout_value <= 0:
        raise ValueError(f"fanout values must be strictly positive, got {value!r}")
    return fanout_value


def build_synthetic_chain_dataset() -> SyntheticDataset:
    """Materialize a small full outer join for exact validation only.

    Column order is data, indicators, then fanouts. Fanout columns are positive
    effective fanouts, including value 1 for unmatched outer-join branches.
    """

    columns = (
        ColumnMetadata("A.value", ColumnKind.DATA, ("a1", "a2", OUTER_MISSING), table="A"),
        ColumnMetadata("B.value", ColumnKind.DATA, ("b1", "b2", "b3", OUTER_MISSING), table="B"),
        ColumnMetadata("C.value", ColumnKind.DATA, ("c1", "c2", OUTER_MISSING), table="C"),
        ColumnMetadata("I_A", ColumnKind.INDICATOR, (0, 1), table="A"),
        ColumnMetadata("I_B", ColumnKind.INDICATOR, (0, 1), table="B"),
        ColumnMetadata("I_C", ColumnKind.INDICATOR, (0, 1), table="C"),
        ColumnMetadata("F_A_to_B", ColumnKind.FANOUT, (1, 2), table="A", fanout_source="A->B"),
        ColumnMetadata("F_B_to_C", ColumnKind.FANOUT, (1, 2, 10), table="B", fanout_source="B->C"),
    )
    rows = (
        ("a1", "b1", "c1", 1, 1, 1, 2, 10),
        ("a1", "b1", "c2", 1, 1, 1, 2, 10),
        ("a1", "b2", "c1", 1, 1, 1, 2, 1),
        ("a2", OUTER_MISSING, OUTER_MISSING, 1, 0, 0, 1, 1),
        (OUTER_MISSING, "b3", OUTER_MISSING, 0, 1, 0, 1, 1),
    )
    metadata = ModelMetadata(
        columns=columns,
        full_join_cardinality=float(len(rows)),
        join_root="A",
        join_tables=("A", "B", "C"),
        join_edges=(("A", "B"), ("B", "C")),
        upstream_attribution={
            "NeuroCard": "full-outer-join tuples, indicators, and fanout semantics",
            "Duet": "predicate-conditioned virtual-token inference",
            "DistJoin": "future factorized output adapter boundary only",
        },
    )
    return SyntheticDataset(metadata, rows, _encode_rows(metadata, rows))


def inspect_encoded_rows(
    metadata: ModelMetadata,
    encoded_rows: np.ndarray,
    decoded_sample_rows: tuple[tuple[object, ...], ...] = (),
) -> SamplerInspection:
    """Summarize ordering, indicators, fanouts, padding, and decoded examples."""

    indicator_frequencies = {}
    padded_percentages = {}
    fanout_domains = {}
    fanout_min_max = {}
    for column_index, column in enumerate(metadata.columns):
        values = encoded_rows[:, column_index]
        decoded_values = np.array([column.domain[index] for index in values], dtype=object)
        if column.kind == ColumnKind.INDICATOR:
            present = np.mean(decoded_values.astype(float))
            indicator_frequencies[column.name] = float(present)
            if column.table is not None:
                padded_percentages[column.table] = float(100.0 * (1.0 - present))
        elif column.kind == ColumnKind.FANOUT:
            numeric = decoded_values.astype(float)
            fanout_domains[column.name] = column.domain
            fanout_min_max[column.name] = (float(np.min(numeric)), float(np.max(numeric)))
    return SamplerInspection(
        join_cardinality=int(metadata.full_join_cardinality),
        column_order=tuple(column.name for column in metadata.columns),
        column_types=tuple(column.kind.value for column in metadata.columns),
        domain_sizes=tuple(column.domain_size for column in metadata.columns),
        indicator_frequencies=indicator_frequencies,
        fanout_domains=fanout_domains,
        fanout_min_max=fanout_min_max,
        padded_percentages=padded_percentages,
        decoded_sample_rows=decoded_sample_rows,
    )
