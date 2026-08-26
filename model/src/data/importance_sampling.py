from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from model.src.data.full_join_sampler import FullJoinBatch, SyntheticFullJoinSampleSource
from model.src.data.strata import (
    ExactRootStratumProvider,
    RootDataStratum,
    membership_matrix,
    predicate_probability_map,
    rho_for_memberships,
)
from model.src.predicates.operators import PredicateOp, PredicateToken


@dataclass
class StreamingWeightStats:
    """Streaming moments plus a bounded deterministic reservoir for percentiles."""

    reservoir_size: int = 100_000
    seed: int = 0
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    reservoir: list[float] = field(default_factory=list)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def update(self, values: np.ndarray | list[float] | tuple[float, ...]) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return
        if np.any(array < 0.0) or not np.all(np.isfinite(array)):
            raise ValueError("streaming weight statistics require finite nonnegative values")
        self.total += float(np.sum(array))
        self.total_squared += float(np.dot(array, array))
        self.minimum = min(self.minimum, float(np.min(array)))
        self.maximum = max(self.maximum, float(np.max(array)))
        for value in array:
            self.count += 1
            if len(self.reservoir) < self.reservoir_size:
                self.reservoir.append(float(value))
                continue
            replacement = int(self._rng.integers(0, self.count))
            if replacement < self.reservoir_size:
                self.reservoir[replacement] = float(value)

    def to_json_dict(self) -> dict[str, float]:
        if self.count == 0:
            return _empty_weight_summary()
        mean = self.total / float(self.count)
        variance = max(0.0, self.total_squared / float(self.count) - mean * mean)
        sample = np.asarray(self.reservoir, dtype=float)
        percentiles = (
            {
                "p50": float(np.percentile(sample, 50)),
                "p90": float(np.percentile(sample, 90)),
                "p95": float(np.percentile(sample, 95)),
                "p99": float(np.percentile(sample, 99)),
                "p999": float(np.percentile(sample, 99.9)),
            }
            if sample.size
            else {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0}
        )
        return {
            "count": int(self.count),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
            "sum": float(self.total),
            "sum_squared": float(self.total_squared),
            "ess": (
                float(self.total * self.total / self.total_squared)
                if self.total_squared > 0.0
                else 0.0
            ),
            "percentile_reservoir_size": int(len(self.reservoir)),
            **percentiles,
        }


@dataclass
class FanoutConditionalContextStats:
    inv_only: StreamingWeightStats = field(default_factory=StreamingWeightStats)
    importance_times_inv: StreamingWeightStats = field(default_factory=StreamingWeightStats)
    relevant_inv_only: StreamingWeightStats = field(default_factory=StreamingWeightStats)
    relevant_importance_times_inv: StreamingWeightStats = field(default_factory=StreamingWeightStats)

    def update(self, inv_value: float, combined_value: float, *, relevant: bool) -> None:
        self.inv_only.update([inv_value])
        self.importance_times_inv.update([combined_value])
        if relevant:
            self.relevant_inv_only.update([inv_value])
            self.relevant_importance_times_inv.update([combined_value])

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "inv_only": self.inv_only.to_json_dict(),
            "importance_times_inv": self.importance_times_inv.to_json_dict(),
            "stratum_relevant_inv_only": self.relevant_inv_only.to_json_dict(),
            "stratum_relevant_importance_times_inv": (
                self.relevant_importance_times_inv.to_json_dict()
            ),
        }


@dataclass
class StratumPredicateContextStats:
    context_count: int = 0
    relevant_context_count: int = 0
    root_predicate_operator_count: Counter[str] = field(default_factory=Counter)
    stratum_relevant_operator_count: Counter[str] = field(default_factory=Counter)
    boundary_literal_count: Counter[str] = field(default_factory=Counter)
    fanouts: dict[str, FanoutConditionalContextStats] = field(
        default_factory=lambda: defaultdict(FanoutConditionalContextStats)
    )

    def update(
        self,
        *,
        token: PredicateToken,
        stratum: RootDataStratum,
        fanout_values: Mapping[str, tuple[float, float]],
    ) -> None:
        relevant = _token_relevant_to_stratum(token, stratum)
        self.context_count += 1
        self.root_predicate_operator_count[token.op.value] += 1
        if relevant:
            self.relevant_context_count += 1
            self.stratum_relevant_operator_count[token.op.value] += 1
        for literal in (token.value, token.upper):
            if literal is not None:
                self.boundary_literal_count[_bounded_literal_key(literal)] += 1
        for fanout_name, (inv_value, combined_value) in fanout_values.items():
            self.fanouts[fanout_name].update(
                float(inv_value),
                float(combined_value),
                relevant=relevant,
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "context_count": int(self.context_count),
            "stratum_relevant_context_count": int(self.relevant_context_count),
            "root_predicate_operator_count": dict(self.root_predicate_operator_count),
            "stratum_relevant_operator_count": dict(self.stratum_relevant_operator_count),
            "boundary_literal_count": dict(self.boundary_literal_count.most_common(256)),
            "fanout_effective_sample_size": {
                name: stats.to_json_dict() for name, stats in self.fanouts.items()
            },
        }


@dataclass
class ImportanceSamplingRunningStats:
    """Audit statistics for q(x) sampling and exact p/q ratios."""

    enabled: bool
    mixture_probability: float
    selected_strata: tuple[RootDataStratum, ...]
    uniform_component_samples: int = 0
    rare_component_samples: int = 0
    selected_stratum_sample_count: Counter[str] = field(default_factory=Counter)
    row_membership_count: Counter[str] = field(default_factory=Counter)
    membership_patterns: Counter[str] = field(default_factory=Counter)
    multiple_membership_samples: int = 0
    rho_stats: StreamingWeightStats = field(default_factory=StreamingWeightStats)
    rho_stats_by_stratum: dict[str, StreamingWeightStats] = field(default_factory=dict)
    context_stats_by_stratum: dict[str, StratumPredicateContextStats] = field(
        default_factory=dict
    )
    ordinary_sampler_seconds: float = 0.0
    conditional_sampler_seconds: float = 0.0
    proposal_selection_seconds: float = 0.0
    importance_weight_seconds: float = 0.0

    def update_batch(
        self,
        *,
        component: np.ndarray,
        selected: list[str | None],
        memberships: np.ndarray,
        rho: np.ndarray,
        timing: Mapping[str, float],
    ) -> None:
        self.uniform_component_samples += int(np.sum(component == "uniform"))
        self.rare_component_samples += int(np.sum(component == "rare"))
        for stratum_id in selected:
            if stratum_id is not None:
                self.selected_stratum_sample_count[stratum_id] += 1
        for row_index in range(memberships.shape[0]):
            ids = [
                self.selected_strata[index].stratum_id
                for index in np.flatnonzero(memberships[row_index])
            ]
            if len(ids) > 1:
                self.multiple_membership_samples += 1
            key = ",".join(ids) if ids else "<none>"
            self.membership_patterns[key] += 1
            for stratum_id in ids:
                self.row_membership_count[stratum_id] += 1
        self.rho_stats.update(rho)
        for stratum_index, stratum in enumerate(self.selected_strata):
            values = rho[memberships[:, stratum_index]]
            if values.size:
                self.rho_stats_by_stratum.setdefault(
                    stratum.stratum_id,
                    StreamingWeightStats(seed=stratum_index + 1),
                ).update(values)
        self.ordinary_sampler_seconds += float(timing.get("ordinary_sampler_seconds", 0.0))
        self.conditional_sampler_seconds += float(timing.get("conditional_sampler_seconds", 0.0))
        self.proposal_selection_seconds += float(timing.get("proposal_selection_seconds", 0.0))
        self.importance_weight_seconds += float(timing.get("importance_weight_seconds", 0.0))

    def update_context_batch(
        self,
        *,
        metadata: Any,
        generation_stats: Any,
        token_rows: list[list[PredicateToken]],
        memberships: np.ndarray,
        inv_only_weights: np.ndarray,
        combined_weights: np.ndarray,
    ) -> None:
        source_indices = tuple(getattr(generation_stats, "source_row_indices", ()) or ())
        if not source_indices:
            if len(token_rows) % memberships.shape[0] != 0:
                raise ValueError("importance context diagnostics cannot align contexts")
            repeats = len(token_rows) // memberships.shape[0]
            source_indices = tuple(np.repeat(np.arange(memberships.shape[0]), repeats))
        if len(source_indices) != len(token_rows):
            raise ValueError("source_row_indices must match generated predicate contexts")
        fanout_indices = tuple(metadata.fanout_indices())
        for context_index, source_row_index in enumerate(source_indices):
            active = np.flatnonzero(memberships[int(source_row_index)])
            if active.size == 0:
                continue
            fanout_values = {
                metadata.columns[fanout_index].name: (
                    float(inv_only_weights[context_index, fanout_index]),
                    float(combined_weights[context_index, fanout_index]),
                )
                for fanout_index in fanout_indices
            }
            token_row = token_rows[context_index]
            for stratum_index in active:
                stratum = self.selected_strata[int(stratum_index)]
                self.context_stats_by_stratum.setdefault(
                    stratum.stratum_id,
                    StratumPredicateContextStats(),
                ).update(
                    token=token_row[stratum.column_index],
                    stratum=stratum,
                    fanout_values=fanout_values,
                )

    def to_json_dict(self) -> dict[str, Any]:
        total = self.uniform_component_samples + self.rare_component_samples
        return {
            "enabled": self.enabled,
            "mixture_probability": self.mixture_probability,
            "number_selected_strata": len(self.selected_strata),
            "selected_strata": [stratum.to_json_dict() for stratum in self.selected_strata],
            "uniform_component_sample_count": self.uniform_component_samples,
            "rare_component_sample_count": self.rare_component_samples,
            "uniform_component_fraction": self.uniform_component_samples / total if total else 0.0,
            "rare_component_fraction": self.rare_component_samples / total if total else 0.0,
            "selected_stratum_sample_count": dict(self.selected_stratum_sample_count),
            "row_membership_count_per_stratum": dict(self.row_membership_count),
            "samples_belonging_to_multiple_selected_strata": self.multiple_membership_samples,
            "unique_stratum_membership_patterns": dict(self.membership_patterns),
            "rho": self.rho_stats.to_json_dict(),
            "rho_by_stratum": {
                stratum_id: stats.to_json_dict()
                for stratum_id, stats in self.rho_stats_by_stratum.items()
            },
            "conditional_context_stats_by_stratum": {
                stratum_id: stats.to_json_dict()
                for stratum_id, stats in self.context_stats_by_stratum.items()
            },
            "ordinary_sampler_seconds": self.ordinary_sampler_seconds,
            "conditional_sampler_seconds": self.conditional_sampler_seconds,
            "proposal_selection_seconds": self.proposal_selection_seconds,
            "importance_weight_computation_seconds": self.importance_weight_seconds,
        }


class ImportanceSamplingSampleSource:
    """Opt-in mixture proposal for uniform-FOJ training.

    Target p(x) remains the uniform full outer join.  The proposal is

        q(x) = (1-lambda)p(x) + lambda r(x)

    where r first chooses a selected root-DATA stratum s with probability
    alpha_s and then samples exactly from p(x | x in A_s).  Training receives
    rho(x)=p(x)/q(x).  This corrects tuple-sampling bias only; INV_FANOUT
    WCE weights still implement query-measure reweighting in the trainer.
    """

    def __init__(
        self,
        base_source: object,
        config: Mapping[str, Any],
    ) -> None:
        self.base_source = base_source
        self.config = dict(config)
        self.importance_config = dict(config.get("importance_sampling", {}))
        self.enabled = bool(self.importance_config.get("enabled", False))
        if not self.enabled:
            raise ValueError("ImportanceSamplingSampleSource requires enabled=true")
        self.mixture_probability = float(self.importance_config.get("mixture_probability", 0.2))
        if not (0.0 <= self.mixture_probability < 1.0):
            raise ValueError("importance_sampling.mixture_probability must be in [0, 1)")
        training = config.get("training", {})
        predicate = config.get("predicate_generation", {})
        discovery = self.importance_config.get("discovery", {})
        n_total = (
            int(training.get("steps_per_epoch", 1))
            * int(training.get("batch_size", 1))
            * int(predicate.get("per_row_contexts", 1))
            * int(training.get("epochs", 1))
        )
        diagnostics_enabled = bool(self.importance_config.get("diagnostics", {}).get("enabled", False))
        started = time.perf_counter()
        if diagnostics_enabled:
            print(
                "[importance_sampling] building exact root stratum provider",
                {
                    "n_total": n_total,
                    "discovery": dict(discovery),
                },
                flush=True,
            )
        self.provider = _provider_for_source(
            base_source,
            {**discovery, "diagnostics": diagnostics_enabled},
        )
        provider_seconds = time.perf_counter() - started
        if diagnostics_enabled:
            print(
                "[importance_sampling] provider ready",
                {
                    "seconds": provider_seconds,
                    "candidate_columns": [
                        self.provider.metadata.columns[index].name
                        for index in self.provider.column_masses
                    ],
                },
                flush=True,
            )
        discover_started = time.perf_counter()
        self.selected_strata = self.provider.discover(
            n_total=n_total,
            predicate_probabilities=predicate_probability_map(predicate),
            minimum_expected_context_support=float(
                discovery.get("minimum_expected_context_support", 100.0)
            ),
            max_selected_strata=int(discovery.get("max_selected_strata", 64)),
        )
        self.discovery_seconds = time.perf_counter() - started
        if diagnostics_enabled:
            print(
                "[importance_sampling] discovery ready",
                {
                    "provider_seconds": provider_seconds,
                    "discover_seconds": time.perf_counter() - discover_started,
                    "total_seconds": self.discovery_seconds,
                    "selected_strata": len(self.selected_strata),
                },
                flush=True,
            )
        if not self.selected_strata:
            raise ValueError(
                "importance sampling discovery selected no strata with both positive "
                "mass and positive support deficit; disable importance_sampling or "
                "lower minimum_expected_context_support"
            )
        prepare_root_strata = getattr(base_source, "prepare_root_strata", None)
        if prepare_root_strata is not None:
            prewarm_started = time.perf_counter()
            prepare_root_strata(self.selected_strata)
            if diagnostics_enabled:
                print(
                    "[importance_sampling] conditional sampler cache ready",
                    {
                        "seconds": time.perf_counter() - prewarm_started,
                        "selected_strata": len(self.selected_strata),
                    },
                    flush=True,
                )
        alpha_sum = sum(stratum.alpha for stratum in self.selected_strata)
        if abs(alpha_sum - 1.0) > 1.0e-8:
            raise ValueError(f"importance stratum alpha values must sum to 1, got {alpha_sum}")
        if any(stratum.probability <= 0.0 for stratum in self.selected_strata):
            raise ValueError("all selected importance strata must have P_s > 0")
        self.stats = ImportanceSamplingRunningStats(
            enabled=True,
            mixture_probability=self.mixture_probability,
            selected_strata=self.selected_strata,
        )
        self._alpha = np.array([stratum.alpha for stratum in self.selected_strata], dtype=float)
        self._batch_calls = 0

    @property
    def join_cardinality(self) -> int:
        return int(self.base_source.join_cardinality)  # type: ignore[attr-defined]

    @property
    def metadata(self) -> object:
        return self.base_source.metadata  # type: ignore[attr-defined]

    @property
    def sampler_run_calls(self) -> int | None:
        return getattr(self.base_source, "sampler_run_calls", None)

    @property
    def distinct_original_rows_seen_estimate(self) -> object:
        return getattr(self.base_source, "distinct_original_rows_seen_estimate", None)

    def discard_buffer(self) -> None:
        discard = getattr(self.base_source, "discard_buffer", None)
        if discard is not None:
            discard()

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        rng = np.random.default_rng(int(seed))
        proposal_start = time.perf_counter()
        rare_mask = rng.random(batch_size) < self.mixture_probability
        component = np.where(rare_mask, "rare", "uniform")
        rare_count = int(rare_mask.sum())
        uniform_count = int(batch_size - rare_count)
        selected_ids: list[str | None] = [None] * batch_size
        proposal_seconds = time.perf_counter() - proposal_start

        pieces: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        ordinary_seconds = 0.0
        conditional_seconds = 0.0
        fresh_rows = 0
        fixture_rows = 0
        if uniform_count:
            start = time.perf_counter()
            uniform_batch = self.base_source.batches(uniform_count, seed=seed)
            ordinary_seconds += time.perf_counter() - start
            pieces.append(uniform_batch.encoded_values)
            positions.append(np.flatnonzero(~rare_mask))
            fresh_rows += int(uniform_batch.fresh_rows_drawn)
            fixture_rows += int(uniform_batch.fixture_rows_reused)
        if rare_count:
            rare_positions = np.flatnonzero(rare_mask)
            selected = rng.choice(
                np.arange(len(self.selected_strata)),
                size=rare_count,
                replace=True,
                p=self._alpha,
            )
            selected_strata = [self.selected_strata[int(index)] for index in selected]
            for row_position, stratum in zip(rare_positions, selected_strata):
                selected_ids[int(row_position)] = stratum.stratum_id
            start = time.perf_counter()
            rows = _sample_conditionals_from_source(
                self.base_source,
                self.provider,
                selected_strata,
                rng,
            )
            conditional_seconds += time.perf_counter() - start
            pieces.append(rows)
            positions.append(rare_positions)
            fresh_rows += int(rare_count)
        encoded = np.empty((batch_size, len(self.metadata.columns)), dtype=np.int64)  # type: ignore[attr-defined]
        for rows, row_positions in zip(pieces, positions):
            encoded[row_positions] = rows
        weight_start = time.perf_counter()
        memberships = membership_matrix(self.metadata, encoded, self.selected_strata)  # type: ignore[arg-type]
        rho = rho_for_memberships(
            memberships,
            self.selected_strata,
            self.mixture_probability,
        )
        weight_seconds = time.perf_counter() - weight_start
        self.stats.update_batch(
            component=component,
            selected=selected_ids,
            memberships=memberships,
            rho=rho,
            timing={
                "ordinary_sampler_seconds": ordinary_seconds,
                "conditional_sampler_seconds": conditional_seconds,
                "proposal_selection_seconds": proposal_seconds,
                "importance_weight_seconds": weight_seconds,
            },
        )
        self._batch_calls += 1
        return FullJoinBatch(
            encoded_values=encoded,
            column_metadata=self.metadata.columns,  # type: ignore[attr-defined]
            fresh_rows_drawn=fresh_rows,
            fixture_rows_reused=fixture_rows,
            importance_weights=rho,
            importance_metadata={
                "component": component,
                "selected_stratum": selected_ids,
                "memberships": memberships,
                "rho": rho,
            },
        )

    def update_importance_context_statistics(
        self,
        *,
        generation_stats: Any,
        token_rows: list[list[PredicateToken]],
        inv_only_weights: np.ndarray,
        combined_weights: np.ndarray,
        batch_metadata: Mapping[str, Any] | None,
    ) -> None:
        if not batch_metadata:
            return
        memberships = np.asarray(batch_metadata.get("memberships"), dtype=bool)
        if memberships.ndim != 2:
            raise ValueError("importance context diagnostics require membership matrix")
        self.stats.update_context_batch(
            metadata=self.metadata,
            generation_stats=generation_stats,
            token_rows=token_rows,
            memberships=memberships,
            inv_only_weights=inv_only_weights,
            combined_weights=combined_weights,
        )

    def importance_sampling_summary(self) -> dict[str, Any]:
        payload = self.stats.to_json_dict()
        payload["discovery_seconds"] = self.discovery_seconds
        payload["root_mass_diagnostics"] = dict(getattr(self.provider, "mass_diagnostics", {}))
        payload["sampler_counters"] = _sampler_counters(self.base_source)
        payload["configuration"] = {
            "enabled": True,
            "mixture_probability": self.mixture_probability,
            "discovery": dict(self.importance_config.get("discovery", {})),
            "allocation": dict(self.importance_config.get("allocation", {})),
            "diagnostics": dict(self.importance_config.get("diagnostics", {})),
            "seed": self.config.get("training", {}).get("seed", 0),
        }
        return payload


def _provider_for_source(
    source: object,
    discovery_config: Mapping[str, Any] | None = None,
) -> ExactRootStratumProvider:
    discovery_config = discovery_config or {}
    if isinstance(source, SyntheticFullJoinSampleSource):
        return ExactRootStratumProvider.from_encoded_rows(
            source.metadata,
            source.dataset.encoded_rows,
            source="synthetic_materialized_full_join",
            root_column_semantics=discovery_config.get("root_column_semantics"),
        )
    dataset = getattr(source, "dataset", None)
    if dataset is not None and hasattr(dataset, "encoded_rows"):
        return ExactRootStratumProvider.from_encoded_rows(
            source.metadata,  # type: ignore[attr-defined]
            dataset.encoded_rows,
            source="materialized_full_join",
            root_column_semantics=discovery_config.get("root_column_semantics"),
        )
    base = getattr(source, "base_source", None)
    if base is not None:
        return _provider_for_source(base, discovery_config)
    sampler = getattr(source, "_sampler", None)
    if sampler is not None:
        return ExactRootStratumProvider.from_neurocard_root_jct(  # type: ignore[attr-defined]
            source.metadata,
            sampler,
            include_categorical=bool(
                discovery_config.get("include_categorical_root_data", True)
            ),
            max_domain_size=discovery_config.get("max_candidate_domain_size"),
            column_names=discovery_config.get("candidate_column_names"),
            root_column_semantics=discovery_config.get("root_column_semantics"),
            diagnostics=bool(discovery_config.get("diagnostics", False))
            or bool(discovery_config.get("log_diagnostics", False)),
        )
    sample_path = getattr(source, "prepared_directory", None)
    if sample_path is not None:
        path = sample_path / "sample_rows.npy"
        if path.exists():
            rows = np.load(path)
            return ExactRootStratumProvider.from_encoded_rows(
                source.metadata,  # type: ignore[attr-defined]
                rows,
                source="prepared_sample_rows_fixture",
                root_column_semantics=discovery_config.get("root_column_semantics"),
            )
    raise NotImplementedError(
        "importance sampling requires exact root-stratum support from a live "
        "NeuroCard sampler, synthetic materialized rows, or a prepared fixture"
    )


def _sample_conditional_from_source(
    source: object,
    provider: ExactRootStratumProvider,
    stratum: RootDataStratum,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    live_method = getattr(source, "sample_root_stratum_rows", None)
    if live_method is not None:
        return live_method(stratum, count, rng=rng)
    base = getattr(source, "base_source", None)
    if base is not None:
        live_method = getattr(base, "sample_root_stratum_rows", None)
        if live_method is not None:
            return live_method(stratum, count, rng=rng)
    return provider.sample_conditional(stratum, count, rng)


def _sample_conditionals_from_source(
    source: object,
    provider: ExactRootStratumProvider,
    strata: list[RootDataStratum],
    rng: np.random.Generator,
) -> np.ndarray:
    batch_method = getattr(source, "sample_root_strata_rows", None)
    if batch_method is not None:
        return batch_method(strata, rng=rng)
    pieces: list[np.ndarray] = []
    for stratum_index, stratum in enumerate(strata):
        rows = _sample_conditional_from_source(source, provider, stratum, 1, rng)
        if rows.shape[0] != 1:
            raise ValueError("conditional single-row sampler returned unexpected row count")
        pieces.append(rows)
    if not pieces:
        return np.empty((0, len(provider.metadata.columns)), dtype=np.int64)
    return np.concatenate(pieces, axis=0)


def _array_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p999": float(np.percentile(values, 99.9)),
    }


def _empty_weight_summary() -> dict[str, float]:
    return {
        "count": 0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "sum": 0.0,
        "sum_squared": 0.0,
        "ess": 0.0,
        "percentile_reservoir_size": 0,
        "p50": 0.0,
        "p90": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "p999": 0.0,
    }


def _token_relevant_to_stratum(token: PredicateToken, stratum: RootDataStratum) -> bool:
    if token.op == PredicateOp.EQUAL:
        return stratum.region_type == "equality" and token.value == stratum.value
    if token.op in {PredicateOp.GREATER_EQUAL, PredicateOp.GREATER_THAN}:
        return (
            stratum.region_type == "lower_tail"
            and token.value is not None
            and _safe_ge(token.value, stratum.lower)
        )
    if token.op in {PredicateOp.LESS_EQUAL, PredicateOp.LESS_THAN}:
        return (
            stratum.region_type == "upper_tail"
            and token.value is not None
            and _safe_le(token.value, stratum.upper)
        )
    if token.op == PredicateOp.RANGE:
        if stratum.region_type == "lower_tail":
            return token.value is not None and _safe_ge(token.value, stratum.lower)
        if stratum.region_type == "upper_tail":
            return token.upper is not None and _safe_le(token.upper, stratum.upper)
        if stratum.region_type == "equality":
            return (
                token.value is not None
                and token.upper is not None
                and _safe_le(token.value, stratum.value)
                and _safe_ge(token.upper, stratum.value)
            )
    return False


def _safe_ge(left: Any, right: Any) -> bool:
    try:
        return float(left) >= float(right)
    except (TypeError, ValueError):
        return left == right


def _safe_le(left: Any, right: Any) -> bool:
    try:
        return float(left) <= float(right)
    except (TypeError, ValueError):
        return left == right


def _bounded_literal_key(value: Any, *, max_len: int = 80) -> str:
    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _sampler_counters(source: object) -> dict[str, Any]:
    return {
        "sampler_run_calls": _nested_attr(source, "sampler_run_calls"),
        "conditional_sampler_batch_calls": _nested_attr(
            source,
            "conditional_sampler_batch_calls",
        ),
        "conditional_rows_drawn": _nested_attr(source, "conditional_rows_drawn"),
        "sample_batches_generated": _nested_attr(source, "sample_batches_generated"),
        "fresh_rows_drawn": _nested_attr(source, "fresh_rows_drawn"),
        "fixture_rows_reused": _nested_attr(source, "fixture_rows_reused"),
    }


def _nested_attr(source: object, name: str) -> Any:
    current = source
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        value = getattr(current, name, None)
        if value is not None:
            return value
        current = getattr(current, "base_source", None)
    return None
