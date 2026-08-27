from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

import numpy as np

from model.src.data.full_join_sampler import FullJoinBatch, SyntheticFullJoinSampleSource
from model.src.data.strata import (
    ExactRootStratumProvider,
    RootDataStratum,
    RootStratumMembershipLookup,
    build_membership_lookup,
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
            if self.reservoir_size <= 0:
                continue
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
            "percentiles_are_approximate": bool(sample.size),
            **percentiles,
        }


@dataclass
class StreamingMomentStats:
    """Constant-size streaming moments and ESS; retains no per-row history."""

    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray | list[float] | tuple[float, ...]) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return
        if np.any(array < 0.0) or not np.all(np.isfinite(array)):
            raise ValueError("moment statistics require finite nonnegative values")
        self.count += int(array.size)
        self.total += float(np.sum(array))
        self.total_squared += float(np.dot(array, array))
        self.minimum = min(self.minimum, float(np.min(array)))
        self.maximum = max(self.maximum, float(np.max(array)))

    def to_json_dict(self) -> dict[str, float]:
        if self.count == 0:
            return _empty_moment_summary()
        mean = self.total / float(self.count)
        variance = max(0.0, self.total_squared / float(self.count) - mean * mean)
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
            "retains_sample_history": False,
        }


@dataclass
class StreamingLogWeightStats:
    """Constant-size log-domain accumulator for global true rho*INV ESS."""

    count: int = 0
    log_sum_w: float = float("-inf")
    log_sum_w2: float = float("-inf")
    minimum_log: float = float("inf")
    maximum_log: float = float("-inf")

    def update_log_weights(self, log_values: np.ndarray) -> None:
        values = np.asarray(log_values, dtype=float).reshape(-1)
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("log weight statistics require finite log weights")
        self.count += int(values.size)
        self.log_sum_w = float(np.logaddexp(self.log_sum_w, _logsumexp(values)))
        self.log_sum_w2 = float(np.logaddexp(self.log_sum_w2, _logsumexp(2.0 * values)))
        self.minimum_log = min(self.minimum_log, float(np.min(values)))
        self.maximum_log = max(self.maximum_log, float(np.max(values)))

    def update_from_weights(self, values: np.ndarray | list[float] | tuple[float, ...]) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return
        if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
            raise ValueError("log weight statistics require finite positive weights")
        self.update_log_weights(np.log(array))

    def to_json_dict(self) -> dict[str, Any]:
        if self.count == 0:
            payload = _empty_moment_summary()
            payload.update({"log_sum": None, "log_sum_squared": None})
            return payload
        log_ess = 2.0 * self.log_sum_w - self.log_sum_w2
        log_mean = self.log_sum_w - np.log(float(self.count))
        log_second_moment = self.log_sum_w2 - np.log(float(self.count))
        mean = _safe_exp(log_mean)
        second_moment = _safe_exp(log_second_moment)
        std = None
        if mean is not None and second_moment is not None:
            std = float(np.sqrt(max(0.0, second_moment - mean * mean)))
        return {
            "count": int(self.count),
            "min": _safe_exp(self.minimum_log),
            "max": _safe_exp(self.maximum_log),
            "mean": mean,
            "std": std,
            "sum": _safe_exp(self.log_sum_w),
            "sum_squared": _safe_exp(self.log_sum_w2),
            "log_sum": float(self.log_sum_w),
            "log_sum_squared": float(self.log_sum_w2),
            "ess": _safe_exp(log_ess),
            "retains_sample_history": False,
            "log_domain": True,
        }


@dataclass
class FanoutConditionalContextStats:
    inv_only: StreamingMomentStats = field(default_factory=StreamingMomentStats)
    importance_times_inv: StreamingLogWeightStats = field(default_factory=StreamingLogWeightStats)
    relevant_inv_only: StreamingMomentStats = field(default_factory=StreamingMomentStats)
    relevant_importance_times_inv: StreamingLogWeightStats = field(default_factory=StreamingLogWeightStats)

    def update_arrays(
        self,
        *,
        inv_values: np.ndarray,
        log_combined_values: np.ndarray,
        relevant_mask: np.ndarray,
    ) -> None:
        self.inv_only.update(inv_values)
        self.importance_times_inv.update_log_weights(log_combined_values)
        if np.any(relevant_mask):
            self.relevant_inv_only.update(inv_values[relevant_mask])
            self.relevant_importance_times_inv.update_log_weights(
                log_combined_values[relevant_mask]
            )

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
    exact_support_event_operator_count: Counter[str] = field(default_factory=Counter)
    equality_literal_counts: Counter[str] = field(default_factory=Counter)
    lower_threshold_counts: Counter[str] = field(default_factory=Counter)
    upper_threshold_counts: Counter[str] = field(default_factory=Counter)
    range_lower_counts: Counter[str] = field(default_factory=Counter)
    range_upper_counts: Counter[str] = field(default_factory=Counter)
    relevant_fanout_token_signature_count: Counter[str] = field(default_factory=Counter)
    fanouts: dict[str, FanoutConditionalContextStats] = field(
        default_factory=lambda: defaultdict(FanoutConditionalContextStats)
    )

    def update_tokens(
        self,
        *,
        tokens: list[PredicateToken],
        stratum: RootDataStratum,
        fanout_signatures: list[str],
    ) -> np.ndarray:
        relevant_mask = np.zeros(len(tokens), dtype=bool)
        self.context_count += len(tokens)
        for context_index, token in enumerate(tokens):
            self.root_predicate_operator_count[token.op.value] += 1
            self._update_boundary_counters(token)
            if _token_exact_support_event_for_stratum(token, stratum):
                self.exact_support_event_operator_count[token.op.value] += 1
            relevant = _token_relevant_to_stratum(token, stratum)
            relevant_mask[context_index] = relevant
            if relevant:
                self.relevant_context_count += 1
                self.stratum_relevant_operator_count[token.op.value] += 1
                self.relevant_fanout_token_signature_count[
                    fanout_signatures[context_index]
                ] += 1
        return relevant_mask

    def update_fanouts(
        self,
        *,
        fanout_name: str,
        inv_values: np.ndarray,
        log_combined_values: np.ndarray,
        relevant_mask: np.ndarray,
    ) -> None:
        self.fanouts[fanout_name].update_arrays(
            inv_values=inv_values,
            log_combined_values=log_combined_values,
            relevant_mask=relevant_mask,
        )

    def _update_boundary_counters(self, token: PredicateToken) -> None:
        if token.op == PredicateOp.EQUAL and token.value is not None:
            self.equality_literal_counts[_bounded_literal_key(token.value)] += 1
        elif token.op in {PredicateOp.GREATER_EQUAL, PredicateOp.GREATER_THAN} and token.value is not None:
            self.lower_threshold_counts[_bounded_literal_key(token.value)] += 1
        elif token.op in {PredicateOp.LESS_EQUAL, PredicateOp.LESS_THAN} and token.value is not None:
            self.upper_threshold_counts[_bounded_literal_key(token.value)] += 1
        elif token.op == PredicateOp.RANGE:
            if token.value is not None:
                self.range_lower_counts[_bounded_literal_key(token.value)] += 1
            if token.upper is not None:
                self.range_upper_counts[_bounded_literal_key(token.upper)] += 1

    def to_json_dict(self) -> dict[str, Any]:
        boundary_counters = {
            "equality_literal_counts": self.equality_literal_counts,
            "lower_threshold_counts": self.lower_threshold_counts,
            "upper_threshold_counts": self.upper_threshold_counts,
            "range_lower_counts": self.range_lower_counts,
            "range_upper_counts": self.range_upper_counts,
        }
        return {
            "context_count": int(self.context_count),
            "stratum_relevant_context_count": int(self.relevant_context_count),
            "root_predicate_operator_count": dict(self.root_predicate_operator_count),
            "stratum_relevant_operator_count": dict(self.stratum_relevant_operator_count),
            "exact_support_event_operator_count": dict(
                self.exact_support_event_operator_count
            ),
            **{
                name: dict(counter.most_common(512))
                for name, counter in boundary_counters.items()
            },
            "boundary_counter_summaries": {
                name: _counter_boundary_summary(counter)
                for name, counter in boundary_counters.items()
            },
            "relevant_fanout_token_signature_count": dict(
                self.relevant_fanout_token_signature_count.most_common(256)
            ),
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
    global_rho_reservoir_size: int = 100_000
    per_stratum_rho_reservoir_size: int = 1_000
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
    membership_computation_seconds: float = 0.0
    context_statistics_seconds: float = 0.0
    smallest_rho_limit: int = 32
    smallest_rho_patterns: list[dict[str, Any]] = field(default_factory=list)

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
        self._update_smallest_rho_patterns(
            component=component,
            selected=selected,
            memberships=memberships,
            rho=rho,
        )
        for stratum_index, stratum in enumerate(self.selected_strata):
            values = rho[memberships[:, stratum_index]]
            if values.size:
                self.rho_stats_by_stratum.setdefault(
                    stratum.stratum_id,
                    StreamingWeightStats(
                        reservoir_size=self.per_stratum_rho_reservoir_size,
                        seed=stratum_index + 1,
                    ),
                ).update(values)
        self.ordinary_sampler_seconds += float(timing.get("ordinary_sampler_seconds", 0.0))
        self.conditional_sampler_seconds += float(timing.get("conditional_sampler_seconds", 0.0))
        self.proposal_selection_seconds += float(timing.get("proposal_selection_seconds", 0.0))
        self.importance_weight_seconds += float(timing.get("importance_weight_seconds", 0.0))
        self.membership_computation_seconds += float(timing.get("membership_computation_seconds", 0.0))

    def update_context_batch(
        self,
        *,
        metadata: Any,
        generation_stats: Any,
        token_rows: list[list[PredicateToken]],
        memberships: np.ndarray,
        inv_only_weights: np.ndarray,
        combined_weights: np.ndarray,
        rho: np.ndarray,
    ) -> None:
        started = time.perf_counter()
        source_indices = tuple(getattr(generation_stats, "source_row_indices", ()) or ())
        if not source_indices:
            if len(token_rows) % memberships.shape[0] != 0:
                raise ValueError("importance context diagnostics cannot align contexts")
            repeats = len(token_rows) // memberships.shape[0]
            source_indices = tuple(np.repeat(np.arange(memberships.shape[0]), repeats))
        if len(source_indices) != len(token_rows):
            raise ValueError("source_row_indices must match generated predicate contexts")
        source_index_array = np.asarray(source_indices, dtype=int)
        context_memberships = memberships[source_index_array]
        log_rho = np.log(np.asarray(rho, dtype=float))
        if log_rho.shape != (len(token_rows),):
            raise ValueError("rho must have one value per generated context")
        fanout_signatures = _fanout_token_signatures(token_rows, metadata)
        fanout_indices = tuple(metadata.fanout_indices())
        for stratum_index, stratum in enumerate(self.selected_strata):
            context_mask = context_memberships[:, stratum_index]
            if not np.any(context_mask):
                continue
            selected_context_indices = np.flatnonzero(context_mask)
            stratum_tokens = [
                token_rows[int(context_index)][stratum.column_index]
                for context_index in selected_context_indices
            ]
            stratum_signatures = [
                fanout_signatures[int(context_index)]
                for context_index in selected_context_indices
            ]
            stratum_stats = self.context_stats_by_stratum.setdefault(
                stratum.stratum_id,
                StratumPredicateContextStats(),
            )
            relevant_mask = stratum_stats.update_tokens(
                tokens=stratum_tokens,
                stratum=stratum,
                fanout_signatures=stratum_signatures,
            )
            for fanout_index in fanout_indices:
                inv_values = inv_only_weights[selected_context_indices, fanout_index]
                log_combined = log_rho[selected_context_indices] + np.log(inv_values)
                stratum_stats.update_fanouts(
                    fanout_name=metadata.columns[fanout_index].name,
                    inv_values=inv_values,
                    log_combined_values=log_combined,
                    relevant_mask=relevant_mask,
                )
        self.context_statistics_seconds += time.perf_counter() - started

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
            "smallest_rho_patterns": list(self.smallest_rho_patterns),
            "ordinary_sampler_seconds": self.ordinary_sampler_seconds,
            "conditional_sampler_seconds": self.conditional_sampler_seconds,
            "proposal_selection_seconds": self.proposal_selection_seconds,
            "importance_weight_computation_seconds": self.importance_weight_seconds,
            "membership_computation_seconds": self.membership_computation_seconds,
            "context_statistics_seconds": self.context_statistics_seconds,
            "statistics_memory_configuration": {
                "global_rho_reservoir_size": int(self.global_rho_reservoir_size),
                "per_stratum_rho_reservoir_size": int(self.per_stratum_rho_reservoir_size),
                "fanout_reservoirs_enabled": False,
            },
        }

    def _update_smallest_rho_patterns(
        self,
        *,
        component: np.ndarray,
        selected: list[str | None],
        memberships: np.ndarray,
        rho: np.ndarray,
    ) -> None:
        if self.smallest_rho_limit <= 0 or rho.size == 0:
            return
        alpha_over_probability = np.array(
            [stratum.alpha / stratum.probability for stratum in self.selected_strata],
            dtype=float,
        )
        candidate_count = min(self.smallest_rho_limit, int(rho.size))
        for row_index in np.argsort(rho)[:candidate_count]:
            matching_indices = np.flatnonzero(memberships[int(row_index)])
            matching_ids = [
                self.selected_strata[int(index)].stratum_id for index in matching_indices
            ]
            self.smallest_rho_patterns.append(
                {
                    "rho": float(rho[int(row_index)]),
                    "component": str(component[int(row_index)]),
                    "selected_proposal_stratum": selected[int(row_index)],
                    "matching_stratum_ids": matching_ids,
                    "alpha_over_probability_sum": float(
                        np.sum(alpha_over_probability[matching_indices])
                    ),
                }
            )
        self.smallest_rho_patterns.sort(key=lambda item: item["rho"])
        del self.smallest_rho_patterns[self.smallest_rho_limit :]


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
        self.maximum_configured_steps = (
            int(training.get("steps_per_epoch", 1)) * int(training.get("epochs", 1))
        )
        self.support_planning_steps = support_planning_steps_from_config(config)
        self.planned_sample_count = (
            self.support_planning_steps
            * int(training.get("batch_size", 1))
            * int(predicate.get("per_row_contexts", 1))
        )
        diagnostics_config = dict(self.importance_config.get("diagnostics", {}))
        diagnostics_enabled = bool(diagnostics_config.get("enabled", False))
        self.global_rho_reservoir_size = int(
            diagnostics_config.get("global_rho_reservoir_size", 100_000)
        )
        self.per_stratum_rho_reservoir_size = int(
            diagnostics_config.get("per_stratum_rho_reservoir_size", 1_000)
        )
        started = time.perf_counter()
        if diagnostics_enabled:
            print(
                "[importance_sampling] building exact root stratum provider",
                {
                    "n_total": self.planned_sample_count,
                    "support_planning_steps": self.support_planning_steps,
                    "maximum_configured_steps": self.maximum_configured_steps,
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
            n_total=self.planned_sample_count,
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
            global_rho_reservoir_size=self.global_rho_reservoir_size,
            per_stratum_rho_reservoir_size=self.per_stratum_rho_reservoir_size,
            rho_stats=StreamingWeightStats(
                reservoir_size=self.global_rho_reservoir_size,
                seed=int(training.get("seed", 0)),
            ),
        )
        self._membership_lookup = build_membership_lookup(self.metadata, self.selected_strata)  # type: ignore[arg-type]
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
        membership_start = time.perf_counter()
        memberships = membership_matrix(
            self.metadata,
            encoded,
            self.selected_strata,
            lookup=self._membership_lookup,
        )  # type: ignore[arg-type]
        membership_seconds = time.perf_counter() - membership_start
        if rare_count:
            self._assert_rare_rows_match_selected_strata(
                rare_positions=np.flatnonzero(rare_mask),
                selected_ids=selected_ids,
                memberships=memberships,
            )
        weight_start = time.perf_counter()
        rho = rho_for_memberships(
            memberships,
            self.selected_strata,
            self.mixture_probability,
        )
        _assert_rho_defensive_bound(rho, self.mixture_probability)
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
                "membership_computation_seconds": membership_seconds,
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

    def _assert_rare_rows_match_selected_strata(
        self,
        *,
        rare_positions: np.ndarray,
        selected_ids: list[str | None],
        memberships: np.ndarray,
    ) -> None:
        stratum_index_by_id = {
            stratum.stratum_id: index for index, stratum in enumerate(self.selected_strata)
        }
        failures: list[dict[str, Any]] = []
        for row_position in rare_positions:
            selected_id = selected_ids[int(row_position)]
            if selected_id is None:
                failures.append({"row_position": int(row_position), "selected_stratum": None})
                continue
            stratum_index = stratum_index_by_id[selected_id]
            if not bool(memberships[int(row_position), stratum_index]):
                failures.append(
                    {
                        "row_position": int(row_position),
                        "selected_stratum": selected_id,
                    }
                )
        if failures:
            raise ValueError(
                "rare importance samples failed selected-stratum membership check: "
                f"{failures[:10]}"
            )

    def update_importance_context_statistics(
        self,
        *,
        generation_stats: Any,
        token_rows: list[list[PredicateToken]],
        inv_only_weights: np.ndarray,
        rho: np.ndarray,
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
            combined_weights=np.empty((0, 0), dtype=float),
            rho=rho,
        )

    def importance_sampling_summary(
        self,
        *,
        actual_optimizer_steps: int | None = None,
        early_stopped: bool | None = None,
        early_stopping_stop_step: int | None = None,
    ) -> dict[str, Any]:
        payload = self.stats.to_json_dict()
        realized_fraction = (
            None
            if actual_optimizer_steps is None
            else float(actual_optimizer_steps) / float(self.support_planning_steps)
        )
        payload["maximum_configured_steps"] = int(self.maximum_configured_steps)
        payload["support_planning_steps"] = int(self.support_planning_steps)
        payload["planned_full_join_sample_count"] = int(self.planned_sample_count)
        payload["actual_optimizer_steps"] = (
            None if actual_optimizer_steps is None else int(actual_optimizer_steps)
        )
        payload["realized_support_fraction"] = realized_fraction
        payload["early_stopped"] = None if early_stopped is None else bool(early_stopped)
        payload["early_stopping_stop_step"] = (
            None if early_stopping_stop_step is None else int(early_stopping_stop_step)
        )
        if early_stopped:
            payload["early_stopping_support_diagnostic"] = (
                "proposal discovery used the fixed support_planning_steps horizon; "
                "realized support values below are post-run diagnostics only"
            )
        payload["selected_strata"] = [
            _stratum_support_summary(stratum, realized_fraction)
            for stratum in self.selected_strata
        ]
        payload["proposal_composition"] = _proposal_composition(self.selected_strata)
        payload["context_amplification_by_stratum"] = _context_amplification_by_stratum(
            self.selected_strata,
            self.stats.context_stats_by_stratum,
            realized_fraction,
        )
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
            "maximum_configured_steps": int(self.maximum_configured_steps),
            "support_planning_steps": int(self.support_planning_steps),
            "planned_full_join_sample_count": int(self.planned_sample_count),
        }
        return payload


class RareSupportSampleSource:
    """Uniform sampler plus an explicit rare-support auxiliary sampler.

    ``batches`` remains the ordinary uniform full-outer-join source.  Rare rows
    are drawn only through ``rare_batches`` and deliberately carry no rho
    correction weights; the trainer uses them for a biased auxiliary objective.
    """

    def __init__(
        self,
        base_source: object,
        config: Mapping[str, Any],
    ) -> None:
        self.base_source = base_source
        self.config = dict(config)
        self.rare_config = dict(config.get("rare_support", {}))
        self.enabled = bool(self.rare_config.get("enabled", False))
        if not self.enabled:
            raise ValueError("RareSupportSampleSource requires rare_support.enabled=true")
        training = config.get("training", {})
        predicate = config.get("predicate_generation", {})
        discovery = self.rare_config.get("discovery", {})
        diagnostics_config = dict(self.rare_config.get("diagnostics", {}))
        diagnostics_enabled = bool(diagnostics_config.get("enabled", False))
        self.maximum_configured_steps = (
            int(training.get("steps_per_epoch", 1)) * int(training.get("epochs", 1))
        )
        self.support_planning_steps = support_planning_steps_from_config(config)
        self.planned_sample_count = (
            self.support_planning_steps
            * int(training.get("batch_size", 1))
            * int(predicate.get("per_row_contexts", 1))
        )
        started = time.perf_counter()
        if diagnostics_enabled:
            print(
                "[rare_support] building exact root stratum provider",
                {
                    "n_total": self.planned_sample_count,
                    "support_planning_steps": self.support_planning_steps,
                    "maximum_configured_steps": self.maximum_configured_steps,
                    "discovery": dict(discovery),
                },
                flush=True,
            )
        self.provider = _provider_for_source(
            base_source,
            {**discovery, "diagnostics": diagnostics_enabled},
        )
        provider_seconds = time.perf_counter() - started
        self.selected_strata = self.provider.discover(
            n_total=self.planned_sample_count,
            predicate_probabilities=predicate_probability_map(predicate),
            minimum_expected_context_support=float(
                discovery.get("minimum_expected_context_support", 100.0)
            ),
            max_selected_strata=int(discovery.get("max_selected_strata", 64)),
        )
        self.discovery_seconds = time.perf_counter() - started
        if diagnostics_enabled:
            print(
                "[rare_support] discovery ready",
                {
                    "provider_seconds": provider_seconds,
                    "total_seconds": self.discovery_seconds,
                    "selected_strata": len(self.selected_strata),
                },
                flush=True,
            )
        if not self.selected_strata:
            raise ValueError(
                "rare support discovery selected no strata with both positive "
                "mass and positive support deficit"
            )
        prepare_root_strata = getattr(base_source, "prepare_root_strata", None)
        if prepare_root_strata is not None:
            prewarm_started = time.perf_counter()
            prepare_root_strata(self.selected_strata)
            if diagnostics_enabled:
                print(
                    "[rare_support] conditional sampler cache ready",
                    {
                        "seconds": time.perf_counter() - prewarm_started,
                        "selected_strata": len(self.selected_strata),
                    },
                    flush=True,
                )
        alpha_sum = sum(stratum.alpha for stratum in self.selected_strata)
        if abs(alpha_sum - 1.0) > 1.0e-8:
            raise ValueError(f"rare support alpha values must sum to 1, got {alpha_sum}")
        if any(stratum.probability <= 0.0 for stratum in self.selected_strata):
            raise ValueError("all selected rare support strata must have P_s > 0")
        self._membership_lookup = build_membership_lookup(self.metadata, self.selected_strata)  # type: ignore[arg-type]
        self._alpha = np.array([stratum.alpha for stratum in self.selected_strata], dtype=float)
        self.rare_rows_drawn = 0
        self.selected_stratum_sample_count: Counter[str] = Counter()

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
        return self.base_source.batches(batch_size, seed=seed)  # type: ignore[attr-defined]

    def rare_batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        if batch_size <= 0:
            raise ValueError("rare auxiliary batch_size must be positive")
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(
            np.arange(len(self.selected_strata)),
            size=int(batch_size),
            replace=True,
            p=self._alpha,
        )
        selected_strata = tuple(self.selected_strata[int(index)] for index in selected)
        rows = _sample_conditionals_from_source(
            self.base_source,
            self.provider,
            list(selected_strata),
            rng,
        )
        memberships = membership_matrix(
            self.metadata,
            rows,
            self.selected_strata,
            lookup=self._membership_lookup,
        )  # type: ignore[arg-type]
        selected_ids = [stratum.stratum_id for stratum in selected_strata]
        ImportanceSamplingSampleSource._assert_rare_rows_match_selected_strata(
            self,
            rare_positions=np.arange(len(selected_strata)),
            selected_ids=selected_ids,
            memberships=memberships,
        )
        self.rare_rows_drawn += int(batch_size)
        self.selected_stratum_sample_count.update(selected_ids)
        return FullJoinBatch(
            encoded_values=rows,
            column_metadata=self.metadata.columns,  # type: ignore[attr-defined]
            fresh_rows_drawn=int(batch_size),
            importance_weights=None,
            importance_metadata={
                "selected_stratum": selected_ids,
                "selected_strata": selected_strata,
                "selected_stratum_column_index": [
                    int(stratum.column_index) for stratum in selected_strata
                ],
                "memberships": memberships,
            },
        )

    def rare_support_summary(
        self,
        *,
        actual_optimizer_steps: int | None = None,
    ) -> dict[str, Any]:
        realized_fraction = (
            None
            if actual_optimizer_steps is None
            else float(actual_optimizer_steps) / float(self.support_planning_steps)
        )
        return {
            "enabled": True,
            "maximum_configured_steps": int(self.maximum_configured_steps),
            "support_planning_steps": int(self.support_planning_steps),
            "planned_full_join_sample_count": int(self.planned_sample_count),
            "actual_optimizer_steps": (
                None if actual_optimizer_steps is None else int(actual_optimizer_steps)
            ),
            "realized_support_fraction": realized_fraction,
            "number_selected_strata": len(self.selected_strata),
            "selected_strata": [
                _stratum_support_summary(stratum, realized_fraction)
                for stratum in self.selected_strata
            ],
            "proposal_composition": _proposal_composition(self.selected_strata),
            "selected_stratum_sample_count": dict(self.selected_stratum_sample_count),
            "rare_rows_drawn": int(self.rare_rows_drawn),
            "discovery_seconds": self.discovery_seconds,
            "root_mass_diagnostics": dict(getattr(self.provider, "mass_diagnostics", {})),
            "sampler_counters": _sampler_counters(self.base_source),
            "configuration": {
                "enabled": True,
                "discovery": dict(self.rare_config.get("discovery", {})),
                "allocation": dict(self.rare_config.get("allocation", {})),
                "diagnostics": dict(self.rare_config.get("diagnostics", {})),
                "seed": self.config.get("training", {}).get("seed", 0),
            },
        }


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


def support_planning_steps_from_config(config: Mapping[str, Any]) -> int:
    training = config.get("training", {})
    support_config = (
        config.get("rare_support", {})
        if bool(config.get("rare_support", {}).get("enabled", False))
        else config.get("importance_sampling", {})
    )
    discovery = support_config.get("discovery", {})
    if "support_planning_steps" in discovery:
        steps = int(discovery["support_planning_steps"])
    else:
        steps = int(training.get("steps_per_epoch", 1)) * int(training.get("epochs", 1))
    if steps <= 0:
        raise ValueError("importance_sampling.discovery.support_planning_steps must be positive")
    return steps


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


def _safe_exp(log_value: float) -> float | None:
    if not isfinite(float(log_value)):
        return None
    if log_value > 700.0:
        return None
    return float(np.exp(log_value))


def _finite_json_number(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _scaled_support(value: float | None, fraction: float | None) -> float | None:
    if value is None or fraction is None:
        return None
    return _finite_json_number(float(value) * float(fraction))


def _stratum_support_summary(
    stratum: RootDataStratum,
    realized_fraction: float | None,
) -> dict[str, Any]:
    payload = stratum.to_json_dict()
    support_fields = (
        "expected_target_rows",
        "expected_equality_count",
        "expected_lower_count",
        "expected_upper_count",
        "expected_range_support",
    )
    for field_name in support_fields:
        planned_value = payload.get(field_name)
        payload[f"planned_{field_name}"] = _finite_json_number(planned_value)
        payload[f"realized_{field_name}"] = _scaled_support(planned_value, realized_fraction)
    return payload


def _proposal_composition(strata: tuple[RootDataStratum, ...]) -> dict[str, Any]:
    by_region: dict[str, dict[str, float | int]] = {}
    by_bottleneck: dict[str, dict[str, float | int]] = {}
    for stratum in strata:
        for mapping, key in (
            (by_region, stratum.region_type),
            (by_bottleneck, stratum.support_bottleneck or "<unknown>"),
        ):
            entry = mapping.setdefault(key, {"selected_count": 0, "alpha_sum": 0.0})
            entry["selected_count"] = int(entry["selected_count"]) + 1
            entry["alpha_sum"] = float(entry["alpha_sum"]) + float(stratum.alpha)
    probabilities = [float(stratum.probability) for stratum in strata]
    return {
        "selected_count": len(strata),
        "alpha_sum": float(sum(stratum.alpha for stratum in strata)),
        "min_probability": min(probabilities) if probabilities else None,
        "max_probability": max(probabilities) if probabilities else None,
        "by_region_type": by_region,
        "by_support_bottleneck": by_bottleneck,
    }


def _context_amplification_by_stratum(
    strata: tuple[RootDataStratum, ...],
    context_stats: Mapping[str, StratumPredicateContextStats],
    realized_fraction: float | None,
) -> dict[str, Any]:
    amplification: dict[str, Any] = {}
    for stratum in strata:
        stats = context_stats.get(stratum.stratum_id)
        exact_observed = (
            dict(stats.exact_support_event_operator_count) if stats is not None else {}
        )
        by_operator: dict[str, Any] = {}
        for op_name, planned in (
            (PredicateOp.EQUAL.value, stratum.expected_equality_count),
            (PredicateOp.GREATER_EQUAL.value, stratum.expected_lower_count),
            (PredicateOp.LESS_EQUAL.value, stratum.expected_upper_count),
            (PredicateOp.RANGE.value, stratum.expected_range_support),
        ):
            if planned is None:
                continue
            expected_uniform = _scaled_support(planned, realized_fraction)
            observed_count = int(exact_observed.get(op_name, 0))
            by_operator[op_name] = {
                "planned_expected_uniform_count": _finite_json_number(planned),
                "expected_uniform_count_at_actual_steps": expected_uniform,
                "observed_exact_support_event_count": observed_count,
                "raw_context_amplification": (
                    None
                    if expected_uniform is None or expected_uniform <= 0.0
                    else float(observed_count) / float(expected_uniform)
                ),
            }
        amplification[stratum.stratum_id] = {
            "support_bottleneck": stratum.support_bottleneck,
            "region_type": stratum.region_type,
            "by_operator": by_operator,
        }
    return amplification


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
        "percentiles_are_approximate": False,
    }


def _empty_moment_summary() -> dict[str, float]:
    return {
        "count": 0,
        "min": 0.0,
        "max": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "sum": 0.0,
        "sum_squared": 0.0,
        "ess": 0.0,
        "retains_sample_history": False,
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
                and _literal_equal(token.value, stratum.value)
                and _literal_equal(token.upper, stratum.value)
            )
        if stratum.region_type == "range":
            return (
                token.value is not None
                and token.upper is not None
                and _safe_ge(token.value, stratum.lower)
                and _safe_le(token.upper, stratum.upper)
            )
    return False


def _token_exact_support_event_for_stratum(
    token: PredicateToken,
    stratum: RootDataStratum,
) -> bool:
    if token.op == PredicateOp.EQUAL:
        return (
            stratum.region_type == "equality"
            and token.value is not None
            and _literal_equal(token.value, stratum.value)
        )
    if token.op == PredicateOp.GREATER_EQUAL:
        return (
            stratum.region_type == "lower_tail"
            and token.value is not None
            and _literal_equal(token.value, stratum.lower)
        )
    if token.op == PredicateOp.LESS_EQUAL:
        return (
            stratum.region_type == "upper_tail"
            and token.value is not None
            and _literal_equal(token.value, stratum.upper)
        )
    if token.op == PredicateOp.RANGE:
        if token.value is None or token.upper is None:
            return False
        if stratum.region_type == "equality":
            return (
                _literal_equal(token.value, stratum.value)
                and _literal_equal(token.upper, stratum.value)
            )
        if stratum.region_type == "lower_tail":
            return _safe_ge(token.value, stratum.lower)
        if stratum.region_type == "upper_tail":
            return _safe_le(token.upper, stratum.upper)
    return False


def _literal_equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


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


def _counter_boundary_summary(counter: Counter[str], *, limit: int = 16) -> dict[str, Any]:
    if not counter:
        return {
            "observed_distinct": 0,
            "least_supported": [],
            "most_supported": [],
        }
    least = sorted(counter.items(), key=lambda item: (item[1], item[0]))[:limit]
    return {
        "observed_distinct": len(counter),
        "least_supported": [{"literal": key, "count": int(value)} for key, value in least],
        "most_supported": [
            {"literal": key, "count": int(value)}
            for key, value in counter.most_common(limit)
        ],
    }


def _fanout_token_signatures(token_rows: list[list[PredicateToken]], metadata: Any) -> list[str]:
    fanout_indices = tuple(metadata.fanout_indices())
    signatures: list[str] = []
    for token_row in token_rows:
        parts = []
        for fanout_index in fanout_indices:
            column_name = metadata.columns[fanout_index].name
            state = "INV" if token_row[fanout_index].op == PredicateOp.INV_FANOUT else "WILDCARD"
            parts.append(f"{column_name}:{state}")
        signatures.append("|".join(parts))
    return signatures


def _assert_rho_defensive_bound(
    rho: np.ndarray,
    mixture_probability: float,
    *,
    tolerance: float = 1.0e-9,
) -> None:
    values = np.asarray(rho, dtype=float)
    if values.size == 0:
        return
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("importance weights rho must be finite and positive")
    upper_bound = 1.0 / (1.0 - float(mixture_probability))
    max_rho = float(np.max(values))
    if max_rho > upper_bound + tolerance:
        raise ValueError(
            f"importance weights exceed defensive mixture bound: max_rho={max_rho}, "
            f"bound={upper_bound}, mixture_probability={mixture_probability}"
        )


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("-inf")
    maximum = float(np.max(values))
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


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
