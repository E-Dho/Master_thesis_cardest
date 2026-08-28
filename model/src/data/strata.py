from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping

import numpy as np

from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ModelMetadata
from model.src.predicates.operators import PredicateOp


SQL_NULL = "__SQL_NULL__"


@dataclass(frozen=True)
class RootDataStratum:
    """A root-table DATA region with exact mass under uniform FOJ tuples."""

    stratum_id: str
    column_index: int
    column_name: str
    region_type: str
    value: Any | None = None
    lower: Any | None = None
    upper: Any | None = None
    foj_count: float = 0.0
    probability: float = 0.0
    expected_target_rows: float = 0.0
    expected_equality_count: float | None = None
    expected_lower_count: float | None = None
    expected_upper_count: float | None = None
    expected_range_support: float | None = None
    support_score: float = 0.0
    support_deficit: float = 0.0
    support_bottleneck: str | None = None
    alpha: float = 0.0
    source: str = "unknown"
    semantic_type: str = "categorical"

    def contains_value(self, value: Any) -> bool:
        if _is_structural_value(value):
            return False
        if self.region_type == "equality":
            return value == self.value
        if self.region_type == "lower_tail":
            return _safe_ge(value, self.lower)
        if self.region_type == "upper_tail":
            return _safe_le(value, self.upper)
        if self.region_type == "range":
            return _safe_ge(value, self.lower) and _safe_le(value, self.upper)
        raise ValueError(f"unsupported stratum region_type {self.region_type!r}")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value"] = _jsonable(payload["value"])
        payload["lower"] = _jsonable(payload["lower"])
        payload["upper"] = _jsonable(payload["upper"])
        return payload


@dataclass(frozen=True)
class RootColumnMass:
    column_index: int
    values: tuple[Any, ...]
    counts: np.ndarray
    total_count: float
    source: str
    semantic_type: str = "categorical"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RootStratumMembershipLookup:
    """Precomputed encoded-domain masks for vectorized root-stratum membership."""

    by_column: Mapping[int, tuple[np.ndarray, np.ndarray]]


class ExactRootStratumProvider:
    """Exact root-DATA stratum support from enumerated or root-JCT-weighted rows."""

    def __init__(
        self,
        metadata: ModelMetadata,
        column_masses: Mapping[int, RootColumnMass],
        *,
        encoded_rows: np.ndarray | None = None,
        row_weights: np.ndarray | None = None,
        source: str = "exact",
        mass_diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata
        self.column_masses = dict(column_masses)
        self.encoded_rows = None if encoded_rows is None else np.asarray(encoded_rows, dtype=np.int64)
        self.row_weights = None if row_weights is None else np.asarray(row_weights, dtype=float)
        self.source = source
        self.mass_diagnostics = dict(mass_diagnostics or {})
        if self.encoded_rows is not None and self.row_weights is None:
            self.row_weights = np.ones(self.encoded_rows.shape[0], dtype=float)

    @classmethod
    def from_encoded_rows(
        cls,
        metadata: ModelMetadata,
        encoded_rows: np.ndarray,
        *,
        source: str = "materialized_full_join",
        root_column_semantics: Mapping[str, str] | None = None,
    ) -> "ExactRootStratumProvider":
        encoded_rows = np.asarray(encoded_rows, dtype=np.int64)
        root = metadata.join_root
        masses: dict[int, RootColumnMass] = {}
        for column_index, column in enumerate(metadata.columns):
            if column.kind != ColumnKind.DATA or column.table != root:
                continue
            counts = np.bincount(
                encoded_rows[:, column_index],
                minlength=column.domain_size,
            ).astype(float)
            masses[column_index] = RootColumnMass(
                column_index=column_index,
                values=column.domain,
                counts=counts,
                total_count=float(counts.sum()),
                source=source,
                semantic_type=_semantic_for_column(column.name, root_column_semantics),
            )
        return cls(metadata, masses, encoded_rows=encoded_rows, source=source)

    @classmethod
    def from_neurocard_root_jct(
        cls,
        metadata: ModelMetadata,
        sampler: Any,
        *,
        include_categorical: bool = True,
        max_domain_size: int | None = None,
        column_names: Iterable[str] | None = None,
        root_column_semantics: Mapping[str, str] | None = None,
        diagnostics: bool = False,
    ) -> "ExactRootStratumProvider":
        root = sampler.join_spec.join_root
        table_actor = next(actor for actor in sampler.dt_actors if actor.table == root)
        if len(table_actor.join_keys) != 1:
            raise NotImplementedError("importance strata currently support one root join key")
        root_key = table_actor.join_keys[0]
        if table_actor.df[root_key].duplicated().any():
            raise NotImplementedError(
                "root DATA stratum mass requires a unique root join key in the root table"
            )
        root_jct = sampler.jct_actors[root].jct
        root_weight_column = f"{root}.weight"
        total_root_mass = float(root_jct[root_weight_column].sum())
        sampler_join_cardinality = _optional_float(getattr(sampler, "join_card", None))
        metadata_join_cardinality = _optional_float(metadata.full_join_cardinality)
        _require_mass_close(
            total_root_mass,
            sampler_join_cardinality,
            "root JCT total weight",
            "sampler join cardinality",
        )
        _require_mass_close(
            total_root_mass,
            metadata_join_cardinality,
            "root JCT total weight",
            "metadata full join cardinality",
        )
        root_values = table_actor.df.set_index(root_key, drop=False)
        allowed_columns = None if column_names is None else set(column_names)
        max_size = None if max_domain_size is None else int(max_domain_size)
        masses: dict[int, RootColumnMass] = {}
        mapped_mass_by_column: dict[str, float] = {}
        mass_difference_by_column: dict[str, float] = {}
        for column_index, column in enumerate(metadata.columns):
            if column.kind != ColumnKind.DATA or column.table != root:
                continue
            semantic_type = _semantic_for_column(column.name, root_column_semantics)
            if allowed_columns is not None and column.name not in allowed_columns:
                if diagnostics:
                    print(
                        "[importance_sampling] skip root column",
                        {"column": column.name, "reason": "not_in_candidate_column_names"},
                        flush=True,
                    )
                continue
            if max_size is not None and column.domain_size > max_size:
                if diagnostics:
                    print(
                        "[importance_sampling] skip root column",
                        {
                            "column": column.name,
                            "reason": "domain_too_large",
                            "domain_size": column.domain_size,
                            "max_domain_size": max_size,
                        },
                        flush=True,
                    )
                continue
            if not include_categorical and semantic_type == "categorical":
                if diagnostics:
                    print(
                        "[importance_sampling] skip root column",
                        {"column": column.name, "reason": "categorical_disabled"},
                        flush=True,
                )
                continue
            if semantic_type == "ordered" and not _ordered_numeric_values(column.domain):
                raise ValueError(
                    f"importance_sampling.discovery.root_column_semantics declares "
                    f"{column.name!r} as ordered, but its complete domain is not "
                    "comparable numeric"
                )
            source_name = column.name.split(":", 1)[1]
            data_column = f"{root}.{source_name}"
            if data_column not in table_actor.df.columns:
                if diagnostics:
                    print(
                        "[importance_sampling] skip root column",
                        {"column": column.name, "reason": "not_in_root_dataframe"},
                        flush=True,
                    )
                continue
            started = time.perf_counter()
            merged = root_jct[[root_key, root_weight_column]].merge(
                root_values[[data_column]],
                how="left",
                left_on=root_key,
                right_index=True,
                indicator=True,
            )
            domain_to_id = _domain_index(column.domain)
            counts = np.zeros(column.domain_size, dtype=float)
            unmapped: Counter[str] = Counter()
            for raw_value, matched, weight in zip(
                merged[data_column].to_numpy(dtype=object),
                merged["_merge"].to_numpy(dtype=object),
                merged[root_weight_column].to_numpy(dtype=float),
            ):
                value = _canonical_root_value(raw_value, matched == "both")
                index = domain_to_id.get(value)
                if index is not None:
                    counts[index] += float(weight)
                else:
                    unmapped[_literal_id(value)] += 1
            mapped_mass = float(counts.sum())
            difference = float(total_root_mass - mapped_mass)
            if unmapped or abs(difference) > max(1.0e-6, abs(total_root_mass) * 1.0e-10):
                raise ValueError(
                    "root JCT DATA mass mapping is not lossless for "
                    f"{column.name!r}: mapped={mapped_mass}, "
                    f"root_jct_total={total_root_mass}, difference={difference}, "
                    f"unmapped_literals={dict(unmapped)}"
                )
            mapped_mass_by_column[column.name] = mapped_mass
            mass_difference_by_column[column.name] = difference
            masses[column_index] = RootColumnMass(
                column_index=column_index,
                values=column.domain,
                counts=counts,
                total_count=float(total_root_mass),
                source="neurocard_root_jct_weight",
                semantic_type=semantic_type,
                diagnostics={
                    "mapped_mass": mapped_mass,
                    "root_jct_total_weight": total_root_mass,
                    "difference": difference,
                    "semantic_type": semantic_type,
                },
            )
            if diagnostics:
                print(
                    "[importance_sampling] root column mass ready",
                    {
                        "column": column.name,
                        "domain_size": column.domain_size,
                        "nonzero_values": int(np.count_nonzero(counts)),
                        "mapped_mass": mapped_mass,
                        "total_count": float(total_root_mass),
                        "semantic_type": semantic_type,
                        "seconds": time.perf_counter() - started,
                    },
                    flush=True,
                )
        return cls(
            metadata,
            masses,
            source="neurocard_root_jct_weight",
            mass_diagnostics={
                "root_jct_total_weight": total_root_mass,
                "sampler_join_cardinality": sampler_join_cardinality,
                "metadata_join_cardinality": metadata_join_cardinality,
                "mapped_mass_by_column": mapped_mass_by_column,
                "mass_difference_by_column": mass_difference_by_column,
            },
        )

    def discover(
        self,
        *,
        n_total: int,
        predicate_probabilities: Mapping[str, float],
        minimum_expected_context_support: float,
        max_selected_strata: int,
    ) -> tuple[RootDataStratum, ...]:
        candidates: list[RootDataStratum] = []
        for column_index, mass in self.column_masses.items():
            column = self.metadata.columns[column_index]
            values = mass.values
            counts = mass.counts
            if mass.semantic_type == "ordered":
                candidates.extend(
                    _numeric_root_candidates(
                        column_index,
                        column.name,
                        values,
                        counts,
                        mass.total_count,
                        n_total=n_total,
                        predicate_probabilities=predicate_probabilities,
                        threshold=minimum_expected_context_support,
                        source=mass.source,
                        semantic_type=mass.semantic_type,
                    )
                )
            else:
                candidates.extend(
                    _categorical_root_candidates(
                        column_index,
                        column.name,
                        values,
                        counts,
                        mass.total_count,
                        n_total=n_total,
                        predicate_probabilities=predicate_probabilities,
                        threshold=minimum_expected_context_support,
                        source=mass.source,
                        semantic_type=mass.semantic_type,
                    )
                )
        selected = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.probability > 0.0 and candidate.support_deficit > 0.0
            ),
            key=lambda item: (-item.support_deficit, item.expected_target_rows, item.stratum_id),
        )[: max(0, int(max_selected_strata))]
        if not selected:
            return ()
        deficits = np.array([max(0.0, item.support_deficit) for item in selected], dtype=float)
        if float(deficits.sum()) <= 0.0:
            return ()
        alpha = deficits / float(deficits.sum())
        return tuple(
            RootDataStratum(**{**item.to_json_dict(), "alpha": float(alpha[index])})
            for index, item in enumerate(selected)
        )

    def sample_conditional(
        self,
        stratum: RootDataStratum,
        num_rows: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.encoded_rows is None:
            raise NotImplementedError("provider does not own materialized rows")
        mask = self.row_membership(self.encoded_rows, stratum)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError(f"stratum {stratum.stratum_id!r} has no materialized rows")
        row_weights = self.row_weights[indices]  # type: ignore[index]
        probabilities = row_weights / float(row_weights.sum())
        selected = rng.choice(indices, size=int(num_rows), replace=True, p=probabilities)
        return self.encoded_rows[selected]

    def row_membership(self, encoded_rows: np.ndarray, stratum: RootDataStratum) -> np.ndarray:
        column = self.metadata.columns[stratum.column_index]
        values = np.array(
            [column.domain[int(index)] for index in np.asarray(encoded_rows)[:, stratum.column_index]],
            dtype=object,
        )
        return np.array([stratum.contains_value(value) for value in values], dtype=bool)

    def membership_lookup(
        self,
        strata: tuple[RootDataStratum, ...],
    ) -> RootStratumMembershipLookup:
        return build_membership_lookup(self.metadata, strata)


def rho_for_memberships(
    memberships: np.ndarray,
    strata: tuple[RootDataStratum, ...],
    mixture_probability: float,
) -> np.ndarray:
    """Return exact rho=p/q for possibly overlapping rare strata.

    q(x) = p(x)[(1-lambda) + lambda sum_{s:x in A_s} alpha_s/P_s],
    so rho(x)=p(x)/q(x) is the reciprocal bracket.  Overlapping strata are
    handled by the membership sum; they are not treated as disjoint.
    """

    lam = float(mixture_probability)
    if lam == 0.0 or not strata:
        return np.ones(memberships.shape[0], dtype=float)
    if not (0.0 <= lam < 1.0):
        raise ValueError("importance_sampling.mixture_probability must be in [0, 1)")
    terms = np.full(memberships.shape[0], 1.0 - lam, dtype=float)
    for index, stratum in enumerate(strata):
        if stratum.probability <= 0.0:
            raise ValueError(f"stratum {stratum.stratum_id!r} has non-positive probability")
        terms += lam * memberships[:, index].astype(float) * (
            float(stratum.alpha) / float(stratum.probability)
        )
    return 1.0 / terms


def membership_matrix(
    metadata: ModelMetadata,
    encoded_rows: np.ndarray,
    strata: tuple[RootDataStratum, ...],
    lookup: RootStratumMembershipLookup | None = None,
) -> np.ndarray:
    rows = np.asarray(encoded_rows, dtype=np.int64)
    if lookup is not None:
        matrix = np.zeros((rows.shape[0], len(strata)), dtype=bool)
        for column_index, (stratum_indices, lookup_matrix) in lookup.by_column.items():
            matrix[:, stratum_indices] = lookup_matrix[rows[:, column_index]]
        return matrix
    matrix = np.zeros((rows.shape[0], len(strata)), dtype=bool)
    for stratum_index, stratum in enumerate(strata):
        column = metadata.columns[stratum.column_index]
        values = [column.domain[int(value)] for value in rows[:, stratum.column_index]]
        matrix[:, stratum_index] = [stratum.contains_value(value) for value in values]
    return matrix


def build_membership_lookup(
    metadata: ModelMetadata,
    strata: tuple[RootDataStratum, ...],
) -> RootStratumMembershipLookup:
    grouped: dict[int, list[int]] = {}
    for stratum_index, stratum in enumerate(strata):
        grouped.setdefault(stratum.column_index, []).append(stratum_index)
    by_column: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for column_index, stratum_indices in grouped.items():
        column = metadata.columns[column_index]
        lookup_matrix = np.zeros((column.domain_size, len(stratum_indices)), dtype=bool)
        for local_index, stratum_index in enumerate(stratum_indices):
            stratum = strata[stratum_index]
            lookup_matrix[:, local_index] = [
                stratum.contains_value(value) for value in column.domain
            ]
        by_column[column_index] = (
            np.asarray(stratum_indices, dtype=int),
            lookup_matrix,
        )
    return RootStratumMembershipLookup(by_column=by_column)


def predicate_probability_map(config: Mapping[str, Any]) -> dict[str, float]:
    return {
        "wildcard": float(config.get("wildcard_probability", 0.2)),
        "equality": float(config.get("equality_probability", 0.2)),
        "lower": float(config.get("lower_bound_probability", 0.2)),
        "upper": float(config.get("upper_bound_probability", 0.2)),
        "range": float(config.get("native_range_probability", 0.2)),
    }


def _numeric_root_candidates(
    column_index: int,
    column_name: str,
    values: tuple[Any, ...],
    counts: np.ndarray,
    total_count: float,
    *,
    n_total: int,
    predicate_probabilities: Mapping[str, float],
    threshold: float,
    source: str,
    semantic_type: str,
) -> Iterable[RootDataStratum]:
    comparable = [(index, float(value)) for index, value in enumerate(values) if _is_plain_number(value)]
    comparable.sort(key=lambda item: item[1])
    if not comparable or total_count <= 0:
        return ()
    ordered_indices = np.array([item[0] for item in comparable], dtype=int)
    ordered_values = [item[1] for item in comparable]
    ordered_counts = counts[ordered_indices].astype(float)
    probabilities = ordered_counts / float(total_count)
    prefix_counts = np.cumsum(ordered_counts)
    suffix_counts = np.cumsum(ordered_counts[::-1])[::-1]
    lower_support_by_threshold = _expected_lower_threshold_counts(
        probabilities,
        n_total=n_total,
        p_lower=float(predicate_probabilities.get("lower", 0.0)),
    )
    upper_support_by_threshold = _expected_upper_threshold_counts(
        probabilities,
        n_total=n_total,
        p_upper=float(predicate_probabilities.get("upper", 0.0)),
    )
    range_equality_support, range_lower_support, range_upper_support = (
        _expected_native_range_interval_counts(
            probabilities,
            n_total=n_total,
            p_range=float(predicate_probabilities.get("range", 0.0)),
        )
    )
    candidates: list[RootDataStratum] = []
    p_equal = float(predicate_probabilities.get("equality", 0.0))
    for rank, (domain_index, value) in enumerate(comparable):
        foj_count = float(ordered_counts[rank])
        probability = float(probabilities[rank])
        if probability <= 0:
            continue
        expected_equality = n_total * probability * p_equal
        candidates.append(
            _with_score(
                RootDataStratum(
                    stratum_id=f"{column_name}:eq:{_literal_id(value)}",
                    column_index=column_index,
                    column_name=column_name,
                    region_type="equality",
                    value=values[domain_index],
                    foj_count=foj_count,
                    probability=probability,
                    expected_target_rows=n_total * probability,
                    expected_equality_count=expected_equality,
                    expected_range_support=float(range_equality_support[rank]),
                    source=source,
                    semantic_type=semantic_type,
                ),
                threshold,
            )
        )
    # Prefix/suffix candidates target low expected lower/upper threshold support.
    for rank, value in enumerate(ordered_values):
        suffix_count = float(suffix_counts[rank])
        probability = suffix_count / float(total_count)
        expected_lower = float(lower_support_by_threshold[rank])
        if probability > 0.0:
            candidates.append(
                _with_score(
                    RootDataStratum(
                        stratum_id=f"{column_name}:ge:{_literal_id(value)}",
                        column_index=column_index,
                        column_name=column_name,
                        region_type="lower_tail",
                        lower=value,
                        foj_count=suffix_count,
                        probability=probability,
                        expected_target_rows=n_total * probability,
                        expected_lower_count=expected_lower,
                        expected_range_support=float(range_lower_support[rank]),
                        source=source,
                        semantic_type=semantic_type,
                    ),
                    threshold,
                )
            )
        prefix_count = float(prefix_counts[rank])
        probability = prefix_count / float(total_count)
        expected_upper = float(upper_support_by_threshold[rank])
        if probability > 0.0:
            candidates.append(
                _with_score(
                    RootDataStratum(
                        stratum_id=f"{column_name}:le:{_literal_id(value)}",
                        column_index=column_index,
                        column_name=column_name,
                        region_type="upper_tail",
                        upper=value,
                        foj_count=prefix_count,
                        probability=probability,
                        expected_target_rows=n_total * probability,
                        expected_upper_count=expected_upper,
                        expected_range_support=float(range_upper_support[rank]),
                        source=source,
                        semantic_type=semantic_type,
                    ),
                    threshold,
                )
            )
    return candidates


def _categorical_root_candidates(
    column_index: int,
    column_name: str,
    values: tuple[Any, ...],
    counts: np.ndarray,
    total_count: float,
    *,
    n_total: int,
    predicate_probabilities: Mapping[str, float],
    threshold: float,
    source: str,
    semantic_type: str,
) -> Iterable[RootDataStratum]:
    p_equal = float(predicate_probabilities.get("equality", 0.0))
    candidates = []
    for index, value in enumerate(values):
        if _is_structural_value(value):
            continue
        probability = float(counts[index]) / float(total_count)
        if probability <= 0.0:
            continue
        expected = n_total * probability * p_equal
        candidates.append(
            _with_score(
                RootDataStratum(
                    stratum_id=f"{column_name}:eq:{_literal_id(value)}",
                    column_index=column_index,
                    column_name=column_name,
                    region_type="equality",
                    value=value,
                    foj_count=float(counts[index]),
                    probability=probability,
                    expected_target_rows=n_total * probability,
                    expected_equality_count=expected,
                    source=source,
                    semantic_type=semantic_type,
                ),
                threshold,
            )
        )
    return candidates


def _expected_lower_threshold_counts(
    probabilities: np.ndarray,
    *,
    n_total: int,
    p_lower: float,
) -> np.ndarray:
    ranks = np.arange(1, probabilities.shape[0] + 1, dtype=float)
    return n_total * p_lower * np.cumsum((probabilities / ranks)[::-1])[::-1]


def _expected_upper_threshold_counts(
    probabilities: np.ndarray,
    *,
    n_total: int,
    p_upper: float,
) -> np.ndarray:
    ranks_from_right = np.arange(probabilities.shape[0], 0, -1, dtype=float)
    return n_total * p_upper * np.cumsum(probabilities / ranks_from_right)


def _expected_native_range_interval_counts(
    probabilities: np.ndarray,
    *,
    n_total: int,
    p_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expected native RANGE coverage for eq/lower-tail/upper-tail intervals.

    Duet row-specific range generation samples a literal row value X, then
    chooses L uniformly among ranks <= X and U uniformly among ranks >= X.
    These vectors return support for [v,v], [v,+inf], and [-inf,v].
    """

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty
    ranks = np.arange(1, probabilities.shape[0] + 1, dtype=float)
    right_counts = np.arange(probabilities.shape[0], 0, -1, dtype=float)
    equality = n_total * p_range * probabilities / (ranks * right_counts)

    suffix_p = np.cumsum(probabilities[::-1])[::-1]
    suffix_p_over_rank = np.cumsum((probabilities / ranks)[::-1])[::-1]
    lower = n_total * p_range * (
        suffix_p - (ranks - 1.0) * suffix_p_over_rank
    )

    prefix_p_over_right = np.cumsum(probabilities / right_counts)
    zero_based_rank = ranks - 1.0
    prefix_zero_rank_p_over_right = np.cumsum(
        probabilities * zero_based_rank / right_counts
    )
    upper = n_total * p_range * (
        ranks * prefix_p_over_right - prefix_zero_rank_p_over_right
    )
    return equality, lower, upper


def _with_score(stratum: RootDataStratum, threshold: float) -> RootDataStratum:
    support_candidates = [
        ("target_rows", stratum.expected_target_rows),
        ("equality", stratum.expected_equality_count),
        ("lower", stratum.expected_lower_count),
        ("upper", stratum.expected_upper_count),
        ("native_range", stratum.expected_range_support),
    ]
    applicable_values = [
        (name, float(value))
        for name, value in support_candidates
        if value is not None and isfinite(float(value))
    ]
    if not applicable_values:
        raise ValueError(f"stratum {stratum.stratum_id!r} has no applicable support metrics")
    support_bottleneck, support = min(applicable_values, key=lambda item: item[1])
    deficit = max(0.0, float(threshold) - float(support))
    return RootDataStratum(
        **{
            **stratum.to_json_dict(),
            "support_score": float(support),
            "support_deficit": float(deficit),
            "support_bottleneck": support_bottleneck,
        }
    )


def _ordered_numeric_values(values: tuple[Any, ...]) -> bool:
    ordinary = [value for value in values if not _is_structural_value(value)]
    return bool(ordinary) and all(_is_plain_number(value) for value in ordinary)


def _is_plain_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))


def _is_structural_value(value: Any) -> bool:
    return value in {OUTER_MISSING, SQL_NULL, None} or (isinstance(value, float) and np.isnan(value))


def _safe_ge(left: Any, right: Any) -> bool:
    try:
        return float(left) >= float(right)
    except (TypeError, ValueError):
        return False


def _safe_le(left: Any, right: Any) -> bool:
    try:
        return float(left) <= float(right)
    except (TypeError, ValueError):
        return False


def _canonical_numeric(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _canonical_root_value(value: Any, matched_root_row: bool) -> Any:
    if not matched_root_row:
        return OUTER_MISSING
    from model.src.data.complete_domain_preparation import canonicalize_base_value

    return canonicalize_base_value(value)


def _domain_index(values: tuple[Any, ...]) -> dict[Any, int]:
    mapping: dict[Any, int] = {}
    for index, value in enumerate(values):
        try:
            mapping[value] = index
        except TypeError as exc:
            raise TypeError(
                "importance root domains must contain hashable canonical values"
            ) from exc
    return mapping


def _semantic_for_column(
    column_name: str,
    root_column_semantics: Mapping[str, Any] | None,
) -> str:
    semantics = root_column_semantics or {}
    configured: Any = semantics.get(column_name)
    if configured is None and ":" in column_name:
        table, source_column = column_name.split(":", 1)
        nested = semantics.get(table)
        if isinstance(nested, Mapping):
            configured = nested.get(source_column)
    value = str(configured or "categorical").lower()
    if value not in {"ordered", "categorical"}:
        raise ValueError(
            "importance_sampling.discovery.root_column_semantics values must be "
            f"'ordered' or 'categorical', got {value!r} for {column_name!r}"
        )
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_mass_close(
    left: float,
    right: float | None,
    left_name: str,
    right_name: str,
) -> None:
    if right is None:
        return
    tolerance = max(1.0e-6, max(abs(left), abs(right)) * 1.0e-10)
    if abs(left - right) > tolerance:
        raise ValueError(
            f"{left_name} ({left}) does not match {right_name} ({right}) "
            f"within tolerance {tolerance}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _literal_id(value: Any) -> str:
    return str(_jsonable(value)).replace(" ", "_").replace(":", "_")
