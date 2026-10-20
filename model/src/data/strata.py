from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from math import isfinite, log
from typing import Any, Iterable, Mapping

import numpy as np

from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.data.schema import ColumnKind, ModelMetadata, root_paths_closure
from model.src.predicates.operators import PredicateOp


SQL_NULL = "__SQL_NULL__"

class RareSupportDiscoveryError(ValueError):
    """A refusal that carries the status that explains it.

    Discovery refuses in the cases a run must not survive, and those are
    exactly the cases where the diagnosis matters most.  Raising a bare
    ValueError strands it: the provider holds a fully populated
    conjunctive_status by the time any of these checks run, but the caller
    never gets the object, so the audit script -- whose whole job is to
    explain a failed configuration -- writes no report for the failure it
    exists to explain.  Subclassing ValueError keeps every existing caller and
    test working.
    """

    def __init__(self, message: str, conjunctive_status: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.conjunctive_status = dict(conjunctive_status or {})


DEFICIT_OBJECTIVES = ("clipped_sum", "log_ratio")
DEFAULT_DEFICIT_OBJECTIVE = "clipped_sum"


@dataclass(frozen=True)
class StratumTerm:
    """One column region inside a (possibly conjunctive) stratum.

    ``realization_family`` is the operator route the region is rarest through,
    which is the route the auxiliary context must actually force: an equality
    region whose native-range support is lower than its equality support has to
    be forced as ``range(v, v)``, not as ``= v``, or training reinforces a
    different event from the one discovery found to be rare.  It travels on the
    term because a conjunction is forced term by term and a scoring string on
    the stratum cannot describe more than one of them.

    ``realization_probability`` is the probability that a row already inside
    this region emits exactly this predicate, through that route.  For an equality term that is
    just ``p_equality``, because the row determines the literal.  For a tail or
    a native range the generator must also pick this exact threshold or
    interval out of all the ones the row admits, and on a high-cardinality
    ordered column that selection probability is small; leaving it out
    overestimates support by the mass-weighted mean number of admissible
    thresholds.  It is filled in from the depth-1 discovery statistics by
    ``with_realization_probabilities`` and is ``None`` until then.
    """

    column_index: int
    column_name: str
    region_type: str
    value: Any | None = None
    lower: Any | None = None
    upper: Any | None = None
    realization_probability: float | None = None
    realization_family: str | None = None

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
        for key in ("value", "lower", "upper"):
            payload[key] = _jsonable(payload[key])
        return payload


@dataclass(frozen=True)
class RootDataStratum:
    """A DATA region with exact mass under uniform FOJ tuples.

    A stratum is a conjunction of per-column terms.  ``terms`` is the full
    conjunction; the legacy scalar fields describe its **last** term, which is
    the one that decides which suffix heads the stratum can serve, so depth-1
    strata behave exactly as before and existing consumers keep working.
    """

    stratum_id: str
    column_index: int
    column_name: str
    region_type: str
    value: Any | None = None
    lower: Any | None = None
    upper: Any | None = None
    terms: tuple["StratumTerm", ...] = ()
    foj_count: float = 0.0
    probability: float = 0.0
    expected_target_rows: float = 0.0
    expected_equality_count: float | None = None
    expected_lower_count: float | None = None
    expected_upper_count: float | None = None
    expected_range_support: float | None = None
    realized_support_mass: float | None = None
    support_score: float = 0.0
    support_deficit: float = 0.0
    support_bottleneck: str | None = None
    alpha: float = 0.0
    source: str = "unknown"
    semantic_type: str = "categorical"

    def __post_init__(self) -> None:
        terms = tuple(
            term if isinstance(term, StratumTerm) else StratumTerm(**dict(term))
            for term in (self.terms or ())
        )
        if not terms:
            terms = (
                StratumTerm(
                    column_index=int(self.column_index),
                    column_name=str(self.column_name),
                    region_type=str(self.region_type),
                    value=self.value,
                    lower=self.lower,
                    upper=self.upper,
                ),
            )
        if len({term.column_index for term in terms}) != len(terms):
            raise ValueError(f"stratum {self.stratum_id!r} repeats a column across terms")
        # Order by column index so that the last term is the one that gates the
        # eligible suffix heads.
        terms = tuple(sorted(terms, key=lambda term: term.column_index))
        object.__setattr__(self, "terms", terms)
        last = terms[-1]
        object.__setattr__(self, "column_index", int(last.column_index))
        object.__setattr__(self, "column_name", str(last.column_name))
        object.__setattr__(self, "region_type", str(last.region_type))
        object.__setattr__(self, "value", last.value)
        object.__setattr__(self, "lower", last.lower)
        object.__setattr__(self, "upper", last.upper)

    @property
    def depth(self) -> int:
        return len(self.terms)

    @property
    def is_conjunctive(self) -> bool:
        return len(self.terms) > 1

    @property
    def column_indices(self) -> tuple[int, ...]:
        return tuple(term.column_index for term in self.terms)

    def tables(self, metadata: ModelMetadata) -> frozenset[str]:
        names = {
            metadata.columns[term.column_index].table
            for term in self.terms
        }
        return frozenset(name for name in names if name is not None)

    def contains_value(self, value: Any) -> bool:
        """Membership for the last term only; see ``contains_row`` for the conjunction."""

        return self.terms[-1].contains_value(value)

    def contains_row(self, encoded_row: np.ndarray, metadata: ModelMetadata) -> bool:
        row = np.asarray(encoded_row)
        for term in self.terms:
            column = metadata.columns[term.column_index]
            if not term.contains_value(column.domain[int(row[term.column_index])]):
                return False
        return True

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value"] = _jsonable(payload["value"])
        payload["lower"] = _jsonable(payload["lower"])
        payload["upper"] = _jsonable(payload["upper"])
        payload["terms"] = tuple(term.to_json_dict() for term in self.terms)
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
        extension_column_masses: Mapping[int, "RootColumnMass"] | None = None,
    ) -> None:
        self.metadata = metadata
        self.column_masses = dict(column_masses)
        # Depth-1 strata stay root-only, because the live sampler imposes them
        # by filtering the root join-count table.  A conjunction is only ever
        # sampled from materialized rows, so its *extension* terms may come
        # from any DATA column; without a separate pool every extension term
        # would be a root column and require_distinct_tables would reject all
        # of them, leaving conjunctive discovery silently inert.
        self.extension_column_masses = dict(extension_column_masses or {})
        # {column name: domain size} for columns a domain cap excluded.
        self.columns_skipped_by_domain_cap: dict[str, int] = {}
        # Columns left out because candidate_column_names did not list them.
        self.columns_excluded_by_name: tuple[str, ...] = ()
        self._non_structural_cache: dict[int, np.ndarray] = {}
        # Diagnostics filled by the beam search; see extend_with_conjunctions.
        self.pruned_joint_counts: dict[int, float] = {}
        self.pruned_joint_counts_sample: dict[int, list[float]] = {}
        self.discovered_columns: dict[int, int] = {}
        self._term_mask_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._term_factor_cache: dict[tuple[Any, ...], np.ndarray] = {}
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
        max_domain_size: int | None = None,
        column_names: Iterable[str] | None = None,
    ) -> "ExactRootStratumProvider":
        encoded_rows = np.asarray(encoded_rows, dtype=np.int64)
        root = metadata.join_root
        masses: dict[int, RootColumnMass] = {}
        extensions: dict[int, RootColumnMass] = {}
        skipped: dict[str, int] = {}
        excluded_by_name: list[str] = []
        limit = None if max_domain_size is None else int(max_domain_size)
        # The root-JCT provider already honours candidate_column_names; the
        # materialized path ignored it, so a fixture had no way to keep a key
        # column such as title.id out of the pool except by capping the domain,
        # which also excludes the high-cardinality columns the search needs.
        allowed = None if column_names is None else set(column_names)
        for column_index, column in enumerate(metadata.columns):
            if column.kind != ColumnKind.DATA:
                continue
            if allowed is not None and column.name not in allowed:
                excluded_by_name.append(column.name)
                continue
            # The same cap the root-JCT provider honours, applied to both pools
            # so widening the extension pool cannot quietly allocate a mass
            # vector per high-cardinality column.  Which columns it excluded is
            # recorded, because "the column I care about is not in the pool" is
            # otherwise indistinguishable from "no conjunction was found".
            if limit is not None and column.domain_size > limit:
                skipped[column.name] = int(column.domain_size)
                continue
            counts = np.bincount(
                encoded_rows[:, column_index],
                minlength=column.domain_size,
            ).astype(float)
            mass = RootColumnMass(
                column_index=column_index,
                values=column.domain,
                counts=counts,
                total_count=float(counts.sum()),
                source=source,
                semantic_type=_semantic_for_column(column.name, root_column_semantics),
            )
            # Every DATA column can extend a conjunction; only root columns can
            # seed a depth-1 stratum.
            extensions[column_index] = mass
            if column.table == root:
                masses[column_index] = mass
        provider = cls(
            metadata,
            masses,
            encoded_rows=encoded_rows,
            source=source,
            extension_column_masses=extensions,
        )
        provider.columns_skipped_by_domain_cap = dict(skipped)
        provider.columns_excluded_by_name = tuple(excluded_by_name)
        return provider

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

    @property
    def supports_conjunctive_strata(self) -> bool:
        """Conjunctions need exact joint counts, which only materialized rows give.

        The live root-JCT provider holds one marginal per root column and no
        way to intersect them, so conjunctive discovery is refused there rather
        than approximated.
        """

        return self.encoded_rows is not None

    PRUNED_SAMPLE_SIZE = 64

    def _record_pruned(self, column_index: int, counts: np.ndarray) -> None:
        """Keep the largest joint counts the floor rejected for a column.

        The maximum alone says whether *one* pairing could be admitted by a
        bigger fixture.  It does not say how much bigger a fixture would have to
        be for a usable *number* of them, which is the question a fixture-growth
        decision actually turns on, so keep the top of the rejected distribution
        rather than only its peak.
        """

        kept = self.pruned_joint_counts_sample.get(column_index, [])
        kept = sorted(
            kept + [float(value) for value in np.asarray(counts, dtype=float) if value > 0],
            reverse=True,
        )[: self.PRUNED_SAMPLE_SIZE]
        self.pruned_joint_counts_sample[column_index] = kept
        if kept:
            self.pruned_joint_counts[column_index] = kept[0]

    def _non_structural_mask(self, column_index: int) -> np.ndarray:
        """Encoded ids that carry a real value, not join padding.

        OUTER_MISSING, SQL NULL and None are padding: a full-outer-join child
        column is full of them.  Candidate builders discard them, so if they are
        ranked and truncated *before* that discard they silently consume slots
        of the candidate budget and the run ends up with fewer -- sometimes
        zero -- usable candidates while real values were available.
        """

        cached = self._non_structural_cache.get(column_index)
        if cached is not None:
            return cached
        mass = self.column_masses.get(column_index) or self.extension_column_masses.get(
            column_index
        )
        values = (
            mass.values if mass is not None else self.metadata.columns[column_index].domain
        )
        mask = np.fromiter(
            (not _is_structural_value(value) for value in values),
            dtype=bool,
            count=len(values),
        )
        self._non_structural_cache[column_index] = mask
        return mask

    def _term_domain_mask(self, term: "StratumTerm") -> np.ndarray:
        """Membership as a boolean lookup over the column's encoded domain.

        Membership depends only on the encoded id, so it is decided once per
        domain value rather than once per row.  Decoding every row to a Python
        object and calling ``contains_value`` on it -- which is what this
        replaced -- costs O(rows) interpreted work on every beam-search pair,
        and the beam evaluates beam_width * extension_width of them.
        """

        key = _term_cache_key(term)
        cached = self._term_mask_cache.get(key)
        if cached is not None:
            return cached
        column = self.metadata.columns[int(term.column_index)]
        mask = np.fromiter(
            (term.contains_value(value) for value in column.domain),
            dtype=bool,
            count=len(column.domain),
        )
        self._term_mask_cache[key] = mask
        return mask

    def _term_factor_vector(
        self,
        term: "StratumTerm",
        predicate_probabilities: Mapping[str, float],
    ) -> np.ndarray:
        # The vector depends on the operator probabilities as well as the term,
        # so they belong in the key: a provider reused across two discovery
        # calls with different probabilities would otherwise answer with the
        # first call's factors.
        key = _term_cache_key(term) + _probability_cache_key(predicate_probabilities)
        cached = self._term_factor_cache.get(key)
        if cached is not None:
            return cached
        mass = self.column_masses.get(int(term.column_index))
        values = (
            mass.values
            if mass is not None
            else self.metadata.columns[int(term.column_index)].domain
        )
        factors = per_value_realization_factors(
            term, values=values, predicate_probabilities=predicate_probabilities
        )
        self._term_factor_cache[key] = factors
        return factors

    def joint_count(self, terms: "tuple[StratumTerm, ...]") -> float:
        if self.encoded_rows is None:
            raise NotImplementedError("joint counts require materialized rows")
        mask = np.ones(self.encoded_rows.shape[0], dtype=bool)
        for term in terms:
            lookup = self._term_domain_mask(term)
            mask &= lookup[self.encoded_rows[:, int(term.column_index)]]
            if not mask.any():
                return 0.0
        weights = self.row_weights
        if weights is None:
            return float(mask.sum())
        return float(weights[mask].sum())

    def realized_support_mass(
        self,
        terms: "tuple[StratumTerm, ...]",
        predicate_probabilities: Mapping[str, float],
    ) -> float:
        """Exact ``E[1[row in S] * prod_i p_i(predicate_i | row)]`` over FOJ rows.

        Multiplying each term's *marginal* average realization probability
        assumes the terms are independent, which they are not: for a tail or a
        range the factor depends on the row's rank, and the other terms change
        the rank distribution inside the conjunction.  Since the provider holds
        the materialized rows and their weights, the product is averaged over
        the joint rows instead, which is exact rather than an approximation.
        """

        if self.encoded_rows is None:
            raise ValueError(
                "realized support mass needs materialized rows; this provider has none"
            )
        rows = self.encoded_rows
        weights = (
            self.row_weights
            if self.row_weights is not None
            else np.ones(rows.shape[0], dtype=float)
        )
        total = float(weights.sum())
        if total <= 0.0:
            return 0.0
        accumulated = weights.astype(float).copy()
        for term in terms:
            factors = self._term_factor_vector(term, predicate_probabilities)
            accumulated = accumulated * factors[rows[:, int(term.column_index)]]
            if not np.any(accumulated):
                return 0.0
        return float(accumulated.sum()) / total

    def _candidates_for_masses(
        self,
        masses: Mapping[int, "RootColumnMass"],
        *,
        n_total: int,
        predicate_probabilities: Mapping[str, float],
        threshold: float,
        max_candidates_per_column: int | None = None,
    ) -> list[RootDataStratum]:
        """Depth-1 region candidates for a set of column masses.

        A categorical column yields one equality region per *observed* value,
        so on a 2M-row fixture a 134k-domain id column really does yield ~134k
        of them.  ``max_candidates_per_column`` keeps the rarest, which are the
        only ones with a support deficit to repair; the rest are zeroed out of
        the counts, which leaves the kept candidates' probabilities exact
        because those are still taken over the full column total.  Ordered
        columns are left alone: their tail regions are cumulative, so dropping
        values would move the thresholds rather than just shorten the list.
        """

        candidates: list[RootDataStratum] = []
        limit = None if max_candidates_per_column is None else int(max_candidates_per_column)
        for column_index, mass in masses.items():
            column = self.metadata.columns[column_index]
            ordered = mass.semantic_type == "ordered"
            builder = _numeric_root_candidates if ordered else _categorical_root_candidates
            counts = np.asarray(mass.counts, dtype=float)
            if limit is not None and limit > 0 and not ordered:
                # Padding is excluded here too, for the same reason: the
                # builder drops it afterwards and the slot is never refilled.
                present = np.flatnonzero(
                    (counts > 0) & self._non_structural_mask(column_index)
                )
                if present.size > limit:
                    order = np.argsort(counts[present], kind="stable")[:limit]
                    trimmed = np.zeros_like(counts)
                    trimmed[present[order]] = counts[present[order]]
                    counts = trimmed
            candidates.extend(
                builder(
                    column_index,
                    column.name,
                    mass.values,
                    counts,
                    mass.total_count,
                    n_total=n_total,
                    predicate_probabilities=predicate_probabilities,
                    threshold=threshold,
                    source=mass.source,
                    semantic_type=mass.semantic_type,
                )
            )
        return candidates

    def _diagnose_uncovered_columns(
        self,
        uncovered: "list[str]",
        candidates: "list[RootDataStratum]",
        selected: "list[RootDataStratum]",
        *,
        minimum_joint_rows: float,
    ) -> dict[str, dict[str, Any]]:
        """Say which of the three stages lost each required column.

        A column can be lost by never reaching the pool, by the joint-row floor
        pruning every pairing, or by being out-ranked in the final deficit sort.
        Those need opposite fixes -- naming the columns, lowering the floor,
        widening selection -- so a failure that only says "not covered" costs a
        whole cluster round to narrow down.
        """

        by_name = {column.name: index for index, column in enumerate(self.metadata.columns)}
        cutoff = (
            min(float(item.support_deficit) for item in selected) if selected else 0.0
        )
        report: dict[str, dict[str, Any]] = {}
        for name in uncovered:
            index = by_name.get(name)
            discovered = [
                item
                for item in candidates
                if item.is_conjunctive and index is not None and index in item.column_indices
            ]
            pruned_best = self.pruned_joint_counts.get(index) if index is not None else None
            entry: dict[str, Any] = {
                "in_extension_pool": index in self.extension_column_masses,
                "conjunctions_discovered": len(discovered),
                "best_pruned_joint_count": pruned_best,
                "minimum_joint_rows": minimum_joint_rows,
                "selection_deficit_cutoff": cutoff,
            }
            # A column missing from the pool entirely is a configuration error
            # and the required_extension_columns guard above has already raised
            # by this point, so there is no branch for it here.
            if not discovered:
                if pruned_best is not None:
                    rows = (
                        0 if self.encoded_rows is None else int(self.encoded_rows.shape[0])
                    )
                    mass = self.extension_column_masses.get(index)
                    observed = (
                        int((np.asarray(mass.counts) > 0).sum()) if mass is not None else None
                    )
                    entry["observed_values"] = observed
                    # Measured scaling, no uniformity assumed: if the best
                    # pairing reached k rows at N, reaching the floor needs
                    # about N * floor / k rows.
                    needed = (
                        int(rows * minimum_joint_rows / pruned_best)
                        if pruned_best > 0 and rows
                        else None
                    )
                    entry["fixture_rows_for_this_floor"] = needed
                    # How much fixture a *usable number* of conjunctions needs,
                    # not just the single best pairing.  Growing a fixture to
                    # the point where one region squeaks past the floor buys one
                    # stratum, which is rarely the experiment anyone wants.
                    sample = self.pruned_joint_counts_sample.get(index, [])
                    entry["fixture_rows_for_n_conjunctions"] = {
                        str(rank): (
                            int(rows * minimum_joint_rows / sample[rank - 1])
                            if rows and len(sample) >= rank and sample[rank - 1] > 0
                            else None
                        )
                        for rank in (1, 5, 10, 25, 50)
                    }
                    entry["pruned_joint_counts_top"] = [
                        float(value) for value in sample[:10]
                    ]
                    entry["explanation"] = (
                        f"every pairing was pruned by minimum_joint_rows={minimum_joint_rows:g}; "
                        f"the largest joint count any pairing reached was {pruned_best:g} over "
                        f"{rows} materialized rows"
                        + (f" across {observed} observed values" if observed else "")
                        + ". No setting of extension_width or max_selected_strata can change "
                        "that, because the pairings were never candidates. Either lower "
                        f"minimum_joint_rows below {pruned_best:g} -- which means estimating "
                        "rarity from a handful of sampled rows -- or use a lower-cardinality "
                        "extension column, or grow the fixture to roughly "
                        f"{needed} rows"
                    )
                else:
                    entry["explanation"] = (
                        "no pairing was even evaluated; the column never entered the "
                        "extension pool within extension_width, so widen extension_width "
                        "or conditional_extension_width"
                    )
            else:
                best = max(float(item.support_deficit) for item in discovered)
                entry["best_discovered_deficit"] = best
                entry["explanation"] = (
                    f"{len(discovered)} conjunctions were discovered but out-ranked: the "
                    f"best had deficit {best:g} against a selection cutoff of {cutoff:g}, "
                    "so raise max_selected_strata or max_stratum_share"
                )
            report[name] = entry
        return report

    def _rows_matching(self, stratum: RootDataStratum) -> np.ndarray:
        mask = np.ones(self.encoded_rows.shape[0], dtype=bool)
        for term in stratum.terms:
            mask &= self._term_domain_mask(term)[
                self.encoded_rows[:, int(term.column_index)]
            ]
        return mask

    def conditional_extension_counts(
        self,
        seeds: "tuple[RootDataStratum, ...]",
        masses: Mapping[int, "RootColumnMass"],
        *,
        minimum_joint_rows: float,
        per_seed_width: int,
        require_distinct_tables: bool,
    ) -> dict[tuple[int, int], float]:
        """Smallest joint count of each (column, value) with any seed.

        Ranking extensions by their *marginal* count is the wrong objective for
        this search: a value can be common overall and almost absent alongside
        a particular seed, and that dependence is exactly what a conjunctive
        stratum is for.  Marginal pruning drops those values before the beam
        ever sees them.

        One ``bincount`` of a column over the rows a seed matches gives the
        joint count for *every* value of that column at once, so conditioning
        costs one pass per (seed, column) rather than one per candidate pair,
        and the count it returns is the exact joint count the beam would have
        computed anyway.
        """

        if self.encoded_rows is None:
            return {}
        weights = self.row_weights
        best: dict[tuple[int, int], float] = {}
        width = int(per_seed_width)
        if width <= 0:
            # Explicitly disabled, which leaves the marginal pool as the only
            # route.  Useful for isolating that path; not a sensible default,
            # because marginal ranking cannot see a joint tail.
            return {}
        for seed in seeds:
            seed_mask = self._rows_matching(seed)
            if not seed_mask.any():
                continue
            seed_columns = set(seed.column_indices)
            seed_tables = seed.tables(self.metadata)
            seed_rows = self.encoded_rows[seed_mask]
            seed_weights = None if weights is None else weights[seed_mask]
            for column_index in masses:
                if column_index in seed_columns:
                    continue
                column = self.metadata.columns[column_index]
                if require_distinct_tables and column.table in seed_tables:
                    continue
                counts = np.bincount(
                    seed_rows[:, column_index],
                    weights=seed_weights,
                    minlength=len(column.domain),
                ).astype(float)
                real = self._non_structural_mask(column_index)
                eligible = np.flatnonzero((counts >= float(minimum_joint_rows)) & real)
                if eligible.size == 0:
                    # The floor also filters here, before the beam ever sees a
                    # pairing.  Record the best it rejected, or a column lost at
                    # this stage looks like one that was merely out-ranked.
                    below = counts[real & (counts > 0)]
                    if below.size:
                        self._record_pruned(int(column_index), below)
                    continue
                rarest = eligible[np.argsort(counts[eligible], kind="stable")[:width]]
                for encoded_id in rarest:
                    key = (int(column_index), int(encoded_id))
                    value = float(counts[int(encoded_id)])
                    if key not in best or value < best[key]:
                        best[key] = value
        return best

    def _conditional_extension_candidates(
        self,
        seeds: "tuple[RootDataStratum, ...]",
        masses: Mapping[int, "RootColumnMass"],
        *,
        n_total: int,
        predicate_probabilities: Mapping[str, float],
        threshold: float,
        minimum_joint_rows: float,
        per_seed_width: int,
        require_distinct_tables: bool,
    ) -> list[RootDataStratum]:
        """Equality candidates for the values that are rarest given a seed."""

        counts = self.conditional_extension_counts(
            seeds,
            masses,
            minimum_joint_rows=minimum_joint_rows,
            per_seed_width=per_seed_width,
            require_distinct_tables=require_distinct_tables,
        )
        if not counts:
            return []
        total = self._total_mass()
        if total <= 0.0:
            return []
        candidates: list[RootDataStratum] = []
        for (column_index, encoded_id), joint_count in counts.items():
            mass = masses.get(column_index)
            if mass is None:
                continue
            value = mass.values[encoded_id]
            if _is_structural_value(value):
                continue
            marginal = float(np.asarray(mass.counts, dtype=float)[encoded_id])
            if marginal <= 0.0 or mass.total_count <= 0.0:
                continue
            probability = marginal / float(mass.total_count)
            candidates.append(
                _with_score(
                    RootDataStratum(
                        stratum_id=(
                            f"{self.metadata.columns[column_index].name}"
                            f":eq:{_literal_id(value)}"
                        ),
                        column_index=column_index,
                        column_name=self.metadata.columns[column_index].name,
                        region_type="equality",
                        value=value,
                        # foj_count carries the *conditional* count here, which
                        # is what ranks the pool; probability stays marginal so
                        # downstream scoring is unchanged.
                        foj_count=float(joint_count),
                        probability=probability,
                        expected_target_rows=n_total * probability,
                        expected_equality_count=(
                            n_total
                            * probability
                            * float(predicate_probabilities.get("equality", 0.0))
                        ),
                        source=f"{self.source}_conditional",
                        semantic_type="categorical",
                    ),
                    threshold,
                )
            )
        return candidates

    def _total_mass(self) -> float:
        if self.row_weights is not None:
            return float(self.row_weights.sum())
        if self.encoded_rows is not None:
            return float(self.encoded_rows.shape[0])
        return float(sum(mass.total_count for mass in self.column_masses.values()))

    def extend_with_conjunctions(
        self,
        seeds: "tuple[RootDataStratum, ...]",
        *,
        n_total: int,
        predicate_probabilities: Mapping[str, float],
        threshold: float,
        max_depth: int,
        beam_width: int,
        minimum_joint_rows: float,
        require_distinct_tables: bool,
        extensions: "tuple[RootDataStratum, ...] | None" = None,
        objective: str = DEFAULT_DEFICIT_OBJECTIVE,
    ) -> "tuple[RootDataStratum, ...]":
        """Beam search for conjunctions, scored by repairable per-head deficit.

        Extension candidates are the seed terms themselves, so the search needs
        no enumeration of value pairs: it intersects boolean masks it can
        already build.  A candidate is dropped as soon as its joint row count
        falls below ``minimum_joint_rows``, because a region with too few rows
        cannot be conditionally sampled without memorizing those rows.
        """

        if not self.supports_conjunctive_strata or max_depth <= 1:
            return ()
        total = float(sum(mass.total_count for mass in self.column_masses.values()))
        if self.row_weights is not None:
            total = float(self.row_weights.sum())
        elif self.encoded_rows is not None:
            total = float(self.encoded_rows.shape[0])
        if total <= 0.0:
            return ()
        root = self.metadata.join_root
        seeds = tuple(
            with_realization_probabilities(seed, predicate_probabilities) for seed in seeds
        )
        # Extension terms default to the seeds' own, which is only useful when
        # the seeds already span several tables.  Discovery passes an explicit
        # all-DATA-column pool so a root-only seed set can still reach across
        # tables.
        extension_strata = tuple(
            with_realization_probabilities(candidate, predicate_probabilities)
            for candidate in (seeds if extensions is None else extensions)
        )
        pool = [candidate.terms[0] for candidate in extension_strata]
        beam = list(seeds)
        produced: dict[str, RootDataStratum] = {}
        for _ in range(max_depth - 1):
            scored: list[RootDataStratum] = []
            for stratum in beam:
                used = set(stratum.column_indices)
                used_tables = set(stratum.tables(self.metadata))
                for term in pool:
                    if term.column_index in used:
                        continue
                    table = self.metadata.columns[term.column_index].table
                    if require_distinct_tables and table in used_tables:
                        continue
                    terms = tuple(sorted(stratum.terms + (term,), key=lambda item: item.column_index))
                    count = self.joint_count(terms)
                    if count < minimum_joint_rows:
                        # Record how close the floor came to being clearable for
                        # this column.  "No conjunction used column X" is
                        # otherwise indistinguishable between "the floor pruned
                        # every pairing" and "they were out-ranked", and those
                        # need opposite fixes.
                        self._record_pruned(int(term.column_index), np.array([count]))
                        continue
                    self.discovered_columns[int(term.column_index)] = (
                        self.discovered_columns.get(int(term.column_index), 0) + 1
                    )
                    candidate = RootDataStratum(
                        stratum_id="&".join(
                            f"{item.column_name}:{item.region_type}:{_literal_id(item.value if item.value is not None else item.lower if item.lower is not None else item.upper)}"
                            for item in terms
                        ),
                        column_index=terms[-1].column_index,
                        column_name=terms[-1].column_name,
                        region_type=terms[-1].region_type,
                        value=terms[-1].value,
                        lower=terms[-1].lower,
                        upper=terms[-1].upper,
                        terms=terms,
                        foj_count=count,
                        probability=count / total,
                        expected_target_rows=n_total * (count / total),
                        realized_support_mass=self.realized_support_mass(
                            terms, predicate_probabilities
                        ),
                        source=self.source,
                        semantic_type="conjunctive",
                    )
                    scored.append(
                        with_conditional_score(
                            candidate,
                            metadata=self.metadata,
                            n_total=n_total,
                            predicate_probabilities=predicate_probabilities,
                            threshold=threshold,
                            root_table=root,
                            objective=objective,
                        )
                    )
            if not scored:
                break
            scored.sort(key=lambda item: (-item.support_deficit, item.stratum_id))
            beam = scored[: max(1, int(beam_width))]
            for candidate in beam:
                produced[candidate.stratum_id] = candidate
        return tuple(produced.values())

    def discover(
        self,
        *,
        n_total: int,
        predicate_probabilities: Mapping[str, float],
        minimum_expected_context_support: float,
        max_selected_strata: int,
        conjunctive: Mapping[str, Any] | None = None,
        discovery_limits: Mapping[str, Any] | None = None,
    ) -> tuple[RootDataStratum, ...]:
        conjunctive = dict(conjunctive or {})
        discovery_limits = dict(discovery_limits or {})
        conjunctive_requested = bool(conjunctive.get("enabled", False))
        objective = str(
            discovery_limits.get("deficit_objective") or DEFAULT_DEFICIT_OBJECTIVE
        )
        self.pruned_joint_counts = {}
        self.pruned_joint_counts_sample = {}
        self.discovered_columns = {}
        self.conjunctive_status: dict[str, Any] = {
            "requested": conjunctive_requested,
            "applied": False,
            "reason": None,
            "conjunctive_selected": 0,
        }
        if conjunctive_requested and self.supports_conjunctive_strata:
            # A conjunction's rarity is estimated from the materialized rows, so
            # the row count bounds what "rare" can even mean here.  The default
            # prepared fixture is sized for validation batches, not for support
            # estimation: 512 rows against a 2.1e12-row join cannot distinguish
            # a rare joint region from an absent one, and minimum_joint_rows=100
            # would then demand that a "rare" conjunction cover a fifth of the
            # sample.  Refuse rather than produce numbers that look like exact
            # joint counts.
            available = int(self.encoded_rows.shape[0])
            floor = int(conjunctive.get("minimum_materialized_rows", 0) or 0)
            if floor > 0 and available < floor:
                raise ValueError(
                    f"conjunctive discovery needs at least {floor} materialized rows "
                    f"and this provider has {available}. Rebuild the prepared fixture "
                    "with python3 -m model.scripts.prepare_neurocard_data --config "
                    "<config> --rebuild-domains --sample-rows <N>, or lower "
                    "rare_support.discovery.conjunctive.minimum_materialized_rows "
                    "to accept the smaller sample"
                )
            required_columns = tuple(conjunctive.get("required_extension_columns", ()) or ())
            if required_columns:
                present = {
                    self.metadata.columns[index].name
                    for index in self.extension_column_masses
                }
                missing = [name for name in required_columns if name not in present]
                if missing:
                    causes = []
                    capped = {
                        name: size
                        for name, size in self.columns_skipped_by_domain_cap.items()
                        if name in missing
                    }
                    if capped:
                        causes.append(
                            f"max_candidate_domain_size excluded {capped} by domain size"
                        )
                    by_name = [
                        name for name in self.columns_excluded_by_name if name in missing
                    ]
                    if by_name:
                        causes.append(f"candidate_column_names does not list {by_name}")
                    if not causes:
                        causes.append("they are not DATA columns of this schema")
                    raise ValueError(
                        f"required_extension_columns {missing!r} are not in the "
                        "extension candidate pool, so no conjunction can use them: "
                        + "; ".join(causes)
                    )
            joint_floor = float(conjunctive.get("minimum_joint_rows", 32.0))
            max_fraction = float(conjunctive.get("max_joint_row_fraction", 0.05) or 0.0)
            if max_fraction > 0.0 and available > 0 and joint_floor / available > max_fraction:
                raise ValueError(
                    f"minimum_joint_rows={joint_floor:g} is "
                    f"{joint_floor / available:.1%} of the {available} materialized rows, "
                    f"above max_joint_row_fraction={max_fraction:.1%}. A region that "
                    "common is not rare, so the search would only ever return regions "
                    "the model already sees"
                )
        if conjunctive_requested and not self.supports_conjunctive_strata:
            self.conjunctive_status["reason"] = (
                "provider has no materialized rows, so exact joint counts are "
                "unavailable; falling back to depth-1 strata"
            )
            # require_conjunctions has to be honoured here too.  Checking it only
            # after discovery meant the oldest fallback of all -- a live sampler
            # with no materialized rows -- walked straight past the fail-closed
            # contract and returned depth-1 strata without error.
            if bool(conjunctive.get("require_conjunctions", False)):
                raise ValueError(
                    "require_conjunctions=true but this provider has no materialized "
                    "rows, so no conjunction can be counted or sampled. Use a fixture "
                    "or prepared-directory source; live NeuroCard sampling filters the "
                    "root join-count table one column at a time and cannot impose a "
                    "conjunction"
                )
            conjunctive_requested = False
        max_candidates_per_column = discovery_limits.get("max_candidates_per_column")
        unsupported_fixed_realization_ids: set[str] = set()

        def fixed_realizable(
            items: Iterable[RootDataStratum],
        ) -> list[RootDataStratum]:
            """Attach realization routes and remove contexts production cannot force."""

            retained = []
            for item in items:
                realized = with_realization_probabilities(item, predicate_probabilities)
                if fixed_forced_predicate_is_supported(realized, self.metadata):
                    retained.append(realized)
                else:
                    unsupported_fixed_realization_ids.add(realized.stratum_id)
            return retained

        candidates = self._candidates_for_masses(
            self.column_masses,
            n_total=n_total,
            predicate_probabilities=predicate_probabilities,
            threshold=minimum_expected_context_support,
            max_candidates_per_column=max_candidates_per_column,
        )
        candidates = fixed_realizable(candidates)
        if conjunctive_requested:
            root = self.metadata.join_root
            require_distinct_tables = bool(conjunctive.get("require_distinct_tables", True))

            def rescore(items: Iterable[RootDataStratum]) -> list[RootDataStratum]:
                scored = [
                    with_conditional_score(
                        item,
                        metadata=self.metadata,
                        n_total=n_total,
                        predicate_probabilities=predicate_probabilities,
                        threshold=minimum_expected_context_support,
                        root_table=root,
                        objective=objective,
                    )
                    for item in items
                    if item.probability > 0.0
                ]
                return fixed_realizable(scored)

            def by_deficit(items: list[RootDataStratum], width: int) -> tuple[RootDataStratum, ...]:
                return tuple(
                    sorted(items, key=lambda item: (-item.support_deficit, item.stratum_id))[
                        : max(1, int(width))
                    ]
                )

            rescored = rescore(candidates)
            seeds = by_deficit(rescored, conjunctive.get("seed_width", 32))
            # Seeds are root-only because a depth-1 stratum has to be samplable
            # on the live path too.  Extension terms are drawn from every DATA
            # column, or a root-only pool would make require_distinct_tables
            # reject every extension and leave discovery inert.
            extension_masses = self.extension_column_masses or self.column_masses
            scored_extensions = rescore(
                self._candidates_for_masses(
                    extension_masses,
                    n_total=n_total,
                    predicate_probabilities=predicate_probabilities,
                    threshold=minimum_expected_context_support,
                    max_candidates_per_column=max_candidates_per_column,
                )
            )
            # Drop candidates that cannot extend any seed *before* applying the
            # width.  Ranking first lets high-scoring but unusable root-table
            # regions consume the whole budget, which on a real root table with
            # many candidate regions makes discovery inert again -- the same
            # failure as a root-only pool, just harder to see.
            eligible = [
                candidate
                for candidate in scored_extensions
                if _can_extend_any_seed(
                    candidate,
                    seeds,
                    metadata=self.metadata,
                    require_distinct_tables=require_distinct_tables,
                )
            ]
            extension_width = max(
                1, int(conjunctive.get("extension_width", conjunctive.get("seed_width", 32)))
            )
            # Conditional candidates come first.  They are the ones chosen for
            # being rare *given a seed*, which is the dependence this search
            # exists to find; the marginal pool above is only a prior and on its
            # own prunes away any value that is common overall.
            conditional = self._conditional_extension_candidates(
                seeds,
                extension_masses,
                n_total=n_total,
                predicate_probabilities=predicate_probabilities,
                threshold=minimum_expected_context_support,
                minimum_joint_rows=float(conjunctive.get("minimum_joint_rows", 32.0)),
                per_seed_width=int(conjunctive.get("conditional_extension_width", 8)),
                require_distinct_tables=require_distinct_tables,
            )
            conditional = [
                with_conditional_score(
                    item,
                    metadata=self.metadata,
                    n_total=n_total,
                    predicate_probabilities=predicate_probabilities,
                    threshold=minimum_expected_context_support,
                    root_table=root,
                    objective=objective,
                )
                for item in conditional
            ]
            chosen: list[RootDataStratum] = []
            seen_ids: set[str] = set()
            for item in sorted(
                conditional, key=lambda entry: (entry.foj_count, entry.stratum_id)
            ):
                if len(chosen) >= extension_width:
                    break
                if item.stratum_id in seen_ids:
                    continue
                seen_ids.add(item.stratum_id)
                chosen.append(item)
            for item in by_deficit(eligible, extension_width) if eligible else ():
                if len(chosen) >= extension_width:
                    break
                if item.stratum_id in seen_ids:
                    continue
                seen_ids.add(item.stratum_id)
                chosen.append(item)
            extension_pool = tuple(chosen)
            self.conjunctive_status["conditional_extension_candidates"] = len(conditional)
            cross_table = tuple(
                candidate
                for candidate in extension_pool
                if self.metadata.columns[candidate.column_index].table != root
            )
            self.conjunctive_status["deficit_objective"] = objective
            self.conjunctive_status["scored_extension_candidates"] = len(scored_extensions)
            self.conjunctive_status["eligible_extension_candidates"] = len(eligible)
            self.conjunctive_status["extension_candidates"] = len(extension_pool)
            self.conjunctive_status["cross_table_extension_candidates"] = len(cross_table)
            conjunctions = self.extend_with_conjunctions(
                seeds,
                n_total=n_total,
                predicate_probabilities=predicate_probabilities,
                threshold=minimum_expected_context_support,
                max_depth=int(conjunctive.get("max_depth", 2)),
                beam_width=int(conjunctive.get("beam_width", 32)),
                minimum_joint_rows=float(conjunctive.get("minimum_joint_rows", 32.0)),
                require_distinct_tables=require_distinct_tables,
                extensions=extension_pool,
                objective=objective,
            )
            candidates = list(rescored) + list(conjunctions)
            self.conjunctive_status["conjunctive_candidates"] = len(conjunctions)
            if not conjunctions:
                if not eligible:
                    self.conjunctive_status["reason"] = (
                        "no extension candidate can extend any seed"
                        + (
                            f"; require_distinct_tables=true and every candidate is on "
                            f"the root table {root!r}"
                            if require_distinct_tables
                            else ""
                        )
                    )
                else:
                    self.conjunctive_status["reason"] = (
                        "the beam search found no region above minimum_joint_rows; "
                        "lower it or widen beam_width/extension_width"
                    )
        candidates = fixed_realizable(candidates)
        self.conjunctive_status[
            "unsupported_fixed_native_range_tail_candidates_excluded"
        ] = len(unsupported_fixed_realization_ids)
        # A row floor for strata of ANY depth.  minimum_joint_rows guards only
        # conjunctions, so the depth-1 path had no floor at all: it was safe
        # only as long as candidate_column_names kept it to dense columns.
        # Opening up a sparse high-cardinality column produced 53 of 64 strata
        # covering a single fixture row apiece, taking 83% of the auxiliary
        # batch -- about 160,000 draws of one row each over a 20k-step run,
        # which is memorisation, not rare-support training.  log_ratio makes
        # this worse rather than better, because an unbounded rarity objective
        # ranks a one-row region top.
        row_floor = float(discovery_limits.get("minimum_stratum_rows") or 0.0)
        eligible_candidates = [
            candidate
            for candidate in candidates
            if candidate.probability > 0.0 and candidate.support_deficit > 0.0
        ]
        starved = []
        # Only meaningful against materialized rows.  On the live NeuroCard
        # path the candidate counts are root-table rows rather than
        # full-outer-join rows, so the same number would silently mean a
        # different thing, and conditional sampling redraws the fanout
        # partners every step, so a stratum's support is not a fixed handful
        # that can be memorized.  A floor that quietly does nothing is worse
        # than no floor at all, so the status reports which of the two
        # happened instead of leaving a configured floor indistinguishable
        # from an unconfigured one.
        row_floor_applied = bool(row_floor > 0.0 and self.encoded_rows is not None)
        if row_floor_applied:
            kept = []
            for candidate in eligible_candidates:
                if float(candidate.foj_count) < row_floor:
                    starved.append(candidate)
                else:
                    kept.append(candidate)
            eligible_candidates = kept
        selected = sorted(
            eligible_candidates,
            key=lambda item: (-item.support_deficit, item.expected_target_rows, item.stratum_id),
        )[: max(0, int(max_selected_strata))]
        self.row_floor_status = {
            "minimum_stratum_rows": row_floor,
            "row_floor_applied": row_floor_applied,
            "candidates_below_row_floor": len(starved),
            "selected_stratum_rows": (
                {
                    "min": float(min(item.foj_count for item in selected)),
                    "median": float(
                        sorted(item.foj_count for item in selected)[len(selected) // 2]
                    ),
                    "max": float(max(item.foj_count for item in selected)),
                    "at_or_below_one": sum(
                        1 for item in selected if float(item.foj_count) <= 1.0
                    ),
                }
                if selected
                else None
            ),
        }
        conjunctive_selected = sum(1 for item in selected if item.is_conjunctive)
        self.conjunctive_status["conjunctive_selected"] = conjunctive_selected
        if conjunctive_requested:
            self.conjunctive_status["materialized_rows"] = (
                None if self.encoded_rows is None else int(self.encoded_rows.shape[0])
            )
            self.conjunctive_status["columns_skipped_by_domain_cap"] = dict(
                self.columns_skipped_by_domain_cap
            )
            # An allowlist is as capable of hiding the column that carries the
            # tail as a domain cap is, and it leaves no other trace, so report
            # both rather than only the one that looks like a limit.
            self.conjunctive_status["columns_excluded_by_name"] = list(
                self.columns_excluded_by_name
            )
            self.conjunctive_status["candidate_columns"] = sorted(
                self.metadata.columns[index].name
                for index in self.extension_column_masses
            )
        # "applied" means conjunctions are actually in the selection.  Reporting
        # applied=true with a null reason while every selected stratum is
        # depth-1 is how a run can look like it exercised the feature when it
        # did not.
        self.conjunctive_status["applied"] = bool(conjunctive_selected)
        self.conjunctive_status.update(self.row_floor_status)
        if conjunctive_requested:
            required_columns = tuple(conjunctive.get("required_extension_columns", ()) or ())
            covered = {
                self.metadata.columns[index].name
                for item in selected
                if item.is_conjunctive
                for index in item.column_indices
            }
            uncovered = [name for name in required_columns if name not in covered]
            self.conjunctive_status["required_columns_covered"] = [
                name for name in required_columns if name in covered
            ]
            self.conjunctive_status["required_columns_uncovered"] = uncovered
            # Being in the candidate pool is not the same as surviving capping,
            # global ranking, the beam and the final deficit sort.  A run can
            # otherwise select conjunctions that use none of the columns the
            # experiment is about and still report success.
            diagnosis = self._diagnose_uncovered_columns(
                uncovered,
                candidates,
                selected,
                minimum_joint_rows=float(conjunctive.get("minimum_joint_rows", 32.0)),
            )
            self.conjunctive_status["required_column_diagnosis"] = diagnosis
            if uncovered and bool(conjunctive.get("require_conjunctions", False)):
                raise RareSupportDiscoveryError(
                    f"required_extension_columns {uncovered!r} appear in no selected "
                    f"conjunctive stratum ({conjunctive_selected} were selected).\n"
                    + "\n".join(
                        f"  {name}: {diagnosis[name]['explanation']}" for name in uncovered
                    ),
                    self.conjunctive_status,
                )
        if conjunctive_requested and not conjunctive_selected:
            if self.conjunctive_status.get("reason") is None:
                self.conjunctive_status["reason"] = (
                    "conjunctions were discovered but none ranked into the top "
                    "max_selected_strata by support deficit"
                )
            if bool(conjunctive.get("require_conjunctions", False)):
                skipped = self.columns_skipped_by_domain_cap
                raise RareSupportDiscoveryError(
                    "require_conjunctions=true but no conjunctive stratum was "
                    f"selected: {self.conjunctive_status['reason']}"
                    + (
                        "; note that max_candidate_domain_size excluded "
                        f"{sorted(skipped)} from the candidate pool"
                        if skipped
                        else ""
                    ),
                    self.conjunctive_status,
                )
        if not selected:
            return ()
        deficits = np.array([max(0.0, item.support_deficit) for item in selected], dtype=float)
        if float(deficits.sum()) <= 0.0:
            return ()
        alpha = deficits / float(deficits.sum())
        alpha = _capped_allocation(alpha, float(conjunctive.get("max_stratum_share", 0.0)))
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
        rows = np.asarray(encoded_rows)
        mask = np.ones(rows.shape[0], dtype=bool)
        for term in stratum.terms:
            column = self.metadata.columns[term.column_index]
            values = [column.domain[int(index)] for index in rows[:, term.column_index]]
            mask &= np.array([term.contains_value(value) for value in values], dtype=bool)
        return mask

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
    # A stratum is a conjunction, so membership starts true and each term's
    # column mask is ANDed in.  Within one column key every stratum appears at
    # most once, which is what makes the in-place AND well defined.
    matrix = np.ones((rows.shape[0], len(strata)), dtype=bool)
    if lookup is not None:
        for column_index, (stratum_indices, lookup_matrix) in lookup.by_column.items():
            matrix[:, stratum_indices] &= lookup_matrix[rows[:, column_index]]
        return matrix
    for stratum_index, stratum in enumerate(strata):
        for term in stratum.terms:
            column = metadata.columns[term.column_index]
            # One decision per domain value, then a gather: the per-row decode
            # this replaced was O(rows) interpreted work per term.
            lookup = np.fromiter(
                (term.contains_value(value) for value in column.domain),
                dtype=bool,
                count=len(column.domain),
            )
            matrix[:, stratum_index] &= lookup[rows[:, term.column_index]]
    return matrix


def build_membership_lookup(
    metadata: ModelMetadata,
    strata: tuple[RootDataStratum, ...],
) -> RootStratumMembershipLookup:
    grouped: dict[int, list[tuple[int, StratumTerm]]] = {}
    for stratum_index, stratum in enumerate(strata):
        for term in stratum.terms:
            grouped.setdefault(term.column_index, []).append((stratum_index, term))
    by_column: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for column_index, entries in grouped.items():
        column = metadata.columns[column_index]
        lookup_matrix = np.zeros((column.domain_size, len(entries)), dtype=bool)
        for local_index, (_, term) in enumerate(entries):
            lookup_matrix[:, local_index] = [
                term.contains_value(value) for value in column.domain
            ]
        by_column[column_index] = (
            np.asarray([stratum_index for stratum_index, _ in entries], dtype=int),
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


TERM_OPERATOR_FAMILY = {
    "equality": "equality",
    "lower_tail": "lower",
    "upper_tail": "upper",
    "range": "range",
}


def table_subset_inclusion_probability(
    required_tables: frozenset[str] | set[str],
    *,
    metadata: ModelMetadata,
    root_table: str | None,
) -> float:
    """Probability that NeuroCard table dropout keeps every required table.

    The upstream law, as recorded in the trainer diagnostics, is
    ``num_dropped ~ Uniform{1..T-1}`` and ``P(drop table | num_dropped) =
    num_dropped / T`` with the root always kept.  Treating the required tables
    as conditionally independent given ``num_dropped`` gives the closed form
    below.  It is an approximation of a without-replacement draw, so it is
    only ever used to rank and threshold candidates, never as an exact mass.
    """

    # Keeping a table also means keeping every table between it and the root:
    # the sampler runs the subset through the root-connected component, so a
    # grandchild whose parent was dropped does not survive either.  Counting
    # only the explicitly required tables overestimates survival on any join
    # deeper than one hop.
    required = {table for table in required_tables if table is not None}
    total = len({column.table for column in metadata.columns if column.table is not None})
    # Scoring calls this once per candidate per served head, and the root-path
    # closure walks the join graph, so memoize on what the answer depends on.
    return _table_subset_inclusion_probability(
        frozenset(required), root_table, total, tuple(metadata.join_edges)
    )


@lru_cache(maxsize=4096)
def _table_subset_inclusion_probability(
    required: frozenset[str],
    root_table: str | None,
    total: int,
    edges: tuple[tuple[str, str], ...],
) -> float:
    if root_table is not None and required:
        required = frozenset(root_paths_closure(set(required), root_table, edges))
    tables = {table for table in required if table != root_table}
    if total <= 1 or not tables:
        return 1.0
    k = len(tables)
    return float(
        sum((1.0 - dropped / total) ** k for dropped in range(1, total))
        / float(total - 1)
    )


EXPECTED_SUPPORT_FIELD = {
    "equality": "expected_equality_count",
    "lower": "expected_lower_count",
    "upper": "expected_upper_count",
    "range": "expected_range_support",
}


def _probability_cache_key(
    predicate_probabilities: Mapping[str, float],
) -> tuple[Any, ...]:
    """The operator probabilities a realization factor can depend on."""

    return tuple(
        float(predicate_probabilities.get(family, 0.0))
        for family in ("equality", "lower", "upper", "range")
    )


def _term_cache_key(term: "StratumTerm") -> tuple[Any, ...]:
    """Identity of a term for caching, independent of literal value types."""

    return (
        int(term.column_index),
        str(term.region_type),
        str(term.realization_family),
        _literal_id(term.value),
        _literal_id(term.lower),
        _literal_id(term.upper),
    )


def per_value_realization_factors(
    term: StratumTerm,
    *,
    values: tuple[Any, ...],
    predicate_probabilities: Mapping[str, float],
) -> np.ndarray:
    """Per-encoded-id probability that a row with that value emits this term.

    These are the per-row factors whose mass-weighted mean is exactly the
    ``_expected_*`` vectors used during depth-1 discovery, written out per row
    so that a conjunction can average their *product* over the joint rows
    instead of multiplying marginal averages.  The predicate generator draws a
    lower bound uniformly among the ranks at or below the row's value and an
    upper bound uniformly among the ranks at or above it, which is where the
    ``1 / rank`` and ``1 / (size - rank + 1)`` factors come from.
    """

    family = term.realization_family or TERM_OPERATOR_FAMILY.get(term.region_type)
    if family is None:
        raise ValueError(f"unsupported region_type {term.region_type!r}")
    factors = np.zeros(len(values), dtype=float)
    if family == "equality" or term.region_type == "equality":
        matching = [index for index, value in enumerate(values) if term.contains_value(value)]
    else:
        matching = None
    if family == "equality":
        probability = float(predicate_probabilities.get("equality", 0.0))
        for index in matching or ():
            factors[index] = probability
        return factors

    comparable = sorted(
        (
            (float(value), index)
            for index, value in enumerate(values)
            if _is_plain_number(value) and not _is_structural_value(value)
        ),
    )
    size = len(comparable)
    if size == 0:
        return factors
    rank_of_index = {index: rank for rank, (_, index) in enumerate(comparable, start=1)}
    boundary = term.value if term.value is not None else (
        term.lower if term.lower is not None else term.upper
    )
    boundary_rank = None
    for rank, (value, index) in enumerate(comparable, start=1):
        if _is_plain_number(boundary) and value == float(boundary):
            boundary_rank = rank
            break
    if boundary_rank is None:
        raise ValueError(
            f"term on {term.column_name!r} has boundary {boundary!r} outside the column domain"
        )

    p_lower = float(predicate_probabilities.get("lower", 0.0))
    p_upper = float(predicate_probabilities.get("upper", 0.0))
    p_range = float(predicate_probabilities.get("range", 0.0))
    for index, rank in rank_of_index.items():
        if not term.contains_value(values[index]):
            continue
        if family == "lower":
            factors[index] = p_lower / float(rank)
        elif family == "upper":
            factors[index] = p_upper / float(size - rank + 1)
        elif family == "range":
            if term.region_type == "equality":
                factors[index] = p_range / float(rank * (size - rank + 1))
            elif term.region_type == "lower_tail":
                factors[index] = p_range * (1.0 - (boundary_rank - 1.0) / float(rank))
            elif term.region_type == "upper_tail":
                factors[index] = p_range * (
                    float(boundary_rank - rank + 1) / float(size - rank + 1)
                )
            else:
                factors[index] = p_range
        else:
            raise ValueError(f"unsupported realization family {family!r}")
    return factors


def _can_extend_any_seed(
    candidate: RootDataStratum,
    seeds: "tuple[RootDataStratum, ...]",
    *,
    metadata: ModelMetadata,
    require_distinct_tables: bool,
) -> bool:
    """Whether this candidate could become a term of at least one seed.

    The extension width is a budget, so it has to be spent on candidates the
    beam search can actually use.  A candidate on a column a seed already pins,
    or on a table a seed already covers when distinct tables are required, can
    never extend that seed; if it can extend none of them it is not a candidate
    at all.
    """

    column_index = int(candidate.column_index)
    table = metadata.columns[column_index].table
    for seed in seeds:
        if column_index in seed.column_indices:
            continue
        if require_distinct_tables and table in seed.tables(metadata):
            continue
        return True
    return False


def with_realization_probabilities(
    stratum: RootDataStratum,
    predicate_probabilities: Mapping[str, float],
) -> RootDataStratum:
    """Fill each term's rarest realization route and its probability.

    A depth-1 candidate carries ``expected_<family>_count`` for every operator
    route that can produce its region: the expected number of contexts that
    generate exactly this predicate through that route, threshold or interval
    selection already included.  The route with the smallest expected count is
    the bottleneck -- the same choice ``_with_score`` makes -- and it is the one
    that has to be recorded, because it decides both the support estimate and
    which predicate the auxiliary context forces.  Dividing by
    ``expected_target_rows`` turns the count back into a per-row probability.

    For an equality region realized through the equality route the ratio
    reduces to ``p_equality`` exactly, which is why such a term can also be
    scored without this step.
    """

    terms = []
    target_rows = float(stratum.expected_target_rows)
    for term in stratum.terms:
        if term.realization_probability is not None and term.realization_family:
            terms.append(term)
            continue
        own_family = TERM_OPERATOR_FAMILY.get(term.region_type)
        if own_family is None:
            raise ValueError(f"unsupported region_type {term.region_type!r}")
        routes: list[tuple[float, str]] = []
        for family in (own_family, "range"):
            expected = getattr(stratum, EXPECTED_SUPPORT_FIELD[family], None)
            if expected is None or not isfinite(float(expected)):
                continue
            if target_rows <= 0.0:
                continue
            routes.append((float(expected) / target_rows, family))
        if not routes:
            # Nothing to derive it from.  An equality region realized through
            # the equality route needs no derivation; anything else is left
            # unset so that scoring refuses it rather than silently falling
            # back to the bare operator probability.
            if own_family == "equality":
                terms.append(
                    StratumTerm(
                        **{
                            **asdict(term),
                            "realization_probability": float(
                                predicate_probabilities.get("equality", 0.0)
                            ),
                            "realization_family": "equality",
                        }
                    )
                )
            else:
                terms.append(term)
            continue
        probability, family = min(routes)
        chosen = probability
        terms.append(
            StratumTerm(
                **{
                    **asdict(term),
                    "realization_probability": probability,
                    "realization_family": family,
                }
            )
        )
    payload = {**stratum.to_json_dict(), "terms": tuple(terms)}
    if payload.get("realized_support_mass") is None and len(terms) == 1:
        # One term means there is no joint to average over, so the marginal
        # per-row probability times the region mass is already exact.
        factor = terms[0].realization_probability
        if factor is not None:
            payload["realized_support_mass"] = float(stratum.probability) * float(factor)
    return RootDataStratum(**payload)


def fixed_forced_predicate_is_supported(
    stratum: RootDataStratum,
    metadata: ModelMetadata,
) -> bool:
    """Whether every term has a production-safe, row-independent realization.

    Equality regions can always use a fixed singleton native range. Tail
    regions can do so only when exactly one comparable domain value belongs to
    the tail. A wider native-range tail would need a sampled-row-dependent
    second boundary, which the production generator deliberately rejects.
    """

    for term in stratum.terms:
        family = term.realization_family
        if family is None:
            if len(stratum.terms) == 1 and stratum.support_bottleneck == "native_range":
                family = "range"
            else:
                family = TERM_OPERATOR_FAMILY.get(term.region_type)
        if family != "range" or term.region_type not in {"lower_tail", "upper_tail"}:
            continue
        matching_values = 0
        for value in metadata.columns[term.column_index].domain:
            try:
                matches = term.contains_value(value)
            except (TypeError, ValueError):
                matches = False
            if matches:
                matching_values += 1
                if matching_values > 1:
                    return False
        if matching_values != 1:
            return False
    return True


def _term_realization(
    stratum: RootDataStratum,
    term: StratumTerm,
    predicate_probabilities: Mapping[str, float],
) -> float:
    family = TERM_OPERATOR_FAMILY.get(term.region_type)
    if family is None:
        raise ValueError(f"unsupported region_type {term.region_type!r}")
    if term.realization_probability is not None:
        return float(term.realization_probability)
    if family == "equality":
        return float(predicate_probabilities.get("equality", 0.0))
    raise ValueError(
        f"stratum {stratum.stratum_id!r} term on {term.column_name!r} is a "
        f"{term.region_type!r} region with no realization probability. Scoring it "
        "with the bare operator probability would ignore threshold selection and "
        "overestimate support; call with_realization_probabilities first."
    )


def conditional_head_support(
    stratum: RootDataStratum,
    head_index: int,
    *,
    metadata: ModelMetadata,
    n_total: int,
    predicate_probabilities: Mapping[str, float],
    root_table: str | None,
) -> float:
    """Expected contexts carrying the whole conjunction with ``head_index`` served.

    This is the quantity the marginal bottleneck missed: a stratum only helps a
    suffix head when every one of its terms is actually emitted as a predicate,
    every table it touches survives table dropout along with the head's own
    table, and the head sits strictly after the last term.
    """

    if head_index <= stratum.column_indices[-1]:
        return 0.0
    mass = stratum.realized_support_mass
    if mass is None:
        if stratum.is_conjunctive:
            raise ValueError(
                f"stratum {stratum.stratum_id!r} is conjunctive and has no realized "
                "support mass. Multiplying the terms' marginal realization "
                "probabilities would assume they are independent, which correlated "
                "terms are not; compute it with "
                "ExactRootStratumProvider.realized_support_mass."
            )
        # Exactly one term, so there is no joint to average over and the
        # region mass times the per-row probability is already exact.
        mass = float(stratum.probability) * _term_realization(
            stratum, stratum.terms[0], predicate_probabilities
        )
    support = float(n_total) * float(mass)
    required = set(stratum.tables(metadata))
    head_table = metadata.columns[head_index].table
    if head_table is not None:
        required.add(head_table)
    return support * table_subset_inclusion_probability(
        frozenset(required), metadata=metadata, root_table=root_table
    )


def auxiliary_eligibility_mask(
    metadata: ModelMetadata,
    selected_column_indices: np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    """Return [rare_rows, original_columns] mask for strict AR suffix heads.

    This lives next to ``served_head_indices`` on purpose: the two state the
    same contract, one per row at training time and one per stratum at
    selection time, and they must not drift apart.
    """

    selected = np.asarray(selected_column_indices, dtype=int).reshape(-1)
    if selected.shape != (row_count,):
        raise ValueError("selected_column_indices must have one entry per rare row")
    column_indices = np.arange(len(metadata.columns), dtype=int)
    return column_indices[None, :] > selected[:, None]


def served_head_indices(stratum: RootDataStratum, metadata: ModelMetadata) -> tuple[int, ...]:
    """Heads the stratum can condition, i.e. strictly after its last term.

    This must match ``auxiliary_eligibility_mask``, which trains every original
    column after the forced prefix, indicator and fanout heads included.  The
    fanout heads are the ones whose effective sample size motivated rare-support
    training in the first place, so scoring only the DATA heads would leave them
    out of the deficit that decides the allocation.
    """

    last = stratum.column_indices[-1]
    return tuple(
        index
        for index, column in enumerate(metadata.columns)
        if index > last
    )


def deficit_from_supports(
    supports: "Iterable[float]",
    *,
    threshold: float,
    objective: str = DEFAULT_DEFICIT_OBJECTIVE,
    epsilon: float = 1.0e-9,
) -> float:
    """Total repairable deficit over a stratum's served heads.

    ``clipped_sum`` is the original objective and saturates: a head whose
    support is already far below the threshold contributes almost exactly
    ``threshold`` no matter how much further below it is.  ``log_ratio``
    contributes ``log(threshold / support)``, which is still monotone in rarity
    across many orders of magnitude, so the ranking keeps measuring the thing
    it is supposed to measure once every candidate is rare.
    """

    if objective not in DEFICIT_OBJECTIVES:
        raise ValueError(
            f"unsupported deficit objective {objective!r}; expected one of "
            f"{list(DEFICIT_OBJECTIVES)}"
        )
    floor = max(float(epsilon), 0.0)
    total = 0.0
    for value in supports:
        support = float(value)
        if support >= threshold:
            continue
        if objective == "clipped_sum":
            total += threshold - support
        else:
            total += log(float(threshold) / max(support, floor))
    return float(total)


def with_conditional_score(
    stratum: RootDataStratum,
    *,
    metadata: ModelMetadata,
    n_total: int,
    predicate_probabilities: Mapping[str, float],
    threshold: float,
    root_table: str | None,
    objective: str = DEFAULT_DEFICIT_OBJECTIVE,
    epsilon: float = 1.0e-9,
) -> RootDataStratum:
    """Score by total repairable deficit across the heads the stratum serves.

    Summing over served heads is what makes allocation prefer a stratum that
    repairs five suffix heads over one that repairs a single head.

    The two objectives differ in what happens once a head's support falls far
    below the threshold.  ``clipped_sum`` contributes ``threshold - support``,
    which is capped at ``threshold``, so the sum saturates at
    ``heads * threshold`` and stops distinguishing a stratum from one a
    thousand times rarer.  In that regime -- which is every stratum this
    feature targets -- ranking collapses onto the served-head count, meaning
    onto the column's position in the schema, and a conjunction on a late
    column can never outrank one on an early column however rare it is.
    ``log_ratio`` contributes ``log(threshold / support)`` instead, which keeps
    growing as support shrinks, so rarity can outweigh a head or two.
    """

    # Fill the per-term realization probabilities here rather than relying on
    # every caller to remember: a tail term scored without one would silently
    # ignore threshold selection, which is the overestimate this guards.
    stratum = with_realization_probabilities(stratum, predicate_probabilities)
    heads = served_head_indices(stratum, metadata)
    supports = [
        conditional_head_support(
            stratum,
            head,
            metadata=metadata,
            n_total=n_total,
            predicate_probabilities=predicate_probabilities,
            root_table=root_table,
        )
        for head in heads
    ]
    deficient = [value for value in supports if value < threshold]
    bottleneck = min(supports) if supports else 0.0
    deficit = deficit_from_supports(
        supports, threshold=threshold, objective=objective, epsilon=epsilon
    )
    return RootDataStratum(
        **{
            **stratum.to_json_dict(),
            "support_score": float(bottleneck),
            "support_deficit": deficit,
            "support_bottleneck": (
                f"conditional_head_support:{len(deficient)}/{len(heads)}_heads"
            ),
        }
    )


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


def _capped_allocation(alpha: np.ndarray, cap: float) -> np.ndarray:
    """Limit any one stratum's share of the auxiliary batch.

    A conjunction can have a far larger deficit than every depth-1 stratum and
    would otherwise take almost the whole batch, which means resampling its few
    rows every step.  Water-fill instead: clip, redistribute, repeat.  A cap at
    or below ``1/n`` is unsatisfiable and is ignored.
    """

    if cap <= 0.0 or alpha.size == 0 or cap <= 1.0 / float(alpha.size):
        return alpha
    capped = alpha.astype(float).copy()
    for _ in range(64):
        excess = np.maximum(0.0, capped - cap)
        if float(excess.sum()) <= 1.0e-12:
            break
        capped = np.minimum(capped, cap)
        headroom = cap - capped
        room = float(headroom.sum())
        if room <= 1.0e-12:
            break
        capped = capped + headroom * (float(excess.sum()) / room)
    total = float(capped.sum())
    return capped / total if total > 0 else alpha


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
