"""Adapted DeepDB JOB-light-ranges schema and order-preserving literal encoding.

The native DeepDB ``imdb-light`` schema deliberately marks six columns that
JOB-light-ranges filters on as irrelevant.  This module defines the explicitly
non-native, project-owned variant that models them:

* the six JOB-light tables and title-centred relationships are unchanged;
* ``title.imdb_index``, ``title.phonetic_code``, ``title.season_nr``,
  ``title.episode_nr``, ``title.series_years`` and ``cast_info.nr_order``
  become modeled attributes;
* the three string columns are replaced by dense, order-preserving integer
  ranks, because DeepDB's SPN leaves only support ``<, <=, >, >=`` on
  numeric columns.  The order is computed by PostgreSQL under the collation
  that reproduces the workload labels (byte-order ``C`` for the published
  JOB-light-ranges labels).

Everything here is deliberately free of DeepDB imports (except the lazily
imported upstream graph classes) so it runs in the legacy Python 3.8 DeepDB
environment and in the modern evaluation environment alike.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .records import FilterPredicate, QueryRecord


VARIANT_ID = "imdb_light_ranges_adapted"
DISPLAY_NAME = "DeepDB JOB-light-ranges adapted"
DOMAIN_FILE = "rank_domains.json"
DOMAIN_SCHEMA_VERSION = 1

TABLES = (
    "title", "movie_info_idx", "movie_info", "cast_info",
    "movie_keyword", "movie_companies",
)

# Upstream gen_job_light_imdb_schema() at DeepDB 655ada13, reproduced so the
# difference to the adapted schema is explicit, testable, and documented.
NATIVE_TABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "title": {
        "attributes": [
            "id", "title", "imdb_index", "kind_id", "production_year", "imdb_id",
            "phonetic_code", "episode_of_id", "season_nr", "episode_nr",
            "series_years", "md5sum",
        ],
        "irrelevant_attributes": [
            "episode_of_id", "title", "imdb_index", "phonetic_code", "season_nr",
            "imdb_id", "episode_nr", "series_years", "md5sum",
        ],
        "no_compression": ["kind_id"],
        "table_size": 3486660,
    },
    "movie_info_idx": {
        "attributes": ["id", "movie_id", "info_type_id", "info", "note"],
        "irrelevant_attributes": ["info", "note"],
        "no_compression": ["info_type_id"],
        "table_size": 3147110,
    },
    "movie_info": {
        "attributes": ["id", "movie_id", "info_type_id", "info", "note"],
        "irrelevant_attributes": ["info", "note"],
        "no_compression": ["info_type_id"],
        "table_size": 24988000,
    },
    "cast_info": {
        "attributes": [
            "id", "person_id", "movie_id", "person_role_id", "note", "nr_order",
            "role_id",
        ],
        "irrelevant_attributes": ["nr_order", "note", "person_id", "person_role_id"],
        "no_compression": ["role_id"],
        "table_size": 63475800,
    },
    "movie_keyword": {
        "attributes": ["id", "movie_id", "keyword_id"],
        "irrelevant_attributes": [],
        "no_compression": ["keyword_id"],
        "table_size": 7522600,
    },
    "movie_companies": {
        "attributes": ["id", "movie_id", "company_id", "company_type_id", "note"],
        "irrelevant_attributes": ["note"],
        "no_compression": ["company_id", "company_type_id"],
        "table_size": 4958300,
    },
}
RELATIONSHIPS = (
    ("movie_info_idx", "movie_id", "title", "id"),
    ("movie_info", "movie_id", "title", "id"),
    ("cast_info", "movie_id", "title", "id"),
    ("movie_keyword", "movie_id", "title", "id"),
    ("movie_companies", "movie_id", "title", "id"),
)

# Columns the native schema excludes but JOB-light-ranges filters on.
ADDED_MODELED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "title": ("imdb_index", "phonetic_code", "season_nr", "episode_nr", "series_years"),
    "cast_info": ("nr_order",),
}
RANKED_STRING_COLUMNS: Tuple[str, ...] = (
    "title.imdb_index",
    "title.phonetic_code",
    "title.series_years",
)
# DeepDB replaces NULL by a per-column sentinel (mean + 1e-4) and subtracts
# the sentinel mass whenever a range contains it.  Histogram compression of
# leaves with > 10,000 distinct values would erase that sentinel and count
# NULL rows inside ranges, contradicting SQL.  The added columns are therefore
# kept uncompressed so their NULL semantics stay exact.
ADDED_NO_COMPRESSION: Dict[str, Tuple[str, ...]] = dict(ADDED_MODELED_COLUMNS)
RANGE_OPERATORS = ("=", "<", "<=", ">", ">=")

# Collations whose order equals Unicode code-point order (i.e. UTF-8 byte
# order).  Even for these, python-side comparisons are only enabled after the
# PostgreSQL-ordered domain was verified to equal the code-point sort.
CODEPOINT_COLLATIONS = frozenset({"C", "POSIX", "C.UTF-8", "C.utf8", "ucs_basic", "pg_c_utf8"})


def adapted_table_specs() -> Dict[str, Dict[str, Any]]:
    """Return the adapted table specifications derived from the native ones."""
    specs: Dict[str, Dict[str, Any]] = {}
    for table, native in NATIVE_TABLE_SPECS.items():
        added = ADDED_MODELED_COLUMNS.get(table, ())
        spec = {key: list(value) if isinstance(value, list) else value for key, value in native.items()}
        spec["irrelevant_attributes"] = [
            column for column in native["irrelevant_attributes"] if column not in added
        ]
        spec["no_compression"] = list(native["no_compression"]) + [
            column for column in ADDED_NO_COMPRESSION.get(table, ())
            if column not in native["no_compression"]
        ]
        specs[table] = spec
    return specs


def gen_job_light_ranges_adapted_schema(csv_path: str, graph_module: Any = None):
    """Project-owned counterpart of upstream ``gen_job_light_imdb_schema``.

    Same six tables, same attribute order (the CSV layout), same five
    ``child.movie_id = title.id`` relationships; only the irrelevant and
    no-compression attribute sets differ, see ``adapted_table_specs``.
    """
    if graph_module is None:
        from ensemble_compilation import graph_representation as graph_module
    schema = graph_module.SchemaGraph()
    for table, spec in adapted_table_specs().items():
        schema.add_table(
            graph_module.Table(
                table,
                attributes=list(spec["attributes"]),
                irrelevant_attributes=list(spec["irrelevant_attributes"]),
                no_compression=list(spec["no_compression"]),
                csv_file_location=csv_path.format(table),
                table_size=spec["table_size"],
            )
        )
    for start, start_attr, end, end_attr in RELATIONSHIPS:
        schema.add_relationship(start, start_attr, end, end_attr)
    return schema


def schema_description() -> Dict[str, Any]:
    specs = adapted_table_specs()
    return {
        "variant_id": VARIANT_ID,
        "native_base": "deepdb schemas.imdb.schema.gen_job_light_imdb_schema",
        "tables": list(TABLES),
        "relationships": [f"{a}.{b} = {c}.{d}" for a, b, c, d in RELATIONSHIPS],
        "added_modeled_columns": {
            table: list(columns) for table, columns in ADDED_MODELED_COLUMNS.items()
        },
        "ranked_string_columns": list(RANKED_STRING_COLUMNS),
        "modeled_attributes": {
            table: [a for a in spec["attributes"] if a not in spec["irrelevant_attributes"]]
            for table, spec in specs.items()
        },
        "no_compression": {table: spec["no_compression"] for table, spec in specs.items()},
    }


def modeled_columns() -> frozenset:
    return frozenset(
        f"{table}.{attribute}"
        for table, spec in adapted_table_specs().items()
        for attribute in spec["attributes"]
        if attribute not in spec["irrelevant_attributes"]
    )


@dataclass(frozen=True)
class RankDomain:
    """Complete non-NULL domain of one string column in collation order.

    ``values[rank]`` is the value with that rank.  ``boundaries`` maps
    literals to ``(bisect_left, bisect_right)`` as computed by PostgreSQL, i.e.
    ``(#values < literal, #values <= literal)`` under the column collation.
    Python-side bisection is only allowed when ``codepoint_order_verified``.
    """

    column: str
    values: Tuple[str, ...]
    collation: str
    boundaries: Mapping[str, Tuple[int, int]] = field(default_factory=dict)
    codepoint_order_verified: bool = False
    _ranks: Dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        ranks = {value: index for index, value in enumerate(self.values)}
        if len(ranks) != len(self.values):
            raise ValueError(f"{self.column} domain contains duplicate values")
        object.__setattr__(self, "_ranks", ranks)

    def __len__(self) -> int:
        return len(self.values)

    def rank(self, literal: str) -> Optional[int]:
        return self._ranks.get(literal)

    def bisect_left(self, literal: str) -> int:
        return self._bounds(literal)[0]

    def bisect_right(self, literal: str) -> int:
        return self._bounds(literal)[1]

    def _bounds(self, literal: str) -> Tuple[int, int]:
        rank = self._ranks.get(literal)
        if rank is not None:
            return rank, rank + 1
        if literal in self.boundaries:
            left, right = self.boundaries[literal]
            return int(left), int(right)
        if self.codepoint_order_verified:
            return (
                bisect.bisect_left(self.values, literal),
                bisect.bisect_right(self.values, literal),
            )
        raise KeyError(
            f"no {self.collation!r} collation boundary is recorded for literal "
            f"{literal!r} of {self.column}; rerun domain preparation with this workload"
        )


class UnsupportedLiteral(ValueError):
    """A literal cannot be placed in the collation order of its domain."""


@dataclass(frozen=True)
class RewrittenPredicate:
    column: str
    operator: str
    literal: Any
    rank_operator: str
    rank_value: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "operator": self.operator,
            "literal": self.literal,
            "rank_operator": self.rank_operator,
            "rank_value": self.rank_value,
        }


@dataclass(frozen=True)
class RewriteResult:
    query: QueryRecord
    predicates: Tuple[RewrittenPredicate, ...]
    empty_reason: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.empty_reason is not None


def rank_predicate(domain: RankDomain, operator: str, literal: str) -> Tuple[str, Optional[int]]:
    """Translate one SQL string predicate into a rank predicate.

    ``=``  -> ``rank = exact rank`` (``None`` if the literal is absent)
    ``>=`` -> ``rank >= bisect_left``    ``>`` -> ``rank >= bisect_right``
    ``<=`` -> ``rank <  bisect_right``   ``<`` -> ``rank <  bisect_left``

    Lower bounds therefore use the left insertion point of an inclusive bound,
    upper bounds the right insertion point.  NULL has no rank, so every rank
    predicate is false for NULL exactly as the SQL predicate is unknown.
    """
    try:
        if operator == "=":
            return "=", domain.rank(literal)
        if operator == ">=":
            return ">=", domain.bisect_left(literal)
        if operator == ">":
            return ">=", domain.bisect_right(literal)
        if operator == "<=":
            return "<", domain.bisect_right(literal)
        if operator == "<":
            return "<", domain.bisect_left(literal)
    except KeyError as exc:
        raise UnsupportedLiteral(str(exc)) from exc
    raise ValueError(f"unsupported operator {operator!r} for ranked column {domain.column}")


def raw_filter_literals(query: QueryRecord) -> List[Any]:
    """Recover the textual literals of CSV workloads.

    ``parse_literal`` converts numeric-looking tokens to numbers; string
    columns must be compared as text, so the untouched token is recovered.
    """
    parts = query.source_line.strip().split("#")
    if len(parts) == 4:
        tokens = [value.strip() for value in parts[2].split(",") if value.strip()]
        if len(tokens) == 3 * len(query.filters):
            return [tokens[3 * index + 2].strip("'\"") for index in range(len(query.filters))]
    return [predicate.value for predicate in query.filters]


def rewrite_query(query: QueryRecord, domains: Mapping[str, RankDomain]) -> RewriteResult:
    """Rewrite ranked-string predicates of ``query`` into rank predicates.

    Returns an explicit empty result when no non-NULL domain value can satisfy
    the conjunction on a ranked column: a missing equality literal, a range
    outside the domain, or contradictory bounds.  Such predicates select no
    tuple in SQL, so the exact cardinality is zero.
    """
    aliases = {table.alias: table.name for table in query.tables}
    raw_literals = raw_filter_literals(query)
    filters: List[FilterPredicate] = []
    rewritten: List[RewrittenPredicate] = []
    per_column: Dict[str, List[Tuple[str, Optional[int]]]] = {}
    empty_reasons: List[str] = []
    for predicate, raw in zip(query.filters, raw_literals):
        alias, _, attribute = predicate.column.partition(".")
        table = aliases.get(alias, alias)
        qualified = f"{table}.{attribute}"
        domain = domains.get(qualified)
        if domain is None:
            filters.append(predicate)
            continue
        if raw is None:
            raise UnsupportedLiteral(f"NULL literal for {qualified}")
        literal = str(raw)
        rank_operator, rank_value = rank_predicate(domain, predicate.operator, literal)
        if rank_value is None:
            empty_reasons.append(
                f"missing_equality_literal:{qualified}={literal!r}"
            )
            continue
        per_column.setdefault(qualified, []).append((rank_operator, rank_value))
        rewritten.append(
            RewrittenPredicate(qualified, predicate.operator, literal, rank_operator, rank_value)
        )
        filters.append(FilterPredicate(predicate.column, rank_operator, rank_value))
    for column, bounds in per_column.items():
        lower, upper = 0, len(domains[column])
        equal = set()
        for operator, value in bounds:
            if operator == ">=":
                lower = max(lower, value)
            elif operator == "<":
                upper = min(upper, value)
            else:
                equal.add(value)
        if len(equal) > 1:
            lower, upper = 1, 0
        elif equal:
            (value,) = tuple(equal)
            lower, upper = max(lower, value), min(upper, value + 1)
        if lower >= upper:
            empty_reasons.append(f"empty_rank_interval:{column}:[{lower},{upper})")
    return RewriteResult(
        query=replace(query, filters=tuple(filters)),
        predicates=tuple(rewritten),
        empty_reason=None if not empty_reasons else ";".join(empty_reasons),
    )


def unsupported_filter_columns(query: QueryRecord) -> Tuple[str, ...]:
    aliases = {table.alias: table.name for table in query.tables}
    columns = modeled_columns()
    unsupported = set()
    for predicate in query.filters:
        alias, _, attribute = predicate.column.partition(".")
        table = aliases.get(alias)
        if table is None or f"{table}.{attribute}" not in columns:
            unsupported.add(predicate.column)
    return tuple(sorted(unsupported))


def string_literals_by_column(queries: Iterable[QueryRecord]) -> Dict[str, List[str]]:
    literals: Dict[str, set] = {column: set() for column in RANKED_STRING_COLUMNS}
    for query in queries:
        aliases = {table.alias: table.name for table in query.tables}
        for predicate, raw in zip(query.filters, raw_filter_literals(query)):
            alias, _, attribute = predicate.column.partition(".")
            qualified = f"{aliases.get(alias, alias)}.{attribute}"
            if qualified in literals and raw is not None:
                literals[qualified].add(str(raw))
    return {column: sorted(values) for column, values in literals.items()}


def domains_to_payload(
    domains: Mapping[str, RankDomain], metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "variant_id": VARIANT_ID,
        **dict(metadata),
        "columns": {
            column: {
                "collation": domain.collation,
                "codepoint_order_verified": domain.codepoint_order_verified,
                "distinct_count": len(domain.values),
                "values": list(domain.values),
                "literal_boundaries": {
                    literal: [int(bounds[0]), int(bounds[1])]
                    for literal, bounds in sorted(domain.boundaries.items())
                },
            }
            for column, domain in sorted(domains.items())
        },
    }


def domains_from_payload(payload: Mapping[str, Any]) -> Dict[str, RankDomain]:
    if int(payload.get("schema_version", 0)) != DOMAIN_SCHEMA_VERSION:
        raise ValueError("unsupported rank-domain schema version")
    return {
        column: RankDomain(
            column=column,
            values=tuple(entry["values"]),
            collation=str(entry["collation"]),
            boundaries={
                literal: (int(bounds[0]), int(bounds[1]))
                for literal, bounds in entry.get("literal_boundaries", {}).items()
            },
            codepoint_order_verified=bool(entry.get("codepoint_order_verified", False)),
        )
        for column, entry in payload["columns"].items()
    }


def write_domains(path: Path, domains: Mapping[str, RankDomain], metadata: Mapping[str, Any]) -> str:
    payload = domains_to_payload(domains, metadata)
    text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_domains(path: Path) -> Dict[str, RankDomain]:
    return domains_from_payload(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_rank_predicates(
    ranks: Sequence[Optional[float]], predicates: Sequence[Tuple[str, int]]
) -> List[bool]:
    """Reference evaluator used by tests: NULL (None/NaN) never matches."""
    result = []
    for value in ranks:
        if value is None or value != value:
            result.append(False)
            continue
        matched = True
        for operator, bound in predicates:
            if operator == "=":
                matched &= value == bound
            elif operator == ">=":
                matched &= value >= bound
            elif operator == "<":
                matched &= value < bound
            elif operator == "<=":
                matched &= value <= bound
            elif operator == ">":
                matched &= value > bound
            else:
                raise ValueError(operator)
        result.append(bool(matched))
    return result
