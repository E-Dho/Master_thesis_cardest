#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DIMENSIONS = ("standard", "temporal", "spatial", "spatio_temporal")
INTERVALS = ("range", "unbounded")
RELATIONS = ("single", "multi")
ORDERED_TYPES = {"numeric", "integer", "timestamp", "temporal_interval", "geometry"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Category:
    dimension: str
    interval: str
    relation: str

    @classmethod
    def parse(cls, value: str) -> "Category":
        parts = value.split(".")
        if len(parts) != 3:
            raise ConfigError(f"Invalid category {value!r}; expected dim.interval.relation")
        category = cls(*parts)
        category.validate()
        return category

    def validate(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ConfigError(f"Invalid dimension {self.dimension!r}")
        if self.interval not in INTERVALS:
            raise ConfigError(f"Invalid interval {self.interval!r}")
        if self.relation not in RELATIONS:
            raise ConfigError(f"Invalid relation {self.relation!r}")

    @property
    def key(self) -> str:
        return f"{self.dimension}.{self.interval}.{self.relation}"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_config(config: Dict[str, Any]) -> None:
    required = {"tables", "joins", "entity", "srid"}
    missing = required - set(config)
    if missing:
        raise ConfigError(f"Missing config keys: {sorted(missing)}")
    if not isinstance(config["tables"], dict) or not config["tables"]:
        raise ConfigError("Config must define at least one table")

    table_names = set(config["tables"])
    entity_table = config["entity"].get("table")
    if entity_table not in table_names:
        raise ConfigError(f"Entity table {entity_table!r} is not configured")

    for table_id, table in config["tables"].items():
        for key in ("name", "alias", "primary_key", "flags", "attributes"):
            if key not in table:
                raise ConfigError(f"Table {table_id!r} is missing {key!r}")
        flags = set(table["flags"])
        invalid_flags = flags - {"standard", "temporal", "spatial"}
        if invalid_flags:
            raise ConfigError(f"Table {table_id!r} has invalid flags {sorted(invalid_flags)}")
        if not isinstance(table["attributes"], list) or not table["attributes"]:
            raise ConfigError(f"Table {table_id!r} must define attributes")
        dimensions_present = {attr.get("dimension") for attr in table["attributes"]}
        for flag in flags:
            if flag not in dimensions_present:
                raise ConfigError(f"Table {table_id!r} has flag {flag!r} but no such attribute")
        for attr in table["attributes"]:
            validate_attribute(table_id, attr)

    if not isinstance(config["joins"], list):
        raise ConfigError("joins must be a list")
    graph = {table_id: set() for table_id in table_names}
    for join in config["joins"]:
        left = join.get("left")
        right = join.get("right")
        if left not in table_names or right not in table_names:
            raise ConfigError(f"Join references unknown tables: {join}")
        if not join.get("condition"):
            raise ConfigError(f"Join {join} is missing condition")
        graph[left].add(right)
        graph[right].add(left)

    if len(table_names) > 1:
        seen = connected_component(next(iter(table_names)), graph)
        if seen != table_names:
            raise ConfigError("Join graph is disconnected")


def validate_attribute(table_id: str, attr: Dict[str, Any]) -> None:
    for key in ("name", "type", "dimension"):
        if key not in attr:
            raise ConfigError(f"Attribute in {table_id!r} is missing {key!r}")
    if attr["dimension"] not in {"standard", "temporal", "spatial"}:
        raise ConfigError(f"Invalid dimension for {table_id}.{attr['name']}")
    attr_type = attr["type"]
    if attr_type in {"numeric", "integer"}:
        require_expression(attr, table_id)
        domain = attr.get("domain")
        if not domain or "min" not in domain or "max" not in domain:
            raise ConfigError(f"{table_id}.{attr['name']} needs numeric domain")
        if float(domain["max"]) <= float(domain["min"]):
            raise ConfigError(f"{table_id}.{attr['name']} has invalid numeric domain")
    elif attr_type == "nominal":
        require_expression(attr, table_id)
        if not attr.get("values"):
            raise ConfigError(f"{table_id}.{attr['name']} needs nominal values")
    elif attr_type == "temporal_interval":
        for key in ("start_expression", "end_expression", "domain"):
            if key not in attr:
                raise ConfigError(f"{table_id}.{attr['name']} needs {key}")
        domain = attr["domain"]
        if "min" not in domain or "max" not in domain:
            raise ConfigError(f"{table_id}.{attr['name']} needs temporal domain")
        parse_timestamp(domain["min"])
        parse_timestamp(domain["max"])
    elif attr_type == "geometry":
        require_expression(attr, table_id)
        domain = attr.get("domain")
        for key in ("min_x", "min_y", "max_x", "max_y"):
            if not domain or key not in domain:
                raise ConfigError(f"{table_id}.{attr['name']} needs spatial domain {key}")
        if float(domain["max_x"]) <= float(domain["min_x"]) or float(domain["max_y"]) <= float(domain["min_y"]):
            raise ConfigError(f"{table_id}.{attr['name']} has invalid spatial domain")
    else:
        raise ConfigError(f"Unsupported type for {table_id}.{attr['name']}: {attr_type}")


def require_expression(attr: Dict[str, Any], table_id: str) -> None:
    if not attr.get("expression"):
        raise ConfigError(f"{table_id}.{attr['name']} needs expression")


def connected_component(start: str, graph: Dict[str, set]) -> set:
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in graph[current]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ConfigError(f"Unsupported timestamp {value!r}")


def timestamp_literal(value: datetime) -> str:
    return "timestamp " + sql_literal(value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip("."))


def sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Invalid numeric literal {value}")
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def table_flags(config: Dict[str, Any], table_ids: Iterable[str]) -> set:
    flags = set()
    for table_id in table_ids:
        flags.update(config["tables"][table_id]["flags"])
    return flags


def required_flags(dimension: str) -> set:
    if dimension == "spatio_temporal":
        return {"spatial", "temporal"}
    return {dimension}


def attribute_dimensions(dimension: str) -> set:
    if dimension == "spatio_temporal":
        return {"spatial", "temporal"}
    return {dimension}


def attrs_for_dimension(config: Dict[str, Any], table_ids: Sequence[str], dimension: str) -> List[Tuple[str, Dict[str, Any]]]:
    dims = attribute_dimensions(dimension)
    attrs = []
    for table_id in table_ids:
        for attr in config["tables"][table_id]["attributes"]:
            if attr["dimension"] in dims:
                attrs.append((table_id, attr))
    return attrs


def is_ordered(attr: Dict[str, Any]) -> bool:
    return attr["type"] in ORDERED_TYPES


def has_required_dimension_mix(attrs: Sequence[Tuple[str, Dict[str, Any]]], dimension: str) -> bool:
    if dimension != "spatio_temporal":
        return bool(attrs)
    dims = {attr["dimension"] for _, attr in attrs}
    return {"spatial", "temporal"} <= dims


def graph_from_config(config: Dict[str, Any]) -> Dict[str, set]:
    graph = {table_id: set() for table_id in config["tables"]}
    for join in config["joins"]:
        graph[join["left"]].add(join["right"])
        graph[join["right"]].add(join["left"])
    return graph


def is_connected_subset(config: Dict[str, Any], subset: Sequence[str]) -> bool:
    graph = graph_from_config(config)
    subset_set = set(subset)
    start = subset[0]
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in graph[current] & subset_set:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == subset_set


def eligible_tables(config: Dict[str, Any], category: Category) -> List[Tuple[str, ...]]:
    table_ids = sorted(config["tables"])
    flags_needed = required_flags(category.dimension)
    eligible = []
    if category.relation == "single":
        for table_id in table_ids:
            if flags_needed <= set(config["tables"][table_id]["flags"]):
                attrs = attrs_for_dimension(config, [table_id], category.dimension)
                if has_required_dimension_mix(attrs, category.dimension):
                    eligible.append((table_id,))
        return eligible

    entity_table = config.get("entity", {}).get("table")
    for size in range(2, len(table_ids) + 1):
        for subset in itertools.combinations(table_ids, size):
            if entity_table and entity_table not in subset:
                continue
            if not is_connected_subset(config, subset):
                continue
            if not flags_needed <= table_flags(config, subset):
                continue
            attrs = attrs_for_dimension(config, subset, category.dimension)
            if has_required_dimension_mix(attrs, category.dimension):
                eligible.append(tuple(subset))
    return eligible


def all_valid_categories(config: Dict[str, Any]) -> List[Category]:
    categories = []
    for dimension in DIMENSIONS:
        for interval in INTERVALS:
            for relation in RELATIONS:
                category = Category(dimension, interval, relation)
                if eligible_tables(config, category):
                    categories.append(category)
    return categories


class QueryExecutor:
    def __init__(self, host=None, port=None, dbname=None, user=None):
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Executing queries requires psycopg; use --no-execute for SQL-only generation") from exc
        self._psycopg = psycopg
        self._kwargs = {k: v for k, v in {"host": host, "port": port, "dbname": dbname, "user": user}.items() if v is not None}
        self._conn = None

    def __enter__(self) -> "QueryExecutor":
        self._conn = self._psycopg.connect(**self._kwargs)
        self._conn.autocommit = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()

    def scalar(self, sql: str) -> Any:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
        return None if row is None else row[0]

    def row(self, sql: str) -> Optional[Tuple[Any, ...]]:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchone()

    def rows(self, sql: str) -> List[Tuple[Any, ...]]:
        assert self._conn is not None
        with self._conn.cursor() as cur:
            cur.execute(sql)
            return [tuple(row) for row in cur.fetchall()]


class LiveCenterCache:
    def __init__(self, config: Dict[str, Any], executor: Optional[QueryExecutor], sample_size: int):
        self.config = config
        self.executor = executor
        self.sample_size = sample_size
        self._cache: Dict[Tuple[str, str], List[Any]] = {}

    def values(self, table_id: str, attr: Dict[str, Any]) -> List[Any]:
        if self.executor is None or self.sample_size <= 0:
            return []
        key = (table_id, attr["name"])
        if key not in self._cache:
            self._cache[key] = self.fetch_values(table_id, attr)
        return self._cache[key]

    def fetch_values(self, table_id: str, attr: Dict[str, Any]) -> List[Any]:
        table = self.config["tables"][table_id]
        attr_type = attr["type"]
        if attr_type in {"numeric", "integer"}:
            select_sql = attr["expression"]
        elif attr_type == "temporal_interval":
            select_sql = f"{attr['start_expression']}, {attr['end_expression']}"
        elif attr_type == "geometry":
            point = f"ST_PointOnSurface({attr['expression']})"
            select_sql = f"ST_X({point}), ST_Y({point})"
        else:
            return []

        sql = f"SELECT {select_sql} FROM {table['name']} {table['alias']} ORDER BY random() LIMIT {int(self.sample_size)}"
        if attr_type in {"numeric", "integer"}:
            return self.fetch_column(sql)
        return self.fetch_rows(sql)

    def fetch_column(self, sql: str) -> List[Any]:
        assert self.executor is not None
        return [row[0] for row in self.executor.rows(sql) if row[0] is not None]

    def fetch_rows(self, sql: str) -> List[Tuple[Any, ...]]:
        assert self.executor is not None
        return [tuple(row) for row in self.executor.rows(sql) if all(value is not None for value in row)]


class QueryGenerator:
    def __init__(
        self,
        config: Dict[str, Any],
        seed: int,
        executor: Optional[QueryExecutor] = None,
        sample_cache_size: int = 2048,
        evaluate_cardinalities: bool = True,
    ):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.executor = executor
        self.live_centers = LiveCenterCache(config, executor, sample_cache_size)
        self.evaluate_cardinalities = evaluate_cardinalities
        self.hash = config_hash(config)
        self.generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def generate(self, categories: Sequence[Category], queries_per_category: int) -> List[Dict[str, Any]]:
        return list(self.iter_generate(categories, queries_per_category))

    def iter_generate(self, categories: Sequence[Category], queries_per_category: int) -> Iterable[Dict[str, Any]]:
        query_ordinal = 0
        for category in categories:
            for category_index in range(queries_per_category):
                query_ordinal += 1
                yield self.generate_one(category, query_ordinal, category_index)

    def generate_one(self, category: Category, query_ordinal: int, category_index: int) -> Dict[str, Any]:
        table_ids = self.choose_tables(category)
        selected_attrs = self.choose_attributes(table_ids, category)
        from_sql, joins = self.build_from_clause(table_ids)
        predicates = self.build_predicates(table_ids, selected_attrs, category)
        where_sql = " AND ".join(f"({p['sql']})" for p in predicates)
        sql = f"SELECT COUNT(*) AS join_cardinality\n{from_sql}\nWHERE {where_sql};"

        entity_sql = None
        if category.relation == "multi":
            entity_expr = self.entity_expression()
            entity_sql = f"SELECT COUNT(DISTINCT {entity_expr}) AS entity_cardinality\n{from_sql}\nWHERE {where_sql};"

        record = {
            "query_id": f"q{query_ordinal:08d}",
            "category": {
                "dimension": category.dimension,
                "interval": category.interval,
                "relation": category.relation,
                "key": category.key,
                "category_index": category_index,
            },
            "tables": list(table_ids),
            "joins": joins,
            "predicates": predicates,
            "sql": sql,
            "entity_sql": entity_sql,
            "center_source": summarize_sources(predicates, "center_source"),
            "range_source": summarize_sources(predicates, "range_source"),
            "seed": self.seed,
            "join_cardinality": None,
            "entity_cardinality": None,
            "config_hash": self.hash,
            "generated_at": self.generated_at,
        }
        if self.executor is not None and self.evaluate_cardinalities:
            record["join_cardinality"] = int(self.executor.scalar(sql))
            if entity_sql:
                record["entity_cardinality"] = int(self.executor.scalar(entity_sql))
        return record

    def choose_tables(self, category: Category) -> Tuple[str, ...]:
        candidates = eligible_tables(self.config, category)
        if not candidates:
            raise ConfigError(f"No eligible tables for {category.key}")
        if category.relation == "single":
            return self.rng.choice(candidates)

        sizes = sorted({len(candidate) for candidate in candidates})
        target_size = self.rng.randint(2, len(self.config["tables"]))
        size_candidates = [candidate for candidate in candidates if len(candidate) == target_size]
        if not size_candidates:
            nearest_size = min(sizes, key=lambda size: abs(size - target_size))
            size_candidates = [candidate for candidate in candidates if len(candidate) == nearest_size]
        return self.rng.choice(size_candidates)

    def choose_attributes(self, table_ids: Sequence[str], category: Category) -> List[Tuple[str, Dict[str, Any]]]:
        attrs = attrs_for_dimension(self.config, table_ids, category.dimension)
        if not attrs:
            raise ConfigError(f"No attributes available for {category.key} on {table_ids}")
        min_count = 2 if category.dimension == "spatio_temporal" else 1
        if len(attrs) < min_count:
            raise ConfigError(f"Not enough attributes for {category.key} on {table_ids}")

        for _ in range(200):
            count = self.rng.randint(min_count, len(attrs))
            selected = self.rng.sample(attrs, count)
            if not has_required_dimension_mix(selected, category.dimension):
                continue
            if category.interval == "unbounded" and not any(is_ordered(attr) for _, attr in selected):
                continue
            return selected
        raise ConfigError(f"Could not sample valid attributes for {category.key} on {table_ids}")

    def build_from_clause(self, table_ids: Sequence[str]) -> Tuple[str, List[Dict[str, str]]]:
        first = table_ids[0]
        first_table = self.config["tables"][first]
        from_sql = f"FROM {first_table['name']} {first_table['alias']}"
        if len(table_ids) == 1:
            return from_sql, []

        included = {first}
        remaining = set(table_ids[1:])
        joins_used = []
        while remaining:
            joined = False
            for join in self.config["joins"]:
                left = join["left"]
                right = join["right"]
                if left in included and right in remaining:
                    table = self.config["tables"][right]
                    from_sql += f"\nJOIN {table['name']} {table['alias']} ON {join['condition']}"
                    joins_used.append(join)
                    included.add(right)
                    remaining.remove(right)
                    joined = True
                    break
                if right in included and left in remaining:
                    table = self.config["tables"][left]
                    from_sql += f"\nJOIN {table['name']} {table['alias']} ON {join['condition']}"
                    joins_used.append(join)
                    included.add(left)
                    remaining.remove(left)
                    joined = True
                    break
            if not joined:
                raise ConfigError(f"Could not connect tables {table_ids}")
        return from_sql, joins_used

    def build_predicates(
        self,
        table_ids: Sequence[str],
        selected_attrs: Sequence[Tuple[str, Dict[str, Any]]],
        category: Category,
    ) -> List[Dict[str, Any]]:
        force_unbounded = set()
        if category.interval == "unbounded":
            ordered_positions = [idx for idx, (_, attr) in enumerate(selected_attrs) if is_ordered(attr)]
            if not ordered_positions:
                raise ConfigError("Unbounded query needs at least one ordered attribute")
            if self.rng.random() < 0.5:
                force_unbounded = set(ordered_positions)
            else:
                first = self.rng.choice(ordered_positions)
                force_unbounded.add(first)
                for idx in ordered_positions:
                    if idx != first and self.rng.random() < 0.5:
                        force_unbounded.add(idx)

        predicates = []
        for idx, (table_id, attr) in enumerate(selected_attrs):
            if category.interval == "unbounded" and idx in force_unbounded and is_ordered(attr):
                predicates.append(self.unbounded_predicate(table_id, attr))
            else:
                predicates.append(self.range_predicate(table_id, attr))
        return predicates

    def range_predicate(self, table_id: str, attr: Dict[str, Any]) -> Dict[str, Any]:
        attr_type = attr["type"]
        if attr_type == "nominal":
            value = self.rng.choice(attr["values"])
            return self.predicate_record(table_id, attr, f"{attr['expression']} = {sql_literal(value)}", "nominal_eq", value=value)
        if attr_type in {"numeric", "integer"}:
            center, center_source = self.scalar_center(table_id, attr)
            width, range_source = self.scalar_width(attr)
            lower = max(float(attr["domain"]["min"]), center - width / 2.0)
            upper = min(float(attr["domain"]["max"]), center + width / 2.0)
            if attr_type == "integer":
                lower = math.floor(lower)
                upper = math.ceil(upper)
            sql = f"{attr['expression']} >= {sql_literal(lower)} AND {attr['expression']} <= {sql_literal(upper)}"
            return self.predicate_record(table_id, attr, sql, "range", lower=lower, upper=upper, center_source=center_source, range_source=range_source)
        if attr_type == "temporal_interval":
            center, center_source = self.temporal_center(table_id, attr)
            width_seconds, range_source = self.temporal_width(attr)
            lower = max(parse_timestamp(attr["domain"]["min"]), center - timedelta(seconds=width_seconds / 2.0))
            upper = min(parse_timestamp(attr["domain"]["max"]), center + timedelta(seconds=width_seconds / 2.0))
            sql = f"{attr['start_expression']} < {timestamp_literal(upper)} AND {attr['end_expression']} >= {timestamp_literal(lower)}"
            return self.predicate_record(table_id, attr, sql, "temporal_overlap", lower=lower.isoformat(sep=" "), upper=upper.isoformat(sep=" "), center_source=center_source, range_source=range_source)
        if attr_type == "geometry":
            center, center_source = self.spatial_center(table_id, attr)
            width, height, range_source = self.spatial_width(attr)
            domain = attr["domain"]
            min_x = max(float(domain["min_x"]), center[0] - width / 2.0)
            max_x = min(float(domain["max_x"]), center[0] + width / 2.0)
            min_y = max(float(domain["min_y"]), center[1] - height / 2.0)
            max_y = min(float(domain["max_y"]), center[1] + height / 2.0)
            sql = self.spatial_intersects_sql(attr, min_x, min_y, max_x, max_y)
            return self.predicate_record(table_id, attr, sql, "spatial_intersects", min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, center_source=center_source, range_source=range_source)
        raise ConfigError(f"Unsupported range predicate for {attr_type}")

    def unbounded_predicate(self, table_id: str, attr: Dict[str, Any]) -> Dict[str, Any]:
        attr_type = attr["type"]
        if attr_type in {"numeric", "integer"}:
            center, center_source = self.scalar_center(table_id, attr)
            op = self.rng.choice(("<", "<=", ">", ">="))
            value = math.floor(center) if attr_type == "integer" else center
            sql = f"{attr['expression']} {op} {sql_literal(value)}"
            return self.predicate_record(table_id, attr, sql, "unbounded", operator=op, value=value, center_source=center_source)
        if attr_type == "temporal_interval":
            center, center_source = self.temporal_center(table_id, attr)
            if self.rng.random() < 0.5:
                op = self.rng.choice(("<", "<="))
                sql = f"{attr['start_expression']} {op} {timestamp_literal(center)}"
            else:
                op = self.rng.choice((">", ">="))
                sql = f"{attr['end_expression']} {op} {timestamp_literal(center)}"
            return self.predicate_record(table_id, attr, sql, "temporal_unbounded", operator=op, value=center.isoformat(sep=" "), center_source=center_source)
        if attr_type == "geometry":
            domain = attr["domain"]
            min_x = float(domain["min_x"])
            max_x = float(domain["max_x"])
            min_y = float(domain["min_y"])
            max_y = float(domain["max_y"])
            center, center_source = self.spatial_center(table_id, attr)
            direction = self.rng.choice(("x_lt", "x_gt", "y_lt", "y_gt"))
            if direction == "x_lt":
                max_x = center[0]
            elif direction == "x_gt":
                min_x = center[0]
            elif direction == "y_lt":
                max_y = center[1]
            else:
                min_y = center[1]
            sql = self.spatial_intersects_sql(attr, min_x, min_y, max_x, max_y)
            return self.predicate_record(table_id, attr, sql, "spatial_unbounded", operator=direction, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, center_source=center_source)
        if attr_type == "nominal":
            return self.range_predicate(table_id, attr)
        raise ConfigError(f"Unsupported unbounded predicate for {attr_type}")

    def predicate_record(self, table_id: str, attr: Dict[str, Any], sql: str, mode: str, **metadata: Any) -> Dict[str, Any]:
        record = {
            "table": table_id,
            "attribute": attr["name"],
            "dimension": attr["dimension"],
            "type": attr["type"],
            "mode": mode,
            "sql": sql,
        }
        record.update(metadata)
        record.setdefault("center_source", None)
        record.setdefault("range_source", None)
        return record

    def scalar_center(self, table_id: str, attr: Dict[str, Any]) -> Tuple[float, str]:
        if self.executor is not None and self.rng.random() < 0.9:
            values = self.live_centers.values(table_id, attr)
            value = self.rng.choice(values) if values else None
            if value is not None:
                return float(value), "live_row"
        domain = attr["domain"]
        return self.rng.uniform(float(domain["min"]), float(domain["max"])), "domain"

    def scalar_width(self, attr: Dict[str, Any]) -> Tuple[float, str]:
        domain = attr["domain"]
        size = float(domain["max"]) - float(domain["min"])
        return self.sample_width(size)

    def temporal_center(self, table_id: str, attr: Dict[str, Any]) -> Tuple[datetime, str]:
        if self.executor is not None and self.rng.random() < 0.9:
            values = self.live_centers.values(table_id, attr)
            row = self.rng.choice(values) if values else None
            if row is not None and row[0] is not None and row[1] is not None:
                start = parse_timestamp(row[0])
                end = parse_timestamp(row[1])
                return start + (end - start) / 2, "live_row"
        lower = parse_timestamp(attr["domain"]["min"])
        upper = parse_timestamp(attr["domain"]["max"])
        seconds = self.rng.uniform(0, (upper - lower).total_seconds())
        return lower + timedelta(seconds=seconds), "domain"

    def temporal_width(self, attr: Dict[str, Any]) -> Tuple[float, str]:
        lower = parse_timestamp(attr["domain"]["min"])
        upper = parse_timestamp(attr["domain"]["max"])
        return self.sample_width((upper - lower).total_seconds())

    def spatial_center(self, table_id: str, attr: Dict[str, Any]) -> Tuple[Tuple[float, float], str]:
        if self.executor is not None and self.rng.random() < 0.9:
            values = self.live_centers.values(table_id, attr)
            row = self.rng.choice(values) if values else None
            if row is not None and row[0] is not None and row[1] is not None:
                return (float(row[0]), float(row[1])), "live_row"
        domain = attr["domain"]
        return (
            self.rng.uniform(float(domain["min_x"]), float(domain["max_x"])),
            self.rng.uniform(float(domain["min_y"]), float(domain["max_y"])),
        ), "domain"

    def spatial_width(self, attr: Dict[str, Any]) -> Tuple[float, float, str]:
        domain = attr["domain"]
        width, source_x = self.sample_width(float(domain["max_x"]) - float(domain["min_x"]))
        height, source_y = self.sample_width(float(domain["max_y"]) - float(domain["min_y"]))
        source = source_x if source_x == source_y else "mixed"
        return width, height, source

    def sample_width(self, size: float) -> Tuple[float, str]:
        if size <= 0:
            raise ConfigError("Domain size must be positive")
        if self.rng.random() < 0.5:
            return max(size * 1e-9, self.rng.uniform(0, size)), "uniform"
        lambd = 10.0 / size
        return min(size, max(size * 1e-9, self.rng.expovariate(lambd))), "exponential"

    def spatial_intersects_sql(self, attr: Dict[str, Any], min_x: float, min_y: float, max_x: float, max_y: float) -> str:
        srid = int(attr.get("srid", self.config["srid"]))
        return (
            f"ST_Intersects({attr['expression']}, "
            f"ST_MakeEnvelope({repr(min_x)}, {repr(min_y)}, {repr(max_x)}, {repr(max_y)}, {srid}))"
        )

    def entity_expression(self) -> str:
        entity = self.config["entity"]
        if "expression" in entity:
            return entity["expression"]
        table = self.config["tables"][entity["table"]]
        return f"{table['alias']}.{entity['key']}"


def summarize_sources(predicates: Sequence[Dict[str, Any]], field: str) -> Optional[str]:
    values = {predicate.get(field) for predicate in predicates if predicate.get(field) is not None}
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def stream_jsonl(path: Path, records: Iterable[Dict[str, Any]], progress_interval: int = 100) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                print(f"generated {count} queries", file=sys.stderr, flush=True)
    return count


def parse_categories(config: Dict[str, Any], category_arg: Optional[str]) -> List[Category]:
    if not category_arg:
        return all_valid_categories(config)
    categories = [Category.parse(value.strip()) for value in category_arg.split(",") if value.strip()]
    for category in categories:
        if not eligible_tables(config, category):
            raise ConfigError(f"Requested category has no eligible tables: {category.key}")
    return categories


def should_execute(args: argparse.Namespace) -> bool:
    if args.execute is not None:
        return args.execute
    return any(value is not None for value in (args.host, args.port, args.dbname, args.user))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate MobilityDB/PostgreSQL COUNT query workloads.")
    parser.add_argument("--config", required=True, help="Path to query generator JSON config.")
    parser.add_argument("--output", required=True, help="Path to output JSONL manifest.")
    parser.add_argument("--queries-per-category", type=int, required=True)
    parser.add_argument("--categories", help="Comma-separated dim.interval.relation categories. Omit for all valid categories.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sample-cache-size", type=int, default=2048, help="Live-row centers cached per ordered attribute in execute mode; use 0 to disable live center caching.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Write a progress line every N generated queries; use 0 to disable.")
    parser.add_argument("--skip-cardinality-execution", action="store_true", help="Connect for live center sampling but leave cardinality fields null.")
    execute_group = parser.add_mutually_exclusive_group()
    execute_group.add_argument("--execute", dest="execute", action="store_true", default=None)
    execute_group.add_argument("--no-execute", dest="execute", action="store_false")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.queries_per_category <= 0:
        raise SystemExit("--queries-per-category must be positive")

    config = load_config(Path(args.config))
    categories = parse_categories(config, args.categories)
    execute = should_execute(args)

    if execute:
        with QueryExecutor(host=args.host, port=args.port, dbname=args.dbname, user=args.user) as executor:
            generator = QueryGenerator(
                config,
                args.seed,
                executor,
                sample_cache_size=args.sample_cache_size,
                evaluate_cardinalities=not args.skip_cardinality_execution,
            )
            written = stream_jsonl(Path(args.output), generator.iter_generate(categories, args.queries_per_category), args.progress_interval)
    else:
        generator = QueryGenerator(config, args.seed, None, sample_cache_size=args.sample_cache_size)
        written = stream_jsonl(Path(args.output), generator.iter_generate(categories, args.queries_per_category), args.progress_interval)

    print(f"wrote {written} queries to {args.output}")


if __name__ == "__main__":
    main()
