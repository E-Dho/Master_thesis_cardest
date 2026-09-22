from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from .records import FilterPredicate, JoinPredicate, QueryRecord, TableRef


_JOIN_RE = re.compile(r"^\s*([^\s=<>!]+)\s*(=|<=|>=|<|>)\s*([^\s=<>!]+)\s*$")
_PREDICATE_RE = re.compile(r"^\s*([^\s=<>!]+)\s*(<=|>=|=|<|>)\s*(.+?)\s*$")
_COUNT_PREFIX = re.compile(r"^\s*SELECT\s+COUNT\s*\(\s*\*\s*\)\s+", re.IGNORECASE)
_SQL_RE = re.compile(
    r"^\s*SELECT\s+(?:COUNT\s*\(\s*\*\s*\)|1)\s+FROM\s+(.*?)"
    r"(?:\s+WHERE\s+(.*?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def load_workload(path: str | Path, workload_id: str) -> list[QueryRecord]:
    records = []
    for source_index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            records.append(parse_csv_query(line, workload_id, len(records), source_index))
    return records


def parse_csv_query(
    line: str,
    workload_id: str,
    query_id: int,
    source_index: int | None = None,
) -> QueryRecord:
    parts = line.strip().split("#")
    if len(parts) != 4:
        location = query_id if source_index is None else source_index
        raise ValueError(f"query line {location} must contain exactly three '#' separators")
    tables_part, joins_part, filters_part, truth_part = parts
    tables: list[TableRef] = []
    aliases: dict[str, str] = {}
    for item in tables_part.split(","):
        pieces = item.strip().split()
        if not pieces:
            continue
        name = pieces[0]
        alias = pieces[1] if len(pieces) > 1 else name
        if len(pieces) > 2:
            raise ValueError(f"invalid table reference {item!r}")
        if alias in aliases:
            raise ValueError(f"duplicate table alias {alias!r}")
        aliases[alias] = name
        tables.append(TableRef(name, alias))

    joins: list[JoinPredicate] = []
    for item in filter(None, (value.strip() for value in joins_part.split(","))):
        match = _JOIN_RE.match(item)
        if match is None:
            raise ValueError(f"invalid join predicate {item!r}")
        joins.append(JoinPredicate(*match.groups()))

    tokens = [value.strip() for value in filters_part.split(",") if value.strip()]
    if len(tokens) % 3:
        raise ValueError(f"filter token count must be divisible by three: {tokens!r}")
    filters = tuple(
        FilterPredicate(tokens[index], tokens[index + 1], parse_literal(tokens[index + 2]))
        for index in range(0, len(tokens), 3)
    )
    return QueryRecord(
        workload=workload_id,
        query_id=query_id,
        tables=tuple(tables),
        joins=tuple(joins),
        filters=filters,
        true_cardinality=int(truth_part),
        source_line=line.rstrip("\n"),
    )


def parse_literal(text: str) -> Any:
    value = text.strip()
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def query_to_sql(query: QueryRecord, *, count: bool = True) -> str:
    select = "SELECT COUNT(*)" if count else "SELECT 1"
    tables = ", ".join(
        table.name if table.alias == table.name else f"{table.name} {table.alias}"
        for table in query.tables
    )
    clauses = [f"{join.left}{join.operator}{join.right}" for join in query.joins]
    clauses.extend(
        f"{predicate.column}{predicate.operator}{sql_literal(predicate.value)}"
        for predicate in query.filters
    )
    where = "" if not clauses else " WHERE " + " AND ".join(clauses)
    return f"{select} FROM {tables}{where};"


def parse_sql_query(
    sql: str,
    workload_id: str,
    query_id: int,
    true_cardinality: int,
) -> QueryRecord:
    match = _SQL_RE.match(sql)
    if match is None:
        raise ValueError("unsupported canonical SQL query")
    tables: list[TableRef] = []
    for table_text in _split_outside_quotes(match.group(1), ","):
        pieces = table_text.strip().split()
        if len(pieces) not in {1, 2}:
            raise ValueError(f"invalid table reference {table_text!r}")
        tables.append(TableRef(pieces[0], pieces[-1]))
    joins: list[JoinPredicate] = []
    filters: list[FilterPredicate] = []
    where = match.group(2)
    for clause in [] if where is None else _split_sql_and(where):
        predicate = _PREDICATE_RE.match(clause)
        if predicate is None:
            raise ValueError(f"invalid SQL predicate {clause!r}")
        left, operator, right = predicate.groups()
        if _looks_like_identifier(right):
            joins.append(JoinPredicate(left, operator, right))
        else:
            filters.append(FilterPredicate(left, operator, parse_sql_literal(right)))
    return QueryRecord(
        workload=workload_id,
        query_id=query_id,
        tables=tuple(tables),
        joins=tuple(joins),
        filters=tuple(filters),
        true_cardinality=int(true_cardinality),
        source_line=sql.strip(),
    )


def rewrite_count_sql_for_explain(sql: str) -> str:
    rewritten, replacements = _COUNT_PREFIX.subn("SELECT 1 ", sql, count=1)
    if replacements != 1:
        raise ValueError("expected a SELECT COUNT(*) query")
    return rewritten


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def parse_sql_literal(text: str) -> Any:
    value = text.strip()
    if value.upper() == "NULL":
        return None
    if value.upper() in {"TRUE", "FALSE"}:
        return value.upper() == "TRUE"
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return parse_literal(value)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workload_statistics(records: list[QueryRecord]) -> dict[str, Any]:
    operator_counts: dict[str, int] = {}
    join_operator_counts: dict[str, int] = {}
    columns: set[str] = set()
    query_shape_counts = {"equality_only": 0, "one_sided_range": 0, "two_sided_range": 0}
    for record in records:
        for join in record.joins:
            join_operator_counts[join.operator] = join_operator_counts.get(join.operator, 0) + 1
        for predicate in record.filters:
            operator_counts[predicate.operator] = operator_counts.get(predicate.operator, 0) + 1
            columns.add(predicate.column)
        bounds: dict[str, set[str]] = {}
        for predicate in record.filters:
            if predicate.operator in {"<", "<="}:
                bounds.setdefault(predicate.column, set()).add("upper")
            elif predicate.operator in {">", ">="}:
                bounds.setdefault(predicate.column, set()).add("lower")
        if any(value == {"lower", "upper"} for value in bounds.values()):
            query_shape_counts["two_sided_range"] += 1
        elif bounds:
            query_shape_counts["one_sided_range"] += 1
        else:
            query_shape_counts["equality_only"] += 1
    truths = [record.true_cardinality for record in records]
    truth_percentiles = (
        {name: None for name in ("p50", "p90", "p95", "p99", "max")}
        if not truths
        else {
            "p50": float(np.percentile(truths, 50)),
            "p90": float(np.percentile(truths, 90)),
            "p95": float(np.percentile(truths, 95)),
            "p99": float(np.percentile(truths, 99)),
            "max": int(max(truths)),
        }
    )
    return {
        "query_count": len(records),
        "operator_counts": dict(sorted(operator_counts.items())),
        "join_operator_counts": dict(sorted(join_operator_counts.items())),
        "query_shape_counts": query_shape_counts,
        "predicate_columns": sorted(columns),
        "predicate_count": sum(len(record.filters) for record in records),
        "table_count_distribution": {
            str(size): sum(len(record.tables) == size for record in records)
            for size in sorted({len(record.tables) for record in records})
        },
        "true_cardinality_min": min(truths) if truths else None,
        "true_cardinality_max": max(truths) if truths else None,
        "true_cardinality_percentiles": truth_percentiles,
        "zero_true_cardinality_count": sum(value == 0 for value in truths),
    }


def _split_sql_and(value: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    index = 0
    quoted = False
    while index < len(value):
        if value[index] == "'":
            if quoted and index + 1 < len(value) and value[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
            index += 1
            continue
        if not quoted and value[index : index + 5].upper() == " AND ":
            pieces.append(value[start:index].strip())
            start = index + 5
            index = start
            continue
        index += 1
    pieces.append(value[start:].strip())
    return pieces


def _split_outside_quotes(value: str, separator: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    quoted = False
    index = 0
    while index < len(value):
        if value[index] == "'":
            if quoted and index + 1 < len(value) and value[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif not quoted and value[index] == separator:
            pieces.append(value[start:index])
            start = index + 1
        index += 1
    pieces.append(value[start:])
    return pieces


def _looks_like_identifier(value: str) -> bool:
    text = value.strip()
    return not text.startswith("'") and bool(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", text)
    )
