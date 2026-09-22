#!/usr/bin/env python3
"""Build the deterministic, exactly labelled MSCN JOB-light-ranges corpus."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pickle
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from evaluation.job_light_imdb_non_trajectory.joblight_eval.mscn_ranges import (
    assert_test_disjoint,
    canonical_query_fingerprint,
    regenerate_table_bitmaps,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
    FilterPredicate,
    QueryRecord,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
    load_workload,
    parse_csv_query,
    query_to_sql,
    sql_literal,
)


SAMPLE_BITS = 1_000
STRING_COLUMNS = {
    "title.imdb_index",
    "title.phonetic_code",
    "title.series_years",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--evaluation-workload", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--query-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--postgres-dsn", required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    prepare_ranges(
        args.dataset_root.resolve(),
        args.evaluation_workload.resolve(),
        args.output_root.resolve(),
        query_count=args.query_count,
        seed=args.seed,
        postgres_dsn=args.postgres_dsn,
        progress_every=args.progress_every,
    )
    return 0


def prepare_ranges(
    dataset_root: Path,
    evaluation_workload: Path,
    output_root: Path,
    *,
    query_count: int,
    seed: int,
    postgres_dsn: str,
    progress_every: int,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "domain_and_samples.pkl"
    evaluation = coerce_query_types(
        load_workload(evaluation_workload, "job_light_ranges")
    )
    started = time.perf_counter()
    if state_path.exists():
        with state_path.open("rb") as handle:
            state = coerce_state_types(pickle.load(handle))
    else:
        state = scan_domains_and_samples(dataset_root, evaluation, seed=seed)
        with state_path.open("wb") as handle:
            pickle.dump(state, handle, pickle.HIGHEST_PROTOCOL)
    domains = state["domains"]
    candidates_path = output_root / "training_candidates.csv"
    if not candidates_path.exists():
        candidates = generate_loosened_range_queries(
            evaluation, domains, count=query_count, seed=seed
        )
        write_query_csv(candidates_path, candidates)
    candidates = coerce_query_types(
        load_workload(candidates_path, "mscn_job_light_ranges_training")
    )
    assert_test_disjoint(candidates, evaluation)

    labelled_path = output_root / "training_labelled.csv"
    labelled = label_queries_resumable(
        candidates, labelled_path, postgres_dsn, progress_every=progress_every
    )
    if len(labelled) != query_count:
        raise RuntimeError(f"expected {query_count} labels, observed {len(labelled)}")
    if any(query.true_cardinality <= 0 for query in labelled):
        raise ValueError("MSCN training labels must all be positive")

    runtime_data = output_root / "data"
    runtime_workloads = output_root / "workloads"
    runtime_data.mkdir(exist_ok=True)
    runtime_workloads.mkdir(exist_ok=True)
    materialize_ranked_workload(
        labelled,
        domains,
        state["sampled_rows_by_table"],
        runtime_data / "train",
    )
    materialize_ranked_workload(
        evaluation,
        domains,
        state["sampled_rows_by_table"],
        runtime_workloads / "job-light-ranges",
    )
    write_column_min_max(runtime_data / "column_min_max_vals.csv", domains)
    manifest = {
        "status": "ready",
        "seed": seed,
        "training_query_count": len(labelled),
        "evaluation_query_count": len(evaluation),
        "sample_bits": SAMPLE_BITS,
        "test_disjoint": True,
        "operators": sorted({predicate.operator for query in labelled for predicate in query.filters}),
        "domain_sizes": {column: len(values) for column, values in sorted(domains.items())},
        "zero_training_labels": sum(query.true_cardinality == 0 for query in labelled),
        "elapsed_seconds": time.perf_counter() - started,
        "paths": {
            "training_prefix": str(runtime_data / "train"),
            "evaluation_prefix": str(runtime_workloads / "job-light-ranges"),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def scan_domains_and_samples(
    dataset_root: Path, queries: list[QueryRecord], *, seed: int
) -> dict[str, Any]:
    columns_by_table: dict[str, set[str]] = {}
    kinds: dict[str, type] = {}
    for query in queries:
        for predicate in query.filters:
            table, column = predicate.column.split(".", 1)
            columns_by_table.setdefault(table, set()).add(column)
            kinds.setdefault(predicate.column, type(predicate.value))
    domains: dict[str, set[Any]] = {
        f"{table}.{column}": set()
        for table, columns in columns_by_table.items()
        for column in columns
    }
    sampled_rows: dict[str, list[dict[str, Any]]] = {}
    for table in sorted({table.name for query in queries for table in query.tables}):
        path = dataset_root / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rng = random.Random(seed + sum(path.name.encode("utf-8")))
        reservoir: list[dict[str, Any]] = []
        selected = columns_by_table.get(table, set())
        with path.open(newline="", encoding="utf-8") as handle:
            for row_index, raw in enumerate(_dict_rows(handle)):
                row = {
                    column: _parse_cell(raw[column], kinds[f"{table}.{column}"])
                    for column in selected
                }
                for column, value in row.items():
                    if value is not None:
                        domains[f"{table}.{column}"].add(value)
                if row_index < SAMPLE_BITS:
                    reservoir.append(row)
                else:
                    replacement = rng.randint(0, row_index)
                    if replacement < SAMPLE_BITS:
                        reservoir[replacement] = row
        if len(reservoir) != SAMPLE_BITS:
            raise ValueError(f"{table} has fewer than {SAMPLE_BITS} rows")
        sampled_rows[table] = reservoir
    ordered_domains = {column: tuple(sorted(values)) for column, values in domains.items()}
    for column, values in ordered_domains.items():
        if not values:
            raise ValueError(f"empty complete domain for {column}")
    return {"domains": ordered_domains, "sampled_rows_by_table": sampled_rows}


def generate_loosened_range_queries(
    templates: list[QueryRecord],
    domains: dict[str, tuple[Any, ...]],
    *,
    count: int,
    seed: int,
) -> list[QueryRecord]:
    if count <= 0:
        raise ValueError("count must be positive")
    eligible = [
        query for query in templates
        if any(predicate.operator in {"<", "<=", ">", ">="} for predicate in query.filters)
    ]
    if not eligible:
        raise ValueError("evaluation workload has no range templates")
    rng = random.Random(seed)
    indexes = {
        column: {value: index for index, value in enumerate(values)}
        for column, values in domains.items()
    }
    forbidden = {canonical_query_fingerprint(query) for query in templates}
    observed = set(forbidden)
    generated: list[QueryRecord] = []
    attempts = 0
    while len(generated) < count and attempts < count * 200:
        attempts += 1
        template = eligible[rng.randrange(len(eligible))]
        predicates: list[FilterPredicate] = []
        changed = False
        for predicate in template.filters:
            if predicate.operator not in {"<", "<=", ">", ">="}:
                predicates.append(predicate)
                continue
            domain = domains[predicate.column]
            original = indexes[predicate.column].get(predicate.value)
            if original is None:
                raise KeyError(f"workload literal {predicate.value!r} absent from {predicate.column}")
            if predicate.operator in {">", ">="}:
                operator = rng.choice((">", ">="))
                maximum = original - 1 if operator == ">" else original
                if maximum < 0:
                    operator, maximum = ">=", original
                rank = rng.randint(0, maximum)
            else:
                operator = rng.choice(("<", "<="))
                minimum = original + 1 if operator == "<" else original
                if minimum >= len(domain):
                    operator, minimum = "<=", original
                rank = rng.randint(minimum, len(domain) - 1)
            replacement = FilterPredicate(predicate.column, operator, domain[rank])
            changed = changed or replacement != predicate
            predicates.append(replacement)
        if not changed:
            continue
        candidate = replace(
            template,
            workload="mscn_job_light_ranges_training",
            query_id=len(generated),
            filters=tuple(predicates),
            true_cardinality=0,
            source_line="",
        )
        fingerprint = canonical_query_fingerprint(candidate)
        if fingerprint in observed:
            continue
        observed.add(fingerprint)
        generated.append(candidate)
    if len(generated) != count:
        raise RuntimeError(f"generated {len(generated)} of {count} queries after {attempts} attempts")
    assert_test_disjoint(generated, templates)
    return generated


def label_queries_resumable(
    candidates: list[QueryRecord],
    output: Path,
    dsn: str,
    *,
    progress_every: int,
) -> list[QueryRecord]:
    completed = (
        coerce_query_types(load_workload(output, "mscn_job_light_ranges_training"))
        if output.exists()
        else []
    )
    if len(completed) > len(candidates):
        raise ValueError("label checkpoint contains more rows than candidate workload")
    for index, observed in enumerate(completed):
        if canonical_query_fingerprint(observed) != canonical_query_fingerprint(candidates[index]):
            raise ValueError(f"label checkpoint diverges from candidates at row {index}")
    if len(completed) == len(candidates):
        return completed
    try:
        import psycopg2
    except ImportError as exc:
        raise RuntimeError("exact MSCN range labels require psycopg2") from exc
    labels = {query.query_id: query.true_cardinality for query in completed}
    cache_path = output.with_name("grouped_exact_labels.jsonl")
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    labels[int(payload["query_id"])] = int(payload["cardinality"])
    groups: dict[tuple[Any, ...], list[QueryRecord]] = collections.defaultdict(list)
    for query in candidates:
        if query.query_id not in labels:
            groups[_label_group_key(query)].append(query)
    with psycopg2.connect(dsn) as connection, connection.cursor() as cursor, cache_path.open(
        "a", encoding="utf-8", buffering=1
    ) as cache:
        cursor.execute("SET statement_timeout = 0")
        for group_index, queries in enumerate(groups.values(), start=1):
            sql, range_columns = _grouped_label_sql(queries[0])
            cursor.execute(sql)
            rows = cursor.fetchall()
            for query in queries:
                cardinality = grouped_cardinality(query, range_columns, rows)
                if cardinality <= 0:
                    raise ValueError(
                        f"loosened query {query.query_id} unexpectedly has "
                        f"non-positive cardinality {cardinality}"
                    )
                labels[query.query_id] = cardinality
                cache.write(json.dumps({
                    "query_id": query.query_id,
                    "cardinality": cardinality,
                }, sort_keys=True) + "\n")
            if group_index % max(1, progress_every) == 0 or group_index == len(groups):
                print(
                    f"labelled_groups={group_index}/{len(groups)} "
                    f"labelled_queries={len(labels)}/{len(candidates)}",
                    flush=True,
                )
    if len(labels) != len(candidates):
        raise RuntimeError(f"missing {len(candidates) - len(labels)} exact labels")
    completed = [
        replace(query, true_cardinality=int(labels[query.query_id]))
        for query in candidates
    ]
    temporary = output.with_suffix(".tmp")
    write_query_csv(temporary, completed)
    temporary.replace(output)
    return completed


def _label_group_key(query: QueryRecord) -> tuple[Any, ...]:
    equality = tuple(
        sorted(
            (predicate.column, predicate.operator, predicate.value)
            for predicate in query.filters
            if predicate.operator == "="
        )
    )
    range_columns = tuple(
        sorted(
            predicate.column
            for predicate in query.filters
            if predicate.operator in {"<", "<=", ">", ">="}
        )
    )
    return (
        tuple((table.name, table.alias) for table in query.tables),
        tuple((join.left, join.operator, join.right) for join in query.joins),
        equality,
        range_columns,
    )


def _grouped_label_sql(query: QueryRecord) -> tuple[str, tuple[str, ...]]:
    range_columns = tuple(
        sorted(
            predicate.column
            for predicate in query.filters
            if predicate.operator in {"<", "<=", ">", ">="}
        )
    )
    if not range_columns:
        raise ValueError("grouped range labeling requires at least one range column")
    aliases = {table.alias: table.name for table in query.tables}
    title_aliases = [alias for alias, table in aliases.items() if table == "title"]
    if len(title_aliases) != 1:
        raise ValueError("JOB-light grouped labeling requires exactly one title table")
    title_alias = title_aliases[0]
    for join in query.joins:
        if join.operator != "=":
            raise ValueError("JOB-light grouped labeling supports equality joins only")
    joined_columns = {join.left for join in query.joins} | {join.right for join in query.joins}
    equality_by_alias: dict[str, list[Any]] = collections.defaultdict(list)
    for predicate in query.filters:
        if predicate.operator == "=":
            equality_by_alias[predicate.column.split(".", 1)[0]].append(predicate)
    range_by_alias: dict[str, list[str]] = collections.defaultdict(list)
    for column in range_columns:
        alias, local_column = column.split(".", 1)
        range_by_alias[alias].append(local_column)

    ctes = []
    joins = []
    weights = []
    for alias, table in aliases.items():
        if alias == title_alias:
            continue
        movie_id = f"{alias}.movie_id"
        if movie_id not in joined_columns:
            raise ValueError(f"table alias {alias!r} has no movie_id join")
        local_ranges = sorted(set(range_by_alias.get(alias, [])))
        selected_ranges = "".join(f", {alias}.{column}" for column in local_ranges)
        clauses = [
            f"{predicate.column}={sql_literal(predicate.value)}"
            for predicate in equality_by_alias.get(alias, [])
        ]
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        group_columns = ", ".join(
            [f"{alias}.movie_id"] + [f"{alias}.{column}" for column in local_ranges]
        )
        cte_name = f"agg_{alias}"
        ctes.append(
            f"{cte_name} AS (SELECT {alias}.movie_id AS __join_id{selected_ranges}, "
            f"COUNT(*)::bigint AS __count FROM {table} {alias}{where} "
            f"GROUP BY {group_columns})"
        )
        joins.append(f"JOIN {cte_name} {alias} ON {title_alias}.id={alias}.__join_id")
        weights.append(f"{alias}.__count::numeric")

    title_clauses = [
        f"{predicate.column}={sql_literal(predicate.value)}"
        for predicate in equality_by_alias.get(title_alias, [])
    ]
    where = "" if not title_clauses else " WHERE " + " AND ".join(title_clauses)
    columns = ", ".join(range_columns)
    weight = " * ".join(weights) if weights else "1::numeric"
    prefix = "" if not ctes else "WITH " + ", ".join(ctes) + " "
    sql = (
        f"{prefix}SELECT {columns}, SUM({weight}) FROM title {title_alias} "
        f"{' '.join(joins)}{where} GROUP BY {columns};"
    )
    return sql, range_columns


def grouped_cardinality(
    query: QueryRecord,
    range_columns: tuple[str, ...],
    grouped_rows: list[tuple[Any, ...]],
) -> int:
    predicates = {
        predicate.column: predicate
        for predicate in query.filters
        if predicate.operator in {"<", "<=", ">", ">="}
    }
    total = 0
    for row in grouped_rows:
        matches = True
        for index, column in enumerate(range_columns):
            predicate = predicates[column]
            value = row[index]
            if value is None or not _matches(value, predicate.operator, predicate.value):
                matches = False
                break
        if matches:
            total += int(row[-1])
    return total


def _matches(value: Any, operator: str, literal: Any) -> bool:
    if operator == "<":
        return value < literal
    if operator == "<=":
        return value <= literal
    if operator == ">":
        return value > literal
    if operator == ">=":
        return value >= literal
    raise ValueError(f"unsupported grouped range operator {operator!r}")


def materialize_ranked_workload(
    queries: list[QueryRecord],
    domains: dict[str, tuple[Any, ...]],
    sampled_rows_by_table: dict[str, list[dict[str, Any]]],
    prefix: Path,
) -> None:
    value_to_rank = {
        column: {value: rank for rank, value in enumerate(values)}
        for column, values in domains.items()
    }
    ranked = [
        replace(
            query,
            filters=tuple(
                FilterPredicate(
                    predicate.column,
                    predicate.operator,
                    value_to_rank[predicate.column][predicate.value],
                )
                for predicate in query.filters
            ),
        )
        for query in queries
    ]
    write_query_csv(prefix.with_suffix(".csv"), ranked)
    bitmaps = regenerate_table_bitmaps(queries, sampled_rows_by_table)
    bitmap_path = prefix.with_suffix(".bitmaps")
    bitmap_path.parent.mkdir(parents=True, exist_ok=True)
    with bitmap_path.open("wb") as handle:
        for query in queries:
            handle.write(len(query.tables).to_bytes(4, byteorder="little"))
            for table in query.tables:
                packed = np.packbits(bitmaps[(query.query_id, table.name)].astype(np.uint8))
                if len(packed) != SAMPLE_BITS // 8:
                    raise ValueError("unexpected materialized-sample bitmap width")
                handle.write(packed.tobytes())


def write_column_min_max(path: Path, domains: dict[str, tuple[Any, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("name", "min", "max", "cardinality", "num_unique_values"))
        for column, values in sorted(domains.items()):
            writer.writerow((column, 0, len(values) - 1, "", len(values)))


def write_query_csv(path: Path, queries: Iterable[QueryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(query_csv_line(query) + "\n")


def query_csv_line(query: QueryRecord) -> str:
    tables = ",".join(
        table.name if table.alias == table.name else f"{table.name} {table.alias}"
        for table in query.tables
    )
    joins = ",".join(f"{join.left}{join.operator}{join.right}" for join in query.joins)
    filters = ",".join(
        token
        for predicate in query.filters
        for token in (predicate.column, predicate.operator, str(predicate.value))
    )
    return f"{tables}#{joins}#{filters}#{query.true_cardinality}"


def _parse_cell(value: str, kind: type) -> Any:
    if value == "":
        return None
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    return value


def coerce_query_types(queries: list[QueryRecord]) -> list[QueryRecord]:
    return [
        replace(
            query,
            filters=tuple(
                FilterPredicate(
                    predicate.column,
                    predicate.operator,
                    str(predicate.value)
                    if predicate.column in STRING_COLUMNS
                    else predicate.value,
                )
                for predicate in query.filters
            ),
        )
        for query in queries
    ]


def coerce_state_types(state: dict[str, Any]) -> dict[str, Any]:
    domains = dict(state["domains"])
    sampled = state["sampled_rows_by_table"]
    for qualified in STRING_COLUMNS:
        if qualified in domains:
            domains[qualified] = tuple(sorted(str(value) for value in domains[qualified]))
        table, column = qualified.split(".", 1)
        for row in sampled.get(table, []):
            if row.get(column) is not None:
                row[column] = str(row[column])
    return {**state, "domains": domains, "sampled_rows_by_table": sampled}


def _dict_rows(handle):
    return csv.DictReader(handle, escapechar="\\")


if __name__ == "__main__":
    raise SystemExit(main())
