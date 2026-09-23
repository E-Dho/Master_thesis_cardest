#!/usr/bin/env python3
"""Adapted DeepDB JOB-light-ranges bridge (explicitly non-native variant).

``deepdb_bridge.py`` runs the published imdb-light schema unchanged.  This
bridge runs the project-owned adapted schema of
``joblight_eval.deepdb_ranges``: six additional modeled columns, three of them
strings replaced by PostgreSQL-collation ranks.  It owns a *separate*
preprocessing root (CSV, rank domains, HDF, sampled HDF) and never reads or
writes the shared native DeepDB preprocessing.

Stages
------
prepare-shared   PostgreSQL rank domains -> ranked, checksummed headerless CSVs
                 -> DeepDB HDF and sampled HDF (all resumable, all timed)
validate         workload support, rank rewrite vs PostgreSQL, collation check
prepare/smoke    run-level manifest; fixture smoke with two adapted queries
build            RDC statistics and SPN ensemble on the adapted HDF
validate-ensemble bounded real-data ensemble build plus evaluation
evaluate         rank-rewrite and native DeepDB inference per query
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.job_light_imdb_non_trajectory.joblight_eval import deepdb_ranges as adapted  # noqa: E402
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (  # noqa: E402
    FilterPredicate,
    QueryRecord,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (  # noqa: E402
    load_workload,
    sha256_file,
    sql_literal,
)


def _load_native_bridge():
    path = Path(__file__).resolve().with_name("deepdb_bridge.py")
    spec = importlib.util.spec_from_file_location("deepdb_native_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


native = _load_native_bridge()

TABLES = adapted.TABLES
SAMPLE_SIZES = native.SAMPLE_SIZES
POST_SAMPLING = native.POST_SAMPLING
BUDGET_FACTOR = 5
MAX_TABLES = 3
RDC_SAMPLE_SIZE = 10_000
CSV_READ_OPTIONS = {"escapechar": "\\", "quotechar": '"', "encoding": "utf-8"}
MANIFEST = "shared_preprocessing_manifest.json"
LATENCY_SCOPE = "rank_literal_rewrite_and_native_deepdb_inference"
# The published JOB-light-ranges labels are reproduced under byte-order "C"
# comparisons, not under a linguistic default collation such as en_US.UTF-8
# (series_years values like '1995-????' sort differently); ``validate``
# re-checks this against the workload labels for the recorded collation.
DEFAULT_COLLATION = "C"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command, dataset_required=True):
        command.add_argument("--source-root", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--dataset-root", required=dataset_required)
        command.add_argument("--output-directory", required=True)
        command.add_argument("--seed", type=int, default=0)

    prepare_parser = subparsers.add_parser("prepare")
    common(prepare_parser)
    prepare_parser.add_argument("--shared-root", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    common(smoke_parser, dataset_required=False)
    smoke_parser.add_argument("--shared-root")
    smoke_parser.add_argument("--queries")

    shared_parser = subparsers.add_parser("prepare-shared")
    common(shared_parser)
    shared_parser.add_argument("--postgres-dsn", required=True)
    shared_parser.add_argument("--workload", action="append", required=True)
    shared_parser.add_argument(
        "--stage", choices=("all", "domains", "dataset", "hdf"), default="all"
    )
    shared_parser.add_argument(
        "--collation", default=DEFAULT_COLLATION,
        help="PostgreSQL collation that orders the rank domains; must be the one "
             "the workload labels were produced with ('column' = column collation)",
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--source-root", required=True)
    validate_parser.add_argument("--revision", required=True)
    validate_parser.add_argument("--shared-root", required=True)
    validate_parser.add_argument("--postgres-dsn", required=True)
    validate_parser.add_argument("--workload", action="append", required=True)
    validate_parser.add_argument("--expected-query-count", type=int, action="append")
    validate_parser.add_argument(
        "--label-check-limit", type=int, default=-1,
        help="collation-dependent workload queries to recount (-1: all, 0: none)",
    )
    validate_parser.add_argument("--statement-timeout-ms", type=int, default=0)
    validate_parser.add_argument(
        "--temp-file-limit", help="PostgreSQL temp_file_limit for the checks, e.g. 50GB",
    )
    validate_parser.add_argument("--output", required=True)

    build_parser = subparsers.add_parser("build")
    common(build_parser)
    build_parser.add_argument("--database", default="imdb_joblight")
    build_parser.add_argument("--pg-host", required=True)
    build_parser.add_argument("--pg-port", type=int, default=55432)
    build_parser.add_argument("--sample-sizes", type=int, nargs=5)
    build_parser.add_argument("--rdc-sample-size", type=int, default=RDC_SAMPLE_SIZE)

    bounded_parser = subparsers.add_parser("validate-ensemble")
    common(bounded_parser)
    bounded_parser.add_argument("--database", default="imdb_joblight")
    bounded_parser.add_argument("--pg-host", required=True)
    bounded_parser.add_argument("--pg-port", type=int, default=55432)
    bounded_parser.add_argument(
        "--sample-sizes", type=int, nargs=5,
        default=[100_000, 100_000, 10_000, 10_000, 10_000],
    )
    bounded_parser.add_argument("--rdc-sample-size", type=int, default=RDC_SAMPLE_SIZE)
    bounded_parser.add_argument("--queries", required=True)
    bounded_parser.add_argument("--query-limit", type=int)
    bounded_parser.add_argument("--repetitions", type=int, default=1)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--source-root", required=True)
    evaluate_parser.add_argument("--revision", required=True)
    evaluate_parser.add_argument("--shared-root", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--queries", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--latency", required=True)
    evaluate_parser.add_argument("--query-limit", type=int)
    evaluate_parser.add_argument("--warmup-passes", type=int, default=1)
    evaluate_parser.add_argument("--repetitions", type=int, default=10)

    args = parser.parse_args(argv)
    # DeepDB's imports require chdir into the checkout: resolve paths first.
    source = native.validate_source(_path(args.source_root), args.revision)
    if args.command == "prepare":
        prepare_run(source, _path(args.dataset_root), _path(args.shared_root), _path(args.output_directory))
    elif args.command == "smoke":
        smoke_run(
            source, _path(args.output_directory),
            _path(args.shared_root),
            _path(args.queries),
        )
    elif args.command == "prepare-shared":
        prepare_shared(
            source, _path(args.dataset_root), _path(args.output_directory),
            args.postgres_dsn, [_path(path) for path in args.workload], stage=args.stage,
            collation=args.collation,
        )
    elif args.command == "validate":
        report = validate_shared(
            source, _path(args.shared_root), args.postgres_dsn,
            [_path(path) for path in args.workload],
            expected_counts=args.expected_query_count,
            label_check_limit=args.label_check_limit,
            statement_timeout_ms=args.statement_timeout_ms,
            temp_file_limit=args.temp_file_limit,
            output=_path(args.output),
        )
        return 0 if report["passed"] else 1
    elif args.command == "build":
        native._set_seed(args.seed)
        build(
            source, _path(args.dataset_root), _path(args.output_directory),
            args.database, args.pg_host, args.pg_port,
            sample_sizes=tuple(args.sample_sizes or SAMPLE_SIZES),
            rdc_sample_size=args.rdc_sample_size,
        )
    elif args.command == "validate-ensemble":
        native._set_seed(args.seed)
        report = validate_ensemble(
            source, _path(args.dataset_root), _path(args.output_directory),
            args.database, args.pg_host, args.pg_port,
            sample_sizes=tuple(args.sample_sizes), rdc_sample_size=args.rdc_sample_size,
            queries=_path(args.queries), query_limit=args.query_limit,
            repetitions=args.repetitions,
        )
        return 0 if report["passed"] else 1
    else:
        evaluate_workload(
            source, _path(args.shared_root), _path(args.checkpoint), _path(args.queries),
            _path(args.predictions), _path(args.latency), query_limit=args.query_limit,
            warmup_passes=args.warmup_passes, repetitions=args.repetitions,
        )
    return 0


def _path(value: Optional[str]) -> Optional[Path]:
    return None if value is None else Path(value).expanduser().resolve()


# ---------------------------------------------------------------------------
# schema / runtime helpers
# ---------------------------------------------------------------------------


def load_runtime(source: Path) -> Dict[str, Any]:
    runtime = native.load_upstream(source)
    from ensemble_compilation import graph_representation

    runtime["graph"] = graph_representation
    return runtime


def make_adapted_schema(runtime: Dict[str, Any], csv_root: Path, table_sizes: Dict[str, int]):
    schema = adapted.gen_job_light_ranges_adapted_schema(
        str(csv_root.resolve() / "{}.csv"), graph_module=runtime["graph"]
    )
    for table in schema.tables:
        table.table_size = int(table_sizes[table.table_name])
    return schema


def read_manifest(shared: Path) -> Dict[str, Any]:
    path = shared / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"adapted DeepDB preprocessing is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("variant_id") != adapted.VARIANT_ID or manifest.get("status") != "ready":
        raise ValueError(f"{path} is not a completed {adapted.VARIANT_ID} preprocessing root")
    return manifest


def load_shared_domains(shared: Path, manifest: Dict[str, Any]) -> Dict[str, adapted.RankDomain]:
    path = shared / adapted.DOMAIN_FILE
    observed = sha256_file(path)
    expected = manifest["domains"]["sha256"]
    if observed != expected:
        raise ValueError(f"rank-domain checksum mismatch: expected {expected}, observed {observed}")
    return adapted.load_domains(path)


def _stage_path(shared: Path, name: str) -> Path:
    return shared / "stages" / f"{name}.json"


def _read_stage(shared: Path, name: str) -> Optional[Dict[str, Any]]:
    path = _stage_path(shared, name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_stage(shared: Path, name: str, payload: Dict[str, Any]) -> None:
    native.write_json(_stage_path(shared, name), payload)


def _file_record(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


# ---------------------------------------------------------------------------
# PostgreSQL rank domains
# ---------------------------------------------------------------------------


def _connect(dsn: str):
    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("the adapted DeepDB preprocessing requires psycopg2") from exc
    connection = psycopg2.connect(dsn)
    connection.set_session(readonly=True, autocommit=True)
    return connection


def postgres_collation_metadata(cursor, columns: Sequence[str]) -> Dict[str, Any]:
    cursor.execute("SELECT current_setting('server_version'), current_database()")
    version, database = cursor.fetchone()
    cursor.execute("SELECT * FROM pg_database WHERE datname = current_database()")
    names = [description[0] for description in cursor.description]
    row = dict(zip(names, cursor.fetchone()))
    database_meta = {
        key: (None if row.get(key) is None else str(row[key]))
        for key in ("datcollate", "datctype", "datlocprovider", "daticulocale",
                    "datlocale", "daticurules", "datcollversion")
        if key in row
    }
    per_column = {}
    for column in columns:
        table, attribute = column.split(".", 1)
        cursor.execute(
            "SELECT c.collname, c.collprovider::text, c.collcollate, c.collctype, "
            "format_type(a.atttypid, a.atttypmod) "
            "FROM pg_attribute a JOIN pg_collation c ON c.oid = a.attcollation "
            "WHERE a.attrelid = %s::regclass AND a.attname = %s",
            (table, attribute),
        )
        found = cursor.fetchone()
        if found is None:
            raise ValueError(f"PostgreSQL column {column} has no collation")
        name, provider, collate, ctype, type_name = found
        effective = database_meta.get("datcollate") if name == "default" else (collate or name)
        per_column[column] = {
            "collation_name": name,
            "collation_provider": provider,
            "collcollate": collate,
            "collctype": ctype,
            "sql_type": type_name,
            "effective_collation": effective,
        }
    return {
        "server_version": version,
        "database": database,
        "database_locale": database_meta,
        "columns": per_column,
    }


def collate_clause(collation: str) -> str:
    """SQL ``COLLATE`` suffix; ``column`` keeps the column's own collation."""
    if collation == "column":
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", collation):
        raise ValueError(f"invalid collation name {collation!r}")
    return f' COLLATE "{collation}"'


def resolve_collation(cursor, collation: str) -> Dict[str, Any]:
    if collation == "column":
        return {"requested": "column", "clause": ""}
    cursor.execute(
        "SELECT collname, collprovider::text, collcollate, collctype FROM pg_collation "
        "WHERE collname = %s ORDER BY collencoding DESC LIMIT 1",
        (collation,),
    )
    found = cursor.fetchone()
    if found is None:
        raise ValueError(f"PostgreSQL has no collation named {collation!r}")
    name, provider, collate, ctype = found
    return {"requested": collation, "clause": collate_clause(collation), "collname": name,
            "collprovider": provider, "collcollate": collate, "collctype": ctype}


def fetch_postgres_domains(
    dsn: str, literals_by_column: Dict[str, List[str]], collation: str = DEFAULT_COLLATION
) -> Tuple[Dict[str, adapted.RankDomain], Dict[str, Any]]:
    """Fetch complete domains and literal boundaries under ``collation``.

    Domains come from ``ORDER BY column COLLATE collation``, the order of the
    comparisons ``column < 'literal' COLLATE collation``.  Boundaries of every
    workload literal are ``#domain values < literal`` and ``<= literal``,
    evaluated by PostgreSQL itself, so no Python collation emulation is used.
    """
    columns = adapted.RANKED_STRING_COLUMNS
    started = time.perf_counter()
    connection = _connect(dsn)
    try:
        with connection.cursor() as cursor:
            metadata = postgres_collation_metadata(cursor, columns)
            selected = resolve_collation(cursor, collation)
            metadata["selected_collation"] = selected
            clause = selected["clause"]
            domains: Dict[str, adapted.RankDomain] = {}
            counts: Dict[str, Dict[str, int]] = {}
            for column in columns:
                table, attribute = column.split(".", 1)
                cursor.execute(
                    f"SELECT {attribute} FROM {table} WHERE {attribute} IS NOT NULL "
                    f"GROUP BY {attribute} ORDER BY {attribute}{clause}"
                )
                values = tuple(row[0] for row in cursor.fetchall())
                cursor.execute(f"SELECT COUNT(*), COUNT({attribute}) FROM {table}")
                total, non_null = cursor.fetchone()
                counts[column] = {
                    "row_count": int(total),
                    "non_null_count": int(non_null),
                    "null_count": int(total) - int(non_null),
                    "distinct_count": len(values),
                }
                boundaries = _postgres_boundaries(
                    cursor, table, attribute, literals_by_column.get(column, []), clause
                )
                verified = list(values) == sorted(values)
                domain = adapted.RankDomain(
                    column=column,
                    values=values,
                    collation=(
                        str(metadata["columns"][column]["effective_collation"])
                        if collation == "column" else collation
                    ),
                    boundaries=boundaries,
                    codepoint_order_verified=verified,
                )
                for literal, (left, right) in boundaries.items():
                    rank = domain.rank(literal)
                    if rank is not None and (left, right) != (rank, rank + 1):
                        raise ValueError(
                            f"PostgreSQL boundary of in-domain literal {literal!r} for "
                            f"{column} is {(left, right)}, expected {(rank, rank + 1)}"
                        )
                domains[column] = domain
    finally:
        connection.close()
    metadata["counts"] = counts
    metadata["extraction_seconds"] = time.perf_counter() - started
    return domains, metadata


def _postgres_boundaries(cursor, table: str, attribute: str, literals: Sequence[str], clause: str = ""):
    boundaries: Dict[str, Tuple[int, int]] = {}
    if not literals:
        return boundaries
    for start in range(0, len(literals), 500):
        batch = list(literals[start:start + 500])
        values = ", ".join(["(%s::text)"] * len(batch))
        cursor.execute(
            f"WITH d AS MATERIALIZED (SELECT DISTINCT {attribute} AS v FROM {table} "
            f"WHERE {attribute} IS NOT NULL), l(lit) AS (VALUES {values}) "
            f"SELECT l.lit, (SELECT COUNT(*) FROM d WHERE d.v < l.lit{clause}), "
            f"(SELECT COUNT(*) FROM d WHERE d.v <= l.lit{clause}) FROM l",
            batch,
        )
        for literal, left, right in cursor.fetchall():
            boundaries[literal] = (int(left), int(right))
    return boundaries


# ---------------------------------------------------------------------------
# adapted dataset
# ---------------------------------------------------------------------------


TITLE_COLUMNS = tuple(adapted.NATIVE_TABLE_SPECS["title"]["attributes"])
TITLE_NUMERIC_COLUMNS = ("id", "kind_id", "production_year", "imdb_id", "episode_of_id",
                         "season_nr", "episode_nr")


def format_csv_field(value: Any) -> str:
    """Render one field so DeepDB's pandas reader recovers it exactly.

    DeepDB reads with ``escapechar='\\'`` which, unlike PostgreSQL, also
    applies outside quotes.  Every string is therefore quoted with ``\\`` and
    ``"`` escaped; NULL is the unquoted empty field; integers are bare.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        raise TypeError("boolean values are not part of the JOB-light schema")
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_ranked_title(
    rows: Iterable[Sequence[Any]], destination: Path, domains: Dict[str, adapted.RankDomain]
) -> Dict[str, Any]:
    """Write headerless title CSV whose string columns are collation ranks."""
    positions = {TITLE_COLUMNS.index(column.split(".", 1)[1]): domains[column]
                 for column in adapted.RANKED_STRING_COLUMNS}
    observed = {column: set() for column in adapted.RANKED_STRING_COLUMNS}
    non_null = {column: 0 for column in adapted.RANKED_STRING_COLUMNS}
    ids: List[int] = []
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            if len(row) != len(TITLE_COLUMNS):
                raise ValueError(f"title row has {len(row)} fields, expected {len(TITLE_COLUMNS)}")
            fields = list(row)
            for position, domain in positions.items():
                value = fields[position]
                if value is None:
                    continue
                rank = domain.rank(value)
                if rank is None:
                    raise ValueError(f"{domain.column} value {value!r} is absent from its rank domain")
                observed[domain.column].add(value)
                non_null[domain.column] += 1
                fields[position] = rank
            ids.append(int(fields[0]))
            handle.write(",".join(format_csv_field(value) for value in fields))
            handle.write("\n")
    temporary.replace(destination)
    return {
        "row_count": len(ids),
        "id_sequence_sha256": hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest(),
        "non_null_counts": non_null,
        "domain_equals_rank_domain": {
            column: observed[column] == set(domains[column].values) for column in observed
        },
    }


def iter_postgres_title(dsn: str, batch_size: int = 100_000):
    """Stream title in physical order with PostgreSQL value semantics."""
    connection = _connect(dsn)
    connection.set_session(readonly=True, autocommit=False)
    try:
        with connection.cursor() as settings:
            settings.execute("SET synchronize_seqscans = off")
        cursor = connection.cursor(name="adapted_title_export")
        cursor.itersize = batch_size
        cursor.execute(f"SELECT {', '.join(TITLE_COLUMNS)} FROM title ORDER BY ctid")
        for row in cursor:
            yield row
        cursor.close()
    finally:
        connection.rollback()
        connection.close()


def load_postgres_title_in_source_order(dsn: str, source_csv: Path) -> List[Tuple[Any, ...]]:
    """PostgreSQL title rows, reordered to the row order of the source CSV.

    COPY's bulk insert may back-fill earlier heap pages, so physical order
    differs from file order.  File order is what the native preprocessing
    sees; keeping it avoids a gratuitous difference in DeepDB's sampling.
    Only the leading ``id`` field of the source is parsed, which no quoting
    difference can shift.
    """
    import pandas as pd

    by_id = {int(row[0]): row for row in iter_postgres_title(dsn)}
    ids = pd.read_csv(source_csv, header=0, usecols=[0], dtype=str, keep_default_na=False,
                      na_filter=False, **CSV_READ_OPTIONS).iloc[:, 0].astype(int).tolist()
    if len(ids) != len(by_id) or set(ids) != set(by_id):
        raise ValueError("source title ids differ from the PostgreSQL title ids")
    return [by_id[identifier] for identifier in ids]


def iter_plain_csv_title(path: Path, *, header: bool):
    """Rows of an escape-free CSV (the smoke fixture); empty field is NULL."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if header:
            next(reader)
        for row in reader:
            yield tuple(
                None if value == "" else int(value) if column in TITLE_NUMERIC_COLUMNS else value
                for column, value in zip(TITLE_COLUMNS, row)
            )


def source_title_ids(path: Path) -> str:
    import pandas as pd

    ids = pd.read_csv(path, header=0, usecols=[0], dtype=str, keep_default_na=False,
                      na_filter=False, **CSV_READ_OPTIONS).iloc[:, 0]
    return hashlib.sha256(",".join(ids.tolist()).encode("ascii")).hexdigest()


def verify_ranked_title(
    ranked_path: Path,
    rows: Iterable[Sequence[Any]],
    domains: Dict[str, adapted.RankDomain],
    *,
    chunksize: int = 250_000,
) -> Dict[str, Any]:
    """Prove DeepDB's reader sees exactly the reference rows with ranks.

    The ranked file is read twice with DeepDB's CSV options: once with its
    default NA handling and numeric conversion (the view the SPNs learn
    from), once as raw text.  Numeric and ranked columns must equal the
    reference values (NULL -> NaN), text columns must equal the reference
    strings.
    """
    import numpy as np
    import pandas as pd

    ranked_positions = {column.split(".", 1)[1]: domains[column] for column in adapted.RANKED_STRING_COLUMNS}
    numeric_view = pd.read_csv(ranked_path, header=None, names=list(TITLE_COLUMNS),
                               chunksize=chunksize, **CSV_READ_OPTIONS)
    text_view = pd.read_csv(ranked_path, header=None, names=list(TITLE_COLUMNS), dtype=str,
                            keep_default_na=False, na_filter=False, chunksize=chunksize,
                            **CSV_READ_OPTIONS)
    iterator = iter(rows)
    checked = 0
    with numeric_view, text_view:
        for numeric, text in zip(numeric_view, text_view):
            reference = [next(iterator) for _ in range(len(numeric))]
            for index, column in enumerate(TITLE_COLUMNS):
                values = [row[index] for row in reference]
                if column in ranked_positions:
                    domain = ranked_positions[column]
                    values = [None if value is None else domain.rank(value) for value in values]
                if column in ranked_positions or column in TITLE_NUMERIC_COLUMNS:
                    expected = np.array([np.nan if value is None else float(value) for value in values])
                    observed = pd.to_numeric(numeric[column]).to_numpy(dtype=float)
                    same = (expected == observed) | (np.isnan(expected) & np.isnan(observed))
                    if not same.all():
                        bad = int(np.flatnonzero(~same)[0])
                        raise ValueError(
                            f"DeepDB reader view of title.{column} differs at row {checked + bad}: "
                            f"expected {expected[bad]!r}, observed {observed[bad]!r}"
                        )
                else:
                    expected_text = ["" if value is None else str(value) for value in values]
                    observed_text = text[column].tolist()
                    if expected_text != observed_text:
                        bad = next(i for i, (a, b) in enumerate(zip(expected_text, observed_text)) if a != b)
                        raise ValueError(
                            f"title.{column} text differs at row {checked + bad}: "
                            f"{expected_text[bad]!r} != {observed_text[bad]!r}"
                        )
            checked += len(numeric)
    if next(iterator, None) is not None:
        raise ValueError("ranked title CSV has fewer rows than the reference")
    return {"verified_rows": checked, "verified_columns": list(TITLE_COLUMNS),
            "reader": "DeepDB read_table_csv options (backslash escapechar, double-quote quotechar)"}


def native_reader_title_mismatches(source_csv: Path, reference_rows: Sequence[Sequence[Any]], *,
                                   chunksize: int = 250_000, limit: int = 20) -> Dict[str, Any]:
    """Quantify rows the native DeepDB CSV reader parses differently from PostgreSQL.

    Upstream reads with a backslash ``escapechar`` that pandas also applies
    outside quotes, whereas PostgreSQL treats unquoted backslashes literally.
    The adapted title export follows PostgreSQL; this records the rows where
    that differs from what the native preprocessing learned from.
    """
    import pandas as pd

    frames = pd.read_csv(source_csv, header=0, dtype=str, keep_default_na=False,
                         na_filter=False, chunksize=chunksize, **CSV_READ_OPTIONS)
    rows = iter(reference_rows)
    mismatches: List[Dict[str, Any]] = []
    count = 0
    checked = 0
    with frames:
        for frame in frames:
            for parsed in frame.itertuples(index=False, name=None):
                reference = next(rows)
                expected = tuple("" if value is None else str(value) for value in reference)
                if int(parsed[0]) != int(reference[0]):
                    return {"comparable": False, "reason": "row order differs from PostgreSQL",
                            "checked_rows": checked}
                if tuple(parsed) != expected:
                    count += 1
                    if len(mismatches) < limit:
                        mismatches.append({
                            "id": int(reference[0]),
                            "columns": [TITLE_COLUMNS[i] for i, (a, b) in enumerate(zip(parsed, expected)) if a != b],
                        })
                checked += 1
    return {"comparable": True, "checked_rows": checked, "mismatch_count": count,
            "examples": mismatches}


def prepare_shared(
    source: Path,
    dataset: Path,
    output: Path,
    dsn: str,
    workloads: Sequence[Path],
    *,
    stage: str = "all",
    collation: str = DEFAULT_COLLATION,
) -> Dict[str, Any]:
    output = output.resolve()
    native_hint = output.name
    if native_hint == "deepdb_shared":
        raise ValueError("refusing to write the adapted variant into the native deepdb_shared root")
    headers = native.validate_headers(dataset)
    for table in TABLES:
        if headers[table] != adapted.NATIVE_TABLE_SPECS[table]["attributes"]:
            raise ValueError(f"{table} header does not match the JOB-light schema: {headers[table]}")
    csv_root = output / "headerless_csv"
    hdf_root = output / "hdf"
    csv_root.mkdir(parents=True, exist_ok=True)
    hdf_root.mkdir(parents=True, exist_ok=True)
    queries = [query for path in workloads for query in load_workload(path, path.stem)]
    literals = adapted.string_literals_by_column(queries)

    domain_stage = _read_stage(output, "domains")
    if domain_stage is None:
        domains, metadata = fetch_postgres_domains(dsn, literals, collation)
        metadata["workloads"] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in workloads
        ]
        digest = adapted.write_domains(output / adapted.DOMAIN_FILE, domains, metadata)
        domain_stage = {
            "status": "complete",
            "seconds": metadata["extraction_seconds"],
            "file": _file_record(output / adapted.DOMAIN_FILE),
            "metadata": dict(metadata),
        }
        if domain_stage["file"]["sha256"] != digest:
            raise RuntimeError("rank-domain file changed while it was written")
        _write_stage(output, "domains", domain_stage)
    domains = adapted.load_domains(output / adapted.DOMAIN_FILE)
    recorded = domain_stage["metadata"]["selected_collation"]["requested"]
    if recorded != collation:
        raise ValueError(
            f"rank domains were built with collation {recorded!r}, requested {collation!r}; "
            f"use a fresh output directory"
        )
    missing = {
        column: [literal for literal in values if literal not in domains[column].boundaries]
        for column, values in literals.items()
    }
    if any(missing.values()):
        raise ValueError(
            "rank domains lack PostgreSQL boundaries for workload literals; delete "
            f"{_stage_path(output, 'domains')} and rerun: {missing}"
        )
    if stage == "domains":
        return domain_stage

    dataset_stage = _read_stage(output, "dataset")
    if dataset_stage is None:
        started = time.perf_counter()
        files: Dict[str, Any] = {}
        conversion: Dict[str, Any] = {}
        for table in TABLES:
            source_csv = dataset / f"{table}.csv"
            destination = csv_root / f"{table}.csv"
            table_started = time.perf_counter()
            if table == "title":
                title_rows = load_postgres_title_in_source_order(dsn, source_csv)
                conversion["title"] = write_ranked_title(title_rows, destination, domains)
                conversion["title"]["source"] = (
                    "PostgreSQL labelling database values in source CSV row order"
                )
                conversion["title"]["row_order_matches_source_csv"] = (
                    source_title_ids(source_csv) == conversion["title"]["id_sequence_sha256"]
                )
                del title_rows
            else:
                native.copy_without_header(source_csv, destination)
            conversion.setdefault(table, {})["seconds"] = time.perf_counter() - table_started
        conversion_seconds = time.perf_counter() - started
        postgres_counts = json.loads(
            (output / adapted.DOMAIN_FILE).read_text(encoding="utf-8")
        )["counts"]
        for column in adapted.RANKED_STRING_COLUMNS:
            exported = conversion["title"]["non_null_counts"][column]
            if exported != postgres_counts[column]["non_null_count"]:
                raise ValueError(
                    f"{column}: export has {exported} non-NULL values, PostgreSQL "
                    f"{postgres_counts[column]['non_null_count']}"
                )
            if not conversion["title"]["domain_equals_rank_domain"][column]:
                raise ValueError(f"{column}: exported values differ from the rank domain")
        verification_started = time.perf_counter()
        title_rows = load_postgres_title_in_source_order(dsn, dataset / "title.csv")
        verification = verify_ranked_title(csv_root / "title.csv", title_rows, domains)
        verification["native_reader_title_mismatches"] = native_reader_title_mismatches(
            dataset / "title.csv", title_rows
        )
        del title_rows
        verification_seconds = time.perf_counter() - verification_started
        for table in TABLES:
            files[table] = {
                "source": _file_record(dataset / f"{table}.csv"),
                "adapted": _file_record(csv_root / f"{table}.csv"),
            }
        dataset_stage = {
            "status": "complete",
            "seconds": conversion_seconds,
            "verification_seconds": verification_seconds,
            "conversion": conversion,
            "verification": verification,
            "files": files,
            "table_sizes": native.table_sizes_from_csv(csv_root),
        }
        _write_stage(output, "dataset", dataset_stage)
    if stage == "dataset":
        return dataset_stage

    runtime = load_runtime(source)
    table_sizes = {table: int(size) for table, size in dataset_stage["table_sizes"].items()}
    schema = make_adapted_schema(runtime, csv_root, table_sizes)
    hdf_stage = _read_stage(output, "hdf")
    if hdf_stage is None:
        hdf_started = time.perf_counter()
        runtime["prepare_all_tables"](
            schema, str(hdf_root), csv_seperator=",", max_table_data=100_000_000
        )
        hdf_seconds = time.perf_counter() - hdf_started
        sample_started = time.perf_counter()
        native.prepare_sample_hdf_compatible(runtime, schema, str(hdf_root), 100_000_000, 10_000)
        sample_seconds = time.perf_counter() - sample_started
        meta = _load_pickle(hdf_root / "meta_data.pkl")
        modeled = {
            table: list(meta[table]["relevant_attributes"]) for table in TABLES
        }
        for table, added in adapted.ADDED_MODELED_COLUMNS.items():
            absent = [column for column in added if column not in modeled[table]]
            if absent:
                raise ValueError(f"DeepDB dropped adapted columns from {table}: {absent}")
        hdf_stage = {
            "status": "complete",
            "hdf_generation_seconds": hdf_seconds,
            "sampled_hdf_generation_seconds": sample_seconds,
            "modeled_attributes": modeled,
            "null_values": {table: list(map(float, meta[table]["null_values_column"])) for table in TABLES},
            "files": {
                path.name: _file_record(path)
                for path in sorted(hdf_root.iterdir()) if path.is_file()
            },
        }
        _write_stage(output, "hdf", hdf_stage)

    domain_file = _file_record(output / adapted.DOMAIN_FILE)
    storage = {
        "headerless_csv_bytes": sum(path.stat().st_size for path in csv_root.glob("*.csv")),
        "hdf_bytes": sum(path.stat().st_size for path in hdf_root.iterdir() if path.is_file()),
        "rank_domain_bytes": domain_file["bytes"],
    }
    storage["total_bytes"] = sum(storage.values())
    storage["total_mb"] = storage["total_bytes"] / 1_000_000
    domain_metadata = domain_stage["metadata"]
    payload = {
        "status": "ready",
        "variant_id": adapted.VARIANT_ID,
        "protocol": "adapted",
        "schema": adapted.schema_description(),
        "headers": headers,
        "table_sizes": table_sizes,
        "domains": {
            "path": domain_file["path"],
            "sha256": domain_file["sha256"],
            "bytes": domain_file["bytes"],
            "selected_collation": domain_metadata["selected_collation"],
            "column_collations": domain_metadata["columns"],
            "database_locale": domain_metadata["database_locale"],
            "server_version": domain_metadata["server_version"],
            "counts": domain_metadata["counts"],
            "codepoint_order_equals_collation": {
                column: domains[column].codepoint_order_verified for column in domains
            },
            "workloads": domain_metadata["workloads"],
        },
        "rank_domain_seconds": domain_stage["seconds"],
        "headerless_conversion_seconds": dataset_stage["seconds"],
        "conversion_verification_seconds": dataset_stage["verification_seconds"],
        "hdf_generation_seconds": hdf_stage["hdf_generation_seconds"],
        "sampled_hdf_generation_seconds": hdf_stage["sampled_hdf_generation_seconds"],
        "preprocessing_seconds": (
            domain_stage["seconds"]
            + dataset_stage["seconds"]
            + hdf_stage["hdf_generation_seconds"]
            + hdf_stage["sampled_hdf_generation_seconds"]
        ),
        "preprocessing_seconds_scope": (
            "rank-domain extraction + ranked CSV conversion + DeepDB HDF + sampled HDF; "
            "conversion verification excluded"
        ),
        "csv_files": dataset_stage["files"],
        "hdf_files": hdf_stage["files"],
        "modeled_attributes": hdf_stage["modeled_attributes"],
        "storage": storage,
        "hdf_root": str(hdf_root),
        "native_shared_preprocessing_reused": False,
    }
    native.write_json(output / MANIFEST, payload)
    print(json.dumps({key: payload[key] for key in ("status", "variant_id", "table_sizes", "preprocessing_seconds", "storage")}, indent=2, sort_keys=True))
    return payload


def _load_pickle(path: Path) -> Any:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# validation against PostgreSQL
# ---------------------------------------------------------------------------


def _numpy_mask(frame, column: str, operator: str, value: Any):
    import numpy as np

    data = frame[column].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        if operator == "=":
            mask = data == float(value)
        elif operator == "<":
            mask = data < float(value)
        elif operator == "<=":
            mask = data <= float(value)
        elif operator == ">":
            mask = data > float(value)
        elif operator == ">=":
            mask = data >= float(value)
        else:
            raise ValueError(operator)
    return mask & ~np.isnan(data)


def _deepdb_title_frame(shared: Path):
    """Read the adapted title CSV with DeepDB's own reader semantics."""
    import pandas as pd

    attributes = adapted.NATIVE_TABLE_SPECS["title"]["attributes"]
    usecols = ["kind_id", "production_year", "imdb_index", "phonetic_code",
               "season_nr", "episode_nr", "series_years"]
    frame = pd.read_csv(
        shared / "headerless_csv" / "title.csv", header=None, names=attributes,
        usecols=usecols, **CSV_READ_OPTIONS,
    )
    return frame.apply(pd.to_numeric, errors="raise")


def validate_shared(
    source: Path,
    shared: Path,
    dsn: str,
    workloads: Sequence[Path],
    *,
    expected_counts: Optional[Sequence[int]],
    label_check_limit: int,
    statement_timeout_ms: int,
    output: Path,
    temp_file_limit: Optional[str] = None,
) -> Dict[str, Any]:
    # Validation may run right after the dataset stage, before the expensive
    # HDF stage; the HDF metadata then refines the modeled-column check.
    domain_stage = _read_stage(shared, "domains")
    dataset_stage = _read_stage(shared, "dataset")
    if domain_stage is None or dataset_stage is None:
        raise FileNotFoundError(f"{shared} lacks completed domain and dataset stages")
    domain_sha = domain_stage["file"]["sha256"]
    if sha256_file(shared / adapted.DOMAIN_FILE) != domain_sha:
        raise ValueError("rank-domain checksum mismatch")
    domains = adapted.load_domains(shared / adapted.DOMAIN_FILE)
    clause = domain_stage["metadata"]["selected_collation"]["clause"]
    runtime = load_runtime(source)
    schema = make_adapted_schema(
        runtime, shared / "headerless_csv",
        {table: int(size) for table, size in dataset_stage["table_sizes"].items()},
    )
    meta_path = shared / "hdf" / "meta_data.pkl"
    if meta_path.exists():
        meta = _load_pickle(meta_path)
        modeled_full = {
            column for table in TABLES for column in meta[table]["relevant_attributes_full"]
        }
        modeled_source = "hdf/meta_data.pkl"
    else:
        modeled_full = set(adapted.modeled_columns())
        modeled_source = "adapted schema (HDF stage not yet run)"
    started = time.perf_counter()
    report: Dict[str, Any] = {
        "variant_id": adapted.VARIANT_ID,
        "shared_root": str(shared),
        "domain_sha256": domain_sha,
        "selected_collation": domain_stage["metadata"]["selected_collation"],
        "column_collations": domain_stage["metadata"]["columns"],
        "database_locale": domain_stage["metadata"]["database_locale"],
        "modeled_columns_source": modeled_source,
        "workloads": {},
    }
    failures: List[str] = []
    title_frame = _deepdb_title_frame(shared)
    connection = _connect(dsn)
    cursor = connection.cursor()
    if statement_timeout_ms:
        cursor.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
    if temp_file_limit:
        cursor.execute("SELECT set_config('temp_file_limit', %s, false)", (temp_file_limit,))
    predicate_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        for index, path in enumerate(workloads):
            queries = load_workload(path, path.stem)
            expected = None if not expected_counts or index >= len(expected_counts) else expected_counts[index]
            support = {"parsed": 0, "empty_predicate": 0, "unsupported": [], "failed": []}
            conjunctions = {"checked": 0, "mismatches": []}
            for query in queries:
                unsupported = adapted.unsupported_filter_columns(query)
                if unsupported:
                    support["unsupported"].append({"query_id": query.query_id, "columns": list(unsupported)})
                    continue
                try:
                    rewrite = adapted.rewrite_query(query, domains)
                except Exception as exc:
                    support["failed"].append({"query_id": query.query_id, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if rewrite.is_empty:
                    support["empty_predicate"] += 1
                else:
                    try:
                        parsed = runtime["parse_query"](
                            native.deepdb_compatible_query_to_sql(rewrite.query), schema
                        )
                        _assert_conditions_modeled(parsed, modeled_full)
                        support["parsed"] += 1
                    except Exception as exc:
                        support["failed"].append({"query_id": query.query_id, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                for predicate in rewrite.predicates:
                    key = (predicate.column, predicate.operator, predicate.literal)
                    if key not in predicate_cache:
                        predicate_cache[key] = _check_single_predicate(
                            cursor, title_frame, predicate, clause
                        )
                checked, mismatch = _check_title_conjunction(cursor, title_frame, query, rewrite, clause)
                conjunctions["checked"] += int(checked)
                if mismatch is not None:
                    conjunctions["mismatches"].append(mismatch)
            supported = support["parsed"] + support["empty_predicate"]
            workload_report = {
                "path": str(path),
                "sha256": sha256_file(path),
                "query_count": len(queries),
                "expected_query_count": expected,
                "supported_count": supported,
                "support": support,
                "title_conjunction_checks": conjunctions,
            }
            if expected is not None and len(queries) != expected:
                failures.append(f"{path.name}: expected {expected} queries, observed {len(queries)}")
            if supported != len(queries):
                failures.append(f"{path.name}: only {supported}/{len(queries)} queries are supported")
            if conjunctions["mismatches"]:
                failures.append(f"{path.name}: {len(conjunctions['mismatches'])} title conjunction mismatches")
            report["workloads"][path.name] = workload_report

        predicate_mismatches = [entry for entry in predicate_cache.values() if not entry["match"]]
        report["predicate_checks"] = {
            "distinct_ranked_predicates": len(predicate_cache),
            "by_operator": _count_by(predicate_cache.keys(), lambda key: f"{key[0]} {key[1]}"),
            "mismatches": predicate_mismatches,
        }
        if predicate_mismatches:
            failures.append(f"{len(predicate_mismatches)} ranked predicates disagree with PostgreSQL")

        all_queries = [query for path in workloads for query in load_workload(path, path.stem)]
        report["new_child_column_checks"] = _check_new_child_columns(cursor, shared, all_queries)
        child_mismatches = [
            entry for entry in report["new_child_column_checks"]["predicates"] if not entry["match"]
        ]
        if child_mismatches:
            failures.append(f"{len(child_mismatches)} new child-column predicates disagree with PostgreSQL")

        report["collation_sensitivity"] = collation_sensitivity(domains)
        report["label_reproduction"] = _label_reproduction(
            cursor, all_queries, label_check_limit, clause
        )
        if report["label_reproduction"]["mismatches"]:
            failures.append(
                f"{len(report['label_reproduction']['mismatches'])} workload labels were not "
                "reproduced by PostgreSQL under the selected rank collation"
            )
        reproduction = report["label_reproduction"]
        if label_check_limit != 0 and reproduction.get("candidates") and not reproduction["checked"]:
            failures.append("no collation-dependent workload label could be recounted")
        if clause:
            # Evidence only: the same queries under the database default collation.
            report["label_reproduction_database_default_collation"] = _label_reproduction(
                cursor, all_queries, label_check_limit, "", results=False
            )
    finally:
        cursor.close()
        connection.close()
    report["seconds"] = time.perf_counter() - started
    report["failures"] = failures
    report["passed"] = not failures
    native.write_json(output, report)
    print(json.dumps({
        "passed": report["passed"],
        "failures": failures,
        "supported": {name: f"{item['supported_count']}/{item['query_count']}" for name, item in report["workloads"].items()},
        "distinct_ranked_predicates": report["predicate_checks"]["distinct_ranked_predicates"],
        "label_reproduction_checked": report["label_reproduction"]["checked"],
    }, indent=2, sort_keys=True))
    return report


def _assert_conditions_modeled(parsed, modeled_full) -> None:
    for table, condition in parsed.conditions:
        column = adapted_condition_column(table, condition)
        if column not in modeled_full:
            raise ValueError(f"condition {table}.{condition} targets an unmodeled column")


def adapted_condition_column(table: str, condition: str) -> str:
    for operator in ("<=", ">=", "<", ">", "="):
        if operator in condition:
            return f"{table}.{condition.split(operator, 1)[0].strip()}"
    raise ValueError(f"unsupported DeepDB condition {condition!r}")


def _count_by(items, key) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        name = key(item)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _check_single_predicate(cursor, frame, predicate, clause: str = "") -> Dict[str, Any]:
    table, attribute = predicate.column.split(".", 1)
    cursor.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {attribute} {predicate.operator} %s{clause}",
        (predicate.literal,),
    )
    expected = int(cursor.fetchone()[0])
    observed = int(_numpy_mask(frame, attribute, predicate.rank_operator, predicate.rank_value).sum())
    return {**predicate.to_dict(), "postgres_count": expected, "rank_count": observed,
            "match": expected == observed}


def _check_title_conjunction(cursor, frame, query: QueryRecord, rewrite,
                             clause: str = "") -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Compare all title filters of a query: PostgreSQL literals vs ranks."""
    import numpy as np

    aliases = {table.alias: table.name for table in query.tables}
    raw = adapted.raw_filter_literals(query)
    original = [
        (predicate.column.split(".", 1)[1], predicate.operator,
         raw_value if f"title.{predicate.column.split('.', 1)[1]}" in adapted.RANKED_STRING_COLUMNS else predicate.value)
        for predicate, raw_value in zip(query.filters, raw)
        if aliases.get(predicate.column.split(".", 1)[0]) == "title"
    ]
    if not original:
        return False, None
    where = " AND ".join(
        f"{column} {operator} %s" + (clause if f"title.{column}" in adapted.RANKED_STRING_COLUMNS else "")
        for column, operator, _ in original
    )
    cursor.execute(f"SELECT COUNT(*) FROM title WHERE {where}", tuple(value for _, _, value in original))
    expected = int(cursor.fetchone()[0])
    if rewrite.is_empty:
        observed = 0
    else:
        mask = np.ones(len(frame), dtype=bool)
        for predicate in rewrite.query.filters:
            if aliases.get(predicate.column.split(".", 1)[0]) != "title":
                continue
            mask &= _numpy_mask(frame, predicate.column.split(".", 1)[1], predicate.operator, predicate.value)
        observed = int(mask.sum())
    if expected == observed:
        return True, None
    return True, {"query_id": query.query_id, "postgres_count": expected, "rank_count": observed,
                  "empty_reason": rewrite.empty_reason}


def _check_new_child_columns(cursor, shared: Path, queries: Sequence[QueryRecord]) -> Dict[str, Any]:
    """Newly modeled numeric child columns (cast_info.nr_order) vs PostgreSQL.

    The adapted CSV is read with DeepDB's reader so the check covers exactly
    the values DeepDB learns from, including NULL (NaN) exclusion.
    """
    import pandas as pd

    started = time.perf_counter()
    predicates = set()
    for query in queries:
        aliases = {table.alias: table.name for table in query.tables}
        for predicate in query.filters:
            alias, attribute = predicate.column.split(".", 1)
            table = aliases.get(alias, alias)
            if table != "title" and attribute in adapted.ADDED_MODELED_COLUMNS.get(table, ()):
                predicates.add((table, attribute, predicate.operator, float(predicate.value)))
    results = []
    for table in sorted({item[0] for item in predicates}):
        attributes = adapted.NATIVE_TABLE_SPECS[table]["attributes"]
        wanted = sorted({item[1] for item in predicates if item[0] == table})
        frame = pd.read_csv(
            shared / "headerless_csv" / f"{table}.csv", header=None, names=attributes,
            usecols=wanted, **CSV_READ_OPTIONS,
        ).apply(pd.to_numeric, errors="raise")
        for _, attribute, operator, value in sorted(item for item in predicates if item[0] == table):
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {attribute} {operator} %s", (value,))
            expected = int(cursor.fetchone()[0])
            observed = int(_numpy_mask(frame, attribute, operator, value).sum())
            results.append({"column": f"{table}.{attribute}", "operator": operator, "value": value,
                            "postgres_count": expected, "csv_count": observed,
                            "match": expected == observed})
        del frame
    return {"distinct_predicates": len(results), "predicates": results,
            "seconds": time.perf_counter() - started}


def collation_sensitivity(domains: Dict[str, adapted.RankDomain]) -> Dict[str, Any]:
    """Quantify how much the workload depends on collation vs code-point order."""
    import bisect

    result = {}
    for column, domain in domains.items():
        codepoint = sorted(domain.values)
        differing = []
        for literal, (left, right) in sorted(domain.boundaries.items()):
            expected = (bisect.bisect_left(codepoint, literal), bisect.bisect_right(codepoint, literal))
            # ranks of the same values under the two orders
            collation_prefix = set(domain.values[:left])
            codepoint_prefix = set(codepoint[:expected[0]])
            if collation_prefix != codepoint_prefix:
                differing.append(literal)
        result[column] = {
            "collation": domain.collation,
            "domain_order_equals_codepoint": list(domain.values) == codepoint,
            "workload_literals": len(domain.boundaries),
            "literals_with_collation_dependent_lower_set": len(differing),
            "examples": differing[:10],
        }
    return result


def collated_query_sql(query: QueryRecord, clause: str) -> str:
    """Original workload SQL with ``clause`` applied to ranked string literals."""
    aliases = {table.alias: table.name for table in query.tables}
    tables = ", ".join(
        table.name if table.alias == table.name else f"{table.name} {table.alias}"
        for table in query.tables
    )
    clauses = [f"{join.left}{join.operator}{join.right}" for join in query.joins]
    for predicate, raw in zip(query.filters, adapted.raw_filter_literals(query)):
        alias, attribute = predicate.column.split(".", 1)
        if f"{aliases.get(alias, alias)}.{attribute}" in adapted.RANKED_STRING_COLUMNS:
            clauses.append(f"{predicate.column}{predicate.operator}{sql_literal(str(raw))}{clause}")
        else:
            clauses.append(f"{predicate.column}{predicate.operator}{sql_literal(predicate.value)}")
    return f"SELECT COUNT(*) FROM {tables} WHERE {' AND '.join(clauses)};"


def _label_reproduction(cursor, queries: Sequence[QueryRecord], limit: int, clause: str = "",
                        *, results: bool = True) -> Dict[str, Any]:
    """Recount workload queries whose string ranges depend on the collation.

    Queries with ``series_years``/``imdb_index`` ranges come first (their
    order differs between byte-order and linguistic collations), then
    ``phonetic_code`` ranges.  ``limit < 0`` checks every candidate.
    """
    if limit == 0:
        return {"checked": 0, "limit": limit, "mismatches": [], "results": []}

    def priority(query: QueryRecord) -> Tuple[int, int]:
        aliases = {table.alias: table.name for table in query.tables}
        columns = {
            f"{aliases.get(p.column.split('.', 1)[0])}.{p.column.split('.', 1)[1]}": p.operator
            for p in query.filters
        }
        ranked_ranges = [c for c, op in columns.items() if c in adapted.RANKED_STRING_COLUMNS and op != "="]
        rank = 0 if any(c != "title.phonetic_code" for c in ranked_ranges) else 1 if ranked_ranges else 2
        return rank, len(query.tables)

    candidates = [query for query in queries if priority(query)[0] < 2]
    candidates.sort(key=lambda query: (priority(query), query.workload, query.query_id))
    import psycopg2

    selected = candidates if limit < 0 else candidates[:limit]
    rows = []
    errors = []
    for query in selected:
        started = time.perf_counter()
        try:
            cursor.execute(collated_query_sql(query, clause))
        except (psycopg2.errors.QueryCanceled, psycopg2.errors.DiskFull,
                psycopg2.errors.ConfigurationLimitExceeded) as exc:
            # Resource limits are not evidence either way; they are reported.
            errors.append({"workload": query.workload, "query_id": query.query_id,
                           "error": type(exc).__name__,
                           "seconds": time.perf_counter() - started})
            continue
        observed = int(cursor.fetchone()[0])
        rows.append({
            "workload": query.workload,
            "query_id": query.query_id,
            "label": query.true_cardinality,
            "postgres_count": observed,
            "match": observed == query.true_cardinality,
            "seconds": time.perf_counter() - started,
        })
    mismatches = [item for item in rows if not item["match"]]
    return {
        "collate_clause": clause or "(column collation)",
        "checked": len(rows),
        "matched": len(rows) - len(mismatches),
        "not_checked_resource_limit": errors,
        "limit": limit,
        "candidates": len(candidates),
        "mismatches": mismatches if results else mismatches[:20],
        "results": rows if results else [],
    }


# ---------------------------------------------------------------------------
# run-level stages
# ---------------------------------------------------------------------------


def prepare_run(source: Path, dataset: Path, shared: Path, output: Path) -> None:
    runtime = load_runtime(source)
    headers = native.validate_headers(dataset)
    manifest = read_manifest(shared)
    load_shared_domains(shared, manifest)
    payload = {
        "status": "ready",
        "variant_id": adapted.VARIANT_ID,
        "protocol": "adapted",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "-C", str(source), "status", "--short"], text=True
        ).splitlines(),
        "python": sys.version,
        "numpy": runtime["np"].__version__,
        "pandas": runtime["pd"].__version__,
        "tables": headers,
        "schema": adapted.schema_description(),
        "shared_root": str(shared),
        "shared_manifest_sha256": sha256_file(shared / MANIFEST),
        "rank_domain_sha256": manifest["domains"]["sha256"],
        "selected_collation": manifest["domains"]["selected_collation"],
        "published_sample_sizes": SAMPLE_SIZES,
        "budget_factor": BUDGET_FACTOR,
        "max_tables_per_ensemble": MAX_TABLES,
    }
    native.write_json(output / "prepare_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


SMOKE_QUERIES = (
    # phonetic_code: ranked string range + equality on the join partner
    "movie_keyword mk,title t#t.id=mk.movie_id#t.phonetic_code,>=,P12,t.kind_id,<=,5#0",
    # cast_info.nr_order and title.season_nr: newly modeled numeric columns
    "cast_info ci,title t#t.id=ci.movie_id#ci.nr_order,<=,4,t.season_nr,>=,3#0",
)


def write_adapted_fixture(root: Path, rows: int) -> None:
    """Native fixture plus NULLs and string values in the adapted columns."""
    native.write_fixture(root, rows=rows)
    path = root / "title.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        data = list(csv.reader(handle))
    columns = adapted.NATIVE_TABLE_SPECS["title"]["attributes"]
    for index, row in enumerate(data, start=1):
        values = dict(zip(columns, row))
        values["phonetic_code"] = "" if index % 5 == 0 else f"P{index % 17}"
        values["imdb_index"] = ["", "I", "II", "III"][index % 4]
        values["series_years"] = "" if index % 3 else f"{1950 + index % 40}-????"
        values["season_nr"] = "" if index % 7 == 0 else str(index % 10)
        row[:] = [values[column] for column in columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(data)


def fixture_domains(root: Path) -> Dict[str, adapted.RankDomain]:
    columns = adapted.NATIVE_TABLE_SPECS["title"]["attributes"]
    with (root / "title.csv").open(newline="", encoding="utf-8") as handle:
        rows = [dict(zip(columns, row)) for row in csv.reader(handle)]
    domains = {}
    for column in adapted.RANKED_STRING_COLUMNS:
        attribute = column.split(".", 1)[1]
        values = tuple(sorted({row[attribute] for row in rows if row[attribute] != ""}))
        domains[column] = adapted.RankDomain(column, values, "C", codepoint_order_verified=True)
    return domains


def _exact_fixture_count(raw: Path, query: QueryRecord) -> int:
    import pandas as pd

    frames = {}
    for table in query.tables:
        attributes = adapted.NATIVE_TABLE_SPECS[table.name]["attributes"]
        frames[table.alias] = pd.read_csv(
            raw / f"{table.name}.csv", header=None, names=attributes, dtype=str,
            keep_default_na=False, na_filter=False,
        )
    title_alias = next(table.alias for table in query.tables if table.name == "title")
    other = next(table.alias for table in query.tables if table.name != "title")
    left = frames[title_alias].add_prefix(f"{title_alias}.")
    right = frames[other].add_prefix(f"{other}.")
    joined = left.merge(right, left_on=f"{title_alias}.id", right_on=f"{other}.movie_id")
    mask = pd.Series(True, index=joined.index)
    for predicate, raw_literal in zip(query.filters, adapted.raw_filter_literals(query)):
        values = joined[predicate.column]
        present = values != ""
        if predicate.column.split(".", 1)[1] in {"phonetic_code", "imdb_index", "series_years"}:
            literal = str(raw_literal)
            compare = values
        else:
            literal = float(raw_literal)
            compare = pd.to_numeric(values.where(present, None), errors="coerce")
        operator = predicate.operator
        result = {
            "=": compare == literal, "<": compare < literal, "<=": compare <= literal,
            ">": compare > literal, ">=": compare >= literal,
        }[operator]
        mask &= present & result
    return int(mask.sum())


def smoke_run(source: Path, output: Path, shared: Optional[Path], queries_path: Optional[Path]) -> None:
    runtime = load_runtime(source)
    raw = output / "fixture_raw"
    fixture = output / "fixture"
    hdf = output / "hdf"
    ensemble = output / "ensemble"
    for path in (raw, fixture, hdf, ensemble):
        path.mkdir(parents=True, exist_ok=True)
    rows = 256
    write_adapted_fixture(raw, rows=rows)
    domains = fixture_domains(raw)
    for table in TABLES:
        if table != "title":
            (fixture / f"{table}.csv").write_bytes((raw / f"{table}.csv").read_bytes())
    header = ",".join(adapted.NATIVE_TABLE_SPECS["title"]["attributes"]) + "\n"
    headered = output / "fixture_title_with_header.csv"
    headered.write_text(header + (raw / "title.csv").read_text(encoding="utf-8"), encoding="utf-8")
    write_ranked_title(iter_plain_csv_title(headered, header=True), fixture / "title.csv", domains)
    verify_ranked_title(fixture / "title.csv", iter_plain_csv_title(headered, header=True), domains)
    schema = make_adapted_schema(runtime, fixture, {table: rows for table in TABLES})
    started = time.perf_counter()
    runtime["prepare_all_tables"](schema, str(hdf), csv_seperator=",", max_table_data=10_000)
    native.prepare_sample_hdf_compatible(runtime, schema, str(hdf), 10_000, 128)
    meta = _load_pickle(hdf / "meta_data.pkl")
    for table, added in adapted.ADDED_MODELED_COLUMNS.items():
        missing = [column for column in added if column not in meta[table]["relevant_attributes"]]
        if missing:
            raise ValueError(f"smoke fixture dropped adapted columns of {table}: {missing}")
    runtime["naive_relationships"](
        schema, str(hdf), 128, str(ensemble), "imdb-light-ranges-adapted-smoke",
        False, 0.3, 10_000, 2,
    )
    ensemble_path = ensemble / "ensemble_relationships_imdb-light-ranges-adapted-smoke_128.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(ensemble_path)
    model = runtime["read_ensemble"](str(ensemble_path), build_reverse_dict=True)
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import parse_csv_query

    results = []
    for index, line in enumerate(SMOKE_QUERIES):
        query = parse_csv_query(line, "adapted_smoke", index)
        rewrite = adapted.rewrite_query(query, domains)
        if rewrite.is_empty:
            raise ValueError(f"smoke query {index} unexpectedly empty: {rewrite.empty_reason}")
        sql = native.deepdb_compatible_query_to_sql(rewrite.query)
        parsed = runtime["parse_query"](sql, schema)
        before = time.perf_counter()
        _, _, estimate, _ = model.cardinality(
            parsed, rdc_spn_selection=False, pairwise_rdc_path=None,
            merge_indicator_exp=True, max_variants=1, exploit_overlapping=True,
            return_factor_values=True,
        )
        latency = (time.perf_counter() - before) * 1_000
        estimate = float(estimate)
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError(f"invalid adapted DeepDB smoke estimate {estimate}")
        truth = _exact_fixture_count(raw, query)
        results.append({
            "query": line,
            "rewritten_sql": sql,
            "rank_predicates": [item.to_dict() for item in rewrite.predicates],
            "estimate": estimate,
            "fixture_true_cardinality": truth,
            "latency_ms": latency,
        })
    real_workload = []
    if shared is not None and queries_path is not None:
        real_workload = _real_workload_smoke(runtime, shared, queries_path)
    payload = {
        "status": "ok",
        "variant_id": adapted.VARIANT_ID,
        "query_count": len(results),
        "queries": results,
        "estimates": [item["estimate"] for item in results],
        "latency_ms": [item["latency_ms"] for item in results],
        "wall_seconds": time.perf_counter() - started,
        "ensemble_bytes": ensemble_path.stat().st_size,
        "fixture_modeled_attributes": {table: meta[table]["relevant_attributes"] for table in TABLES},
        "real_workload_rewrite_and_parse": real_workload,
        "native_hdf_generation": True,
        "native_relationship_spn_training": True,
    }
    native.write_json(output / "smoke_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _real_workload_smoke(runtime, shared: Path, queries_path: Path) -> List[Dict[str, Any]]:
    manifest = read_manifest(shared)
    domains = load_shared_domains(shared, manifest)
    schema = make_adapted_schema(
        runtime, shared / "headerless_csv",
        {table: int(size) for table, size in manifest["table_sizes"].items()},
    )
    queries = load_workload(queries_path, queries_path.stem)
    chosen = []
    phonetic = next(q for q in queries if any(p.column.endswith(".phonetic_code") for p in q.filters))
    chosen.append(phonetic)
    others = {"nr_order", "season_nr", "episode_nr", "series_years", "imdb_index"}
    chosen.append(next(
        q for q in queries if q is not phonetic
        and any(p.column.split(".", 1)[1] in others for p in q.filters)
    ))
    results = []
    for query in chosen:
        rewrite = adapted.rewrite_query(query, domains)
        sql = None if rewrite.is_empty else native.deepdb_compatible_query_to_sql(rewrite.query)
        if sql is not None:
            runtime["parse_query"](sql, schema)
        results.append({
            "query_id": query.query_id,
            "source_line": query.source_line,
            "rewritten_sql": sql,
            "empty_reason": rewrite.empty_reason,
            "rank_predicates": [item.to_dict() for item in rewrite.predicates],
        })
    return results


def build(
    source: Path,
    shared: Path,
    output: Path,
    database: str,
    pg_host: str,
    pg_port: int,
    *,
    sample_sizes: Tuple[int, ...] = SAMPLE_SIZES,
    rdc_sample_size: int = RDC_SAMPLE_SIZE,
) -> Path:
    runtime = load_runtime(source)
    manifest = read_manifest(shared)
    load_shared_domains(shared, manifest)
    schema = make_adapted_schema(
        runtime, shared / "headerless_csv",
        {table: int(size) for table, size in manifest["table_sizes"].items()},
    )
    native.configure_database(runtime, pg_host, pg_port)
    ensemble = output / "ensemble"
    ensemble.mkdir(parents=True, exist_ok=True)
    pairwise = ensemble / "pairwise_rdc.pkl"
    started = time.perf_counter()
    runtime["candidate_evaluation"](
        schema,
        str(shared / "hdf"),
        rdc_sample_size,
        list(sample_sizes),
        100_000_000,
        str(ensemble),
        database,
        list(POST_SAMPLING),
        BUDGET_FACTOR,
        MAX_TABLES,
        0.3,
        str(pairwise),
    )
    elapsed = time.perf_counter() - started
    checkpoint = ensemble / f"ensemble_join_{MAX_TABLES}_budget_{BUDGET_FACTOR}_{sample_sizes[0]}.pkl"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    spns = _ensemble_structure(runtime, checkpoint)
    metrics = {
        "variant_id": adapted.VARIANT_ID,
        "protocol": "adapted",
        "training_seconds": elapsed,
        "preprocessing_seconds": manifest["preprocessing_seconds"],
        "total_build_seconds": manifest["preprocessing_seconds"] + elapsed,
        "sample_sizes": list(sample_sizes),
        "published_sample_sizes": list(SAMPLE_SIZES),
        "uses_published_sample_sizes": tuple(sample_sizes) == tuple(SAMPLE_SIZES),
        "rdc_sample_size": rdc_sample_size,
        "post_sampling_factors": list(POST_SAMPLING),
        "budget_factor": BUDGET_FACTOR,
        "max_tables_per_ensemble": MAX_TABLES,
        "seed": int(os.environ.get("JOBLIGHT_SEED", "0")),
        "rank_domain_sha256": manifest["domains"]["sha256"],
        "shared_manifest_sha256": sha256_file(shared / MANIFEST),
        "reused_native_ensemble": False,
        "reused_native_preprocessing": False,
        "ensemble_structure": spns,
    }
    artifact = {
        "parameter_count": None,
        "serialized_model_mb": checkpoint.stat().st_size / 1_000_000,
        "full_checkpoint_mb": sum(path.stat().st_size for path in ensemble.glob("*")) / 1_000_000,
        "rank_domain_mb": manifest["domains"]["bytes"] / 1_000_000,
        "inference_artifact_mb": (checkpoint.stat().st_size + (ensemble / "pairwise_rdc.pkl").stat().st_size
                                  + manifest["domains"]["bytes"]) / 1_000_000,
        "checkpoint": str(checkpoint),
        "checkpoint_format": "DeepDB pickled SPNEnsemble plus pairwise RDC pickle and JSON rank domains",
        "training": metrics,
    }
    native.write_json(output / "build_stage_metrics.json", metrics)
    native.write_json(output / "artifact_manifest.json", artifact)
    print(json.dumps({key: value for key, value in metrics.items() if key != "ensemble_structure"}, indent=2, sort_keys=True))
    return checkpoint


def _ensemble_structure(runtime, checkpoint: Path) -> List[Dict[str, Any]]:
    model = runtime["read_ensemble"](str(checkpoint), build_reverse_dict=False)
    structure = []
    for spn in model.spns:
        structure.append({
            "tables": sorted(getattr(spn, "table_set", ())),
            "column_count": len(getattr(spn, "column_names", ()) or ()),
            "columns": list(getattr(spn, "column_names", ()) or ()),
            "full_join_size": float(getattr(spn, "full_join_size", float("nan"))),
            "full_sample_size": float(getattr(spn, "full_sample_size", float("nan"))),
        })
    return structure


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_workload(
    source: Path,
    shared: Path,
    checkpoint: Path,
    queries_path: Path,
    predictions_path: Path,
    latency_path: Path,
    *,
    query_limit: Optional[int],
    warmup_passes: int,
    repetitions: int,
) -> Dict[str, Any]:
    if warmup_passes < 0 or repetitions <= 0:
        raise ValueError("warmup passes must be nonnegative and repetitions positive")
    runtime = load_runtime(source)
    manifest = read_manifest(shared)
    domains = load_shared_domains(shared, manifest)
    schema = make_adapted_schema(
        runtime, shared / "headerless_csv",
        {table: int(size) for table, size in manifest["table_sizes"].items()},
    )
    model = runtime["read_ensemble"](str(checkpoint), build_reverse_dict=True)
    records = load_workload(queries_path, queries_path.stem)
    if query_limit is not None:
        records = records[:query_limit]
    prepared: List[Tuple[QueryRecord, Any, Any]] = []
    failures: Dict[int, str] = {}
    unsupported: Dict[int, str] = {}
    rewrite_rows: List[Dict[str, Any]] = []
    for record in records:
        columns = adapted.unsupported_filter_columns(record)
        if columns:
            unsupported[record.query_id] = (
                "columns outside the adapted DeepDB schema: " + ", ".join(columns)
            )
            continue
        try:
            rewrite = adapted.rewrite_query(record, domains)
            parsed = None if rewrite.is_empty else runtime["parse_query"](
                native.deepdb_compatible_query_to_sql(rewrite.query), schema
            )
        except Exception as exc:
            failures[record.query_id] = f"{type(exc).__name__}: {exc}"
            continue
        prepared.append((record, rewrite, parsed))
        rewrite_rows.append({
            "query_id": record.query_id,
            "empty_reason": rewrite.empty_reason or "",
            "rank_predicates": json.dumps([item.to_dict() for item in rewrite.predicates], sort_keys=True),
        })

    pairwise = checkpoint.parent / "pairwise_rdc.pkl"

    def estimate(record: QueryRecord, parsed) -> Tuple[float, float]:
        rewrite_started = time.perf_counter()
        rewrite = adapted.rewrite_query(record, domains)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1_000
        if rewrite.is_empty:
            return 0.0, rewrite_ms
        model_started = time.perf_counter()
        _, _, value, _ = model.cardinality(
            parsed,
            rdc_spn_selection=True,
            pairwise_rdc_path=str(pairwise),
            merge_indicator_exp=True,
            max_variants=1,
            exploit_overlapping=True,
            return_factor_values=True,
        )
        model_ms = (time.perf_counter() - model_started) * 1_000
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid estimate {value}")
        return value, rewrite_ms + model_ms

    for _ in range(warmup_passes):
        for record, _, parsed in prepared:
            if record.query_id in failures:
                continue
            try:
                estimate(record, parsed)
            except Exception as exc:
                failures[record.query_id] = f"{type(exc).__name__}: {exc}"
    estimates: Dict[int, float] = {}
    latency_rows: List[Dict[str, Any]] = []
    for repetition in range(repetitions):
        for record, _, parsed in prepared:
            if record.query_id in failures:
                continue
            try:
                value, elapsed = estimate(record, parsed)
                estimates.setdefault(record.query_id, value)
                latency_rows.append({
                    "query_id": record.query_id,
                    "repetition": repetition,
                    "latency_ms": elapsed,
                    "scope": LATENCY_SCOPE,
                })
            except Exception as exc:
                failures[record.query_id] = f"{type(exc).__name__}: {exc}"
    empty = {record.query_id: rewrite.empty_reason for record, rewrite, _ in prepared if rewrite.is_empty}
    prediction_rows = [
        {
            "query_id": record.query_id,
            "estimated_cardinality": "" if record.query_id in failures else estimates.get(record.query_id, ""),
            "status": (
                "unsupported" if record.query_id in unsupported
                else "failed" if record.query_id in failures or record.query_id not in estimates
                else "ok"
            ),
            "diagnostic": unsupported.get(
                record.query_id,
                failures.get(
                    record.query_id,
                    "" if record.query_id not in empty else f"empty_predicate:{empty[record.query_id]}",
                ),
            ),
        }
        for record in records
    ]
    native.write_csv(
        predictions_path, prediction_rows,
        ("query_id", "estimated_cardinality", "status", "diagnostic"),
    )
    native.write_csv(latency_path, latency_rows, ("query_id", "repetition", "latency_ms", "scope"))
    native.write_csv(
        predictions_path.with_name(predictions_path.stem + "_rank_rewrite.csv"),
        rewrite_rows, ("query_id", "empty_reason", "rank_predicates"),
    )
    summary = {
        "variant_id": adapted.VARIANT_ID,
        "query_count": len(records),
        "success_count": sum(row["status"] == "ok" for row in prediction_rows),
        "unsupported_count": len(unsupported),
        "failure_count": sum(row["status"] == "failed" for row in prediction_rows),
        "empty_predicate_count": len(empty),
        "latency_scope": LATENCY_SCOPE,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def validate_ensemble(
    source: Path,
    shared: Path,
    output: Path,
    database: str,
    pg_host: str,
    pg_port: int,
    *,
    sample_sizes: Tuple[int, ...],
    rdc_sample_size: int,
    queries: Path,
    query_limit: Optional[int],
    repetitions: int,
) -> Dict[str, Any]:
    """Bounded real-data ensemble: small samples, full pipeline, not reported."""
    import numpy as np
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.metrics import (
        raw_q_error,
        smoothed_q_error,
    )

    started = time.perf_counter()
    checkpoint = build(
        source, shared, output, database, pg_host, pg_port,
        sample_sizes=sample_sizes, rdc_sample_size=rdc_sample_size,
    )
    predictions = output / "bounded_predictions.csv"
    summary = evaluate_workload(
        source, shared, checkpoint, queries, predictions, output / "bounded_latency.csv",
        query_limit=query_limit, warmup_passes=0, repetitions=repetitions,
    )
    truth = {record.query_id: record.true_cardinality for record in load_workload(queries, queries.stem)}
    with predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ok = [row for row in rows if row["status"] == "ok"]
    raw = [raw_q_error(float(row["estimated_cardinality"]), truth[int(row["query_id"])]) for row in ok]
    smooth = [smoothed_q_error(float(row["estimated_cardinality"]), truth[int(row["query_id"])]) for row in ok]
    positive = [raw_q_error(float(row["estimated_cardinality"]), truth[int(row["query_id"])])
                for row in ok if truth[int(row["query_id"])] > 0]

    def percentiles(values):
        if not values:
            return None
        result = {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)}
        result["max"] = float(max(values))
        return result

    report = {
        "purpose": "bounded real-ensemble validation; not a reported result",
        "variant_id": adapted.VARIANT_ID,
        "sample_sizes": list(sample_sizes),
        "query_count": len(rows),
        "status_counts": _count_by(rows, lambda row: row["status"]),
        "evaluation_summary": summary,
        "raw_q_error": percentiles(raw),
        "raw_q_error_true_positive": percentiles(positive),
        "true_zero_query_count": sum(truth[int(row["query_id"])] == 0 for row in rows),
        "smoothed_q_error": percentiles(smooth),
        "failures": [row for row in rows if row["status"] != "ok"][:50],
        "seconds": time.perf_counter() - started,
    }
    report["passed"] = bool(rows) and len(ok) == len(rows)
    native.write_json(output / "bounded_validation.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "query_count", "status_counts", "raw_q_error_true_positive", "smoothed_q_error")}, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    raise SystemExit(main())
