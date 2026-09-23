#!/usr/bin/env python3
"""Native DeepDB bridge with shared preprocessing and result instrumentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABLES = (
    "title", "movie_info_idx", "movie_info", "cast_info",
    "movie_keyword", "movie_companies",
)
SAMPLE_SIZES = (10_000_000, 10_000_000, 1_000_000, 1_000_000, 1_000_000)
POST_SAMPLING = (10, 10, 5, 1, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "prepare-shared", "build"):
        command = subparsers.add_parser(name)
        command.add_argument("--source-root", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--output-directory", required=True)
        command.add_argument("--seed", type=int, default=0)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source-root", required=True)
    evaluate.add_argument("--revision", required=True)
    evaluate.add_argument("--shared-root", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--latency", required=True)
    evaluate.add_argument("--query-limit", type=int)
    evaluate.add_argument("--warmup-passes", type=int, default=1)
    evaluate.add_argument("--repetitions", type=int, default=10)
    command = subparsers.choices["build"]
    command.add_argument("--database", default="imdb_joblight")
    command.add_argument("--pg-host", required=True)
    command.add_argument("--pg-port", type=int, default=55432)
    args = parser.parse_args()
    source = validate_source(Path(args.source_root), args.revision)
    if args.command == "prepare":
        prepare(source, Path(args.dataset_root), Path(args.output_directory))
    elif args.command == "smoke":
        smoke(source, Path(args.output_directory))
    elif args.command == "prepare-shared":
        prepare_shared(source, Path(args.dataset_root), Path(args.output_directory))
    elif args.command == "build":
        _set_seed(args.seed)
        build(
            source, Path(args.dataset_root), Path(args.output_directory),
            args.database, args.pg_host, args.pg_port,
        )
    else:
        evaluate_workload(
            source,
            Path(args.shared_root),
            Path(args.checkpoint),
            Path(args.queries),
            Path(args.predictions),
            Path(args.latency),
            query_limit=args.query_limit,
            warmup_passes=args.warmup_passes,
            repetitions=args.repetitions,
        )
    return 0


def prepare(source: Path, dataset: Path, output: Path) -> None:
    runtime = load_upstream(source)
    headers = validate_headers(dataset)
    payload = {
        "status": "ready",
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
        "published_sample_sizes": SAMPLE_SIZES,
        "budget_factor": 5,
        "max_tables_per_ensemble": 3,
    }
    write_json(output / "prepare_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def smoke(source: Path, output: Path) -> None:
    runtime = load_upstream(source)
    fixture = output / "fixture"
    hdf = output / "hdf"
    ensemble = output / "ensemble"
    for path in (fixture, hdf, ensemble):
        path.mkdir(parents=True, exist_ok=True)
    write_fixture(fixture, rows=256)
    schema = make_schema(runtime, fixture, table_size=256)
    started = time.perf_counter()
    runtime["prepare_all_tables"](schema, str(hdf), csv_seperator=",", max_table_data=10_000)
    prepare_sample_hdf_compatible(runtime, schema, str(hdf), 10_000, 128)
    runtime["naive_relationships"](
        schema, str(hdf), 128, str(ensemble), "imdb-light-smoke",
        False, 0.3, 10_000, 2,
    )
    ensemble_path = ensemble / "ensemble_relationships_imdb-light-smoke_128.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError(ensemble_path)
    model = runtime["read_ensemble"](str(ensemble_path), build_reverse_dict=True)
    queries = (
        "SELECT COUNT(*) FROM title t, movie_keyword mk "
        "WHERE t.id=mk.movie_id AND t.kind_id=2",
        "SELECT COUNT(*) FROM title t, movie_info_idx mi_idx "
        "WHERE t.id=mi_idx.movie_id AND mi_idx.info_type_id=5",
    )
    estimates = []
    latencies = []
    for sql in queries:
        query = runtime["parse_query"](sql, schema)
        before = time.perf_counter()
        _, _, estimate, _ = model.cardinality(
            query, rdc_spn_selection=False, pairwise_rdc_path=None,
            merge_indicator_exp=True, max_variants=1,
            exploit_overlapping=True, return_factor_values=True,
        )
        latencies.append((time.perf_counter() - before) * 1_000)
        estimates.append(float(estimate))
    if len(estimates) != 2 or not all(value >= 0 for value in estimates):
        raise ValueError(f"invalid DeepDB smoke estimates: {estimates}")
    payload = {
        "status": "ok",
        "query_count": 2,
        "estimates": estimates,
        "latency_ms": latencies,
        "wall_seconds": time.perf_counter() - started,
        "ensemble_bytes": ensemble_path.stat().st_size,
        "native_hdf_generation": True,
        "native_relationship_spn_training": True,
    }
    write_json(output / "smoke_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def prepare_shared(source: Path, dataset: Path, output: Path) -> None:
    runtime = load_upstream(source)
    headers = validate_headers(dataset)
    headerless = output / "headerless_csv"
    hdf = output / "hdf"
    headerless.mkdir(parents=True, exist_ok=True)
    hdf.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    conversion_started = time.perf_counter()
    for table in TABLES:
        destination = headerless / f"{table}.csv"
        source_path = dataset / f"{table}.csv"
        if not destination.exists() or destination.stat().st_mtime < source_path.stat().st_mtime:
            copy_without_header(source_path, destination)
    conversion_seconds = time.perf_counter() - conversion_started
    sizes = table_sizes_from_csv(headerless)
    schema = make_schema(runtime, headerless, table_sizes=sizes)
    hdf_started = time.perf_counter()
    if not (hdf / "meta_data.pkl").exists():
        runtime["prepare_all_tables"](
            schema, str(hdf), csv_seperator=",", max_table_data=100_000_000
        )
    hdf_seconds = time.perf_counter() - hdf_started
    sample_started = time.perf_counter()
    if not (hdf / "meta_data_sampled.pkl").exists():
        prepare_sample_hdf_compatible(runtime, schema, str(hdf), 100_000_000, 10_000)
    sample_seconds = time.perf_counter() - sample_started
    payload = {
        "status": "ready",
        "headers": headers,
        "table_sizes": sizes,
        "headerless_conversion_seconds": conversion_seconds,
        "hdf_generation_seconds": hdf_seconds,
        "sampled_hdf_generation_seconds": sample_seconds,
        "preprocessing_seconds": time.perf_counter() - started,
        "hdf_root": str(hdf),
    }
    write_json(output / "shared_preprocessing_manifest.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _set_seed(seed: int) -> None:
    os.environ["JOBLIGHT_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def build(
    source: Path,
    shared: Path,
    output: Path,
    database: str,
    pg_host: str,
    pg_port: int,
) -> None:
    runtime = load_upstream(source)
    manifest = json.loads((shared / "shared_preprocessing_manifest.json").read_text())
    schema = make_schema(
        runtime, shared / "headerless_csv", table_sizes=manifest["table_sizes"]
    )
    configure_database(runtime, pg_host, pg_port)
    ensemble = output / "ensemble"
    ensemble.mkdir(parents=True, exist_ok=True)
    pairwise = ensemble / "pairwise_rdc.pkl"
    started = time.perf_counter()
    runtime["candidate_evaluation"](
        schema,
        str(shared / "hdf"),
        10_000,
        list(SAMPLE_SIZES),
        100_000_000,
        str(ensemble),
        database,
        list(POST_SAMPLING),
        5,
        3,
        0.3,
        str(pairwise),
    )
    elapsed = time.perf_counter() - started
    checkpoint = ensemble / "ensemble_join_3_budget_5_10000000.pkl"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    metrics = {
        "training_seconds": elapsed,
        "total_build_seconds": manifest["preprocessing_seconds"] + elapsed,
        "published_sample_sizes": SAMPLE_SIZES,
        "budget_factor": 5,
        "max_tables_per_ensemble": 3,
        "seed": int(os.environ.get("JOBLIGHT_SEED", "0")),
    }
    artifact = {
        "parameter_count": None,
        "serialized_model_mb": checkpoint.stat().st_size / 1_000_000,
        "full_checkpoint_mb": sum(path.stat().st_size for path in ensemble.glob("*")) / 1_000_000,
        "checkpoint": str(checkpoint),
        "training": metrics,
    }
    write_json(output / "build_stage_metrics.json", metrics)
    write_json(output / "artifact_manifest.json", artifact)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_workload(
    source: Path,
    shared: Path,
    checkpoint: Path,
    queries_path: Path,
    predictions_path: Path,
    latency_path: Path,
    *,
    query_limit: int | None,
    warmup_passes: int,
    repetitions: int,
) -> None:
    if warmup_passes < 0 or repetitions <= 0:
        raise ValueError("warmup passes must be nonnegative and repetitions positive")
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
        load_workload,
    )

    runtime = load_upstream(source)
    manifest = json.loads((shared / "shared_preprocessing_manifest.json").read_text())
    schema = make_schema(
        runtime, shared / "headerless_csv", table_sizes=manifest["table_sizes"]
    )
    model = runtime["read_ensemble"](str(checkpoint), build_reverse_dict=True)
    records = load_workload(queries_path, queries_path.stem)
    if query_limit is not None:
        records = records[:query_limit]
    parsed: list[tuple[int, Any]] = []
    failures: dict[int, str] = {}
    unsupported: dict[int, str] = {}
    for record in records:
        try:
            unsupported_columns = deepdb_unsupported_filter_columns(record, schema)
            if unsupported_columns:
                unsupported[record.query_id] = (
                    "columns excluded by the native DeepDB JOB-light schema: "
                    + ", ".join(unsupported_columns)
                )
                continue
            parsed.append(
                (
                    record.query_id,
                    runtime["parse_query"](deepdb_compatible_query_to_sql(record), schema),
                )
            )
        except Exception as exc:
            failures[record.query_id] = f"{type(exc).__name__}: {exc}"

    pairwise = checkpoint.parent / "pairwise_rdc.pkl"

    def estimate(query) -> float:
        _, _, value, _ = model.cardinality(
            query,
            rdc_spn_selection=True,
            pairwise_rdc_path=str(pairwise),
            merge_indicator_exp=True,
            max_variants=1,
            exploit_overlapping=True,
            return_factor_values=True,
        )
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid estimate {value}")
        return value

    for _ in range(warmup_passes):
        for query_id, query in parsed:
            if query_id in failures:
                continue
            try:
                estimate(query)
            except Exception as exc:
                failures[query_id] = f"{type(exc).__name__}: {exc}"
    estimates: dict[int, float] = {}
    latency_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for query_id, query in parsed:
            if query_id in failures:
                continue
            try:
                started = time.perf_counter()
                value = estimate(query)
                elapsed = (time.perf_counter() - started) * 1_000
                estimates.setdefault(query_id, value)
                latency_rows.append(
                    {
                        "query_id": query_id,
                        "repetition": repetition,
                        "latency_ms": elapsed,
                        "scope": "predicate_encoding_and_native_deepdb_inference",
                    }
                )
            except Exception as exc:
                failures[query_id] = f"{type(exc).__name__}: {exc}"
    prediction_rows = [
        {
            "query_id": record.query_id,
            "estimated_cardinality": estimates.get(record.query_id, ""),
            "status": (
                "unsupported"
                if record.query_id in unsupported
                else "failed"
                if record.query_id in failures
                else "ok"
            ),
            "diagnostic": unsupported.get(
                record.query_id, failures.get(record.query_id, "")
            ),
        }
        for record in records
    ]
    write_csv(
        predictions_path, prediction_rows,
        ("query_id", "estimated_cardinality", "status", "diagnostic"),
    )
    write_csv(
        latency_path, latency_rows,
        ("query_id", "repetition", "latency_ms", "scope"),
    )
    print(json.dumps({
        "query_count": len(records),
        "success_count": len(records) - len(failures) - len(unsupported),
        "unsupported_count": len(unsupported),
        "failure_count": len(failures),
    }, indent=2, sort_keys=True))


def deepdb_compatible_query_to_sql(query: Any) -> str:
    """Render JOB-light joins using DeepDB's title-centered relationship schema.

    JOB-light-ranges sometimes encodes its connected movie-id join tree with
    child-to-child edges.  DeepDB registers only child.movie_id = title.id
    relationships.  For a valid connected JOB-light equijoin, both forms are
    relationally equivalent, so canonicalize the tree before native parsing.
    """
    from dataclasses import replace

    from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import (
        JoinPredicate,
    )
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
        query_to_sql,
    )

    aliases = {table.alias: table.name for table in query.tables}
    title_aliases = [alias for alias, name in aliases.items() if name == "title"]
    if len(title_aliases) != 1:
        return query_to_sql(query)
    title_alias = title_aliases[0]

    adjacency = {alias: set() for alias in aliases}
    for join in query.joins:
        if join.operator != "=":
            raise ValueError("DeepDB JOB-light canonicalization requires equality joins")
        try:
            left_alias, left_column = join.left.split(".", 1)
            right_alias, right_column = join.right.split(".", 1)
        except ValueError as exc:
            raise ValueError(f"invalid qualified JOB-light join: {join}") from exc
        for alias, column in ((left_alias, left_column), (right_alias, right_column)):
            expected = "id" if alias == title_alias else "movie_id"
            if alias not in aliases or column != expected:
                raise ValueError(f"unsupported JOB-light relationship endpoint: {alias}.{column}")
        adjacency[left_alias].add(right_alias)
        adjacency[right_alias].add(left_alias)

    reachable = {title_alias}
    frontier = [title_alias]
    while frontier:
        new_aliases = adjacency[frontier.pop()] - reachable
        reachable.update(new_aliases)
        frontier.extend(new_aliases)
    if reachable != set(aliases):
        raise ValueError("JOB-light join graph must connect every table to title")

    canonical_joins = tuple(
        JoinPredicate(f"{alias}.movie_id", "=", f"{title_alias}.id")
        for alias, table_name in aliases.items()
        if table_name != "title"
    )
    return query_to_sql(replace(query, joins=canonical_joins))


def deepdb_unsupported_filter_columns(query: Any, schema: Any) -> tuple[str, ...]:
    """Return filters that the selected native DeepDB schema cannot model."""
    aliases = {table.alias: table.name for table in query.tables}
    unsupported = set()
    for predicate in query.filters:
        try:
            alias, attribute = predicate.column.split(".", 1)
        except ValueError:
            unsupported.add(predicate.column)
            continue
        table_name = aliases.get(alias)
        table = None if table_name is None else schema.table_dictionary.get(table_name)
        if (
            table is None
            or attribute not in table.attributes
            or attribute in table.irrelevant_attributes
        ):
            unsupported.add(predicate.column)
    return tuple(sorted(unsupported))


def load_upstream(source: Path) -> dict[str, Any]:
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import pandas as pd
    import ensemble_creation.rdc_based as rdc_based
    from data_preparation.join_data_preparation import prepare_sample_hdf
    from data_preparation.prepare_single_tables import prepare_all_tables
    from ensemble_compilation.spn_ensemble import read_ensemble
    from ensemble_creation.naive import naive_every_relationship_ensemble
    from evaluation.utils import parse_query
    from schemas.imdb.schema import gen_job_light_imdb_schema
    return {
        "np": np, "pd": pd, "rdc_based": rdc_based,
        "prepare_sample_hdf": prepare_sample_hdf,
        "prepare_all_tables": prepare_all_tables,
        "read_ensemble": read_ensemble,
        "naive_relationships": naive_every_relationship_ensemble,
        "parse_query": parse_query,
        "schema": gen_job_light_imdb_schema,
        "candidate_evaluation": rdc_based.candidate_evaluation,
    }


def make_schema(runtime, csv_root: Path, table_size: int | None = None, table_sizes=None):
    schema = runtime["schema"](str(csv_root.resolve() / "{}.csv"))
    for table in schema.tables:
        table.table_size = int(table_size if table_sizes is None else table_sizes[table.table_name])
    return schema


def configure_database(runtime, host: str, port: int) -> None:
    base = runtime["rdc_based"].DBConnection

    class ClusterConnection(base):
        def __init__(self, db="imdb_joblight", **_kwargs):
            super().__init__(db_user=os.environ.get("USER", "sunip956"), db_password="",
                             db_host=host, db_port=str(port), db=db)

    runtime["rdc_based"].DBConnection = ClusterConnection


def prepare_sample_hdf_compatible(
    runtime: dict[str, Any], schema, hdf_path: str, max_table_data: int, sample_size: int
) -> None:
    """Run the pinned sampler across pandas' stricter index/column check.

    DeepDB deliberately retains each join key both as an index and a column,
    then passes the column name to ``merge(left_on=...)``. Pandas 1.5 rejects
    that formerly accepted, unambiguous case before selecting the column. The
    index and column contain the same key values here, so suppress only this
    precondition check for the duration of the native upstream function.
    """

    frame = runtime["pd"].DataFrame
    original = frame._check_label_or_level_ambiguity
    frame._check_label_or_level_ambiguity = lambda self, key, axis=0: None
    try:
        runtime["prepare_sample_hdf"](
            schema, hdf_path, max_table_data, sample_size
        )
    finally:
        frame._check_label_or_level_ambiguity = original


def validate_source(source: Path, revision: str) -> Path:
    source = source.resolve()
    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != revision:
        raise ValueError(f"DeepDB revision mismatch: expected {revision}, got {observed}")
    return source


def validate_headers(dataset: Path) -> dict[str, list[str]]:
    result = {}
    for table in TABLES:
        path = dataset / f"{table}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            result[table] = next(csv.reader(handle, escapechar="\\"))
        if not result[table] or result[table][0] != "id":
            raise ValueError(f"missing canonical header in {path}")
    return result


def copy_without_header(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        reader.readline()
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    temporary.replace(destination)


def table_sizes_from_csv(root: Path) -> dict[str, int]:
    sizes = {}
    for table in TABLES:
        with (root / f"{table}.csv").open("rb") as handle:
            sizes[table] = sum(1 for line in handle if line.strip())
    return sizes


def write_fixture(root: Path, rows: int) -> None:
    schemas = {
        "title": ["id", "title", "imdb_index", "kind_id", "production_year", "imdb_id", "phonetic_code", "episode_of_id", "season_nr", "episode_nr", "series_years", "md5sum"],
        "movie_info_idx": ["id", "movie_id", "info_type_id", "info", "note"],
        "movie_info": ["id", "movie_id", "info_type_id", "info", "note"],
        "cast_info": ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"],
        "movie_keyword": ["id", "movie_id", "keyword_id"],
        "movie_companies": ["id", "movie_id", "company_id", "company_type_id", "note"],
    }
    for table, columns in schemas.items():
        with (root / f"{table}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for index in range(1, rows + 1):
                values = {
                    "id": index, "title": f"title-{index}", "imdb_index": "",
                    "kind_id": index % 7 + 1, "production_year": 1950 + index % 70,
                    "imdb_id": index, "phonetic_code": f"P{index % 17}", "episode_of_id": "",
                    "season_nr": index % 10, "episode_nr": index % 30, "series_years": "",
                    "md5sum": f"hash-{index}", "movie_id": index, "info_type_id": index % 11 + 1,
                    "info": str(index), "note": "", "person_id": index,
                    "person_role_id": index, "nr_order": index % 10, "role_id": index % 11 + 1,
                    "keyword_id": index % 101 + 1, "company_id": index % 53 + 1,
                    "company_type_id": index % 2 + 1,
                }
                writer.writerow([values[column] for column in columns])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
