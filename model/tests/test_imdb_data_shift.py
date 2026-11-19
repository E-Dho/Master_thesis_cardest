"""Data-shift ablation on JOB-light-ranges: model/scripts/imdb_data_shift.py and fine-tuning.

On a synthetic JOB-light data set (six CSVs with NeuroCard's quoting: escaped
quotes, embedded newlines and commas, NULLs; titles in random file order):

- the full-outer-join count equals an explicit enumeration of the join;
- the temporal split keeps the earliest 80 % with NULL years last;
- the subset CSVs hold exactly the kept records, byte for byte;
- the exact counter equals SQLite's COUNT(*) on the full and the kept data, and
  a wrong label in the query file is refused;
- the prepared subset keeps the full domains, takes the sampler's join
  cardinality (a mismatch is refused) and trains; fine-tuning loads its weights
  exactly; a checkpoint with other domains is refused;
- the private NeuroCard working directory links the package and reads the kept CSVs.
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model.scripts import imdb_data_shift as shift

REPO = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = REPO / "model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml"
TRICKY = ['plain', 'with, comma', 'with "quote"', 'two\nlines', 'tab\there', 'end quote"', '',
          'back\\slash in "quotes"']
# an unquoted text ending in a backslash: PostgreSQL reads it literally, pandas' escapechar
# escapes the following comma and shifts the later columns of that record (as in IMDB)
SHIFTING = 'ends in a backslash\\'
# the columns the model sees (a small "content" projection); the CSVs carry more
MODEL_COLUMNS = {
    "title": ["kind_id", "production_year", "phonetic_code", "series_years"],
    "cast_info": ["role_id", "nr_order"],
    "movie_companies": ["company_type_id"],
    "movie_info": ["info_type_id"],
    "movie_keyword": ["keyword_id"],
    "movie_info_idx": ["info_type_id"],
}
HEADERS = {
    # text columns before numeric ones, as in IMDB's title and cast_info
    "title": ["id", "kind_id", "phonetic_code", "series_years", "title", "production_year"],
    "cast_info": ["id", "movie_id", "note", "role_id", "nr_order"],
    "movie_companies": ["id", "movie_id", "company_type_id"],
    "movie_info": ["id", "movie_id", "info_type_id", "info"],
    "movie_keyword": ["id", "movie_id", "keyword_id"],
    "movie_info_idx": ["id", "movie_id", "info_type_id"],
}


def _pg_field(value) -> str:
    """A value as PostgreSQL's COPY ... TO (FORMAT CSV, NULL '', ESCAPE '\\') writes it."""

    if value is None:
        return ""
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(_pg_field(value) for value in row) + "\n")


def _imdb(directory: Path, *, titles: int = 60, orphans: bool = False, seed: int = 0) -> dict:
    """Synthetic JOB-light CSVs; the true values are kept in ``_truth.json`` next to them."""

    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True)
    ids = rng.permutation(np.arange(1, titles + 1) * 7)
    rows = []
    for index, title_id in enumerate(ids):
        year = None if rng.random() < 0.12 else float(rng.integers(1950, 2021))
        code = None if rng.random() < 0.1 else f"{'ABS'[rng.integers(0, 3)]}{rng.integers(100, 999)}"
        if index in (3, 4):
            code = ["", "NA"][index - 3]  # not NULL in PostgreSQL, NaN in pandas
        series = None if rng.random() < 0.5 else f"{rng.integers(1990, 2000)}-{rng.integers(2000, 2010)}"
        text = SHIFTING if index % 9 == 2 else TRICKY[index % len(TRICKY)]
        rows.append([int(title_id), int(rng.integers(1, 8)), code, series, text, year])
    tables = {"title": rows}
    values = {
        "cast_info": lambda i: [SHIFTING if i % 7 == 3 else TRICKY[i % len(TRICKY)], int(rng.integers(1, 5)),
                                None if rng.random() < 0.3 else int(rng.integers(1, 6))],
        "movie_companies": lambda i: [int(rng.integers(1, 3))],
        "movie_info": lambda i: [int(rng.integers(1, 6)), TRICKY[(i + 3) % len(TRICKY)]],
        "movie_keyword": lambda i: [int(rng.integers(1, 9))],
        "movie_info_idx": lambda i: [int(rng.integers(99, 102))],
    }
    for table, value in values.items():
        table_rows, row_id = [], 0
        movie_ids = list(ids) + ([99999, 99999] if orphans and table == "movie_keyword" else [])
        for movie_id in movie_ids:
            for _ in range(int(rng.integers(0, 4)) if movie_id != 99999 else 1):
                row_id += 1
                table_rows.append([row_id, int(movie_id), *value(row_id)])
        order = rng.permutation(len(table_rows))
        tables[table] = [table_rows[i] for i in order]
    for table, table_rows in tables.items():
        _write_csv(directory / f"{table}.csv", HEADERS[table], table_rows)
    (directory / "_truth.json").write_text(json.dumps(tables))
    return {"title_ids": np.asarray(ids, dtype=np.int64)}


def _truth_db(full_directory: Path, kept_title_ids=None) -> sqlite3.Connection:
    """SQLite with the true values (not parsed from the CSVs), optionally only the kept titles' data."""

    tables = json.loads((Path(full_directory) / "_truth.json").read_text())
    kept = None if kept_title_ids is None else set(int(v) for v in kept_title_ids)
    all_ids = {row[0] for row in tables["title"]}
    connection = sqlite3.connect(":memory:")
    for table, rows in tables.items():
        if kept is not None:
            rows = [row for row in rows if (row[0] in kept) if table == "title"] if table == "title" else \
                [row for row in rows if row[1] in kept or row[1] not in all_ids]
        columns = HEADERS[table]
        connection.execute(f"CREATE TABLE {table} ({', '.join(columns)})")
        connection.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(columns))})", rows)
    return connection


def _frames(directory: Path, columns=None) -> dict:
    out = {}
    for table in shift.JOB_LIGHT_TABLES:
        usecols = None if columns is None else [shift.KEY[table], *columns[table]]
        out[table] = pd.read_csv(directory / f"{table}.csv", usecols=usecols, escapechar="\\")
    return out


def _enumerate_join(directory: Path) -> list:
    """Every row of the full outer join, as dicts in NeuroCard's column naming."""

    frames = _frames(directory, MODEL_COLUMNS)
    children = [table for table in shift.JOB_LIGHT_TABLES if table != "title"]
    by_movie = {table: frames[table].groupby("movie_id") for table in children}
    groups = {table: {key: group for key, group in by_movie[table]} for table in children}
    rows = []
    for _, title in frames["title"].iterrows():
        options = []
        for table in children:
            group = groups[table].get(int(title["id"]))
            options.append([None] if group is None else [r for _, r in group.iterrows()])
        for combo in itertools.product(*options):
            row = {f"title:{c}": title[c] for c in MODEL_COLUMNS["title"]}
            row["__in_title"] = 1
            for table, child in zip(children, combo):
                count = 0 if groups[table].get(int(title["id"])) is None else len(groups[table][int(title["id"])])
                for c in MODEL_COLUMNS[table]:
                    row[f"{table}:{c}"] = np.nan if child is None else child[c]
                row[f"__in_{table}"] = np.nan if child is None else 1
                row[f"__fanout_{table}"] = max(count, 1)
            rows.append(row)
    title_ids = set(frames["title"]["id"].astype(int))
    for table in children:  # orphan child rows: rows of their own
        for _, child in frames[table].iterrows():
            if int(child["movie_id"]) not in title_ids:
                row = {"__in_title": np.nan}
                for c in MODEL_COLUMNS["title"]:
                    row[f"title:{c}"] = np.nan
                for other in children:
                    for c in MODEL_COLUMNS[other]:
                        row[f"{other}:{c}"] = child[c] if other == table else np.nan
                    row[f"__in_{other}"] = 1 if other == table else np.nan
                    row[f"__fanout_{other}"] = 1
                rows.append(row)
    return rows


def _sql_literal(value) -> str:
    return f"'{value}'" if isinstance(value, str) else repr(value)


def _queries(directory: Path, count: int, seed: int = 1) -> list:
    """Random JOB-light-ranges-like queries (NeuroCard's line format) with SQLite labels on the true values."""

    rng = np.random.default_rng(seed)
    truth = json.loads((Path(directory) / "_truth.json").read_text())
    connection = _truth_db(directory)
    children = [table for table in shift.JOB_LIGHT_TABLES if table != "title"]
    lines = []
    for _ in range(count):
        tables = ["title", *rng.choice(children, size=int(rng.integers(1, 4)), replace=False)]
        predicates = []
        for table in tables:
            for column in MODEL_COLUMNS[table]:
                if rng.random() < 0.3:
                    index = HEADERS[table].index(column)
                    values = [row[index] for row in truth[table] if row[index] not in (None, "")]
                    value = values[int(rng.integers(0, len(values)))]
                    predicates.append((table, column, str(rng.choice(["=", "<=", ">="])), value))
        joins = ",".join(f"title.id={t}.movie_id" for t in tables[1:])
        filters = ",".join(f"{t}.{c},{op},{v}" for t, c, op, v in predicates)
        where = [f"title.id={t}.movie_id" for t in tables[1:]] + [f"{t}.{c} {op} {_sql_literal(v)}" for t, c, op, v in predicates]
        sql = f"SELECT COUNT(*) FROM {', '.join(tables)}" + (f" WHERE {' AND '.join(where)}" if where else "")
        lines.append((f"{','.join(tables)}#{joins}#{filters}", sql, connection.execute(sql).fetchone()[0]))
    return lines


class ImdbDataShiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.full = cls.root / "full"
        _imdb(cls.full)
        cls.expected_full = len(_enumerate_join(cls.full))
        cls.selection = shift.select(cls.full, "temporal", expected_full=cls.expected_full)
        cls.split = cls.root / "split"
        cls.record = shift.write_split(cls.split, cls.selection)
        cls.subset = cls.root / "subset"
        cls.subset_summary = shift.subset_csvs(cls.full, cls.split, cls.subset)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_csv_round_trip_is_what_pandas_reads(self) -> None:
        notes = pd.read_csv(self.full / "cast_info.csv", escapechar="\\", keep_default_na=False)["note"]
        self.assertTrue(set(TRICKY) <= set(notes.tolist()))

    def test_full_outer_join_count_matches_enumeration(self) -> None:
        self.assertEqual(self.record["full_join_cardinality_full"], self.expected_full)
        self.assertEqual(self.record["full_join_cardinality_subset"], len(_enumerate_join(self.subset)))
        with tempfile.TemporaryDirectory() as tmp:
            orphans = Path(tmp) / "orphans"
            _imdb(orphans, orphans=True, seed=3)
            count = shift.select(orphans, "random", expected_full=None)["record"]
            self.assertEqual(count["full_join_cardinality_full"], len(_enumerate_join(orphans)))
            self.assertEqual(count["orphan_child_rows_full"]["movie_keyword"], 2)
            with self.assertRaisesRegex(ValueError, "NeuroCard's constant"):
                shift.select(orphans, "random", expected_full=count["full_join_cardinality_full"] + 1)

    def test_temporal_split_keeps_the_earliest_titles_null_years_last(self) -> None:
        title = pd.read_csv(self.full / "title.csv", escapechar="\\")
        title["null"] = title["production_year"].isna()
        ranked = title.sort_values(["null", "production_year", "id"])
        keep = int(round(0.8 * len(title)))
        self.assertEqual(sorted(ranked["id"].iloc[:keep]), self.selection["kept_title_ids"].tolist())
        self.assertEqual(self.record["titles_kept"], keep)
        self.assertGreater(self.record["null_production_year_titles"], 0)
        random = shift.select(self.full, "random", expected_full=self.expected_full)
        self.assertEqual(random["record"]["titles_kept"], keep)
        self.assertNotEqual(random["kept_title_ids"].tolist(), self.selection["kept_title_ids"].tolist())

    def test_subset_csvs_hold_exactly_the_kept_records(self) -> None:
        kept = set(self.selection["kept_title_ids"].tolist())
        for table in shift.JOB_LIGHT_TABLES:
            full = pd.read_csv(self.full / f"{table}.csv", escapechar="\\", keep_default_na=False, dtype=str)
            subset = pd.read_csv(self.subset / f"{table}.csv", escapechar="\\", keep_default_na=False, dtype=str)
            mask = full[shift.KEY[table]].astype(int).isin(kept)
            self.assertTrue(full[mask].reset_index(drop=True).equals(subset), table)
            self.assertEqual(self.subset_summary["tables"][table]["rows_kept"], int(mask.sum()))
        # byte for byte: every subset record occurs verbatim in the source
        source = (self.full / "movie_info.csv").read_bytes()
        records = [text for _fields, text in shift._raw_records(self.subset / "movie_info.csv")]
        self.assertTrue(any("\n" in text.rstrip("\n") for text in records))  # a multi-line record
        self.assertTrue(all(text.encode() in source for text in records))
        with self.assertRaises(FileExistsError):
            shift.subset_csvs(self.full, self.split, self.subset)

    def test_relabel_equals_sqlite_and_checks_the_full_labels(self) -> None:
        lines = _queries(self.full, 60)
        queries = self.root / "queries.csv"
        queries.write_text("\n".join(f"{line}#{truth}" for line, _sql, truth in lines) + "\n")
        output = self.root / "queries_subset.csv"
        summary = shift.relabel(queries, self.full, self.subset, self.split, output)
        subset_db = _truth_db(self.full, self.selection["kept_title_ids"])
        expected = [subset_db.execute(sql).fetchone()[0] for _line, sql, _truth in lines]
        written = [int(line.rsplit("#", 1)[1]) for line in output.read_text().splitlines()]
        self.assertEqual(written, expected)
        self.assertEqual([line.rsplit("#", 1)[0] for line in output.read_text().splitlines()],
                         [line for line, _sql, _truth in lines])
        self.assertEqual(summary["zero_subset_labels"], expected.count(0))
        self.assertGreater(sum(truth > 0 for *_rest, truth in lines), 30)  # mostly non-empty queries
        self.assertTrue(any(0 < count < truth for count, (*_r, truth) in zip(expected, lines)))
        self.assertTrue(json.loads(output.with_suffix(".json").read_text())["full_data_labels_reproduced"] == 60)
        # pandas (NeuroCard's loader) reads the backslash records differently; PostgreSQL semantics are used
        differences = summary["rows_pandas_reads_differently"]
        self.assertGreater(differences["title"]["rows"], 0)
        self.assertGreater(differences["cast_info"]["rows"], 0)
        self.assertEqual(differences["movie_keyword"]["rows"], 0)
        bad = self.root / "queries_bad.csv"
        line, _sql, truth = lines[0]
        bad.write_text(f"{line}#{truth + 1}\n")
        with self.assertRaisesRegex(ValueError, "does not reproduce 1 of 1"):
            shift.relabel(bad, self.full, self.subset, self.split, self.root / "bad_subset.csv")

    def test_postgres_csv_reading(self) -> None:
        truth = json.loads((self.full / "_truth.json").read_text())
        for table in ("title", "cast_info", "movie_info"):
            frame = shift.read_pg_csv(self.full / f"{table}.csv", HEADERS[table])
            self.assertEqual(len(frame), len(truth[table]))
            for row, expected in zip(frame.itertuples(index=False), truth[table]):
                for value, true in zip(row, expected):
                    if true is None:
                        self.assertTrue(pd.isna(value), (table, expected))
                    else:
                        self.assertEqual(value, true, (table, expected))
        pandas_title = pd.read_csv(self.full / "title.csv", escapechar="\\")
        shifted = [row[0] for row in truth["title"] if row[4] == SHIFTING]
        self.assertTrue(shifted)
        self.assertTrue(pandas_title.loc[pandas_title["id"].isin(shifted), "production_year"].isna().all())

    def test_neurocard_sampler_lets_worker_processes_import_the_package(self) -> None:
        """Ray's workers (NeuroCard's prepare() on a cache miss) are separate processes: they
        must import NeuroCard's modules from the inherited environment (23236408)."""

        import os
        import sys as system

        shared = self.root / "neurocard_fake"
        package = shared / "neurocard"
        package.mkdir(parents=True)
        (shared / "datasets").mkdir()
        (shared / "datasets" / "job").symlink_to(self.full)
        (package / "experiments.py").write_text("JOB_LIGHT_BASE = {}\n")
        (package / "join_utils.py").write_text(
            "class Spec:\n    join_tables = ['title']\n    join_name = 'job-light-test'\n"
            "def get_join_spec(config):\n    return Spec()\n")
        (package / "datasets.py").write_text(
            "def LoadImdb(table, data_dir, use_cols, try_load_parsed):\n    return (table, data_dir, use_cols)\n")
        (package / "factorized_sampler.py").write_text(
            "import os, subprocess, sys\n"
            "import pandas as pd\n"
            "class prepare_utils:\n"
            "    @staticmethod\n"
            "    def check_required_files(spec):\n        return False\n"
            "class FactorizedSampler:\n"
            "    def __init__(self, tables, spec, rows, rng, disambiguate_column_names):\n"
            "        worker = subprocess.run([sys.executable, '-c', 'import join_utils, os; print(os.getcwd())'],\n"
            "                                capture_output=True, text=True, env=dict(os.environ))\n"
            "        if worker.returncode:\n            raise RuntimeError(worker.stderr)\n"
            "        self.worker_cwd = worker.stdout.strip()\n"
            "        import time\n"
            "        time.sleep(float(os.environ.get('FAKE_SAMPLER_SLEEP', '0')))\n"
            "        self.join_card = 5\n"
            "    def run(self):\n        return pd.DataFrame({'worker_cwd': [self.worker_cwd]})\n")
        # a stand-in for Ray: records how it was started
        (package / "ray.py").write_text(
            "import json, os\n"
            "_up = False\n"
            "def is_initialized():\n    return _up\n"
            "def init(**kwargs):\n"
            "    global _up\n    _up = True\n"
            "    open(os.environ['FAKE_RAY_LOG'], 'w').write(json.dumps(kwargs))\n")
        os.environ["FAKE_RAY_LOG"] = str(self.root / "ray_init.json")
        copy = shift.neurocard_workdir(package, self.full, self.subset, self.root / "nc_fake_subset")
        before = os.environ.get("PYTHONPATH")
        path_before = list(system.path)
        try:
            count, frame = shift.neurocard_sampler(copy, self.subset, "content", 4)
        finally:
            for name in ("datasets", "experiments", "factorized_sampler", "join_utils"):
                system.modules.pop(name, None)
            system.path[:] = path_before
        self.assertEqual(count, 5)
        self.assertEqual(Path(frame["worker_cwd"].iloc[0]).resolve(), copy.parent.resolve())
        self.assertEqual(os.environ.get("PYTHONPATH"), before)  # restored afterwards
        # Ray was started by us, node-locally, before NeuroCard's own ray.init (PREPARE 23323812 hung there)
        started = json.loads((self.root / "ray_init.json").read_text())
        self.assertEqual(started["_node_ip_address"], "127.0.0.1")
        self.assertFalse(started["include_dashboard"])
        self.assertTrue(started["_temp_dir"].startswith("/tmp/ray_ids_") and len(started["_temp_dir"]) < 40)
        system.modules.pop("ray", None)
        # a sampler that does not finish is aborted (exit 3) instead of holding the node
        script = (f"import sys; sys.path.insert(0, {str(REPO)!r})\n"
                  "from model.scripts import imdb_data_shift as shift\n"
                  f"shift.neurocard_sampler(__import__('pathlib').Path({str(copy)!r}), "
                  f"__import__('pathlib').Path({str(self.subset)!r}), 'content', 4)\n")
        import subprocess
        hung = subprocess.run([system.executable, "-c", script], capture_output=True, text=True, timeout=120,
                              env={**os.environ, "FAKE_SAMPLER_SLEEP": "60", "IMDB_SHIFT_SAMPLER_TIMEOUT": "2"})
        self.assertEqual(hung.returncode, 3, hung.stderr[-2000:])
        self.assertIn("aborting", hung.stdout)
        self.assertIn("Thread", hung.stderr)
        os.environ.pop("FAKE_RAY_LOG", None)

    def test_neurocard_workdir_links_the_package_and_reads_the_kept_csvs(self) -> None:
        shared = self.root / "neurocard_shared"
        package = shared / "neurocard"
        (package / "factorized_sampler_lib").mkdir(parents=True)
        for name in ("datasets.py", "factorized_sampler.py"):
            (package / name).write_text("# stub\n")
        (package / "__pycache__").mkdir()
        (shared / "datasets").mkdir()
        (shared / "datasets" / "job").symlink_to(self.full)
        (shared / "cache").mkdir()
        copy = shift.neurocard_workdir(package, self.full, self.subset, self.root / "nc_subset")
        self.assertEqual(copy, self.root / "nc_subset" / "neurocard")
        self.assertEqual(sorted(p.name for p in copy.iterdir()), ["datasets.py", "factorized_sampler.py", "factorized_sampler_lib"])
        self.assertTrue(all(p.is_symlink() for p in copy.iterdir()))
        self.assertEqual((copy.parent / "datasets" / "job").resolve(), self.subset.resolve())
        self.assertFalse((copy.parent / "cache").exists())
        with self.assertRaisesRegex(ValueError, "layout"):
            shift.neurocard_workdir(package, self.subset, self.subset, self.root / "nc_other")


def build_prepared(root: Path) -> dict:
    """Synthetic full data with its full preparation, a temporal split, the subset CSVs and
    the subset preparation (NeuroCard's sampler replaced by an enumeration of the join)."""

    from model.src.data.complete_domain_preparation import (
        CompleteDomainSpec, build_complete_metadata, build_manifest_payload, encode_sample_dataframe,
        preparation_stats, write_prepared_artifacts)
    from model.src.model.factorization import FactorizationConfig

    full_csv = root / "full"
    _imdb(full_csv, titles=40, seed=5)
    full_rows = _enumerate_join(full_csv)
    spec = CompleteDomainSpec(
        join_tables=shift.JOB_LIGHT_TABLES, join_root="title",
        join_keys={table: (shift.KEY[table],) for table in shift.JOB_LIGHT_TABLES},
        join_cardinality=len(full_rows), dataset_name="job_light_test", dataset_type="neurocard_full_join")
    metadata = build_complete_metadata(_frames(full_csv, MODEL_COLUMNS), spec)
    names = [column.name for column in metadata.columns]
    frame = pd.DataFrame(full_rows)[names]
    encoded = encode_sample_dataframe(frame, metadata, strict=True)
    full_prepared = root / "prepared_full"
    write_prepared_artifacts(
        prepared_directory=full_prepared,
        manifest_payload=build_manifest_payload(metadata=metadata, spec=spec, sample_rows=len(frame),
                                                neurocard_use_cols="content"),
        encoded_rows=encoded.encoded_rows,
        stats=preparation_stats(metadata=metadata, encoded_sample=encoded, factorization_config=FactorizationConfig()))
    split = root / "split"
    shift.write_split(split, shift.select(full_csv, "temporal", expected_full=len(full_rows)))
    subset_csv = root / "subset"
    shift.subset_csvs(full_csv, split, subset_csv)

    def fake_sampler(package, csv_directory, use_cols, sample_rows):
        assert use_cols == "content" and Path(csv_directory) == subset_csv
        rows = _enumerate_join(csv_directory)
        return len(rows), pd.DataFrame(rows)[names]

    subset_prepared = root / "prepared_subset"
    summary = shift.prepare(full_prepared, subset_csv, split, subset_prepared, package=root / "unused",
                            sampler=fake_sampler)
    return {"full_csv": full_csv, "full_prepared": full_prepared, "split": split, "subset_csv": subset_csv,
            "subset_prepared": subset_prepared, "summary": summary, "fake_sampler": fake_sampler, "metadata": metadata}


def fixture_config(root: Path, name: str, prepared: Path, extra: str = "") -> Path:
    """The evaluated model's training config, tiny, on a prepared directory's fixture rows."""

    text = SOURCE_CONFIG.read_text()
    for pattern, value in ((r"(?m)^  prepared_directory: .*", f"  prepared_directory: {prepared}"),
                           (r"(?m)^  sampling_mode: live", "  sampling_mode: fixture"),
                           (r"hidden_sizes: \[512, 512, 512, 512\]", "hidden_sizes: [16, 16]"),
                           (r"(?m)^  batch_size: 2048", "  batch_size: 16"),
                           (r"(?m)^  steps_per_epoch: 40000", "  steps_per_epoch: 4"),
                           (r"(?m)^  checkpoint_interval_steps: 500", "  checkpoint_interval_steps: 2"),
                           (r"(?m)^  validation_interval_steps: 250", "  validation_interval_steps: 2"),
                           (r"(?m)^  device: auto", "  device: cpu"),
                           (r"(?m)^rare_support:\n  enabled: true", "rare_support:\n  enabled: false"),
                           (r"(?m)^rare_auxiliary:\n  enabled: true", "rare_auxiliary:\n  enabled: false"),
                           (r"(?m)^  output_directory: .*", f"  output_directory: {root / 'runs' / name}")):
        text, n = re.subn(pattern, value, text)
        assert n >= 1, pattern
    if extra:
        text = text.replace("training:\n", "training:\n" + extra, 1)
    path = root / f"{name}.yaml"
    path.write_text(text)
    return path


def train(path: Path):
    from model.src.config import load_simple_yaml, validate_config
    from model.src.data.sample_sources import sample_source_from_config
    from model.src.training.resmade_trainer import train_resmade_sample_source

    config = load_simple_yaml(str(path))
    validate_config(config)
    return train_resmade_sample_source(sample_source_from_config(config), config), config


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "needs torch")
class ImdbPrepareAndFineTuneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        built = build_prepared(cls.root)
        for key, value in built.items():
            setattr(cls, key, staticmethod(value) if key == "fake_sampler" else value)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _config(self, name: str, prepared: Path, extra: str = "") -> Path:
        return fixture_config(self.root, name, prepared, extra)

    def _train(self, path: Path):
        return train(path)

    def test_subset_preparation_keeps_the_domains_and_takes_the_kept_join(self) -> None:
        full = json.loads((self.full_prepared / "manifest.json").read_text())
        subset = json.loads((self.subset_prepared / "manifest.json").read_text())
        self.assertEqual([c["domain"] for c in subset["metadata"]["columns"]],
                         [c["domain"] for c in full["metadata"]["columns"]])
        record = json.loads((self.split / "split.json").read_text())
        self.assertEqual(subset["join_cardinality"], record["full_join_cardinality_subset"])
        self.assertLess(subset["join_cardinality"], full["join_cardinality"])
        self.assertEqual(subset["metadata"]["full_join_cardinality"], float(subset["join_cardinality"]))
        self.assertNotEqual(subset["metadata"]["schema_hash"], full["metadata"]["schema_hash"])
        self.assertEqual(self.summary["subset_join_cardinality"], subset["join_cardinality"])
        self.assertEqual(json.loads((self.subset_prepared / "subset.json").read_text())["full_join_cardinality"],
                         full["join_cardinality"])

        def wrong(package, csv_directory, use_cols, sample_rows):
            count, frame = self.fake_sampler(package, csv_directory, use_cols, sample_rows)
            return count + 1, frame

        with self.assertRaisesRegex(ValueError, "NeuroCard's sampler counts"):
            shift.prepare(self.full_prepared, self.subset_csv, self.split, self.root / "prepared_wrong",
                          package=self.root / "unused", sampler=wrong)

    def test_stale_training_and_fine_tuning_from_its_weights(self) -> None:
        import torch

        from model.src.config import load_simple_yaml
        from model.src.training.resmade_trainer import _initialize_from_checkpoint, build_resmade_from_config

        stale, _ = self._train(self._config("stale", self.subset_prepared))
        payload = torch.load(stale.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["metadata"]["full_join_cardinality"], float(self.summary["subset_join_cardinality"]))
        path = self._config("finetune", self.full_prepared, f"  init_checkpoint: {stale.checkpoint_path}\n")
        tuned, config = self._train(path)
        summary = json.loads(Path(tuned.summary_path).read_text())
        self.assertEqual(summary["init_checkpoint"]["path"], str(stale.checkpoint_path))
        tuned_payload = torch.load(tuned.checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(tuned_payload["metadata"]["full_join_cardinality"], float(self.summary["full_join_cardinality"]))
        # the weights are loaded exactly
        from model.src.data.sample_sources import sample_source_from_config

        source = sample_source_from_config(load_simple_yaml(str(path)))
        model = build_resmade_from_config(source.metadata, config)
        _initialize_from_checkpoint(model, stale.checkpoint_path, source.metadata, "cpu")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, payload["model_state_dict"][key]), key)
        # a checkpoint whose domains differ is refused
        other = dict(payload)
        metadata = json.loads(json.dumps(payload["metadata"]))
        metadata["columns"][0]["domain"] = metadata["columns"][0]["domain"][:-1]
        other["metadata"] = metadata
        bad = self.root / "other_domains.pt"
        torch.save(other, bad)
        with self.assertRaisesRegex(ValueError, "value domains"):
            _initialize_from_checkpoint(build_resmade_from_config(source.metadata, config), bad, source.metadata, "cpu")


if __name__ == "__main__":
    unittest.main()
