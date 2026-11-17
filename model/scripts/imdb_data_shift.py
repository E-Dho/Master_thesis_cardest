"""Data-shift ablation on JOB-light-ranges (IMDB): the own model trained on 80 % of the titles.

Decided 2026-11-17 as the IMDB counterpart of the trajectory ablation (branch
model/data-shift): a stale model trained on 80 % of the data is evaluated against
the labels of the current (100 %) data; variants refresh its join cardinality,
fine-tune it on the full data, or evaluate it on the 80 % data (in-distribution
control). JOB-light is a star around ``title`` (every query joins ``title``), so
the split drops whole titles together with the rows of the five child tables
that reference them (``movie_id``):

- ``random80``: 80 % of the titles, uniformly (seed 0);
- ``temporal80``: the earliest 80 % by ``production_year`` (NULL years last, as
  ``ORDER BY production_year`` in PostgreSQL; ties by id).

Subcommands (all refuse existing outputs):

``select``      the kept titles, the per-table row counts and both full-outer-join
                cardinalities; the full one must equal NeuroCard's constant for the
                six JOB-light tables (2,128,877,229,383), which checks the formula.
``subset-csv``  the six CSVs of the kept data, each record copied byte for byte
                (quoted fields, escapes and embedded newlines included) and the
                output re-read with pandas against the expected rows.
``relabel``     the JOB-light-ranges queries counted exactly on the kept data
                (in-distribution control). The counter is first checked on the full
                data: every one of the 1,000 labels must be reproduced.
``prepare``     the model preparation of the kept data: the full preparation's
                manifest (columns and value domains unchanged, so fine-tuning can
                load the stale weights), the kept data's join cardinality from
                NeuroCard's sampler (checked against ``select``), a validation
                sample encoded strictly against the full domains, and a private
                NeuroCard working directory (``datasets/job`` -> the kept CSVs,
                its own ``cache``), because NeuroCard reads its data tables and
                caches its join-count tables relative to the working directory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

JOB_LIGHT_TABLES = ("cast_info", "movie_companies", "movie_info", "movie_keyword", "title", "movie_info_idx")
CHILD_TABLES = tuple(table for table in JOB_LIGHT_TABLES if table != "title")
KEY = {table: ("id" if table == "title" else "movie_id") for table in JOB_LIGHT_TABLES}
# datasets.JoinOrderBenchmark.TRUE_FULL_OUTER_CARDINALITY of NeuroCard for these six tables
NEUROCARD_FULL_OUTER_CARDINALITY = 2128877229383
MODES = ("random", "temporal")
FRACTION = 0.8
SPLIT_VERSION = 1
SPLIT_FILE, KEPT_FILE, ALL_FILE = "split.json", "kept_title_ids.npy", "all_title_ids.npy"
CSV_OPTIONS = {"escapechar": "\\", "low_memory": False}
NEUROCARD_SKIP = {"cache", "cache_fair", "datasets", "__pycache__"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh(output: Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; use a new directory")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    return partial


def _keys(csv_directory: Path, table: str) -> np.ndarray:
    import pandas as pd

    frame = pd.read_csv(Path(csv_directory) / f"{table}.csv", usecols=[KEY[table]], **CSV_OPTIONS)
    values = frame[KEY[table]]
    if values.isna().any():
        raise ValueError(f"{table}.{KEY[table]} has NULL values")
    return values.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------- join cardinality


def full_outer_join_cardinality(title_ids: np.ndarray, child_keys: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Rows of the JOB-light full outer join (root ``title``, children on ``movie_id``).

    A title contributes the product over the children of max(its matching rows, 1);
    a child row whose movie_id is no title is a row of its own (its join to
    ``title.id`` fails, so it joins no other child row either).
    """

    order = np.argsort(title_ids)
    sorted_ids = title_ids[order]
    if sorted_ids.size and np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("title ids are not unique")
    per_title = np.ones(sorted_ids.size, dtype=np.int64)
    as_float = np.ones(sorted_ids.size, dtype=np.float64)
    orphans = {}
    for table, keys in child_keys.items():
        position = np.searchsorted(sorted_ids, keys)
        position[position >= sorted_ids.size] = 0
        matched = (sorted_ids[position] == keys) if sorted_ids.size else np.zeros(keys.size, dtype=bool)
        counts = np.bincount(position[matched], minlength=sorted_ids.size).astype(np.int64)
        per_title *= np.maximum(counts, 1)
        as_float *= np.maximum(counts, 1)
        orphans[table] = int(np.count_nonzero(~matched))
    total = int(per_title.sum()) + sum(orphans.values())
    if not np.isclose(float(as_float.sum()) + sum(orphans.values()), float(total), rtol=1e-9):
        raise OverflowError("join cardinality overflowed int64")
    return {"rows": total, "orphan_child_rows": orphans}


# ---------------------------------------------------------------- select


def select(csv_directory: Path, mode: str, *, fraction: float = FRACTION, seed: int = 0,
           expected_full: Optional[int] = NEUROCARD_FULL_OUTER_CARDINALITY) -> Dict[str, Any]:
    import pandas as pd

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    csv_directory = Path(csv_directory)
    title = pd.read_csv(csv_directory / "title.csv", usecols=["id", "production_year"], **CSV_OPTIONS)
    ids = title["id"].to_numpy(dtype=np.int64)
    year = title["production_year"].to_numpy(dtype=np.float64)
    total = int(ids.size)
    keep_count = int(round(fraction * total))
    record: Dict[str, Any] = {"split_version": SPLIT_VERSION, "mode": mode, "fraction": fraction}
    kept_mask = np.zeros(total, dtype=bool)
    if mode == "random":
        order = np.random.default_rng(seed).permutation(total)
        record["seed"] = seed
    else:
        missing = np.isnan(year)
        # production_year ascending, NULL years last (PostgreSQL's ORDER BY ... ASC), ties by id
        order = np.lexsort((ids, np.where(missing, 0.0, year), missing))
        last, first_dropped = order[keep_count - 1], order[keep_count] if keep_count < total else None
        record.update(
            order="production_year ASC NULLS LAST, id",
            null_production_year_titles=int(missing.sum()),
            null_production_year_titles_kept=int(missing[order[:keep_count]].sum()),
            last_kept_production_year=None if missing[last] else float(year[last]),
            first_dropped_production_year=(None if first_dropped is None or missing[first_dropped]
                                           else float(year[first_dropped])),
        )
    kept_mask[order[:keep_count]] = True
    kept_ids = np.sort(ids[kept_mask])
    child_keys = {table: _keys(csv_directory, table) for table in CHILD_TABLES}
    full = full_outer_join_cardinality(ids, child_keys)
    if expected_full is not None and full["rows"] != int(expected_full):
        raise ValueError(f"the full outer join has {full['rows']} rows by this count, NeuroCard's constant is "
                         f"{expected_full}: not the JOB-light data or not NeuroCard's join semantics")
    dropped = np.sort(ids[~kept_mask])
    rows_total = {"title": total, **{table: int(keys.size) for table, keys in child_keys.items()}}
    rows_kept = {"title": int(kept_ids.size)}
    kept_child_keys = {}
    for table, keys in child_keys.items():
        keep = ~np.isin(keys, dropped)
        rows_kept[table] = int(keep.sum())
        kept_child_keys[table] = keys[keep]
    subset = full_outer_join_cardinality(kept_ids, kept_child_keys)
    record.update(
        titles_total=total, titles_kept=int(kept_ids.size),
        rows_total=rows_total, rows_kept=rows_kept,
        full_join_cardinality_full=full["rows"], full_join_cardinality_subset=subset["rows"],
        orphan_child_rows_full=full["orphan_child_rows"], orphan_child_rows_subset=subset["orphan_child_rows"],
        csv_directory=str(csv_directory), title_csv_sha256=_sha256(csv_directory / "title.csv"),
    )
    return {"record": record, "kept_title_ids": kept_ids, "all_title_ids": ids}


def write_split(output: Path, selection: Dict[str, Any]) -> Dict[str, Any]:
    partial = _fresh(output)
    np.save(partial / KEPT_FILE, np.asarray(selection["kept_title_ids"], dtype=np.int64))
    np.save(partial / ALL_FILE, np.asarray(selection["all_title_ids"], dtype=np.int64))
    record = dict(selection["record"])
    record["kept_title_ids_sha256"] = _sha256(partial / KEPT_FILE)
    record["all_title_ids_sha256"] = _sha256(partial / ALL_FILE)
    record["created_at_utc"] = _now()
    (partial / SPLIT_FILE).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return record


def read_split(split: Path) -> Dict[str, Any]:
    split = Path(split)
    record = json.loads((split / SPLIT_FILE).read_text(encoding="utf-8"))
    for file, key in ((KEPT_FILE, "kept_title_ids_sha256"), (ALL_FILE, "all_title_ids_sha256")):
        if _sha256(split / file) != record[key]:
            raise ValueError(f"{split / file} does not match its split record")
    return {"record": record, "kept": np.load(split / KEPT_FILE), "all": np.load(split / ALL_FILE)}


# ---------------------------------------------------------------- subset CSVs


def _raw_records(path: Path) -> Iterator[Tuple[List[str], str]]:
    """Yield (parsed fields, exact source text) per CSV record, header first.

    The csv module pulls one physical line at a time and more only for a quoted
    field that spans lines, so the lines consumed for a record are its text.
    """

    with Path(path).open("r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        consumed: List[str] = []

        def lines() -> Iterator[str]:
            for line in handle:
                consumed.append(line)
                yield line

        reader = csv.reader(lines(), escapechar="\\", quotechar='"', doublequote=True, strict=True)
        for fields in reader:
            text = "".join(consumed)
            consumed.clear()
            yield fields, text


def _column_digests(path: Path, key: str, keep: Optional[Callable[[np.ndarray], np.ndarray]],
                   chunk_rows: int = 1_000_000) -> Tuple[int, Tuple[str, ...], Tuple[str, ...]]:
    """Row count, header and one digest per column of a CSV as pandas parses it (as text),
    after keeping the rows ``keep`` selects by key (streamed in chunks)."""

    import pandas as pd

    digests: Optional[List[Any]] = None
    columns: Tuple[str, ...] = ()
    rows = 0
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_rows, **CSV_OPTIONS):
        if keep is not None:
            chunk = chunk[keep(chunk[key].astype(np.int64).to_numpy())]
        if digests is None:
            columns = tuple(str(column) for column in chunk.columns)
            digests = [hashlib.sha256() for _ in columns]
        rows += len(chunk)
        for digest, column in zip(digests, columns):
            values = chunk[column].tolist()
            if values:  # each value ends with a separator, so chunk boundaries do not matter
                digest.update(("\x1e".join(values) + "\x1e").encode("utf-8", "surrogateescape"))
    return rows, columns, tuple(digest.hexdigest() for digest in digests or [])


def subset_csvs(csv_directory: Path, split: Path, output: Path) -> Dict[str, Any]:
    import pandas as pd

    csv_directory = Path(csv_directory)
    loaded = read_split(split)
    record = loaded["record"]
    if _sha256(csv_directory / "title.csv") != record["title_csv_sha256"]:
        raise ValueError("title.csv is not the one the split was selected from")
    kept = set(int(value) for value in loaded["kept"])
    dropped_sorted = np.setdiff1d(loaded["all"], loaded["kept"])
    dropped = set(int(value) for value in dropped_sorted)
    partial = _fresh(output)
    summary: Dict[str, Any] = {"tables": {}}
    for table in JOB_LIGHT_TABLES:
        source = csv_directory / f"{table}.csv"
        target = partial / f"{table}.csv"
        records = _raw_records(source)
        header, header_text = next(records)
        key_index = header.index(KEY[table])
        written = read = 0
        with target.open("w", encoding="utf-8", errors="surrogateescape", newline="") as out:
            out.write(header_text)
            for fields, text in records:
                read += 1
                key = int(fields[key_index])
                if (key in kept) if table == "title" else (key not in dropped):
                    out.write(text)
                    written += 1
        expected_rows = record["rows_kept"][table]
        if read != record["rows_total"][table] or written != expected_rows:
            raise ValueError(f"{table}: {read} records read / {written} written, the split expects "
                             f"{record['rows_total'][table]} / {expected_rows} (CSV record boundaries differ from pandas)")
        # the output must parse with pandas to exactly the kept rows of the source, every column, in order
        source_digest = _column_digests(source, KEY[table], lambda keys: np.isin(keys, loaded["kept"])
                                        if table == "title" else ~np.isin(keys, dropped_sorted))
        if source_digest != _column_digests(target, KEY[table], None):
            raise ValueError(f"{table}: the written CSV does not parse to the kept rows of {source}")
        summary["tables"][table] = {"rows_total": read, "rows_kept": written, "sha256": _sha256(target)}
    summary.update(split_directory=str(split), source_directory=str(csv_directory),
                   kept_title_ids_sha256=record["kept_title_ids_sha256"], created_at_utc=_now())
    (partial / "subset_csv.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return summary


# ---------------------------------------------------------------- PostgreSQL CSV semantics


def _pg_fields(handle: Iterator[str], line: str) -> List[Optional[str]]:
    """One record as PostgreSQL's COPY ... (FORMAT CSV, NULL '', ESCAPE '\\') reads it.

    The benchmark database (scripts/postgres_admin.py) was loaded this way, and its
    labels follow it: a quote starts or ends a quoted section anywhere in a field;
    inside quotes a backslash escapes a following quote or backslash (elsewhere it is
    a literal character, unlike pandas' ``escapechar``); only an unquoted empty field
    is NULL. A quoted section may span lines (``handle`` supplies them).
    """

    fields: List[Optional[str]] = []
    buffer: List[str] = []
    saw_quote = in_quote = False
    i = 0
    while True:
        if i >= len(line):
            if not in_quote:
                break
            nxt = next(handle, None)
            if nxt is None:
                raise ValueError("unterminated quoted field at the end of the file")
            line, i = nxt, 0
            continue
        c = line[i]
        if in_quote:
            if c == "\\" and i + 1 < len(line) and line[i + 1] in ('"', "\\"):
                buffer.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                in_quote = False
            else:
                buffer.append(c)
            i += 1
            continue
        if c == '"':
            in_quote = saw_quote = True
        elif c == ",":
            fields.append(None if not saw_quote and not buffer else "".join(buffer))
            buffer, saw_quote = [], False
        elif c in "\r\n":
            break
        else:
            buffer.append(c)
        i += 1
    fields.append(None if not saw_quote and not buffer else "".join(buffer))
    return fields


def read_pg_csv(path: Path, columns: Sequence[str]) -> Any:
    """The named columns of a JOB CSV as PostgreSQL loads it (see _pg_fields), typed:
    a column whose non-NULL values all parse as numbers is numeric, otherwise text."""

    import pandas as pd

    with Path(path).open("r", encoding="utf-8", errors="surrogateescape", newline="") as handle:
        header = _pg_fields(handle, next(handle))
        indexes = [header.index(column) for column in columns]
        width = len(header)
        values: List[List[Optional[str]]] = [[] for _ in columns]
        for line in handle:
            if '"' not in line:  # no quoted section: commas split, a backslash is literal
                fields = line.rstrip("\r\n").split(",")
                fields = [None if field == "" else field for field in fields]
            else:
                fields = _pg_fields(handle, line)
            if len(fields) != width:
                raise ValueError(f"{path}: a record has {len(fields)} fields, the header {width}")
            for target, index in zip(values, indexes):
                target.append(fields[index])
    frame = {}
    for column, raw in zip(columns, values):
        series = pd.Series(raw, dtype=object)
        try:
            frame[column] = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError):
            frame[column] = series
    return pd.DataFrame(frame)


# ---------------------------------------------------------------- exact counts


class JobLightCounter:
    """Exact COUNT(*) of JOB-light(-ranges) queries: inner joins of title with child
    tables on movie_id, conjunctive filters, SQL NULL semantics (a comparison with
    NULL is false). Strings compare in code-point order. The CSVs are read as the
    benchmark's PostgreSQL database loaded them (read_pg_csv), not as pandas with
    ``escapechar`` reads them (NeuroCard's loader): the two differ on records whose
    unquoted text ends in a backslash, which shifts pandas' later columns."""

    def __init__(self, csv_directory: Path, columns: Dict[str, set], *,
                 expected_rows: Optional[Dict[str, int]] = None) -> None:
        self.csv_directory = Path(csv_directory)
        self.frames: Dict[str, Any] = {}
        for table in JOB_LIGHT_TABLES:
            usecols = sorted({KEY[table], *columns.get(table, set())})
            self.frames[table] = read_pg_csv(self.csv_directory / f"{table}.csv", usecols)
            if expected_rows is not None and len(self.frames[table]) != int(expected_rows[table]):
                raise ValueError(f"{table}: {len(self.frames[table])} records as PostgreSQL reads them, "
                                 f"{expected_rows[table]} as pandas reads them")
        title_ids = self.frames["title"]["id"].to_numpy(dtype=np.int64)
        self.order = np.argsort(title_ids)
        self.sorted_ids = title_ids[self.order]
        self.title_count = title_ids.size
        self.position: Dict[str, np.ndarray] = {}
        self.matched: Dict[str, np.ndarray] = {}
        for table in CHILD_TABLES:
            keys = self.frames[table]["movie_id"].to_numpy(dtype=np.int64)
            position = np.searchsorted(self.sorted_ids, keys)
            position[position >= self.sorted_ids.size] = 0
            self.position[table] = position
            self.matched[table] = self.sorted_ids[position] == keys if self.sorted_ids.size else np.zeros(keys.size, bool)
        self._counts: Dict[Tuple[str, Tuple[Any, ...]], np.ndarray] = {}

    @staticmethod
    def _mask(series: Any, op: str, literal: Any) -> np.ndarray:
        present = series.notna().to_numpy()
        values = series[present]
        if op == "=":
            hit = values == literal
        elif op == "<":
            hit = values < literal
        elif op == ">":
            hit = values > literal
        elif op == "<=":
            hit = values <= literal
        elif op == ">=":
            hit = values >= literal
        else:
            raise ValueError(f"unsupported operator {op!r}")
        mask = np.zeros(len(series), dtype=bool)
        mask[np.flatnonzero(present)] = hit.to_numpy(dtype=bool)
        return mask

    def _table_mask(self, table: str, predicates: Sequence[Tuple[str, str, Any]]) -> np.ndarray:
        frame = self.frames[table]
        mask = np.ones(len(frame), dtype=bool)
        for column, op, literal in predicates:
            mask &= self._mask(frame[column], op, literal)
        return mask

    def _per_title(self, table: str, predicates: Tuple[Tuple[str, str, Any], ...]) -> np.ndarray:
        cache_key = (table, predicates)
        if cache_key not in self._counts:
            if table == "title":
                counts = self._table_mask(table, predicates)[self.order].astype(np.int64)
            else:
                mask = self._table_mask(table, predicates) & self.matched[table]
                counts = np.bincount(self.position[table][mask], minlength=self.title_count).astype(np.int64)
            self._counts[cache_key] = counts
        return self._counts[cache_key]

    def count(self, tables: set, predicates: Sequence[Tuple[str, str, Any]]) -> int:
        if "title" not in tables:
            raise ValueError("every JOB-light query joins title")
        by_table: Dict[str, List[Tuple[str, str, Any]]] = {table: [] for table in tables}
        for column, op, literal in predicates:
            table, name = column.split(":", 1)
            by_table[table].append((name, op, literal))
        product = self._per_title("title", tuple(sorted(by_table["title"], key=repr))).copy()
        as_float = product.astype(np.float64)
        for table in sorted(tables - {"title"}):
            counts = self._per_title(table, tuple(sorted(by_table[table], key=repr)))
            product *= counts
            as_float *= counts
        total = int(product.sum())
        if not np.isclose(float(as_float.sum()), float(total), rtol=1e-9):
            raise OverflowError("count overflowed int64")
        return total


def _predicate_fields(predicate: Any) -> Tuple[str, str, Any]:
    names = {field.name for field in dataclasses.fields(predicate)}
    column = predicate.column
    op = getattr(predicate, "op", None) if "op" in names else getattr(predicate, "operator", None)
    literal = getattr(predicate, "literal", None) if "literal" in names else getattr(predicate, "value", None)
    return column, str(getattr(op, "value", op)), literal


def load_queries(path: Path) -> List[Tuple[str, set, List[Tuple[str, str, Any]], int]]:
    from model.scripts.evaluate_job_light_queries import parse_query

    parsed = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tables, predicates, truth = parse_query(line)
        parsed.append((line, set(tables), [_predicate_fields(p) for p in predicates], int(truth)))
    return parsed


def pandas_parse_differences(csv_directory: Path, columns: Dict[str, set]) -> Dict[str, Any]:
    """Per table, the rows whose query columns pandas (``escapechar``, as NeuroCard loads
    the data) reads differently from PostgreSQL: count and up to five keys."""

    import pandas as pd

    out: Dict[str, Any] = {}
    for table in JOB_LIGHT_TABLES:
        usecols = sorted({KEY[table], *columns.get(table, set())})
        pg = read_pg_csv(Path(csv_directory) / f"{table}.csv", usecols)
        pandas_frame = pd.read_csv(Path(csv_directory) / f"{table}.csv", usecols=usecols, **CSV_OPTIONS)
        if len(pg) != len(pandas_frame):
            out[table] = {"rows": None, "note": f"{len(pg)} vs {len(pandas_frame)} records"}
            continue
        differs = np.zeros(len(pg), dtype=bool)
        for column in usecols:
            a, b = pg[column].astype(object), pandas_frame[column].astype(object)
            both_null = a.isna().to_numpy() & b.isna().to_numpy()
            equal = np.array([x == y or (isinstance(x, (int, float)) and isinstance(y, (int, float)) and float(x) == float(y))
                              for x, y in zip(a.tolist(), b.tolist())], dtype=bool)
            differs |= ~(both_null | equal)
        keys = pg[KEY[table]].to_numpy()[differs][:5].tolist()
        out[table] = {"rows": int(differs.sum()), "example_keys": [int(k) if not isinstance(k, str) else k for k in keys]}
    return out


def relabel(queries: Path, full_csv: Path, subset_csv: Path, split: Path, output: Path) -> Dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    loaded = read_split(split)
    subset_record = json.loads((Path(subset_csv) / "subset_csv.json").read_text(encoding="utf-8"))
    if subset_record["kept_title_ids_sha256"] != loaded["record"]["kept_title_ids_sha256"]:
        raise ValueError("the subset CSVs belong to another split")
    parsed = load_queries(queries)
    columns: Dict[str, set] = {}
    for _line, _tables, predicates, _truth in parsed:
        for column, _op, _literal in predicates:
            table, name = column.split(":", 1)
            columns.setdefault(table, set()).add(name)
    full = JobLightCounter(full_csv, columns, expected_rows=loaded["record"]["rows_total"])
    parse_differences = pandas_parse_differences(full_csv, columns)
    mismatches = [(index, truth, full.count(tables, predicates))
                  for index, (_line, tables, predicates, truth) in enumerate(parsed)]
    mismatches = [item for item in mismatches if item[1] != item[2]]
    if mismatches:
        raise ValueError(f"the counter does not reproduce {len(mismatches)} of {len(parsed)} labels on the full data; "
                         f"first (query index, label, count): {mismatches[:5]}")
    del full
    counter = JobLightCounter(subset_csv, columns, expected_rows=loaded["record"]["rows_kept"])
    lines, counts = [], []
    for line, tables, predicates, truth in parsed:
        count = counter.count(tables, predicates)
        if count > truth:
            raise ValueError(f"a query counts more rows on the subset ({count}) than on the full data ({truth})")
        head, _old = line.rsplit("#", 1)
        lines.append(f"{head}#{count}")
        counts.append(count)
    partial = output.with_name(output.name + ".partial")
    partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts_array = np.asarray(counts, dtype=np.float64)
    truths = np.asarray([truth for *_rest, truth in parsed], dtype=np.float64)
    summary = {
        "queries": str(queries), "queries_sha256": _sha256(Path(queries)), "query_count": len(parsed),
        "full_data_labels_reproduced": len(parsed),
        "csv_reading": "PostgreSQL COPY (FORMAT CSV, NULL '', ESCAPE '\\\\') semantics, as the benchmark database",
        "rows_pandas_reads_differently": parse_differences,
        "zero_subset_labels": int(np.count_nonzero(counts_array == 0)),
        "subset_to_full_label_ratio": {q: float(np.percentile(counts_array / truths, q)) for q in (1, 10, 50, 90, 99)}
        if np.all(truths > 0) else None,
        "subset_csv": str(subset_csv), "kept_title_ids_sha256": loaded["record"]["kept_title_ids_sha256"],
        "output": str(output), "created_at_utc": _now(),
    }
    partial.rename(output)
    summary["output_sha256"] = _sha256(output)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- preparation


def neurocard_workdir(shared_package: Path, full_csv: Path, subset_csv: Path, output: Path) -> Path:
    """A NeuroCard working directory of its own for the kept data.

    NeuroCard's factorized sampler reads its data tables from ``datasets/job`` and
    writes its join-count tables to ``cache/<join name>`` relative to the working
    directory (the package's parent), so the shared checkout would mix the full
    and the kept data. The package entries are linked, ``datasets/job`` points
    at the kept CSVs and the cache starts empty. Returns the package path.
    """

    shared_package = Path(shared_package).resolve()
    shared_data = (shared_package.parent / "datasets" / "job").resolve()
    if shared_data != Path(full_csv).resolve():
        raise ValueError(f"the shared NeuroCard working directory reads {shared_data}, not {full_csv}; "
                         "its layout is not the one this copy mirrors")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    package = output / shared_package.name
    package.mkdir(parents=True)
    for entry in sorted(shared_package.iterdir()):
        if entry.name in NEUROCARD_SKIP:
            continue
        (package / entry.name).symlink_to(entry)
    (output / "datasets").mkdir()
    (output / "datasets" / "job").symlink_to(Path(subset_csv).resolve())
    return package


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def neurocard_sampler(package: Path, csv_directory: Path, use_cols: str, sample_rows: int) -> Tuple[int, Any]:
    """NeuroCard's join cardinality of the data under ``package``'s working directory and a validation sample."""

    from model.src.data.neurocard_schema import scoped_neurocard_prepare_cache_bypass

    package = Path(package)
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    with _pushd(package.parent):
        import datasets  # type: ignore
        import experiments  # type: ignore
        import factorized_sampler  # type: ignore
        import join_utils  # type: ignore

        spec = join_utils.get_join_spec(experiments.JOB_LIGHT_BASE)
        tables = [datasets.LoadImdb(table, data_dir=str(csv_directory) + "/", use_cols=use_cols, try_load_parsed=True)
                  for table in spec.join_tables]
        with scoped_neurocard_prepare_cache_bypass(factorized_sampler, spec):
            sampler = factorized_sampler.FactorizedSampler(tables, spec, sample_rows, rng=np.random.default_rng(0),
                                                           disambiguate_column_names=True)
        return int(sampler.join_card), sampler.run()


def prepare(full_prepared: Path, csv_directory: Path, split: Path, output: Path, *, package: Path,
            sample_rows: int = 2048,
            sampler: Callable[[Path, Path, str, int], Tuple[int, Any]] = neurocard_sampler) -> Dict[str, Any]:
    from model.src.data.complete_domain_preparation import (
        encode_sample_dataframe, source_csv_fingerprints, validate_prepared_manifest, write_prepared_artifacts)
    from model.src.data.schema import ModelMetadata

    full_prepared, csv_directory, output = Path(full_prepared), Path(csv_directory), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists")
    loaded = read_split(split)
    record = loaded["record"]
    subset_record = json.loads((csv_directory / "subset_csv.json").read_text(encoding="utf-8"))
    if subset_record["kept_title_ids_sha256"] != record["kept_title_ids_sha256"]:
        raise ValueError("the subset CSVs belong to another split")
    for table, entry in subset_record["tables"].items():
        if _sha256(csv_directory / f"{table}.csv") != entry["sha256"]:
            raise ValueError(f"{table}.csv changed since subset-csv")
    validate_prepared_manifest(full_prepared)
    manifest = json.loads((full_prepared / "manifest.json").read_text(encoding="utf-8"))
    metadata = ModelMetadata.from_json_dict(manifest["metadata"])
    if int(manifest["join_cardinality"]) != int(record["full_join_cardinality_full"]):
        raise ValueError(f"the full preparation has join cardinality {manifest['join_cardinality']}, the split "
                         f"counts {record['full_join_cardinality_full']}: not the same data")
    use_cols = manifest.get("neurocard_use_cols", "simple")  # the projection the full preparation was built with
    join_card, frame = sampler(Path(package), csv_directory, use_cols, sample_rows)
    if join_card != int(record["full_join_cardinality_subset"]):
        raise ValueError(f"NeuroCard's sampler counts {join_card} join rows on the kept data, the split "
                         f"{record['full_join_cardinality_subset']}")
    encoded = encode_sample_dataframe(frame, metadata, strict=True)  # the kept data within the full domains
    subset_metadata = dataclasses.replace(metadata, full_join_cardinality=float(join_card), schema_hash=None)
    subset_metadata_json = subset_metadata.to_json_dict()
    fingerprints = source_csv_fingerprints([csv_directory / f"{table}.csv" for table in JOB_LIGHT_TABLES])
    shift = {
        "split": record, "split_directory": str(split), "subset_csv": str(csv_directory),
        "source_prepared_directory": str(full_prepared), "source_manifest_sha256": _sha256(full_prepared / "manifest.json"),
        "source_schema_hash": metadata.schema_hash or metadata.stable_schema_hash(),
        "full_join_cardinality": int(manifest["join_cardinality"]), "subset_join_cardinality": join_card,
        "neurocard_package": str(package),
        "domains": "unchanged: the full preparation's (values of the dropped titles included)",
        "created_at_utc": _now(),
    }
    new_manifest = dict(manifest)
    new_manifest.update(join_cardinality=join_card, metadata=subset_metadata_json, sample_rows=int(encoded.encoded_rows.shape[0]),
                        source_csv_fingerprints=fingerprints, data_shift_subset=shift)
    stats_path = full_prepared / "preparation_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats.update(sample_row_count=int(encoded.encoded_rows.shape[0]), sample_encoding_ood_values=len(encoded.issues),
                 source_csv_fingerprints=fingerprints, schema_hash=subset_metadata_json["schema_hash"],
                 data_shift_subset=shift)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    write_prepared_artifacts(prepared_directory=partial, manifest_payload=new_manifest,
                             encoded_rows=encoded.encoded_rows, stats=stats)
    (partial / "join_counts").mkdir(exist_ok=True)
    (partial / "indexes").mkdir(exist_ok=True)
    summary = {
        "full_join_cardinality": int(manifest["join_cardinality"]), "subset_join_cardinality": join_card,
        "split_mode": record["mode"], "kept_title_ids_sha256": record["kept_title_ids_sha256"],
        "schema_hash": subset_metadata_json["schema_hash"], "prepared_directory": str(output),
        "neurocard_package": str(package),
    }
    (partial / "subset.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    partial.rename(output)
    return summary


# ---------------------------------------------------------------- CLI


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--csv-directory", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--fraction", type=float, default=FRACTION)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("subset-csv")
    p.add_argument("--csv-directory", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("relabel")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--full-csv", type=Path, required=True)
    p.add_argument("--subset-csv", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--full-prepared", type=Path, required=True)
    p.add_argument("--full-csv", type=Path, required=True)
    p.add_argument("--subset-csv", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--shared-neurocard", type=Path, required=True, help="the shared NeuroCard package (datasets.py)")
    p.add_argument("--neurocard-workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-rows", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.command == "select":
        result = write_split(args.output, select(args.csv_directory, args.mode, fraction=args.fraction, seed=args.seed))
    elif args.command == "subset-csv":
        result = subset_csvs(args.csv_directory, args.split, args.output)
    elif args.command == "relabel":
        result = relabel(args.queries, args.full_csv, args.subset_csv, args.split, args.output)
    else:
        package = neurocard_workdir(args.shared_neurocard, args.full_csv, args.subset_csv, args.neurocard_workdir)
        result = prepare(args.full_prepared, args.subset_csv, args.split, args.output, package=package,
                         sample_rows=args.sample_rows)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
