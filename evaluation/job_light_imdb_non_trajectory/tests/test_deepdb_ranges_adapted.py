from __future__ import annotations

import bisect
import importlib.util
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from evaluation.job_light_imdb_non_trajectory.joblight_eval import deepdb_ranges as adapted
from evaluation.job_light_imdb_non_trajectory.joblight_eval.config import (
    load_experiment_config,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.report import (
    aggregate_runs,
    compare_aggregates,
)
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import (
    load_workload,
    parse_csv_query,
)
from model.src.config import load_simple_yaml

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepdb_ranges_adapted_bridge.py"
SPEC = importlib.util.spec_from_file_location("deepdb_ranges_adapted_bridge", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)

EXCLUDED = {
    "title.phonetic_code", "cast_info.nr_order", "title.season_nr",
    "title.episode_nr", "title.series_years", "title.imdb_index",
}
SQL_OPERATORS = {
    "=": lambda a, b: a == b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}


def _domain(values, *, key=None, collation="C", literals=()):
    """Domain ordered by ``key`` with PostgreSQL-style recorded boundaries."""
    ordered = tuple(sorted(set(values), key=key))
    boundaries = {}
    for literal in literals:
        rank_key = (lambda value: value) if key is None else key
        boundaries[literal] = (
            sum(rank_key(value) < rank_key(literal) for value in ordered),
            sum(rank_key(value) <= rank_key(literal) for value in ordered),
        )
    return adapted.RankDomain(
        "title.phonetic_code", ordered, collation, boundaries,
        codepoint_order_verified=key is None,
    )


class _Graph:
    class Table:
        def __init__(self, name, **kwargs):
            self.table_name = name
            self.__dict__.update(kwargs)

    class SchemaGraph:
        def __init__(self):
            self.tables, self.relationships, self.table_dictionary = [], [], {}

        def add_table(self, table):
            self.tables.append(table)
            self.table_dictionary[table.table_name] = table

        def add_relationship(self, *args):
            self.relationships.append(args)


class AdaptedSchemaTests(unittest.TestCase):
    def test_only_the_six_excluded_columns_become_modeled(self):
        native = {
            f"{table}.{attribute}"
            for table, spec in adapted.NATIVE_TABLE_SPECS.items()
            for attribute in spec["attributes"]
            if attribute not in spec["irrelevant_attributes"]
        }
        self.assertEqual(set(adapted.modeled_columns()) - native, EXCLUDED)
        self.assertTrue(native <= set(adapted.modeled_columns()))

    def test_tables_attributes_and_relationships_match_upstream(self):
        schema = adapted.gen_job_light_ranges_adapted_schema("/data/{}.csv", graph_module=_Graph)
        self.assertEqual([table.table_name for table in schema.tables], list(adapted.TABLES))
        for table in schema.tables:
            native = adapted.NATIVE_TABLE_SPECS[table.table_name]
            self.assertEqual(table.attributes, native["attributes"])
            self.assertEqual(table.csv_file_location, f"/data/{table.table_name}.csv")
            self.assertTrue(set(native["no_compression"]) <= set(table.no_compression))
        self.assertEqual(
            schema.relationships,
            [(child, "movie_id", "title", "id") for child in adapted.TABLES[1:]],
        )
        title = schema.table_dictionary["title"]
        self.assertEqual(
            sorted(title.irrelevant_attributes),
            ["episode_of_id", "imdb_id", "md5sum", "title"],
        )
        self.assertIn("phonetic_code", title.no_compression)
        self.assertEqual(schema.table_dictionary["cast_info"].irrelevant_attributes,
                         ["note", "person_id", "person_role_id"])

    def test_adapted_schema_accepts_excluded_columns(self):
        query = parse_csv_query(
            "title t,cast_info ci#t.id=ci.movie_id#t.phonetic_code,=,S123,"
            "ci.nr_order,<=,3,t.season_nr,>=,2#1", "ranges", 0,
        )
        self.assertEqual(adapted.unsupported_filter_columns(query), ())
        text = parse_csv_query("title t##t.title,=,x#1", "ranges", 1)
        self.assertEqual(adapted.unsupported_filter_columns(text), ("t.title",))


class RankRewriteTests(unittest.TestCase):
    def test_rank_boundaries_for_each_operator(self):
        domain = _domain(["A", "B", "D"])
        self.assertEqual(adapted.rank_predicate(domain, "=", "B"), ("=", 1))
        self.assertEqual(adapted.rank_predicate(domain, "=", "C"), ("=", None))
        self.assertEqual(adapted.rank_predicate(domain, ">=", "B"), (">=", 1))
        self.assertEqual(adapted.rank_predicate(domain, ">", "B"), (">=", 2))
        self.assertEqual(adapted.rank_predicate(domain, ">=", "C"), (">=", 2))
        self.assertEqual(adapted.rank_predicate(domain, "<=", "B"), ("<", 2))
        self.assertEqual(adapted.rank_predicate(domain, "<", "B"), ("<", 1))
        self.assertEqual(adapted.rank_predicate(domain, "<=", "C"), ("<", 2))
        self.assertEqual(adapted.rank_predicate(domain, "<", "0"), ("<", 0))
        self.assertEqual(adapted.rank_predicate(domain, ">", "Z"), (">=", 3))

    def test_ranked_predicates_equal_sql_semantics_including_null(self):
        rng = random.Random(7)
        alphabet = "AB1?-"
        for _ in range(200):
            values = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3)))
                      for _ in range(rng.randint(1, 12))]
            column = values + [None] * rng.randint(0, 3)
            domain = _domain(v for v in column if v is not None)
            ranks = [None if v is None else domain.rank(v) for v in column]
            literal = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 3)))
            for operator, compare in SQL_OPERATORS.items():
                expected = [v is not None and compare(v, literal) for v in column]
                rank_operator, bound = adapted.rank_predicate(domain, operator, literal)
                if bound is None:
                    self.assertFalse(any(expected))
                    continue
                observed = adapted.evaluate_rank_predicates(ranks, [(rank_operator, bound)])
                self.assertEqual(observed, expected, (column, operator, literal))

    def test_non_codepoint_collation_uses_recorded_postgres_boundaries(self):
        # Emulate a linguistic collation that ignores '?' at the first level.
        def key(value):
            return (value.replace("?", ""), value)

        values = ["1995-????", "1995-1999", "1996-????", "1986-1990"]
        literals = ["1995-2000", "1995-????", "1990-????"]
        domain = _domain(values, key=key, collation="en_US.UTF-8", literals=literals)
        self.assertFalse(domain.codepoint_order_verified)
        ranks = [domain.rank(value) for value in values]
        for literal in literals:
            for operator, compare in SQL_OPERATORS.items():
                expected = [compare(key(value), key(literal)) for value in values]
                rank_operator, bound = adapted.rank_predicate(domain, operator, literal)
                if bound is None:
                    self.assertFalse(any(expected))
                    continue
                observed = adapted.evaluate_rank_predicates(ranks, [(rank_operator, bound)])
                self.assertEqual(observed, expected, (operator, literal))
        with self.assertRaises(adapted.UnsupportedLiteral):
            adapted.rank_predicate(domain, "<", "1997-2001")

    def test_rewrite_keeps_numeric_looking_string_literals_as_text(self):
        domains = {
            "title.imdb_index": adapted.RankDomain(
                "title.imdb_index", ("1", "I", "II"), "C", codepoint_order_verified=True
            )
        }
        query = parse_csv_query(
            "title t##t.imdb_index,=,1,t.production_year,>=,2004.0#5", "ranges", 0
        )
        self.assertEqual(query.filters[0].value, 1)
        result = adapted.rewrite_query(query, domains)
        self.assertFalse(result.is_empty)
        self.assertEqual(result.query.filters[0].operator, "=")
        self.assertEqual(result.query.filters[0].value, 0)
        self.assertEqual(result.query.filters[1], query.filters[1])
        self.assertEqual(result.predicates[0].literal, "1")

    def test_missing_equality_and_empty_intervals_are_explicit(self):
        domains = {"title.phonetic_code": _domain(["A", "B", "D"])}
        cases = {
            "t.phonetic_code,=,C": "missing_equality_literal",
            "t.phonetic_code,>=,D,t.phonetic_code,<=,B": "empty_rank_interval",
            "t.phonetic_code,>,D": "empty_rank_interval",
            "t.phonetic_code,<,A": "empty_rank_interval",
            "t.phonetic_code,=,A,t.phonetic_code,>=,B": "empty_rank_interval",
        }
        for filters, reason in cases.items():
            query = parse_csv_query(f"title t##{filters}#0", "ranges", 0)
            result = adapted.rewrite_query(query, domains)
            self.assertTrue(result.is_empty, filters)
            self.assertIn(reason, result.empty_reason)
        ok = parse_csv_query("title t##t.phonetic_code,>=,B,t.phonetic_code,<=,C#1", "r", 0)
        result = adapted.rewrite_query(ok, domains)
        self.assertFalse(result.is_empty)
        self.assertEqual([(f.operator, f.value) for f in result.query.filters], [(">=", 1), ("<", 2)])

    def test_rewritten_sql_is_numeric_for_deepdb(self):
        domains = {"title.phonetic_code": _domain(["A", "B", "D"])}
        query = parse_csv_query(
            "movie_companies,title#title.id=movie_companies.movie_id#"
            "title.phonetic_code,<=,C,movie_companies.company_type_id,=,1#3",
            "ranges", 0,
        )
        sql = BRIDGE.native.deepdb_compatible_query_to_sql(adapted.rewrite_query(query, domains).query)
        self.assertIn("title.phonetic_code<2", sql)
        self.assertNotIn("'", sql)

    def test_domain_payload_round_trip_and_checksum(self):
        domains = {"title.phonetic_code": _domain(["A", "B"], literals=["AA"])}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / adapted.DOMAIN_FILE
            digest = adapted.write_domains(path, domains, {"collation": "C"})
            self.assertEqual(BRIDGE.sha256_file(path), digest)
            loaded = adapted.load_domains(path)
        self.assertEqual(loaded["title.phonetic_code"].values, ("A", "B"))
        self.assertEqual(loaded["title.phonetic_code"].bisect_left("AA"), 1)

    def test_string_literals_are_collected_as_text(self):
        queries = [parse_csv_query("title t##t.imdb_index,<=,1,t.series_years,>=,1995-????#1", "r", 0)]
        literals = adapted.string_literals_by_column(queries)
        self.assertEqual(literals["title.imdb_index"], ["1"])
        self.assertEqual(literals["title.series_years"], ["1995-????"])


@unittest.skipIf(pd is None, "pandas is required")
class AdaptedDatasetTests(unittest.TestCase):
    def _domains(self, rows):
        domains = {}
        for column in adapted.RANKED_STRING_COLUMNS:
            index = BRIDGE.TITLE_COLUMNS.index(column.split(".", 1)[1])
            values = tuple(sorted({row[index] for row in rows if row[index] is not None}))
            domains[column] = adapted.RankDomain(column, values, "C", codepoint_order_verified=True)
        return domains

    def test_csv_fields_round_trip_through_deepdb_reader(self):
        values = ["\\Frag'ile\\", 'a"b', 'x\\"y', "multi\nline", "trail\\", "", "NA", "c,d", '"']
        text = "".join(
            f"{BRIDGE.format_csv_field(value)},{BRIDGE.format_csv_field(index)},"
            f"{BRIDGE.format_csv_field(None)}\n"
            for index, value in enumerate(values)
        )
        raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False,
                          na_filter=False, **BRIDGE.CSV_READ_OPTIONS)
        self.assertEqual(raw[0].tolist(), values)
        parsed = pd.read_csv(io.StringIO(text), header=None, **BRIDGE.CSV_READ_OPTIONS)
        self.assertEqual(parsed[1].tolist(), list(range(len(values))))
        self.assertTrue(parsed[2].isna().all())

    def test_ranked_title_export_is_verified_against_reference_rows(self):
        rows = [
            (1, "\\Frag'ile\\", None, 1, 2010, None, "F624", None, None, None, None, "abc"),
            (2, 'Say "hi"', "I", 7, 1999, None, "S1", 1, 1, 2, "1995-????", "def"),
            (3, "NA", "II", 7, None, None, None, 1, None, None, "1995-1999", "ghi"),
        ]
        domains = self._domains(rows)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "title.csv"
            stats = BRIDGE.write_ranked_title(rows, path, domains)
            self.assertEqual(stats["row_count"], 3)
            self.assertEqual(stats["non_null_counts"]["title.phonetic_code"], 2)
            report = BRIDGE.verify_ranked_title(path, rows, domains)
            self.assertEqual(report["verified_rows"], 3)
            deepdb_view = pd.read_csv(path, header=None, names=list(BRIDGE.TITLE_COLUMNS),
                                      **BRIDGE.CSV_READ_OPTIONS)
            self.assertEqual(deepdb_view["kind_id"].tolist(), [1, 7, 7])
            self.assertEqual(deepdb_view["phonetic_code"].tolist()[:2], [0.0, 1.0])
            self.assertTrue(pd.isna(deepdb_view["phonetic_code"].iloc[2]))
            tampered = list(rows)
            tampered[1] = tampered[1][:3] + (8,) + tampered[1][4:]
            with self.assertRaises(ValueError):
                BRIDGE.verify_ranked_title(path, tampered, domains)

    def test_unknown_value_is_rejected(self):
        rows = [(1, "t", "I", 1, 2000, None, "A1", None, None, None, None, "m")]
        domains = self._domains(rows)
        domains["title.phonetic_code"] = adapted.RankDomain(
            "title.phonetic_code", ("B",), "C", codepoint_order_verified=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "absent from its rank domain"):
                BRIDGE.write_ranked_title(rows, Path(temporary) / "title.csv", domains)


class BridgeCommandLineTests(unittest.TestCase):
    def test_every_subcommand_dispatches_to_its_stage(self):
        from unittest import mock

        base = ["--source-root", "/src", "--revision", "r"]
        out = ["--output-directory", "/out"]
        cases = {
            "prepare_run": ["prepare", *base, "--dataset-root", "/d", *out, "--shared-root", "/s"],
            "smoke_run": ["smoke", *base, *out],
            "prepare_shared": ["prepare-shared", *base, "--dataset-root", "/d", *out,
                               "--postgres-dsn", "x", "--workload", "/w.csv"],
            "validate_shared": ["validate", *base, "--shared-root", "/s", "--postgres-dsn", "x",
                                "--workload", "/w.csv", "--output", "/o.json"],
            "build": ["build", *base, "--dataset-root", "/s", *out, "--pg-host", "/sock"],
            "validate_ensemble": ["validate-ensemble", *base, "--dataset-root", "/s", *out,
                                  "--pg-host", "/sock", "--queries", "/q.csv"],
            "evaluate_workload": ["evaluate", *base, "--shared-root", "/s", "--checkpoint", "/c.pkl",
                                  "--queries", "/q.csv", "--predictions", "/p.csv", "--latency", "/l.csv"],
        }
        for function, argv in cases.items():
            with mock.patch.object(BRIDGE.native, "validate_source", return_value=Path("/src")), \
                    mock.patch.object(BRIDGE.native, "_set_seed"), \
                    mock.patch.object(BRIDGE, function, return_value={"passed": True}) as stage:
                self.assertEqual(BRIDGE.main(argv), 0, function)
                stage.assert_called_once()
        self.assertEqual(BRIDGE.DEFAULT_COLLATION, "C")


class AdaptedConfigAndReportTests(unittest.TestCase):
    def test_adapted_config_identity_is_separate_from_native(self):
        adapted_config = load_simple_yaml(ROOT / "configs" / "deepdb_job_light_ranges_adapted.yaml")
        native_config = load_simple_yaml(ROOT / "configs" / "deepdb_job_light.yaml")
        identity = adapted_config["experiment"]
        self.assertEqual(identity["method_id"], "deepdb")
        self.assertEqual(identity["variant_id"], adapted.VARIANT_ID)
        self.assertEqual(identity["display_name"], adapted.DISPLAY_NAME)
        self.assertEqual(identity["protocol"], "adapted")
        self.assertNotEqual(identity["variant_id"], native_config["experiment"]["variant_id"])
        self.assertNotIn("protocol", native_config["experiment"])
        commands = " ".join(
            str(value) for key, value in adapted_config["adapter"].items() if key.endswith("_command")
        )
        self.assertIn("deepdb_ranges_adapted_bridge.py", commands)
        self.assertIn("deepdb_ranges_adapted_shared", commands)
        self.assertNotIn("deepdb_shared ", commands + " ")
        self.assertEqual(list(adapted_config["workloads"]), ["job_light_ranges"])

    def _write_config(self, root: Path, extra: str) -> Path:
        queries = root / "q.csv"
        queries.write_text("title t##t.id,=,1#1\n", encoding="utf-8")
        path = root / "c.yaml"
        path.write_text(
            "schema_version: 1\nexperiment:\n  experiment_id: e\n  method_id: deepdb\n"
            f"  variant_id: v\n{extra}  seeds: [0]\nsource:\n  url: u\n  revision: r\n"
            f"paths:\n  results_root: {root}\nworkloads:\n  job_light_ranges:\n"
            f"    queries_csv: {queries}\nadapter:\n  type: deepdb\nartifacts:\n  x: y\n",
            encoding="utf-8",
        )
        return path

    def test_protocol_defaults_to_native_and_is_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(load_experiment_config(self._write_config(root, "")).protocol, "native")
            config = load_experiment_config(
                self._write_config(root, "  protocol: adapted\n  adaptation: extended schema\n")
            )
            self.assertEqual((config.protocol, config.adaptation), ("adapted", "extended schema"))
            with self.assertRaises(ValueError):
                load_experiment_config(self._write_config(root, "  protocol: adapted\n"))
            with self.assertRaises(ValueError):
                load_experiment_config(self._write_config(root, "  protocol: tuned\n"))

    def test_comparison_labels_adapted_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregates = []
            for name, protocol in (("native", None), ("adapted", "adapted")):
                run = root / name / "run"
                run.mkdir(parents=True)
                (run / "run_manifest.json").write_text(json.dumps({"status": "complete"}))
                stats = {"p50": 1.0, "p90": 1.0, "p95": 1.0, "p99": 1.0, "max": 1.0}
                summary = {
                    "experiment_id": "e", "method_id": "deepdb", "variant_id": name,
                    "display_name": f"DeepDB {name}", "config_hash": name, "seed": 0,
                    "workloads": {"job_light_ranges": {
                        "accuracy": {
                            "query_count": 1, "scored_query_count": 1, "coverage_fraction": 1.0,
                            "true_zero_matching_count": 0, "estimate_lt_1_count": 0,
                            "estimate_lt_0_1_count": 0, "estimate_lt_0_01_count": 0,
                            "zero_estimate_count": 0, "raw_q_error": stats,
                            "raw_q_error_true_positive": stats,
                            "smoothed_q_error_true_zero": stats, "smoothed_q_error": stats,
                        },
                        "inference": {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0,
                                      "p99_ms": 1.0, "throughput_queries_per_second": 1.0},
                    }},
                }
                if protocol:
                    summary.update(protocol=protocol, adaptation="extended schema")
                (run / "summary.json").write_text(json.dumps(summary))
                aggregate = aggregate_runs([run], root / name / "aggregate")
                self.assertEqual(aggregate["protocol"], protocol or "native")
                aggregates.append(root / name / "aggregate" / "comparison.json")
            comparison = compare_aggregates(aggregates, root / "comparison")
            self.assertEqual([row["protocol"] for row in comparison["results"]], ["native", "adapted"])
            markdown = (root / "comparison" / "comparison.md").read_text()
            self.assertIn("| deepdb | adapted | DeepDB adapted | adapted |", markdown)
            self.assertIn("| deepdb | native | DeepDB native | native |", markdown)
            self.assertIn("Protocol: **adapted**", (root / "adapted" / "aggregate" / "comparison.md").read_text())


@unittest.skipUnless(os.environ.get("JOBLIGHT_RANGES_CSV"), "set JOBLIGHT_RANGES_CSV to the upstream workload")
class UpstreamWorkloadCoverageTests(unittest.TestCase):
    def test_adapted_schema_covers_all_range_queries(self):
        queries = load_workload(os.environ["JOBLIGHT_RANGES_CSV"], "job_light_ranges")
        self.assertEqual(len(queries), 1000)
        self.assertEqual(sum(bool(adapted.unsupported_filter_columns(q)) for q in queries), 0)
        native_modeled = {
            f"{table}.{attribute}"
            for table, spec in adapted.NATIVE_TABLE_SPECS.items()
            for attribute in spec["attributes"]
            if attribute not in spec["irrelevant_attributes"]
        }

        def native_supported(query):
            aliases = {table.alias: table.name for table in query.tables}
            return all(
                f"{aliases[p.column.split('.')[0]]}.{p.column.split('.')[1]}" in native_modeled
                for p in query.filters
            )

        self.assertEqual(sum(native_supported(q) for q in queries), 197)


if __name__ == "__main__":
    unittest.main()
