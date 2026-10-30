"""P4: planner-estimate adapter. SQL rendering without a server; the v3_sql
variant end to end on PostgreSQL + PostGIS when TRAJ_EVAL_TEST_PG is set."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Dict

from evaluation.eval_core.artifacts import read_predictions
from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.runner import run_seed
from evaluation.eval_core.workload_formats import load_queries
from evaluation.trajectory_pol_porto.traj_eval.adapters.mobilitydb import PlannerSql, parallel_divisor, plan_rows
from evaluation.trajectory_pol_porto.traj_eval.workloads import load_traj_workload

from query_generation.benchmark_v3 import synthetic

from ._fixture import MULTI, SINGLE, V3_CONFIG, synthetic_settings, write_benchmark

PG = os.environ.get("TRAJ_EVAL_TEST_PG")
STBOX = {
    "xt": "stbox 'SRID={srid};STBOX XT((({xmin},{ymin}),({xmax},{ymax})),[{tmin}, {tmax}])'",
    "x": "stbox 'SRID={srid};STBOX X(({xmin},{ymin}),({xmax},{ymax}))'",
    "t": "stbox 'STBOX T([{tmin}, {tmax}])'",
}


class RenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        path = write_benchmark(Path(cls.tmp.name) / "benchmark.jsonl")
        cls.records = load_traj_workload(path, "w")
        cls.settings = synthetic_settings()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def find(self, category, predicate=lambda r: True):
        return next(r for r in self.records if r.category == category and predicate(r))

    def test_v3_sql_plans_the_label_query(self) -> None:
        renderer = PlannerSql(self.settings, "v3_sql", {})
        multi = self.find("spatio_temporal.range.multi")
        sql, note = renderer.render(multi)
        self.assertEqual(note, "")
        self.assertTrue(sql.startswith("SELECT t.trip_id\nFROM"))
        self.assertTrue(sql.endswith("GROUP BY t.trip_id"))
        self.assertIn("GREATEST(0.0::float8", sql)  # the exact Liang-Barsky test is kept
        single = self.find("spatial.range.single")
        sql, _ = renderer.render(single)
        self.assertTrue(sql.startswith("SELECT 1\nFROM pol.segments s"))

    def test_segment_stbox_replaces_only_segment_space_and_time(self) -> None:
        renderer = PlannerSql(self.settings, "segment_stbox", STBOX)
        record = self.find("spatio_temporal.range.multi", lambda r: r.addon_count > 0)
        sql, note = renderer.render(record)
        self.assertEqual(note, "approx:segment_stbox")
        self.assertIn("s.segment_tgeom && stbox 'SRID=26916;STBOX XT(((", sql)
        self.assertNotIn("GREATEST(0.0::float8", sql)
        self.assertEqual(sql.count("&& stbox"), 1)
        standard = self.find("standard.range.multi")
        sql, note = renderer.render(standard)
        self.assertEqual(note, "")  # no box involved: the exact v3 SQL
        unbounded = self.find("temporal.unbounded.multi", lambda r: any(
            p["table"] == "segments" for p in r.predicates))
        sql, _ = renderer.render(unbounded)
        self.assertIn("STBOX T([", sql)  # a missing bound falls back to the data domain

    def test_trip_stbox_is_trip_level_and_refuses_what_a_box_cannot_express(self) -> None:
        renderer = PlannerSql(self.settings, "trip_stbox", STBOX)
        record = self.find("spatial.range.multi", lambda r: all(
            p["table"] != "segments" or p["kind"] == "spatial" for p in r.predicates))
        sql, note = renderer.render(record)
        self.assertEqual(note, "approx:trip_stbox")
        self.assertIn("t.trip_tgeom && stbox 'SRID=26916;STBOX X(", sql)
        self.assertNotIn("pol.segments", sql)
        self.assertEqual(renderer.render(self.find("spatial.range.single"))[0], None)
        segment_attribute = next((r for r in self.records if r.relation == "multi" and any(
            p["table"] == "segments" and p["kind"] not in ("spatial", "spatio_temporal", "temporal")
            for p in r.predicates)), None)
        self.assertIsNotNone(segment_attribute)
        sql, reason = renderer.render(segment_attribute)
        self.assertIsNone(sql)
        self.assertIn("segment-level attribute predicate", reason)

    def test_missing_template_and_unknown_variant_are_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "adapter.variant"):
            PlannerSql(self.settings, "nope", {})
        renderer = PlannerSql(self.settings, "segment_stbox", {"x": STBOX["x"]})
        with self.assertRaisesRegex(ValueError, "adapter.stbox.xt"):
            renderer.render(self.find("spatio_temporal.range.multi"))

    def test_plan_rows_reads_the_aggregate_and_its_input(self) -> None:
        payload = [{"Plan": {"Node Type": "Aggregate", "Plan Rows": 12, "Plans": [
            {"Node Type": "Hash Join", "Plan Rows": 345}]}}]
        self.assertEqual(plan_rows(payload), (12.0, "Aggregate", 345.0, "Hash Join", None))
        self.assertEqual(plan_rows(json.dumps([{"Plan": {"Node Type": "Seq Scan", "Plan Rows": 7}}])),
                         (7.0, "Seq Scan", None, None, None))

    def test_parallel_plans_report_total_input_rows(self) -> None:
        # Finalize Group <- Gather Merge (2 workers) <- Partial Aggregate <- Sort <- Parallel Hash Join
        payload = [{"Plan": {"Node Type": "Group", "Plan Rows": 900, "Plans": [
            {"Node Type": "Gather Merge", "Plan Rows": 800, "Workers Planned": 2, "Plans": [
                {"Node Type": "Aggregate", "Partial Mode": "Partial", "Plan Rows": 400, "Plans": [
                    {"Node Type": "Sort", "Plan Rows": 1000, "Plans": [
                        {"Node Type": "Hash Join", "Plan Rows": 1000}]}]}]}]}}]
        rows, node, input_rows, input_node, workers = plan_rows(payload)
        self.assertEqual((rows, node, input_node, workers), (900.0, "Group", "Hash Join", 2))
        self.assertAlmostEqual(input_rows, 1000 * 2.4)  # 2 workers + leader 1 - 0.6
        self.assertAlmostEqual(plan_rows(payload, leader_participation=False)[2], 2000.0)
        self.assertAlmostEqual(parallel_divisor(4), 4.0)  # the leader no longer contributes

    def test_literal_checks_cover_each_template_with_the_srid(self) -> None:
        self.assertEqual(PlannerSql(self.settings, "v3_sql", {}).literal_checks(), [])
        for variant, column in (("segment_stbox", "s.segment_tgeom"), ("trip_stbox", "t.trip_tgeom")):
            checks = dict(PlannerSql(self.settings, variant, STBOX).literal_checks())
            self.assertEqual(sorted(checks), ["t", "x", "xt"])
            for template, sql in checks.items():
                self.assertTrue(sql.startswith(f"SELECT {column} && stbox '"), sql)
                self.assertTrue(sql.endswith("LIMIT 1"), sql)
                if template != "t":
                    self.assertIn("SRID=26916;", sql)


SHIPPED = Path(__file__).resolve().parents[1] / "configs" / "pol_v3"
VARIANT_KEYS = ("variant_id:", "display_name:", "adaptation:", "variant:", "# P4:", "# one seed")


def shipped_configs(root: Path, dsn: str = "host=/nonexistent dbname=none") -> Dict[str, Path]:
    """The shipped mobilitydb_* configs with only the cluster paths and the DSN swapped."""
    import re

    target = root / "shipped"
    target.mkdir(exist_ok=True)
    out = {}
    for path in [SHIPPED / "_base.yaml", *sorted(SHIPPED.glob("mobilitydb_*.yaml"))]:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"/work_beegfs/\S+/benchmark\.jsonl", str(root / "benchmark.jsonl"), text)
        text = text.replace("../../../../query_generation/benchmark_v3/pol_v3_config.json", str(V3_CONFIG))
        text = text.replace("expected_query_count: 6000", f"expected_query_count: {8 * MULTI}")
        text = text.replace("expected_query_count: 2000", f"expected_query_count: {8 * SINGLE}")
        text = text.replace("results_root: ../../results", f"results_root: {root / 'shipped_results'}")
        text = re.sub(r"^  dsn: .*$", f"  dsn: {dsn}", text, flags=re.M)
        (target / path.name).write_text(text, encoding="utf-8")
        if path.name.startswith("mobilitydb_"):
            out[path.stem.removeprefix("mobilitydb_")] = target / path.name
    return out


class ShippedConfigTest(unittest.TestCase):
    def test_the_three_variants_differ_only_in_their_variant_keys(self) -> None:
        texts = {p.name: [line for line in p.read_text().splitlines() if not line.strip().startswith(VARIANT_KEYS)]
                 for p in SHIPPED.glob("mobilitydb_*.yaml")}
        self.assertEqual(sorted(texts), ["mobilitydb_segment_stbox.yaml", "mobilitydb_trip_stbox.yaml",
                                         "mobilitydb_v3_sql.yaml"])
        first = next(iter(texts.values()))
        for name, lines in texts.items():
            self.assertEqual(lines, first, name)

    def test_shipped_configs_load_and_their_templates_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = load_traj_workload(write_benchmark(root / "benchmark.jsonl"), "w")
            for variant, path in shipped_configs(root).items():
                config = load_experiment_config(path)
                self.assertEqual((config.method_id, config.variant_id, config.adapter["variant"]),
                                 ("mobilitydb", variant, variant))
                self.assertEqual(config.seeds, (0,))
                self.assertEqual(config.adapter["session_timezone"], "UTC")
                renderer = PlannerSql(synthetic_settings(), variant, dict(config.adapter["stbox"]))
                for record in records:  # every template has every placeholder it needs
                    sql, note = renderer.render(record)
                    if sql is not None and "stbox" in note:
                        self.assertNotIn("{", sql)
                        if "STBOX T" in sql or "STBOX XT" in sql:
                            self.assertIn("+00", sql)


SCHEMA = """
CREATE EXTENSION postgis;
CREATE SCHEMA pol;
CREATE TABLE pol.agents (agent_id bigint PRIMARY KEY, age double precision NOT NULL,
  "educationLevel" text NOT NULL, interest text NOT NULL, joviality double precision NOT NULL,
  family_size integer NOT NULL);
CREATE TABLE pol.trips (trip_id bigint PRIMARY KEY, agent_id bigint NOT NULL REFERENCES pol.agents,
  start_time timestamp NOT NULL, end_time timestamp NOT NULL, num_of_segments integer NOT NULL);
CREATE TABLE pol.segments (trip_id bigint NOT NULL REFERENCES pol.trips, segment_idx integer NOT NULL,
  s_x double precision NOT NULL, s_y double precision NOT NULL, e_x double precision NOT NULL,
  e_y double precision NOT NULL, t_s timestamp NOT NULL, t_e timestamp NOT NULL,
  segment_geom geometry(LineString, 26916), PRIMARY KEY (trip_id, segment_idx));
"""


def _load_database(psycopg, dsn: str) -> None:
    _, rows = synthetic.build(synthetic_settings())
    with psycopg.connect(dsn) as connection:
        connection.execute(SCHEMA)
        with connection.cursor() as cursor:
            for table, columns, items in (
                ("pol.agents", ["agent_id", "age", "educationLevel", "interest", "joviality", "family_size"], rows["agents"]),
                ("pol.trips", ["trip_id", "agent_id", "start_time", "end_time", "num_of_segments"],
                 [{**r, "start_time": r["start"], "end_time": r["end"]} for r in rows["trips"]]),
                ("pol.segments", ["trip_id", "segment_idx", "s_x", "s_y", "e_x", "e_y", "t_s", "t_e"],
                 [{**r, "s_x": r["sx"], "s_y": r["sy"], "e_x": r["ex"], "e_y": r["ey"], "t_s": r["ts"], "t_e": r["te"]}
                  for r in rows["segments"]]),
            ):
                quoted = ", ".join(f'"{c}"' for c in columns)
                with cursor.copy(f"COPY {table} ({quoted}) FROM STDIN") as copy:
                    for item in items:
                        copy.write_row([item[c] for c in columns])
            cursor.execute("UPDATE pol.segments SET segment_geom = ST_SetSRID(ST_MakeLine("
                           "ST_MakePoint(s_x, s_y), ST_MakePoint(e_x, e_y)), 26916)")
        connection.commit()
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute("ANALYZE")


def _config(root: Path, dsn: str, variant: str = "v3_sql", **adapter) -> Path:
    extra = "".join(f"  {key}: {value}\n" for key, value in adapter.items())
    path = root / f"mobilitydb_{variant}_{uuid.uuid4().hex[:6]}.yaml"
    path.write_text(f"""schema_version: 1
experiment:
  experiment_id: traj_p4_fixture
  method_id: mobilitydb
  variant_id: {variant}
  display_name: planner {variant}
  seeds: [0]
source:
  url: https://github.com/E-Dho/Master_thesis_cardest.git
  installed_version: working tree
paths:
  results_root: {root / 'results'}
workloads:
  pol_v3_multi:
    format: traj_v3_jsonl
    queries_jsonl: {root / 'benchmark.jsonl'}
    select:
      relation: multi
    expected_query_count: {8 * MULTI}
  pol_v3_single:
    format: traj_v3_jsonl
    queries_jsonl: {root / 'benchmark.jsonl'}
    select:
      relation: single
    expected_query_count: {8 * SINGLE}
timing:
  device: cpu
  warmup_passes: 0
  repetitions: 2
resources:
  profile: test
artifacts:
  database: test
adapter:
  type: traj_mobilitydb
  variant: {variant}
  benchmark_config: {V3_CONFIG}
  dsn: {dsn}
{extra}""", encoding="utf-8")
    return path


@unittest.skipUnless(PG, "set TRAJ_EVAL_TEST_PG to run the PostgreSQL + PostGIS tests")
class PlannerEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import psycopg

        cls.psycopg = psycopg
        cls.dbname = f"traj_p4_test_{uuid.uuid4().hex[:8]}"
        with psycopg.connect(f"{PG} dbname=postgres", autocommit=True) as admin:
            admin.execute(f"CREATE DATABASE {cls.dbname}")
        cls.dsn = f"{PG} dbname={cls.dbname}"
        _load_database(psycopg, cls.dsn)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        with cls.psycopg.connect(f"{PG} dbname=postgres", autocommit=True) as admin:
            admin.execute(f"DROP DATABASE {cls.dbname}")

    def test_v3_sql_runs_every_stage_and_records_the_planner(self) -> None:
        config = load_experiment_config(_config(self.root, self.dsn))
        run = run_seed(config, 0, run_id="v3sql")
        predictions = read_predictions(run / "predictions.csv")
        self.assertTrue(predictions)
        for prediction in predictions:
            self.assertEqual(prediction.status, "ok", prediction)
            self.assertGreaterEqual(prediction.estimated_cardinality, 1.0)
            self.assertEqual(prediction.diagnostic, "")
        details = list(csv.DictReader((run / "planner_details.csv").open()))
        self.assertEqual(len(details), len(predictions))
        multi = [d for d in details if d["workload"] == "pol_v3_multi"]
        self.assertTrue(all(d["top_node"] in ("Aggregate", "Group", "Unique") for d in multi), multi[:3])
        self.assertTrue(all(d["input_plan_rows"] for d in multi))
        identity = json.loads((run / "database_identity.json").read_text())
        self.assertIn("postgis", identity["extensions"])
        self.assertEqual(identity["stbox_literal_checks"], [])
        self.assertEqual(len(identity["statistics_fingerprint"]), 64)
        self.assertFalse(identity["analyzed_by_this_run"])
        artifacts = json.loads((run / "build_metrics.json").read_text())["artifacts"]
        self.assertGreater(artifacts["serialized_bytes"], 0)
        self.assertEqual(artifacts["variant"], "v3_sql")

    def test_shipped_v3_sql_config_runs(self) -> None:
        config = load_experiment_config(shipped_configs(self.root, self.dsn)["v3_sql"])
        run = run_seed(config, 0, run_id="shipped")
        summary = json.loads((run / "summary.json").read_text())
        self.assertIn("extension", summary["workloads"]["pol_v3_multi"])
        identity = json.loads((run / "database_identity.json").read_text())
        self.assertEqual(identity["settings"]["TimeZone"], "UTC")
        from evaluation.eval_core.report import aggregate_runs, compare_aggregates

        aggregate = aggregate_runs([run], self.root / "agg")  # one seed, no database needed
        self.assertIn("extension", aggregate["workloads"]["pol_v3_single"])
        compare_aggregates([self.root / "agg" / "comparison.json"], self.root / "compare")
        self.assertTrue((self.root / "compare" / "trajectory_summary_v1.md").exists())

    def test_changed_statistics_stop_the_run(self) -> None:
        from evaluation.eval_core.runner import initialize_staged_run, run_stage

        config = load_experiment_config(_config(self.root, self.dsn))
        run = initialize_staged_run(config, 0, run_id="stats")
        run_stage(config, 0, run, "prepare")
        with self.psycopg.connect(self.dsn, autocommit=True) as connection:
            connection.execute("ALTER TABLE pol.segments ALTER COLUMN s_x SET STATISTICS 7")
            connection.execute("ANALYZE pol.segments")
        try:
            with self.assertRaisesRegex(ValueError, "statistics changed between prepare and build"):
                run_stage(config, 0, run, "build")
        finally:
            with self.psycopg.connect(self.dsn, autocommit=True) as connection:
                connection.execute("ALTER TABLE pol.segments ALTER COLUMN s_x SET STATISTICS -1")
                connection.execute("ANALYZE pol.segments")


if __name__ == "__main__":
    unittest.main()
