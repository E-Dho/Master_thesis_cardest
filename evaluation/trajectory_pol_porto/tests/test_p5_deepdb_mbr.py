"""P5 DeepDB-MBR: trip-table export, query translation, bridge wiring.

The translation must ask exactly Exact-MBR's question (tested on every fixture
query without DeepDB). With TRAJ_EVAL_DEEPDB_PYTHON (a Python 3.8 DeepDB
environment) and TRAJ_EVAL_DEEPDB_SOURCE (a deepdb-public checkout at the
pinned revision) the real bridge runs through the pipeline stages.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_core.artifacts import read_predictions
from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.runner import run_seed
from evaluation.trajectory_pol_porto.traj_eval import deepdb_mbr
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap
from evaluation.trajectory_pol_porto.traj_eval.adapters.exact_mbr import ExactMbrModel, align_trip_tables
from evaluation.trajectory_pol_porto.traj_eval.workloads import load_traj_workload

from query_generation.benchmark_v3.config import load_settings

from ._fixture import MULTI, SINGLE, V3_CONFIG, write_benchmark, write_synthetic_snapshot

REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "evaluation" / "trajectory_pol_porto" / "scripts" / "deepdb_mbr_bridge.py"
SHIPPED = REPO / "evaluation" / "trajectory_pol_porto" / "configs" / "pol_v3" / "deepdb_mbr.yaml"
DEEPDB_PYTHON = os.environ.get("TRAJ_EVAL_DEEPDB_PYTHON")
DEEPDB_SOURCE = os.environ.get("TRAJ_EVAL_DEEPDB_SOURCE")
REVISION = "655ada13a043a4226a239b0c0a65bcdefef87a02"
OPERATORS = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater, "=": np.equal}


def read_table(shared: Path, layout):
    names = [column["name"] for column in layout["columns"]]
    kinds = {column["name"]: column["kind"] for column in layout["columns"]}
    with (shared / deepdb_mbr.CSV_NAME).open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    table = {}
    for index, name in enumerate(names):
        values = [row[index] for row in rows]
        table[name] = np.asarray(values) if kinds[name] == "nominal" else np.asarray(values, dtype=np.float64)
    return table


def count(table, conditions) -> int:
    mask = np.ones(len(table["trip_id"]), dtype=bool)
    for column, operator, value in conditions:
        mask &= OPERATORS[operator](table[column], value)
    return int(mask.sum())


class _Shared(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.settings = load_settings(V3_CONFIG)
        cls.snapshot = write_synthetic_snapshot(cls.root / "snapshot")
        cls.benchmark = write_benchmark(cls.root / "benchmark.jsonl")
        cls.shared = cls.root / "shared"
        cls.manifest = deepdb_mbr.export(cls.settings, cls.snapshot, cls.shared, log=lambda message: None)
        cls.layout = deepdb_mbr.read_layout(cls.shared)
        cls.rows = [json.loads(line) for line in cls.benchmark.read_text().splitlines()]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def exact_mbr(self) -> ExactMbrModel:
        data = snap.load_snapshot(self.snapshot)
        trips = snap.load_trip_index(self.snapshot)
        arrays = {f"box/{name}": trips[name] for name in ("xmin", "ymin", "xmax", "ymax", "tmin_us", "tmax_us")}
        for table_id, columns in align_trip_tables(self.settings, data, trips).items():
            for column, values in columns.items():
                arrays[f"table/{table_id}/{column}"] = values
        return ExactMbrModel(self.settings, arrays)


class TranslationTest(_Shared):
    def test_layout_is_unambiguous_for_deepdb(self) -> None:
        names = [column["name"] for column in self.layout["columns"]]
        self.assertEqual(names[0], "trip_id")
        self.assertEqual(names[-6:], list(deepdb_mbr.BOX_COLUMNS))
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertEqual(name, name.lower())
            self.assertRegex(name, r"^[a-z0-9_]+$")
        self.assertEqual(self.manifest["rows"], len(snap.load_trip_index(self.snapshot)["trip_id"]))

    def test_translation_asks_exactly_the_exact_mbr_question(self) -> None:
        model = self.exact_mbr()
        table = read_table(self.shared, self.layout)
        records = load_traj_workload(self.benchmark, "w")
        checked = 0
        for row, record in zip(self.rows, records):
            conditions, note = deepdb_mbr.translate(row, self.layout)
            status, estimate, diagnostic = model.estimate(record)
            if conditions is None:
                self.assertEqual(status, "unsupported", note)
                continue
            self.assertEqual(status, "ok")
            self.assertEqual(note, diagnostic)
            self.assertEqual(count(table, conditions), int(estimate), row["query_id"])
            checked += 1
        self.assertGreater(checked, 30)

    def test_sql_literals_round_trip_without_exponents(self) -> None:
        for row in self.rows:
            conditions, _ = deepdb_mbr.translate(row, self.layout)
            if conditions is None:
                continue
            sql = deepdb_mbr.to_sql(conditions)
            literals = re.findall(r" (?:<=|>=|<|>|=) ('[^']*'|-?[0-9.]+)(?= AND|$)", sql)
            self.assertEqual(len(literals), len(conditions), sql)
            for literal, (_, _, value) in zip(literals, conditions):
                self.assertNotIn("e", literal.lower().strip("'") if not isinstance(value, str) else "")
                parsed = literal.strip("'") if isinstance(value, str) else float(literal)
                self.assertEqual(parsed, value)
        self.assertEqual(deepdb_mbr._literal(1e-07), "0.0000001")
        self.assertEqual(float(deepdb_mbr._literal(-123456.75)), -123456.75)

    def test_epoch_seconds_match_the_snapshot_microseconds(self) -> None:
        self.assertEqual(deepdb_mbr.epoch_seconds("2020-03-29 02:30:00.000001"), 1585449000.000001)
        self.assertEqual(deepdb_mbr.epoch_seconds("1970-01-01 00:00:00"), 0.0)


class ShippedConfigTest(unittest.TestCase):
    def test_config_loads_and_commands_render(self) -> None:
        import tempfile as _tempfile

        text = SHIPPED.read_text(encoding="utf-8")
        with _tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_benchmark(root / "benchmark.jsonl")
            base = (SHIPPED.parent / "_base.yaml").read_text(encoding="utf-8")
            base = re.sub(r"/work_beegfs/\S+/benchmark\.jsonl", str(root / "benchmark.jsonl"), base)
            base = base.replace("../../../../query_generation/benchmark_v3/pol_v3_config.json", str(V3_CONFIG))
            base = base.replace("expected_query_count: 6000", f"expected_query_count: {8 * MULTI}")
            base = base.replace("expected_query_count: 2000", f"expected_query_count: {8 * SINGLE}")
            (root / "_base.yaml").write_text(base, encoding="utf-8")
            (root / "deepdb_mbr.yaml").write_text(text, encoding="utf-8")
            config = load_experiment_config(root / "deepdb_mbr.yaml")
        self.assertEqual((config.method_id, config.variant_id, config.protocol), ("deepdb_mbr", "k1_trip_box", "adapted"))
        self.assertEqual(config.seeds, (0, 1, 2))
        self.assertEqual(config.adapter_type, "external")
        self.assertTrue(config.adapter["require_smoke_before_build"])
        workload = config.workloads[0]
        from evaluation.eval_core.workload_formats import format_of

        values = {"seed": "0", "exchange_directory": "/x/exchange", "predictions_csv": "/x/p.csv",
                  "latency_csv": "/x/l.csv", **format_of(workload).placeholders(workload)}
        for key in ("evaluate_command",):
            argv = shlex.split(config.adapter[key].format(**values))
            self.assertEqual(json.loads(argv[argv.index("--select-json") + 1]), {"relation": "multi"})
            self.assertIn(REVISION, argv)
        timing = shlex.split(config.adapter["timing_commands"]["cpu_1core"].format(**values))
        self.assertEqual(timing[timing.index("--repetitions") + 1], "3")
        for stage in ("prepare", "smoke", "build"):
            self.assertIn(f"deepdb_mbr_bridge.py {stage}", config.adapter[f"{stage}_command"])


@unittest.skipUnless(DEEPDB_PYTHON and DEEPDB_SOURCE, "set TRAJ_EVAL_DEEPDB_PYTHON and TRAJ_EVAL_DEEPDB_SOURCE")
class RealDeepDbTest(_Shared):
    def bridge(self, *args: str) -> None:
        subprocess.run([DEEPDB_PYTHON, "-W", "ignore", str(BRIDGE), *args, "--source-root", DEEPDB_SOURCE,
                        "--revision", REVISION, "--shared-root", str(self.shared)],
                       check=True, capture_output=True, text=True)

    def test_compatibility_shims(self) -> None:
        """NumPy >= 1.24 ragged domains; the CCA fallback only for verified degenerate (constant, finite) pairs."""
        script = f"""
import sys, importlib.util
sys.path.insert(0, {str(DEEPDB_SOURCE)!r})
spec = importlib.util.spec_from_file_location("bridge", {str(BRIDGE)!r})
bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
bridge.install_compat(); bridge.install_compat()
import numpy as np, rspn.rspn as r, spn.algorithms.splitting.RDC as rdc
domains = r.np.asanyarray([[0.0, 1.0], np.arange(0, 5)])
assert domains.dtype == object and domains.shape == (2,), domains
assert r.np.asanyarray([[0, 1], [2, 3]]).shape == (2, 2)
rs = np.random.RandomState(0)
constant = np.ones((50, 3)); varied = rs.rand(50, 3); other = rs.rand(50, 3)
original = rdc.rdc_cca.__wrapped__
def raises(call):
    try: call()
    except (ValueError, np.linalg.LinAlgError) as error: return type(error)
    return None
def nan(value): return value != value
# 1. one side constant: upstream raises, the shim returns NaN (SPFlow: independent)
assert raises(lambda: original((0, 1, [constant, varied]))) is not None
assert nan(rdc.rdc_cca((0, 1, [constant, varied])))
# 2. the case found in the DeepDB fixture build: a 4-row slice where both variables take two values
#    in orthogonal patterns (A B B A vs. C C D D), so the centred cross-covariance is exactly zero
a, b, c, d = (np.r_[np.sin(rs.rand(4)) * 0.1, 1.0] for _ in range(4))
x_tiny, y_tiny = np.array([a, b, b, a]), np.array([c, c, d, d])
assert raises(lambda: original((0, 1, [x_tiny, y_tiny]))) is not None
assert bridge.degenerate_pair((0, 1, [x_tiny, y_tiny]))
assert nan(rdc.rdc_cca((0, 1, [x_tiny, y_tiny])))
assert rdc.rdc_cca.fallbacks == [(0, 1), (0, 1)], rdc.rdc_cca.fallbacks
assert bridge.cca_fallback_count() == 2
# 3. valid pairs are unchanged, including tiny and near-constant ones that upstream answers itself
for pair in ([varied, other], [constant + 1e-13 * rs.rand(50, 3), varied], [x_tiny, np.array([c, d, d, d])]):
    assert rdc.rdc_cca((0, 1, pair)) == original((0, 1, pair))
# 4. non-constant input with NaN or inf: upstream rejects it, so must the shim (no silent independence);
#    a constant side with a non-finite entry is not verified degenerate either
for bad in (np.nan, np.inf):
    broken = varied.copy(); broken[3, 1] = bad
    assert raises(lambda: original((0, 1, [broken, other]))) is not None
    assert raises(lambda: rdc.rdc_cca((0, 1, [broken, other]))) is not None, bad
    constant_bad = constant.copy(); constant_bad[0, 0] = bad
    assert not bridge.degenerate_pair((0, 1, [constant_bad, varied]))
    assert raises(lambda: rdc.rdc_cca((0, 1, [constant_bad, varied]))) is not None, bad
# 5. other failures re-raise (mismatched rows)
assert raises(lambda: rdc.rdc_cca((0, 1, [varied, rs.rand(40, 3)]))) is not None
assert not bridge.degenerate_pair((0, 1, [varied, other]))
assert rdc.rdc_cca.fallbacks == [(0, 1), (0, 1)]  # no further fallbacks were taken
assert not hasattr(rdc.rdc_cca.__wrapped__, "__wrapped__")  # wrapped once
print("ok")
"""
        result = subprocess.run([DEEPDB_PYTHON, "-W", "ignore", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip().splitlines()[-1:], ["ok"], result.stderr[-2000:])

    def test_pipeline_run_matches_exact_mbr_coverage(self) -> None:
        self.bridge("prepare-shared")
        common = (f"--source-root {DEEPDB_SOURCE} --revision {REVISION} --shared-root {self.shared} --seed {{seed}}")
        evaluate = (f"{DEEPDB_PYTHON} -W ignore {BRIDGE} evaluate {common} --checkpoint-directory "
                    "{exchange_directory}/ensemble --queries-jsonl {queries_jsonl} "
                    "--select-json '{workload_select_json}' --predictions {predictions_csv} --latency {latency_csv}")
        source_line = f"  checkout_path: {DEEPDB_SOURCE}\n  revision: {REVISION}\n"
        path = self.root / "deepdb_mbr.yaml"
        path.write_text(f"""schema_version: 1
experiment:
  experiment_id: traj_p5_fixture
  method_id: deepdb_mbr
  variant_id: k1
  display_name: DeepDB-MBR fixture
  protocol: adapted
  adaptation: fixture
  seeds: [0]
source:
  url: https://github.com/DataManagementLab/deepdb-public.git
{source_line}paths:
  results_root: {self.root / 'results'}
workloads:
  pol_v3_multi:
    format: traj_v3_jsonl
    queries_jsonl: {self.benchmark}
    select:
      relation: multi
    expected_query_count: {8 * MULTI}
  pol_v3_single:
    format: traj_v3_jsonl
    queries_jsonl: {self.benchmark}
    select:
      relation: single
    expected_query_count: {8 * SINGLE}
timing:
  device: cpu
  warmup_passes: 0
  repetitions: 1
resources:
  profile: test
artifacts:
  deepdb_shared: {self.shared}
adapter:
  type: external
  working_directory: {REPO}
  environment:
    PYTHONPATH: null
  require_smoke_before_build: true
  prepare_command: {DEEPDB_PYTHON} -W ignore {BRIDGE} prepare {common} --output {{exchange_directory}}/prepare_manifest.json
  smoke_command: {DEEPDB_PYTHON} -W ignore {BRIDGE} smoke {common} --output-directory {{exchange_directory}}/smoke --rows 120
  build_command: {DEEPDB_PYTHON} -W ignore {BRIDGE} build {common} --output-directory {{exchange_directory}} --sample-size 1000
  evaluate_command: {evaluate}
""", encoding="utf-8")
        config = load_experiment_config(path)
        run = run_seed(config, 0, run_id="deepdb")
        predictions = {(p.workload, p.query_id): p for p in read_predictions(run / "predictions.csv")}
        model = self.exact_mbr()
        q_errors = []
        for workload in config.workloads:
            for record in load_traj_workload(self.benchmark, workload.workload_id,
                                             select=dict(workload.select)):
                status, estimate, diagnostic = model.estimate(record)
                prediction = predictions[(workload.workload_id, record.query_id)]
                self.assertEqual(prediction.status, status, (workload.workload_id, record.query_id))
                if status == "ok":
                    self.assertEqual(prediction.diagnostic, diagnostic)
                    q_errors.append(max(prediction.estimated_cardinality, 1) / max(estimate, 1)
                                    if prediction.estimated_cardinality > estimate
                                    else max(estimate, 1) / max(prediction.estimated_cardinality, 1))
        self.assertGreater(len(q_errors), 30)
        self.assertLess(float(np.median(q_errors)), 2.0)  # SPN on all 240 trips
        artifacts = json.loads((run / "exchange" / "artifact_manifest.json").read_text())
        self.assertGreater(artifacts["serialized_bytes"], 0)
        build = json.loads((run / "exchange" / "build_stage_metrics.json").read_text())
        self.assertIsInstance(build["cca_degenerate_fallbacks"], int)
        self.assertEqual(len(build["numpy_compat"]), 2)
        summary = json.loads((run / "summary.json").read_text())
        self.assertIn("extension", summary["workloads"]["pol_v3_multi"])


if __name__ == "__main__":
    unittest.main()
