"""The numpy label backend of generate.py: labelling on the shared snapshot."""

import json
import tempfile
import unittest
from pathlib import Path

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import binding, generate_category, main, run_paths
from query_generation.benchmark_v3.merge import GateError, merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.query_generator import Category

CONFIG = Path(__file__).with_name("pol_v3_config.json")
QUIET = dict(progress_interval=0, log=lambda message: None)
SINGLE, MULTI = 3, 4
SQL_BINDING_KEYS = {"excluded_sha256", "state_version", "semantics_version", "config_sha256",
                    "pools_sha256", "category", "count", "base_seed"}


class NumpyLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluation.trajectory_pol_porto.tests._fixture import write_synthetic_snapshot

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        # the grid of this run differs from the snapshot's (full) grid on purpose
        cls.settings = load_settings(CONFIG, single_count=SINGLE, multi_count=MULTI)
        cls.data, rows = synthetic.build(load_settings(CONFIG, single_count=6, multi_count=8))
        cls.pools = synthetic.pools_from(cls.settings, rows)
        cls.pools_path = cls.root / "pools.json"
        cls.pools.save(cls.pools_path)
        cls.snapshot = write_synthetic_snapshot(cls.root / "snapshot", cls.data)
        cls.reference = NumpyBackend(cls.settings, cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def generate(self, out: Path, *extra: str) -> None:
        for category in self.settings.categories():
            code = main(["--config", str(CONFIG), "--pools", str(self.pools_path), "--category", category.key,
                         "--output-dir", str(out), "--base-seed", "9", "--single-count", str(SINGLE),
                         "--multi-count", str(MULTI), "--progress-interval", "0", "--backend", "numpy",
                         "--snapshot", str(self.snapshot), *extra])
            self.assertEqual(code, 0)

    def test_cli_labels_on_the_snapshot_and_records_the_source(self):
        out = self.root / "numpy_run"
        self.generate(out)
        result = merge(self.settings, out)
        manifest = __import__("hashlib").sha256((self.snapshot / "manifest.json").read_bytes()).hexdigest()
        self.assertEqual(result["summary"]["label_backend"], "numpy_snapshot")
        self.assertEqual(result["summary"]["label_snapshot_manifest_sha256"], manifest)
        for row in result["rows"]:
            self.assertEqual(row["cardinality"], self.reference.count(row))
            self.assertEqual(row["label_backend"], "numpy_snapshot")
        summary = json.loads(run_paths(out, Category.parse("spatial.range.multi"))["summary"].read_text())
        self.assertEqual(summary["label_source"]["label_backend"], "numpy_snapshot")

    def test_same_queries_as_the_count_function_path(self):
        """Backend choice changes provenance only: the accepted queries and labels are identical."""
        numpy_dir, direct_dir = self.root / "same_numpy", self.root / "same_direct"
        self.generate(numpy_dir)
        for category in self.settings.categories():
            generate_category(self.settings, self.pools, category, base_seed=9, count_fn=self.reference.count,
                              output_dir=direct_dir, **QUIET)
        strip = lambda rows: [(r["slot_id"], r["predicate_hash"], r["cardinality"]) for r in rows]
        self.assertEqual(strip(merge(self.settings, numpy_dir)["rows"]),
                         strip(merge(self.settings, direct_dir)["rows"]))

    def test_sql_binding_is_unchanged_and_backends_never_mix(self):
        category = Category.parse("temporal.range.multi")
        self.assertEqual(set(binding(self.settings, self.pools, category, 9)), SQL_BINDING_KEYS)
        out = self.root / "mixed"
        generate_category(self.settings, self.pools, category, base_seed=9, count_fn=self.reference.count,
                          output_dir=out, **QUIET)  # the SQL-path binding (no label source)
        with self.assertRaisesRegex(SystemExit, "label_backend"):
            main(["--config", str(CONFIG), "--pools", str(self.pools_path), "--category", category.key,
                  "--output-dir", str(out), "--base-seed", "9", "--single-count", str(SINGLE),
                  "--multi-count", str(MULTI), "--progress-interval", "0", "--backend", "numpy",
                  "--snapshot", str(self.snapshot)])

    def test_merge_refuses_rows_from_two_backends(self):
        out = self.root / "two_backends"
        self.generate(out)
        path = run_paths(out, Category.parse("standard.range.single"))["accepted"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop("label_backend")
            row.pop("label_snapshot_manifest_sha256")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(GateError, "label_backend"):
            merge(self.settings, out)

    def test_numpy_backend_needs_a_snapshot_of_this_config(self):
        with self.assertRaisesRegex(SystemExit, "--snapshot"):
            main(["--config", str(CONFIG), "--pools", str(self.pools_path), "--category", "temporal.range.multi",
                  "--output-dir", str(self.root / "nosnap"), "--base-seed", "9", "--backend", "numpy"])
        bad = self.root / "bad_snapshot"
        bad.mkdir()
        manifest = json.loads((self.snapshot / "manifest.json").read_text())
        (bad / "manifest.json").write_text(json.dumps({**manifest, "config_sha256": "0" * 64}))
        with self.assertRaisesRegex(ValueError, "different v3 config"):
            main(["--config", str(CONFIG), "--pools", str(self.pools_path), "--category", "temporal.range.multi",
                  "--output-dir", str(self.root / "badsnap"), "--base-seed", "9", "--backend", "numpy",
                  "--snapshot", str(bad)])


if __name__ == "__main__":
    unittest.main()
