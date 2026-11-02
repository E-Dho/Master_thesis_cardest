"""P8b MSCN: featurisation, published model structure, training and the pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation.eval_core.artifacts import read_predictions
from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.runner import run_seed
from evaluation.trajectory_pol_porto.traj_eval.adapters import mscn
from evaluation.trajectory_pol_porto.traj_eval.workloads import load_traj_workload

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import generate_category, load_excluded_hashes
from query_generation.benchmark_v3.merge import merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend

from ._fixture import V3_CONFIG, synthetic_data, synthetic_settings, write_benchmark, write_synthetic_snapshot
from .test_p3_adapters import _config

RECIPE = "\n    epochs: 20\n    batch_size: 64\n    sample_bits: 100\n    hidden_size: 32"


def _training_set(root: Path, benchmark: Path) -> Path:
    """A disjoint v3 workload on the same synthetic data (another seed, fixture excluded)."""
    settings = load_settings(V3_CONFIG, single_count=10, multi_count=20)
    data, rows = synthetic.build(synthetic_settings())
    pools = synthetic.pools_from(settings, rows)
    backend = NumpyBackend(settings, data)
    hashes, sources = load_excluded_hashes([benchmark])
    for category in settings.categories():
        generate_category(settings, pools, category, base_seed=123, count_fn=backend.count,
                          output_dir=root / "train", progress_interval=0, log=lambda message: None,
                          excluded_hashes=hashes, excluded_sources=sources)
    rows = merge(settings, root / "train", [benchmark])["rows"]
    path = root / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class _Root(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.benchmark = write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")
        cls.training = _training_set(cls.root, cls.benchmark)
        cls.settings = load_settings(V3_CONFIG)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()


class FeaturizerTest(_Root):
    def setUp(self) -> None:
        data = synthetic_data()
        self.samples = mscn.draw_samples(data, list(self.settings.base["tables"]), 50, seed=0)
        self.featurizer = mscn.Featurizer(self.settings, self.samples, 50)
        self.records = load_traj_workload(self.benchmark, "w")

    def test_bitmaps_are_the_exact_v3_masks_on_the_samples(self) -> None:
        backend = NumpyBackend(self.settings, self.samples)
        for record in self.records:
            samples, predicates, joins = self.featurizer.encode(list(record.tables), list(record.predicates))
            self.assertEqual(samples.shape, (len(record.tables), 3 + 50))
            for row, table_id in enumerate(record.tables):
                expected = backend.table_mask(table_id, list(record.predicates)).astype(np.float32)
                np.testing.assert_array_equal(samples[row, 3:3 + expected.size], expected)
                self.assertEqual(samples[row, :3].sum(), 1.0)
            self.assertEqual(predicates.shape[0], max(1, len(record.predicates)))
            if record.relation == "single":
                self.assertEqual(joins.shape, (1, self.featurizer.join_features))
                self.assertEqual(joins[0, self.featurizer.joins.index(mscn.NO_JOIN)], 1.0)
            else:
                self.assertEqual(joins[:, self.featurizer.joins.index(mscn.NO_JOIN)].sum(), 0.0)

    def test_predicate_vectors_hold_normalised_bounds_and_presence(self) -> None:
        for record in self.records:
            for predicate in record.predicates:
                vector = self.featurizer.predicate_vector(predicate)
                values = vector[-12:-6]
                present = vector[-6:]
                self.assertTrue(np.all((values >= 0) & (values <= 1)))
                self.assertTrue(set(np.unique(present)) <= {0.0, 1.0})
                np.testing.assert_array_equal(values[present == 0], 0.0)
                kind = predicate["kind"]
                if kind in ("spatial", "spatio_temporal"):
                    np.testing.assert_array_equal(present[2:], 1.0)
                if kind == "scalar_range":
                    np.testing.assert_array_equal(present, [1, 1, 0, 0, 0, 0])
                if kind == "nominal_eq":
                    self.assertEqual(present.sum(), 0.0)
                    start = len(self.featurizer.keys) + len(mscn.KINDS)
                    self.assertEqual(vector[start:start + len(self.featurizer.nominal)].sum(), 1.0)

    def test_model_is_the_published_set_convolution(self) -> None:
        torch = mscn._torch()
        network = mscn.make_model(self.featurizer, 16)
        self.assertEqual(sorted({name.split(".")[0] for name, _ in network.named_parameters()}),
                         ["join_mlp1", "join_mlp2", "out_mlp1", "out_mlp2", "predicate_mlp1", "predicate_mlp2",
                          "sample_mlp1", "sample_mlp2"])
        record = next(r for r in self.records if r.relation == "multi" and len(r.predicates) > 1)
        encoded = self.featurizer.encode(list(record.tables), list(record.predicates))
        short = self.featurizer.encode(["trips"], [])
        model = mscn.MscnModel(self.featurizer, network, 0.0, 5.0)
        alone = model.predict([encoded])
        padded = model.predict([encoded, short])  # padding the batch must not change an estimate
        self.assertAlmostEqual(float(alone[0]), float(padded[0]), places=4)
        self.assertTrue(np.all((padded >= 1.0 - 1e-9) & (padded <= np.exp(5.0) + 1e-6)))
        self.assertIsInstance(torch.__version__, str)


class PipelineTest(_Root):
    def config(self, name: str, **extra) -> Path:
        return _config(self.root, "mscn", "traj_mscn", name, training_jsonl=self.training,
                       validation_jsonl=self.training, recipe=RECIPE, **extra)

    def test_run_answers_everything_and_reports_training(self) -> None:
        config = load_experiment_config(self.config("both"))
        run = run_seed(config, 0, run_id="mscn")
        predictions = read_predictions(run / "predictions.csv")
        self.assertTrue(predictions)
        self.assertTrue(all(p.status == "ok" and p.estimated_cardinality >= 1.0 for p in predictions))
        summary = json.loads((run / "summary.json").read_text())
        self.assertIn("extension", summary["workloads"]["pol_v3_multi"])
        meta = json.loads((run / "model" / "model_meta.json").read_text())
        self.assertEqual(meta["training"]["training_queries"], len(self.training.read_text().splitlines()))
        self.assertIn("validation_q_error", meta["training"])
        self.assertLess(meta["training"]["training_q_error"]["p50"], 3.0)
        artifacts = json.loads((run / "build_metrics.json").read_text())["artifacts"]
        self.assertGreater(artifacts["serialized_bytes"], 0)
        self.assertIsInstance(artifacts["parameter_count"], int)
        again = run_seed(config, 0, run_id="mscn_again")  # seeded: identical estimates
        self.assertEqual([p.estimated_cardinality for p in predictions],
                         [p.estimated_cardinality for p in read_predictions(again / "predictions.csv")])

    def test_multi_only_variant_rejects_single_queries(self) -> None:
        config = load_experiment_config(self.config("multi", training_relations="[multi]"))
        run = run_seed(config, 0, run_id="multi")
        predictions = read_predictions(run / "predictions.csv")
        single = [p for p in predictions if p.workload == "pol_v3_single"]
        self.assertTrue(single and all(p.status == "unsupported" for p in single))
        meta = json.loads((run / "model" / "model_meta.json").read_text())
        rows = [json.loads(line) for line in self.training.read_text().splitlines()]
        self.assertEqual(meta["training"]["training_queries"],
                         sum(row["category"]["relation"] == "multi" for row in rows))

    def test_training_set_overlapping_the_test_set_is_refused(self) -> None:
        leaky = self.root / "leaky.jsonl"
        leaky.write_text(self.training.read_text() + self.benchmark.read_text().splitlines()[0] + "\n")
        path = _config(self.root, "mscn", "traj_mscn", "leaky", training_jsonl=leaky, recipe=RECIPE)
        with self.assertRaisesRegex(ValueError, "also occur in an evaluated workload"):
            run_seed(load_experiment_config(path), 0, run_id="leaky")


@unittest.skipUnless(hasattr(os, "sched_setaffinity"), "needs Linux CPU affinity")
class TimingTest(_Root):
    def test_time_stage_on_one_pinned_core(self) -> None:
        from evaluation.eval_core import timing_guard as guard
        from evaluation.eval_core.timing import run_timing_stage

        path = _config(self.root, "mscn", "traj_mscn", "timed", training_jsonl=self.training, recipe=RECIPE)
        path.write_text(path.read_text().replace(
            "  repetitions: 2\n", "  repetitions: 1\n  timing_warmup_passes: 1\n  timing_repetitions: 2\n"
            "  estimate_consistency:\n    mode: deterministic\n    relative_tolerance: 1.0e-4\n"
            "    absolute_tolerance: 1.0\n"))
        config = load_experiment_config(path)
        run = run_seed(config, 0, run_id="timed")
        spec = self.root / "timing_hardware.json"
        spec.write_text(json.dumps({"schema_version": 1, "profiles": {
            "cpu_1core": {"cpu_model": guard.cpu_model()["model_name"]}}}))
        original = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(guard.thread_siblings(min(original))) & original)
        existing = {child["pid"] for child in guard.child_processes()}
        children = guard.child_processes
        try:
            with mock.patch.dict(os.environ, {**guard.profile_environment("cpu_1core"),
                                              guard.ALLOW_MISSING_THREADPOOLCTL_ENV: "1",
                                              guard.HARDWARE_SPEC_ENV: str(spec)}), \
                    mock.patch.object(guard, "child_processes", lambda *a, **k: [
                        c for c in children(*a, **k) if c["pid"] not in existing]):
                directory = run_timing_stage(config, 0, run, "cpu_1core")
        finally:
            os.sched_setaffinity(0, original)
        summary = json.loads((directory / "timing_summary.json").read_text())
        self.assertTrue(summary["reportable"], summary.get("failures"))
        self.assertTrue(summary["estimate_consistency"]["passed"])
        self.assertEqual((summary["warmup_passes"], summary["repetitions"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
