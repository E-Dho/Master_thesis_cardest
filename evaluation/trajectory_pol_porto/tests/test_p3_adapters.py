"""P3 estimators through the real pipeline stages on the synthetic snapshot."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_core.artifacts import read_predictions
from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.report import aggregate_runs, compare_aggregates
from evaluation.eval_core.runner import run_seed
from evaluation.eval_core.workload_formats import load_queries
from evaluation.trajectory_pol_porto.traj_eval import snapshot as snap
from evaluation.trajectory_pol_porto.traj_eval.adapters.base import query_row
from evaluation.trajectory_pol_porto.traj_eval.adapters.sampling import SampleModel, draw_sample

from ._fixture import MULTI, SINGLE, V3_CONFIG, synthetic_settings, write_benchmark, write_synthetic_snapshot


def _config(root: Path, method: str, adapter_type: str, variant: str, seeds="[0]", **adapter) -> Path:
    extra = "".join(f"  {key}: {value}\n" for key, value in adapter.items())
    path = root / f"{method}_{variant}.yaml"
    path.write_text(f"""schema_version: 1
experiment:
  experiment_id: traj_p3_fixture
  method_id: {method}
  variant_id: {variant}
  display_name: {method} {variant}
  seeds: {seeds}
source:
  url: https://github.com/E-Dho/Master_thesis_cardest.git
  revision: working-tree
paths:
  results_root: {root / 'results'}
workloads:
  pol_v3_multi:
    format: traj_v3_jsonl
    queries_jsonl: {root / 'benchmark.jsonl'}
    select:
      relation: multi
    expected_query_count: {8 * MULTI}
    benchmark_config: {V3_CONFIG}
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
  primary_profile: cpu_1core
  profiles: [cpu_1core]
resources:
  profile: test
artifacts:
  snapshot_root: {root / 'snapshot'}
adapter:
  type: {adapter_type}
  benchmark_config: {V3_CONFIG}
{extra}""", encoding="utf-8")
    return path


class _Root(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_benchmark(cls.root / "benchmark.jsonl")
        write_synthetic_snapshot(cls.root / "snapshot")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def run_method(self, method, adapter_type, variant, **adapter):
        config = load_experiment_config(_config(self.root, method, adapter_type, variant, **adapter))
        run = run_seed(config, config.seeds[0], run_id="fixed")
        predictions = {(p.workload, p.query_id): p for p in read_predictions(run / "predictions.csv")}
        records = {w.workload_id: list(load_queries(w)) for w in config.workloads}
        return config, run, predictions, records


class ExactMbrTest(_Root):
    def test_never_below_truth_exact_without_segment_conditions_and_honest_coverage(self) -> None:
        _, run, predictions, records = self.run_method("exact_mbr", "traj_exact_mbr", "boxes")
        for record in records["pol_v3_single"]:
            self.assertEqual(predictions[("pol_v3_single", record.query_id)].status, "unsupported")
        above = exact = 0
        for record in records["pol_v3_multi"]:
            prediction = predictions[("pol_v3_multi", record.query_id)]
            segment = [p for p in record.predicates if p["table"] == "segments"]
            segment_attribute = [p for p in segment if p["kind"] not in ("spatial", "spatio_temporal", "temporal")]
            if segment_attribute:
                self.assertEqual(prediction.status, "unsupported", record.query_id)
                continue
            self.assertEqual(prediction.status, "ok", record.query_id)
            self.assertGreaterEqual(prediction.estimated_cardinality, record.true_cardinality, record.query_id)
            if not segment:
                self.assertEqual(prediction.estimated_cardinality, record.true_cardinality, record.query_id)
                self.assertEqual(prediction.diagnostic, "", record.query_id)
                exact += 1
            else:
                # the box answers a surrogate question: an approximation by the pipeline's convention
                self.assertEqual(prediction.diagnostic, "approx:trajectory_box_overlap", record.query_id)
            above += prediction.estimated_cardinality > record.true_cardinality
        self.assertGreater(exact, 0)
        self.assertGreater(above, 0, "the box representation should overcount somewhere")
        coverage = json.loads((run / "summary.json").read_text())["workloads"]["pol_v3_multi"]["extension"]["coverage"]
        self.assertEqual(coverage["ok_exact_count"], exact)
        self.assertGreater(coverage["ok_approx_count"], 0)
        metadata = json.loads((run / "build_metrics.json").read_text())["artifacts"]
        self.assertGreater(metadata["serialized_bytes"], 0)
        self.assertEqual(metadata["trip_count"], 240)


class LatencyTest(_Root):
    def test_only_answered_queries_are_timed_and_rejections_are_separate(self) -> None:
        from evaluation.eval_core.artifacts import read_latencies

        config, run, predictions, records = self.run_method("exact_mbr", "traj_exact_mbr", "latency")
        latencies = read_latencies(run / "latency.csv")
        repetitions = int(config.timing["repetitions"])
        answered = {(p.workload, p.query_id) for p in predictions.values() if p.status == "ok"}
        timed = [(l.workload, l.query_id) for l in latencies]
        self.assertEqual(set(timed), answered)
        self.assertEqual(len(timed), len(answered) * repetitions)
        summary = json.loads((run / "summary.json").read_text())
        single = summary["workloads"]["pol_v3_single"]
        self.assertEqual(single["inference"]["observation_count"], 0)
        self.assertIsNone(single["inference"]["mean_ms"])
        rejections = summary["evaluation_detail"]["rejections"]
        self.assertEqual(rejections["pol_v3_single"]["rejected_queries"], 8 * SINGLE)
        self.assertEqual(rejections["pol_v3_single"]["timed_rejections"], 8 * SINGLE * repetitions)
        unsupported_multi = sum(p.status == "unsupported" for (w, _), p in predictions.items() if w == "pol_v3_multi")
        self.assertEqual(rejections["pol_v3_multi"]["rejected_queries"], unsupported_multi)

    def test_answered_queries_without_latency_still_fail_validation(self) -> None:
        from unittest.mock import patch

        from evaluation.eval_core.adapters.base import AdapterEvaluation, StageMetrics
        from evaluation.eval_core.records import PredictionRecord

        config = load_experiment_config(_config(self.root, "exact_mbr", "traj_exact_mbr", "nolatency"))

        class Untimed:
            def __init__(self, *args):
                pass

            def prepare(self):
                return StageMetrics()

            smoke = build = lambda self, *args, **kwargs: StageMetrics()

            def evaluate(self, workloads):
                rows = [PredictionRecord(w.workload_id, r.query_id, "ok", r.true_cardinality, 1.0)
                        for w in workloads for r in load_queries(w)]
                return AdapterEvaluation(tuple(rows), ())

            def artifact_metadata(self):
                return {}

        with patch("evaluation.eval_core.runner.create_adapter", Untimed):
            with self.assertRaisesRegex(ValueError, "no primary-device latency observations"):
                run_seed(config, 0, run_id="untimed")


class SamplingTest(_Root):
    def test_full_sample_reproduces_every_supported_label(self) -> None:
        _, _, predictions, records = self.run_method("traj_sampling", "traj_sampling", "full", sample_fraction=1.0)
        supported = 0
        for workload, items in records.items():
            for record in items:
                prediction = predictions[(workload, record.query_id)]
                if record.relation == "single" and record.tables[0] == "agents":
                    self.assertEqual(prediction.status, "unsupported")
                    continue
                self.assertEqual(prediction.status, "ok")
                self.assertEqual(prediction.estimated_cardinality, record.true_cardinality, record.query_id)
                supported += 1
        self.assertGreater(supported, 8 * MULTI)

    def test_seeds_give_independent_reproducible_samples(self) -> None:
        settings = synthetic_settings()
        data = snap.load_snapshot(self.root / "snapshot")
        trips = snap.load_trip_index(self.root / "snapshot")
        ids = lambda seed: draw_sample(settings, data, trips, 60, seed)["trips"]["trip_id"]
        self.assertTrue(np.array_equal(ids(0), ids(0)))
        self.assertFalse(np.array_equal(ids(0), ids(1)))
        sample = draw_sample(settings, data, trips, 60, 0)
        index = {int(t): i for i, t in enumerate(trips["trip_id"])}
        expected_segments = sum(int(trips["segment_count"][index[int(t)]]) for t in sample["trips"]["trip_id"])
        self.assertEqual(sample["segments"]["trip_id"].size, expected_segments)
        self.assertTrue(set(sample["segments"]["trip_id"].tolist()) <= set(sample["trips"]["trip_id"].tolist()))

    def test_estimates_are_unbiased_across_seeds(self) -> None:
        settings = synthetic_settings()
        data = snap.load_snapshot(self.root / "snapshot")
        trips = snap.load_trip_index(self.root / "snapshot")
        config = load_experiment_config(_config(self.root, "traj_sampling", "traj_sampling", "u", sample_trips=60))
        workload = next(w for w in config.workloads if w.workload_id == "pol_v3_multi")
        records = [r for r in load_queries(workload) if r.true_cardinality >= 5][:8]
        self.assertTrue(records)
        estimates = np.zeros((300, len(records)))
        for seed in range(300):
            model = SampleModel(settings, draw_sample(settings, data, trips, 60, seed), 240)
            estimates[seed] = [model.estimate(r)[1] for r in records]
        for column, record in enumerate(records):
            mean, se = estimates[:, column].mean(), estimates[:, column].std(ddof=1) / np.sqrt(300)
            self.assertLess(abs(mean - record.true_cardinality), 4 * se + 1e-9, record.query_id)

    def test_budget_rule_sizes_the_sample_and_reports_the_bytes(self) -> None:
        snapshot_bytes = sum(p.stat().st_size for p in (self.root / "snapshot" / "tables").rglob("*.npy"))
        budget = snapshot_bytes // 2
        _, run, _, _ = self.run_method("traj_sampling", "traj_sampling", "budget", budget_bytes=budget)
        artifacts = json.loads((run / "build_metrics.json").read_text())["artifacts"]
        self.assertEqual(artifacts["sample_size_rule"], "budget_bytes")
        self.assertEqual(artifacts["budget_bytes"], budget)
        self.assertLess(abs(artifacts["serialized_bytes"] - budget) / budget, 0.25, artifacts)
        self.assertAlmostEqual(artifacts["sample_fraction"], 0.5, delta=0.1)

    def test_exactly_one_size_rule_is_required(self) -> None:
        config = load_experiment_config(_config(self.root, "traj_sampling", "traj_sampling", "bad",
                                                sample_trips=10, sample_fraction=0.5))
        with self.assertRaisesRegex(ValueError, "exactly one of"):
            run_seed(config, 0, run_id="fixed")


class ReportTest(_Root):
    def test_both_methods_aggregate_and_compare_with_strata(self) -> None:
        aggregates = []
        for method, adapter_type, variant, extra in (
            ("exact_mbr", "traj_exact_mbr", "boxes", {}),
            ("traj_sampling", "traj_sampling", "half", {"sample_fraction": 0.5}),
        ):
            config = load_experiment_config(_config(self.root, method, adapter_type, variant,
                                                    seeds="[0]" if method == "exact_mbr" else "[0, 1]", **extra))
            runs = [run_seed(config, seed, run_id="cmp") for seed in config.seeds]
            aggregate_runs(runs, self.root / "agg" / method)
            aggregates.append(self.root / "agg" / method / "comparison.json")
        compare_aggregates(aggregates, self.root / "compare")
        rows = list(csv.DictReader((self.root / "compare" / "trajectory_summary_v1.csv").open()))
        overall = {(r["method_id"], r["workload"]): r for r in rows if r["stratum"] == "all"}
        self.assertEqual(float(overall[("exact_mbr", "pol_v3_single")]["coverage_fraction.mean"]), 0.0)
        self.assertGreater(float(overall[("traj_sampling", "pol_v3_single")]["coverage_fraction.mean"]), 0.0)


if __name__ == "__main__":
    unittest.main()


class ShippedConfigTest(_Root):
    """The configs in configs/pol_v3 run as shipped, with only the cluster paths swapped."""

    def test_every_pol_v3_config_runs_on_the_fixture(self) -> None:
        import re
        import shutil

        source = Path(__file__).resolve().parents[1] / "configs" / "pol_v3"
        target = self.root / "shipped"
        target.mkdir()
        for path in source.glob("*.yaml"):
            text = path.read_text(encoding="utf-8")
            text = re.sub(r"/work_beegfs/\S+/benchmark\.jsonl", str(self.root / "benchmark.jsonl"), text)
            text = re.sub(r"/work_beegfs/\S+/traj_snapshots/pol_v3", str(self.root / "snapshot"), text)
            text = text.replace("../../../../query_generation/benchmark_v3/pol_v3_config.json", str(V3_CONFIG))
            text = text.replace("expected_query_count: 6000", f"expected_query_count: {8 * MULTI}")
            text = text.replace("expected_query_count: 2000", f"expected_query_count: {8 * SINGLE}")
            text = text.replace("results_root: ../../results", f"results_root: {self.root / 'shipped_results'}")
            text = re.sub(r"sample_trips: \d+", "sample_trips: 60", text)
            (target / path.name).write_text(text, encoding="utf-8")
        configs = sorted(p for p in target.glob("*.yaml") if not p.name.startswith("_"))
        self.assertEqual(len(configs), 5)
        for path in configs:
            config = load_experiment_config(path)
            self.assertEqual(config.protocol, "adapted")
            self.assertEqual([w.workload_id for w in config.workloads], ["pol_v3_multi", "pol_v3_single"])
            run = run_seed(config, config.seeds[0], run_id="shipped")
            summary = json.loads((run / "summary.json").read_text())
            self.assertIn("extension", summary["workloads"]["pol_v3_multi"], path.name)
