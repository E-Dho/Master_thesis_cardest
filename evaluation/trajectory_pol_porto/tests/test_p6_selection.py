"""P6 selection under the memory budget, and the own-model inference-only size."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.trajectory_pol_porto.traj_eval import select_tuned
from evaluation.trajectory_pol_porto.traj_eval.tune import COLUMNS


def _write(path: Path, rows, coverage="cov2:" + "c" * 64, run_directory="r"):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for parameters, p50, p90, p99, size in rows:
            writer.writerow({"variant_id": "v", "parameters": json.dumps(parameters), "query_count": 1800,
                             "scored_count": 678, "coverage_fraction": 678 / 1800, "q_p50": p50, "q_p90": p90,
                             "q_p95": p90, "q_p99": p99, "q_max": p99, "zero_estimate_count": 0,
                             "serialized_bytes": size, "theoretical_bytes": size, "build_seconds": 1,
                             "coverage_sha256": coverage, "run_directory": run_directory})


def _predictions(run: Path, statuses, benchmark_sha="b" * 64):
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_manifest.json").write_text(json.dumps({"workloads": {"validation_multi": {
        "queries_jsonl_sha256": benchmark_sha, "select": {"relation": "multi"},
        "query_id_rule": "0-based position within the selection, in file order", "query_count": len(statuses)}}}))
    lines = ["workload,query_id,status,true_cardinality,estimated_cardinality,raw_q_error,smoothed_q_error,diagnostic"]
    lines += [f"validation_multi,{i},{status},1,,,," for i, status in enumerate(statuses)]
    (run / "predictions.csv").write_text("\n".join(lines) + "\n")


class SelectionTest(unittest.TestCase):
    def test_budget_and_best_picks_across_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            _write(first, [({"unit_size_m": 25, "time_buckets": 400}, 1.03, 16.8, 174, 1_768_869_066),
                           ({"unit_size_m": 400, "time_buckets": 400}, 1.23, 20.9, 350, 6_534_850),
                           ({"unit_size_m": 400, "time_buckets": 100}, 1.28, 80.3, 1354, 1_626_850)])
            _write(second, [({"unit_size_m": 400, "time_buckets": 3200}, 1.20, 4.0, 60, 52_000_000),
                            ({"unit_size_m": 400, "time_buckets": 1600}, 1.21, 7.0, 90, 26_000_000)])
            rows = select_tuned.read_rows([first, second])
            result = select_tuned.select(rows, budget_bytes=30_000_000)
        self.assertEqual(result["settings"], 5)
        self.assertEqual(result["best"]["parameters"], {"unit_size_m": 400, "time_buckets": 3200})
        self.assertEqual(result["budget"]["parameters"], {"unit_size_m": 400, "time_buckets": 1600})
        self.assertEqual(result["smallest"]["parameters"], {"unit_size_m": 400, "time_buckets": 100})
        self.assertAlmostEqual(result["best_over_budget_factor"], 52 / 30)

    def test_duplicate_settings_must_agree_in_every_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            _write(first, [({"u": 1}, 1.0, 2.0, 3.0, 100)])
            _write(second, [({"u": 1}, 1.0, 2.0, 3.0, 100)])
            self.assertEqual(len(select_tuned.read_rows([first, second])), 1)  # identical repeat is fine
            _write(second, [({"u": 1}, 1.0, 2.0, 3.0, 999)])  # same p90, different size
            with self.assertRaisesRegex(ValueError, "serialized_bytes"):
                select_tuned.read_rows([first, second])
            _write(second, [({"u": 1}, 1.0, 2.0, 3.0, 100)], coverage="cov2:" + "d" * 64)
            with self.assertRaisesRegex(ValueError, "coverage_sha256"):
                select_tuned.read_rows([first, second])

    def test_coverage_is_the_answered_query_set_not_its_size(self) -> None:
        from evaluation.trajectory_pol_porto.traj_eval.tune import coverage_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _predictions(root / "r1", ["ok", "unsupported", "ok"])
            _predictions(root / "r2", ["ok", "ok", "unsupported"])  # same count, other queries
            self.assertNotEqual(coverage_sha256(root / "r1"), coverage_sha256(root / "r2"))
            # same ids and statuses, but another benchmark file: other queries
            _predictions(root / "r3", ["ok", "unsupported", "ok"], benchmark_sha="e" * 64)
            self.assertNotEqual(coverage_sha256(root / "r1"), coverage_sha256(root / "r3"))
            _predictions(root / "r4", ["ok", "unsupported", "ok"])
            self.assertEqual(coverage_sha256(root / "r1"), coverage_sha256(root / "r4"))
            (root / "r4" / "run_manifest.json").write_text(json.dumps({"workloads": {}}))
            with self.assertRaisesRegex(ValueError, "queries_jsonl_sha256"):
                coverage_sha256(root / "r4")
            # files written before the column existed: recomputed from the run directories
            a, b = root / "a.csv", root / "b.csv"
            _write(a, [({"u": 1}, 1.0, 2.0, 3.0, 100)], coverage="", run_directory=str(root / "r1"))
            _write(b, [({"u": 2}, 1.0, 2.5, 3.0, 50)], coverage="", run_directory=str(root / "r2"))
            rows = select_tuned.read_rows([a, b])
            self.assertEqual(rows[0]["coverage_sha256"], coverage_sha256(root / "r1"))
            with self.assertRaisesRegex(ValueError, "different query sets"):
                select_tuned.select(rows, budget_bytes=1000)
            # a fingerprint stored by the previous, unbound version is not trusted: recomputed
            old = "0" * 64
            _write(a, [({"u": 1}, 1.0, 2.0, 3.0, 100)], coverage=old, run_directory=str(root / "r1"))
            self.assertEqual(select_tuned.read_rows([a])[0]["coverage_sha256"], coverage_sha256(root / "r1"))
            self.assertTrue(coverage_sha256(root / "r1").startswith("cov2:"))
            _write(a, [({"u": 1}, 1.0, 2.0, 3.0, 100)], coverage=old, run_directory=str(root / "missing"))
            with self.assertRaisesRegex(ValueError, "no coverage_sha256"):
                select_tuned.read_rows([a])
            _write(a, [({"u": 1}, 1.0, 2.0, 3.0, 100)], coverage="", run_directory=str(root / "r1"))
            _write(b, [({"u": 2}, 1.0, 2.5, 3.0, 50)], coverage="", run_directory=str(root / "missing"))
            with self.assertRaisesRegex(ValueError, "no coverage_sha256"):
                select_tuned.read_rows([a, b])

    def test_nothing_under_the_budget_and_ties(self) -> None:
        rows = [{"parameters": {"u": 1}, "scored_count": 5, "q_p50": 1, "q_p90": 2, "q_p99": 3,
                 "serialized_bytes": 100, "variant_id": "a", "coverage_fraction": 1, "q_max": 3,
                 "zero_estimate_count": 0, "theoretical_bytes": 100, "coverage_sha256": "x", "source": "s"},
                {"parameters": {"u": 2}, "scored_count": 5, "q_p50": 1, "q_p90": 2, "q_p99": 3,
                 "serialized_bytes": 50, "variant_id": "b", "coverage_fraction": 1, "q_max": 3,
                 "zero_estimate_count": 0, "theoretical_bytes": 50, "coverage_sha256": "x", "source": "s"}]
        result = select_tuned.select(rows, budget_bytes=10)
        self.assertIsNone(result["budget"])
        self.assertEqual(result["best"]["parameters"], {"u": 2})  # tie -> smaller model
        with self.assertRaises(ValueError):  # different coverage would make accuracy incomparable
            select_tuned.select([rows[0], dict(rows[1], scored_count=4)], budget_bytes=10)
        with self.assertRaises(ValueError):  # same count, other queries
            select_tuned.select([rows[0], dict(rows[1], coverage_sha256="y")], budget_bytes=10)


def _size_module():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "own_model_size.py"
    spec = importlib.util.spec_from_file_location("own_model_size", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OwnModelSizeTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_parameters_exclude_buffers_and_the_optimizer_is_dropped(self) -> None:
        import torch

        class Masked(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(200, 100)
                self.register_buffer("mask", torch.ones(100, 200))

        model = Masked()
        optimizer = torch.optim.Adam(model.parameters())
        model.linear(torch.randn(4, 200)).sum().backward()
        optimizer.step()

        def rebuild(payload):
            module = Masked()
            module.load_state_dict(payload["model_state_dict"])
            return module

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "step": 7, "trajectory_distinct": {"enabled": True},
                        "trajectory_target_semantics_version": "pol_exact_st_v3", "metadata": {}}, path)
            before = path.read_bytes()
            row = _size_module().measure(path, build_module=rebuild)
            self.assertEqual(path.read_bytes(), before)  # read-only
        self.assertEqual(row["parameters"], 200 * 100 + 100)  # the mask buffer is not a parameter
        self.assertEqual(row["parameter_bytes"], 4 * (200 * 100 + 100))
        self.assertEqual(row["buffers"], 100 * 200)
        self.assertEqual(row["state_dict_tensor_bytes"], row["parameter_bytes"] + row["buffer_bytes"])
        self.assertTrue(row["has_optimizer_state"])
        self.assertLess(row["inference_only_bytes"], row["checkpoint_bytes"])
        self.assertGreater(row["inference_only_bytes"], row["state_dict_tensor_bytes"])
        self.assertEqual(row["step"], 7)

    def test_real_resmade_checkpoint(self) -> None:
        import torch

        from model.src.config import load_simple_yaml
        from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
        from model.src.model.checkpoint import save_resmade_checkpoint
        from model.src.predicates.vocabulary import PredicateVocabularies
        from model.src.training.resmade_trainer import build_resmade_from_config

        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        model = build_resmade_from_config(source.metadata, config)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_resmade_checkpoint(path, model, optimizer, epoch=0, step=3, metadata=source.metadata,
                                    predicate_vocabularies=PredicateVocabularies.from_metadata(source.metadata),
                                    config=config)
            row = _size_module().measure(path)  # default: rebuilds the ResMADE
        self.assertEqual(row["parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(row["buffers"], sum(b.numel() for b in model.buffers()))
        self.assertGreater(row["buffers"], 0)  # MADE masks


if __name__ == "__main__":
    unittest.main()
