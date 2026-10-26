from __future__ import annotations

import csv
import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "distjoin_bridge.py"
SPEC = importlib.util.spec_from_file_location("distjoin_bridge", SCRIPT)
BRIDGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE)


class DistJoinBridgeTests(unittest.TestCase):
    def test_evaluator_receives_explicit_experiment_mark(self):
        completed = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(BRIDGE.subprocess, "run", return_value=completed) as run:
            BRIDGE._run_distjoin_evaluator(
                source=Path("/source"),
                overlay=Path("/overlay"),
                python=Path("/python"),
                queries=Path("/queries"),
                base_cardinalities=Path("/basecards"),
                predictions=Path("/predictions"),
                latencies=Path("/latencies"),
                warmup_passes=1,
                repetitions=2,
                experiment_mark="smoke",
            )
        command = run.call_args.args[0]
        mark_index = command.index("--experiment-mark")
        self.assertEqual(command[mark_index + 1], "smoke")
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_gpu_evaluator_keeps_allocated_cuda_device_visible(self):
        completed = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(BRIDGE.subprocess, "run", return_value=completed) as run:
            BRIDGE._run_distjoin_evaluator(
                source=Path("/source"), overlay=Path("/overlay"),
                python=Path("/python"), queries=Path("/queries"),
                base_cardinalities=Path("/basecards"),
                predictions=Path("/predictions"), latencies=Path("/latencies"),
                warmup_passes=1, repetitions=2, device="cuda",
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertNotEqual(
            run.call_args.kwargs["env"].get("CUDA_VISIBLE_DEVICES"), ""
        )

    def test_absolute_smoke_warmup_is_one_step(self):
        self.assertEqual(
            BRIDGE._effective_warmup_steps(1, batches_per_epoch=1, epochs=1),
            1,
        )

    def test_fractional_tiny_smoke_warmup_rejects_zero_steps(self):
        with self.assertRaisesRegex(ValueError, "resolves to zero"):
            BRIDGE._effective_warmup_steps(0.1, batches_per_epoch=1, epochs=1)

    def test_parameter_count_excludes_autoregressive_mask_buffers(self):
        state = {
            "net.0.weight": types.SimpleNamespace(numel=lambda: 12),
            "net.0.bias": types.SimpleNamespace(numel=lambda: 3),
            "net.0.mask": types.SimpleNamespace(numel=lambda: 12),
            "direct_io_layer.mask": types.SimpleNamespace(numel=lambda: 9),
        }
        self.assertEqual(BRIDGE._parameter_count_from_state_dict(state), 15)

    def test_fixture_contains_all_job_light_tables_with_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            BRIDGE._write_fixture(root)
            observed = BRIDGE._validate_dataset(root)
            self.assertEqual(set(observed), set(BRIDGE.JOB_LIGHT_TABLES))
            self.assertIn("production_year", observed["title"]["header"])
            self.assertIn("keyword_id", observed["movie_keyword"]["header"])
            with (root / "title.csv").open(newline="", encoding="utf-8") as handle:
                title_rows = list(csv.DictReader(handle))
            for column in ("imdb_index", "series_years"):
                values = {row[column] for row in title_rows}
                self.assertNotIn("", values)
                self.assertGreater(len(values), 1)

    def test_smoke_workload_covers_two_native_join_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.csv"
            BRIDGE._write_smoke_workload(path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("movie_keyword", lines[0])
            self.assertIn("movie_info_idx", lines[1])

    def test_production_config_rejects_recipe_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            config = {
                "seed": 42, "datasets": ["imdb"], "num_gpu": 1,
                "excludes": [], "exclude_gpus": [], "tag": "", "data_dir": "",
                "factorize": True,
                "train": {
                    "use_pregen_data": False, "use_ANPM": True, "model_type": "MADE",
                    "bs": 16_384, "sample_bs": 4_096, "max_steps": 255, "epochs": 20,
                },
            }
            path = source / "Configs" / "IMDB"
            path.mkdir(parents=True)
            class FakeYaml:
                @staticmethod
                def safe_load(_text):
                    return config

            original = BRIDGE._yaml_module
            BRIDGE._yaml_module = lambda: FakeYaml
            self.addCleanup(setattr, BRIDGE, "_yaml_module", original)
            (path / "IMDB.yaml").write_text("test fixture")
            with self.assertRaisesRegex(ValueError, "max_steps"):
                BRIDGE._production_config(source, source, 0)

    def test_production_config_preserves_imdb_ordering_sentinel(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            config = {
                "seed": 42, "datasets": ["imdb"], "num_gpu": 1,
                "excludes": [], "exclude_gpus": [], "tag": "", "data_dir": "",
                "factorize": True, "test": {},
                "train": {
                    "use_pregen_data": False, "use_ANPM": True, "model_type": "MADE",
                    "bs": 16_384, "sample_bs": 4_096, "max_steps": 256, "epochs": 20,
                },
            }
            path = source / "Configs" / "IMDB"
            path.mkdir(parents=True)
            class FakeYaml:
                @staticmethod
                def safe_load(_text):
                    return config

            original = BRIDGE._yaml_module
            BRIDGE._yaml_module = lambda: FakeYaml
            self.addCleanup(setattr, BRIDGE, "_yaml_module", original)
            (path / "IMDB.yaml").write_text("test fixture")
            observed = BRIDGE._production_config(source, source, 3)
            self.assertEqual(observed["datasets"], ["imdb"])
            self.assertEqual(observed["test"]["glob"], "{}-seed3-19.pt")

if __name__ == "__main__":
    unittest.main()
