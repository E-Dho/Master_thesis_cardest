"""The shared core: plugin registry, workload formats, and the IMDB aliases."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.eval_core import plugins
from evaluation.eval_core.config import WorkloadConfig, load_experiment_config


class PluginRegistryTest(unittest.TestCase):
    def tearDown(self) -> None:
        plugins._plugins.cache_clear()

    def test_imdb_plugin_provides_its_adapters_and_format(self) -> None:
        from evaluation.eval_core.adapters.external import ExternalCommandAdapter
        from evaluation.job_light_imdb_non_trajectory.joblight_eval.adapters.postgres import (
            PostgresAdapter,
        )

        self.assertIs(plugins.adapter_class("external"), ExternalCommandAdapter)
        self.assertIs(plugins.adapter_class("postgres"), PostgresAdapter)
        self.assertEqual(plugins.workload_format("joblight_csv").name, "joblight_csv")
        self.assertIn("foj_sampling", plugins.in_process_timing_adapters())

    def test_unknown_names_fail_with_the_known_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown adapter type"):
            plugins.adapter_class("nope")
        with self.assertRaisesRegex(ValueError, "joblight_csv"):
            plugins.workload_format("nope")

    def test_a_name_defined_twice_is_an_error(self) -> None:
        clash = types.ModuleType("clash_plugin")
        clash.ADAPTERS = {"postgres": object}
        real = plugins._plugins()
        plugins._plugins.cache_clear()
        with patch.object(plugins, "_plugins", return_value=real + (clash,)):
            with self.assertRaisesRegex(ValueError, "defined by both"):
                plugins.adapter_class("postgres")

    def test_lazy_adapter_factories_are_resolved(self) -> None:
        lazy = types.ModuleType("lazy_plugin")
        lazy.ADAPTERS = {"lazy_kind": lambda: dict}
        real = plugins._plugins()
        plugins._plugins.cache_clear()
        with patch.object(plugins, "_plugins", return_value=real + (lazy,)):
            self.assertIs(plugins.adapter_class("lazy_kind"), dict)


class ImdbAliasTest(unittest.TestCase):
    def test_old_module_paths_are_the_core_modules(self) -> None:
        names = ["config", "artifacts", "metrics", "records", "report", "runner",
                 "timing", "timing_guard", "resources"]
        for name in names:
            old = __import__(
                f"evaluation.job_light_imdb_non_trajectory.joblight_eval.{name}",
                fromlist=["_"],
            )
            new = __import__(f"evaluation.eval_core.{name}", fromlist=["_"])
            self.assertIs(old, new, name)

    def test_workloads_module_does_not_import_the_core_config(self) -> None:
        # The DeepDB bridges import it from a Python 3.8 environment.
        source = Path(
            sys.modules["evaluation.eval_core.config"].__file__
        ).parents[1] / "job_light_imdb_non_trajectory" / "joblight_eval" / "workloads.py"
        header = source.read_text(encoding="utf-8").split("class JobLightCsvFormat")[0]
        top_level = [line for line in header.splitlines() if line.startswith(("from ", "import "))]
        self.assertFalse(any("eval_core" in line for line in top_level), top_level)


class WorkloadFormatDispatchTest(unittest.TestCase):
    def test_default_format_is_joblight_csv_and_positional_construction_still_works(self) -> None:
        workload = WorkloadConfig("job_light", Path("queries.csv"))
        self.assertEqual(workload.format, "joblight_csv")
        self.assertEqual(workload.queries_file, Path("queries.csv"))
        self.assertEqual(workload.select, ())

    def test_config_routes_parsing_through_the_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "q.csv").write_text("title t##t.id,=,1#3\n\n", encoding="utf-8")
            config_path = root / "c.yaml"
            config_path.write_text(
                f"""schema_version: 1
experiment:
  experiment_id: x
  method_id: m
  variant_id: v
source:
  url: https://example.invalid/r.git
  revision: abc
paths:
  results_root: {root / 'results'}
workloads:
  job_light:
    queries_csv: {root / 'q.csv'}
    expected_query_count: 1
artifacts:
  checkpoint: none
adapter:
  type: external
""",
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)
            self.assertEqual(config.workloads[0].format, "joblight_csv")
            text = config_path.read_text(encoding="utf-8").replace(
                "expected_query_count: 1", "expected_query_count: 2"
            )
            config_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 2 queries, observed 1"):
                load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
