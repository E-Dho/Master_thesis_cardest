from __future__ import annotations

import importlib.util
import collections
import subprocess
import sys
import json
import pickle
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation.job_light_imdb_non_trajectory.joblight_eval import inference_size as size
from evaluation.job_light_imdb_non_trajectory.scripts import deepdb_bridge, deepdb_ranges_adapted_bridge, mscn_bridge, neurocard_bridge, distjoin_bridge


class InferenceSizeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def file(self, name, content=b"test"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_deepdb_counts_the_ensemble_and_rdc_used_by_cardinality(self):
        checkpoint = self.file("ensemble/model.pkl")
        rdc = self.file("ensemble/pairwise_rdc.pkl", b"rdc")
        report = deepdb_bridge.inference_artifact_metadata(checkpoint)
        self.assertEqual({x["path"] for x in report["inference_files"]}, {str(checkpoint), str(rdc)})
        self.assertEqual(report["inference_artifact_mb"], 7 / 1_000_000)

    def test_adapted_deepdb_also_counts_the_literal_rank_domains(self):
        checkpoint = self.file("ensemble/model.pkl")
        rdc = self.file("ensemble/pairwise_rdc.pkl")
        domains = self.file("rank_domains.json", b"{}")
        report = deepdb_ranges_adapted_bridge.inference_artifact_metadata(checkpoint, domains)
        self.assertEqual({x["path"] for x in report["inference_files"]},
                         {str(checkpoint), str(rdc), str(domains)})

    def test_mscn_published_counts_the_exact_fixed_workload_bitmap(self):
        ckpt = self.file("model.pt")
        meta = self.file("model_metadata.json", b"{}")
        bitmap = self.file("source/workloads/job-light.bitmaps", b"bitmap")
        report = mscn_bridge.inference_artifact_metadata(ckpt, meta, self.root / "source")
        self.assertEqual({x["path"] for x in report["inference_files"]},
                         {str(ckpt), str(meta), str(bitmap)})
        self.assertEqual(report["inference_artifact_mb"], 12 / 1_000_000)

    def test_mscn_adapted_counts_the_literal_encoder_domains_and_samples(self):
        ckpt = self.file("model.pt")
        meta = self.file("metadata.json", b"{}")
        state = self.file("runtime/domain_and_samples.pkl", pickle.dumps({"domains": {}, "sampled_rows_by_table": {}}))
        report = mscn_bridge.inference_artifact_metadata(ckpt, meta, self.root / "source", self.root / "runtime")
        self.assertEqual({x["path"] for x in report["inference_files"]}, {str(ckpt), str(meta), str(state)})

    def test_foj_counts_its_sample_array_and_the_sidecar_evaluate_reads(self):
        reservoir = self.root / "sample.npy"
        np.save(reservoir, np.arange(24).reshape(8, 3))
        metadata = self.file("sample.json", b'{"metadata": {}}')
        report = size.foj_size(reservoir, metadata, 4)
        self.assertEqual({x["path"] for x in report["inference_files"]}, {str(reservoir), str(metadata)})
        self.assertAlmostEqual(report["inference_artifact_mb"],
                               (reservoir.stat().st_size - 4 * 3 * 8 + metadata.stat().st_size) / 1_000_000)
        with self.assertRaises(ValueError):
            size.foj_size(reservoir, metadata, 9)

    def test_neurocard_measures_loaded_schemas_and_domains_without_base_rows(self):
        ckpt = self.file("model.pt")
        column = types.SimpleNamespace(name="x", data=np.arange(1000), all_distinct_values=np.array([1, 2, 3]))
        table = types.SimpleNamespace(columns=[column], data=np.zeros((1000, 20)), cardinality=1000)
        runtime = types.SimpleNamespace(table=table, config={"word_size_bits": 11}, join_spec={"tables": ["t"]},
                                        train_data=types.SimpleNamespace(columns=[column], table_num_columns=[1],
                                                                         combined_columns_types=[0], table_indexes=[0]))
        report = neurocard_bridge.inference_artifact_metadata(runtime, ckpt)
        payload = size.schema_payload({"table": runtime.table,
                                       "factorized_columns": runtime.train_data.columns,
                                       "table_num_columns": [1], "column_types": [0], "table_indexes": [0],
                                       "join_spec": runtime.join_spec, "config": runtime.config})
        self.assertEqual(report["inference_metadata_bytes"], len(pickle.dumps(payload, protocol=4)))
        restored = pickle.loads(pickle.dumps(payload, protocol=4))
        self.assertIsNone(restored["table"].data)
        self.assertIsNone(restored["table"].columns[0].data)
        np.testing.assert_array_equal(restored["table"].columns[0].all_distinct_values, [1, 2, 3])
        self.assertEqual(report["inference_files"][0]["path"], str(ckpt))
        self.assertIsNotNone(runtime.table.data)

    def test_derived_schema_keeps_domain_masks_but_not_training_row_buffers(self):
        column = types.SimpleNamespace(name="__fanout_t", data=np.arange(1000),
                                       nan_ind=np.zeros(1000, dtype=bool), has_none=True,
                                       all_distinct_values=np.array([0, 1, 2]))
        schema = types.SimpleNamespace(columns=[column], bit_masks=np.array([3, 7]),
                                        factorized_tuples_np=np.ones((10000, 10)),
                                        new_tuples=np.ones((10000, 10)), inp=np.ones((10000, 10)),
                                        table_dataset=types.SimpleNamespace(tuples_np=np.ones((10000, 10))))
        result = pickle.loads(pickle.dumps(size.schema_payload(schema), protocol=4))
        for name in ["factorized_tuples_np", "new_tuples", "inp", "table_dataset"]:
            self.assertIsNone(getattr(result, name))
        np.testing.assert_array_equal(result.bit_masks, [3, 7])
        np.testing.assert_array_equal(result.columns[0].all_distinct_values, [0, 1, 2])
        self.assertIsNone(result.columns[0].data)
        self.assertIsNone(result.columns[0].nan_ind)
        self.assertTrue(result.columns[0].has_none)
        self.assertLess(len(pickle.dumps(result, protocol=4)), 2000)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_distjoin_counts_only_the_checkpoint_paths_load_checkpoints_opens(self):
        from evaluation.job_light_imdb_non_trajectory.scripts import distjoin_eval_runner as runner
        names = list(distjoin_bridge.JOB_LIGHT_TABLES)
        directory = self.root / "Configs/IMDB/model/production"
        expected = [self.file("Configs/IMDB/model/production/" + n + "-seed0-19.pt") for n in names]
        self.file("Configs/IMDB/model/production/title-seed0-18.pt", b"old epoch")
        paths = size.distjoin_checkpoints(directory, "{}-seed0-19.pt", names)
        self.assertEqual(paths, expected)
        module = types.SimpleNamespace(config={"glob": "{}-seed0-19.pt"}, util=types.SimpleNamespace(get_device=lambda: "cpu"))
        estimator = types.SimpleNamespace(models={n: types.SimpleNamespace(load_state_dict=lambda s: None, eval=lambda: None) for n in names})
        model_module = types.ModuleType("model")
        model_module.made = types.SimpleNamespace(MADE=type("Made", (), {}))
        with mock.patch.dict("sys.modules", {"model": model_module}), mock.patch.object(runner.torch, "load", return_value={}) as load:
            runner._load_checkpoints(module, estimator, self.root, "production")
        self.assertEqual([c.args[0] for c in load.call_args_list], paths)

    def test_distjoin_counts_optional_cardinality_pickle_only_when_it_is_opened(self):
        cards = self.file("normalization.json", b"{}")
        self.assertEqual(size.distjoin_cardinality_files(self.root, cards), [cards])
        optional = self.file("queries/basecard.pkl", pickle.dumps({}))
        self.assertEqual(size.distjoin_cardinality_files(self.root, cards), [optional, cards])

    def test_postgres_does_not_mislabel_whole_catalog_statistics(self):
        artifact = {"database_manifest": {"statistics_storage_bytes": 704512}}
        self.assertIsNone(size.postgres_size(artifact)["inference_artifact_mb"])
        artifact["database_manifest"].update(statistics_storage_scope="benchmark_tables", statistics_storage_source="pg_statistic benchmark relations")
        self.assertEqual(size.postgres_size(artifact)["inference_artifact_mb"], .704512)

    def test_serialized_schemas_keep_the_loaded_mapping_type_and_factory(self):
        mapping = collections.defaultdict(list, {"factor": [1, 2]})
        loaded = pickle.loads(pickle.dumps(size.schema_payload(mapping), protocol=4))
        self.assertIsInstance(loaded, collections.defaultdict)
        self.assertIs(loaded.default_factory, list)
        self.assertEqual(loaded["factor"], [1, 2])

    def test_standalone_bridges_resolve_size_code_without_pythonpath(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        names = ["distjoin_bridge.py", "mscn_bridge.py", "neurocard_bridge.py"]
        if importlib.util.find_spec("torch"):
            names.append("distjoin_eval_runner.py")
        for name in names:
            with self.subTest(bridge=name):
                code = ("import runpy; runpy.run_path(" + repr(str(scripts / name)) + "); "
                        "from evaluation.job_light_imdb_non_trajectory.joblight_eval.inference_size import files_size")
                completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=self.root,
                                           capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_files_are_not_modified_and_missing_inputs_are_rejected(self):
        p = self.file("artifact", b"unchanged")
        stat = p.stat();before = p.read_bytes()
        size.files_size([p])
        self.assertEqual(p.read_bytes(), before)
        self.assertEqual(p.stat().st_mtime_ns, stat.st_mtime_ns)
        with self.assertRaises(FileNotFoundError):
            size.files_size([self.root / "missing"])


if __name__ == "__main__":
    unittest.main()
