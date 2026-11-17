"""Data-shift ablation: the evaluation configs in configs/{pol_v3,porto_v3}_shift."""

from __future__ import annotations

import json
import re
import shlex
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.workload_formats import format_of

from . import _fixture
from .test_p7_own_model import REPO, _bridge, _csv, _metadata, _model_available, _write_training_config

B = "/work_beegfs/sunip956/master_thesis_trajectories"
CONFIGS = REPO / "evaluation/trajectory_pol_porto/configs"
VARIANTS = ("stale", "refreshed", "finetuned", "control")
SPLITS = ("random80", "temporal80")
ALLOWED = ("  variant_id:", "  display_name:", "  seeds:", "  installed_version:")


def _body(text: str) -> str:
    return text[text.index("extends:"):]


class ShiftConfigTextTest(unittest.TestCase):
    def test_each_config_differs_from_the_own_model_only_where_it_should(self) -> None:
        for dataset in ("pol_v3", "porto_v3"):
            reference = _body((CONFIGS / dataset / "own_model.yaml").read_text())
            folder = CONFIGS / f"{dataset}_shift"
            names = sorted(p.name for p in folder.glob("*.yaml"))
            self.assertEqual(names, sorted([f"_base_control_{s}.yaml" for s in SPLITS]
                                           + [f"own_model_shift_{s}_{v}.yaml" for s in SPLITS for v in VARIANTS]))
            for split in SPLITS:
                for variant in VARIANTS:
                    text = _body((folder / f"own_model_shift_{split}_{variant}.yaml").read_text())
                    kind = "finetune" if variant == "finetuned" else "stale"
                    models = f"{B}/p7_models/{dataset}_shift/{split}_{kind}/seed_{{seed}}"
                    self.assertNotIn(f"{B}/p7_models/{dataset}/seed_", text)
                    restored = text.replace(models, f"{B}/p7_models/{dataset}/seed_{{seed}}")
                    restored = restored.replace(f" --refresh-join-cardinality {B}/data_shift/{dataset}/{split}/prepared/subset.json", "")
                    restored = re.sub(r"^extends: .*$", "extends: _base.yaml", restored, count=1, flags=re.M)
                    self.assertEqual(len(restored.splitlines()), len(reference.splitlines()))
                    changed = [b for a, b in zip(reference.splitlines(), restored.splitlines()) if a != b]
                    self.assertTrue(all(line.startswith(ALLOWED) for line in changed), (dataset, split, variant, changed))
                    self.assertEqual("--refresh-join-cardinality" in text, variant == "refreshed")
                    expected_base = f"_base_control_{split}.yaml" if variant == "control" else f"../{dataset}/_base.yaml"
                    self.assertTrue(text.startswith(f"extends: {expected_base}\n"))
                base = (CONFIGS / dataset / "_base.yaml").read_text()
                control = (folder / f"_base_control_{split}.yaml").read_text()
                control_body = control[control.index("schema_version"):]
                expected = base[base.index("schema_version"):]
                self.assertIn(f"{B}/data_shift/{dataset}/{split}/benchmark_subset.jsonl", control_body)
                self.assertEqual(len(control_body.splitlines()), len(expected.splitlines()))
                changed = [b for a, b in zip(expected.splitlines(), control_body.splitlines()) if a != b]
                self.assertEqual(len(changed), 4, changed)  # two workload ids, two query files


@unittest.skipUnless(_model_available(), "needs torch and the model package")
class ShiftConfigRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from model.src.config import load_simple_yaml
        from model.src.model.checkpoint import save_resmade_checkpoint
        from model.src.training.resmade_trainer import build_resmade_from_config, predicate_vocabularies_from_config

        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        torch.manual_seed(0)
        metadata = _metadata(_fixture.synthetic_data())
        cls.config_path = cls.root / "training_config.yaml"
        _write_training_config(cls.config_path)
        config = load_simple_yaml(str(cls.config_path))
        model = build_resmade_from_config(metadata, config)
        cls.checkpoint = cls.root / "checkpoint_best.pt"
        save_resmade_checkpoint(cls.checkpoint, model, torch.optim.Adam(model.parameters()), epoch=0, step=7,
                                metadata=metadata, predicate_vocabularies=predicate_vocabularies_from_config(metadata, config),
                                config=config)
        cls.join = metadata.full_join_cardinality
        cls.benchmark = _fixture.write_benchmark(cls.root / "benchmark.jsonl")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_every_pol_config_renders_and_runs(self) -> None:
        dataset = "pol_v3"
        root = self.root / "configs"
        (root / dataset).mkdir(parents=True, exist_ok=True)
        (root / f"{dataset}_shift").mkdir(exist_ok=True)

        def localize(text: str) -> str:
            text = re.sub(r"/work_beegfs/\S+/benchmark(_subset)?\.jsonl", str(self.benchmark), text)
            text = text.replace("../../../../query_generation/benchmark_v3/pol_v3_config.json", str(_fixture.V3_CONFIG))
            text = text.replace("expected_query_count: 6000", f"expected_query_count: {8 * _fixture.MULTI}")
            return text.replace("expected_query_count: 2000", f"expected_query_count: {8 * _fixture.SINGLE}")

        (root / dataset / "_base.yaml").write_text(localize((CONFIGS / dataset / "_base.yaml").read_text()))
        for split in SPLITS:
            (root / f"{dataset}_shift" / f"_base_control_{split}.yaml").write_text(
                localize((CONFIGS / f"{dataset}_shift" / f"_base_control_{split}.yaml").read_text()))
            subset = self.root / f"data_shift/{split}/prepared/subset.json"
            subset.parent.mkdir(parents=True, exist_ok=True)
            subset.write_text(json.dumps({"subset_join_cardinality": self.join, "full_join_cardinality": self.join * 1.25}))
            estimates = {}
            for variant in VARIANTS:
                kind = "finetune" if variant == "finetuned" else "stale"
                models = self.root / f"p7_models/{split}_{kind}/seed_0"
                if not models.exists():
                    models.mkdir(parents=True)
                    (models / "checkpoint_best.pt").symlink_to(self.checkpoint)
                    (models / "training_config.yaml").symlink_to(self.config_path)
                text = (CONFIGS / f"{dataset}_shift" / f"own_model_shift_{split}_{variant}.yaml").read_text()
                text = text.replace(f"{B}/p7_models/{dataset}_shift/{split}_{kind}/seed_{{seed}}",
                                    str(self.root / f"p7_models/{split}_{kind}/seed_{{seed}}"))
                text = text.replace(f"{B}/data_shift/{dataset}/{split}/prepared/subset.json", str(subset))
                path = root / f"{dataset}_shift" / f"own_model_shift_{split}_{variant}.yaml"
                path.write_text(text)
                config = load_experiment_config(path)
                self.assertEqual((config.method_id, config.variant_id, config.seeds),
                                 ("own_model", f"shift_{split}_{variant}", (0,)))
                ids = sorted(w.workload_id for w in config.workloads)
                expected_ids = ([f"{dataset}_multi_subset_{split}", f"{dataset}_single_subset_{split}"]
                                if variant == "control" else [f"{dataset}_multi", f"{dataset}_single"])
                self.assertEqual(ids, sorted(expected_ids))
                workload = next(w for w in config.workloads if "multi" in w.workload_id)
                out = self.root / f"rendered_{split}_{variant}"
                values = {"seed": "0", "exchange_directory": str(out), "predictions_csv": str(out / "p.csv"),
                          "latency_csv": str(out / "l.csv"), "query_limit": "5", **format_of(workload).placeholders(workload)}
                argv = shlex.split(config.adapter["evaluate_command"].format(**values))
                self.assertEqual(_bridge().main(argv[2:]), 0)
                estimates[variant] = [float(p["estimated_cardinality"]) for p in _csv(out / "p.csv") if p["status"] == "ok"]
            # the refreshed variant is the stale one times |J_full| / |J_subset|
            self.assertEqual(len(estimates["refreshed"]), len(estimates["stale"]))
            for stale, refreshed in zip(estimates["stale"], estimates["refreshed"]):
                self.assertAlmostEqual(refreshed, 1.25 * stale, delta=1e-9 * max(1.0, stale))


if __name__ == "__main__":
    unittest.main()
