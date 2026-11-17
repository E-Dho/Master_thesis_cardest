"""Data-shift ablation on JOB-light-ranges: the evaluation configs and the count refresh.

- configs/own_model_data_shift_<split>_<variant>.yaml differ from the evaluated
  model's config (own_model_job_light_ranges_full_schema_step_11000.yaml) only in
  identity, run directory, checkpoint step, model checkout, and for ``refreshed``
  the refresh option, for ``control`` the relabelled query file;
- own_model_eval_runner.py --refresh-join-cardinality scales every estimate of
  the stale model by |J_full| / |J_subset| and refuses another model's subset.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "evaluation/job_light_imdb_non_trajectory/configs"
SCRIPTS = REPO / "evaluation/job_light_imdb_non_trajectory/scripts"
B = "/work_beegfs/sunip956/master_thesis_trajectories"
SPLITS = ("random80", "temporal80")
VARIANTS = ("stale", "refreshed", "finetuned", "control")


class DataShiftConfigTest(unittest.TestCase):
    def test_configs_differ_from_the_evaluated_model_only_where_they_should(self) -> None:
        reference = (CONFIGS / "own_model_job_light_ranges_full_schema_step_11000.yaml").read_text()
        names = sorted(p.name for p in CONFIGS.glob("own_model_data_shift_*.yaml"))
        self.assertEqual(names, sorted(f"own_model_data_shift_{s}_{v}.yaml" for s in SPLITS for v in VARIANTS))
        revision = None
        for split in SPLITS:
            data = f"{B}/data_shift/imdb/{split}"
            for variant in VARIANTS:
                text = (CONFIGS / f"own_model_data_shift_{split}_{variant}.yaml").read_text()
                body = text[text.index("schema_version"):]
                kind, step = ("finetune", 2000) if variant == "finetuned" else ("stale", 11000)
                restored = body.replace(f"{B}/joblight_evaluation/own_model_data_shift/{split}_{kind}/seed_{{seed}}",
                                        f"{B}/joblight_evaluation/own_model_job_light_ranges_rare_aux_20k/seed_{{seed}}")
                restored = restored.replace(f"checkpoint_step_{step}.pt", "checkpoint_step_11000.pt")
                restored = restored.replace(f" --refresh-join-cardinality {data}/prepared/subset.json", "")
                restored = restored.replace(f"{data}/job_light_ranges_subset.csv",
                                            f"{B}/external/distjoin/queries/job-light-ranges.csv")
                restored = restored.replace(f"  job_light_ranges_subset_{split}:\n", "  job_light_ranges:\n")
                self.assertEqual(len(restored.splitlines()), len(reference.splitlines()))
                changed = [(a, b) for a, b in zip(reference.splitlines(), restored.splitlines()) if a != b]
                allowed = ("  variant_id:", "  display_name:", "  seeds:", "  revision:", "  checkout_path:", "    PYTHONPATH:")
                self.assertTrue(all(b.startswith(allowed) for _a, b in changed), (split, variant, changed))
                self.assertIn(f"variant_id: data_shift_{split}_{variant}\n", body)
                self.assertIn("seeds: [0]\n", body)
                self.assertIn(f"    PYTHONPATH: {B}/external/master_thesis_cardest_03decaf\n", body)
                # the adapter checks source.revision against the checkout's HEAD: a checkout pinned to it
                self.assertIn(f"  checkout_path: {B}/external/master_thesis_cardest_03decaf\n", body)
                self.assertEqual(body.count("--refresh-join-cardinality"), 6 if variant == "refreshed" else 0)
                self.assertEqual("job_light_ranges_subset.csv" in body, variant == "control")
                revision = re.search(r"(?m)^  revision: (\w+)$", body).group(1)
                self.assertTrue(revision.startswith("03decaf"), revision)


def _predictions(path: Path) -> dict:
    with path.open() as handle:
        return {row["query_id"]: row for row in csv.DictReader(handle)}


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "needs torch")
class RefreshTest(unittest.TestCase):
    def test_refresh_scales_every_estimate_and_checks_the_model(self) -> None:
        from model.tests import test_imdb_data_shift as fixture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = fixture.build_prepared(root)
            stale, _ = fixture.train(fixture.fixture_config(root, "stale", built["subset_prepared"]))
            lines = fixture._queries(built["full_csv"], 12)
            queries = root / "queries.csv"
            queries.write_text("\n".join(f"{line}#{truth}" for line, _sql, truth in lines) + "\n")
            subset_json = built["subset_prepared"] / "subset.json"

            def run(name: str, *extra: str) -> subprocess.CompletedProcess:
                return subprocess.run(
                    [sys.executable, str(SCRIPTS / "own_model_eval_runner.py"), "--checkpoint", str(stale.checkpoint_path),
                     "--queries", str(queries), "--predictions", str(root / f"{name}.csv"),
                     "--latencies", str(root / f"{name}_latency.csv"), "--warmup-passes", "0", "--repetitions", "1",
                     "--device", "cpu", *extra],
                    env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)

            self.assertEqual(run("stale").returncode, 0)
            result = run("refreshed", "--refresh-join-cardinality", str(subset_json))
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            summary = json.loads(subset_json.read_text())
            ratio = summary["full_join_cardinality"] / summary["subset_join_cardinality"]
            self.assertGreater(ratio, 1.0)
            plain, refreshed = _predictions(root / "stale.csv"), _predictions(root / "refreshed.csv")
            self.assertEqual(plain.keys(), refreshed.keys())
            scored = 0
            for query_id, row in plain.items():
                self.assertEqual(row["status"], refreshed[query_id]["status"])
                if row["status"] == "ok":
                    scored += 1
                    self.assertAlmostEqual(float(refreshed[query_id]["estimated_cardinality"]),
                                           ratio * float(row["estimated_cardinality"]),
                                           delta=1e-9 * max(1.0, float(row["estimated_cardinality"])))
            self.assertGreater(scored, 0)
            # the bridge passes the option on (the configs call the bridge)
            bridge = subprocess.run(
                [sys.executable, str(SCRIPTS / "own_model_bridge.py"), "evaluate", "--config", str(queries),
                 "--checkpoint", str(stale.checkpoint_path), "--output-directory", str(root / "bridge"),
                 "--queries", str(queries), "--predictions", str(root / "bridge.csv"),
                 "--latencies", str(root / "bridge_latency.csv"), "--warmup-passes", "0", "--repetitions", "1",
                 "--device", "cpu", "--refresh-join-cardinality", str(subset_json)],
                env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
            self.assertEqual(bridge.returncode, 0, bridge.stderr[-2000:])
            self.assertEqual(_predictions(root / "bridge.csv"), refreshed)
            other = root / "other_subset.json"
            other.write_text(json.dumps({**summary, "subset_join_cardinality": summary["subset_join_cardinality"] + 7}))
            result = run("other", "--refresh-join-cardinality", str(other))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("trained with", result.stderr)


if __name__ == "__main__":
    unittest.main()
