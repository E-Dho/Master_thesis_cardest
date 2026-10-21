"""End to end on synthetic data: generate all categories, resume, merge gates."""

import dataclasses
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from query_generation.benchmark_v3 import synthetic
from query_generation.benchmark_v3.config import load_settings
from query_generation.benchmark_v3.generate import SlotExhausted, generate_category, run_paths
from query_generation.benchmark_v3.merge import GateError, merge
from query_generation.benchmark_v3.numpy_backend import NumpyBackend
from query_generation.benchmark_v3.sampler import Sampler
from query_generation.query_generator import Category

CONFIG = Path(__file__).with_name("pol_v3_config.json")
QUIET = dict(progress_interval=0, log=lambda message: None)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings(CONFIG, single_count=6, multi_count=8)
        cls.data, rows = synthetic.build(cls.settings)
        cls.pools = synthetic.pools_from(cls.settings, rows)
        cls.backend = NumpyBackend(cls.settings, cls.data)

    def run_all(self, out: Path, count_fn=None, settings=None):
        settings = settings or self.settings
        for category in settings.categories():
            generate_category(settings, self.pools, category, base_seed=9,
                              count_fn=count_fn or self.backend.count, output_dir=out, **QUIET)

    def test_full_run_merges_with_exact_positive_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_all(out)
            result = merge(self.settings, out)
            rows = result["rows"]
            self.assertEqual(len(rows), 8 * 6 + 8 * 8)
            self.assertEqual([r["query_id"] for r in rows], [f"q{i:08d}" for i in range(1, len(rows) + 1)])
            for row in rows:
                self.assertGreater(row["cardinality"], 0)
                self.assertEqual(row["cardinality"], self.backend.count(row))
            kinds = result["summary"]["label_kinds"]
            self.assertEqual(kinds, {"row_count": 48, "distinct_trips": 64})

    def test_resume_after_interruption_gives_identical_output(self):
        category = Category.parse("spatio_temporal.range.multi")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate_category(self.settings, self.pools, category, base_seed=9,
                              count_fn=self.backend.count, output_dir=Path(a), **QUIET)
            calls = {"n": 0}

            def flaky(record):
                calls["n"] += 1
                if calls["n"] == 5:
                    raise KeyboardInterrupt("simulated kill")
                return self.backend.count(record)

            with self.assertRaises(KeyboardInterrupt):
                generate_category(self.settings, self.pools, category, base_seed=9,
                                  count_fn=flaky, output_dir=Path(b), **QUIET)
            resumed_calls = {"n": 0}

            def counting(record):
                resumed_calls["n"] += 1
                return self.backend.count(record)

            generate_category(self.settings, self.pools, category, base_seed=9,
                              count_fn=counting, output_dir=Path(b), **QUIET)
            strip = lambda path: [
                {k: v for k, v in json.loads(line).items() if k not in ("label_seconds", "labeled_at")}
                for line in path.read_text().splitlines()
            ]
            pa, pb = run_paths(Path(a), category), run_paths(Path(b), category)
            self.assertEqual(strip(pa["accepted"]), strip(pb["accepted"]))
            self.assertEqual(calls["n"] - 1 + resumed_calls["n"],
                             len(strip(pa["accepted"])) + sum(1 for r in strip(pa["rejected"]) if r["reason"] != "duplicate"),
                             "no label query is repeated after the resume")

    def test_resume_refuses_a_changed_run(self):
        category = Category.parse("temporal.range.single")
        with tempfile.TemporaryDirectory() as tmp:
            generate_category(self.settings, self.pools, category, base_seed=9,
                              count_fn=self.backend.count, output_dir=Path(tmp), **QUIET)
            with self.assertRaises(SystemExit):
                generate_category(self.settings, self.pools, category, base_seed=10,
                                  count_fn=self.backend.count, output_dir=Path(tmp), **QUIET)

    def test_deadline_stops_cleanly(self):
        category = Category.parse("spatial.range.single")
        with tempfile.TemporaryDirectory() as tmp:
            summary = generate_category(self.settings, self.pools, category, base_seed=9,
                                        count_fn=self.backend.count, output_dir=Path(tmp),
                                        deadline=time.time() - 1, **QUIET)
            self.assertFalse(summary["finished"])
            self.assertEqual(summary["accepted"], 0)

    def test_zero_labels_are_rejected_and_logged(self):
        category = Category.parse("standard.range.single")
        seen = set()

        def zero_first(record):
            slot = record["slot"]["index"]
            if slot not in seen:
                seen.add(slot)
                return 0
            return self.backend.count(record)

        with tempfile.TemporaryDirectory() as tmp:
            summary = generate_category(self.settings, self.pools, category, base_seed=9,
                                        count_fn=zero_first, output_dir=Path(tmp), **QUIET)
            self.assertTrue(summary["finished"])
            self.assertGreaterEqual(summary["rejected"]["zero"], 6)
            rejected = [json.loads(l) for l in run_paths(Path(tmp), category)["rejected"].read_text().splitlines()]
            self.assertTrue(all(r["cardinality"] == 0 for r in rejected if r["reason"] == "zero"))

    def test_duplicates_are_rejected(self):
        category = Category.parse("standard.range.single")
        original = Sampler.candidate

        def duplicate_slot_one(self_, cat, index, attempt):
            if index == 1 and attempt == 0:
                record = original(self_, cat, 0, 0)
                record["slot"] = dict(record["slot"], index=1, attempt=0)
                record["slot_id"] = record["slot_id"].replace("#0000", "#0001")
                return record
            return original(self_, cat, index, attempt)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Sampler, "candidate", duplicate_slot_one):
            generate_category(self.settings, self.pools, category, base_seed=9,
                              count_fn=lambda r: max(1, self.backend.count(r)), output_dir=Path(tmp), **QUIET)
            rejected = [json.loads(l) for l in run_paths(Path(tmp), category)["rejected"].read_text().splitlines()]
            self.assertIn("duplicate", {r["reason"] for r in rejected})

    def test_exhausted_slot_fails_loudly(self):
        settings = dataclasses.replace(self.settings, v3={**self.settings.v3, "max_attempts": 3})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(SlotExhausted):
            generate_category(settings, self.pools, Category.parse("spatial.range.multi"), base_seed=9,
                              count_fn=lambda record: 0, output_dir=Path(tmp), **QUIET)

    def test_merge_gates_catch_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_all(out)
            path = run_paths(out, Category.parse("spatio_temporal.range.multi"))["accepted"]
            original = path.read_text()
            tampering = [
                lambda r: r.update(cardinality=0, entity_cardinality=0),
                lambda r: r["predicates"][0].update(min_x=r["predicates"][0]["min_x"] - 1.0),
                lambda r: r.update(label_kind="row_count"),
                lambda r: r.update(semantics_version="segment_coupled_v2"),
            ]
            for tamper in tampering:
                rows = [json.loads(l) for l in original.splitlines()]
                tamper(rows[0])
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(GateError):
                    merge(self.settings, out)
            path.write_text("".join(l + "\n" for l in original.splitlines()[:-1]))
            with self.assertRaises(GateError):
                merge(self.settings, out)


if __name__ == "__main__":
    unittest.main()
