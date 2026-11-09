"""P7: v3 -> own-model translation and the own-model bridge.

The semantic test builds model metadata from the synthetic POL data the
fixture benchmark was labelled on and evaluates the model's query tokens
exactly on those rows: for every non-spatial query the token semantics must
reproduce the v3 label, for spatial ones the segment-MBR count (the
approximation the model is told to answer, ``mbr_overcount``'s ``mbr_count``).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.trajectory_pol_porto.tests import _fixture
from evaluation.trajectory_pol_porto.traj_eval import own_model_translate as tr

REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "evaluation/trajectory_pol_porto/scripts/own_model_traj_bridge.py"


def _row(kind_predicates, tables, relation="multi", query_id="q1"):
    return {"query_id": query_id, "tables": tables, "category": {"relation": relation},
            "predicates": kind_predicates}


class TranslateTest(unittest.TestCase):
    def test_kinds_and_closed_intervals(self) -> None:
        row = _row([
            {"kind": "nominal_eq", "table": "agents", "attribute": "interest", "value": "A"},
            {"kind": "scalar_range", "table": "agents", "attribute": "age", "lower": 20, "upper": 30},
            {"kind": "scalar_unbounded", "table": "trips", "attribute": "num_of_segments", "operator": "<",
             "value": 5},
            {"kind": "temporal", "table": "trips", "attribute": "trip_time", "lower": "2020-01-01 00:00:00",
             "upper": "2020-01-02 00:00:00"},
        ], ["agents", "trips"])
        out = tr.translate(row)
        self.assertEqual(out.route, "join")
        self.assertEqual(out.notes, ())
        modes = [(p["attribute"], p["mode"], p.get("operator")) for p in out.record["predicates"]]
        self.assertEqual(modes, [("interest", "nominal_eq", None), ("age", "range", None),
                                 ("num_of_segments", "unbounded", "<"),
                                 ("trip_time", "temporal_unbounded", "<="),
                                 ("trip_time", "temporal_unbounded", ">=")])
        # closed overlap: start <= U and end >= L
        self.assertEqual(out.record["predicates"][3]["value"], "2020-01-02 00:00:00")
        self.assertEqual(out.record["predicates"][4]["value"], "2020-01-01 00:00:00")

    def test_one_sided_interval_has_one_part(self) -> None:
        row = _row([{"kind": "temporal", "table": "segments", "attribute": "segment_time", "lower": None,
                     "upper": "2020-01-02 00:00:00"}], ["segments"], relation="single")
        out = tr.translate(row)
        self.assertEqual(out.route, "rows")
        self.assertEqual([(p["operator"], p["value"]) for p in out.record["predicates"]],
                         [("<=", "2020-01-02 00:00:00")])

    def test_spatial_and_spatio_temporal_are_marked_approximate(self) -> None:
        window = {"min_x": 0.0, "min_y": 1.0, "max_x": 2.0, "max_y": 3.0}
        spatial = tr.translate(_row([{"kind": "spatial", "table": "segments", "attribute": "segment_geom",
                                      **window}], ["segments", "trips"]), srid=26916)
        self.assertEqual(spatial.route, "distinct")
        self.assertEqual(spatial.notes, (tr.APPROX_MBR,))
        self.assertEqual(spatial.record["predicates"][0]["srid"], 26916)
        st = tr.translate(_row([{"kind": "spatio_temporal", "table": "segments", "attribute": "segment_geom",
                                 **window, "lower": "2020-01-01 00:00:00", "upper": None}], ["segments", "trips"]))
        self.assertEqual(st.notes, (tr.APPROX_MBR, tr.APPROX_DECOUPLED))
        self.assertEqual([(p["mode"], p.get("attribute"), p.get("operator")) for p in st.record["predicates"]],
                         [("spatial_intersects", "segment_geom", None),
                          ("temporal_unbounded", "segment_time", ">=")])

    def test_clamp_keeps_the_null_value_out(self) -> None:
        clamp = tr.parse_clamp(["trips:origin_stand=1:63"])
        self.assertEqual(clamp, {"trips:origin_stand": (1, 63)})

        def one(predicate):
            return tr.translate(_row([{"table": "trips", "attribute": "origin_stand", "type": "integer",
                                       **predicate}], ["trips"], relation="single"), clamp=clamp
                                ).record["predicates"][0]

        self.assertEqual(one({"kind": "scalar_unbounded", "operator": "<", "value": 10}),
                         {"table": "trips", "attribute": "origin_stand", "mode": "range", "lower": 1, "upper": 9})
        self.assertEqual(one({"kind": "scalar_unbounded", "operator": ">=", "value": 60})["lower"], 60)
        self.assertEqual(one({"kind": "scalar_unbounded", "operator": ">", "value": 60})["upper"], 63)
        self.assertEqual(one({"kind": "scalar_range", "lower": 0, "upper": 5})["lower"], 1)
        self.assertEqual(one({"kind": "nominal_eq", "value": 7})["mode"], "nominal_eq")
        with self.assertRaises(tr.TranslationError):
            one({"kind": "scalar_unbounded", "operator": "<", "value": 1})
        with self.assertRaises(tr.TranslationError):
            one({"kind": "nominal_eq", "value": 0})

    def test_refusals(self) -> None:
        with self.assertRaises(tr.TranslationError):  # one condition per column
            tr.translate(_row([{"kind": "scalar_range", "table": "trips", "attribute": "num_of_segments",
                                "lower": 1, "upper": 2},
                               {"kind": "scalar_unbounded", "table": "trips", "attribute": "num_of_segments",
                                "operator": ">", "value": 0}], ["trips"], relation="single"))
        with self.assertRaises(tr.TranslationError):  # multi without trips
            tr.translate(_row([], ["agents", "segments"]))

    def test_every_fixture_row_translates(self) -> None:
        routes = {}
        for row in _fixture.benchmark_rows():
            out = tr.translate(row, srid=26916)
            routes[out.route] = routes.get(out.route, 0) + 1
            expected = ("rows" if row["category"]["relation"] == "single"
                        else "distinct" if "segments" in row["tables"] else "join")
            self.assertEqual(out.route, expected)
            spatial = any(p["kind"] in ("spatial", "spatio_temporal") for p in row["predicates"])
            self.assertEqual(tr.APPROX_MBR in out.notes, spatial)
        self.assertEqual(sum(routes.values()), len(_fixture.benchmark_rows()))
        self.assertTrue(all(routes.get(route) for route in tr.ROUTES), routes)


def _model_available() -> bool:
    try:
        import torch  # noqa: F401

        from model.src.evaluation import pol_query_adapter  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


DATA_COLUMNS = (
    ("agents", "age"), ("agents", "educationLevel"), ("agents", "interest"), ("agents", "joviality"),
    ("agents", "family_size"), ("trips", "num_of_segments"), ("trips", "start_time"), ("trips", "end_time"),
    ("segments", "segment_idx"), ("segments", "t_s"), ("segments", "t_e"), ("segments", "s_x"),
    ("segments", "s_y"), ("segments", "e_x"), ("segments", "e_y"), ("segments", "seg_min_x"),
    ("segments", "seg_max_x"), ("segments", "seg_min_y"), ("segments", "seg_max_y"),
)


def _raw_columns(data):
    """Model column name -> raw values per table (times as epoch seconds of the naive timestamps)."""
    agents, trips, segments = data["agents"], data["trips"], data["segments"]
    seconds = lambda us: np.asarray(us, dtype=np.int64) / 1e6  # noqa: E731
    return {
        "agents": {f"agents:{name}": np.asarray(agents[name]) for name in
                   ("age", "educationLevel", "interest", "joviality", "family_size")},
        "trips": {"trips:num_of_segments": np.asarray(trips["num_of_segments"]),
                  "trips:start_time": seconds(trips["trip_time.start"]),
                  "trips:end_time": seconds(trips["trip_time.end"])},
        "segments": {"segments:segment_idx": np.asarray(segments["segment_idx"]),
                     "segments:t_s": seconds(segments["ts"]), "segments:t_e": seconds(segments["te"]),
                     "segments:s_x": np.asarray(segments["sx"]), "segments:s_y": np.asarray(segments["sy"]),
                     "segments:e_x": np.asarray(segments["ex"]), "segments:e_y": np.asarray(segments["ey"]),
                     "segments:seg_min_x": np.minimum(segments["sx"], segments["ex"]),
                     "segments:seg_max_x": np.maximum(segments["sx"], segments["ex"]),
                     "segments:seg_min_y": np.minimum(segments["sy"], segments["ey"]),
                     "segments:seg_max_y": np.maximum(segments["sy"], segments["ey"])},
    }


def _metadata(data):
    from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata

    raw = _raw_columns(data)
    columns = []
    for table, name in DATA_COLUMNS:
        values = raw[table][f"{table}:{name}"]
        domain = tuple(sorted({v.item() if hasattr(v, "item") else v for v in values}))
        columns.append(ColumnMetadata(f"{table}:{name}", ColumnKind.DATA, domain, table=table))
    for table in ("agents", "trips", "segments"):
        columns.append(ColumnMetadata(f"I_{table}", ColumnKind.INDICATOR, (0, 1), table=table))
    return ModelMetadata(columns=tuple(columns), full_join_cardinality=len(data["segments"]["trip_id"]),
                         join_root="trips", join_tables=("agents", "trips", "segments"),
                         join_edges=(("trips", "agents"), ("trips", "segments")))


@unittest.skipUnless(_model_available(), "needs torch and the model package")
class TokenSemanticsTest(unittest.TestCase):
    """The tokens of every translated fixture query, evaluated exactly on the labelled data."""

    def test_tokens_reproduce_labels_or_the_mbr_count(self) -> None:
        from model.src.evaluation.pol_query_adapter import pol_workload_record_to_context

        from evaluation.trajectory_pol_porto.traj_eval import mbr_overcount

        data = _fixture.synthetic_data()
        metadata = _metadata(data)
        raw = _raw_columns(data)
        trips, agents, segments = data["trips"], data["agents"], data["segments"]
        trip_row = {int(t): i for i, t in enumerate(trips["trip_id"])}
        agent_row = {int(a): i for i, a in enumerate(agents["agent_id"])}
        trip_agent = np.array([agent_row[int(a)] for a in trips["agent_id"]])
        seg_trip = np.array([trip_row[int(t)] for t in segments["trip_id"]])
        backend = mbr_overcount._backend_class()(_fixture.synthetic_settings(), data)

        def satisfied(table, tokens):
            mask = np.ones(len(next(iter(raw[table].values()))), dtype=bool)
            for column, values in raw[table].items():
                token = tokens.get(column)
                if token is not None:
                    mask &= np.array([token.satisfies(v.item() if hasattr(v, "item") else v) for v in values])
            return mask

        checked = {"exact": 0, "mbr": 0}
        for row in _fixture.benchmark_rows():
            out = tr.translate(row, srid=26916)
            context = pol_workload_record_to_context(out.record, metadata, srid=26916,
                                                     trajectory_spatial={"enabled": True,
                                                                         "representation": "segment_mbr"})
            tokens = dict(context.ordinary_predicates)
            masks = {table: satisfied(table, tokens) for table in ("agents", "trips", "segments")}
            if out.route == "rows":
                count = int(masks[row["tables"][0]].sum())
            else:
                trip_ok = masks["trips"].copy()
                if "agents" in row["tables"]:
                    trip_ok &= masks["agents"][trip_agent]
                if out.route == "join":
                    count = int(trip_ok.sum())
                else:
                    seg_ok = masks["segments"] & trip_ok[seg_trip]
                    count = int(np.unique(np.asarray(segments["trip_id"])[seg_ok]).size)
            predicate = mbr_overcount.spatial_predicate(row)
            if predicate is None:
                self.assertEqual(count, row["cardinality"], row["query_id"])
                checked["exact"] += 1
            else:
                classes = mbr_overcount.segment_classes(segments, predicate)
                box = np.zeros(len(segments["sx"]), dtype=bool)
                box[classes["index"]] = True
                expected = backend.count_with(row, predicate, box)
                self.assertEqual(count, expected, row["query_id"])
                self.assertGreaterEqual(count, row["cardinality"])
                checked["mbr"] += 1
        self.assertGreater(checked["exact"], 50)
        self.assertGreater(checked["mbr"], 30)


def _write_training_config(path: Path) -> None:
    static = ["agents:age", "agents:educationLevel", "agents:interest", "agents:joviality", "agents:family_size",
              "trips:num_of_segments", "trips:start_time", "trips:end_time"]
    varying = [f"segments:{n}" for n in ("segment_idx", "t_s", "t_e", "s_x", "s_y", "e_x", "e_y", "seg_min_x",
                                          "seg_max_x", "seg_min_y", "seg_max_y")]
    # the production encoding (pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml), small sizes
    lines = ["model:", "  type: predicate_resmade", "  hidden_sizes: [16]", "  residual_connections: true",
             "  direct_io_connections: true", "  input_encoding: duet_binary", "  output_encoding: embed",
             "  output_embedding_size: 8", "  embedding_size: 4", "  fixed_ordering: true",
             "predicate_encoding:", "  mode: two_slot_binary_duet", "  multi_predicate_merge: sum",
             "  max_predicates_per_column: 2",
             "trajectory_spatial:", "  enabled: true", "  representation: segment_mbr",
             "factorization:", "  enabled: false",
             "inference:", "  progressive_sampling: false", "  use_log_space_product: true",
             "predicate_generation:", "  trajectory_srid: 26916",
             "trajectory_distinct:", "  enabled: true", "  head_name: traj_dedup_factor", "  loss: mse",
             "  output_activation: sigmoid", "  entity_table: trips", "  segment_table: segments",
             "  trajectory_key: trip_id", "  segment_key: trip_id,segment_idx", "  predicate_scope: segment_query",
             "  srid: 26916", "  trajectory_static_columns:", *[f"    - {c}" for c in static],
             "  segment_varying_columns:", *[f"    - {c}" for c in varying]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bridge():
    spec = importlib.util.spec_from_file_location("own_model_traj_bridge", BRIDGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["own_model_traj_bridge"] = module
    spec.loader.exec_module(module)
    return module


def _csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@unittest.skipUnless(_model_available(), "needs torch and the model package")
class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        from model.src.config import load_simple_yaml
        from model.src.model.checkpoint import save_resmade_checkpoint
        from model.src.training.resmade_trainer import build_resmade_from_config, predicate_vocabularies_from_config

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        torch.manual_seed(0)
        metadata = _metadata(_fixture.synthetic_data())
        cls.config_path = root / "training_config.yaml"
        _write_training_config(cls.config_path)
        config = load_simple_yaml(str(cls.config_path))
        model = build_resmade_from_config(metadata, config)
        vocab = predicate_vocabularies_from_config(metadata, config)
        cls.checkpoint = root / "checkpoint_best.pt"
        save_resmade_checkpoint(cls.checkpoint, model, torch.optim.Adam(model.parameters()), epoch=0, step=7,
                                metadata=metadata, predicate_vocabularies=vocab, config=config)
        cls.benchmark = _fixture.write_benchmark(root / "benchmark.jsonl")
        cls.root = root

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def _evaluate(self, select, name, *extra):
        out = self.root / name
        args = ["evaluate", "--training-config", str(self.config_path), "--checkpoint", str(self.checkpoint),
                "--queries-jsonl", str(self.benchmark), "--select-json", json.dumps(select),
                "--predictions", str(out / "p.csv"), "--latency", str(out / "l.csv"), "--repetitions", "2",
                *extra]
        self.assertEqual(_bridge().main(args), 0)
        return _csv(out / "p.csv"), _csv(out / "l.csv"), _csv(out / "p_components.csv"), out

    def test_evaluate_routes_components_and_the_ablation(self) -> None:
        predictions, latency, components, out = self._evaluate({"relation": "multi"}, "multi")
        rows = [r for r in _fixture.benchmark_rows() if r["category"]["relation"] == "multi"]
        self.assertEqual(len(predictions), len(rows))
        self.assertEqual([int(p["query_id"]) for p in predictions], list(range(len(rows))))
        self.assertTrue(all(p["status"] == "ok" for p in predictions), predictions[:3])
        for prediction, component, row in zip(predictions, components, rows):
            route = component["route"]
            self.assertEqual(route, "distinct" if "segments" in row["tables"] else "join")
            self.assertIn(f"route:{route}", prediction["diagnostic"])
            if route == "distinct" and component["zero_support"] == "False":
                matching, factor = float(component["matching_segment_estimate"]), float(component["traj_dedup_factor"])
                self.assertAlmostEqual(float(prediction["estimated_cardinality"]), matching * factor, places=6)
                self.assertTrue(0.0 <= factor <= 1.0)
        self.assertEqual(len(latency), 2 * len(rows))  # every answered query, both repetitions
        provenance = json.loads((out / "p_components_provenance.json").read_text())
        self.assertEqual(Path(provenance["model_package_file"]).resolve().parent,
                         (REPO / "model" / "src").resolve())

        self.assertEqual({c["estimate_kind"] for c in components}, {"dedup"})
        ablation, _, ablation_components, _ = self._evaluate({"relation": "multi"}, "multi_ablation",
                                                             "--estimate", "matching_segments")
        self.assertEqual({c["estimate_kind"] for c in ablation_components}, {"matching_segments"})
        # the error split credits each variant with the factor it applied (review of ab48208)
        from evaluation.trajectory_pol_porto.traj_eval import own_model_error_split as split

        truth = {"query_id": "q", "matching_segments_model": "100", "distinct_model": "20", "distinct_exact": "10"}
        dedup_terms, ablation_terms = [], []
        for component, ablation_component in zip(components, ablation_components):
            if component["route"] == "distinct" and component["zero_support"] == "False":
                dedup_terms.append(split.split_query(component, truth)["dedup"])
                ablation_terms.append(split.split_query(ablation_component, truth)["dedup"])
        self.assertTrue(dedup_terms)
        self.assertTrue(all(abs(value - math.log10(1 / 0.2)) < 1e-9 for value in ablation_terms))
        self.assertNotEqual(dedup_terms, ablation_terms)
        for prediction, component, base in zip(ablation, ablation_components, components):
            if component["route"] == "distinct" and component["zero_support"] == "False":
                self.assertEqual(float(prediction["estimated_cardinality"]),
                                 float(base["matching_segment_estimate"]))
            else:
                self.assertEqual(prediction["estimated_cardinality"], base["estimated_cardinality"])

    def test_single_queries_use_the_row_route(self) -> None:
        predictions, _, components, _ = self._evaluate({"relation": "single"}, "single")
        self.assertTrue(predictions and all(p["status"] == "ok" for p in predictions))
        self.assertEqual({c["route"] for c in components}, {"rows"})

    def test_prepare_build_and_smoke(self) -> None:
        bridge = _bridge()
        common = ["--training-config", str(self.config_path), "--checkpoint", str(self.checkpoint),
                  "--output-directory", str(self.root / "exchange")]
        self.assertEqual(bridge.main(["prepare", *common]), 0)
        prepared = json.loads((self.root / "exchange" / "prepare_manifest.json").read_text())
        self.assertEqual(prepared["checkpoint_bytes"], self.checkpoint.stat().st_size)
        self.assertEqual(bridge.main(["build", *common]), 0)
        artifact = json.loads((self.root / "exchange" / "artifact_manifest.json").read_text())
        self.assertLess(artifact["serialized_bytes"], artifact["checkpoint_bytes"])  # optimizer state dropped
        self.assertGreater(artifact["parameter_count"], 0)
        self.assertEqual(bridge.main(["smoke", *common, "--queries-jsonl", str(self.benchmark),
                                      "--select-json", json.dumps({"relation": "multi"}), "--query-limit", "5"]), 0)
        smoke = json.loads((self.root / "exchange" / "smoke_metrics.json").read_text())
        self.assertEqual((smoke["status"], smoke["query_count"], smoke["failed"]), ("ok", 5, 0))

    def test_shipped_configs_render_and_run(self) -> None:
        import re
        import shlex

        from evaluation.eval_core.config import load_experiment_config
        from evaluation.eval_core.workload_formats import format_of

        shipped = REPO / "evaluation/trajectory_pol_porto/configs/pol_v3"
        root = self.root / "configs"
        root.mkdir(exist_ok=True)
        base = (shipped / "_base.yaml").read_text(encoding="utf-8")
        base = re.sub(r"/work_beegfs/\S+/benchmark\.jsonl", str(self.benchmark), base)
        base = base.replace("../../../../query_generation/benchmark_v3/pol_v3_config.json", str(_fixture.V3_CONFIG))
        base = base.replace("expected_query_count: 6000", f"expected_query_count: {8 * _fixture.MULTI}")
        base = base.replace("expected_query_count: 2000", f"expected_query_count: {8 * _fixture.SINGLE}")
        (root / "_base.yaml").write_text(base, encoding="utf-8")
        models = self.root / "p7_models" / "seed_0"
        models.mkdir(parents=True, exist_ok=True)
        for name, source in (("checkpoint_best.pt", self.checkpoint), ("training_config.yaml", self.config_path)):
            if not (models / name).exists():
                (models / name).symlink_to(source)
        estimates = {}
        for name, estimate in (("own_model", "dedup"), ("own_model_no_dedup", "matching_segments")):
            text = (shipped / f"{name}.yaml").read_text(encoding="utf-8")
            text = text.replace("/work_beegfs/sunip956/master_thesis_trajectories/p7_models/pol_v3/seed_{seed}",
                                str(self.root / "p7_models" / "seed_{seed}"))
            (root / f"{name}.yaml").write_text(text, encoding="utf-8")
            config = load_experiment_config(root / f"{name}.yaml")
            self.assertEqual((config.method_id, config.seeds, config.adapter_type), ("own_model", (0, 1, 2), "external"))
            self.assertEqual(tuple(config.timing["profiles"]), ("cpu_1core", "gpu_single_query"))
            workload = next(w for w in config.workloads if w.workload_id == "pol_v3_multi")
            out = self.root / f"rendered_{name}"
            values = {"seed": "0", "exchange_directory": str(out), "predictions_csv": str(out / "p.csv"),
                      "latency_csv": str(out / "l.csv"), "query_limit": "3",
                      **format_of(workload).placeholders(workload)}
            argv = shlex.split(config.adapter["evaluate_command"].format(**values))
            self.assertTrue(argv[1].endswith("own_model_traj_bridge.py"))
            self.assertEqual(argv[argv.index("--estimate") + 1], estimate)
            self.assertEqual(json.loads(argv[argv.index("--select-json") + 1]), {"relation": "multi"})
            self.assertEqual(_bridge().main(argv[2:]), 0)
            estimates[name] = _csv(out / "p.csv")
            smoke = shlex.split(config.adapter["smoke_command"].format(**values))
            self.assertEqual(json.loads(smoke[smoke.index("--select-json") + 1]), {"relation": "multi"})
            for profile, device in (("cpu_1core", "cpu"), ("gpu_single_query", "cuda")):
                timing = shlex.split(config.adapter["timing_commands"][profile].format(**values))
                self.assertEqual((timing[timing.index("--device") + 1], timing[timing.index("--repetitions") + 1]),
                                 (device, "3"))
        differs = [a["query_id"] for a, b in zip(estimates["own_model"], estimates["own_model_no_dedup"])
                   if a["estimated_cardinality"] != b["estimated_cardinality"]]
        self.assertTrue(differs)  # the variants differ exactly on the distinct routes


class SubmitP7Test(unittest.TestCase):
    """STAGE=run must not evaluate other checkpoints than P7_MODELS names (review of ab48208)."""

    def _submit(self, env_extra):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "sbatch"
            log = Path(tmp) / "sbatch.log"
            fake.write_text('#!/bin/bash\necho "$@" >> "$FAKE_SBATCH_LOG"\necho 4242\n')
            fake.chmod(0o755)
            env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "FAKE_SBATCH_LOG": str(log),
                   "REPO_ROOT": str(REPO), "STAGE": "run", **env_extra}
            if "P7_MODELS" not in env_extra:
                env.pop("P7_MODELS", None)
            result = subprocess.run(["bash", str(REPO / "evaluation/trajectory_pol_porto/slurm/submit_p7.sh")],
                                    env=env, capture_output=True, text=True)
            submitted = log.read_text().splitlines() if log.exists() else []
        return result, submitted

    def test_default_directory_submits_both_variants(self) -> None:
        result, submitted = self._submit({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(submitted), 3)  # two seed arrays and the comparison
        self.assertTrue(all("--array=0,1,2" in line for line in submitted[:2]))

    def test_other_directory_is_refused_before_submitting(self) -> None:
        result, submitted = self._submit({"P7_MODELS": "/elsewhere/p7_models/pol_v3"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("edit the config", result.stderr)
        self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
