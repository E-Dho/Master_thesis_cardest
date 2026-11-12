#!/usr/bin/env python3
"""External-adapter bridge for the own model (ResMADE with ``traj_dedup_factor``) on v3 workloads (P7).

Commands (the ``external`` adapter's stages):

``prepare``   checks the checkpoint against its training config and records
              provenance: checkpoint size and sha256, the training config's
              sha256, the model code root (``--model-root``) with its git
              revision and the file the ``model`` package was imported from.
``smoke``     ``evaluate`` on the first ``--query-limit`` rows, CPU, one pass.
``build``     no training (pretrained checkpoint); writes
              ``artifact_manifest.json`` with the parameter count and the
              inference-only serialized size (``own_model_size.py``: the
              checkpoint without optimizer state), the 1x memory budget.
``evaluate``  every row of a v3 workload selection: translation
              (``traj_eval/own_model_translate.py``), the model's query context
              (``pol_workload_record_to_context``), and per route
              (``distinct`` / ``join`` / ``rows``) the estimate. Writes the
              prediction and latency CSVs of the exchange contract and a
              components CSV (matching-segment estimate, ``traj_dedup_factor``,
              forward calls) for the error split.

``--estimate matching_segments`` reports the matching-segment estimate instead
of the distinct-trajectory estimate on ``distinct`` routes: the ablation
without the dedup head (decided 2026-11-09). ``join`` and ``rows`` routes are
identical in both variants.

``--model-root`` puts that checkout's ``model`` package first on ``sys.path``,
so the same bridge can run the inference code of another checkout (the pinned
training state) for the equality check; ``evaluation`` is always taken from
this repository.

Latency: one query = building the model's query context (predicate encoding)
plus the estimate, with CUDA synchronized; translating the JSON row is
parsing and is not timed. A query answered without a forward pass (a window
outside the column domains: zero support) is timed the same way.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
LATENCY_SCOPE = "predicate_encoding_and_estimation"
COMPONENT_FIELDS = ("query_id", "route", "status", "estimate_kind", "estimated_cardinality", "matching_segment_estimate",
                    "traj_dedup_factor", "distinct_trajectory_estimate", "model_forward_calls", "zero_support",
                    "diagnostic")


def _install_paths(model_root: Path) -> None:
    for path in (str(REPO_ROOT), str(model_root)):  # model_root ends up first
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> Dict[str, Any]:
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "model/src"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "model_src_modified": None}
    return {"revision": revision, "model_src_modified": bool(dirty)}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


class OwnModel:
    """Checkpoint, training config and estimator; one call per query."""

    def __init__(self, training_config: Path, checkpoint: Path, *, device: str = "cpu",
                 clamp: Optional[Dict[str, Any]] = None) -> None:
        import torch

        from model.src.config import load_simple_yaml, validate_config
        from model.src.data.schema import ModelMetadata
        from model.src.data.trajectory_distinct import TrajectoryDistinctRuntimeConfig
        from model.src.evaluation.pol_query_adapter import assert_checkpoint_trajectory_config_compatible
        from model.src.inference.estimator import OnePassEstimator
        from model.src.inference.torch_estimator import TorchDistributionModel
        from model.src.model.checkpoint import load_resmade_checkpoint
        from model.src.predicates.vocabulary import PredicateVocabularies

        self.translate = _load_module("own_model_translate",
                                      REPO_ROOT / "evaluation/trajectory_pol_porto/traj_eval/own_model_translate.py")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA inference requested but CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.config = load_simple_yaml(str(training_config))
        validate_config(self.config)
        model, payload = load_resmade_checkpoint(checkpoint, map_location=str(self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        self.metadata = ModelMetadata.from_json_dict(payload["metadata"])
        vocabularies = PredicateVocabularies.from_json_dict(payload["predicate_vocabularies"], self.metadata)
        self.runtime = TrajectoryDistinctRuntimeConfig.from_dict(self.config.get("trajectory_distinct", {}))
        self.trajectory_spatial = dict(self.config.get("trajectory_spatial", {}) or {})
        assert_checkpoint_trajectory_config_compatible(payload, self.runtime, self.trajectory_spatial)
        self.srid = self.runtime.srid if self.runtime.srid is not None else self.config.get(
            "predicate_generation", {}).get("trajectory_srid")
        self.wrapped = TorchDistributionModel(model, self.metadata, vocabularies, device=str(self.device))
        self.estimator = OnePassEstimator(self.wrapped, self.metadata)
        self.use_log = bool(self.config.get("inference", {}).get("use_log_space_product", True))
        self.clamp = dict(clamp or {})
        self.payload_step = payload.get("step")

    def translate_row(self, row: Dict[str, Any]):
        translation = self.translate.translate(row, clamp=self.clamp, srid=self.srid)
        tables, added = root_connected_tables(self.metadata, translation.record["tables"])
        if not added:
            return translation
        record = {**translation.record, "tables": tables}
        return type(translation)(record=record, route=translation.route,
                                 notes=translation.notes + (f"fk_ancestors:{','.join(added)}",))

    def estimate(self, translation) -> Dict[str, Any]:
        from model.src.data.trajectory_distinct import trajectory_query_has_zero_support
        from model.src.evaluation.pol_query_adapter import pol_workload_record_to_context

        context = pol_workload_record_to_context(translation.record, self.metadata, srid=self.srid,
                                                 trajectory_spatial=self.trajectory_spatial)
        if trajectory_query_has_zero_support(context.trajectory_query):
            return {"matching_segment_estimate": 0.0, "traj_dedup_factor": None, "distinct_trajectory_estimate": 0.0,
                    "join_estimate": 0.0, "model_forward_calls": 0, "zero_support": True}
        if translation.route == "distinct":
            result = self.estimator.estimate_distinct_trajectories(
                list(context.tokens), context=context, trajectory_config=self.runtime,
                use_log_space_product=self.use_log)
            return {"matching_segment_estimate": float(result.matching_segment_estimate),
                    "traj_dedup_factor": float(result.traj_dedup_factor),
                    "distinct_trajectory_estimate": float(result.distinct_trajectory_estimate),
                    "join_estimate": None, "model_forward_calls": int(result.model_forward_calls),
                    "zero_support": False}
        result = self.estimator.estimate(list(context.tokens), use_log_space_product=self.use_log)
        return {"matching_segment_estimate": None, "traj_dedup_factor": None, "distinct_trajectory_estimate": None,
                "join_estimate": float(result.estimated_cardinality), "model_forward_calls": 1,
                "zero_support": False}

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def device_name(self) -> str:
        if self.device.type == "cuda":
            return str(self.torch.cuda.get_device_name(self.device))
        return "CPU"


def root_connected_tables(metadata, tables) -> tuple:
    """The query's tables plus their join ancestors, when that changes the fanout scaling.

    The model is trained on root-connected table subsets (NeuroCard's rooted
    table dropout) and scales a subset by the inverse fanout of every edge
    whose child table is missing. A subset that leaves out an ancestor of an
    included table (POL single queries on ``segments``: no ``agents``, no
    ``trips``) would be divided by the ``agents -> trips`` fanout although
    every segment row occurs once in the full join (P7 review 2026-11-12:
    estimates about 1/300 of the truth). Along a foreign key every child row
    has exactly one parent row, so adding the ancestors leaves the count
    unchanged and gives the subset the model was trained on. Subsets whose
    fanout scaling does not change (every multi query, single ``trips`` and
    ``agents``) are returned as they are. Returns ``(tables, added)``.
    """
    from model.src.predicates.generation import infer_join_graph, inverse_fanouts_for_table_subset

    graph = infer_join_graph(metadata)
    parent = {child: parent_table for parent_table, child in graph.edges}
    closure = set(tables)
    for table in tables:
        while table in parent:
            table = parent[table]
            closure.add(table)
    if inverse_fanouts_for_table_subset(metadata, set(tables)) == inverse_fanouts_for_table_subset(metadata, closure):
        return list(tables), ()
    added = tuple(table for table in graph.tables if table in closure - set(tables))
    return [table for table in graph.tables if table in closure], added


def reported_estimate(route: str, parts: Dict[str, Any], estimate_kind: str) -> float:
    if parts["zero_support"]:
        return 0.0
    if route != "distinct":
        return float(parts["join_estimate"])
    if estimate_kind == "matching_segments":
        return float(parts["matching_segment_estimate"])
    return float(parts["distinct_trajectory_estimate"])


def _timing():
    directory = str(REPO_ROOT / "evaluation" / "job_light_imdb_non_trajectory" / "scripts")
    if directory not in sys.path:
        sys.path.append(directory)
    import _timing

    return _timing


def evaluate_workload(model: OwnModel, rows: List[Dict[str, Any]], predictions_path: Path, latency_path: Path,
                      components_path: Path, *, estimate_kind: str, warmup_passes: int, repetitions: int,
                      device: str, timing_session_name: str = "own_model_traj_bridge") -> Dict[str, Any]:
    import numpy as np

    from model.src.data.trajectory_distinct import TrajectoryDistinctNotApplicable

    if warmup_passes < 0 or repetitions <= 0:
        raise ValueError("warm-up passes must be nonnegative and repetitions positive")
    if estimate_kind not in ("dedup", "matching_segments"):
        raise ValueError(f"unknown --estimate {estimate_kind!r}")
    timing = _timing()
    session = timing.session_from_environment(timing_session_name, device=device)
    if session is not None:
        session.configure(torch=model.torch)
    translations: Dict[int, Any] = {}
    unsupported: Dict[int, str] = {}
    for query_id, row in enumerate(rows):
        try:
            translations[query_id] = model.translate_row(row)
        except model.translate.TranslationError as exc:
            unsupported[query_id] = f"translation:{exc}"
    failures: Dict[int, str] = {}
    parts_of: Dict[int, Dict[str, Any]] = {}
    estimates: Dict[int, float] = {}
    latency_rows: List[Dict[str, Any]] = []
    device_name = model.device_name()

    def run(query_id: int) -> Dict[str, Any]:
        try:
            return model.estimate(translations[query_id])
        except TrajectoryDistinctNotApplicable as exc:
            unsupported[query_id] = f"not_applicable:{exc}"
            raise

    if session is not None:
        session.configure_torch(model.torch)
        session.note(query_count=len(rows), translated_query_count=len(translations))
        session.verify("pre_timing")
    with model.torch.no_grad(), timing.measured(session, "own_model_traj_workload"):
        for _ in range(warmup_passes):
            for query_id in translations:
                if query_id in failures or query_id in unsupported:
                    continue
                try:
                    run(query_id)
                except TrajectoryDistinctNotApplicable:
                    continue
                except Exception as exc:  # noqa: BLE001 - recorded per query
                    failures[query_id] = f"{type(exc).__name__}: {exc}"
        model.synchronize()
        for repetition in range(repetitions):
            for query_id in translations:
                if query_id in failures or query_id in unsupported:
                    continue
                model.synchronize()
                started = time.perf_counter()
                try:
                    parts = run(query_id)
                except TrajectoryDistinctNotApplicable:
                    continue
                except Exception as exc:  # noqa: BLE001
                    failures[query_id] = f"{type(exc).__name__}: {exc}"
                    continue
                model.synchronize()
                elapsed = (time.perf_counter() - started) * 1000.0
                value = reported_estimate(translations[query_id].route, parts, estimate_kind)
                if not (math.isfinite(value) and value >= 0.0):
                    failures[query_id] = f"invalid estimate {value!r}"
                    continue
                if query_id in estimates and not np.isclose(estimates[query_id], value, rtol=1e-7, atol=1e-9):
                    raise RuntimeError(f"own-model estimate changed between repetitions for query {query_id}")
                estimates.setdefault(query_id, value)
                parts_of.setdefault(query_id, parts)
                latency_rows.append({"query_id": query_id, "repetition": repetition, "latency_ms": elapsed,
                                     "scope": LATENCY_SCOPE, "device": device, "device_name": device_name})
    if session is not None:
        session.verify("post_timing")
        session.write()
    bad = set(failures) | set(unsupported)
    latency_rows = [row for row in latency_rows if row["query_id"] not in bad]
    prediction_rows, component_rows = [], []
    for query_id in range(len(rows)):
        translation = translations.get(query_id)
        route = translation.route if translation is not None else ""
        if query_id in unsupported:
            status, value, diagnostic = "unsupported", "", unsupported[query_id]
        elif query_id in failures:
            status, value, diagnostic = "failed", "", failures[query_id]
        else:
            status, value = "ok", repr(estimates[query_id])
            diagnostic = ";".join([f"route:{route}", *translation.notes])
        prediction_rows.append({"query_id": query_id, "estimated_cardinality": value, "status": status,
                                "diagnostic": diagnostic})
        parts = parts_of.get(query_id, {}) if status == "ok" else {}
        component_rows.append({
            "query_id": query_id, "route": route, "status": status, "estimate_kind": estimate_kind,
            "estimated_cardinality": value,
            "matching_segment_estimate": _text(parts.get("matching_segment_estimate")),
            "traj_dedup_factor": _text(parts.get("traj_dedup_factor")),
            "distinct_trajectory_estimate": _text(parts.get("distinct_trajectory_estimate")),
            "model_forward_calls": _text(parts.get("model_forward_calls")),
            "zero_support": _text(parts.get("zero_support")), "diagnostic": diagnostic})
    write_csv(predictions_path, prediction_rows, ("query_id", "estimated_cardinality", "status", "diagnostic"))
    write_csv(latency_path, latency_rows, ("query_id", "repetition", "latency_ms", "scope", "device", "device_name"))
    write_csv(components_path, component_rows, COMPONENT_FIELDS)
    summary = {"query_count": len(rows), "ok": len(estimates), "unsupported": len(unsupported),
               "failed": len(failures), "estimate": estimate_kind, "latency_scope": LATENCY_SCOPE,
               "routes": {route: sum(1 for t in translations.values() if t.route == route)
                          for route in ("distinct", "join", "rows")}}
    print(json.dumps(summary, sort_keys=True))
    return summary


def _text(value: Any) -> str:
    return "" if value is None else repr(value)


def _rows(queries_jsonl: Path, select_json: str, query_limit: Optional[int]) -> List[Dict[str, Any]]:
    from evaluation.trajectory_pol_porto.traj_eval.workloads import select_rows

    rows = select_rows(Path(queries_jsonl).resolve(), json.loads(select_json))
    return rows if query_limit is None else rows[:query_limit]


def provenance(args: argparse.Namespace) -> Dict[str, Any]:
    import model.src as model_src

    return {
        "status": "ready",
        "training_config": str(Path(args.training_config).resolve()),
        "training_config_sha256": _sha256(args.training_config),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model_root": str(Path(args.model_root).resolve()),
        "model_root_git": _git_revision(Path(args.model_root)),
        "model_package_file": str(Path(model_src.__file__).resolve()),
        "bridge_git": _git_revision(REPO_ROOT),
        "clamp": args.clamp,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "smoke", "build", "evaluate"))
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=REPO_ROOT,
                        help="checkout whose model/ package is imported (default: this repository)")
    parser.add_argument("--clamp", action="append", default=[], help="table:attribute=lo:hi (repeatable)")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--queries-jsonl", type=Path)
    parser.add_argument("--select-json")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--components", type=Path, help="default: next to --predictions, *_components.csv")
    parser.add_argument("--estimate", choices=("dedup", "matching_segments"), default="dedup")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--warmup-passes", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    _install_paths(args.model_root.resolve())
    for path in (args.training_config, args.checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    clamp = _load_module("own_model_translate",
                         REPO_ROOT / "evaluation/trajectory_pol_porto/traj_eval/own_model_translate.py"
                         ).parse_clamp(args.clamp)
    if args.command == "prepare":
        OwnModel(args.training_config, args.checkpoint, device="cpu", clamp=clamp)  # compatibility checks
        write_json(_required(args.output_directory, "--output-directory") / "prepare_manifest.json",
                   provenance(args))
        return 0
    if args.command == "build":
        size_script = _load_module("own_model_size", REPO_ROOT / "evaluation/trajectory_pol_porto/scripts/own_model_size.py")
        sizes = size_script.measure(Path(args.checkpoint))
        output = _required(args.output_directory, "--output-directory")
        write_json(output / "artifact_manifest.json", {
            "parameter_count": sizes["parameters"],
            "buffer_count": sizes["buffers"],
            "serialized_bytes": sizes["inference_only_bytes"],
            "serialized_model_mb": sizes["inference_only_bytes"] / 1_000_000,
            "serialized_bytes_definition": "checkpoint re-serialized without optimizer state (own_model_size.py)",
            "checkpoint_bytes": sizes["checkpoint_bytes"],
            "state_dict_tensor_bytes": sizes["state_dict_tensor_bytes"],
            "checkpoint": sizes["checkpoint"], "checkpoint_sha256": sizes["checkpoint_sha256"],
            "step": sizes["step"], **{f"provenance_{k}": v for k, v in provenance(args).items()},
        })
        write_json(output / "build_stage_metrics.json", {"status": "pretrained_checkpoint",
                                                         "total_build_seconds": None})
        return 0
    queries = _required(args.queries_jsonl, "--queries-jsonl")
    select = _required(args.select_json, "--select-json")
    if args.command == "smoke":
        output = _required(args.output_directory, "--output-directory")
        started = time.perf_counter()
        model = OwnModel(args.training_config, args.checkpoint, device="cpu", clamp=clamp)
        rows = _rows(queries, select, args.query_limit or 20)
        summary = evaluate_workload(model, rows, output / "smoke_predictions.csv", output / "smoke_latency.csv",
                                    output / "smoke_components.csv", estimate_kind=args.estimate, warmup_passes=0,
                                    repetitions=1, device="cpu")
        if summary["failed"]:
            raise RuntimeError(f"smoke failed on {summary['failed']} queries")
        write_json(output / "smoke_metrics.json", {"status": "ok", **summary,
                                                   "wall_seconds": time.perf_counter() - started})
        return 0
    predictions = _required(args.predictions, "--predictions")
    latency = _required(args.latency, "--latency")
    components = args.components or predictions.with_name(predictions.stem + "_components.csv")
    model = OwnModel(args.training_config, args.checkpoint, device=args.device, clamp=clamp)
    rows = _rows(queries, select, args.query_limit)
    evaluate_workload(model, rows, predictions, latency, components, estimate_kind=args.estimate,
                      warmup_passes=args.warmup_passes, repetitions=args.repetitions, device=args.device)
    write_json(components.with_name(components.stem + "_provenance.json"), provenance(args))
    return 0


def _required(value: Any, flag: str) -> Any:
    if value is None:
        raise SystemExit(f"{flag} is required for this command")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
