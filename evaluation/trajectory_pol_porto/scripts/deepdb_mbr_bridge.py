#!/usr/bin/env python3
"""DeepDB-MBR bridge: native DeepDB (pinned revision, Python 3.8 environment)
on the one-row-per-trajectory table exported by ``traj_eval/deepdb_mbr.py``.

Commands
--------
``prepare-shared``  CSV -> DeepDB HDF (``prepare_all_tables``), once per
                    export, into ``<shared>/hdf``; seed-independent.
``prepare``         checks the source revision and the shared files.
``smoke``           a throwaway single-table SPN on the first rows of the
                    export, two box queries; estimates must be finite and
                    within [0, rows].
``build``           one single-table SPN (``create_naive_all_split_ensemble``)
                    per seed from ``--sample-size`` sampled trips.
``evaluate``        every query of a v3 workload selection: translation to
                    box-overlap SQL (``traj_eval.deepdb_mbr.translate``),
                    DeepDB's ``parse_query`` and ``SPNEnsemble.cardinality``.

Latency per answered query (scope ``LATENCY_SCOPE``) covers translation, SQL
rendering, parsing and inference; ``model_core_ms`` is ``cardinality`` alone.
As in the IMDB bridge, DeepDB's cross-query expectation cache is cleared
before every query, outside the timed region. Unsupported queries (single
table, ``segment_idx``) get no latency rows. Only the standard library,
numpy/pandas of the DeepDB environment and the stdlib-only pipeline modules
``traj_eval.workloads`` and ``traj_eval.deepdb_mbr`` are imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.trajectory_pol_porto.traj_eval import deepdb_mbr  # noqa: E402
from evaluation.trajectory_pol_porto.traj_eval.workloads import select_rows  # noqa: E402

LATENCY_SCOPE = "deepdb_mbr_translation_sql_parse_and_native_inference"
DATASET = "trips_mbr"
DEFAULTS = {"sample_size": 10_000_000, "rdc_threshold": 0.3, "post_sampling_factor": 1,
            "max_table_data": 100_000_000}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-shared", "prepare", "smoke", "build", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--source-root", required=True, type=Path)
        command.add_argument("--revision", required=True)
        command.add_argument("--shared-root", required=True, type=Path)
        command.add_argument("--seed", type=int, default=0)
    sub.choices["prepare"].add_argument("--output", required=True, type=Path)
    sub.choices["smoke"].add_argument("--output-directory", required=True, type=Path)
    sub.choices["smoke"].add_argument("--rows", type=int, default=5_000)
    build = sub.choices["build"]
    build.add_argument("--output-directory", required=True, type=Path)
    build.add_argument("--sample-size", type=int, default=DEFAULTS["sample_size"])
    build.add_argument("--rdc-threshold", type=float, default=DEFAULTS["rdc_threshold"])
    build.add_argument("--post-sampling-factor", type=int, default=DEFAULTS["post_sampling_factor"])
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--checkpoint-directory", required=True, type=Path)
    evaluate.add_argument("--queries-jsonl", required=True, type=Path)
    evaluate.add_argument("--select-json", required=True)
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--latency", required=True, type=Path)
    evaluate.add_argument("--query-limit", type=int)
    evaluate.add_argument("--warmup-passes", type=int)
    evaluate.add_argument("--repetitions", type=int)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():  # load_upstream changes the working directory
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    source = validate_source(args.source_root, args.revision)
    set_seed(args.seed)
    if args.command == "prepare-shared":
        prepare_shared(source, args.shared_root)
    elif args.command == "prepare":
        prepare(source, args.shared_root, args.output)
    elif args.command == "smoke":
        smoke(source, args.shared_root, args.output_directory, args.rows)
    elif args.command == "build":
        build_model(source, args.shared_root, args.output_directory, args.sample_size,
                    args.rdc_threshold, args.post_sampling_factor, args.seed)
    else:
        warmup = args.warmup_passes if args.warmup_passes is not None else int(
            os.environ.get("JOBLIGHT_TIMING_WARMUP_PASSES", "0"))
        repetitions = args.repetitions if args.repetitions is not None else int(
            os.environ.get("JOBLIGHT_TIMING_REPETITIONS", "1"))
        evaluate_workload(source, args.shared_root, args.checkpoint_directory, args.queries_jsonl,
                          json.loads(args.select_json), args.predictions, args.latency,
                          query_limit=args.query_limit, warmup_passes=warmup, repetitions=repetitions)
    return 0


# -- upstream -----------------------------------------------------------------

def validate_source(source: Path, revision: str) -> Path:
    source = source.resolve()
    observed = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if observed != revision:
        raise ValueError(f"DeepDB revision mismatch: expected {revision}, got {observed}")
    return source


def load_upstream(source: Path) -> Dict[str, Any]:
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(source)
    import numpy as np
    import pandas as pd
    from data_preparation.prepare_single_tables import prepare_all_tables
    from ensemble_compilation.graph_representation import SchemaGraph, Table
    from ensemble_compilation.spn_ensemble import read_ensemble
    from ensemble_creation.naive import create_naive_all_split_ensemble
    from evaluation.utils import parse_query

    install_compat()
    return {"np": np, "pd": pd, "prepare_all_tables": prepare_all_tables, "SchemaGraph": SchemaGraph,
            "Table": Table, "read_ensemble": read_ensemble,
            "single_ensemble": create_naive_all_split_ensemble, "parse_query": parse_query}


NUMPY_COMPAT = [
    "rspn.rspn.build_ds_context: ragged column domains become a 1-d object array, as numpy < 1.24 did "
    "(numpy >= 1.24 raises for np.asanyarray of [min, max] pairs mixed with the value lists of "
    "categorical columns)",
    "spn.algorithms.splitting.RDC.rdc_cca: only when CCA raises AND the pair is verified degenerate "
    "(both feature matrices finite with equal row counts, and every canonical correlation, computed "
    "scale-free from orthonormal bases of the centred matrices, at most 1e-8; e.g. one variable constant "
    "over the slice, or two-valued variables with orthogonal patterns on a tiny slice) the pair gets "
    "rdc = NaN. rdc = 0 is then the exact answer, and SPFlow maps NaN to 0, i.e. independence "
    "(getIndependentRDCGroups_py); scikit-learn 1.2's NIPALS divides 0 by 0 there and scipy's gesdd "
    "then raises. Variation below 1e-12 per entry counts as rounding noise. Every other failure, including "
    "non-finite input, is re-raised; the number of fallbacks is recorded (cca_degenerate_fallbacks)",
]


class _RaggedAsObject:
    """``numpy`` for ``rspn.rspn`` only, restoring numpy < 1.24's ragged ``asanyarray``.

    ``build_ds_context`` stores one domain per column: ``[min, max]`` for real
    columns and the list of codes for categorical ones. Older numpy turned the
    mixed list into an object array (with a deprecation warning) and DeepDB
    indexes it per column; numpy 1.24 raises instead. Everything else is the
    real numpy.
    """

    def __getattr__(self, name: str) -> Any:
        import numpy

        return getattr(numpy, name)

    def asanyarray(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        import numpy

        try:
            return numpy.asanyarray(value, *args, **kwargs)
        except ValueError:
            if args or kwargs or not isinstance(value, (list, tuple)):
                raise
            out = numpy.empty(len(value), dtype=object)
            for index, item in enumerate(value):
                out[index] = item
            return out


# Variation below NOISE_FLOOR per entry is rounding noise: RDC features are sin(.) of
# copula-rank projections plus a ones column (|value| <= 1), and real variation between distinct
# values is many orders of magnitude larger.
NOISE_FLOOR = 1e-12
# Largest canonical correlation (scale-free, in [0, 1]) that still counts as independent.
CANONICAL_TOLERANCE = 1e-8


def _centred_basis(matrix: Any) -> Any:
    """Orthonormal basis of the column space of the centred matrix (numerical rank)."""
    import numpy

    centred = matrix - matrix.mean(axis=0)
    u, singular, _ = numpy.linalg.svd(centred, full_matrices=False)
    if singular.size == 0:
        return u[:, :0]
    rows, columns = centred.shape
    cutoff = max(NOISE_FLOOR * math.sqrt(rows * columns),
                 float(singular[0]) * max(rows, columns) * numpy.finfo(float).eps)
    return u[:, singular > cutoff]


def canonical_correlations(x: Any, y: Any) -> Any:
    """All canonical correlations of two finite feature matrices (empty if a side has no variation)."""
    import numpy

    basis_x, basis_y = _centred_basis(x), _centred_basis(y)
    if basis_x.shape[1] == 0 or basis_y.shape[1] == 0:
        return numpy.zeros(0)
    return numpy.linalg.svd(basis_x.T @ basis_y, compute_uv=False)


def degenerate_pair(indexes: Any) -> bool:
    """Finite features, same rows, and every canonical correlation of the pair (numerically) 0.

    Computed scale-free from orthonormal bases of the centred feature matrices,
    so a small but genuine dependent component is not mistaken for
    independence. A side without variation (a constant variable) has no
    canonical correlations at all. For such a pair rdc = 0 is the exact answer.
    """
    import numpy

    i, j, features = indexes
    try:
        x = numpy.asarray(features[i], dtype=float)
        y = numpy.asarray(features[j], dtype=float)
    except (TypeError, ValueError):
        return False
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2 or x.size == 0 or y.size == 0:
        return False
    if not (numpy.all(numpy.isfinite(x)) and numpy.all(numpy.isfinite(y))):
        return False
    try:
        rho = canonical_correlations(x, y)
    except numpy.linalg.LinAlgError:
        return False
    return rho.size == 0 or float(rho.max()) <= CANONICAL_TOLERANCE


def _nan_on_degenerate_cca(original):
    """``rdc_cca`` with NaN for a verified degenerate pair whose CCA raised; anything else re-raises."""
    fallbacks: List[Any] = []

    def rdc_cca(indexes):
        import numpy

        try:
            return original(indexes)
        except (ValueError, numpy.linalg.LinAlgError):
            if not degenerate_pair(indexes):
                raise
            fallbacks.append((indexes[0], indexes[1]))  # list.append is atomic (threading backend)
            return float("nan")

    rdc_cca.__wrapped__ = original
    rdc_cca.fallbacks = fallbacks
    return rdc_cca


def cca_fallback_count() -> Optional[int]:
    import spn.algorithms.splitting.RDC as rdc_module

    fallbacks = getattr(rdc_module.rdc_cca, "fallbacks", None)
    return None if fallbacks is None else len(fallbacks)


def install_compat() -> None:
    """The two compatibility shims of ``NUMPY_COMPAT`` (idempotent)."""
    import rspn.rspn as rspn_module
    import spn.algorithms.splitting.RDC as rdc_module

    if not isinstance(rspn_module.np, _RaggedAsObject):
        rspn_module.np = _RaggedAsObject()
    if not hasattr(rdc_module.rdc_cca, "__wrapped__"):
        rdc_module.rdc_cca = _nan_on_degenerate_cca(rdc_module.rdc_cca)


def set_seed(seed: int) -> None:
    os.environ["JOBLIGHT_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def make_schema(runtime: Dict[str, Any], csv_path: Path, layout: Dict[str, Any], rows: int):
    schema = runtime["SchemaGraph"]()
    names = [column["name"] for column in layout["columns"]]
    schema.add_table(runtime["Table"](
        deepdb_mbr.TABLE, attributes=names, irrelevant_attributes=["trip_id"], primary_key=["trip_id"],
        csv_file_location=str(csv_path), table_size=int(rows)))
    return schema


def shared_files(shared: Path) -> Dict[str, Any]:
    manifest = json.loads((shared / "export_manifest.json").read_text(encoding="utf-8"))
    return {"layout": deepdb_mbr.read_layout(shared), "manifest": manifest,
            "csv": shared / deepdb_mbr.CSV_NAME, "hdf": shared / "hdf"}


# -- stages ---------------------------------------------------------------------

def prepare_shared(source: Path, shared: Path) -> None:
    runtime = load_upstream(source)
    files = shared_files(shared)
    files["hdf"].mkdir(parents=True, exist_ok=True)
    marker = files["hdf"] / "prepared_for.json"
    expected = {"csv_sha256": files["manifest"]["csv_sha256"], "revision": _revision(source)}
    if marker.exists() and json.loads(marker.read_text()) == expected and (files["hdf"] / "meta_data.pkl").exists():
        print(f"HDF already prepared for this export: {files['hdf']}")
        return
    started = time.perf_counter()
    schema = make_schema(runtime, files["csv"], files["layout"], files["manifest"]["rows"])
    runtime["prepare_all_tables"](schema, str(files["hdf"]), csv_seperator=",",
                                  max_table_data=DEFAULTS["max_table_data"])
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True))
    write_json(shared / "hdf_preparation.json", {"seconds": time.perf_counter() - started, **expected,
                                                 "rows": files["manifest"]["rows"]})
    print(f"prepared {files['hdf']} in {time.perf_counter() - started:.1f} s")


def _revision(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def prepare(source: Path, shared: Path, output: Path) -> None:
    runtime = load_upstream(source)
    files = shared_files(shared)
    marker = files["hdf"] / "prepared_for.json"
    if not marker.exists() or json.loads(marker.read_text())["csv_sha256"] != files["manifest"]["csv_sha256"]:
        raise ValueError(f"{files['hdf']} is not prepared for the current export; run prepare-shared")
    payload = {"status": "ready", "source_revision": _revision(source), "python": sys.version,
               "numpy": runtime["np"].__version__, "pandas": runtime["pd"].__version__,
               "numpy_compat": NUMPY_COMPAT,
               "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__,
               "rows": files["manifest"]["rows"], "columns": files["manifest"]["columns"],
               "export_csv_sha256": files["manifest"]["csv_sha256"],
               "snapshot_manifest_sha256": files["manifest"]["snapshot_manifest_sha256"]}
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _box_query(layout: Dict[str, Any], csv_path: Path, rows: int) -> List[str]:
    """Two queries for the smoke model: the full box, and a quarter of it."""
    names = [column["name"] for column in layout["columns"]]
    lows: Dict[str, float] = {}
    highs: Dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for number, values in enumerate(csv.reader(handle)):
            if number >= rows:
                break
            for name, value in zip(names, values):
                if name.startswith("b_"):
                    lows[name] = min(lows.get(name, math.inf), float(value))
                    highs[name] = max(highs.get(name, -math.inf), float(value))
    full = [("b_xmin", "<=", highs["b_xmax"]), ("b_xmax", ">=", lows["b_xmin"])]
    middle_x = (lows["b_xmin"] + highs["b_xmax"]) / 2
    middle_y = (lows["b_ymin"] + highs["b_ymax"]) / 2
    quarter = [("b_xmin", "<=", middle_x), ("b_ymin", "<=", middle_y)]
    return [deepdb_mbr.to_sql(full), deepdb_mbr.to_sql(quarter)]


def smoke(source: Path, shared: Path, output: Path, rows: int) -> None:
    runtime = load_upstream(source)
    files = shared_files(shared)
    output.mkdir(parents=True, exist_ok=True)
    head = output / deepdb_mbr.CSV_NAME
    with files["csv"].open(encoding="utf-8") as reader, head.open("w", encoding="utf-8") as writer:
        count = 0
        for line in reader:
            if count >= rows:
                break
            writer.write(line)
            count += 1
    schema = make_schema(runtime, head, files["layout"], count)
    hdf = output / "hdf"
    ensemble = output / "ensemble"
    hdf.mkdir(exist_ok=True)
    ensemble.mkdir(exist_ok=True)
    started = time.perf_counter()
    runtime["prepare_all_tables"](schema, str(hdf), csv_seperator=",", max_table_data=count)
    runtime["single_ensemble"](schema, str(hdf), count, str(ensemble), DATASET, False, 0.3, count, 1, 0)
    model = runtime["read_ensemble"](str(_checkpoint(ensemble)), build_reverse_dict=True)
    estimates = []
    for sql in _box_query(files["layout"], head, count):
        model.cached_expecation_vals = dict()
        estimates.append(_cardinality(model, runtime["parse_query"](sql, schema)))
    if not all(math.isfinite(value) and -1e-6 <= value <= count * (1 + 1e-6) for value in estimates):
        raise ValueError(f"invalid DeepDB-MBR smoke estimates {estimates} for {count} rows")
    payload = {"status": "ok", "rows": count, "estimates": estimates,
               "wall_seconds": time.perf_counter() - started, "cca_degenerate_fallbacks": cca_fallback_count()}
    write_json(output / "smoke_metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _checkpoint(directory: Path) -> Path:
    found = sorted(directory.glob(f"ensemble_single_{DATASET}_*.pkl"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one ensemble_single_{DATASET}_*.pkl in {directory}, found {found}")
    return found[0]


def _cardinality(model: Any, query: Any) -> float:
    _, _, value, _ = model.cardinality(query, rdc_spn_selection=False, pairwise_rdc_path=None,
                                       merge_indicator_exp=True, max_variants=1,
                                       exploit_overlapping=True, return_factor_values=True)
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid estimate {value}")
    return value


def build_model(source: Path, shared: Path, output: Path, sample_size: int, rdc_threshold: float,
                post_sampling_factor: int, seed: int) -> None:
    runtime = load_upstream(source)
    files = shared_files(shared)
    schema = make_schema(runtime, files["csv"], files["layout"], files["manifest"]["rows"])
    ensemble = output / "ensemble"
    ensemble.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime["single_ensemble"](schema, str(files["hdf"]), int(sample_size), str(ensemble), DATASET, False,
                               float(rdc_threshold), DEFAULTS["max_table_data"], int(post_sampling_factor), 0)
    seconds = time.perf_counter() - started
    checkpoint = _checkpoint(ensemble)
    preparation = shared / "hdf_preparation.json"
    metrics = {"training_seconds": seconds, "seed": seed, "sample_size": int(sample_size),
               "effective_samples": min(int(sample_size), int(files["manifest"]["rows"])),
               "rdc_threshold": float(rdc_threshold), "post_sampling_factor": int(post_sampling_factor),
               "shared_hdf_seconds": (json.loads(preparation.read_text())["seconds"]
                                      if preparation.exists() else None),
               "note": "trip-table export and HDF preparation are shared, reported once",
               "numpy_compat": NUMPY_COMPAT, "cca_degenerate_fallbacks": cca_fallback_count()}
    write_json(output / "build_stage_metrics.json", metrics)
    write_json(output / "artifact_manifest.json", {
        "parameter_count": None,
        "parameter_count_reason": "sum-product network (pickled ensemble)",
        "serialized_bytes": checkpoint.stat().st_size,
        "serialized_model_mb": checkpoint.stat().st_size / 1_000_000,
        "checkpoint": str(checkpoint),
        "training": metrics,
    })
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _timing():
    directory = str(PROJECT_ROOT / "evaluation" / "job_light_imdb_non_trajectory" / "scripts")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import _timing

    return _timing


def evaluate_workload(source: Path, shared: Path, checkpoint_directory: Path, queries_jsonl: Path,
                      select: Dict[str, Any], predictions_path: Path, latency_path: Path, *,
                      query_limit: Optional[int], warmup_passes: int, repetitions: int) -> None:
    if warmup_passes < 0 or repetitions <= 0:
        raise ValueError("warm-up passes must be nonnegative and repetitions positive")
    timing = _timing()
    session = timing.session_from_environment("deepdb_mbr_bridge", device="cpu")
    rows = select_rows(Path(queries_jsonl).resolve(), select)
    runtime = load_upstream(source)
    files = shared_files(shared)
    schema = make_schema(runtime, files["csv"], files["layout"], files["manifest"]["rows"])
    model = runtime["read_ensemble"](str(_checkpoint(checkpoint_directory)), build_reverse_dict=True)
    if query_limit is not None:
        rows = rows[:query_limit]
    layout = files["layout"]
    notes: Dict[int, str] = {}
    unsupported: Dict[int, str] = {}
    failures: Dict[int, str] = {}
    supported = []
    for query_id, row in enumerate(rows):
        conditions, note = deepdb_mbr.translate(row, layout)
        if conditions is None:
            unsupported[query_id] = note
        else:
            notes[query_id] = note
            supported.append((query_id, row))

    def estimate(row: Dict[str, Any]):
        conditions, _ = deepdb_mbr.translate(row, layout)
        query = runtime["parse_query"](deepdb_mbr.to_sql(conditions), schema)
        core = time.perf_counter()
        value = _cardinality(model, query)
        return value, (time.perf_counter() - core) * 1000.0

    if session is not None:
        session.configure()
        session.note(query_count=len(rows), supported_query_count=len(supported))
        session.verify("pre_timing")
    estimates: Dict[int, float] = {}
    latency_rows: List[Dict[str, Any]] = []
    with timing.measured(session, "deepdb_mbr_workload"):
        for _ in range(warmup_passes):
            for query_id, row in supported:
                if query_id in failures:
                    continue
                try:
                    model.cached_expecation_vals = dict()
                    estimate(row)
                except Exception as exc:  # noqa: BLE001 - recorded per query
                    failures[query_id] = f"{type(exc).__name__}: {exc}"
        for repetition in range(repetitions):
            for query_id, row in supported:
                if query_id in failures:
                    continue
                try:
                    model.cached_expecation_vals = dict()
                    started = time.perf_counter()
                    value, core_ms = estimate(row)
                    elapsed = (time.perf_counter() - started) * 1000.0
                except Exception as exc:  # noqa: BLE001
                    failures[query_id] = f"{type(exc).__name__}: {exc}"
                    continue
                if query_id in estimates and estimates[query_id] != value:
                    raise RuntimeError(f"DeepDB-MBR estimate changed between repetitions for query {query_id}")
                estimates[query_id] = value
                latency_rows.append({"query_id": query_id, "repetition": repetition, "latency_ms": elapsed,
                                     "scope": LATENCY_SCOPE, "device": "cpu", "device_name": "CPU",
                                     "model_core_ms": core_ms})
    if session is not None:
        session.verify("post_timing")
        session.write()
    # a query that failed in a later repetition has no valid latency set
    latency_rows = [row for row in latency_rows if row["query_id"] not in failures]
    prediction_rows = []
    for query_id in range(len(rows)):
        if query_id in unsupported:
            prediction_rows.append({"query_id": query_id, "estimated_cardinality": "", "status": "unsupported",
                                    "diagnostic": unsupported[query_id]})
        elif query_id in failures:
            prediction_rows.append({"query_id": query_id, "estimated_cardinality": "", "status": "failed",
                                    "diagnostic": failures[query_id]})
        else:
            prediction_rows.append({"query_id": query_id, "estimated_cardinality": repr(estimates[query_id]),
                                    "status": "ok", "diagnostic": notes[query_id]})
    write_csv(predictions_path, prediction_rows, ("query_id", "estimated_cardinality", "status", "diagnostic"))
    write_csv(latency_path, latency_rows,
              ("query_id", "repetition", "latency_ms", "scope", "device", "device_name", "model_core_ms"))
    print(json.dumps({"query_count": len(rows), "ok": len(estimates), "unsupported": len(unsupported),
                      "failed": len(failures), "latency_scope": LATENCY_SCOPE}, sort_keys=True))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
