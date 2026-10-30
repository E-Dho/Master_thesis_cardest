"""Shared base of the in-process trajectory estimators.

An in-process estimator builds a model artifact from the shared snapshot
(``artifacts.snapshot_root``) and answers v3 queries one at a time. Subclasses
implement :meth:`_build_model` (returns the arrays to store),
:meth:`_load_model` and :meth:`_estimate`; this class handles the snapshot
and v3 config, the model file in the run directory, the evaluate loop with
warm-up and repetitions, and profile-controlled timing under the core's
timing guard.

Config keys:

``artifacts.snapshot_root``
    the P2 snapshot directory (relative paths resolve against the config file).
``adapter.benchmark_config``
    the v3 config JSON the snapshot was exported for; the snapshot's
    ``config_sha256`` must match.
``adapter.verify_snapshot_checksums`` (default false)
    re-hash every snapshot array in ``prepare`` (a few minutes for POL).
``timing.warmup_passes`` / ``timing.repetitions``
    passes of the accuracy ``evaluate`` stage.
``timing.timing_warmup_passes`` / ``timing.timing_repetitions``
    passes of the profile-controlled ``time`` stage (default: the two above).
    Timing keys never change estimates or the accuracy hash, so a timing run
    may use more passes than its accuracy run.

``_estimate`` returns ``(status, estimate, diagnostic)``. Queries a method
cannot express are ``("unsupported", None, reason)``; an estimate that answers
an approximating question would use status ``ok`` with an ``approx:``
diagnostic (see ``strata.py``).
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from evaluation.eval_core.adapters.base import Adapter, AdapterEvaluation, StageMetrics, TimingOutcome
from evaluation.eval_core.config import WorkloadConfig
from evaluation.eval_core.records import LatencyRecord, PredictionRecord
from evaluation.eval_core.workload_formats import load_queries

from .. import snapshot as snapshot_module

Estimate = Tuple[str, Optional[float], str]
MODEL_FILE = "model.npz"
MODEL_META = "model_meta.json"


def query_row(record: Any) -> Dict[str, Any]:
    """The minimal v3 row ``NumpyBackend.count`` reads, from a workload record."""
    return {
        "predicates": list(record.predicates),
        "tables": list(record.tables),
        "category": {"relation": record.relation},
    }


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class InProcessTrajectoryAdapter(Adapter):
    #: latency scope recorded with every timed query
    latency_scope = "estimate"
    #: role recorded in the timing-guard report
    timing_role = "in_process_trajectory_estimator"

    # -- configuration ------------------------------------------------------

    def _resolve(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.config.source_path.parent / path
        return path.resolve()

    @property
    def snapshot_root(self) -> Path:
        value = self.config.artifacts.get("snapshot_root")
        if not value:
            raise ValueError("artifacts.snapshot_root is required")
        return self._resolve(value)

    @property
    def settings(self) -> Any:
        if not hasattr(self, "_settings"):
            from query_generation.benchmark_v3.config import load_settings

            value = self.config.adapter.get("benchmark_config")
            if not value:
                raise ValueError("adapter.benchmark_config is required")
            self._settings = load_settings(self._resolve(value))
        return self._settings

    def _snapshot_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        return snapshot_module.load_snapshot(
            self.snapshot_root, expected_config_sha256=self.settings.sha256
        )

    @property
    def model_directory(self) -> Path:
        return self.run_directory / "model"

    # -- subclass hooks -----------------------------------------------------

    def _build_model(self, data: Mapping[str, Mapping[str, np.ndarray]],
                     trips: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Arrays to store as the model, plus JSON metadata."""
        raise NotImplementedError

    def _load_model(self, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def _estimate(self, model: Any, record: Any) -> Estimate:
        raise NotImplementedError

    # -- stages -------------------------------------------------------------

    def prepare(self) -> StageMetrics:
        started = time.perf_counter()
        manifest = snapshot_module.read_manifest(self.snapshot_root)
        if manifest.get("config_sha256") != self.settings.sha256:
            raise ValueError("snapshot was exported under a different v3 config")
        problems: List[str] = []
        if self.config.adapter.get("verify_snapshot_checksums") is True:
            problems = snapshot_module.verify_snapshot(self.snapshot_root)
            if problems:
                raise ValueError(f"snapshot checksum problems: {problems[:5]}")
        return StageMetrics(
            wall_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
            detail={
                "snapshot_root": str(self.snapshot_root),
                "snapshot_manifest_sha256": _sha256(self.snapshot_root / "manifest.json"),
                "snapshot_created_at_utc": manifest.get("created_at_utc"),
                "checksums_verified": self.config.adapter.get("verify_snapshot_checksums") is True,
                "note": "snapshot export is shared and reported once, outside this method's build time",
            },
        )

    def smoke(self, workloads: tuple[WorkloadConfig, ...], query_limit: int = 2) -> StageMetrics:
        started = time.perf_counter()
        model = self._load_model(*self._build_model(self._snapshot_data(),
                                                    snapshot_module.load_trip_index(self.snapshot_root)))
        results = []
        for workload in workloads:
            for record in list(load_queries(workload))[:query_limit]:
                status, estimate, diagnostic = self._estimate(model, record)
                if status == "failed":
                    raise ValueError(f"smoke failed on {workload.workload_id} query {record.query_id}: {diagnostic}")
                results.append({"workload": workload.workload_id, "query_id": record.query_id,
                                "status": status, "estimate": estimate})
        return StageMetrics(wall_seconds=time.perf_counter() - started, peak_rss_bytes=_peak_rss_bytes(),
                            detail={"queries": results})

    def build(self) -> StageMetrics:
        started = time.perf_counter()
        arrays, meta = self._build_model(self._snapshot_data(),
                                         snapshot_module.load_trip_index(self.snapshot_root))
        self.model_directory.mkdir(parents=True, exist_ok=True)
        path = self.model_directory / MODEL_FILE
        np.savez(path, **arrays)
        meta = {**meta, "serialized_bytes": path.stat().st_size, "model_sha256": _sha256(path),
                "arrays": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in arrays.items()}}
        (self.model_directory / MODEL_META).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return StageMetrics(wall_seconds=time.perf_counter() - started, peak_rss_bytes=_peak_rss_bytes(),
                            detail={"model": str(path), **{k: v for k, v in meta.items() if k != "arrays"}})

    def _model(self) -> Any:
        path = self.model_directory / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; run the build stage first")
        meta = json.loads((self.model_directory / MODEL_META).read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return self._load_model(arrays, meta)

    def evaluate(self, workloads: tuple[WorkloadConfig, ...]) -> AdapterEvaluation:
        return self._evaluate(workloads)

    def time(self, profile: str, workloads: tuple[WorkloadConfig, ...], output_directory: Path,
             environment: dict[str, str]) -> TimingOutcome:
        from evaluation.eval_core.timing_guard import TimingSession

        report = output_directory / f"{self.config.adapter_type}_timing_environment.json"
        session = TimingSession(profile, report, role=self.timing_role)
        session.configure()
        evaluation = self._evaluate(workloads, session=session, profile=profile)
        session.verify("post_timing")
        session.write()
        return TimingOutcome(
            predictions=evaluation.predictions,
            latencies=evaluation.latencies,
            guard_reports={workload.workload_id: report for workload in workloads},
            detail=evaluation.detail,
        )

    def _evaluate(self, workloads: tuple[WorkloadConfig, ...], *, session: Any = None,
                  profile: str = "") -> AdapterEvaluation:
        model = self._model()
        warmup_passes = int(self.config.timing.get("warmup_passes", 1))
        repetitions = max(1, int(self.config.timing.get("repetitions", 1)))
        if session is not None:  # the profile-controlled timing stage
            warmup_passes = int(self.config.timing.get("timing_warmup_passes", warmup_passes))
            repetitions = max(1, int(self.config.timing.get("timing_repetitions", repetitions)))
        predictions: List[PredictionRecord] = []
        latencies: List[LatencyRecord] = []
        rejections: Dict[str, Dict[str, Any]] = {}
        if session is not None:
            session.verify("pre_timing")
        measured = session.measure(f"{self.config.adapter_type}_workloads") if session is not None else nullcontext()
        with measured:
            for workload in workloads:
                records = list(load_queries(workload))
                for _ in range(warmup_passes):
                    for record in records:
                        self._estimate(model, record)
                first: Dict[int, Estimate] = {}
                rejected_ms: List[float] = []
                for repetition in range(repetitions):
                    for record in records:
                        start = time.perf_counter()
                        result = self._estimate(model, record)
                        elapsed = (time.perf_counter() - start) * 1000.0
                        # Only answered queries are inference: the time to reject
                        # an unsupported (or failed) query is reported separately.
                        if result[0] == "ok":
                            latencies.append(LatencyRecord(
                                workload.workload_id, record.query_id, repetition, elapsed,
                                scope=self.latency_scope, device="cpu", profile=profile,
                            ))
                        else:
                            rejected_ms.append(elapsed)
                        if repetition == 0:
                            first[record.query_id] = result
                        elif result != first[record.query_id]:
                            raise RuntimeError(
                                f"{self.config.adapter_type} estimate changed between repetitions "
                                f"for query {record.query_id}"
                            )
                rejections[workload.workload_id] = {
                    "rejected_queries": sum(result[0] != "ok" for result in first.values()),
                    "timed_rejections": len(rejected_ms),
                    "mean_ms": float(np.mean(rejected_ms)) if rejected_ms else None,
                    "note": "not part of the inference latency",
                }
                for record in records:
                    status, estimate, diagnostic = first[record.query_id]
                    predictions.append(PredictionRecord(
                        workload.workload_id, record.query_id, status, record.true_cardinality,
                        None if estimate is None else float(estimate), diagnostic=diagnostic,
                    ))
        return AdapterEvaluation(tuple(predictions), tuple(latencies),
                                 detail={"model_meta": self._meta_without_arrays(),
                                         "rejections": rejections})

    def _meta_without_arrays(self) -> Dict[str, Any]:
        path = self.model_directory / MODEL_META
        if not path.exists():
            return {}
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.pop("arrays", None)
        return meta

    def artifact_metadata(self) -> dict[str, Any]:
        meta = self._meta_without_arrays()
        serialized = meta.get("serialized_bytes")
        budget = self.config.adapter.get("budget_bytes")
        return {
            "parameter_count": None,
            "parameter_count_reason": "non-parametric estimator (stored arrays)",
            "serialized_bytes": serialized,
            "serialized_model_mb": None if serialized is None else serialized / 1_000_000.0,
            "budget_bytes": None if budget in (None, "") else int(budget),
            "within_budget": None if budget in (None, "") or serialized is None else serialized <= int(budget),
            **{key: value for key, value in meta.items() if key not in ("serialized_bytes",)},
        }
