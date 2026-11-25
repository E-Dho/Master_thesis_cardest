#!/usr/bin/env python3
"""Measure one immutable inference bundle in an isolated, CPU-only process."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

BASE = Path("/work_beegfs/sunip956/master_thesis_trajectories")


def install_read_only_guard(output_directory: Path) -> dict[Path, tuple[int, int]]:
    """Reject writes outside generated reports/temp space and bind input versions."""
    reads: dict[Path, tuple[int, int]] = {}
    roots = [output_directory.resolve(), Path(os.environ.get("TMPDIR", str(output_directory))).resolve()]

    def writable(path: Path) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            if not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0])).absolute()
            mode, flags = args[1:3]
            writing = ((isinstance(mode, str) and any(char in mode for char in "wax+")) or
                       (isinstance(flags, int) and bool(flags &
                        (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))))
            if writing and not writable(path):
                raise PermissionError(f"read-only size measurement attempted write: {path}")
            if not writing and BASE in path.parents and path.is_file():
                stat = path.stat()
                reads.setdefault(path, (stat.st_size, stat.st_mtime_ns))
        elif event in ("os.remove", "os.rename", "os.rmdir", "os.mkdir"):
            path = Path(os.fsdecode(args[0])).absolute()
            if not writable(path) and not (event == "os.mkdir" and path.exists()):
                raise PermissionError(f"read-only size measurement attempted mutation: {path}")
            if event == "os.rename" and not writable(Path(os.fsdecode(args[1])).absolute()):
                raise PermissionError("read-only size measurement attempted rename into an input directory")

    sys.addaudithook(guard)
    return reads


def measure_request(request: dict[str, Any]) -> dict[str, Any]:
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.inference_size import foj_size, postgres_size
    method, artifact, variant = request["method"], request["artifacts"], request["variant"]
    if method == "postgresql":
        return postgres_size(artifact)
    if method == "foj_sampling":
        reservoir = Path(artifact["reservoir_path"])
        return foj_size(reservoir, reservoir.with_suffix(".json"), int(artifact["sample_size"]))
    if method == "mscn":
        from evaluation.job_light_imdb_non_trajectory.scripts.mscn_bridge import inference_artifact_metadata
        runtime = BASE / "joblight_evaluation/mscn_job_light_ranges_adapted" if "adapted" in variant else None
        return inference_artifact_metadata(Path(artifact["checkpoint"]), Path(artifact["metadata"]),
                                           BASE / "external/mscn", runtime,
                                           [BASE / "external/distjoin/queries" / ("job-light-ranges.csv" if runtime else "job-light.csv")])
    if method == "deepdb":
        from evaluation.job_light_imdb_non_trajectory.scripts.deepdb_bridge import inference_artifact_metadata
        rank_domains = BASE / "joblight_evaluation/deepdb_ranges_adapted_shared/rank_domains.json" if "adapted" in variant else None
        return inference_artifact_metadata(Path(artifact["checkpoint"]), rank_domains)
    if method == "distjoin":
        from evaluation.job_light_imdb_non_trajectory.scripts.distjoin_bridge import inference_artifact_metadata
        overlay = Path(artifact["checkpoint_directory"]).parents[3]
        return inference_artifact_metadata(BASE / "external/distjoin", overlay,
                                           BASE / "joblight_evaluation/distjoin_base_cardinalities.json")
    if method == "neurocard":
        from evaluation.job_light_imdb_non_trajectory.scripts.neurocard_bridge import _NativeRuntime, inference_artifact_metadata
        runtime = _NativeRuntime(source_root=BASE / "external/neurocard", dataset_root=BASE / "datasets/job",
                                 model_kind="job_light_ranges" if "ranges" in variant else "job_light",
                                 seed=request["seed"], device="cpu", cpu_threads=2,
                                 checkpoint=Path(artifact["checkpoint"]), training=False)
        return inference_artifact_metadata(runtime, Path(artifact["checkpoint"]))
    raise ValueError(f"unsupported method: {method}")


def main() -> None:
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.request, args.output = args.request.resolve(), args.output.resolve()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    reads = install_read_only_guard(args.output.parent)
    result = measure_request(request)
    existing = {item["path"] for item in result.get("inference_files", []) + result.get("inference_metadata_sources", [])}
    sources = []
    for path, before in sorted(reads.items()):
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != before:
            raise RuntimeError(f"input changed during size measurement: {path}")
        if (args.output.parent in path.parents or path.suffix not in (".table", ".jct", ".pkl", ".json", ".yaml", ".npy")
                or "envs" in path.parts or str(path.resolve()) in existing):
            continue
        sources.append({"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    result.setdefault("inference_metadata_sources", []).extend(sources)
    from evaluation.job_light_imdb_non_trajectory.joblight_eval.inference_size import measurement_version
    result.update(method=request["method"], variant=request["variant"], seed=request["seed"],
                  measurement_version=measurement_version(request["method"]))
    result["artifact_identity"] = {key: request["artifacts"][key] for key in
                                   ("checkpoint", "metadata", "checkpoint_directory", "reservoir_path", "sample_size", "database_manifest")
                                   if key in request["artifacts"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SIZE_RESULT", request["key"], result.get("inference_artifact_mb"), result.get("parameter_count"), flush=True)


if __name__ == "__main__":
    main()
