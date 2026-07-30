from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.complete_domain_preparation import (
    CompleteDomainSpec,
    build_complete_metadata,
    build_manifest_payload,
    encode_sample_dataframe,
    preparation_stats,
    source_csv_fingerprints,
    validate_prepared_manifest,
    write_prepared_artifacts,
)
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.src.model.factorization import FactorizationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NeuroCard full-join sampler artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--rebuild-domains",
        action="store_true",
        help="rebuild manifest domains from complete base tables and join metadata",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="optional validation sample size written to sample_rows.npy",
    )
    parser.add_argument(
        "--neurocard-path",
        default=None,
        help="path to the NeuroCard python package directory",
    )
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    dataset = config["dataset"]
    prepared_directory = Path(dataset["prepared_directory"])
    prepared_directory.mkdir(parents=True, exist_ok=True)
    if dataset["type"] == "synthetic_full_join":
        source = SyntheticFullJoinSampleSource()
        manifest = {
            "dataset_name": dataset["name"],
            "dataset_type": dataset["type"],
            "join_cardinality": source.join_cardinality,
            "metadata": source.metadata.to_json_dict(),
            "source": "synthetic_full_join",
            "metadata_source": "synthetic_complete_rows",
            "sample_source": "synthetic_full_join",
            "sample_rows": len(source.dataset.encoded_rows),
            "domains_complete": True,
            "format_version": 2,
        }
        (prepared_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"prepared_manifest={prepared_directory / 'manifest.json'}")
        return

    if dataset["type"] != "neurocard_full_join":
        raise SystemExit(f"unsupported dataset.type {dataset['type']!r}")
    manifest_path = prepared_directory / "manifest.json"
    if manifest_path.exists() and not args.rebuild_domains:
        try:
            validate_prepared_manifest(prepared_directory)
        except Exception as exc:
            raise SystemExit(
                f"existing manifest is not a valid complete-domain manifest: {exc}\n"
                "Rebuild it explicitly with --rebuild-domains."
            ) from exc
        print(f"prepared_manifest={manifest_path}")
        print("domains_complete=true")
        return
    csv_directory = Path(dataset["csv_directory"])
    if not csv_directory.exists():
        raise SystemExit(
            f"missing CSV directory {csv_directory}. Download/export dataset CSVs there first."
        )
    csv_files = sorted(csv_directory.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no CSV files found in {csv_directory}")
    missing_headers = [path for path in csv_files if not path.read_text(encoding="utf-8").splitlines()]
    if missing_headers:
        raise SystemExit(f"empty CSV files without headers: {missing_headers}")
    if not args.rebuild_domains:
        raise SystemExit(
            "No complete-domain manifest exists. Run this command again with "
            "--rebuild-domains to construct domains from complete base tables."
        )
    artifacts = _prepare_neurocard_job_light(
        config,
        csv_directory=csv_directory,
        prepared_directory=prepared_directory,
        sample_rows=(
            args.sample_rows
            if args.sample_rows is not None
            else int(dataset.get("sample_batch_size", 512))
        ),
        neurocard_path=args.neurocard_path,
    )
    print(f"prepared_manifest={artifacts.manifest_path}")
    print(f"sample_rows={artifacts.encoded_rows.shape[0]}")
    print(f"preparation_stats={artifacts.stats_path}")
    print(f"schema_hash={artifacts.metadata.stable_schema_hash()}")
    print(
        "original_output_width="
        f"{artifacts.stats['total_original_output_width']}"
    )
    print(
        "factorized_output_width="
        f"{artifacts.stats['total_factorized_output_width']}"
    )


def _prepare_neurocard_job_light(
    config: dict[str, Any],
    *,
    csv_directory: Path,
    prepared_directory: Path,
    sample_rows: int,
    neurocard_path: str | None,
) -> Any:
    """Load complete JOB-light tables, sample validation rows, and write artifacts."""

    if sample_rows <= 0:
        raise SystemExit("--sample-rows must be positive")
    neurocard_package = _resolve_neurocard_package(neurocard_path)
    import sys

    sys.path.insert(0, str(neurocard_package))
    import datasets  # type: ignore
    import experiments  # type: ignore
    import factorized_sampler  # type: ignore
    import join_utils  # type: ignore

    cfg = experiments.JOB_LIGHT_BASE
    spec = join_utils.get_join_spec(cfg)
    if config.get("dataset", {}).get("name") not in {"job_light", "job_light_factorized_anpm"}:
        print(
            "warning=using NeuroCard JOB_LIGHT_BASE because only JOB-light "
            "complete-domain preparation is currently implemented"
        )
    with _pushd(neurocard_package.parent):
        tables = [
            datasets.LoadImdb(
                table,
                data_dir=str(csv_directory) + "/",
                use_cols=cfg["use_cols"],
                try_load_parsed=True,
            )
            for table in spec.join_tables
        ]
        table_by_name = {table.name: table for table in tables}
        join_cardinality = float(
            datasets.JoinOrderBenchmark.GetFullOuterCardinalityOrFail(spec.join_tables)
        )
        complete_spec = CompleteDomainSpec(
            join_tables=tuple(spec.join_tables),
            join_root=spec.join_root,
            join_keys={
                table: tuple(keys) for table, keys in spec.join_keys.items()
            },
            join_cardinality=join_cardinality,
            dataset_name=config["dataset"]["name"],
            dataset_type=config["dataset"]["type"],
        )
        metadata = build_complete_metadata(table_by_name, complete_spec)
        sampler = factorized_sampler.FactorizedSampler(
            tables,
            spec,
            sample_rows,
            rng=np.random.default_rng(0),
            disambiguate_column_names=True,
        )
        sample_frame = sampler.run()
        if int(sampler.join_card) != int(join_cardinality):
            raise ValueError(
                f"sampler join cardinality {sampler.join_card} does not match "
                f"static JOB-light cardinality {join_cardinality}"
            )
    encoded_sample = encode_sample_dataframe(sample_frame, metadata, strict=True)
    csv_paths = [csv_directory / f"{table}.csv" for table in spec.join_tables]
    fingerprints = source_csv_fingerprints(csv_paths)
    manifest = build_manifest_payload(
        metadata=metadata,
        spec=complete_spec,
        sample_rows=encoded_sample.encoded_rows.shape[0],
        source_csv_fingerprints=fingerprints,
    )
    stats = preparation_stats(
        metadata=metadata,
        encoded_sample=encoded_sample,
        factorization_config=FactorizationConfig.from_dict(config.get("factorization", {})),
        source_csv_fingerprints=fingerprints,
    )
    (prepared_directory / "join_counts").mkdir(parents=True, exist_ok=True)
    (prepared_directory / "indexes").mkdir(parents=True, exist_ok=True)
    return write_prepared_artifacts(
        prepared_directory=prepared_directory,
        manifest_payload=manifest,
        encoded_rows=encoded_sample.encoded_rows,
        stats=stats,
    )


def _resolve_neurocard_package(explicit_path: str | None) -> Path:
    """Find the NeuroCard package used for JOB-light preparation."""

    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get("NEUROCARD_PATH"):
        candidates.append(Path(os.environ["NEUROCARD_PATH"]))
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "external/neurocard/neurocard",
            cwd.parent / "external/neurocard/neurocard",
            Path("/work_beegfs/sunip956/master_thesis_trajectories/external/neurocard/neurocard"),
        ]
    )
    for candidate in candidates:
        if (candidate / "datasets.py").exists() and (candidate / "factorized_sampler.py").exists():
            return candidate.resolve()
    raise SystemExit(
        "could not locate NeuroCard package. Pass --neurocard-path or set "
        "NEUROCARD_PATH to the directory containing datasets.py."
    )


@contextlib.contextmanager
def _pushd(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
