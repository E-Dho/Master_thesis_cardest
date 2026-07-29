from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NeuroCard full-join sampler artifacts.")
    parser.add_argument("--config", required=True)
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
        }
        (prepared_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"prepared_manifest={prepared_directory / 'manifest.json'}")
        return

    if dataset["type"] != "neurocard_full_join":
        raise SystemExit(f"unsupported dataset.type {dataset['type']!r}")
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
    expected = [
        prepared_directory / "join_counts",
        prepared_directory / "indexes",
        prepared_directory / "manifest.json",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit(
            "NeuroCard preparation wrapper validated CSV presence, but required "
            f"Exact Weight artifacts are missing: {missing}. Build or rsync "
            "NeuroCard join-count/index artifacts on the cluster, then rerun."
        )
    print(f"prepared_manifest={prepared_directory / 'manifest.json'}")


if __name__ == "__main__":
    main()

