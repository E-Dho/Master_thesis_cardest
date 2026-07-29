from __future__ import annotations

import argparse
from dataclasses import asdict

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource, inspect_encoded_rows
from model.src.data.sample_sources import sample_source_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect full-join sampler metadata and batches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-rows", type=int, default=5)
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    source = sample_source_from_config(config)
    if isinstance(source, SyntheticFullJoinSampleSource):
        inspection = source.inspect(sample_rows=args.sample_rows)
    else:
        batch = source.batches(args.sample_rows, seed=0)
        inspection = inspect_encoded_rows(source.metadata, batch.encoded_values, batch.raw_values or ())
    for key, value in asdict(inspection).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

