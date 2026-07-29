from __future__ import annotations

import argparse

from model.src.config import load_simple_yaml, validate_config
from model.src.data.sample_sources import sample_source_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predicate-conditioned ResMADE.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    source = sample_source_from_config(config)
    try:
        from model.src.training.resmade_trainer import train_resmade_sample_source
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    result = train_resmade_sample_source(source, config)
    print(f"checkpoint={result.checkpoint_path}")
    print(f"parameter_count={result.parameter_count}")
    print(f"first_loss={result.first_loss:.6f}")
    print(f"last_loss={result.last_loss:.6f}")


if __name__ == "__main__":
    main()
