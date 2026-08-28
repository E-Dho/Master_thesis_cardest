from __future__ import annotations

import argparse
import json

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
    if result.best_checkpoint_path is not None:
        print(f"best_checkpoint={result.best_checkpoint_path}")
    print(f"parameter_count={result.parameter_count}")
    print(f"parameter_size_bytes={result.parameter_size_bytes}")
    print(f"backbone_parameter_count={result.backbone_parameter_count}")
    print(f"anpm_parameter_count={result.anpm_parameter_count}")
    print(f"first_loss={result.first_loss:.6f}")
    print(f"last_loss={result.last_loss:.6f}")
    print(f"total_sampled_tuples={result.total_sampled_tuples}")
    print(f"nominal_rows_seen={result.nominal_rows_seen}")
    print(f"training_seconds={result.training_seconds:.6f}")
    print(f"output_width_original={result.output_width_original}")
    print(f"output_width_factorized={result.output_width_factorized}")
    if result.peak_gpu_memory_bytes is not None:
        print(f"peak_gpu_memory_bytes={result.peak_gpu_memory_bytes}")
    if result.last_original_column_losses:
        print(
            "last_original_column_losses="
            f"{json.dumps(result.last_original_column_losses, sort_keys=True)}"
        )
    if result.last_factor_losses:
        print(
            "last_factor_losses="
            f"{json.dumps(result.last_factor_losses, sort_keys=True)}"
        )
    print(f"metrics_path={result.metrics_path}")
    print(f"summary_path={result.summary_path}")
    if result.early_stopping_summary.get("enabled"):
        print(
            "early_stopping_summary="
            f"{json.dumps(result.early_stopping_summary, sort_keys=True)}"
        )
    if result.validation_summary.get("enabled"):
        print(f"validation_summary={json.dumps(result.validation_summary, sort_keys=True)}")
    for fanout_name, stats in result.fanout_effective_sample_size.items():
        print(
            f"fanout_effective_sample_size[{fanout_name}]="
            f"last:{stats['last']:.6f},mean:{stats['mean']:.6f},"
            f"min:{stats['min']:.6f},max:{stats['max']:.6f}"
        )


if __name__ == "__main__":
    main()
