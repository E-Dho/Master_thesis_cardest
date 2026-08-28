from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any

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
        if key == "fanout_domains":
            value = {
                name: _domain_summary(domain)
                for name, domain in value.items()
            }
        print(f"{key}: {value}")
    plan = source.metadata.factorization_plan
    if plan.enabled:
        print(f"factorization_enabled: {plan.enabled}")
        print(f"factorization_strategy: {plan.strategy}")
        print(f"original_output_width: {plan.original_output_width}")
        print(f"factorized_output_width: {plan.factorized_output_width}")
        ratio = plan.factorized_output_width / max(plan.original_output_width, 1)
        print(f"factorized_output_ratio: {ratio:.6f}")
        for factorization in plan.original_column_factorizations:
            column = source.metadata.columns[factorization.original_column_index]
            print(
                "factorized_column: "
                f"{column.name}, factors={len(factorization.factor_column_indices)}, "
                f"domains={factorization.factor_domains}, "
                f"bit_widths={factorization.bit_widths}, "
                f"invalid_combinations={factorization.invalid_combination_count}"
            )
        for column_index, column in enumerate(source.metadata.columns):
            if column.kind.value != "data":
                continue
            factorization = plan.factorization_for_column(column_index)
            if factorization is None:
                print(
                    "ordinary_column_factorization: "
                    f"{column.name}, original_domain_size={column.domain_size}, "
                    "selected_for_factorization=false, factor_count=1, "
                    f"factor_domains=({column.domain_size},)"
                )
            else:
                print(
                    "ordinary_column_factorization: "
                    f"{column.name}, original_domain_size={column.domain_size}, "
                    "selected_for_factorization=true, "
                    f"factor_count={len(factorization.factor_column_indices)}, "
                    f"factor_domains={factorization.factor_domains}"
                )
    try:
        from model.src.training.resmade_trainer import build_resmade_from_config

        model = build_resmade_from_config(source.metadata, config)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        print(f"estimated_parameter_count: {model.parameter_count()}")
        print(f"estimated_parameter_size_bytes: {int(parameter_bytes)}")
        print(f"estimated_adam_state_bytes: {int(parameter_bytes * 2)}")
    except ImportError as exc:
        print(f"estimated_parameter_count: unavailable ({exc})")


def _domain_summary(domain: Any) -> Any:
    values = tuple(domain)
    if len(values) <= 16:
        return values
    return {
        "size": len(values),
        "min": min(values),
        "max": max(values),
        "first": values[:4],
        "last": values[-4:],
    }


if __name__ == "__main__":
    main()
