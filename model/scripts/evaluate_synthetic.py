from __future__ import annotations

import argparse

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import build_synthetic_chain_dataset
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.evaluation.metrics import q_error
from model.src.inference.estimator import OnePassEstimator
from model.src.model.predicate_made import PredicateConditionedTableModel
from model.src.predicates.generation import tokens_for_query_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the synthetic INV_FANOUT prototype.")
    parser.add_argument("--config", default="model/configs/inv_fanout_baseline.yaml")
    parser.add_argument("--checkpoint", default="model/examples/synthetic_checkpoint.json")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    dataset = build_synthetic_chain_dataset()
    model = PredicateConditionedTableModel.load(args.checkpoint)
    tokens = tokens_for_query_tables(
        dataset.metadata,
        {"A", "B", "C"},
        {"F_A_to_B", "F_B_to_C"},
    )
    estimate = OnePassEstimator(model, dataset.metadata).estimate(
        tokens,
        use_log_space_product=config["inference"]["use_log_space_product"],
    )
    true_inverse_expectation = ExactOracle(dataset.metadata, dataset.encoded_rows).exact_weighted_product_for_fanouts(
        ("F_A_to_B", "F_B_to_C")
    )
    true_cardinality_factor = dataset.metadata.full_join_cardinality * true_inverse_expectation
    print(f"estimated_cardinality={estimate.estimated_cardinality:.6f}")
    print(f"true_cardinality={true_cardinality_factor:.6f}")
    print(f"q_error={q_error(estimate.estimated_cardinality, true_cardinality_factor):.6f}")
    print(f"inference_latency_seconds={estimate.latency_seconds:.8f}")


if __name__ == "__main__":
    main()

