from __future__ import annotations

import argparse

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import build_synthetic_chain_dataset
from model.src.evaluation.exact_evaluator import ExactOracle
from model.src.evaluation.metrics import q_error
from model.src.inference.estimator import OnePassEstimator
from model.src.predicates.generation import tokens_for_query_tables
from model.src.predicates.vocabulary import PredicateVocabularies


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicate-conditioned ResMADE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_simple_yaml(args.config)
    validate_config(config)
    try:
        from model.src.inference.torch_estimator import TorchDistributionModel
        from model.src.model.checkpoint import load_resmade_checkpoint
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    model, payload = load_resmade_checkpoint(args.checkpoint, map_location="cpu")
    metadata = payload["metadata"]
    from model.src.data.schema import ModelMetadata

    model_metadata = ModelMetadata.from_json_dict(metadata)
    requested_factorization = bool(config.get("factorization", {}).get("enabled", False))
    if requested_factorization != model_metadata.factorization_plan.enabled:
        raise SystemExit(
            "checkpoint factorization mode does not match config: "
            f"checkpoint={model_metadata.factorization_plan.enabled}, "
            f"config={requested_factorization}"
        )
    vocabularies = PredicateVocabularies.from_json_dict(payload["predicate_vocabularies"])
    tokens = tokens_for_query_tables(
        model_metadata,
        {column.table for column in model_metadata.columns if column.table is not None},
        {column.name for column in model_metadata.columns if column.kind.value == "fanout"},
    )
    wrapped = TorchDistributionModel(model, model_metadata, vocabularies)
    forward_calls_before = model.forward_calls
    result = OnePassEstimator(wrapped, model_metadata).estimate(
        tokens,
        use_log_space_product=bool(config["inference"].get("use_log_space_product", True)),
    )
    forward_calls = model.forward_calls - forward_calls_before
    print(f"estimated_cardinality={result.estimated_cardinality:.6f}")
    print(f"backbone_forward_seconds={wrapped.last_backbone_seconds:.8f}")
    print(f"anpm_decode_seconds={wrapped.last_decode_seconds:.8f}")
    print(f"inference_latency_seconds={result.latency_seconds:.8f}")
    print(f"model_forward_calls={forward_calls}")
    print(f"factorization_enabled={model_metadata.factorization_plan.enabled}")
    if model_metadata.factorization_plan.enabled:
        print(f"decode_chunk_size={model.anpm_config.decode_chunk_size}")
        print(
            "materialized_original_distributions="
            f"{model.anpm_config.materialize_original_distribution_for_debug}"
        )
    if config["dataset"]["type"] == "synthetic_full_join":
        dataset = build_synthetic_chain_dataset()
        true_inverse = ExactOracle(
            dataset.metadata, dataset.encoded_rows
        ).exact_weighted_product_for_fanouts(("F_A_to_B", "F_B_to_C"))
        true_cardinality = dataset.metadata.full_join_cardinality * true_inverse
        print(f"true_cardinality={true_cardinality:.6f}")
        print(f"q_error={q_error(result.estimated_cardinality, true_cardinality):.6f}")


if __name__ == "__main__":
    main()
