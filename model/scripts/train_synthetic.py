from __future__ import annotations

import argparse
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import build_synthetic_chain_dataset
from model.src.model.predicate_made import PredicateConditionedTableModel
from model.src.predicates.generation import tokens_for_query_tables
from model.src.training.losses import cumulative_inverse_fanout_weights, summarize_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the synthetic INV_FANOUT prototype.")
    parser.add_argument("--config", default="model/configs/inv_fanout_baseline.yaml")
    parser.add_argument("--checkpoint", default="model/examples/synthetic_checkpoint.json")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    dataset = build_synthetic_chain_dataset()
    token_rows = [
        tokens_for_query_tables(
            dataset.metadata,
            {"A", "B", "C"},
            {"F_A_to_B", "F_B_to_C"},
        )
        for _ in dataset.decoded_rows
    ]
    weights = cumulative_inverse_fanout_weights(
        dataset.encoded_rows,
        token_rows,
        dataset.metadata,
        compute_in_log_space=config["fanout"]["compute_weights_in_log_space"],
    )
    smoothing = float(config["training"]["smoothing"])
    model = PredicateConditionedTableModel(dataset.metadata, smoothing=smoothing)
    pre_loss, post_loss = model.fit_weighted_counts(dataset.encoded_rows, token_rows, weights)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint_path)
    fanout_weight_stats = summarize_weights(weights[:, -1])
    print(f"pre_loss={pre_loss:.6f}")
    print(f"post_loss={post_loss:.6f}")
    print(f"checkpoint={checkpoint_path}")
    print(f"last_fanout_ess={fanout_weight_stats.effective_sample_size:.6f}")


if __name__ == "__main__":
    main()

