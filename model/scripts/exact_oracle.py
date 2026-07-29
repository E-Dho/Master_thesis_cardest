from __future__ import annotations

from model.src.data.full_join_sampler import build_synthetic_chain_dataset
from model.src.evaluation.exact_evaluator import ExactOracle


def main() -> None:
    dataset = build_synthetic_chain_dataset()
    oracle = ExactOracle(dataset.metadata, dataset.encoded_rows)
    value = oracle.exact_weighted_product_for_fanouts(("F_A_to_B", "F_B_to_C"))
    print(f"E[1/(F_A_to_B*F_B_to_C)]={value:.6f}")


if __name__ == "__main__":
    main()

