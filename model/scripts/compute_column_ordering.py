"""Choose an autoregressive column order from join statistics alone.

Draws full-outer-join rows through the same sampler a training run uses, then
ranks the DATA columns by a statistic of those rows and writes an ordering file
that ``permute_prepared_columns`` consumes.  Nothing here reads a query, a
result or a cardinality: an order derived from a previous run's errors would
make any improvement on that benchmark unfalsifiable.

    python3 -m model.scripts.compute_column_ordering \
        --config model/configs/<the control arm>.yaml \
        --rows 2000000 --sampler-seed 11 \
        --output-directory model/orderings/

Writes one file per criterion plus ``column_statistics.json`` holding the
figures they were ranked on, so an arm can be read back to the numbers that
chose it.

Criteria:

``coupling``
    DATA columns by descending total mutual information with the other DATA
    columns.  The model conditions each column on the predicates of the
    columns before it, so the column that says most about the rest goes first.
    This is the arm the trajectory benchmark's "harder columns first" result
    predicts.

``coupling_reversed``
    The same ranking, inverted.  It is the falsification arm: if ordering acts
    through coupling, this should be worse than ``coupling`` rather than
    merely different, and if both move the same way the statistic is not what
    is doing the work.

``entropy``
    DATA columns by descending marginal entropy, a cheaper story -- put the
    columns with most to predict where they are conditioned on most.

``domain_size``
    DATA columns by descending domain size.  Needs no sample at all and is the
    naive baseline the other three have to beat to be worth their cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.column_statistics import (
    bin_columns,
    data_column_indices,
    marginal_entropies,
    ordering_from_scores,
    pairwise_mutual_information,
    reversed_ordering,
)
from model.src.data.sample_sources import sample_source_from_config

CRITERIA = ("coupling", "coupling_reversed", "entropy", "domain_size")


def draw_rows(source: Any, total_rows: int, batch_size: int) -> np.ndarray:
    """Collect ``total_rows`` encoded rows from the sampler, in batches."""

    if total_rows <= 0:
        raise SystemExit("--rows must be positive")
    collected = []
    seen = 0
    while seen < total_rows:
        batch = source.batches(min(batch_size, total_rows - seen), seed=0)
        values = np.asarray(batch.encoded_values)
        if values.shape[0] == 0:
            raise SystemExit("the sampler returned no rows")
        collected.append(values.astype(np.int32, copy=False))
        seen += values.shape[0]
    rows = np.concatenate(collected, axis=0) if len(collected) > 1 else collected[0]
    return rows[:total_rows]


def build_orderings(metadata: Any, rows: np.ndarray, max_bins: int) -> dict[str, Any]:
    """Return every criterion's column order plus the figures behind them."""

    indices = data_column_indices(metadata)
    if not indices:
        raise SystemExit("the schema has no DATA columns to order")
    entropies = marginal_entropies(rows, metadata)
    binned = bin_columns(rows, metadata, max_bins=max_bins)
    matrix = pairwise_mutual_information(binned, indices)
    coupling = {index: float(matrix[position].sum()) for position, index in enumerate(indices)}
    entropy_scores = {index: float(entropies[index]) for index in indices}
    domain_scores = {
        index: float(metadata.columns[index].domain_size) for index in indices
    }
    coupling_order = ordering_from_scores(metadata, coupling)
    orders = {
        "coupling": coupling_order,
        # Built by reversing the arm it falsifies, so the two are opposites
        # even where the statistic ties.
        "coupling_reversed": reversed_ordering(coupling_order, indices),
        "entropy": ordering_from_scores(metadata, entropy_scores),
        "domain_size": ordering_from_scores(metadata, domain_scores),
    }
    names = [metadata.columns[index].name for index in indices]
    statistics = {
        "rows": int(rows.shape[0]),
        "max_bins": int(max_bins),
        "bins_used": {
            metadata.columns[index].name: binned.bin_counts[index] for index in indices
        },
        "data_columns": names,
        "marginal_entropy_bits": {
            metadata.columns[index].name: entropy_scores[index] for index in indices
        },
        "total_mutual_information_bits": {
            metadata.columns[index].name: coupling[index] for index in indices
        },
        "pairwise_mutual_information_bits": {
            names[first]: {
                names[second]: float(matrix[first, second])
                for second in range(len(names))
                if second != first
            }
            for first in range(len(names))
        },
    }
    return {"orders": orders, "statistics": statistics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the arm's training config")
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-bins", type=int, default=64)
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="override dataset.sampler_seed so the draw is reproducible and "
        "independent of the seed any arm trains with",
    )
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    if args.sampler_seed is not None:
        config.setdefault("dataset", {})["sampler_seed"] = int(args.sampler_seed)
    # Statistics describe the join, not a training objective: rare support
    # would reweight the stream towards the regions it targets and the order
    # would then be chosen from a distribution no arm actually trains on.
    config.setdefault("rare_support", {})["enabled"] = False
    config.setdefault("importance_sampling", {})["enabled"] = False

    source = sample_source_from_config(config)
    rows = draw_rows(source, int(args.rows), int(args.batch_size))
    report = build_orderings(source.metadata, rows, int(args.max_bins))

    directory = Path(args.output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": str(Path(args.config)),
        "sampler_seed": args.sampler_seed,
        "rows": int(rows.shape[0]),
        "max_bins": int(args.max_bins),
        "source": type(source).__name__,
    }
    for criterion in CRITERIA:
        payload = {
            "order": report["orders"][criterion],
            "criterion": criterion,
            **provenance,
        }
        (directory / f"{criterion}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (directory / "column_statistics.json").write_text(
        json.dumps({**provenance, **report["statistics"]}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {directory} from {rows.shape[0]} rows")
    for criterion in CRITERIA:
        moved = sum(
            1
            for position, name in enumerate(report["orders"][criterion])
            if name != source.metadata.columns[position].name
        )
        print(f"  {criterion}: {moved} columns moved")


if __name__ == "__main__":
    main()
