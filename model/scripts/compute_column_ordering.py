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
    DATA columns by descending marginal entropy.  Same mechanism as
    ``coupling``, read through one column at a time: an early column's own
    factor is an unconditioned marginal while its predicate conditions every
    later factor, so the column carrying the most information is the one whose
    predicate is worth conditioning the rest on.

``entropy_reversed``
    The inverse, and not merely a formality.  The competing hypothesis is that
    a high-entropy column is the one that most needs conditioning itself, which
    argues for placing it last, where its factor is conditioned on everything.
    The two readings predict opposite orders, so both are generated and the
    runs decide rather than the wording.

``domain_size``
    DATA columns by descending domain size.  Needs no sample at all and is the
    naive baseline the other three have to beat to be worth their cost.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

CRITERIA = (
    "coupling",
    "coupling_reversed",
    "entropy",
    "entropy_reversed",
    "domain_size",
)


@dataclass(frozen=True)
class DrawnRows:
    """The sample, plus how much of it the sampler actually drew fresh."""

    rows: np.ndarray
    fresh_rows_drawn: int
    fixture_rows_reused: int


def resolve_sampler_seed(config: dict[str, Any], requested: int | None) -> int:
    """Return the one seed the sampler and the batch sequence both use.

    Written back into the config as well as returned, because the live source
    takes its seed from ``dataset.sampler_seed`` at construction while the
    batch sequence takes it from here.  Leaving the two to be resolved
    separately is how a run was built with the config's seed, stepped through
    batches from zero, and reported null.  The fallback chain is
    sample_source_from_config's own, so omitting the flag changes nothing about
    which sample is drawn -- only that the provenance now names it.
    """

    dataset = config.setdefault("dataset", {})
    if requested is not None:
        seed = int(requested)
    elif "sampler_seed" in dataset:
        seed = int(dataset["sampler_seed"])
    else:
        seed = int(config.get("training", {}).get("seed", 0))
    dataset["sampler_seed"] = seed
    return seed


def draw_rows(
    source: Any, total_rows: int, batch_size: int, *, base_seed: int = 0
) -> DrawnRows:
    """Collect ``total_rows`` encoded rows from the sampler, in batches.

    Each batch gets its own seed.  The fixture and synthetic sources build a
    fresh ``default_rng(seed)`` per call and index into their rows with it, so
    passing one seed every time makes a two-million-row sample thirty-one
    copies of the same sixty-five-thousand-row draw while the report still says
    two million.  The live NeuroCard source ignores the argument and advances
    its own sampler, so it is unaffected either way.
    """

    if total_rows <= 0:
        raise SystemExit("--rows must be positive")
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    collected = []
    seen = 0
    fresh = 0
    reused = 0
    while seen < total_rows:
        batch = source.batches(
            min(batch_size, total_rows - seen), seed=int(base_seed) + len(collected)
        )
        values = np.asarray(batch.encoded_values)
        returned = int(values.shape[0])
        if returned == 0:
            raise SystemExit("the sampler returned no rows")
        kept = min(returned, total_rows - seen)
        batch_fresh = int(getattr(batch, "fresh_rows_drawn", 0) or 0)
        batch_reused = int(getattr(batch, "fixture_rows_reused", 0) or 0)
        if kept < returned:
            # Only rows that stay in the sample are counted.  The two counters
            # partition a batch for both shipped sources -- every row is either
            # freshly drawn or resampled from the fixture -- so the kept part
            # is split in that same proportion.  A source that accounts for its
            # batch some other way is capped rather than rescaled, which
            # understates rather than invents.
            if batch_fresh + batch_reused == returned:
                batch_fresh = round(batch_fresh * kept / returned)
                batch_reused = kept - batch_fresh
            else:
                batch_fresh = min(batch_fresh, kept)
                batch_reused = min(batch_reused, kept)
        collected.append(values[:kept].astype(np.int32, copy=False))
        seen += kept
        fresh += batch_fresh
        reused += batch_reused
    rows = np.concatenate(collected, axis=0) if len(collected) > 1 else collected[0]
    return DrawnRows(rows, fresh, reused)


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
    entropy_order = ordering_from_scores(metadata, entropy_scores)
    orders = {
        "coupling": coupling_order,
        # Built by reversing the arm it falsifies, so the two are opposites
        # even where the statistic ties.
        "coupling_reversed": reversed_ordering(coupling_order, indices),
        "entropy": entropy_order,
        "entropy_reversed": reversed_ordering(entropy_order, indices),
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
    effective_seed = resolve_sampler_seed(config, args.sampler_seed)
    # Statistics describe the join, not a training objective: rare support
    # would reweight the stream towards the regions it targets and the order
    # would then be chosen from a distribution no arm actually trains on.
    config.setdefault("rare_support", {})["enabled"] = False
    config.setdefault("importance_sampling", {})["enabled"] = False

    source = sample_source_from_config(config)
    drawn = draw_rows(
        source, int(args.rows), int(args.batch_size), base_seed=effective_seed
    )
    rows = drawn.rows
    report = build_orderings(source.metadata, rows, int(args.max_bins))

    directory = Path(args.output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "config": str(Path(args.config)),
        "sampler_seed": effective_seed,
        "sampler_seed_requested": args.sampler_seed,
        "rows": int(rows.shape[0]),
        # The seed sequence runs one seed per batch, so the batch size is part
        # of what has to be repeated to draw this sample again.
        "batch_size": int(args.batch_size),
        "max_bins": int(args.max_bins),
        "source": type(source).__name__,
        # A fixture source resamples a fixed set of rows with replacement, so a
        # large sample is not a large number of distinct rows.  Saying which it
        # was keeps the figures honest about what they were estimated from.
        "fresh_rows_drawn": drawn.fresh_rows_drawn,
        "fixture_rows_reused": drawn.fixture_rows_reused,
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
