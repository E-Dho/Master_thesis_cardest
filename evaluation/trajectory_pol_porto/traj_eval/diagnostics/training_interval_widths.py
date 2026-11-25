#!/usr/bin/env python3
"""Audit time-interval widths emitted by the POL training-context generator.

This diagnostic deliberately reuses the training configuration, prepared
fixture source, row sampler, and ``PredicateTrainingContextGenerator``.  It
does not construct a model or change model behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.sample_sources import sample_source_from_config
from model.src.data.trajectory_distinct import SegmentTemporalPredicate
from model.src.predicates.generation import (
    GeneratedTrainingContext,
    PredicateTrainingContextGenerator,
)
from model.src.predicates.operators import PredicateOp, PredicateToken


SECONDS_PER_DAY = 86_400.0
BIN_ORDER = (
    "inside",
    "lt_1_min",
    "1_min_to_1_h",
    "1_h_to_1_day",
    "1_to_30_days",
    "30_to_180_days",
    "gt_180_days",
)
BIN_LABELS = {
    "inside": "inside (U < L)",
    "lt_1_min": "< 1 min",
    "1_min_to_1_h": "1 min-1 h",
    "1_h_to_1_day": "1 h-1 day",
    "1_to_30_days": "1-30 days",
    "30_to_180_days": "30-180 days",
    "gt_180_days": "> 180 days",
}


def main_predicate_rng_seed(generator_seed: int, global_step: int) -> int:
    """Match ``resmade_trainer.main_predicate_rng_seed`` without importing torch."""
    return int(generator_seed) + int(global_step)


def interval_report_key(kind: str, source: str) -> str:
    """Map extractor labels onto the stable JSON report keys."""
    try:
        return {
            ("segment_time", "trajectory_temporal"): "segment_trajectory_temporal",
            ("segment_time", "ordinary"): "segment_ordinary",
            ("trip_time", "ordinary"): "trip_ordinary",
        }[(kind, source)]
    except KeyError as error:
        raise ValueError(f"unexpected interval source: {kind}/{source}") from error


def one_sided_report_key(key: str) -> str:
    """Map a serialized one-sided extractor key onto its report key."""
    try:
        return {
            "segment_time_ordinary": "segment_ordinary",
            "trip_time_ordinary": "trip_ordinary",
        }[key]
    except KeyError as error:
        raise ValueError(f"unexpected one-sided interval source: {key}") from error


@dataclass(frozen=True)
class IntervalObservation:
    """One paired lower/upper temporal interval found in a context."""

    kind: str
    source: str
    lower: float
    upper: float

    @property
    def width_seconds(self) -> float:
        return self.upper - self.lower


def _predicate_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Match the trainer's configuration hand-off exactly."""
    predicate_config = dict(config.get("predicate_generation", {}))
    if "trajectory_spatial" not in predicate_config:
        predicate_config["trajectory_spatial"] = dict(config.get("trajectory_spatial", {}))
    return predicate_config


def _seconds(value: Any) -> float:
    """Convert training literals or benchmark timestamps to epoch seconds."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.datetime64):
        return float(value.astype("datetime64[ns]").astype(np.int64)) / 1.0e9
    if isinstance(value, datetime):
        return float(value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp())
    if isinstance(value, str):
        normalized = value.replace(" ", "T")
        parsed = datetime.fromisoformat(normalized)
        return float(parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp())
    raise TypeError(f"unsupported temporal literal {value!r}")


def width_bin(width_seconds: float) -> str:
    if width_seconds < 0:
        return "inside"
    if width_seconds < 60:
        return "lt_1_min"
    if width_seconds < 3600:
        return "1_min_to_1_h"
    if width_seconds < SECONDS_PER_DAY:
        return "1_h_to_1_day"
    if width_seconds < 30 * SECONDS_PER_DAY:
        return "1_to_30_days"
    if width_seconds < 180 * SECONDS_PER_DAY:
        return "30_to_180_days"
    return "gt_180_days"


def _upper_bound(token: PredicateToken | None) -> Any | None:
    if token is None:
        return None
    if token.op in {PredicateOp.LESS_THAN, PredicateOp.LESS_EQUAL}:
        return token.value
    if token.op == PredicateOp.RANGE:
        return token.upper
    return None


def _lower_bound(token: PredicateToken | None) -> Any | None:
    if token is None:
        return None
    if token.op in {PredicateOp.GREATER_THAN, PredicateOp.GREATER_EQUAL}:
        return token.value
    if token.op == PredicateOp.RANGE:
        return token.value
    return None


def _semantic_temporal_columns(context: GeneratedTrainingContext) -> frozenset[str]:
    query = context.trajectory_query
    if query is None:
        return frozenset()
    return frozenset(
        column_name
        for predicate in getattr(query, "temporal_predicates", ())
        for column_name in (predicate.start_column, predicate.end_column)
    )


def extract_context_intervals(
    context: GeneratedTrainingContext,
) -> tuple[list[IntervalObservation], Counter[str], Counter[str]]:
    """Extract paired intervals, one-sided contexts, and equality tokens.

    Trajectory temporal predicates are counted separately. Their scalar tokens
    also live in ``ordinary_predicates`` for model encoding, so their owned
    columns are excluded from the ordinary-predicate measurement to avoid
    double counting.
    """
    observations: list[IntervalObservation] = []
    one_sided: Counter[str] = Counter()
    equalities: Counter[str] = Counter()
    query = context.trajectory_query
    for predicate in getattr(query, "temporal_predicates", ()) if query is not None else ():
        if not isinstance(predicate, SegmentTemporalPredicate):
            continue
        if predicate.lower is not None and predicate.upper is not None:
            observations.append(
                IntervalObservation(
                    kind="segment_time",
                    source="trajectory_temporal",
                    lower=_seconds(predicate.lower),
                    upper=_seconds(predicate.upper),
                )
            )
        elif predicate.lower is not None or predicate.upper is not None:
            one_sided["segment_trajectory_temporal"] += 1

    semantic_columns = _semantic_temporal_columns(context)
    ordinary = context.ordinary_predicates
    for column_name in (
        "segments:t_s",
        "segments:t_e",
        "trips:start_time",
        "trips:end_time",
    ):
        token = ordinary.get(column_name)
        if token is not None and token.op == PredicateOp.EQUAL:
            equalities[column_name] += 1

    for kind, source, start_name, end_name in (
        ("segment_time", "ordinary", "segments:t_s", "segments:t_e"),
        ("trip_time", "ordinary", "trips:start_time", "trips:end_time"),
    ):
        if start_name in semantic_columns or end_name in semantic_columns:
            continue
        upper = _upper_bound(ordinary.get(start_name))
        lower = _lower_bound(ordinary.get(end_name))
        key = f"{kind}_{source}"
        if upper is not None and lower is not None:
            observations.append(
                IntervalObservation(
                    kind=kind,
                    source=source,
                    lower=_seconds(lower),
                    upper=_seconds(upper),
                )
            )
        elif upper is not None or lower is not None:
            one_sided[key] += 1
    return observations, one_sided, equalities


def _empty_interval_stats() -> dict[str, Any]:
    return {
        "two_sided_context_count": 0,
        "one_sided_context_count": 0,
        "widths_seconds": [],
    }


def _finalize_interval_stats(
    stats: Mapping[str, Any],
    *,
    total_contexts: int,
    projected_contexts: int | None = None,
) -> dict[str, Any]:
    widths = [float(width) for width in stats["widths_seconds"]]
    bins = Counter(width_bin(width) for width in widths)
    count = len(widths)
    result = {
        "two_sided_context_count": count,
        "share_of_all_contexts": count / total_contexts if total_contexts else 0.0,
        "one_sided_context_count": int(stats["one_sided_context_count"]),
        "one_sided_share_of_all_contexts": (
            int(stats["one_sided_context_count"]) / total_contexts if total_contexts else 0.0
        ),
        "median_width_seconds": statistics.median(widths) if widths else None,
        "bins": {
            name: {
                "count": int(bins[name]),
                "share_of_two_sided": bins[name] / count if count else 0.0,
                "share_of_all_contexts": bins[name] / total_contexts if total_contexts else 0.0,
                "projected_count": (
                    bins[name] / total_contexts * projected_contexts
                    if total_contexts and projected_contexts is not None
                    else None
                ),
            }
            for name in BIN_ORDER
        },
    }
    result["projected_two_sided_context_count"] = (
        count / total_contexts * projected_contexts
        if total_contexts and projected_contexts is not None
        else None
    )
    return result


def analyze_training_contexts(
    *,
    source: Any,
    config: Mapping[str, Any],
    batches: int,
    seed: int,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Generate the same contexts as training and collect interval statistics."""
    if batches <= 0:
        raise ValueError("batches must be positive")
    training = config["training"]
    batch_size = int(training["batch_size"])
    predicate_config = _predicate_generation_config(config)
    generator_seed = int(predicate_config.get("seed", training.get("seed", 0)))
    generator = PredicateTrainingContextGenerator(predicate_config)
    if generator.strategy != "duet_batch_bounds":
        raise ValueError("diagnostic requires predicate_generation.strategy=duet_batch_bounds")
    per_row_contexts = int(generator.per_row_contexts)
    interval_stats = {
        "segment_trajectory_temporal": _empty_interval_stats(),
        "segment_ordinary": _empty_interval_stats(),
        "trip_ordinary": _empty_interval_stats(),
    }
    equality_counts: Counter[str] = Counter()
    total_contexts = 0
    rejected_contexts = 0
    for global_step in range(batches):
        batch = source.batches(batch_size, seed=seed + global_step)
        contexts, _rows, generation_stats = generator.generate_batch(
            encoded_rows=batch.encoded_values,
            metadata=source.metadata,
            rng=np.random.default_rng(main_predicate_rng_seed(generator_seed, global_step)),
        )
        rejected_contexts += int(generation_stats.rejected_unsatisfied_contexts)
        for context in contexts:
            total_contexts += 1
            observations, one_sided, equalities = extract_context_intervals(context)
            equality_counts.update(equalities)
            for key, count in one_sided.items():
                report_key = one_sided_report_key(key)
                interval_stats[report_key]["one_sided_context_count"] += count
            for observation in observations:
                key = interval_report_key(observation.kind, observation.source)
                interval_stats[key]["widths_seconds"].append(observation.width_seconds)
        if progress_every and (global_step + 1) % progress_every == 0:
            print(f"[interval-widths] completed_batches={global_step + 1}/{batches}", flush=True)
    projected_contexts = int(training["steps_per_epoch"]) * batch_size * per_row_contexts
    return {
        "sampling": {
            "seed": int(seed),
            "batches": int(batches),
            "batch_size": batch_size,
            "per_row_contexts": per_row_contexts,
            "sampled_contexts": total_contexts,
            "projected_training_contexts": projected_contexts,
            "context_validation": "PredicateTrainingContextGenerator.generate_batch(validate_contexts=True)",
            "rejected_unsatisfied_contexts": rejected_contexts,
            "context_satisfaction_failures": 0,
        },
        "intervals": {
            key: _finalize_interval_stats(
                value,
                total_contexts=total_contexts,
                projected_contexts=projected_contexts,
            )
            for key, value in interval_stats.items()
        },
        "equality_predicates": {
            column: int(equality_counts[column])
            for column in ("segments:t_s", "segments:t_e", "trips:start_time", "trips:end_time")
        },
    }


def analyze_benchmark(path: Path) -> dict[str, Any]:
    """Measure core two-sided temporal intervals in the POL v3 workload."""
    grouped: dict[str, list[float]] = defaultdict(list)
    total_records = 0
    total_intervals = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            dimension = str(record.get("category", {}).get("dimension", "unknown"))
            if dimension not in {"temporal", "spatio_temporal"}:
                continue
            for predicate in record.get("predicates", ()):
                if predicate.get("role") != "core" or predicate.get("kind") != "temporal":
                    continue
                if predicate.get("lower") is None or predicate.get("upper") is None:
                    continue
                attribute = str(predicate.get("attribute"))
                if attribute not in {"segment_time", "trip_time"}:
                    continue
                width = float(predicate.get("width_seconds", _seconds(predicate["upper"]) - _seconds(predicate["lower"])))
                grouped[f"{attribute}/{dimension}"].append(width)
                total_intervals += 1
    return {
        "benchmark_path": str(path),
        "records": total_records,
        "two_sided_core_intervals": total_intervals,
        "intervals": {
            key: _finalize_interval_stats(
                {"widths_seconds": widths, "one_sided_context_count": 0},
                total_contexts=total_intervals,
            )
            for key, widths in sorted(grouped.items())
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.3g}%"


def _width_summary(stats: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{BIN_LABELS[name]}: {stats['bins'][name]['count']} ({_percent(stats['bins'][name]['share_of_two_sided'])})"
        for name in BIN_ORDER
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    training = report["training"]
    benchmark = report["benchmark"]
    lines = [
        "# POL Training Interval Widths",
        "",
        "## Training contexts",
        "",
        "| Interval kind | Two-sided contexts | Share of all contexts | One-sided | Median width | Width bins |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, stats in training["intervals"].items():
        median = "--" if stats["median_width_seconds"] is None else f"{stats['median_width_seconds']:.3g} s"
        lines.append(
            f"| {key} | {stats['two_sided_context_count']} | {_percent(stats['share_of_all_contexts'])} | "
            f"{stats['one_sided_context_count']} | {median} | {_width_summary(stats)} |"
        )
    lines += [
        "",
        "## POL v3 benchmark core predicates",
        "",
        "| Interval kind/category | Two-sided predicates | Share of all two-sided predicates | Median width | Width bins |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, stats in benchmark["intervals"].items():
        median = "--" if stats["median_width_seconds"] is None else f"{stats['median_width_seconds']:.3g} s"
        lines.append(
            f"| {key} | {stats['two_sided_context_count']} | {_percent(stats['share_of_all_contexts'])} | "
            f"{median} | {_width_summary(stats)} |"
        )
    return "\n".join(lines) + "\n"


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    source = sample_source_from_config(config)
    training = analyze_training_contexts(
        source=source,
        config=config,
        batches=args.batches,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    benchmark = analyze_benchmark(args.benchmark)
    report = {
        "git_revision": _git_revision(),
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "benchmark_path": str(args.benchmark),
        "training": training,
        "benchmark": benchmark,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "training_interval_widths.json"
    markdown_path = args.output_dir / "training_interval_widths.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()
