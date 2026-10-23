#!/usr/bin/env python3
"""Merge the 16 category files into the v3 benchmark and enforce every gate.

The merge refuses to write ``benchmark.jsonl`` unless all checks pass
(SEMANTICS.md): exact grid, strictly positive labels of the right kind,
unique queries, the structural rules of v3, and that every stored SQL string
is exactly what the stored predicates render to.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from query_generation.generate_pol_segment_coupled_benchmark import write_json_atomic

from . import SEMANTICS_VERSION
from .config import Settings, load_settings
from .generate import run_paths
from .query_model import label_kind, label_sql, predicate_hash, predicate_sql
from .sampler import quotas

SPATIAL_KINDS = {"spatial", "spatio_temporal"}
TEMPORAL_KINDS = {"temporal", "spatio_temporal"}
STANDARD_KINDS = {"nominal_eq", "scalar_range", "scalar_unbounded"}


class GateError(ValueError):
    pass


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_row(settings: Settings, row: Dict[str, Any]) -> List[str]:
    """All structural problems of one accepted row (empty list = valid)."""

    problems: List[str] = []
    category = row["category"]
    relation, dimension = category["relation"], category["dimension"]
    predicates = row["predicates"]
    kinds = [p["kind"] for p in predicates]
    core = [p for p in predicates if p.get("role") == "core"]
    addons = [p for p in predicates if p.get("role") == "addon"]
    seg_table = settings.segments.table

    if row.get("semantics_version") != SEMANTICS_VERSION:
        problems.append("semantics_version")
    card = row.get("cardinality")
    if not isinstance(card, int) or card <= 0:
        problems.append("cardinality not a positive integer")
    if row.get("label_kind") != label_kind(relation):
        problems.append("label_kind does not match relation")
    if relation == "multi" and (row.get("entity_cardinality") != card or row.get("join_cardinality") is not None):
        problems.append("multi rows must carry only the distinct-trip label")
    if relation == "single" and (row.get("join_cardinality") != card or row.get("entity_cardinality") is not None):
        problems.append("single rows must carry only the row-count label")

    if sum(k in SPATIAL_KINDS for k in kinds) > 1:
        problems.append("more than one spatial window")
    if sum(k in TEMPORAL_KINDS for k in kinds) > 1:
        problems.append("more than one temporal interval")
    for p in predicates:
        if p["kind"] in SPATIAL_KINDS and p["table"] != seg_table:
            problems.append(f"{p['kind']} predicate outside {seg_table}")
        if p["table"] not in row["tables"]:
            problems.append(f"predicate on table {p['table']} that is not joined")
        if "trip_geom" in p.get("sql", "") or "tgeom" in p.get("sql", ""):
            problems.append("trip geometry or tgeom used")
        if p.get("sql") != predicate_sql(settings, p):
            problems.append(f"predicate SQL of {p['attribute']} does not match its parameters")

    expected_core = {
        "standard": lambda: core and all(p["kind"] in STANDARD_KINDS for p in core),
        "temporal": lambda: [p["kind"] for p in core] == ["temporal"],
        "spatial": lambda: [p["kind"] for p in core] == ["spatial"],
        "spatio_temporal": lambda: [p["kind"] for p in core] == ["spatio_temporal"],
    }[dimension]
    if not expected_core():
        problems.append(f"core predicates do not match dimension {dimension}")
    if len(core) + len(addons) != len(predicates):
        problems.append("predicate without role")
    if addons and (relation != "multi" or dimension == "standard"):
        problems.append("add-on predicates outside non-standard multi queries")
    if len(addons) != row.get("addon_predicates") or len(addons) != row.get("addon_target"):
        problems.append("add-on count differs from the slot's assigned add-on target")
    if any(p["kind"] not in STANDARD_KINDS or p["kind"] == "scalar_unbounded" for p in addons):
        problems.append("add-ons must be bounded standard predicates")

    if relation == "single" and len(row["tables"]) != 1:
        problems.append("single query with several tables")
    if relation == "multi" and (settings.entity_table not in row["tables"] or len(row["tables"]) < 2):
        problems.append("multi query must join the entity table and another table")

    unbounded = [p for p in core if p.get("mode") == "unbounded"]
    if category["interval"] == "unbounded" and not unbounded:
        problems.append("unbounded category without a one-sided core component")
    if category["interval"] == "range" and unbounded:
        problems.append("range category with a one-sided component")

    if row.get("sql") != label_sql(settings, relation, row["tables"], predicates):
        problems.append("label SQL does not match the predicates")
    if row.get("predicate_hash") != predicate_hash(category["key"], row["tables"], predicates):
        problems.append("predicate_hash does not match the predicates")
    return problems


def merge(settings: Settings, run_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    rejected_by_category: Dict[str, Counter] = {}
    problems: List[str] = []
    for category in settings.categories():
        paths = run_paths(run_dir, category)
        accepted = load_jsonl(paths["accepted"])
        rejected = load_jsonl(paths["rejected"])
        rejected_by_category[category.key] = Counter(row["reason"] for row in rejected)
        target = settings.count_for(category)
        if len(accepted) != target:
            problems.append(f"{category.key}: {len(accepted)} accepted rows, expected {target}")
        indexes = [row["slot"]["index"] for row in accepted]
        if indexes != list(range(len(accepted))):
            problems.append(f"{category.key}: slot indexes are not 0..n-1 in order")
        for row in accepted:
            if row["category"]["key"] != category.key:
                problems.append(f"{row['slot_id']}: stored in the file of {category.key}")
            for problem in check_row(settings, row):
                problems.append(f"{row['slot_id']}: {problem}")
        rows.extend(accepted)

    for category in settings.categories():
        if category.relation != "multi" or category.dimension == "standard":
            continue
        observed = Counter(row["addon_predicates"] for row in rows if row["category"]["key"] == category.key)
        counts, probabilities = settings.addon_distribution
        expected = {c: q for c, q in zip(counts, quotas(settings.count_for(category), probabilities)) if q}
        if dict(observed) != expected and len(rows) == settings.total_queries:
            problems.append(f"{category.key}: add-on split {dict(observed)} differs from quotas {expected}")

    for field in ("config_sha256", "pools_sha256"):
        values = {row.get(field) for row in rows}
        if len(values) > 1:
            problems.append(f"rows disagree on {field}: {sorted(map(str, values))}")
    if rows and rows[0].get("config_sha256") != settings.sha256:
        problems.append("rows were generated with a different config or grid")
    seeds = {row["slot"]["base_seed"] for row in rows}
    if len(seeds) > 1:
        problems.append(f"rows use different base seeds: {sorted(seeds)}")
    duplicates = [h for h, n in Counter(row["predicate_hash"] for row in rows).items() if n > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate queries across the benchmark")
    if len(rows) != settings.total_queries and not any("accepted rows" in p for p in problems):
        problems.append(f"{len(rows)} rows, expected {settings.total_queries}")
    if problems:
        raise GateError("\n".join(problems[:200]))

    for number, row in enumerate(rows, start=1):
        row["query_id"] = f"q{number:08d}"
    return {"rows": rows, "summary": build_summary(settings, rows, rejected_by_category)}


def _quantiles(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]
    return {"p05": pick(0.05), "p25": pick(0.25), "p50": pick(0.5), "p75": pick(0.75), "p95": pick(0.95)}


def build_summary(settings: Settings, rows, rejected_by_category) -> Dict[str, Any]:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]["key"]].append(row)
    categories = {}
    for key, items in by_category.items():
        cards = [row["cardinality"] for row in items]
        rejected = rejected_by_category.get(key, Counter())
        categories[key] = {
            "rows": len(items),
            "label_kind": items[0]["label_kind"],
            "cardinality": _quantiles(cards),
            "cardinality_min": min(cards),
            "cardinality_max": max(cards),
            "rejected": dict(rejected),
            "acceptance_rate": len(items) / (len(items) + rejected.get("zero", 0) + rejected.get("timeout", 0)),
            "label_seconds_total": sum(row["label_seconds"] for row in items),
            "addon_counts": dict(Counter(row["addon_predicates"] for row in items)),
            "center_sources": dict(Counter(p.get("center_source") for row in items for p in row["predicates"] if p.get("center_source"))),
            "tables": dict(Counter("+".join(row["tables"]) for row in items)),
        }
    widths = [
        p.get("width_seconds") or p.get("temporal_width_seconds")
        for row in rows for p in row["predicates"]
        if p["kind"] in TEMPORAL_KINDS and (p.get("width_seconds") or p.get("temporal_width_seconds"))
    ]
    spatial_widths = [
        p[f"width_{axis}_meters"]
        for row in rows for p in row["predicates"] for axis in ("x", "y")
        if p.get(f"width_{axis}_meters")
    ]
    return {
        "semantics_version": SEMANTICS_VERSION,
        "rows": len(rows),
        "spatial_width_meters": _quantiles(spatial_widths),
        "grid": settings.grid,
        "config_sha256": settings.sha256,
        "pools_sha256": rows[0]["pools_sha256"] if rows else None,
        "base_seed": rows[0]["slot"]["base_seed"] if rows else None,
        "label_kinds": dict(Counter(row["label_kind"] for row in rows)),
        "temporal_width_seconds": _quantiles(widths),
        "label_seconds_total": sum(row["label_seconds"] for row in rows),
        "categories": dict(sorted(categories.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--single-count", type=int)
    parser.add_argument("--multi-count", type=int)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(Path(args.config), single_count=args.single_count, multi_count=args.multi_count)
    try:
        result = merge(settings, Path(args.run_dir))
    except GateError as error:
        print(f"merge refused:\n{error}", file=sys.stderr)
        return 1
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    write_json_atomic(Path(args.summary), result["summary"])
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "categories"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
