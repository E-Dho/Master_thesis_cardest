#!/usr/bin/env python3
"""Rewrite POL spatio-temporal workload rows to a same-segment measure.

The original generator may combine a trajectory-wide ``trip_geom`` predicate
with a trajectory-wide temporal predicate.  Such a row can pass because two
different segments satisfy the two conditions.  This tool freezes the sampled
bounds but moves every trip spatial/temporal predicate onto the joined segment
row, so all spatial and temporal conditions are evaluated for one segment.
"""

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TRIP_TO_SEGMENT = {
    "trip_geom": ("segment_geom", (("t.trip_geom", "s.segment_geom"),)),
    "trip_time": (
        "segment_time",
        (("t.start_time", "s.t_s"), ("t.end_time", "s.t_e")),
    ),
}


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def is_trip_spatiotemporal_predicate(predicate: Dict[str, Any]) -> bool:
    return predicate.get("table") == "trips" and predicate.get("attribute") in TRIP_TO_SEGMENT


def needs_segment_correction(row: Dict[str, Any]) -> bool:
    category = row.get("category", {})
    return category.get("dimension") == "spatio_temporal" and any(
        is_trip_spatiotemporal_predicate(predicate) for predicate in row.get("predicates", [])
    )


def rewrite_predicate(predicate: Dict[str, Any]) -> Dict[str, Any]:
    rewritten = copy.deepcopy(predicate)
    attribute = rewritten.get("attribute")
    if not is_trip_spatiotemporal_predicate(rewritten):
        return rewritten
    segment_attribute, replacements = TRIP_TO_SEGMENT[attribute]
    rewritten["table"] = "segments"
    rewritten["attribute"] = segment_attribute
    sql = str(rewritten["sql"])
    for old, new in replacements:
        sql = sql.replace(old, new)
    rewritten["sql"] = sql
    return rewritten


def build_from_clause(table_ids: Sequence[str]) -> Tuple[str, List[Dict[str, str]]]:
    """Build the canonical POL join tree for a connected table subset."""

    tables = {
        "agents": ("pol.agents", "a"),
        "trips": ("pol.trips", "t"),
        "segments": ("pol.segments", "s"),
    }
    joins = (
        {"left": "agents", "right": "trips", "condition": "a.agent_id = t.agent_id"},
        {"left": "trips", "right": "segments", "condition": "t.trip_id = s.trip_id"},
    )
    first = table_ids[0]
    table_name, alias = tables[first]
    from_sql = f"FROM {table_name} {alias}"
    included = {first}
    remaining = set(table_ids[1:])
    joins_used: List[Dict[str, str]] = []
    while remaining:
        for join in joins:
            left, right = join["left"], join["right"]
            if left in included and right in remaining:
                table_name, alias = tables[right]
                from_sql += f"\nJOIN {table_name} {alias} ON {join['condition']}"
                included.add(right)
                remaining.remove(right)
                joins_used.append(dict(join))
                break
            if right in included and left in remaining:
                table_name, alias = tables[left]
                from_sql += f"\nJOIN {table_name} {alias} ON {join['condition']}"
                included.add(left)
                remaining.remove(left)
                joins_used.append(dict(join))
                break
        else:
            raise ValueError(f"Could not connect POL table subset {table_ids}")
    return from_sql, joins_used


def rewrite_row(row: Dict[str, Any], source_workload: str) -> Dict[str, Any]:
    if not needs_segment_correction(row):
        raise ValueError(f"Query {row.get('query_id')} does not need segment correction")

    output = copy.deepcopy(row)
    original = {
        "category": copy.deepcopy(row["category"]),
        "tables": list(row["tables"]),
        "joins": copy.deepcopy(row["joins"]),
        "predicates": copy.deepcopy(row["predicates"]),
        "sql": row["sql"],
        "entity_sql": row.get("entity_sql"),
        "join_cardinality": row.get("join_cardinality"),
        "entity_cardinality": row.get("entity_cardinality"),
    }

    tables = tuple(sorted(set(row["tables"]) | {"segments"}))
    from_sql, joins = build_from_clause(tables)
    predicates = [rewrite_predicate(predicate) for predicate in row["predicates"]]
    where_sql = " AND ".join(f"({predicate['sql']})" for predicate in predicates)

    category = dict(row["category"])
    category["relation"] = "multi"
    category["key"] = f"{category['dimension']}.{category['interval']}.multi"
    output.update(
        {
            "category": category,
            "tables": list(tables),
            "joins": joins,
            "predicates": predicates,
            "sql": f"SELECT COUNT(*) AS join_cardinality\n{from_sql}\nWHERE {where_sql};",
            "entity_sql": f"SELECT COUNT(DISTINCT t.trip_id) AS entity_cardinality\n{from_sql}\nWHERE {where_sql};",
            "entity_key": "trip_id",
            "join_cardinality": None,
            "entity_cardinality": None,
            "join_evaluation_seconds": None,
            "entity_evaluation_seconds": None,
            "evaluated_at": None,
            "semantic_correction": {
                "name": "segment_coupled_spatio_temporal_v1",
                "description": "Trip spatial and temporal predicates were moved to one joined segments row.",
                "source_workload": source_workload,
                "original": original,
            },
            "semantic_rewritten_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create same-segment POL spatio-temporal workload rows.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    rows = list(load_jsonl(input_path))
    affected = [rewrite_row(row, str(input_path)) for row in rows if needs_segment_correction(row)]
    written = write_jsonl(Path(args.output), affected)
    direct_trip_geom = sum(
        any(predicate.get("table") == "trips" and predicate.get("attribute") == "trip_geom" for predicate in row["predicates"])
        for row in rows
        if needs_segment_correction(row)
    )
    summary = {
        "source_workload": str(input_path),
        "source_rows": len(rows),
        "rewritten_rows": written,
        "direct_trip_geom_rows": direct_trip_geom,
        "semantic_correction": "segment_coupled_spatio_temporal_v1",
        "all_rows_include_segments": all("segments" in row["tables"] for row in affected),
        "all_rows_have_entity_sql": all(row["entity_sql"] is not None for row in affected),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
