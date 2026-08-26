#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from query_generator import QueryExecutor


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stream_jsonl(path: Path, rows: Iterable[Dict[str, Any]], progress_interval: int) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            written.append(row)
            if progress_interval > 0 and len(written) % progress_interval == 0:
                print(f"evaluated {len(written)} queries", file=sys.stderr, flush=True)
    return written


def row_in_slice(row_number: int, start_index: int, end_index: Optional[int]) -> bool:
    if row_number < start_index:
        return False
    if end_index is not None and row_number >= end_index:
        return False
    return True


def evaluate_rows(
    rows: Iterable[Dict[str, Any]],
    executor: QueryExecutor,
    start_index: int,
    end_index: Optional[int],
) -> Iterable[Dict[str, Any]]:
    evaluated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row_number, row in enumerate(rows):
        if not row_in_slice(row_number, start_index, end_index):
            continue

        output = dict(row)
        join_started = time.monotonic()
        output["join_cardinality"] = int(executor.scalar(output["sql"]))
        output["join_evaluation_seconds"] = round(time.monotonic() - join_started, 6)

        entity_sql = output.get("entity_sql")
        if entity_sql:
            entity_started = time.monotonic()
            output["entity_cardinality"] = int(executor.scalar(entity_sql))
            output["entity_evaluation_seconds"] = round(time.monotonic() - entity_started, 6)
        else:
            output["entity_cardinality"] = None
            output["entity_evaluation_seconds"] = None

        output["evaluated_at"] = evaluated_at
        output["source_row_index"] = row_number
        yield output


def write_summary(path: Path, rows: List[Dict[str, Any]], input_path: Path, start_index: int, end_index: Optional[int]) -> None:
    category_counts = Counter(row["category"]["key"] for row in rows)
    multi = [row for row in rows if row["category"]["relation"] == "multi"]
    bad_multi = [
        row["query_id"]
        for row in multi
        if row["entity_cardinality"] is None
        or row["join_cardinality"] is None
        or row["entity_cardinality"] > row["join_cardinality"]
    ]
    summary = {
        "rows": len(rows),
        "input_jsonl": str(input_path),
        "start_index": start_index,
        "end_index": end_index,
        "category_counts": dict(sorted(category_counts.items())),
        "multi_rows": len(multi),
        "bad_multi_entity_gt_join": bad_multi,
        "min_join_cardinality": min(row["join_cardinality"] for row in rows) if rows else None,
        "max_join_cardinality": max(row["join_cardinality"] for row in rows) if rows else None,
        "join_evaluation_seconds": round(sum(row.get("join_evaluation_seconds") or 0 for row in rows), 6),
        "entity_evaluation_seconds": round(sum(row.get("entity_evaluation_seconds") or 0 for row in rows), 6),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if bad_multi:
        raise SystemExit("entity_cardinality exceeded join_cardinality")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen COUNT-query JSONL rows against PostgreSQL/MobilityDB.")
    parser.add_argument("--input", required=True, help="Input JSONL with sql/entity_sql fields.")
    parser.add_argument("--output", required=True, help="Output JSONL with cardinality fields populated.")
    parser.add_argument("--summary", help="Optional summary JSON path.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based input row offset to start from.")
    parser.add_argument("--end-index", type=int, help="Exclusive zero-based input row offset to stop before.")
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if args.end_index is not None and args.end_index <= args.start_index:
        raise SystemExit("--end-index must be greater than --start-index")

    input_path = Path(args.input)
    output_path = Path(args.output)
    with QueryExecutor(host=args.host, port=args.port, dbname=args.dbname, user=args.user) as executor:
        evaluated = stream_jsonl(
            output_path,
            evaluate_rows(load_jsonl(input_path), executor, args.start_index, args.end_index),
            args.progress_interval,
        )
    if args.summary:
        write_summary(Path(args.summary), evaluated, input_path, args.start_index, args.end_index)
    print(f"evaluated {len(evaluated)} queries to {args.output}")


if __name__ == "__main__":
    main()
