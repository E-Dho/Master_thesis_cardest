#!/usr/bin/env python3
"""Re-evaluate the recorded JOB-light-ranges selection candidates under one source revision.

The input inventory is the versioned ``own_model_selection.md``.  Every array
task snapshots its selected row alongside canonical predictions and accuracy;
``--summarize`` then ranks the complete 1,000-query results without using a
benchmark query to construct the candidate set.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.job_light_imdb_non_trajectory.joblight_eval.artifacts import write_predictions
from evaluation.job_light_imdb_non_trajectory.joblight_eval.metrics import summarize_predictions
from evaluation.job_light_imdb_non_trajectory.joblight_eval.own_model_selection import require_complete_candidate
from evaluation.job_light_imdb_non_trajectory.joblight_eval.records import PredictionRecord
from evaluation.job_light_imdb_non_trajectory.joblight_eval.workloads import load_workload


def _unquote(value: str) -> str:
    return value.strip().strip("`")


def candidates(selection: Path) -> list[dict[str, Any]]:
    text = selection.read_text(encoding="utf-8")
    try:
        section = text.split("### job_light_ranges", 1)[1].split("## Re-running", 1)[0]
    except IndexError as exc:
        raise ValueError(f"cannot locate JOB-light-ranges inventory in {selection}") from exc
    result = []
    for line in section.splitlines():
        if not line.startswith("| ") or "Checkpoint" in line or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 17:
            continue
        checkpoint = _unquote(cells[3])
        if not checkpoint.endswith(".pt"):
            continue
        result.append({
            "index": len(result),
            "checkpoint": checkpoint,
            "training_config": _unquote(cells[2]),
            "step": int(cells[4]),
            "seed": int(cells[5]),
            "old_smoothed": {
                "p50": float(cells[8]), "p90": float(cells[9]), "p95": float(cells[10]),
                "p99": float(cells[11]), "max": float(cells[12]),
            },
            "old_raw": {"p50": float(cells[13]), "p99": float(cells[14])},
            "old_estimate_lt_1_count": int(cells[15]),
            "old_mark": cells[0],
        })
    if len(result) != 16:
        raise ValueError(f"expected 16 eligible JOB-light-ranges candidates, found {len(result)}")
    return result


def evaluate(candidate: dict[str, Any], queries: Path, output: Path, *, device: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    raw_predictions = output / "runner_predictions.csv"
    raw_latencies = output / "runner_latencies.csv"
    command = [
        sys.executable, str(Path(__file__).with_name("own_model_eval_runner.py")),
        "--checkpoint", candidate["checkpoint"], "--queries", str(queries),
        "--predictions", str(raw_predictions), "--latencies", str(raw_latencies),
        "--warmup-passes", "0", "--repetitions", "1", "--device", device,
    ]
    subprocess.run(command, check=True)
    workload = load_workload(queries, "job_light_ranges")
    truth = {query.query_id: query.true_cardinality for query in workload}
    records = []
    with raw_predictions.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query_id = int(row["query_id"])
            estimate = row.get("estimated_cardinality")
            records.append(PredictionRecord(
                workload="job_light_ranges", query_id=query_id, status=row["status"],
                true_cardinality=truth[query_id],
                estimated_cardinality=None if estimate in (None, "") else float(estimate),
                diagnostic=row.get("diagnostic", ""),
            ))
    accuracy = summarize_predictions(records)
    require_complete_candidate("job_light_ranges", accuracy)
    write_predictions(output / "predictions.csv", records)
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps({"candidate": candidate, "accuracy": accuracy}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.glob("candidate_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate, accuracy = payload["candidate"], payload["accuracy"]
        require_complete_candidate("job_light_ranges", accuracy)
        rows.append({
            "index": candidate["index"], "checkpoint": candidate["checkpoint"], "step": candidate["step"],
            "seed": candidate["seed"], "old": candidate["old_smoothed"],
            "new": accuracy["smoothed_q_error"], "raw": accuracy["raw_q_error"],
            "estimate_lt_1_count": accuracy["estimate_lt_1_count"],
        })
    if len(rows) != 16:
        raise ValueError(f"expected 16 completed candidates, found {len(rows)}")
    old_p50 = min(rows, key=lambda row: row["old"]["p50"])
    old_p99 = min(rows, key=lambda row: row["old"]["p99"])
    new_p50 = min(rows, key=lambda row: row["new"]["p50"])
    new_p99 = min(rows, key=lambda row: row["new"]["p99"])
    rows.sort(key=lambda row: row["new"]["p50"])
    payload = {
        "candidate_count": len(rows), "old_pick_a": old_p50, "old_pick_b": old_p99,
        "new_pick_a": new_p50, "new_pick_b": new_p99, "ranked_by_new_smoothed_p50": rows,
        "ranked_by_new_smoothed_p99": sorted(rows, key=lambda row: row["new"]["p99"]),
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.output_root)
        return 0
    if args.candidate_index is None:
        raise ValueError("--candidate-index is required unless --summarize is used")
    inventory = candidates(args.selection)
    candidate = inventory[args.candidate_index]
    evaluate(candidate, args.queries, args.output_root / f"candidate_{candidate['index']:02d}", device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
