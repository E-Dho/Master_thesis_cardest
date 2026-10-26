"""Deterministic fake estimator speaking the subprocess contract on v3 workloads.

usage: fake_traj_estimator.py QUERIES_JSONL SELECT_JSON PREDICTIONS LATENCY SALT
Every 11th query is unsupported, every 13th an approximation, the rest are the
truth times a seeded factor between 1/10 and 10.
"""
import csv
import hashlib
import json
import sys

from evaluation.trajectory_pol_porto.traj_eval.workloads import select_rows

queries, select_json, predictions, latency, salt = sys.argv[1:6]
rows = select_rows(queries, json.loads(select_json))
with open(predictions, "w", newline="") as p, open(latency, "w", newline="") as t:
    pw, tw = csv.writer(p), csv.writer(t)
    pw.writerow(["query_id", "estimated_cardinality", "status", "diagnostic"])
    tw.writerow(["query_id", "repetition", "latency_ms", "scope", "device", "device_name"])
    for query_id, row in enumerate(rows):
        h = int(hashlib.sha256(f"{salt}:{row['predicate_hash']}".encode()).hexdigest(), 16)
        if query_id % 11 == 5:
            pw.writerow([query_id, "", "unsupported", "attribute add-on"])
        else:
            factor = 10 ** (((h % 2001) / 1000.0) - 1.0)
            diagnostic = "approx:mbr surrogate" if query_id % 13 == 7 else ""
            pw.writerow([query_id, row["cardinality"] * factor, "ok", diagnostic])
        tw.writerow([query_id, 0, 1.0 + (h % 100) / 100.0, "end_to_end", "cpu", "FakeCPU"])
