#!/bin/bash
#SBATCH --job-name=own_model_sentinel_check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
OLD_SOURCE=$ROOT/external/master_thesis_cardest_0d0030d6e81883fc3db3c139ae29255697f6bba9
NEW_SOURCE=$ROOT/external/master_thesis_cardest_7c156b247031d7f5dfbd96e812f111abe14a75f2
CHECKPOINT=$ROOT/Master_thesis_cardest/model/runs/job_light_ranges_duet_binary_native_anpm_20k_no_rare_support_live_dropout_0_0_seed_0/checkpoint_step_20000.pt
QUERIES=$ROOT/external/distjoin/queries/job-light-ranges.csv
REFERENCE=$ROOT/Master_thesis_cardest/model/runs/job_light_ranges_duet_binary_native_anpm_20k_no_rare_support_live_dropout_0_0_seed_0/job_light_ranges_results.csv
OUT=$REPO/evaluation/job_light_imdb_non_trajectory/results/joblight_phase1/own_model/range_sentinel_validation_20k

mkdir -p "$OUT"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "old_source=$(git -C "$OLD_SOURCE" rev-parse HEAD)"
echo "new_source=$(git -C "$NEW_SOURCE" rev-parse HEAD)"

PYTHONPATH="$OLD_SOURCE:$REPO" "$PYTHON" \
  evaluation/job_light_imdb_non_trajectory/scripts/own_model_eval_runner.py \
  --checkpoint "$CHECKPOINT" --queries "$QUERIES" \
  --predictions "$OUT/predictions_0d0030d.csv" --latencies "$OUT/latency_0d0030d.csv" \
  --warmup-passes 0 --repetitions 1 --device cuda
PYTHONPATH="$NEW_SOURCE:$REPO" "$PYTHON" \
  evaluation/job_light_imdb_non_trajectory/scripts/own_model_eval_runner.py \
  --checkpoint "$CHECKPOINT" --queries "$QUERIES" \
  --predictions "$OUT/predictions_7c156b2.csv" --latencies "$OUT/latency_7c156b2.csv" \
  --warmup-passes 0 --repetitions 1 --device cuda

REFERENCE="$REFERENCE" OLD="$OUT/predictions_0d0030d.csv" NEW="$OUT/predictions_7c156b2.csv" OUT="$OUT/summary.json" \
  "$PYTHON" - <<'PY'
import csv
import json
import math
import os


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(row["query_id"]): row for row in csv.DictReader(handle)}


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


reference = rows(os.environ["REFERENCE"])
old = rows(os.environ["OLD"])
new = rows(os.environ["NEW"])
if set(old) != set(new) or set(new) != set(reference):
    raise RuntimeError("prediction and reference query IDs differ")
changed = [
    query_id for query_id in sorted(new)
    if not math.isclose(
        float(old[query_id]["estimated_cardinality"]),
        float(new[query_id]["estimated_cardinality"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    )
]
errors = []
for query_id, row in new.items():
    estimate = max(1.0, float(row["estimated_cardinality"]))
    truth = max(1.0, float(reference[query_id]["true_cardinality"]))
    errors.append(max(estimate / truth, truth / estimate))
summary = {
    "old_revision": "0d0030d6e81883fc3db3c139ae29255697f6bba9",
    "new_revision": "7c156b247031d7f5dfbd96e812f111abe14a75f2",
    "query_count": len(new),
    "changed_estimate_count": len(changed),
    "changed_query_ids": changed,
    "smoothed_q_error": {
        "p50": percentile(errors, 0.50),
        "p90": percentile(errors, 0.90),
        "p95": percentile(errors, 0.95),
        "p99": percentile(errors, 0.99),
        "max": max(errors),
    },
}
with open(os.environ["OUT"], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, sort_keys=True))
PY
