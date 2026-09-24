#!/bin/bash
#SBATCH --job-name=deepdb_ranges_adapted_seed
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%A_%a.err
#
# Production run of the adapted variant, one seed per array task.  Tasks run
# one at a time (%1) because they share the PostgreSQL cluster that DeepDB
# queries for exact full-join sizes.  Requires a completed
# deepdb_ranges_adapted_prepare.sbatch and a passed bounded validation.
# The evaluate stage here produces accuracy only; standardized latency is
# measured afterwards with slurm/timing_cpu_1core.sbatch on the finished runs.
# Aggregate afterwards with:
#   python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli aggregate --config "$CONFIG_ADAPTED"

set -euo pipefail
source "$(dirname "$0")/_deepdb_ranges_adapted_common.sh" 2>/dev/null \
  || source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_deepdb_ranges_adapted_common.sh
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
export PYTHONPATH="$PACKAGES:$PYTHONPATH"
for gate in "$ADAPTED_SHARED/validation/pre_hdf_validation.json" "$ADAPTED_SHARED/validation/post_hdf_validation.json"; do
  "$CLI_PYTHON" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["passed"] else 1)' "$gate" \
    || { echo "validation gate not passed: $gate" >&2; exit 1; }
done
start_postgres
CLI=(-m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli)
RUN_DIRECTORY=$("$CLI_PYTHON" "${CLI[@]}" prepare --config "$CONFIG_ADAPTED" --seed "$SEED")
echo "run_directory=$RUN_DIRECTORY"
for stage in smoke build evaluate summarize; do
  "$CLI_PYTHON" "${CLI[@]}" "$stage" --config "$CONFIG_ADAPTED" --seed "$SEED" --run-directory "$RUN_DIRECTORY"
done
cat "$RUN_DIRECTORY/summary.json" | head -80
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
