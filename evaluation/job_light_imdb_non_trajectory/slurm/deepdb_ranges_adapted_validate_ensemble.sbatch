#!/bin/bash
#SBATCH --job-name=deepdb_ranges_adapted_bounded
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Bounded real-ensemble validation of the adapted DeepDB variant: real adapted
# HDF, reduced SPN sample sizes, all 1,000 JOB-light-ranges queries, one
# timed repetition.  Output is a gate, not a reported result.

set -euo pipefail
source "$(dirname "$0")/_deepdb_ranges_adapted_common.sh" 2>/dev/null \
  || source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_deepdb_ranges_adapted_common.sh
SAMPLE_SIZES=${SAMPLE_SIZES:-"100000 100000 10000 10000 10000"}
OUTPUT=${OUTPUT:-$ROOT/joblight_evaluation/deepdb_ranges_adapted_validation/bounded_${SLURM_JOB_ID:-manual}}
start_postgres
# shellcheck disable=SC2086
"$DEEPDB_PYTHON" "$BRIDGE" validate-ensemble --source-root "$DEEPDB_SOURCE" \
  --revision "$DEEPDB_REVISION" --dataset-root "$ADAPTED_SHARED" --output-directory "$OUTPUT" \
  --database "$DATABASE" --pg-host "$SOCKET" --pg-port "$PORT" --seed 0 \
  --sample-sizes $SAMPLE_SIZES --queries "$RANGES" --repetitions 1
cat "$OUTPUT/bounded_validation.json"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
