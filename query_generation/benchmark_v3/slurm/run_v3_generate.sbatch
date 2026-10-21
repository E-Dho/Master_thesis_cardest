#!/bin/bash
#SBATCH --job-name=qv3_generate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_generate_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_generate_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
# One array task per category (0-15), one task at a time: every task owns the
# PostgreSQL instance while it runs. Exit 75 = stopped at the deadline; resubmit
# the same array (same RUN_LABEL and SEED) and it resumes where it stopped.
task_id=${SLURM_ARRAY_TASK_ID:?run as a Slurm array 0-15}
CATEGORY=${V3_CATEGORIES[$task_id]}
DEADLINE_SECONDS=${DEADLINE_SECONDS:-84600}
mkdir -p "$CATEGORY_DIR"
start_postgres
trap 'stop_postgres' EXIT
set +e
python -m query_generation.benchmark_v3.generate \
  --config "$V3_CONFIG" --pools "$POOLS_JSON" --category "$CATEGORY" \
  --output-dir "$CATEGORY_DIR" --base-seed "$SEED" "${GRID_ARGS[@]}" \
  --deadline-seconds "$DEADLINE_SECONDS" --progress-interval 10 "${DB_ARGS[@]}"
status=$?
set -e
exit $status
