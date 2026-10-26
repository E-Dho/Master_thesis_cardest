#!/bin/bash
#SBATCH --job-name=qv3_verify
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_verify_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_verify_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
# Every verified query is appended to verify_report.json.results.jsonl. The job
# stops cleanly before the time limit (exit 75); resubmit with STAGE=verify and
# it continues where it stopped.
PER_CATEGORY=${PER_CATEGORY:-25}   # VALIDATION=1 sets 10 in v3_env.sh
DEADLINE_SECONDS=${DEADLINE_SECONDS:-84600}
if ! python -c "import numpy" 2>/dev/null; then
  echo "The verification job needs numpy in $MOBILITYDB_ENV (pools, generation and merge do not)." >&2
  echo "Install it once, e.g.: micromamba install -p $MOBILITYDB_ENV -c conda-forge numpy" >&2
  exit 3
fi
start_postgres
trap 'stop_postgres' EXIT
set +e
python -m query_generation.benchmark_v3.verify \
  --config "$V3_CONFIG" --benchmark "$RUN_DIR/benchmark.jsonl" "${GRID_ARGS[@]}" \
  --report "$RUN_DIR/verify_report.json" --per-category "$PER_CATEGORY" \
  --deadline-seconds "$DEADLINE_SECONDS" "${DB_ARGS[@]}"
status=$?
set -e
exit $status
