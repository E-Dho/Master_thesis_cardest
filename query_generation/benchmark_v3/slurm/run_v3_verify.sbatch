#!/bin/bash
#SBATCH --job-name=qv3_verify
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
PER_CATEGORY=${PER_CATEGORY:-25}
start_postgres
trap 'stop_postgres' EXIT
python -m query_generation.benchmark_v3.verify \
  --config "$V3_CONFIG" --benchmark "$RUN_DIR/benchmark.jsonl" "${GRID_ARGS[@]}" \
  --report "$RUN_DIR/verify_report.json" --per-category "$PER_CATEGORY" "${DB_ARGS[@]}"
