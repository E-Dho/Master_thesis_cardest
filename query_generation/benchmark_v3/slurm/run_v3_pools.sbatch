#!/bin/bash
#SBATCH --job-name=qv3_pools
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_pools_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_pools_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
start_postgres
trap 'stop_postgres' EXIT
if [ -f "$POOLS_JSON" ]; then
  echo "pools already exist: $POOLS_JSON"
  exit 0
fi
python -m query_generation.benchmark_v3.build_pools \
  --config "$V3_CONFIG" --output "$POOLS_JSON" --base-seed "$SEED" "${DB_ARGS[@]}"
