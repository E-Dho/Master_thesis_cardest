#!/bin/bash
#SBATCH --job-name=qv3_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_merge_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_merge_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
# Pure Python; does not need the database.
python -m query_generation.benchmark_v3.merge \
  --config "$V3_CONFIG" --run-dir "$CATEGORY_DIR" "${GRID_ARGS[@]}" \
  --output "$RUN_DIR/benchmark.jsonl" --summary "$RUN_DIR/benchmark_summary.json"
