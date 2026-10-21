#!/bin/bash
#SBATCH --job-name=qmerge_pol_shards
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_pol_shards_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_pol_shards_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/query_generation/merge_pol_segment_coupled_shards.py" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
activate_mobilitydb_env

QUERIES_PER_CATEGORY=${QUERIES_PER_CATEGORY:-500}
RUN_LABEL=${RUN_LABEL:-qgen_pol_50m_${QUERIES_PER_CATEGORY}qpc_segment_coupled_v2}
RUN_DIR="$MOBILITYDB_ROOT/query_runs/$RUN_LABEL"

cd "$REPO_ROOT"
python3 query_generation/merge_pol_segment_coupled_shards.py \
  --config query_generation/pol_query_config.json \
  --shard-dir "$RUN_DIR/shards" \
  --output "$RUN_DIR/queries.jsonl" \
  --summary "$RUN_DIR/benchmark_summary.json" \
  --queries-per-category "$QUERIES_PER_CATEGORY"

cat "$RUN_DIR/benchmark_summary.json"
