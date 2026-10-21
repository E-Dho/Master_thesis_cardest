#!/bin/bash
#SBATCH --job-name=qgen_pol_shard
#SBATCH --array=0-63%1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_shard_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_shard_%A_%a.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
activate_mobilitydb_env

QUERIES_PER_CATEGORY=${QUERIES_PER_CATEGORY:-500}
CHUNKS_PER_CATEGORY=${CHUNKS_PER_CATEGORY:-4}
SEED=${SEED:-20261020}
RUN_LABEL=${RUN_LABEL:-qgen_pol_50m_${QUERIES_PER_CATEGORY}qpc_segment_coupled_v2}
RUN_DIR="$MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
SHARD_DIR="$RUN_DIR/shards"

if [ "$CHUNKS_PER_CATEGORY" -ne 4 ]; then
  echo "This 0-63 array is defined for CHUNKS_PER_CATEGORY=4" >&2
  exit 2
fi

categories=(
  standard.range.single
  standard.range.multi
  standard.unbounded.single
  standard.unbounded.multi
  temporal.range.single
  temporal.range.multi
  temporal.unbounded.single
  temporal.unbounded.multi
  spatial.range.single
  spatial.range.multi
  spatial.unbounded.single
  spatial.unbounded.multi
  spatio_temporal.range.single
  spatio_temporal.range.multi
  spatio_temporal.unbounded.single
  spatio_temporal.unbounded.multi
)

task_id=${SLURM_ARRAY_TASK_ID:?run this script as a Slurm array}
category_index=$((task_id / CHUNKS_PER_CATEGORY))
chunk_index=$((task_id % CHUNKS_PER_CATEGORY))
CATEGORY=${categories[$category_index]}
chunk_size=$(((QUERIES_PER_CATEGORY + CHUNKS_PER_CATEGORY - 1) / CHUNKS_PER_CATEGORY))
start_index=$((chunk_index * chunk_size))
remaining=$((QUERIES_PER_CATEGORY - start_index))
query_count=$chunk_size
if [ "$remaining" -lt "$query_count" ]; then
  query_count=$remaining
fi
if [ "$query_count" -le 0 ]; then
  echo "No work for task $task_id" >&2
  exit 0
fi

shard_prefix="$SHARD_DIR/$CATEGORY.chunk-$(printf '%02d' "$chunk_index")"
OUTPUT_JSONL="$shard_prefix.jsonl"
SUMMARY_JSON="$shard_prefix.summary.json"
STATE_JSON="$shard_prefix.state.json"

mkdir -p "$SHARD_DIR"
start_postgres
trap 'stop_postgres' EXIT

user_args=()
if [ -n "${MOBILITYDB_USER:-}" ]; then
  user_args+=(--user "$MOBILITYDB_USER")
fi

cd "$REPO_ROOT"
python query_generation/generate_pol_segment_coupled_benchmark.py \
  --config query_generation/pol_query_config.json \
  --output "$OUTPUT_JSONL" \
  --summary "$SUMMARY_JSON" \
  --queries-per-category "$QUERIES_PER_CATEGORY" \
  --categories "$CATEGORY" \
  --start-index "$start_index" \
  --query-count "$query_count" \
  --state "$STATE_JSON" \
  --progress-interval 10 \
  --seed "$SEED" \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB" \
  "${user_args[@]}"

cat "$SUMMARY_JSON"
