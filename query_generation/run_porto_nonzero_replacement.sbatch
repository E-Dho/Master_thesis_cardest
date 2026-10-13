#!/bin/bash
#SBATCH --job-name=qreplace_porto_nonzero
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qreplace_porto_nonzero_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qreplace_porto_nonzero_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
activate_mobilitydb_env

INPUT_JSONL=${INPUT_JSONL:-$PORTO_MOBILITYDB_ROOT/query_runs/porto_taxi_50m_8000_segment_coupled_v1/queries.jsonl}
RUN_LABEL=${RUN_LABEL:-porto_taxi_50m_8000_segment_coupled_nonzero_v1}
QUERY_CONFIG=${QUERY_CONFIG:-$PORTO_MOBILITYDB_ROOT/query_configs/porto_taxi_query_config.json}
REPLACEMENT_SEED=${REPLACEMENT_SEED:-20261013}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-1000}
SAMPLE_CACHE_SIZE=${SAMPLE_CACHE_SIZE:-2048}
RUN_DIR="$PORTO_MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
SUMMARY_JSON="$RUN_DIR/benchmark_summary.json"
PROGRESS_JSONL="$RUN_DIR/queries.jsonl.partial.jsonl"
CENTERS_JSON="$RUN_DIR/queries.jsonl.centers.json"
DB_USER=${MOBILITYDB_USER:-}

mkdir -p "$RUN_DIR"
test -f "$INPUT_JSONL"
test -f "$QUERY_CONFIG"

start_postgres
trap 'stop_postgres' EXIT

user_args=()
if [ -n "$DB_USER" ]; then
  user_args+=(--user "$DB_USER")
fi

cd "$REPO_ROOT"
python query_generation/resample_nonzero_benchmark.py \
  --input "$INPUT_JSONL" \
  --config "$QUERY_CONFIG" \
  --output "$OUTPUT_JSONL" \
  --summary "$SUMMARY_JSON" \
  --progress "$PROGRESS_JSONL" \
  --centers-cache "$CENTERS_JSON" \
  --seed "$REPLACEMENT_SEED" \
  --max-attempts "$MAX_ATTEMPTS" \
  --sample-cache-size "$SAMPLE_CACHE_SIZE" \
  --queries-per-category 500 \
  --expected-distinct-rows 4000 \
  "${user_args[@]}" \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB"

cat "$SUMMARY_JSON"
