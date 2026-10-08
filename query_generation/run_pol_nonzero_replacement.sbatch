#!/bin/bash
#SBATCH --job-name=qreplace_pol_nonzero
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qreplace_pol_nonzero_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qreplace_pol_nonzero_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
activate_mobilitydb_env

INPUT_JSONL=${INPUT_JSONL:-$MOBILITYDB_ROOT/query_runs/qgen_pol_50m_500qpc_20260824_segment_coupled_v1/queries.jsonl}
RUN_LABEL=${RUN_LABEL:-qgen_pol_50m_500qpc_20260824_segment_coupled_nonzero_v1}
REPLACEMENT_SEED=${REPLACEMENT_SEED:-20261008}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-1000}
RUN_DIR="$MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
SUMMARY_JSON="$RUN_DIR/benchmark_summary.json"

mkdir -p "$RUN_DIR"
test -f "$INPUT_JSONL"

start_postgres
trap 'stop_postgres' EXIT

cd "$REPO_ROOT"
python query_generation/resample_nonzero_benchmark.py \
  --input "$INPUT_JSONL" \
  --config query_generation/pol_query_config.json \
  --output "$OUTPUT_JSONL" \
  --summary "$SUMMARY_JSON" \
  --seed "$REPLACEMENT_SEED" \
  --max-attempts "$MAX_ATTEMPTS" \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB"

cat "$SUMMARY_JSON"
