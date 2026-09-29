#!/bin/bash
#SBATCH --job-name=qeval_porto
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qeval_porto_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qeval_porto_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
activate_mobilitydb_env
INPUT_JSONL=${INPUT_JSONL:?INPUT_JSONL is required}
START_INDEX=${START_INDEX:?START_INDEX is required}
END_INDEX=${END_INDEX:?END_INDEX is required}
RUN_LABEL=${RUN_LABEL:?RUN_LABEL is required}
RUN_DIR="$PORTO_MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
mkdir -p "$RUN_DIR"
start_postgres
trap 'stop_postgres' EXIT
start_seconds=$(date +%s)
cd "$REPO_ROOT"
python query_generation/evaluate_queries.py --input "$INPUT_JSONL" --output "$RUN_DIR/queries.jsonl" --summary "$RUN_DIR/summary.json" \
  --start-index "$START_INDEX" --end-index "$END_INDEX" --progress-interval 1 \
  --host "$MOBILITYDB_SOCKET_DIR" --port "$MOBILITYDB_PORT" --dbname "$MOBILITYDB_DB"
elapsed=$(( $(date +%s) - start_seconds ))
rows=$(wc -l < "$RUN_DIR/queries.jsonl" | tr -d ' ')
printf 'run_label=%s\nstart_index=%s\nend_index=%s\nelapsed_seconds=%s\nevaluated_queries=%s\navg_seconds_per_query=%s\n' "$RUN_LABEL" "$START_INDEX" "$END_INDEX" "$elapsed" "$rows" "$(python -c "print($elapsed / $rows if $rows else float('nan'))")" > "$RUN_DIR/run_metadata.txt"
