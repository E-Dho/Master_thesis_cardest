#!/bin/bash
#SBATCH --job-name=qeval_pol
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qeval_pol_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qeval_pol_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
activate_mobilitydb_env

QUERY_GENERATION_DIR="$REPO_ROOT/query_generation"
RUN_ROOT=/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/query_runs
INPUT_JSONL=${INPUT_JSONL:?INPUT_JSONL is required}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:-}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-5}
RUN_LABEL=${RUN_LABEL:-qeval_pol_${START_INDEX}_${END_INDEX:-end}_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$RUN_ROOT/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
SUMMARY_JSON="$RUN_DIR/summary.json"
METADATA="$RUN_DIR/run_metadata.txt"

mkdir -p "$RUN_DIR"

start_postgres

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)

write_partial_metadata() {
  local status="$1"
  local end_utc
  local end_seconds
  local elapsed_seconds
  local evaluated_queries
  end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  end_seconds=$(date +%s)
  elapsed_seconds=$((end_seconds - START_SECONDS))
  if [ -f "$OUTPUT_JSONL" ]; then
    evaluated_queries=$(wc -l < "$OUTPUT_JSONL" | tr -d ' ')
  else
    evaluated_queries=0
  fi
  {
    echo "run_label=$RUN_LABEL"
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "status=$status"
    echo "input_jsonl=$INPUT_JSONL"
    echo "start_index=$START_INDEX"
    echo "end_index=$END_INDEX"
    echo "start_utc=$START_UTC"
    echo "end_utc=$end_utc"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "evaluated_queries=$evaluated_queries"
    echo "output_jsonl=$OUTPUT_JSONL"
  } > "$METADATA.partial"
}

cleanup() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    write_partial_metadata "failed_or_interrupted"
  fi
  stop_postgres
  exit "$status"
}
trap cleanup EXIT

end_args=()
if [ -n "$END_INDEX" ]; then
  end_args+=(--end-index "$END_INDEX")
fi

cd "$REPO_ROOT"
python "$QUERY_GENERATION_DIR/evaluate_queries.py" \
  --input "$INPUT_JSONL" \
  --output "$OUTPUT_JSONL" \
  --summary "$SUMMARY_JSON" \
  --start-index "$START_INDEX" \
  "${end_args[@]}" \
  --progress-interval "$PROGRESS_INTERVAL" \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB"

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))
EVALUATED_QUERIES=$(wc -l < "$OUTPUT_JSONL" | tr -d ' ')
AVG_SECONDS_PER_QUERY=$(python - "$ELAPSED_SECONDS" "$EVALUATED_QUERIES" <<'PY'
import sys

elapsed = float(sys.argv[1])
queries = int(sys.argv[2])
print(f"{elapsed / queries:.6f}" if queries else "nan")
PY
)

{
  echo "run_label=$RUN_LABEL"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "input_jsonl=$INPUT_JSONL"
  echo "start_index=$START_INDEX"
  echo "end_index=$END_INDEX"
  echo "start_utc=$START_UTC"
  echo "end_utc=$END_UTC"
  echo "elapsed_seconds=$ELAPSED_SECONDS"
  echo "evaluated_queries=$EVALUATED_QUERIES"
  echo "avg_seconds_per_query=$AVG_SECONDS_PER_QUERY"
  echo "output_jsonl=$OUTPUT_JSONL"
  echo "summary_json=$SUMMARY_JSON"
} > "$METADATA"

cat "$METADATA"
cat "$SUMMARY_JSON"
