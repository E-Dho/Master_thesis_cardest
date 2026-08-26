#!/bin/bash
#SBATCH --job-name=qgen_pol
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_%j.err

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
QUERIES_PER_CATEGORY=${QUERIES_PER_CATEGORY:-10}
SEED=${SEED:-20260715}
SAMPLE_CACHE_SIZE=${SAMPLE_CACHE_SIZE:-2048}
CATEGORIES=${CATEGORIES:-}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-100}
SKIP_CARDINALITY_EXECUTION=${SKIP_CARDINALITY_EXECUTION:-0}
RUN_LABEL=${RUN_LABEL:-qgen_pol_${QUERIES_PER_CATEGORY}qpc_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$RUN_ROOT/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
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
  local generated_queries
  end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  end_seconds=$(date +%s)
  elapsed_seconds=$((end_seconds - START_SECONDS))
  if [ -f "$OUTPUT_JSONL" ]; then
    generated_queries=$(wc -l < "$OUTPUT_JSONL" | tr -d ' ')
  else
    generated_queries=0
  fi
  {
    echo "run_label=$RUN_LABEL"
    echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
    echo "status=$status"
    echo "queries_per_category=$QUERIES_PER_CATEGORY"
    echo "categories=$CATEGORIES"
    echo "seed=$SEED"
    echo "sample_cache_size=$SAMPLE_CACHE_SIZE"
    echo "skip_cardinality_execution=$SKIP_CARDINALITY_EXECUTION"
    echo "start_utc=$START_UTC"
    echo "end_utc=$end_utc"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "generated_queries=$generated_queries"
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

category_args=()
if [ -n "$CATEGORIES" ]; then
  category_args+=(--categories "$CATEGORIES")
fi
cardinality_args=()
if [ "$SKIP_CARDINALITY_EXECUTION" = "1" ]; then
  cardinality_args+=(--skip-cardinality-execution)
fi

cd "$REPO_ROOT"
python "$QUERY_GENERATION_DIR/query_generator.py" \
  --config "$QUERY_GENERATION_DIR/pol_query_config.json" \
  --output "$OUTPUT_JSONL" \
  --queries-per-category "$QUERIES_PER_CATEGORY" \
  "${category_args[@]}" \
  --seed "$SEED" \
  --sample-cache-size "$SAMPLE_CACHE_SIZE" \
  --progress-interval "$PROGRESS_INTERVAL" \
  "${cardinality_args[@]}" \
  --execute \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB"

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

python - "$OUTPUT_JSONL" "$SKIP_CARDINALITY_EXECUTION" > "$RUN_DIR/summary.json" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
allow_missing_cardinalities = sys.argv[2] == "1"
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
category_counts = Counter(row["category"]["key"] for row in rows)
multi = [row for row in rows if row["category"]["relation"] == "multi"]
bad_multi = []
for row in multi:
    entity_cardinality = row["entity_cardinality"]
    join_cardinality = row["join_cardinality"]
    if entity_cardinality is None or join_cardinality is None:
        if not allow_missing_cardinalities:
            bad_multi.append(row["query_id"])
        continue
    if entity_cardinality > join_cardinality:
        bad_multi.append(row["query_id"])
summary = {
    "rows": len(rows),
    "category_counts": dict(sorted(category_counts.items())),
    "multi_rows": len(multi),
    "bad_multi_entity_gt_join": bad_multi,
    "min_join_cardinality": min((row["join_cardinality"] for row in rows if row["join_cardinality"] is not None), default=None),
    "max_join_cardinality": max((row["join_cardinality"] for row in rows if row["join_cardinality"] is not None), default=None),
}
print(json.dumps(summary, indent=2, sort_keys=True))
if bad_multi:
    raise SystemExit("entity_cardinality exceeded join_cardinality")
PY

GENERATED_QUERIES=$(python - "$OUTPUT_JSONL" <<'PY'
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(sum(1 for _ in handle))
PY
)
AVG_SECONDS_PER_QUERY=$(python - "$ELAPSED_SECONDS" "$GENERATED_QUERIES" <<'PY'
import sys

elapsed = float(sys.argv[1])
queries = int(sys.argv[2])
print(f"{elapsed / queries:.6f}" if queries else "nan")
PY
)

{
  echo "run_label=$RUN_LABEL"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "queries_per_category=$QUERIES_PER_CATEGORY"
  echo "categories=$CATEGORIES"
  echo "seed=$SEED"
  echo "sample_cache_size=$SAMPLE_CACHE_SIZE"
  echo "skip_cardinality_execution=$SKIP_CARDINALITY_EXECUTION"
  echo "start_utc=$START_UTC"
  echo "end_utc=$END_UTC"
  echo "elapsed_seconds=$ELAPSED_SECONDS"
  echo "generated_queries=$GENERATED_QUERIES"
  echo "avg_seconds_per_query=$AVG_SECONDS_PER_QUERY"
  echo "output_jsonl=$OUTPUT_JSONL"
  echo "summary_json=$RUN_DIR/summary.json"
} > "$METADATA"

cat "$METADATA"
cat "$RUN_DIR/summary.json"
