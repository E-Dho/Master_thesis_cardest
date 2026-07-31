#!/bin/bash
#SBATCH --job-name=qgen_pol
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qgen_pol_%j.err

set -euo pipefail

source /zfshome/sunip956/master_thesis_trajectories/dataset_generation/mobilitydb_loader/common.sh
activate_mobilitydb_env

QUERY_GENERATION_DIR=/zfshome/sunip956/master_thesis_trajectories/query_generation
RUN_ROOT=/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/query_runs
QUERIES_PER_CATEGORY=${QUERIES_PER_CATEGORY:-10}
SEED=${SEED:-20260715}
SAMPLE_CACHE_SIZE=${SAMPLE_CACHE_SIZE:-2048}
RUN_LABEL=${RUN_LABEL:-qgen_pol_${QUERIES_PER_CATEGORY}qpc_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$RUN_ROOT/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
METADATA="$RUN_DIR/run_metadata.txt"

mkdir -p "$RUN_DIR"

start_postgres
trap 'stop_postgres' EXIT

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)

cd /zfshome/sunip956/master_thesis_trajectories
python "$QUERY_GENERATION_DIR/query_generator.py" \
  --config "$QUERY_GENERATION_DIR/pol_query_config.json" \
  --output "$OUTPUT_JSONL" \
  --queries-per-category "$QUERIES_PER_CATEGORY" \
  --seed "$SEED" \
  --sample-cache-size "$SAMPLE_CACHE_SIZE" \
  --execute \
  --host "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" \
  --dbname "$MOBILITYDB_DB"

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

python - "$OUTPUT_JSONL" > "$RUN_DIR/summary.json" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
category_counts = Counter(row["category"]["key"] for row in rows)
multi = [row for row in rows if row["category"]["relation"] == "multi"]
bad_multi = [
    row["query_id"] for row in multi
    if row["entity_cardinality"] is None
    or row["join_cardinality"] is None
    or row["entity_cardinality"] > row["join_cardinality"]
]
summary = {
    "rows": len(rows),
    "category_counts": dict(sorted(category_counts.items())),
    "multi_rows": len(multi),
    "bad_multi_entity_gt_join": bad_multi,
    "min_join_cardinality": min(row["join_cardinality"] for row in rows),
    "max_join_cardinality": max(row["join_cardinality"] for row in rows),
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
  echo "seed=$SEED"
  echo "sample_cache_size=$SAMPLE_CACHE_SIZE"
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
