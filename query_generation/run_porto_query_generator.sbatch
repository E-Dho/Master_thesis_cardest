#!/bin/bash
#SBATCH --job-name=qfreeze_porto
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qfreeze_porto_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qfreeze_porto_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
activate_mobilitydb_env

QUERY_CONFIG=${QUERY_CONFIG:-$PORTO_MOBILITYDB_ROOT/query_configs/porto_taxi_query_config.json}
RUN_LABEL=${RUN_LABEL:-porto_taxi_50m_8000_segment_coupled_v1_frozen}
RUN_DIR="$PORTO_MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
OUTPUT_JSONL="$RUN_DIR/queries.jsonl"
mkdir -p "$RUN_DIR"
test -f "$QUERY_CONFIG"
start_postgres
trap 'stop_postgres' EXIT
cd "$REPO_ROOT"
python query_generation/query_generator.py --config "$QUERY_CONFIG" --output "$OUTPUT_JSONL" \
  --queries-per-category 500 --seed 20260929 --sample-cache-size 2048 --progress-interval 25 \
  --skip-cardinality-execution --execute --host "$MOBILITYDB_SOCKET_DIR" --port "$MOBILITYDB_PORT" --dbname "$MOBILITYDB_DB"
python - "$OUTPUT_JSONL" > "$RUN_DIR/summary.json" <<'PY'
import json
import sys
from collections import Counter
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
counts = Counter(row['category']['key'] for row in rows)
if len(rows) != 8000:
    raise SystemExit(f'expected 8000 frozen queries, got {len(rows)}')
if len(counts) != 16 or any(value != 500 for value in counts.values()):
    raise SystemExit(f'expected 16 categories of 500 queries, got {dict(sorted(counts.items()))}')
if any(row['join_cardinality'] is not None for row in rows):
    raise SystemExit('frozen workload must leave cardinalities unevaluated')
print(json.dumps({'rows': len(rows), 'category_counts': dict(sorted(counts.items())), 'seed': 20260929}, indent=2, sort_keys=True))
PY
