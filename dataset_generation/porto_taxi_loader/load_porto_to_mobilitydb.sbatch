#!/bin/bash
#SBATCH --job-name=load_porto_50m
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_porto_50m_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_porto_50m_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
fi
source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
activate_mobilitydb_env

TARGET_SEGMENTS=${TARGET_SEGMENTS:-50000000}
SELECTION_SEED=${SELECTION_SEED:-porto_taxi_50m_v1}
# Parsing train.csv is the slowest step and is fully determined by the input
# file, target and seed.  Resubmit with the same RUN_ID to reuse the staging
# directory instead of re-parsing; the default starts a fresh one.
RUN_ID=${RUN_ID:-porto_taxi_50m_load_$(date -u +%Y%m%dT%H%M%SZ)}
STAGING_DIR="$MOBILITYDB_STAGING_ROOT/$RUN_ID"
mkdir -p "$STAGING_DIR" "$MOBILITYDB_LOG_DIR" "$MOBILITYDB_RUN_DIR"
if [ ! -f "$PORTO_TRAIN_CSV" ]; then
  echo "Porto input CSV not found: $PORTO_TRAIN_CSV" >&2
  exit 1
fi

if [ ! -f "$MOBILITYDB_DATA_DIR/PG_VERSION" ]; then
  bash "$REPO_ROOT/dataset_generation/porto_taxi_loader/init_porto_mobilitydb.sh"
fi
start_postgres
trap 'stop_postgres' EXIT

python "$REPO_ROOT/dataset_generation/porto_taxi_loader/parse_porto_to_staging.py" \
  --input "$PORTO_TRAIN_CSV" \
  --staging-dir "$STAGING_DIR" \
  --target-segments "$TARGET_SEGMENTS" \
  --selection-seed "$SELECTION_SEED" \
  --skip-if-complete \
  2>&1 | tee "$MOBILITYDB_LOG_DIR/${RUN_ID}_staging.log"

psql_mobility -v ON_ERROR_STOP=1 -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/schema.sql" 2>&1 | tee "$MOBILITYDB_LOG_DIR/${RUN_ID}_load.log"
psql_mobility -v ON_ERROR_STOP=1 \
  -v taxis_file="'$STAGING_DIR/taxis.tsv'" \
  -v trips_file="'$STAGING_DIR/trips.tsv'" \
  -v segments_file="'$STAGING_DIR/segments_wgs84.tsv'" \
  -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/load.sql" 2>&1 | tee -a "$MOBILITYDB_LOG_DIR/${RUN_ID}_load.log"
psql_mobility -v ON_ERROR_STOP=1 -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/validate.sql" 2>&1 | tee "$MOBILITYDB_LOG_DIR/${RUN_ID}_validation.log"

python "$REPO_ROOT/query_generation/build_porto_taxi_query_config.py" \
  --host "$MOBILITYDB_SOCKET_DIR" --port "$MOBILITYDB_PORT" --dbname "$MOBILITYDB_DB" \
  --output "$PORTO_MOBILITYDB_ROOT/query_configs/porto_taxi_query_config.json"

{
  echo "run_id=$RUN_ID"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "staging_dir=$STAGING_DIR"
  echo "query_config=$PORTO_MOBILITYDB_ROOT/query_configs/porto_taxi_query_config.json"
  echo "database=$MOBILITYDB_DB"
  echo "port=$MOBILITYDB_PORT"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$STAGING_DIR/slurm_load_metadata.txt"
