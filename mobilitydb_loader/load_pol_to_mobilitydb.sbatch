#!/bin/bash
#SBATCH --job-name=load_pol_mobilitydb
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_pol_mobilitydb_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_pol_mobilitydb_%j.err

set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh
activate_mobilitydb_env

RUN_ID="pol_mobilitydb_load_$(date -u +%Y%m%dT%H%M%SZ)"
STAGING_DIR="$MOBILITYDB_STAGING_ROOT/$RUN_ID"
LOAD_LOG="$MOBILITYDB_LOG_DIR/${RUN_ID}.log"
VALIDATION_LOG="$MOBILITYDB_LOG_DIR/${RUN_ID}_validation.log"
mkdir -p "$STAGING_DIR" "$MOBILITYDB_LOG_DIR" "$MOBILITYDB_RUN_DIR"

if [ ! -f "$MOBILITYDB_DATA_DIR/PG_VERSION" ]; then
  "$MOBILITYDB_LOADER_DIR/init_mobilitydb.sh"
fi

start_postgres
trap 'stop_postgres' EXIT

python "$MOBILITYDB_LOADER_DIR/parse_pol_to_staging.py" \
  --dataset "$POL_DATASET_DIR" \
  --staging-dir "$STAGING_DIR" \
  --srid "$POL_SRID" \
  2>&1 | tee "$LOAD_LOG"

psql_mobility -v ON_ERROR_STOP=1 -f "$MOBILITYDB_LOADER_DIR/schema.sql" 2>&1 | tee -a "$LOAD_LOG"
psql_mobility -v ON_ERROR_STOP=1 \
  -v agents_file="'$STAGING_DIR/agents.tsv'" \
  -v trips_file="'$STAGING_DIR/trips.tsv'" \
  -v segments_file="'$STAGING_DIR/segments.tsv'" \
  -f "$MOBILITYDB_LOADER_DIR/load.sql" 2>&1 | tee -a "$LOAD_LOG"

psql_mobility -v ON_ERROR_STOP=1 -f "$MOBILITYDB_LOADER_DIR/validate.sql" 2>&1 | tee "$VALIDATION_LOG"

{
  echo "run_id=$RUN_ID"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "dataset=$POL_DATASET_DIR"
  echo "staging_dir=$STAGING_DIR"
  echo "load_log=$LOAD_LOG"
  echo "validation_log=$VALIDATION_LOG"
  echo "database=$MOBILITYDB_DB"
  echo "socket_dir=$MOBILITYDB_SOCKET_DIR"
  echo "port=$MOBILITYDB_PORT"
} > "$STAGING_DIR/slurm_load_metadata.txt"
