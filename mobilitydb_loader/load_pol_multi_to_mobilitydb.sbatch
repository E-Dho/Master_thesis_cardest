#!/bin/bash
#SBATCH --job-name=load_pol_50m_mobilitydb
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_pol_50m_mobilitydb_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/load_pol_50m_mobilitydb_%j.err

set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh
activate_mobilitydb_env

DATASET_LIST="${POL_DATASET_LIST:-/work_beegfs/sunip956/master_thesis_trajectories/runs/pol_atl_5000a_365d_5min_trajbench_50m/datasets.txt}"
EXPECTED_AGENTS="${POL_EXPECTED_AGENTS:-35000}"
MIN_TOTAL_TUPLES="${POL_MIN_TOTAL_TUPLES:-50000000}"
RUN_ID="pol_mobilitydb_50m_load_$(date -u +%Y%m%dT%H%M%SZ)"
STAGING_DIR="$MOBILITYDB_STAGING_ROOT/$RUN_ID"
LOAD_LOG="$MOBILITYDB_LOG_DIR/${RUN_ID}.log"
VALIDATION_LOG="$MOBILITYDB_LOG_DIR/${RUN_ID}_validation.log"
mkdir -p "$STAGING_DIR" "$MOBILITYDB_LOG_DIR" "$MOBILITYDB_RUN_DIR"

if [ ! -f "$DATASET_LIST" ]; then
  echo "Dataset list not found: $DATASET_LIST" >&2
  exit 1
fi

if [ ! -f "$MOBILITYDB_DATA_DIR/PG_VERSION" ]; then
  "$MOBILITYDB_LOADER_DIR/init_mobilitydb.sh"
fi

start_postgres
trap 'stop_postgres' EXIT

python "$MOBILITYDB_LOADER_DIR/parse_pol_to_staging.py" \
  --dataset-list "$DATASET_LIST" \
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

psql_mobility -v ON_ERROR_STOP=1 \
  -v expected_agents="$EXPECTED_AGENTS" \
  -v min_total_tuples="$MIN_TOTAL_TUPLES" <<'SQL' 2>&1 | tee -a "$VALIDATION_LOG"
WITH counts AS (
  SELECT
    (SELECT COUNT(*) FROM pol.agents) AS agents,
    (SELECT COUNT(*) FROM pol.agents)
      + (SELECT COUNT(*) FROM pol.trips)
      + (SELECT COUNT(*) FROM pol.segments) AS total_tuples
)
SELECT
  'benchmark_size_guard' AS check_name,
  agents = :expected_agents::bigint AND total_tuples >= :min_total_tuples::bigint AS ok,
  agents,
  total_tuples,
  CASE
    WHEN agents = :expected_agents::bigint AND total_tuples >= :min_total_tuples::bigint THEN 1
    ELSE 1 / 0
  END AS fail_if_bad
FROM counts;
SQL

{
  echo "run_id=$RUN_ID"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "dataset_list=$DATASET_LIST"
  echo "staging_dir=$STAGING_DIR"
  echo "load_log=$LOAD_LOG"
  echo "validation_log=$VALIDATION_LOG"
  echo "database=$MOBILITYDB_DB"
  echo "socket_dir=$MOBILITYDB_SOCKET_DIR"
  echo "port=$MOBILITYDB_PORT"
  echo "expected_agents=$EXPECTED_AGENTS"
  echo "min_total_tuples=$MIN_TOTAL_TUPLES"
} > "$STAGING_DIR/slurm_load_metadata.txt"
