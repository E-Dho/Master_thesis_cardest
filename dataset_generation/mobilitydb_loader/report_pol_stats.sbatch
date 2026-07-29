#!/bin/bash
#SBATCH --job-name=pol_db_stats
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/pol_db_stats_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/pol_db_stats_%j.err

set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/dataset_generation/mobilitydb_loader/common.sh
activate_mobilitydb_env

REPORT="${POL_STATS_REPORT:-/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/logs/pol_db_stats_${SLURM_JOB_ID}.txt}"
DATASET="${POL_STATS_DATASET:-/work_beegfs/sunip956/master_thesis_trajectories/datasets/pol_atl_5000a_365d_5min_seed1_trajbench_50m}"
STAGING_DIR="${POL_STATS_STAGING_DIR:-/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/staging/pol_mobilitydb_load_20260729T073314Z}"
LOAD_JOB="${POL_STATS_LOAD_JOB:-19590003}"

start_postgres
trap stop_postgres EXIT

{
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "database=$MOBILITYDB_DB"
  echo "load_job=$LOAD_JOB"
  echo "dataset=$DATASET"
  echo "staging_dir=$STAGING_DIR"

  echo
  echo "== row_counts =="
  psql_mobility -X -A -F $'\t' -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'agents' AS table_name, COUNT(*) AS rows FROM pol.agents
UNION ALL SELECT 'trips', COUNT(*) FROM pol.trips
UNION ALL SELECT 'segments', COUNT(*) FROM pol.segments
UNION ALL SELECT 'total_logical_tuples',
  (SELECT COUNT(*) FROM pol.agents)
  + (SELECT COUNT(*) FROM pol.trips)
  + (SELECT COUNT(*) FROM pol.segments);
SQL

  echo
  echo "== trip_segment_stats =="
  psql_mobility -X -A -F $'\t' -v ON_ERROR_STOP=1 <<'SQL'
SELECT
  COUNT(*) AS trips,
  MIN(num_of_segments) AS min_segments_per_trip,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY num_of_segments) AS median_segments_per_trip,
  AVG(num_of_segments)::numeric(12,3) AS avg_segments_per_trip,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY num_of_segments) AS p95_segments_per_trip,
  MAX(num_of_segments) AS max_segments_per_trip
FROM pol.trips;
SQL

  echo
  echo "== temporal_spatial_extent =="
  psql_mobility -X -A -F $'\t' -v ON_ERROR_STOP=1 <<'SQL'
SELECT
  (SELECT MIN(start_time) FROM pol.trips) AS min_trip_start,
  (SELECT MAX(end_time) FROM pol.trips) AS max_trip_end,
  (SELECT MIN(t_s) FROM pol.segments) AS min_segment_start,
  (SELECT MAX(t_e) FROM pol.segments) AS max_segment_end,
  (SELECT ST_AsText(ST_Extent(segment_geom)) FROM pol.segments) AS segment_extent;
SQL

  echo
  echo "== table_sizes =="
  psql_mobility -X -A -F $'\t' -v ON_ERROR_STOP=1 <<'SQL'
SELECT
  n.nspname || '.' || c.relname AS relation,
  pg_size_pretty(pg_relation_size(c.oid)) AS table_size,
  pg_size_pretty(pg_indexes_size(c.oid)) AS index_size,
  pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
  pg_relation_size(c.oid) AS table_bytes,
  pg_indexes_size(c.oid) AS index_bytes,
  pg_total_relation_size(c.oid) AS total_bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'pol' AND c.relkind IN ('r','p')
ORDER BY pg_total_relation_size(c.oid) DESC;
SQL

  echo
  echo "== database_sizes =="
  psql_mobility -X -A -F $'\t' -v ON_ERROR_STOP=1 <<'SQL'
SELECT
  current_database() AS database,
  pg_size_pretty(pg_database_size(current_database())) AS database_size,
  pg_database_size(current_database()) AS database_bytes;

SELECT
  'pol_schema_total' AS scope,
  pg_size_pretty(SUM(pg_total_relation_size(c.oid))) AS size,
  SUM(pg_total_relation_size(c.oid)) AS bytes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'pol' AND c.relkind IN ('r','p');
SQL

  echo
  echo "== filesystem_sizes =="
  du -sh "$MOBILITYDB_DATA_DIR"
  du -sh "$STAGING_DIR"
  du -sh "$DATASET"
} > "$REPORT"

echo "$REPORT"
