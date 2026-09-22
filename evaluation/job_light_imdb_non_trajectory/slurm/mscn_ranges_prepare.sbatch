#!/bin/bash
#SBATCH --job-name=mscn_ranges_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PG_BIN=$ROOT/envs/mobilitydb/bin
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
OUTPUT=${OUTPUT:-$ROOT/joblight_evaluation/mscn_job_light_ranges_adapted}
QUERY_COUNT=${QUERY_COUNT:-100000}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
STARTED_POSTGRES=0
if "$PG_BIN/pg_ctl" -D "$PGDATA" -o "-k $SOCKET -p $PORT -c listen_addresses=''" -w start 2>/dev/null; then
  STARTED_POSTGRES=1
else
  "$PG_BIN/pg_ctl" -D "$PGDATA" status
fi
cleanup() {
  if [[ "$STARTED_POSTGRES" -eq 1 ]]; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop || true
  fi
}
trap cleanup EXIT

# pg_ctl can return after its fixed startup timeout while crash recovery is still
# finishing on BeeGFS.  Do not hand work to psycopg until it accepts queries.
for attempt in $(seq 1 180); do
  if "$PG_BIN/pg_isready" -h "$SOCKET" -p "$PORT" -d imdb_joblight >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" -eq 180 ]]; then
    echo "PostgreSQL did not become ready within 15 minutes" >&2
    exit 1
  fi
  sleep 5
done

"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/mscn_ranges_prepare.py \
  --dataset-root "$ROOT/datasets/job" \
  --evaluation-workload "$ROOT/external/neurocard/neurocard/queries/job-light-ranges.csv" \
  --output-root "$OUTPUT" \
  --query-count "$QUERY_COUNT" \
  --seed 0 \
  --postgres-dsn "dbname=imdb_joblight host=$SOCKET port=$PORT" \
  --progress-every 100
