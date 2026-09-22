#!/bin/bash
#SBATCH --job-name=distjoin_basecards
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PG_BIN=$ROOT/envs/mobilitydb/bin
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
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
for attempt in $(seq 1 180); do
  "$PG_BIN/pg_isready" -h "$SOCKET" -p "$PORT" -d imdb_joblight >/dev/null 2>&1 && break
  [[ "$attempt" -lt 180 ]] || { echo "PostgreSQL readiness timeout" >&2; exit 1; }
  sleep 5
done
"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/distjoin_basecards.py \
  --workload "$ROOT/external/distjoin/queries/job-light.csv" \
  --workload "$ROOT/external/distjoin/queries/job-light-ranges.csv" \
  --postgres-dsn "dbname=imdb_joblight host=$SOCKET port=$PORT" \
  --output "$ROOT/joblight_evaluation/distjoin_base_cardinalities.json"
