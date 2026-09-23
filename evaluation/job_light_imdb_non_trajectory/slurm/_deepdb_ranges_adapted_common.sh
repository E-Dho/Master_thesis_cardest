# Shared settings for the adapted DeepDB JOB-light-ranges launchers (sourced).
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
DEEPDB_PYTHON=$ROOT/joblight_evaluation/envs/deepdb/bin/python
CLI_PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages
PG_BIN=$ROOT/envs/mobilitydb/bin
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
DATABASE=imdb_joblight
DSN="dbname=$DATABASE host=$SOCKET port=$PORT"
DEEPDB_SOURCE=$ROOT/external/deepdb
DEEPDB_REVISION=655ada13a043a4226a239b0c0a65bcdefef87a02
DATASET=$ROOT/datasets/job
# Separate from the native $ROOT/joblight_evaluation/deepdb_shared root.
ADAPTED_SHARED=$ROOT/joblight_evaluation/deepdb_ranges_adapted_shared
RANGES=$ROOT/external/distjoin/queries/job-light-ranges.csv
NORMAL=$ROOT/external/distjoin/queries/job-light.csv
CONFIG_ADAPTED=$REPO/evaluation/job_light_imdb_non_trajectory/configs/deepdb_job_light_ranges_adapted.yaml
BRIDGE=evaluation/job_light_imdb_non_trajectory/scripts/deepdb_ranges_adapted_bridge.py

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-unknown} array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

STARTED_POSTGRES=0
start_postgres() {
  if "$PG_BIN/pg_ctl" -D "$PGDATA" -o "-k $SOCKET -p $PORT -c listen_addresses=''" -w start 2>/dev/null; then
    STARTED_POSTGRES=1
  else
    "$PG_BIN/pg_ctl" -D "$PGDATA" status
  fi
  for attempt in $(seq 1 180); do
    "$PG_BIN/pg_isready" -h "$SOCKET" -p "$PORT" -d "$DATABASE" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "PostgreSQL readiness timeout" >&2
  return 1
}
stop_postgres() {
  if [[ "$STARTED_POSTGRES" -eq 1 ]]; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop || true
  fi
}
trap stop_postgres EXIT
