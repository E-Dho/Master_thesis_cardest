# Locked start/stop of the shared PostgreSQL server (sourced).
#
# PGDATA is on BeeGFS and visible from every node, but pg_ctl status only sees
# local processes: without a cluster-wide lock two nodes could each start a
# postmaster on the same data directory.  Every launcher sources this file and
# uses start_postgres_locked / stop_postgres_locked (see scripts/pg_cluster_lock.py).
#
# Requires PG_BIN PGDATA SOCKET PORT REPO; optional PG_LOCK_PYTHON (default
# python3), JOBLIGHT_PG_LOCK_WAIT_SECONDS (default 0: fail fast when another
# job holds the lock), PG_READY_DATABASE (default imdb_joblight).
PG_LOCK_SCRIPT=${PG_LOCK_SCRIPT:-$REPO/evaluation/job_light_imdb_non_trajectory/scripts/pg_cluster_lock.py}
PG_LOCK_TOKEN=""
PG_LOCK_OWNED=0
STARTED_POSTGRES=0

pg_lock_acquire() {
  local output mode
  output=$("${PG_LOCK_PYTHON:-python3}" "$PG_LOCK_SCRIPT" acquire --pgdata "$PGDATA" \
    --holder-pid "$$" --label "${SLURM_JOB_NAME:-$(basename "$0")}" \
    --wait-seconds "${JOBLIGHT_PG_LOCK_WAIT_SECONDS:-0}") || return $?
  read -r PG_LOCK_TOKEN mode <<<"$output"
  export JOBLIGHT_PG_LOCK_TOKEN=$PG_LOCK_TOKEN
  if [[ "$mode" == owner ]]; then PG_LOCK_OWNED=1; fi
}

pg_lock_release() {
  if [[ "$PG_LOCK_OWNED" -eq 1 ]]; then
    "${PG_LOCK_PYTHON:-python3}" "$PG_LOCK_SCRIPT" release --pgdata "$PGDATA" --token "$PG_LOCK_TOKEN" || true
    PG_LOCK_OWNED=0
  fi
}

start_postgres_locked() {
  pg_lock_acquire || { echo "cannot lock $PGDATA (see message above)" >&2; return 1; }
  if [[ "$PG_LOCK_OWNED" -eq 1 ]]; then
    # The lock guarantees no server runs anywhere on this PGDATA; start ours.
    # Mark it started first: a start that times out may still be in crash
    # recovery on BeeGFS and must be stopped by stop_postgres_locked.
    STARTED_POSTGRES=1
    if ! "$PG_BIN/pg_ctl" -D "$PGDATA" -o "-k $SOCKET -p $PORT -c listen_addresses=''" -w -t 900 start; then
      if ! "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
        echo "PostgreSQL failed to start" >&2
        return 1
      fi
      echo "pg_ctl start timed out; waiting for the server to finish recovery" >&2
    fi
  else
    # Re-entered under a parent's lock: the parent's server must be running here.
    "$PG_BIN/pg_ctl" -D "$PGDATA" status
  fi
  local attempt
  for attempt in $(seq 1 180); do
    "$PG_BIN/pg_isready" -h "$SOCKET" -p "$PORT" -d "${PG_READY_DATABASE:-imdb_joblight}" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "PostgreSQL readiness timeout" >&2
  return 1
}

stop_postgres_locked() {
  if [[ "$STARTED_POSTGRES" -eq 1 ]] && "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    if ! "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w -t 900 stop; then
      # Keep the lock: it turns stale when this job ends (and Slurm kills the
      # server with it), so no other job can start a second postmaster before.
      echo "pg_ctl stop failed; leaving $PGDATA locked until this job ends" >&2
      return 0
    fi
  fi
  STARTED_POSTGRES=0
  pg_lock_release
}
