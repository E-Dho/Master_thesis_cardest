#!/usr/bin/env bash
set -euo pipefail

export THESIS_HOME_ROOT=${THESIS_HOME_ROOT:-/zfshome/sunip956/master_thesis_trajectories}
export THESIS_WORK_ROOT=${THESIS_WORK_ROOT:-/work_beegfs/sunip956/master_thesis_trajectories}
export MOBILITYDB_LOADER_DIR="$THESIS_HOME_ROOT/dataset_generation/mobilitydb_loader"
export MOBILITYDB_ROOT="$THESIS_WORK_ROOT/mobilitydb"
export MOBILITYDB_ENV="$THESIS_WORK_ROOT/envs/mobilitydb"
export MOBILITYDB_SRC="$MOBILITYDB_ROOT/src/MobilityDB"
export MOBILITYDB_BUILD="$MOBILITYDB_ROOT/build/MobilityDB"
export MOBILITYDB_LOG_DIR="$MOBILITYDB_ROOT/logs"
export MOBILITYDB_RUN_DIR="$MOBILITYDB_ROOT/run"
export MOBILITYDB_DATA_DIR="$MOBILITYDB_ROOT/pgdata"
export MOBILITYDB_STAGING_ROOT="$MOBILITYDB_ROOT/staging"
export MOBILITYDB_PORT=${MOBILITYDB_PORT:-55432}
export MOBILITYDB_DB=${MOBILITYDB_DB:-pol_mobilitydb}
export MOBILITYDB_SOCKET_DIR="$MOBILITYDB_RUN_DIR"
export POL_DATASET_DIR=${POL_DATASET_DIR:-/work_beegfs/sunip956/master_thesis_trajectories/datasets/pol_atl_1000a_365d_5min_seed1_enriched_traj_20260714T111045Z}
export POL_SRID=${POL_SRID:-26916}

load_micromamba() {
  module load micromamba/1.4.2
  export MAMBA_ROOT_PREFIX=/work_beegfs/sunip956/micromamba
}

activate_mobilitydb_env() {
  load_micromamba
  eval "$(micromamba shell hook -s bash)"
  micromamba activate "$MOBILITYDB_ENV"
}

psql_mobility() {
  psql -h "$MOBILITYDB_SOCKET_DIR" -p "$MOBILITYDB_PORT" -d "$MOBILITYDB_DB" "$@"
}

pg_is_running() {
  pg_ctl -D "$MOBILITYDB_DATA_DIR" status >/dev/null 2>&1
}

start_postgres() {
  mkdir -p "$MOBILITYDB_RUN_DIR" "$MOBILITYDB_LOG_DIR"
  if pg_is_running; then
    return 0
  fi
  pg_ctl -D "$MOBILITYDB_DATA_DIR" \
    -l "$MOBILITYDB_LOG_DIR/postgres_$(date -u +%Y%m%dT%H%M%SZ).log" \
    -o "-k $MOBILITYDB_SOCKET_DIR -p $MOBILITYDB_PORT" \
    -w start
}

stop_postgres() {
  if pg_is_running; then
    pg_ctl -D "$MOBILITYDB_DATA_DIR" -m fast -w stop
  fi
}
