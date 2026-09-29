#!/usr/bin/env bash
set -euo pipefail

export THESIS_HOME_ROOT=${THESIS_HOME_ROOT:-/zfshome/sunip956/master_thesis_trajectories}
export THESIS_WORK_ROOT=${THESIS_WORK_ROOT:-/work_beegfs/sunip956/master_thesis_trajectories}
export PORTO_LOADER_DIR=${PORTO_LOADER_DIR:-$THESIS_HOME_ROOT/dataset_generation/porto_taxi_loader}
export PORTO_MOBILITYDB_ROOT=${PORTO_MOBILITYDB_ROOT:-$THESIS_WORK_ROOT/mobilitydb_porto}
export MOBILITYDB_ENV=${MOBILITYDB_ENV:-$THESIS_WORK_ROOT/envs/mobilitydb}
export MOBILITYDB_DATA_DIR=${MOBILITYDB_DATA_DIR:-$PORTO_MOBILITYDB_ROOT/pgdata}
export MOBILITYDB_LOG_DIR=${MOBILITYDB_LOG_DIR:-$PORTO_MOBILITYDB_ROOT/logs}
export MOBILITYDB_RUN_DIR=${MOBILITYDB_RUN_DIR:-$PORTO_MOBILITYDB_ROOT/run}
export MOBILITYDB_STAGING_ROOT=${MOBILITYDB_STAGING_ROOT:-$PORTO_MOBILITYDB_ROOT/staging}
export MOBILITYDB_PORT=${MOBILITYDB_PORT:-55433}
export MOBILITYDB_DB=${MOBILITYDB_DB:-porto_taxi_mobilitydb}
export MOBILITYDB_SOCKET_DIR=${MOBILITYDB_SOCKET_DIR:-$MOBILITYDB_RUN_DIR}
export PORTO_TRAIN_CSV=${PORTO_TRAIN_CSV:-$THESIS_WORK_ROOT/datasets/porto_taxi/raw/train.csv}
export PORTO_SRID=${PORTO_SRID:-3763}

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
  if pg_is_running; then return 0; fi
  pg_ctl -D "$MOBILITYDB_DATA_DIR" \
    -l "$MOBILITYDB_LOG_DIR/postgres_$(date -u +%Y%m%dT%H%M%SZ).log" \
    -o "-k $MOBILITYDB_SOCKET_DIR -p $MOBILITYDB_PORT" -w start
}

stop_postgres() {
  if pg_is_running; then pg_ctl -D "$MOBILITYDB_DATA_DIR" -m fast -w stop; fi
}
