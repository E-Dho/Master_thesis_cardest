#!/usr/bin/env bash
set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh

activate_mobilitydb_env
mkdir -p "$MOBILITYDB_DATA_DIR" "$MOBILITYDB_RUN_DIR" "$MOBILITYDB_LOG_DIR"

if [ ! -f "$MOBILITYDB_DATA_DIR/PG_VERSION" ]; then
  initdb -D "$MOBILITYDB_DATA_DIR" --encoding=UTF8 --locale=C
  cat >> "$MOBILITYDB_DATA_DIR/postgresql.conf" <<CONF
listen_addresses = '127.0.0.1'
port = $MOBILITYDB_PORT
unix_socket_directories = '$MOBILITYDB_SOCKET_DIR'
shared_preload_libraries = 'postgis-3'
max_locks_per_transaction = 128
shared_buffers = 2GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_wal_size = 8GB
checkpoint_timeout = 30min
CONF
  cat >> "$MOBILITYDB_DATA_DIR/pg_hba.conf" <<CONF
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
CONF
fi

start_postgres

if ! psql -h "$MOBILITYDB_SOCKET_DIR" -p "$MOBILITYDB_PORT" -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname = '$MOBILITYDB_DB'" | grep -qx 1; then
  createdb -h "$MOBILITYDB_SOCKET_DIR" -p "$MOBILITYDB_PORT" "$MOBILITYDB_DB"
fi

psql_mobility -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql_mobility -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS mobilitydb;"

{
  echo "initialized_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "data_dir=$MOBILITYDB_DATA_DIR"
  echo "database=$MOBILITYDB_DB"
  echo "port=$MOBILITYDB_PORT"
  echo "socket_dir=$MOBILITYDB_SOCKET_DIR"
  psql_mobility -Atc "SELECT postgis_full_version();"
  psql_mobility -Atc "SELECT mobilitydb_version();"
} > "$MOBILITYDB_ROOT/init_metadata.txt"

stop_postgres
