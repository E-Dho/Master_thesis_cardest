#!/bin/bash
# Select the loaded database of a v3 dataset (sourced by every v3 and
# trajectory-evaluation job before anything else). Needs REPO_ROOT.
#   V3_DATASET=pol    (default) POL, dataset_generation/mobilitydb_loader/common.sh
#   V3_DATASET=porto  Porto taxi, dataset_generation/porto_taxi_loader/common.sh
# The database variables are cleared first: --export=ALL passes the submitting
# shell's environment, and a stale MOBILITYDB_* value of the other dataset
# would otherwise point this job at the wrong data directory or port.
V3_DATASET=${V3_DATASET:-pol}
unset MOBILITYDB_DATA_DIR MOBILITYDB_LOG_DIR MOBILITYDB_RUN_DIR MOBILITYDB_STAGING_ROOT \
  MOBILITYDB_PORT MOBILITYDB_DB MOBILITYDB_SOCKET_DIR MOBILITYDB_ROOT
case "$V3_DATASET" in
  pol)
    source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
    V3_EXPECTED_DATA_DIR="$THESIS_WORK_ROOT/mobilitydb/pgdata"
    ;;
  porto)
    source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
    export MOBILITYDB_ROOT="$PORTO_MOBILITYDB_ROOT"  # query_runs/ lives next to the Porto pgdata
    V3_EXPECTED_DATA_DIR="$PORTO_MOBILITYDB_ROOT/pgdata"
    ;;
  *)
    echo "V3_DATASET must be pol or porto, got $V3_DATASET" >&2
    exit 2
    ;;
esac
if [ "$MOBILITYDB_DATA_DIR" != "$V3_EXPECTED_DATA_DIR" ]; then
  echo "V3_DATASET=$V3_DATASET resolved PGDATA $MOBILITYDB_DATA_DIR, expected $V3_EXPECTED_DATA_DIR" >&2
  exit 2
fi
