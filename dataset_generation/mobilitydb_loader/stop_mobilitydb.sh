#!/usr/bin/env bash
set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/dataset_generation/mobilitydb_loader/common.sh
activate_mobilitydb_env
rm -f "$MOBILITYDB_RUN_DIR/keepalive"
stop_postgres
