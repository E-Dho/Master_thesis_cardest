#!/bin/bash
#SBATCH --job-name=mobilitydb_server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/mobilitydb_server_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/mobilitydb_server_%j.err

set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/dataset_generation/mobilitydb_loader/common.sh
activate_mobilitydb_env
mkdir -p "$MOBILITYDB_RUN_DIR" "$MOBILITYDB_LOG_DIR"

KEEPALIVE="$MOBILITYDB_RUN_DIR/keepalive"
touch "$KEEPALIVE"
start_postgres

cat > "$MOBILITYDB_RUN_DIR/connection.env" <<CONN
PGHOST=$MOBILITYDB_SOCKET_DIR
PGPORT=$MOBILITYDB_PORT
PGDATABASE=$MOBILITYDB_DB
CONN

trap 'stop_postgres' EXIT
echo "MobilityDB server started. Remove $KEEPALIVE or cancel this job to stop."
while [ -f "$KEEPALIVE" ]; do
  sleep 10
done
