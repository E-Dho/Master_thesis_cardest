#!/bin/bash
#SBATCH --job-name=validate_mobilitydb
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/validate_mobilitydb_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/validate_mobilitydb_%j.err

set -euo pipefail

source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh
activate_mobilitydb_env
start_postgres
trap 'stop_postgres' EXIT

VALIDATION_OUT="$MOBILITYDB_ROOT/validation_$(date -u +%Y%m%dT%H%M%SZ).txt"
psql_mobility -v ON_ERROR_STOP=1 -f "$MOBILITYDB_LOADER_DIR/validate.sql" | tee "$VALIDATION_OUT"
echo "$VALIDATION_OUT" > "$MOBILITYDB_ROOT/latest_validation_path.txt"
