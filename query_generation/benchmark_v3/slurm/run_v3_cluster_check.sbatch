#!/bin/bash
#SBATCH --job-name=qv3_check
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_check_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_check_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
python -c "import sys, numpy, psycopg; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'psycopg', psycopg.__version__)"
start_postgres
trap 'stop_postgres' EXIT
REPORT="$MOBILITYDB_ROOT/query_runs/v3_cluster_check_$(date -u +%Y%m%dT%H%M%SZ).txt"
psql -X -h "$MOBILITYDB_SOCKET_DIR" -p "$MOBILITYDB_PORT" -d "$MOBILITYDB_DB" \
  -f query_generation/benchmark_v3/cluster_check.sql 2>&1 | tee "$REPORT"
echo "report=$REPORT"
