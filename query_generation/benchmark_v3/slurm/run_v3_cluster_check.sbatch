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
REPORT="$MOBILITYDB_ROOT/query_runs/v3_cluster_check_$(date -u +%Y%m%dT%H%M%SZ).txt"
# Report the Python packages of the job environment (informational, never fatal):
# psycopg is needed by every stage, numpy only by the verification job.
python - <<'PY' | tee "$REPORT"
import importlib, sys
print("python", sys.version.split()[0])
for name, needed_by in (("psycopg", "all stages"), ("numpy", "verify only")):
    try:
        module = importlib.import_module(name)
        print(f"{name} {module.__version__} ({needed_by})")
    except ImportError:
        print(f"{name} MISSING ({needed_by})")
PY
start_postgres
trap 'stop_postgres' EXIT
psql -X -h "$MOBILITYDB_SOCKET_DIR" -p "$MOBILITYDB_PORT" -d "$MOBILITYDB_DB" \
  -f query_generation/benchmark_v3/cluster_check.sql 2>&1 | tee -a "$REPORT"
echo "report=$REPORT"
