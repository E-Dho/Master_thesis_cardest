#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_snapshot_export_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_snapshot_export_%j.err
# Export the benchmark database into the shared numpy snapshot (P2), then
# re-check every checksum. Submit through submit_snapshot.sh: it runs under the
# shared database job name with --dependency=singleton, so it never overlaps
# another job that starts or stops PostgreSQL.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
FORCE_ARGS=()
if [ "${FORCE:-0}" = "1" ]; then FORCE_ARGS=(--force); fi
echo "snapshot: $SNAPSHOT_DIR  config: $V3_CONFIG  commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "$(dirname "$SNAPSHOT_DIR")"
start_postgres
trap 'stop_postgres' EXIT
python -m evaluation.trajectory_pol_porto.traj_eval.snapshot export \
  --config "$V3_CONFIG" --output "$SNAPSHOT_DIR" "${DB_ARGS[@]}" "${FORCE_ARGS[@]}"
stop_postgres
trap - EXIT
python -m evaluation.trajectory_pol_porto.traj_eval.snapshot verify --snapshot "$SNAPSHOT_DIR"
