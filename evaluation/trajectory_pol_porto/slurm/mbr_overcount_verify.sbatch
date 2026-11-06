#!/bin/bash
#SBATCH --job-name=traj_mbr_overcount_verify
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_verify_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_verify_%j.err
# Check every snapshot array against its manifest checksum once and record the
# result (snapshot_verification.json) for the overcount shards and merge, which
# refuse to start without it. Exit 1 on any checksum problem.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
OVERCOUNT_ROOT=${OVERCOUNT_ROOT:-$SNAPSHOT_DIR/analysis/mbr_overcount/v$(python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount version)}
python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount verify \
  --snapshot "$SNAPSHOT_DIR" --results-dir "$OVERCOUNT_ROOT/$WORKLOAD"
