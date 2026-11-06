#!/bin/bash
#SBATCH --job-name=traj_mbr_overcount
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_%A_%a.err
# One shard of the segment-MBR overcount statistic (traj_eval/mbr_overcount.py;
# no database needed). WORKLOAD is test or validation; OVERCOUNT_SHARDS must
# equal the array size. A shard stops cleanly before the time limit (exit 75);
# resubmitting the same array resumes.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount run \
  --snapshot "$SNAPSHOT_DIR" --config "$V3_CONFIG" --benchmark "$(traj_benchmark_for "$WORKLOAD")" \
  --results-dir "$SNAPSHOT_DIR/analysis/mbr_overcount/$WORKLOAD" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "${OVERCOUNT_SHARDS:?}" --deadline-seconds "${DEADLINE_SECONDS:-13800}"
