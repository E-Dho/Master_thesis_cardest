#!/bin/bash
#SBATCH --job-name=traj_segment_truth
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_segment_truth_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_segment_truth_%A_%a.err
# One shard of the true matching-segment counts (traj_eval/segment_truth.py, P7
# error split; no database). SEGTRUTH_STEP=verify|run|merge; run needs
# SEGTRUTH_SHARDS = the array size. A shard stops cleanly before the time limit
# (exit 75); resubmitting the same array resumes. Submit with submit_p7.sh STAGE=truth.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
SEGTRUTH_ROOT=${SEGTRUTH_ROOT:?}
module=evaluation.trajectory_pol_porto.traj_eval.segment_truth
case "${SEGTRUTH_STEP:?}" in
  verify) python -m "$module" verify --snapshot "$SNAPSHOT_DIR" --results-dir "$SEGTRUTH_ROOT/$WORKLOAD" ;;
  run) python -m "$module" run --snapshot "$SNAPSHOT_DIR" --config "$V3_CONFIG" \
         --benchmark "$(traj_benchmark_for "$WORKLOAD")" --results-dir "$SEGTRUTH_ROOT/$WORKLOAD" \
         --shard "$SLURM_ARRAY_TASK_ID" --shards "${SEGTRUTH_SHARDS:?}" --deadline-seconds "${DEADLINE_SECONDS:-13800}" ;;
  merge) python -m "$module" merge --snapshot "$SNAPSHOT_DIR" --benchmark "$(traj_benchmark_for "$WORKLOAD")" \
           --results-dir "$SEGTRUTH_ROOT/$WORKLOAD" --output-prefix "$SEGTRUTH_ROOT/${WORKLOAD}_segment_truth" ;;
  *) echo "SEGTRUTH_STEP must be verify, run or merge" >&2; exit 2 ;;
esac
