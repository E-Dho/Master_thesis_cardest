#!/bin/bash
#SBATCH --job-name=traj_gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_gate_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_gate_%A_%a.err
# One shard of the label-reproduction gate (no database needed). WORKLOAD is
# test or validation; GATE_SHARDS must equal the array size. A shard stops
# cleanly before the time limit (exit 75); resubmitting the same array resumes.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
BENCHMARK=$(traj_benchmark_for "$WORKLOAD")
if [ ! -f "$BENCHMARK" ]; then
  echo "benchmark $BENCHMARK does not exist (has its generation and merge finished?)" >&2
  exit 2
fi
python -m evaluation.trajectory_pol_porto.traj_eval.snapshot_gate run \
  --snapshot "$SNAPSHOT_DIR" --config "$V3_CONFIG" --benchmark "$BENCHMARK" \
  --results-dir "$SNAPSHOT_DIR/gate/$WORKLOAD" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "${GATE_SHARDS:?}" --deadline-seconds "${DEADLINE_SECONDS:-13800}"
