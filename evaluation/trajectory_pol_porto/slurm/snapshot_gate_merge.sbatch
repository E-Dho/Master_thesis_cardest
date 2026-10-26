#!/bin/bash
#SBATCH --job-name=traj_gate_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_gate_merge_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_gate_merge_%j.err
# Combine the gate shards of one workload into gate/<workload>_report.json.
# Exit 1 on any mismatch, missing or duplicate query.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
python -m evaluation.trajectory_pol_porto.traj_eval.snapshot_gate merge \
  --benchmark "$(traj_benchmark_for "$WORKLOAD")" --results-dir "$SNAPSHOT_DIR/gate/$WORKLOAD" \
  --snapshot "$SNAPSHOT_DIR" --report "$SNAPSHOT_DIR/gate/${WORKLOAD}_report.json"
