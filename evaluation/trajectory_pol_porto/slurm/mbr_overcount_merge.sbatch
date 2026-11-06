#!/bin/bash
#SBATCH --job-name=traj_mbr_overcount_merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_merge_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mbr_overcount_merge_%j.err
# Combine the shards into analysis/mbr_overcount/<workload>_report.{json,md}
# and <workload>_report_per_query.csv. Exit 1 on a label mismatch or a gap.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
WORKLOAD=${WORKLOAD:?set WORKLOAD=test or validation}
python -m evaluation.trajectory_pol_porto.traj_eval.mbr_overcount merge \
  --snapshot "$SNAPSHOT_DIR" --benchmark "$(traj_benchmark_for "$WORKLOAD")" \
  --results-dir "$SNAPSHOT_DIR/analysis/mbr_overcount/$WORKLOAD" \
  --report-prefix "$SNAPSHOT_DIR/analysis/mbr_overcount/${WORKLOAD}_report"
