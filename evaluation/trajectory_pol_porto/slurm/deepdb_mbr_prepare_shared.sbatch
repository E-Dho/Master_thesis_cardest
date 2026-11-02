#!/bin/bash
#SBATCH --job-name=traj_deepdb_mbr_shared
#SBATCH --partition=base
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# P5 shared preparation (no database): export the one-row-per-trajectory table
# from the P2 snapshot (mobilitydb environment, numpy), then DeepDB's HDF
# preparation in the DeepDB environment. Both outputs are shared by every seed.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
DEEPDB_PYTHON=${DEEPDB_PYTHON:-$THESIS_WORK_ROOT/joblight_evaluation/envs/deepdb/bin/python}
DEEPDB_SOURCE=${DEEPDB_SOURCE:-$THESIS_WORK_ROOT/external/deepdb}
DEEPDB_REVISION=${DEEPDB_REVISION:-655ada13a043a4226a239b0c0a65bcdefef87a02}
SHARED=${DEEPDB_SHARED:-$SNAPSHOT_ROOT/${TRAJ_DATASET}_deepdb_mbr}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown) shared=$SHARED snapshot=$SNAPSHOT_DIR"
activate_mobilitydb_env
python -m evaluation.trajectory_pol_porto.traj_eval.deepdb_mbr \
  --snapshot "$SNAPSHOT_DIR" --config "$V3_CONFIG" --output "$SHARED"
"$DEEPDB_PYTHON" evaluation/trajectory_pol_porto/scripts/deepdb_mbr_bridge.py prepare-shared \
  --source-root "$DEEPDB_SOURCE" --revision "$DEEPDB_REVISION" --shared-root "$SHARED"
