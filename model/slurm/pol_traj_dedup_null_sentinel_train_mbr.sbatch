#!/bin/bash
#SBATCH --job-name=pol_mbr_null_train
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:-/work_beegfs/sunip956/master_thesis_trajectories}
REPO=${REPO:-$ROOT/Master_thesis_cardest_git}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
CONFIG=${CONFIG:-model/configs/pol_50m_traj_dedup_single_anchor_mbr_null_sentinel.yaml}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "git_sha=$(git rev-parse HEAD)"
echo "training_only=true"

# Training prints one null_sentinel=... line at startup, before the first step:
#   catalog_pairs / void_cells   how much provable void signal was found
#   scanned_rows                 rows behind the co-occurrence catalog
#   width_changed_columns        columns whose head width grew by one bit
#                                because their real domain size was an exact
#                                power of two; parameter counts will not match
#                                a baseline checkpoint for those columns
# void_cells=0 or catalog_available=false means the run is effectively a
# baseline, so check that line before spending 24 hours on it.
"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
