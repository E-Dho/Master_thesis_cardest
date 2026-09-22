#!/bin/bash
#SBATCH --job-name=deepdb_prepare_shared
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/deepdb/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/deepdb_bridge.py prepare-shared \
  --source-root "$ROOT/external/deepdb" \
  --revision 655ada13a043a4226a239b0c0a65bcdefef87a02 \
  --dataset-root "$ROOT/datasets/job" \
  --output-directory "$ROOT/joblight_evaluation/deepdb_shared"
