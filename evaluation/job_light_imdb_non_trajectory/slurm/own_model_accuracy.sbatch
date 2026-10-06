#!/bin/bash
#SBATCH --job-name=own_model_accuracy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages
CONFIG=${CONFIG:?set CONFIG to an own-model evaluation config path}
SEED=${SEED:?set SEED to 0, 1, or 2}
RUN_ID=${RUN_ID:-production_step_11000}

cd "$REPO"
export PYTHONPATH="$PACKAGES:$REPO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "seed=$SEED"
echo "run_id=$RUN_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
  run --config "$CONFIG" --seed "$SEED" --run-id "$RUN_ID"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
