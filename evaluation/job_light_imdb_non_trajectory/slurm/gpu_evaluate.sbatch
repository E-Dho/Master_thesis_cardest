#!/bin/bash
#SBATCH --job-name=joblight_gpu_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages

CONFIG=${CONFIG:?set CONFIG to an evaluation config path}
RUN_DIRECTORY=${RUN_DIRECTORY:?set RUN_DIRECTORY to a built run}
SEED=${SEED:-0}

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
echo "run_directory=$RUN_DIRECTORY"
echo "seed=$SEED"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
  evaluate --config "$CONFIG" --seed "$SEED" --run-directory "$RUN_DIRECTORY"
"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
  summarize --config "$CONFIG" --seed "$SEED" --run-directory "$RUN_DIRECTORY"

cat "$RUN_DIRECTORY/summary.json"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
