#!/bin/bash
#SBATCH --job-name=joblight_neurocard_build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages

CONFIG=${CONFIG:?set CONFIG to a NeuroCard evaluation config path}
RUN_DIRECTORY=${RUN_DIRECTORY:?set RUN_DIRECTORY to a prepared, smoke-passed run}
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
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
  build --config "$CONFIG" --seed "$SEED" --run-directory "$RUN_DIRECTORY"

cat "$RUN_DIRECTORY/build_stage_metrics.json"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
