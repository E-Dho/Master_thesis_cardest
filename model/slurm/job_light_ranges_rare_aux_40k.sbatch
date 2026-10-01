#!/bin/bash
#SBATCH --job-name=joblight_ranges_40k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=${CONFIG:-model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml}
RUN_DIR=model/runs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary_seed_0
STARTUP=$RUN_DIR/startup_check_${SLURM_JOB_ID:-manual}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "batch_size=2048"
echo "rare_batch_size=512"
echo "steps_per_epoch=40000"
echo "early_stopping_patience_steps=5000"
echo "support_planning_steps=40000"
echo "use_cols=content"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -e "$RUN_DIR/training_summary.json" ]]; then
  echo "refusing to overwrite completed run: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
