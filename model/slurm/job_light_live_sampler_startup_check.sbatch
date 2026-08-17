#!/bin/bash
#SBATCH --job-name=joblight_live_startup
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python

CONFIG=${CONFIG:-model/configs/job_light_duet_batch_native_range_7m.yaml}
RUN_DIR=${RUN_DIR:-model/runs/job_light_duet_batch_native_range_7m}
CHECK_DIR=${CHECK_DIR:-$RUN_DIR/startup_checks/${SLURM_JOB_ID:-manual}}
DRAW_ROWS=${DRAW_ROWS:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "check_dir=$CHECK_DIR"
echo "draw_rows=$DRAW_ROWS"

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$CHECK_DIR" \
  --draw-rows "$DRAW_ROWS"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
