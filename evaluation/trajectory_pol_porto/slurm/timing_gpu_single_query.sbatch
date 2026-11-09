#!/bin/bash
#SBATCH --job-name=traj_timing_gpu_single_query
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# GPU latency of the trajectory neural estimators (profile gpu_single_query):
# one L40S, four pinned support physical cores, PyTorch/BLAS CPU threads limited
# to one, synchronized CUDA timing; its own table, never mixed with CPU latency.
# The timing guard enforces gpu_name "NVIDIA L40S" and the declared cpu_model
# (configs/timing_hardware.json). Submit through submit_timing.sh.
set -euo pipefail
PROFILE=gpu_single_query
source "${REPO_ROOT:?}/evaluation/trajectory_pol_porto/slurm/_timing_common.sh"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv,noheader || true
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread --gres=gpu:1 \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile gpu_single_query --plan "$PLAN" --cli-module "$CLI_MODULE"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
