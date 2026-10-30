#!/bin/bash
#SBATCH --job-name=traj_timing_cpu_1core
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# Primary CPU latency of the in-process trajectory estimators (profile
# cpu_1core): one physical core, one compute thread; every plan entry on this
# node. Submit through submit_timing.sh.
set -euo pipefail
PROFILE=cpu_1core
source "${REPO_ROOT:?}/evaluation/trajectory_pol_porto/slurm/_timing_common.sh"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_1core --plan "$PLAN" --cli-module "$CLI_MODULE"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
