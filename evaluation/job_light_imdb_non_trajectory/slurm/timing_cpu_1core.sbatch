#!/bin/bash
#SBATCH --job-name=joblight_timing_cpu_1core
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Primary CPU latency (profile cpu_1core): Slurm selects one base node, then
# the launcher binds the profile to its exact CPU model. One physical core,
# one thread for BLAS/OpenMP/PyTorch; every plan entry runs on this node.
#   PLAN=... sbatch slurm/timing_cpu_1core.sbatch

set -euo pipefail
source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_timing_common.sh
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_1core --plan "$PLAN"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
