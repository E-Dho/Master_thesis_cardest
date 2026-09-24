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
# Primary CPU latency (profile cpu_1core): one physical core with its SMT
# sibling reserved, one thread for BLAS/OpenMP/PyTorch.  Runs every plan
# entry back to back on this node.  Submit e.g.
#   PLAN=... sbatch --constraint=<node-feature> slurm/timing_cpu_1core.sbatch

set -euo pipefail
source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_timing_common.sh
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_1core --plan "$PLAN"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
