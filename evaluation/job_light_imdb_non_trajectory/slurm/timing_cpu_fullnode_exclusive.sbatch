#!/bin/bash
#SBATCH --job-name=joblight_timing_cpu_fullnode
#SBATCH --partition=base
#SBATCH --nodes=1
# Declared CPU class (configs/timing_hardware.json); override with
# sbatch --nodelist=<another verified AMD EPYC 7313 node>.
#SBATCH --nodelist=n237
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Supplementary scalability timing (profile cpu_fullnode_exclusive): exclusive
# node, every logical CPU visible, thread pools sized to the physical core
# count.  Reported separately; never mixed with the one-core table.

set -euo pipefail
source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_timing_common.sh
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_ON_NODE
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" --cpu-bind=verbose,none --exclusive \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_fullnode_exclusive --plan "$PLAN"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
