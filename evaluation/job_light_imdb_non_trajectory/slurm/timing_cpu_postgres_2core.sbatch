#!/bin/bash
#SBATCH --job-name=joblight_timing_cpu_postgres_2core
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# PostgreSQL latency (profile cpu_postgres_2core): the launcher starts the
# server inside this step pinned to the second physical core (autovacuum off)
# and pins the client to the first.  The server must not already be running.

set -euo pipefail
source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_timing_common.sh
CLI_PYTHON=${CLI_PYTHON_POSTGRES:-$CLI_PYTHON}   # needs psycopg>=3, numpy, threadpoolctl
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_postgres_2core --plan "$PLAN" \
  --pg-bin "$PG_BIN" --pgdata "$PGDATA" --socket-dir "$SOCKET" --port "$PORT"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
