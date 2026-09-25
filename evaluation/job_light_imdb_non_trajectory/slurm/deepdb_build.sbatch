#!/bin/bash
#SBATCH --job-name=joblight_deepdb_build
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/deepdb/bin/python
PG_BIN=$ROOT/envs/mobilitydb/bin
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
CONFIG=${CONFIG:?set CONFIG to the DeepDB evaluation config}
RUN_DIRECTORY=${RUN_DIRECTORY:?set RUN_DIRECTORY to a prepared, smoke-passed run}
SEED=${SEED:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Shared PGDATA: start/stop only under the cluster-visible lock.
PG_LOCK_PYTHON=$PYTHON
source "$REPO/evaluation/job_light_imdb_non_trajectory/slurm/_pg_lock.sh"
trap stop_postgres_locked EXIT
start_postgres_locked
"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli build \
  --config "$CONFIG" --seed "$SEED" --run-directory "$RUN_DIRECTORY"
cat "$RUN_DIRECTORY/build_stage_metrics.json"
