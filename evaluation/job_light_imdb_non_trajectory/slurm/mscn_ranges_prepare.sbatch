#!/bin/bash
#SBATCH --job-name=mscn_ranges_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PG_BIN=$ROOT/envs/mobilitydb/bin
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
OUTPUT=${OUTPUT:-$ROOT/joblight_evaluation/mscn_job_light_ranges_adapted}
QUERY_COUNT=${QUERY_COUNT:-100000}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Shared PGDATA: start/stop only under the cluster-visible lock.
PG_LOCK_PYTHON=$PYTHON
source "$REPO/evaluation/job_light_imdb_non_trajectory/slurm/_pg_lock.sh"
trap stop_postgres_locked EXIT
start_postgres_locked


"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/mscn_ranges_prepare.py \
  --dataset-root "$ROOT/datasets/job" \
  --evaluation-workload "$ROOT/external/neurocard/neurocard/queries/job-light-ranges.csv" \
  --output-root "$OUTPUT" \
  --query-count "$QUERY_COUNT" \
  --seed 0 \
  --postgres-dsn "dbname=imdb_joblight host=$SOCKET port=$PORT" \
  --progress-every 100
