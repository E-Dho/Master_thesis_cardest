#!/bin/bash
#SBATCH --job-name=distjoin_basecards
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PG_BIN=$ROOT/envs/mobilitydb/bin
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Shared PGDATA: start/stop only under the cluster-visible lock.
PG_LOCK_PYTHON=$PYTHON
source "$REPO/evaluation/job_light_imdb_non_trajectory/slurm/_pg_lock.sh"
trap stop_postgres_locked EXIT
start_postgres_locked
"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/distjoin_basecards.py \
  --workload "$ROOT/external/distjoin/queries/job-light.csv" \
  --workload "$ROOT/external/distjoin/queries/job-light-ranges.csv" \
  --postgres-dsn "dbname=imdb_joblight host=$SOCKET port=$PORT" \
  --output "$ROOT/joblight_evaluation/distjoin_base_cardinalities.json"
