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
STARTED_POSTGRES=0
if "$PG_BIN/pg_ctl" -D "$PGDATA" -o "-k $SOCKET -p $PORT -c listen_addresses=''" -w start 2>/dev/null; then
  STARTED_POSTGRES=1
else
  "$PG_BIN/pg_ctl" -D "$PGDATA" status
fi
cleanup() {
  if [[ "$STARTED_POSTGRES" -eq 1 ]]; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop || true
  fi
}
trap cleanup EXIT
for attempt in $(seq 1 180); do
  "$PG_BIN/pg_isready" -h "$SOCKET" -p "$PORT" -d imdb_joblight >/dev/null 2>&1 && break
  [[ "$attempt" -lt 180 ]] || { echo "PostgreSQL readiness timeout" >&2; exit 1; }
  sleep 5
done
"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli build \
  --config "$CONFIG" --seed "$SEED" --run-directory "$RUN_DIRECTORY"
cat "$RUN_DIRECTORY/build_stage_metrics.json"
