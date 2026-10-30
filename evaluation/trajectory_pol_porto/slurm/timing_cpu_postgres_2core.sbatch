#!/bin/bash
#SBATCH --job-name=pol_mobilitydb
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_timing_cpu_postgres_2core_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_timing_cpu_postgres_2core_%j.err
# Planner latency of the MobilityDB/PostgreSQL configs (profile
# cpu_postgres_2core): the launcher starts the POL server inside this step,
# pinned to the second physical core with autovacuum off (planner settings and
# statistics unchanged, so the estimates equal the accuracy run's), and pins
# the client to the first. It starts PostgreSQL on the shared POL data
# directory, so it runs under the database job name pol_mobilitydb with
# --dependency=singleton (submit_timing.sh); the server must not be running.
set -euo pipefail
PROFILE=cpu_postgres_2core
source "${REPO_ROOT:?}/evaluation/trajectory_pol_porto/slurm/_timing_common.sh"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --hint=nomultithread \
  --cpu-bind=verbose,cores \
  "$CLI_PYTHON" "$LAUNCHER" --profile cpu_postgres_2core --plan "$PLAN" --cli-module "$CLI_MODULE" \
  --pg-bin "$MOBILITYDB_ENV/bin" --pgdata "$MOBILITYDB_DATA_DIR" --socket-dir "$MOBILITYDB_SOCKET_DIR" \
  --port "$MOBILITYDB_PORT" --database "$MOBILITYDB_DB"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
