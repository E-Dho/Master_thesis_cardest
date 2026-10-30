#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mobilitydb_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_mobilitydb_%j.err
# P4: the MobilityDB planner configs in CONFIGS (colon separated), one after
# the other against one PostgreSQL start (prepare, smoke, build, evaluate,
# summarize; seed 0). Submit through submit_p4.sh: it runs under the shared
# database job name with --dependency=singleton. Every config is attempted;
# the job fails if any of them failed.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
CONFIGS=${CONFIGS:?set CONFIGS to colon-separated mobilitydb_*.yaml paths}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
activate_mobilitydb_env
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown) host=$(hostname) python=$(command -v python)"
start_postgres
trap 'stop_postgres' EXIT
failed=()
IFS=: read -r -a config_list <<< "$CONFIGS"
for config in "${config_list[@]}"; do
  echo "== $config"
  if ! python -m evaluation.trajectory_pol_porto.traj_eval.cli run --config "$config" --seed 0; then
    echo "FAILED: $config" >&2
    failed+=("$config")
  fi
done
stop_postgres
trap - EXIT
if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed configs: ${failed[*]}" >&2
  exit 1
fi
