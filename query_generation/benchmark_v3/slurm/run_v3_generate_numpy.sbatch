#!/bin/bash
#SBATCH --job-name=qv3_generate_numpy
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_generate_numpy_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qv3_generate_numpy_%A_%a.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"
source "$REPO_ROOT/query_generation/benchmark_v3/slurm/v3_env.sh"
v3_resolve_repo_root
activate_mobilitydb_env
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# One array task per category (0-15), all in parallel: labels come from
# NumpyBackend on the shared snapshot, which needs no database (memory: the
# snapshot is memory-mapped, about 40 GB peak as in the label gate). Exit 75 =
# stopped at the deadline; resubmit the same array (same RUN_LABEL and SEED)
# and it resumes where it stopped.
task_id=${SLURM_ARRAY_TASK_ID:?run as a Slurm array 0-15}
CATEGORY=${V3_CATEGORIES[$task_id]}
DEADLINE_SECONDS=${DEADLINE_SECONDS:-41400}
if [ ! -f "$SNAPSHOT_DIR/manifest.json" ]; then
  echo "snapshot $SNAPSHOT_DIR does not exist" >&2
  exit 2
fi
mkdir -p "$CATEGORY_DIR"
set +e
python -m query_generation.benchmark_v3.generate \
  --config "$V3_CONFIG" --pools "$POOLS_JSON" --category "$CATEGORY" \
  --output-dir "$CATEGORY_DIR" --base-seed "$SEED" "${GRID_ARGS[@]}" "${EXCLUDE_ARGS[@]}" \
  --deadline-seconds "$DEADLINE_SECONDS" --progress-interval 10 \
  --backend numpy --snapshot "$SNAPSHOT_DIR"
status=$?
set -e
exit $status
