#!/bin/bash
#SBATCH --job-name=traj_tune
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_tune_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_tune_%j.err
# Parameter sweep of one in-process trajectory method on the VALIDATION workload
# (traj_eval/tune.py); never touches the test benchmark. No database.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
TUNE_CONFIG=${TUNE_CONFIG:?set TUNE_CONFIG}
VALIDATION_JSONL=${VALIDATION_JSONL:?set VALIDATION_JSONL}
# GRIDS: key=v1/v2/...+key=v1/v2 (no commas or spaces: sbatch --export splits on commas)
GRIDS=${GRIDS:?set GRIDS, e.g. unit_size_m=25/50+time_buckets=50/100}
OUTPUT=${OUTPUT:?set OUTPUT}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
echo "config=$TUNE_CONFIG grids=$GRIDS commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown) host=$(hostname)"
grid_args=()
IFS='+' read -r -a grids <<< "$GRIDS"
for grid in "${grids[@]}"; do grid_args+=(--grid "${grid//\//,}"); done
expected=()
if [ -n "${EXPECTED_QUERIES:-}" ]; then expected=(--expected-queries "$EXPECTED_QUERIES"); fi
"$PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.tune --config "$TUNE_CONFIG" \
  --validation-jsonl "$VALIDATION_JSONL" "${expected[@]}" "${grid_args[@]}" --output "$OUTPUT"
