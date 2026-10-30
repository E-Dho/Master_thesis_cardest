#!/bin/bash
#SBATCH --job-name=traj_eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_eval_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_eval_%A_%a.err
# One seed (= array task id) of an in-process trajectory estimator config:
# prepare, smoke, build, evaluate, summarize. Needs the P2 snapshot, no database.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
CONFIG=${CONFIG:?set CONFIG to a config under evaluation/trajectory_pol_porto/configs}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
echo "config=$CONFIG seed=$SEED commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown) host=$(hostname)"
"$PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.cli run --config "$CONFIG" --seed "$SEED"
