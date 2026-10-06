#!/bin/bash
#SBATCH --job-name=joblight_aggregate
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages
CONFIG=${CONFIG:?set CONFIG to an evaluation config path}

cd "$REPO"
export PYTHONPATH="$PACKAGES:$REPO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
  aggregate --config "$CONFIG"
