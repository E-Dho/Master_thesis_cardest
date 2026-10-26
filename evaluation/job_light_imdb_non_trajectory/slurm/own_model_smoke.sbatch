#!/bin/bash
#SBATCH --job-name=own_model_eval_smoke
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
MODEL_CONFIG=${MODEL_CONFIG:?MODEL_CONFIG must name the model YAML}
CHECKPOINT=${CHECKPOINT:?CHECKPOINT must name the checkpoint}
OUTPUT=${OUTPUT:-$ROOT/joblight_evaluation/own_model_smoke}
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/own_model_bridge.py smoke \
  --config "$MODEL_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output-directory "$OUTPUT" \
  --queries "$ROOT/external/neurocard/neurocard/queries/job-light.csv" \
  --query-limit 2 \
  --warmup-passes 1 \
  --repetitions 2
