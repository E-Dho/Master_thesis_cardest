#!/bin/bash
#SBATCH --job-name=joblight_tail
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light.csv

CONFIG=${CONFIG:?set CONFIG to a model config path}
RUN_DIR=${RUN_DIR:?set RUN_DIR to the config output directory}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-3500}
CHECKPOINT=$RUN_DIR/checkpoint_step_${CHECKPOINT_STEP}.pt
RESULTS=$RUN_DIR/job_light_query_results_native_ranges.csv
SUMMARY=$RUN_DIR/job_light_query_summary_native_ranges.json
ANALYSIS_DIR=$RUN_DIR/tail_analysis

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "run_dir=$RUN_DIR"
echo "checkpoint_step=$CHECKPOINT_STEP"

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "training_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" -m model.scripts.evaluate_job_light_queries \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --queries "$QUERIES" \
  --results "$RESULTS" \
  --summary "$SUMMARY"

"$PYTHON" -m model.scripts.analyze_job_light_tail \
  --results "$RESULTS" \
  --checkpoint "$CHECKPOINT" \
  --training-metrics "$RUN_DIR/training_metrics.jsonl" \
  --output-dir "$ANALYSIS_DIR"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
