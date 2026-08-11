#!/bin/bash
#SBATCH --job-name=joblight_live_d_7m
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_live_d_7m_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_live_d_7m_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=model/configs/job_light_native_ranges_live_validation_7m.yaml
QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light.csv
RUN_DIR=model/runs/job_light_native_ranges_live_validation_7m
CHECKPOINT_LAST=$RUN_DIR/checkpoint_step_3500.pt
CHECKPOINT_BEST=$RUN_DIR/checkpoint_best.pt

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "batch_size=2048"
echo "steps_per_epoch=3500"
echo "expected_nominal_rows_seen=7168000"
echo "sampling_mode=live"
echo "sampler_batch_size=16384"
echo "validation_interval_steps=250"
echo "validation_fresh_sampler_batches=10"
echo "validation_selection_metric=validation_weighted_nll"

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "training_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" -m model.scripts.evaluate_job_light_queries \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_LAST" \
  --queries "$QUERIES" \
  --results "$RUN_DIR/job_light_query_results_native_ranges_last.csv" \
  --summary "$RUN_DIR/job_light_query_summary_native_ranges_last.json"

"$PYTHON" -m model.scripts.evaluate_job_light_queries \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_BEST" \
  --queries "$QUERIES" \
  --results "$RUN_DIR/job_light_query_results_native_ranges_best.csv" \
  --summary "$RUN_DIR/job_light_query_summary_native_ranges_best.json"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
