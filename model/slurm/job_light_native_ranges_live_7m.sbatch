#!/bin/bash
#SBATCH --job-name=joblight_live_c_7m
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_live_c_7m_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_live_c_7m_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=model/configs/job_light_native_ranges_live_7m.yaml
QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light.csv
RUN_DIR=model/runs/job_light_native_ranges_live_7m
CHECKPOINT=$RUN_DIR/checkpoint_step_3500.pt
RESULTS=$RUN_DIR/job_light_query_results_native_ranges.csv
SUMMARY=$RUN_DIR/job_light_query_summary_native_ranges.json

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

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "training_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$PYTHON" -m model.scripts.evaluate_job_light_queries \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --queries "$QUERIES" \
  --results "$RESULTS" \
  --summary "$SUMMARY"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
