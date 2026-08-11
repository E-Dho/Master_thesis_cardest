#!/bin/bash
#SBATCH --job-name=joblight_c_sweep
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_c_sweep_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/joblight_c_sweep_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=model/configs/job_light_native_ranges_live_7m.yaml
QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light.csv
C_RUN=model/runs/job_light_native_ranges_live_7m
SWEEP_DIR=$C_RUN/tail_ablations/checkpoint_sweep
ANALYSIS_DIR=$C_RUN/tail_ablations/analysis

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$SWEEP_DIR" "$ANALYSIS_DIR"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"

for step in 500 1000 1500 2000 2500 3000 3500; do
  checkpoint=$C_RUN/checkpoint_step_${step}.pt
  results=$SWEEP_DIR/job_light_query_results_step_${step}.csv
  summary=$SWEEP_DIR/job_light_query_summary_step_${step}.json
  echo "evaluating_step=$step checkpoint=$checkpoint"
  "$PYTHON" -m model.scripts.evaluate_job_light_queries \
    --config "$CONFIG" \
    --checkpoint "$checkpoint" \
    --queries "$QUERIES" \
    --results "$results" \
    --summary "$summary"
done

"$PYTHON" -m model.scripts.analyze_job_light_tail \
  --results "$C_RUN/job_light_query_results_native_ranges.csv" \
  --checkpoint "$C_RUN/checkpoint_step_3500.pt" \
  --training-metrics "$C_RUN/training_metrics.jsonl" \
  --sweep-dir "$SWEEP_DIR" \
  --output-dir "$ANALYSIS_DIR"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
