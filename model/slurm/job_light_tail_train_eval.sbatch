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
RESULTS=$RUN_DIR/job_light_query_results_native_ranges.csv
SUMMARY=$RUN_DIR/job_light_query_summary_native_ranges.json
ANALYSIS_DIR=$RUN_DIR/tail_analysis
PRETRAIN_STARTUP_CHECK=${PRETRAIN_STARTUP_CHECK:-0}
STARTUP_CHECK_DRAW_ROWS=${STARTUP_CHECK_DRAW_ROWS:-0}
STARTUP_CHECK_DIR=${STARTUP_CHECK_DIR:-$RUN_DIR/startup_checks/${SLURM_JOB_ID:-manual}_pretrain}

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
echo "pretrain_startup_check=$PRETRAIN_STARTUP_CHECK"

if [[ "$PRETRAIN_STARTUP_CHECK" == "1" ]]; then
  echo "startup_check_dir=$STARTUP_CHECK_DIR"
  echo "startup_check_draw_rows=$STARTUP_CHECK_DRAW_ROWS"
  "$PYTHON" -m model.scripts.check_live_sampler_startup \
    --config "$CONFIG" \
    --output-dir "$STARTUP_CHECK_DIR" \
    --draw-rows "$STARTUP_CHECK_DRAW_ROWS"
  echo "startup_check_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "training_done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$CHECKPOINT_STEP" == "auto" ]]; then
  CHECKPOINT=$("$PYTHON" - "$RUN_DIR/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(summary["checkpoint"])
PY
)
else
  CHECKPOINT=$RUN_DIR/checkpoint_step_${CHECKPOINT_STEP}.pt
fi
echo "checkpoint=$CHECKPOINT"

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
