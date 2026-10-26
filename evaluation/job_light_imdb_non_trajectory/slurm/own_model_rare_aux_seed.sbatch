#!/bin/bash
#SBATCH --job-name=own_rare_aux_seed
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
MODEL_REPO=$ROOT/Master_thesis_cardest
EVAL_REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
SEED=${SEED:?set SEED to 1 or 2}

case "$SEED" in
  1|2) ;;
  *) echo "SEED must be 1 or 2, received $SEED" >&2; exit 2 ;;
esac

SOURCE_CONFIG=$EVAL_REPO/evaluation/job_light_imdb_non_trajectory/training_configs/own_model_rare_aux_40k_seed_${SEED}.yaml
OUTPUT=$ROOT/joblight_evaluation/own_model_rare_aux_40k/seed_${SEED}
CONFIG=$OUTPUT/training_config.yaml
STARTUP=$OUTPUT/startup_check
EXPECTED_TRAINER_SHA=bb3aa5ec11753bc9b6c77509f5ac5ff4887b1eb009d95b0f1ca689658a6be45e

cd "$MODEL_REPO"
export PYTHONPATH="$MODEL_REPO:$EVAL_REPO:${PYTHONPATH:-}"
export PYTHONHASHSEED=$SEED
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

ACTUAL_TRAINER_SHA=$(sha256sum model/src/training/resmade_trainer.py | awk '{print $1}')
if [[ "$ACTUAL_TRAINER_SHA" != "$EXPECTED_TRAINER_SHA" ]]; then
  echo "trainer checksum mismatch: $ACTUAL_TRAINER_SHA != $EXPECTED_TRAINER_SHA" >&2
  exit 3
fi
if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "missing seed config: $SOURCE_CONFIG" >&2
  exit 4
fi
if [[ -e "$OUTPUT/training_summary.json" ]]; then
  echo "refusing to overwrite completed/in-progress output: $OUTPUT" >&2
  exit 5
fi

mkdir -p "$OUTPUT"
cp "$SOURCE_CONFIG" "$CONFIG"

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "seed=$SEED"
echo "config=$CONFIG"
echo "output_directory=$OUTPUT"
echo "trainer_sha256=$ACTUAL_TRAINER_SHA"
echo "main_batch=2048"
echo "rare_batch=512"
echo "beta=0.25"
echo "support_planning_steps=40000"
echo "steps_per_epoch=40000"
echo "early_stopping_patience_steps=5000"

"$PYTHON" - "$CONFIG" "$SEED" "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

path, seed, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
config = load_simple_yaml(path)
validate_config(config)
assert config["dataset"]["sampler_seed"] == seed
assert config["predicate_generation"]["seed"] == seed
assert config["training"]["seed"] == seed
assert Path(config["logging"]["output_directory"]) == output
assert config["training"]["batch_size"] == 2048
assert config["rare_auxiliary"]["batch_size"] == 512
assert config["rare_auxiliary"]["beta"] == 0.25
assert config["rare_support"]["discovery"]["support_planning_steps"] == 40000
assert config["training"]["steps_per_epoch"] == 40000
assert config["training"]["early_stopping_patience_steps"] == 5000
PY

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
