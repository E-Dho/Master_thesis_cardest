#!/bin/bash
#SBATCH --job-name=own_ranges_40k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
SEED=${SEED:?set SEED to 0, 1, or 2}

case "$SEED" in
  0|1|2) ;;
  *) echo "SEED must be 0, 1, or 2; received $SEED" >&2; exit 2 ;;
esac

SOURCE_CONFIG=${SOURCE_CONFIG:-model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml}
RUN_DIR=$ROOT/joblight_evaluation/own_model_job_light_ranges_rare_aux_40k/seed_${SEED}
CONFIG=$RUN_DIR/training_config.yaml
STARTUP=$RUN_DIR/startup_check_${SLURM_JOB_ID:-manual}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "seed=$SEED"
echo "source_config=$SOURCE_CONFIG"
echo "config=$CONFIG"
echo "output_directory=$RUN_DIR"
echo "batch_size=2048"
echo "rare_batch_size=512"
echo "steps_per_epoch=40000"
echo "early_stopping_patience_steps=5000"
echo "support_planning_steps=40000"
echo "use_cols=content"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -e "$RUN_DIR/training_summary.json" || -e "$RUN_DIR/checkpoint_step_500.pt" ]]; then
  echo "refusing to overwrite completed or started run: $RUN_DIR" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
sed \
  -e "s/^  sampler_seed: 0$/  sampler_seed: $SEED/" \
  -e "s/^  seed: 0$/  seed: $SEED/g" \
  -e "s#^  output_directory:.*#  output_directory: $RUN_DIR#" \
  "$SOURCE_CONFIG" > "$CONFIG"

"$PYTHON" - "$CONFIG" "$SEED" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

path, seed, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
config = load_simple_yaml(path)
validate_config(config)
assert config["dataset"]["use_cols"] == "content"
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
