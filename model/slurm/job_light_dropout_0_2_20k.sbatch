#!/bin/bash
#SBATCH --job-name=own_dropout02_20k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=model/configs/job_light_duet_binary_native_anpm_20k_rare_auxiliary_dropout_0_2.yaml
BASE_CONFIG=model/configs/job_light_duet_binary_native_anpm_20k_rare_auxiliary.yaml
RUN_DIR=model/runs/job_light_duet_binary_native_anpm_20k_rare_auxiliary_dropout_0_2
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
echo "config=$CONFIG"
echo "output_directory=$RUN_DIR"
echo "residual_dropout=0.2"
echo "steps_per_epoch=20000"
echo "early_stopping_patience_steps=3000"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -e "$RUN_DIR/training_summary.json" || -e "$RUN_DIR/checkpoint_step_500.pt" ]]; then
  echo "refusing to overwrite completed or started run: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" - "$CONFIG" "$BASE_CONFIG" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config
from model.src.model.resmade import PredicateResMADEConfig

path, base_path = map(Path, sys.argv[1:])
config = load_simple_yaml(path)
base_config = load_simple_yaml(base_path)
validate_config(config)
assert config["model"]["residual_dropout"] == 0.2
assert base_config["model"]["residual_dropout"] == 0.0
assert PredicateResMADEConfig.__dataclass_fields__["residual_dropout"].default == 0.0
assert config["training"]["steps_per_epoch"] == 20000
assert config["training"]["early_stopping_patience_steps"] == 3000
assert config["training"]["checkpoint_interval_steps"] == 500
assert config["training"]["batch_size"] == 2048
assert config["rare_auxiliary"]["batch_size"] == 512
assert config["rare_auxiliary"]["beta"] == 0.25
assert Path(config["logging"]["output_directory"]) == Path(
    "model/runs/job_light_duet_binary_native_anpm_20k_rare_auxiliary_dropout_0_2"
)
PY

mkdir -p "$RUN_DIR"
"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
