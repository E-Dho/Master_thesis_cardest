#!/bin/bash
#SBATCH --job-name=own_main_rare_off
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

# Main IMDB models with rare support off (decided 2026-11-25), three seeds per workload
# (WORKLOAD=job_light or job_light_ranges). Since 2026-11-26 the recipe of the POL/Porto
# mains for both workloads: 20,000 steps, no early stopping, validation every 250 steps on
# fresh sampler batches, and checkpoint_best.pt (lowest validation weighted NLL) is the
# evaluated checkpoint (the template's validation section; steps, patience and planning set
# here). Training runs under learned-gpu through the config's runtime section.

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=${REPO:?set REPO to a checkout of model/imdb-ablations}
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
SEED=${SEED:?set SEED to 0, 1, or 2}
WORKLOAD=${WORKLOAD:?set WORKLOAD to job_light or job_light_ranges}

case "$SEED" in
  0|1|2) ;;
  *) echo "SEED must be 0, 1, or 2; received $SEED" >&2; exit 2 ;;
esac

CSV_DIRECTORY=$ROOT/datasets/job
case "$WORKLOAD" in
  job_light_ranges)
    SOURCE_CONFIG=model/configs/imdb_main_job_light_ranges_rare_off.yaml
    PREPARED_DIRECTORY=$ROOT/Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content
    RUN_DIR=$ROOT/joblight_evaluation/own_model_job_light_ranges_rare_off_20k/seed_${SEED}
    ;;
  job_light)
    SOURCE_CONFIG=model/configs/imdb_main_job_light_rare_off.yaml
    PREPARED_DIRECTORY=$ROOT/Master_thesis_cardest/data/neurocard_prepared/job_light
    RUN_DIR=$ROOT/joblight_evaluation/own_model_job_light_rare_off_20k/seed_${SEED}
    ;;
  *) echo "unknown WORKLOAD $WORKLOAD" >&2; exit 2 ;;
esac
STEPS=20000; PATIENCE=0; PLANNING=20000
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
echo "workload=$WORKLOAD"
echo "git_sha=$(git rev-parse HEAD)"
echo "source_config=$SOURCE_CONFIG"
echo "prepared_directory=$PREPARED_DIRECTORY"
echo "config=$CONFIG"
echo "output_directory=$RUN_DIR"
echo "steps_per_epoch=$STEPS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

if [[ -e "$RUN_DIR/training_summary.json" || -e "$RUN_DIR/checkpoint_step_500.pt" ]]; then
  echo "refusing to overwrite completed or started run: $RUN_DIR" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
sed \
  -e "s#^  csv_directory:.*#  csv_directory: $CSV_DIRECTORY#" \
  -e "s#^  prepared_directory:.*#  prepared_directory: $PREPARED_DIRECTORY#" \
  -e "s/^  sampler_seed: 0$/  sampler_seed: $SEED/" \
  -e "s/^  seed: 0$/  seed: $SEED/g" \
  -e "s/^    support_planning_steps: 40000$/    support_planning_steps: $PLANNING/" \
  -e "s/^  steps_per_epoch: 40000$/  steps_per_epoch: $STEPS/" \
  -e "s/^  early_stopping_patience_steps: 5000$/  early_stopping_patience_steps: $PATIENCE/" \
  -e "s#^  output_directory:.*#  output_directory: $RUN_DIR#" \
  "$SOURCE_CONFIG" > "$CONFIG"

"$PYTHON" - "$CONFIG" "$SEED" "$RUN_DIR" "$CSV_DIRECTORY" "$PREPARED_DIRECTORY" "$STEPS" "$PATIENCE" "$PLANNING" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

path, seed, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
csv_directory, prepared_directory = Path(sys.argv[4]), Path(sys.argv[5])
steps, patience, planning = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
config = load_simple_yaml(path)
validate_config(config)
assert Path(config["dataset"]["csv_directory"]) == csv_directory
assert Path(config["dataset"]["prepared_directory"]) == prepared_directory
assert (prepared_directory / "manifest.json").is_file()
assert config["dataset"]["sampler_seed"] == seed
assert config["predicate_generation"]["seed"] == seed
assert config["training"]["seed"] == seed
assert Path(config["logging"]["output_directory"]) == output
assert config["training"]["batch_size"] == 2048
assert config["training"]["steps_per_epoch"] == steps
assert config["training"]["early_stopping_patience_steps"] == patience
assert config["rare_support"]["discovery"]["support_planning_steps"] == planning
assert config["rare_support"]["enabled"] is False
assert config["rare_auxiliary"]["enabled"] is False
assert config["runtime"]["require_cuda"] is True
validation = config["validation"]
assert validation["enabled"] is True and validation["interval_steps"] == 250
assert validation["fresh_sampler_batches"] == 4 and validation["batch_size"] == 2048
assert validation["selection_metric"] == "validation_weighted_nll" and validation["minimize"] is True
PY

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  # Check the generated config and its asserts only (e.g. on a login node).
  echo "dry_run=ok config=$CONFIG"
  exit 0
fi

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
