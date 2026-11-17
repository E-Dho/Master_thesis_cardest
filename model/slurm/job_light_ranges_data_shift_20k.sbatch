#!/bin/bash
#SBATCH --job-name=own_shift_20k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# Data-shift ablation on JOB-light-ranges (decided 2026-11-17), seed 0:
#   KIND=stale     the evaluated model's 20k recipe (job_light_ranges_rare_aux_20k.sbatch) on the
#                  80 % data of SPLIT: its CSVs, preparation and NeuroCard working directory
#                  ($ROOT/data_shift/imdb/$SPLIT, model/scripts/imdb_data_shift.py);
#   KIND=finetune  the stale run's checkpoint_step_11000 (the evaluated model's step) trained
#                  2,000 more steps (10 % of the training) at a tenth of the learning rate (1e-5)
#                  on the full data, rare-support planning over those steps.
# REPO = a checkout of model/imdb-data-shift. Run directory:
# $ROOT/joblight_evaluation/own_model_data_shift/<SPLIT>_<KIND>/seed_0.
set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=${REPO:?set REPO to a checkout of model/imdb-data-shift}
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
KIND=${KIND:?set KIND to stale or finetune}
SPLIT=${SPLIT:?set SPLIT to random80 or temporal80}
case "$SPLIT" in random80|temporal80) ;; *) echo "SPLIT must be random80 or temporal80" >&2; exit 2 ;; esac
SEED=0
SOURCE_CONFIG=model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml
DATA=$ROOT/data_shift/imdb/$SPLIT
RUN_ROOT=$ROOT/joblight_evaluation/own_model_data_shift
case "$KIND" in
  stale)
    CSV_DIRECTORY=$DATA/csv
    PREPARED_DIRECTORY=$DATA/prepared
    NEUROCARD_PATH=$DATA/neurocard/neurocard
    STEPS=20000; SUPPORT_STEPS=20000; LEARNING_RATE=0.0001; INIT_CHECKPOINT="" ;;
  finetune)
    CSV_DIRECTORY=$ROOT/datasets/job
    PREPARED_DIRECTORY=$ROOT/Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content
    NEUROCARD_PATH=$ROOT/external/neurocard/neurocard
    STEPS=2000; SUPPORT_STEPS=2000; LEARNING_RATE=0.00001
    INIT_CHECKPOINT=$RUN_ROOT/${SPLIT}_stale/seed_0/checkpoint_step_11000.pt
    if [[ ! -f "$INIT_CHECKPOINT" ]]; then echo "missing $INIT_CHECKPOINT" >&2; exit 3; fi ;;
  *) echo "KIND must be stale or finetune" >&2; exit 2 ;;
esac
RUN_DIR=$RUN_ROOT/${SPLIT}_${KIND}/seed_${SEED}
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
echo "kind=$KIND split=$SPLIT seed=$SEED"
echo "git_sha=$(git rev-parse HEAD)"
echo "csv_directory=$CSV_DIRECTORY"
echo "prepared_directory=$PREPARED_DIRECTORY"
echo "neurocard_path=$NEUROCARD_PATH"
echo "init_checkpoint=${INIT_CHECKPOINT:-none}"
echo "steps_per_epoch=$STEPS support_planning_steps=$SUPPORT_STEPS learning_rate=$LEARNING_RATE"
echo "output_directory=$RUN_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -e "$RUN_DIR" ]]; then
  echo "refusing to reuse an existing run directory: $RUN_DIR" >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
# The evaluated model's 20k recipe (job_light_ranges_rare_aux_20k.sbatch), then the data-shift changes.
sed \
  -e "s#^  csv_directory:.*#  csv_directory: $CSV_DIRECTORY#" \
  -e "s#^  prepared_directory:.*#  prepared_directory: $PREPARED_DIRECTORY#" \
  -e "s#^  neurocard_path:.*#  neurocard_path: $NEUROCARD_PATH#" \
  -e "s/^  sampler_seed: 0$/  sampler_seed: $SEED/" \
  -e "s/^  seed: 0$/  seed: $SEED/g" \
  -e "s/^    support_planning_steps: 40000$/    support_planning_steps: $SUPPORT_STEPS/" \
  -e "s/^  steps_per_epoch: 40000$/  steps_per_epoch: $STEPS/" \
  -e "s/^  learning_rate: 0.0001$/  learning_rate: $LEARNING_RATE/" \
  -e "s/^  early_stopping_patience_steps: 5000$/  early_stopping_patience_steps: 3000/" \
  -e "s#^  output_directory:.*#  output_directory: $RUN_DIR#" \
  "$SOURCE_CONFIG" > "$CONFIG"
if [[ -n "$INIT_CHECKPOINT" ]]; then
  awk -v line="  init_checkpoint: $INIT_CHECKPOINT" '{ print } $0 == "training:" { print line }' "$CONFIG" > "$CONFIG.tmp"
  mv "$CONFIG.tmp" "$CONFIG"
fi

"$PYTHON" - "$CONFIG" "$RUN_DIR" "$CSV_DIRECTORY" "$PREPARED_DIRECTORY" "$NEUROCARD_PATH" "$KIND" "$STEPS" "$LEARNING_RATE" "${INIT_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

path, output = Path(sys.argv[1]), Path(sys.argv[2])
csv_directory, prepared_directory, neurocard = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
kind, steps, learning_rate, init = sys.argv[6], int(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
config = load_simple_yaml(path)
validate_config(config)
dataset, training = config["dataset"], config["training"]
assert dataset["use_cols"] == "content" and dataset["sampling_mode"] == "live"
assert Path(dataset["csv_directory"]) == csv_directory
assert Path(dataset["prepared_directory"]) == prepared_directory
assert Path(dataset["neurocard_path"]) == neurocard
assert (prepared_directory / "manifest.json").is_file()
assert dataset["sampler_seed"] == 0 and config["predicate_generation"]["seed"] == 0 and training["seed"] == 0
assert Path(config["logging"]["output_directory"]) == output
assert training["batch_size"] == 2048
assert config["rare_auxiliary"]["batch_size"] == 512 and config["rare_auxiliary"]["beta"] == 0.25
assert config["rare_support"]["discovery"]["support_planning_steps"] == steps
assert training["steps_per_epoch"] == steps
assert training["early_stopping_patience_steps"] == 3000
assert float(training["learning_rate"]) == learning_rate
assert (training.get("init_checkpoint") or "") == init
if kind == "stale":
    subset = json.loads((prepared_directory / "subset.json").read_text())
    assert Path(subset["neurocard_package"]) == neurocard, subset
    manifest = json.loads((prepared_directory / "manifest.json").read_text())
    assert manifest["join_cardinality"] == subset["subset_join_cardinality"] < subset["full_join_cardinality"]
print(f"config_checked kind={kind}")
PY

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
