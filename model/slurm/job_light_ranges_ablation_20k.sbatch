#!/bin/bash
#SBATCH --job-name=own_ablation_20k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
# A checkout of model/imdb-ablations (not the evaluation worktree).
REPO=${REPO:?set REPO to a checkout of model/imdb-ablations}
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
SEED=${SEED:?set SEED to 0, 1, or 2}
ABLATION=${ABLATION:?set ABLATION to imdb_ablation_no_native_ranges, imdb_ablation_embed_input or imdb_ablation_direct_io_off}
case "$ABLATION" in
  imdb_ablation_no_native_ranges|imdb_ablation_embed_input|imdb_ablation_direct_io_off) ;;
  *) echo "unknown ABLATION $ABLATION" >&2; exit 2 ;;
esac

case "$SEED" in
  0|1|2) ;;
  *) echo "SEED must be 0, 1, or 2; received $SEED" >&2; exit 2 ;;
esac

# Same 20k recipe (the sed below) as job_light_ranges_rare_aux_20k.sbatch, which trained
# the evaluated model; only the source config and the run directory differ.
SOURCE_CONFIG=model/configs/$ABLATION.yaml
CSV_DIRECTORY=$ROOT/datasets/job
PREPARED_DIRECTORY=$ROOT/Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content
RUN_DIR=$ROOT/joblight_evaluation/own_model_ablations/$ABLATION/seed_${SEED}
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
echo "ablation=$ABLATION"
echo "git_sha=$(git rev-parse HEAD)"
echo "source_config=$SOURCE_CONFIG"
echo "csv_directory=$CSV_DIRECTORY"
echo "prepared_directory=$PREPARED_DIRECTORY"
echo "config=$CONFIG"
echo "output_directory=$RUN_DIR"
echo "batch_size=2048"
echo "rare_batch_size=512"
echo "steps_per_epoch=20000"
echo "early_stopping_patience_steps=3000"
echo "support_planning_steps=20000"
echo "use_cols=content"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

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
  -e "s/^    support_planning_steps: 40000$/    support_planning_steps: 20000/" \
  -e "s/^  steps_per_epoch: 40000$/  steps_per_epoch: 20000/" \
  -e "s/^  early_stopping_patience_steps: 5000$/  early_stopping_patience_steps: 3000/" \
  -e "s#^  output_directory:.*#  output_directory: $RUN_DIR#" \
  "$SOURCE_CONFIG" > "$CONFIG"

"$PYTHON" - "$CONFIG" "$SEED" "$RUN_DIR" "$CSV_DIRECTORY" "$PREPARED_DIRECTORY" "$ABLATION" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

path, seed, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
csv_directory, prepared_directory = Path(sys.argv[4]), Path(sys.argv[5])
config = load_simple_yaml(path)
validate_config(config)
assert config["dataset"]["use_cols"] == "content"
assert Path(config["dataset"]["csv_directory"]) == csv_directory
assert Path(config["dataset"]["prepared_directory"]) == prepared_directory
assert (prepared_directory / "manifest.json").is_file()
assert config["dataset"]["sampler_seed"] == seed
assert config["predicate_generation"]["seed"] == seed
assert config["training"]["seed"] == seed
assert Path(config["logging"]["output_directory"]) == output
assert config["training"]["batch_size"] == 2048
assert config["rare_auxiliary"]["batch_size"] == 512
assert config["rare_auxiliary"]["beta"] == 0.25
assert config["rare_support"]["discovery"]["support_planning_steps"] == 20000
assert config["training"]["steps_per_epoch"] == 20000
assert config["training"]["early_stopping_patience_steps"] == 3000
ablation = sys.argv[6]
pg, enc, model = config["predicate_generation"], config["predicate_encoding"], config["model"]
expected = {
    "imdb_ablation_no_native_ranges": (False, 0.0, "duet_binary", "two_slot_binary_duet", True),
    "imdb_ablation_embed_input": (True, 0.2, "embed", "two_slot", True),
    "imdb_ablation_direct_io_off": (True, 0.2, "duet_binary", "two_slot_binary_duet", False),
}[ablation]
observed = (pg["enable_native_range_tokens"], pg["native_range_probability"], model["input_encoding"],
            enc["mode"], model["direct_io_connections"])
assert observed == expected, (ablation, observed, expected)
PY

"$PYTHON" -m model.scripts.check_live_sampler_startup \
  --config "$CONFIG" \
  --output-dir "$STARTUP" \
  --draw-rows 2048

"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
