#!/bin/bash
#SBATCH --job-name=jlr_conj_20k_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=${CONFIG:-model/configs/job_light_duet_binary_native_anpm_20k_conjunctive_rare_auxiliary.yaml}
RUN_DIR=${RUN_DIR:-model/runs/job_light_ranges_duet_binary_native_anpm_20k_conjunctive_rare_auxiliary_seed_0}
EXPECTED_DROPOUT=${EXPECTED_DROPOUT:-0.0}
RANGE_QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light-ranges.csv
NORMAL_QUERIES=$ROOT/external/neurocard/neurocard/queries/job-light.csv
RESULTS=$RUN_DIR/job_light_ranges_results.csv
SUMMARY=$RUN_DIR/job_light_ranges_summary.json

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
echo "run_dir=$RUN_DIR"
echo "residual_dropout=$EXPECTED_DROPOUT"
echo "evaluation_workload=$RANGE_QUERIES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ -e "$RUN_DIR/training_summary.json" || -e "$RUN_DIR/checkpoint_step_500.pt" ]]; then
  echo "refusing to overwrite completed or started run: $RUN_DIR" >&2
  exit 3
fi

"$PYTHON" - "$CONFIG" "$RANGE_QUERIES" "$NORMAL_QUERIES" "$RUN_DIR" "$EXPECTED_DROPOUT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from model.src.config import load_simple_yaml, validate_config

config_path, range_path, normal_path = map(Path, sys.argv[1:4])
run_dir = Path(sys.argv[4])
expected_dropout = float(sys.argv[5])
config = load_simple_yaml(config_path)
validate_config(config)
range_count = sum(bool(line.strip()) for line in range_path.read_text(encoding="utf-8").splitlines())
normal_count = sum(bool(line.strip()) for line in normal_path.read_text(encoding="utf-8").splitlines())
assert range_path.name == "job-light-ranges.csv"
assert normal_path.name == "job-light.csv"
assert range_count == 1_000, range_count
assert normal_count == 70, normal_count
assert range_path.resolve() != normal_path.resolve()
dataset = config["dataset"]
training = config["training"]
conjunctive = config["rare_support"]["discovery"]["conjunctive"]
prepared = Path(dataset["prepared_directory"])
manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
rows = np.load(prepared / "sample_rows.npy", mmap_mode="r")
assert dataset["sampling_mode"] == "fixture"
assert dataset["use_cols"] == "content"
assert manifest["neurocard_use_cols"] == "content"
assert rows.shape == (2_000_000, 24), rows.shape
assert training["batch_size"] == 2048
assert training["steps_per_epoch"] == 20_000
assert training["early_stopping_patience_steps"] == 3_000
assert training["checkpoint_interval_steps"] == 500
assert config["rare_auxiliary"]["batch_size"] == 512
assert config["rare_auxiliary"]["beta"] == 0.25
assert conjunctive["require_conjunctions"] is False
assert config["model"]["residual_dropout"] == expected_dropout
assert Path(config["logging"]["output_directory"]) == run_dir
print(f"confirmed_range_workload_queries={range_count}")
print(f"confirmed_normal_workload_queries={normal_count}")
print("confirmed_evaluation_workload=job-light-ranges.csv")
print(f"confirmed_residual_dropout={expected_dropout}")
PY

mkdir -p "$RUN_DIR"
"$PYTHON" -m model.scripts.train_resmade --config "$CONFIG"

CHECKPOINT=$("$PYTHON" - "$RUN_DIR/training_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(summary["checkpoint"])
PY
)
echo "training_checkpoint=$CHECKPOINT"

"$PYTHON" -m model.scripts.evaluate_job_light_queries \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --queries "$RANGE_QUERIES" \
  --results "$RESULTS" \
  --summary "$SUMMARY"

"$PYTHON" - "$SUMMARY" "$RANGE_QUERIES" <<'PY'
import json
import sys
from pathlib import Path

summary_path, expected_queries = map(Path, sys.argv[1:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
assert summary["queries_total"] == 1_000, summary["queries_total"]
assert Path(summary["queries_path"]).resolve() == expected_queries.resolve()
print(f"validated_evaluation_queries_total={summary['queries_total']}")
print(f"validated_evaluation_queries_path={summary['queries_path']}")
PY

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
