#!/bin/bash
#SBATCH --job-name=joblight_ranges_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=${CONFIG:-model/configs/job_light_ranges_duet_binary_native_anpm_40k_rare_auxiliary.yaml}

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

"$PYTHON" -m model.scripts.prepare_neurocard_data \
  --config "$CONFIG" \
  --rebuild-domains \
  --sample-rows 2048

"$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from model.src.config import load_simple_yaml

config = load_simple_yaml(sys.argv[1])
prepared = Path(config["dataset"]["prepared_directory"])
manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
columns = {column["name"] for column in manifest["metadata"]["columns"]}
required = {
    "title:imdb_index",
    "title:phonetic_code",
    "title:season_nr",
    "title:episode_nr",
    "title:series_years",
    "cast_info:nr_order",
}
missing = sorted(required - columns)
if missing:
    raise SystemExit(f"prepared content schema is missing columns: {missing}")
print(f"prepared_column_count={len(columns)}")
print("required_range_columns_present=true")
PY

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
