#!/bin/bash
#SBATCH --job-name=jlr_conj_prepare_2m
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
CONFIG=model/configs/job_light_duet_binary_native_anpm_20k_conjunctive_rare_auxiliary.yaml
PREPARED=data/neurocard_prepared/job_light_ranges_conjunctive_2m

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "prepared_directory=$PREPARED"
echo "sample_rows=2000000"

if [[ -e "$PREPARED/manifest.json" || -e "$PREPARED/sample_rows.npy" ]]; then
  echo "refusing to overwrite prepared artifacts: $PREPARED" >&2
  exit 3
fi

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from model.src.config import load_simple_yaml, validate_config

config = load_simple_yaml(Path(sys.argv[1]))
validate_config(config)
dataset = config["dataset"]
conjunctive = config["rare_support"]["discovery"]["conjunctive"]
assert dataset["sampling_mode"] == "fixture"
assert dataset["use_cols"] == "content"
assert Path(dataset["prepared_directory"]) == Path(
    "data/neurocard_prepared/job_light_ranges_conjunctive_2m"
)
assert conjunctive["enabled"] is True
assert conjunctive["require_conjunctions"] is True
assert conjunctive["minimum_materialized_rows"] == 2_000_000
PY

"$PYTHON" -m model.scripts.prepare_neurocard_data \
  --config "$CONFIG" \
  --rebuild-domains \
  --sample-rows 2000000

"$PYTHON" - "$PREPARED" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

prepared = Path(sys.argv[1])
manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
rows = np.load(prepared / "sample_rows.npy", mmap_mode="r")
columns = [column["name"] for column in manifest["metadata"]["columns"]]
required = {
    "title:production_year",
    "title:phonetic_code",
    "title:kind_id",
    "movie_keyword:keyword_id",
    "movie_info:info_type_id",
    "movie_info_idx:info_type_id",
}
assert manifest["neurocard_use_cols"] == "content"
assert manifest["sample_rows"] == 2_000_000
assert rows.shape == (2_000_000, 24), rows.shape
assert required.issubset(columns), sorted(required - set(columns))
print(f"validated_fixture_shape={rows.shape}")
print(f"validated_projection={manifest['neurocard_use_cols']}")
print(f"validated_columns={columns}")
PY

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
