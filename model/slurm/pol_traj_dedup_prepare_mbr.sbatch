#!/bin/bash
#SBATCH --job-name=pol_mbr_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=${ROOT:-/work_beegfs/sunip956/master_thesis_trajectories}
REPO=${REPO:-$ROOT/Master_thesis_cardest_git}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
CONFIG=${CONFIG:-model/configs/pol_50m_traj_dedup_single_anchor_mbr.yaml}
STAGING_DIR=${STAGING_DIR:-$ROOT/mobilitydb/staging/pol_mobilitydb_50m_load_20260730T121326Z}
SAMPLE_ROWS=${SAMPLE_ROWS:-0}
FORCE_REBUILD_INDEX=${FORCE_REBUILD_INDEX:-0}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "config=$CONFIG"
echo "staging_dir=$STAGING_DIR"
echo "sample_rows=$SAMPLE_ROWS"
echo "force_rebuild_index=$FORCE_REBUILD_INDEX"
echo "git_sha=$(git rev-parse HEAD)"

ARGS=(
  -m model.scripts.prepare_pol_staging_data
  --config "$CONFIG"
  --staging-dir "$STAGING_DIR"
  --sample-rows "$SAMPLE_ROWS"
)
if [[ "$FORCE_REBUILD_INDEX" == "1" ]]; then
  ARGS+=(--force-rebuild-index)
fi

"$PYTHON" "${ARGS[@]}"

echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
