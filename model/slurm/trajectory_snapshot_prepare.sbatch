#!/bin/bash
#SBATCH --job-name=traj_snapshot_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# Model training data from a v3 evaluation snapshot (model/scripts/prepare_trajectory_snapshot.py).
#   MODE=pol_check  prepare POL from its snapshot into CHECK_DIR and compare with the staging-TSV
#                   preparation that trained the POL model (columns, domains, every encoded row,
#                   provenance arrays, trajectory index arrays); exit 1 on any difference.
#   MODE=porto      prepare Porto into the config's dataset.prepared_directory / trajectory_index_path.
set -euo pipefail

ROOT=${ROOT:-/work_beegfs/sunip956/master_thesis_trajectories}
REPO=${REPO:?set REPO to the checkout of model/porto-segments-first}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
MODE=${MODE:?set MODE=pol_check or porto}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "job_id=${SLURM_JOB_ID:-manual} host=$(hostname) start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_sha=$(git rev-parse HEAD) status=$(git status --porcelain -- model | wc -l)"
"$PYTHON" -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"

case "$MODE" in
  pol_check)
    CONFIG=${CONFIG:-model/configs/pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k.yaml}
    SNAPSHOT=${SNAPSHOT:-$ROOT/traj_snapshots/pol_v3}
    CHECK_DIR=${CHECK_DIR:?set CHECK_DIR to a new directory}
    REFERENCE=$("$PYTHON" -c "from model.src.config import load_simple_yaml as l; print(l('$CONFIG')['dataset']['prepared_directory'])")
    REFERENCE_INDEX=$("$PYTHON" -c "from model.src.config import load_simple_yaml as l; print(l('$CONFIG')['dataset']['trajectory_index_path'])")
    if [ -e "$CHECK_DIR" ]; then echo "$CHECK_DIR exists" >&2; exit 2; fi
    "$PYTHON" -m model.scripts.prepare_trajectory_snapshot prepare --dataset pol --config "$CONFIG" \
      --snapshot "$SNAPSHOT" --prepared-directory "$CHECK_DIR/prepared"
    "$PYTHON" -m model.scripts.prepare_trajectory_snapshot compare --prepared "$CHECK_DIR/prepared" \
      --reference "$REFERENCE" --index "$CHECK_DIR/prepared/trajectory_segment_index" \
      --reference-index "$REFERENCE_INDEX" --output "$CHECK_DIR/compare.json"
    ;;
  porto)
    CONFIG=${CONFIG:-model/configs/porto_traj_dedup_single_anchor_mbr_segments_first_20k.yaml}
    SNAPSHOT=${SNAPSHOT:-$ROOT/traj_snapshots/porto_v3}
    "$PYTHON" -m model.scripts.prepare_trajectory_snapshot prepare --dataset porto --config "$CONFIG" \
      --snapshot "$SNAPSHOT"
    ;;
  *) echo "MODE must be pol_check or porto" >&2; exit 2 ;;
esac
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
