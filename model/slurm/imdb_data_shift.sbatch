#!/bin/bash
#SBATCH --job-name=imdb_data_shift
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
# Data-shift ablation on JOB-light-ranges (decided 2026-11-17), one step for one split
# (submit_imdb_data_shift.sh chains them); DATA = $ROOT/data_shift/imdb/$SPLIT:
#   STEP=select   the kept titles and both join cardinalities        -> $DATA/split
#   STEP=subset   the six CSVs of the kept data                       -> $DATA/csv
#   STEP=relabel  the query file counted on the kept data (control)   -> $DATA/job_light_ranges_subset.csv
#   STEP=prepare  NeuroCard working directory + model preparation     -> $DATA/neurocard, $DATA/prepared
# REPO = a checkout of model/imdb-data-shift. Every step refuses existing outputs.
set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=${REPO:?set REPO to a checkout of model/imdb-data-shift}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
STEP=${STEP:?set STEP to select, subset, relabel or prepare}
SPLIT=${SPLIT:?set SPLIT to random80 or temporal80}
case "$SPLIT" in random80) MODE=random ;; temporal80) MODE=temporal ;; *) echo "SPLIT must be random80 or temporal80" >&2; exit 2 ;; esac
FULL_CSV=${FULL_CSV:-$ROOT/datasets/job}
FULL_PREPARED=${FULL_PREPARED:-$ROOT/Master_thesis_cardest/data/neurocard_prepared/job_light_ranges_content}
SHARED_NEUROCARD=${SHARED_NEUROCARD:-$ROOT/external/neurocard/neurocard}
QUERIES=${QUERIES:-$ROOT/external/distjoin/queries/job-light-ranges.csv}
DATA=$ROOT/data_shift/imdb/$SPLIT
mkdir -p "$DATA"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
echo "step=$STEP split=$SPLIT commit=$(git rev-parse HEAD) host=$(hostname) start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
case "$STEP" in
  select) "$PYTHON" -m model.scripts.imdb_data_shift select --csv-directory "$FULL_CSV" --mode "$MODE" --seed 0 \
            --output "$DATA/split" ;;
  subset) "$PYTHON" -m model.scripts.imdb_data_shift subset-csv --csv-directory "$FULL_CSV" --split "$DATA/split" \
            --output "$DATA/csv" ;;
  relabel) "$PYTHON" -m model.scripts.imdb_data_shift relabel --queries "$QUERIES" --full-csv "$FULL_CSV" \
             --subset-csv "$DATA/csv" --split "$DATA/split" --output "$DATA/job_light_ranges_subset.csv" ;;
  prepare) "$PYTHON" -m model.scripts.imdb_data_shift prepare --full-prepared "$FULL_PREPARED" --full-csv "$FULL_CSV" \
             --subset-csv "$DATA/csv" --split "$DATA/split" --shared-neurocard "$SHARED_NEUROCARD" \
             --neurocard-workdir "$DATA/neurocard" --output "$DATA/prepared" ;;
  *) echo "STEP must be select, subset, relabel or prepare" >&2; exit 2 ;;
esac
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
