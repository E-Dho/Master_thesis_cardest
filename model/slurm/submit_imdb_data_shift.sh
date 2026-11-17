#!/bin/bash
# Data-shift ablation on JOB-light-ranges, data side for one split (decided 2026-11-17):
#   SPLIT=random80|temporal80 REPO=<checkout of model/imdb-data-shift> model/slurm/submit_imdb_data_shift.sh
# select -> subset -> relabel and subset -> prepare. Prints the job ids; the stale training
# (job_light_ranges_data_shift_20k.sbatch KIND=stale) is submitted after PREPARE has been checked.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPLIT=${SPLIT:?set SPLIT=random80 or temporal80}
REPO=${REPO:?set REPO to a checkout of model/imdb-data-shift}
case "$SPLIT" in random80|temporal80) ;; *) echo "SPLIT must be random80 or temporal80" >&2; exit 2 ;; esac
DATA=/work_beegfs/sunip956/master_thesis_trajectories/data_shift/imdb/$SPLIT
if [ -e "$DATA" ]; then echo "$DATA exists; refusing to submit over it" >&2; exit 2; fi
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
export_vars="ALL,REPO=$REPO,SPLIT=$SPLIT"
select_job=$(submit --mem=64G --time=02:00:00 --export="$export_vars,STEP=select" "$SCRIPT_DIR/imdb_data_shift.sbatch")
subset_job=$(submit --dependency="afterok:$select_job" --mem=96G --time=06:00:00 \
  --export="$export_vars,STEP=subset" "$SCRIPT_DIR/imdb_data_shift.sbatch")
relabel_job=$(submit --dependency="afterok:$subset_job" --mem=96G --time=06:00:00 \
  --export="$export_vars,STEP=relabel" "$SCRIPT_DIR/imdb_data_shift.sbatch")
prepare_job=$(submit --dependency="afterok:$subset_job" --mem=128G --time=12:00:00 \
  --export="$export_vars,STEP=prepare" "$SCRIPT_DIR/imdb_data_shift.sbatch")
echo "SELECT_JOB=$select_job SUBSET_JOB=$subset_job RELABEL_JOB=$relabel_job PREPARE_JOB=$prepare_job"
echo "data: $DATA"
