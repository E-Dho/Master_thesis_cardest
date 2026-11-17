#!/bin/bash
# Data-shift ablation, data side, for one dataset and split (decided 2026-11-17):
#   TRAJ_DATASET=pol_v3|porto_v3 SPLIT=random80|temporal80 MODEL_REPO=<checkout of model/data-shift> ./submit_data_shift.sh
# select -> subset snapshot -> gate array (GATE_SHARDS, default 16) -> relabel, and
# select -> prepare (the 80 % model preparation, from the full one). Prints the job ids;
# PREPARE_JOB is what the stale model's training waits for.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
TRAJ_DATASET=${TRAJ_DATASET:-pol_v3}
SPLIT=${SPLIT:?set SPLIT=random80 or temporal80}
MODEL_REPO=${MODEL_REPO:?set MODEL_REPO to a checkout of model/data-shift}
GATE_SHARDS=${GATE_SHARDS:-16}
_work=/work_beegfs/sunip956/master_thesis_trajectories
case "$TRAJ_DATASET" in
  pol_v3) FULL_PREPARED=${FULL_PREPARED:-$_work/Master_thesis_cardest/data/pol_prepared/pol_50m_mbr} ;;
  porto_v3) FULL_PREPARED=${FULL_PREPARED:-$_work/model_data/porto_v3_mbr} ;;
  *) echo "TRAJ_DATASET must be pol_v3 or porto_v3" >&2; exit 2 ;;
esac
DATA=$_work/data_shift/$TRAJ_DATASET/$SPLIT
if [ -e "$DATA" ]; then echo "$DATA exists; refusing to submit over it" >&2; exit 2; fi
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
export_vars="ALL,REPO_ROOT=$REPO_ROOT,TRAJ_DATASET=$TRAJ_DATASET,SPLIT=$SPLIT"
select_job=$(submit --mem=32G --time=01:00:00 --export="$export_vars,STEP=select" "$SCRIPT_DIR/data_shift.sbatch")
subset_job=$(submit --dependency="afterok:$select_job" --mem=64G --time=02:00:00 \
  --export="$export_vars,STEP=subset" "$SCRIPT_DIR/data_shift.sbatch")
gate_job=$(submit --dependency="afterok:$subset_job" --array="0-$((GATE_SHARDS - 1))" --cpus-per-task=1 --mem=40G \
  --time=06:00:00 --export="$export_vars,STEP=gate,GATE_SHARDS=$GATE_SHARDS" "$SCRIPT_DIR/data_shift.sbatch")
relabel_job=$(submit --dependency="afterok:$gate_job" --mem=8G --time=00:30:00 \
  --export="$export_vars,STEP=relabel" "$SCRIPT_DIR/data_shift.sbatch")
prepare_job=$(submit --dependency="afterok:$select_job" --mem=180G --time=06:00:00 \
  --export="$export_vars,STEP=prepare,MODEL_REPO=$MODEL_REPO,FULL_PREPARED=$FULL_PREPARED" "$SCRIPT_DIR/data_shift.sbatch")
echo "SELECT_JOB=$select_job SUBSET_JOB=$subset_job GATE_JOB=$gate_job RELABEL_JOB=$relabel_job PREPARE_JOB=$prepare_job"
echo "data: $DATA"
