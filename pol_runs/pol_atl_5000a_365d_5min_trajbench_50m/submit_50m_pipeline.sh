#!/usr/bin/env bash
set -euo pipefail

THESIS_HOME=/zfshome/sunip956/master_thesis_trajectories
THESIS_WORK=/work_beegfs/sunip956/master_thesis_trajectories
RUN_ROOT="$THESIS_WORK/runs/pol_atl_5000a_365d_5min_trajbench_50m"
HOME_RUN_ROOT="$THESIS_HOME/pol_runs/pol_atl_5000a_365d_5min_trajbench_50m"
DATASET_LIST="$RUN_ROOT/datasets.txt"
LOAD_SCRIPT="$THESIS_HOME/mobilitydb_loader/load_pol_multi_to_mobilitydb.sbatch"

mkdir -p "$RUN_ROOT" "$THESIS_WORK/slurm_logs"
: > "$DATASET_LIST"
for seed in 1 2 3 4 5 6 7; do
  echo "$THESIS_WORK/datasets/pol_atl_5000a_365d_5min_seed${seed}_trajbench_50m" >> "$DATASET_LIST"
done

smoke_job=$(sbatch --parsable "$HOME_RUN_ROOT/run_smoke.sbatch")
seed1_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" "$HOME_RUN_ROOT/run_seed1.sbatch")
array_job=$(sbatch --parsable --dependency=afterok:"$seed1_job" "$HOME_RUN_ROOT/run_seeds_2_7_array.sbatch")
load_job=$(sbatch --parsable --dependency=afterok:"$array_job" --export=ALL,POL_DATASET_LIST="$DATASET_LIST" "$LOAD_SCRIPT")

cat > "$RUN_ROOT/submission_metadata.txt" <<META
submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
smoke_job=$smoke_job
seed1_job=$seed1_job
seeds_2_7_array_job=$array_job
load_job=$load_job
dataset_list=$DATASET_LIST
META

cat "$RUN_ROOT/submission_metadata.txt"
