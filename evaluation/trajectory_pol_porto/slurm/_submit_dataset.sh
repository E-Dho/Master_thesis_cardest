#!/bin/bash
# Dataset of the submit_p3/p4/p5/p6/p8b/timing scripts (sourced after REPO_ROOT;
# no database access on the login node). submit_p7.sh uses traj_env.sh directly.
#   TRAJ_DATASET=pol_v3   (default) configs/pol_v3, results/traj_phase1_pol_v3
#   TRAJ_DATASET=porto_v3 configs/porto_v3, results/traj_phase1_porto_v3, the Porto
#                         database job name and validation workload
# TRAJ_DATASET is passed to every job (base_export), so the jobs' traj_env.sh
# selects the same dataset (snapshot, database, benchmarks).
TRAJ_DATASET=${TRAJ_DATASET:-pol_v3}
_traj_runs=/work_beegfs/sunip956/master_thesis_trajectories
case "$TRAJ_DATASET" in
  pol_v3)
    DATASET_DB_JOB_NAME=pol_mobilitydb
    DATASET_VALIDATION_JSONL=$_traj_runs/mobilitydb/query_runs/qgen_pol_v3_validation_seed20261026/benchmark.jsonl
    ;;
  porto_v3)
    DATASET_DB_JOB_NAME=porto_mobilitydb
    DATASET_VALIDATION_JSONL=$_traj_runs/mobilitydb_porto/query_runs/qgen_porto_v3_validation_seed20261103/benchmark.jsonl
    ;;
  *) echo "TRAJ_DATASET must be pol_v3 or porto_v3, got $TRAJ_DATASET" >&2; exit 2 ;;
esac
unset _traj_runs
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/$TRAJ_DATASET"
RESULTS_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_$TRAJ_DATASET"
base_export="ALL,REPO_ROOT=$REPO_ROOT,TRAJ_DATASET=$TRAJ_DATASET"
