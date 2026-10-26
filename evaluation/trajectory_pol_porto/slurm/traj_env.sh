#!/bin/bash
# Shared settings of the trajectory evaluation jobs. Needs REPO_ROOT.
#   TRAJ_DATASET=pol_v3        snapshot name (directory under SNAPSHOT_ROOT)
#   V3_CONFIG=...              v3 benchmark config the snapshot is exported for
#   SNAPSHOT_DIR=...           override the snapshot directory
#   TEST_BENCHMARK=...         benchmark.jsonl of the test workload
#   VALIDATION_BENCHMARK=...   benchmark.jsonl of the validation workload
source "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh"

TRAJ_DATASET=${TRAJ_DATASET:-pol_v3}
V3_CONFIG=${V3_CONFIG:-$REPO_ROOT/query_generation/benchmark_v3/pol_v3_config.json}
SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-$THESIS_WORK_ROOT/traj_snapshots}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-$SNAPSHOT_ROOT/$TRAJ_DATASET}
TEST_BENCHMARK=${TEST_BENCHMARK:-$MOBILITYDB_ROOT/query_runs/qgen_pol_v3_exact_st_seed20261021/benchmark.jsonl}
VALIDATION_BENCHMARK=${VALIDATION_BENCHMARK:-$MOBILITYDB_ROOT/query_runs/qgen_pol_v3_validation_seed20261026/benchmark.jsonl}
DB_ARGS=(--host "$MOBILITYDB_SOCKET_DIR" --port "$MOBILITYDB_PORT" --dbname "$MOBILITYDB_DB")
if [ -n "${MOBILITYDB_USER:-}" ]; then
  DB_ARGS+=(--user "$MOBILITYDB_USER")
fi

traj_resolve_repo_root() {
  REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
  if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/evaluation/trajectory_pol_porto/traj_eval/snapshot.py" ]; then
    echo "Set REPO_ROOT to a checkout of feature/trajectory-evaluation" >&2
    exit 2
  fi
}

traj_benchmark_for() {
  case "$1" in
    test) echo "$TEST_BENCHMARK" ;;
    validation) echo "$VALIDATION_BENCHMARK" ;;
    *) echo "unknown workload $1 (test or validation)" >&2; return 2 ;;
  esac
}
