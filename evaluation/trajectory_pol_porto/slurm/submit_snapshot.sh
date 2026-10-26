#!/bin/bash
# P2: export the shared snapshot, then recount every label on it.
#   ./submit_snapshot.sh                      export + gate for test and validation
#   STAGE=gate ./submit_snapshot.sh           gate only (snapshot exists)
#   WORKLOADS=test ./submit_snapshot.sh       only the test workload's gate
#   AFTER_JOB=<id> ./submit_snapshot.sh       the export also waits for job <id>
#   FORCE=1 ./submit_snapshot.sh              replace an existing snapshot
#
# The export starts PostgreSQL, so it is submitted under the shared database job
# name (DB_JOB_NAME, default pol_mobilitydb) with --dependency=singleton, like
# every v3 database job: Slurm runs it only when no other job of that name runs.
# The gate needs no database; the test gate uses 16 shards, validation 8.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
source "$SCRIPT_DIR/traj_env.sh"
traj_resolve_repo_root
STAGE=${STAGE:-all}
WORKLOADS=${WORKLOADS:-"test validation"}
DB_JOB_NAME=${DB_JOB_NAME:-pol_mobilitydb}
export_vars="ALL,REPO_ROOT=$REPO_ROOT,SNAPSHOT_DIR=$SNAPSHOT_DIR,V3_CONFIG=$V3_CONFIG,TEST_BENCHMARK=$TEST_BENCHMARK,VALIDATION_BENCHMARK=$VALIDATION_BENCHMARK,FORCE=${FORCE:-0}"

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

gate_dependency=""
if [ "$STAGE" = "all" ] || [ "$STAGE" = "export" ]; then
  deps="singleton"
  if [ -n "${AFTER_JOB:-}" ]; then deps="afterany:$AFTER_JOB,$deps"; fi
  export_job=$(submit --job-name="$DB_JOB_NAME" --dependency="$deps" --export="$export_vars" "$SCRIPT_DIR/snapshot_export.sbatch")
  echo "export_job=$export_job (job name $DB_JOB_NAME, singleton)"
  gate_dependency="afterok:$export_job"
fi
if [ "$STAGE" = "export" ]; then exit 0; fi

for workload in $WORKLOADS; do
  case "$workload" in
    test) shards=${TEST_SHARDS:-16} ;;
    validation) shards=${VALIDATION_SHARDS:-8} ;;
    *) echo "unknown workload $workload" >&2; exit 2 ;;
  esac
  benchmark=$(traj_benchmark_for "$workload")
  if [ ! -f "$benchmark" ]; then
    echo "warning: $benchmark does not exist yet; its gate fails unless it exists when the gate starts" >&2
  fi
  dependency_args=()
  if [ -n "$gate_dependency" ]; then dependency_args=(--dependency="$gate_dependency"); fi
  gate=$(submit "${dependency_args[@]}" --array=0-$((shards - 1)) \
    --export="$export_vars,WORKLOAD=$workload,GATE_SHARDS=$shards" "$SCRIPT_DIR/snapshot_gate.sbatch")
  merge=$(submit --dependency="afterany:$gate" --export="$export_vars,WORKLOAD=$workload" \
    "$SCRIPT_DIR/snapshot_gate_merge.sbatch")
  echo "${workload}_gate_array=$gate (${shards} shards)"
  echo "${workload}_gate_merge=$merge -> $SNAPSHOT_DIR/gate/${workload}_report.json"
done
