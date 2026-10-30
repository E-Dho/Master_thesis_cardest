#!/bin/bash
# Submit the v3 pipeline: pools -> 16 category tasks -> merge -> verify.
#   PILOT=1 ./submit_v3.sh        10 queries per category
#   SKIP_POOLS=1 ./submit_v3.sh   resume generation after a deadline stop
#   STAGE=check ./submit_v3.sh    only the Phase 1 cluster check
#   STAGE=verify ./submit_v3.sh   only (re)run the verification of a merged run
#   VALIDATION=1 ./submit_v3.sh   validation workload (see v3_env.sh)
#   TRAINING=1 ./submit_v3.sh     MSCN training workload, labelled on the numpy
#                                 snapshot (LABEL_BACKEND=numpy: the 16 category
#                                 tasks run in parallel without the database)
#   AFTER_JOB=<id> ./submit_v3.sh database jobs also wait until job <id> has ended
#
# Every job that starts PostgreSQL is submitted under one shared job name with
# --dependency=singleton, so Slurm never runs two of them at the same time: they
# share one data directory, and each stops PostgreSQL when it finishes.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
export_vars="ALL,REPO_ROOT=$REPO_ROOT,PILOT=${PILOT:-0},VALIDATION=${VALIDATION:-0},TRAINING=${TRAINING:-0}"
for name in SEED RUN_LABEL SINGLE_COUNT MULTI_COUNT EXCLUDE_HASHES PER_CATEGORY LABEL_BACKEND SNAPSHOT_DIR; do
  if [ -n "${!name:-}" ]; then export_vars="$export_vars,$name=${!name}"; fi
done
if [ $(( ${PILOT:-0} + ${VALIDATION:-0} + ${TRAINING:-0} )) -gt 1 ]; then
  echo "PILOT, VALIDATION and TRAINING are mutually exclusive" >&2
  exit 2
fi
label_backend=${LABEL_BACKEND:-sql}
if [ "${TRAINING:-0}" = "1" ]; then label_backend=${LABEL_BACKEND:-numpy}; fi
DB_JOB_NAME=${DB_JOB_NAME:-pol_mobilitydb}

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

# db_submit [extra dependency] sbatch-args...: database job, one at a time.
# AFTER_JOB covers a database job submitted under another name (e.g. an older
# qv3_verify), which singleton does not see.
db_submit() {
  local deps="singleton"
  if [ -n "$1" ]; then deps="$1,singleton"; fi
  shift
  if [ -n "${AFTER_JOB:-}" ]; then deps="afterany:$AFTER_JOB,$deps"; fi
  submit --job-name="$DB_JOB_NAME" --dependency="$deps" "$@"
}

if [ "${STAGE:-all}" = "check" ]; then
  echo "cluster_check_job=$(db_submit "" --export="$export_vars" "$SCRIPT_DIR/run_v3_cluster_check.sbatch")"
  exit 0
fi

if [ "${STAGE:-all}" = "verify" ]; then
  echo "verify_job=$(db_submit "" --export="$export_vars" "$SCRIPT_DIR/run_v3_verify.sbatch")"
  exit 0
fi

pools_dependency=""
if [ "${SKIP_POOLS:-0}" != "1" ]; then
  pools=$(db_submit "" --export="$export_vars" "$SCRIPT_DIR/run_v3_pools.sbatch")
  echo "pools_job=$pools"
  pools_dependency="afterok:$pools"
fi
if [ "$label_backend" = "numpy" ]; then
  # no database: all 16 categories at once, after the pools job
  generate_dependency=()
  if [ -n "$pools_dependency" ]; then generate_dependency=(--dependency="$pools_dependency"); fi
  generate=$(submit "${generate_dependency[@]}" --array=0-15 --export="$export_vars" \
    "$SCRIPT_DIR/run_v3_generate_numpy.sbatch")
else
  generate=$(db_submit "$pools_dependency" --array=0-15%1 --export="$export_vars" "$SCRIPT_DIR/run_v3_generate.sbatch")
fi
merge=$(submit --dependency="afterok:$generate" --export="$export_vars" "$SCRIPT_DIR/run_v3_merge.sbatch")
verify=$(db_submit "afterok:$merge" --export="$export_vars" "$SCRIPT_DIR/run_v3_verify.sbatch")
echo "generate_array_job=$generate"
echo "merge_job=$merge"
echo "verify_job=$verify"
echo "database jobs run under the job name $DB_JOB_NAME, one at a time (singleton)"
