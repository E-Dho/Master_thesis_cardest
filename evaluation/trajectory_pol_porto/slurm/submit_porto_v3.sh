#!/bin/bash
# Porto v3 benchmarks, labelling as decided on 2026-11-03 (option c):
#   1. test workload (8 x 250 single + 8 x 750 multi) labelled in SQL;
#   2. the Porto snapshot export (independent, runs as soon as the database is free);
#   3. the label gate on the test set: every SQL label recounted on the snapshot
#      with NumpyBackend; it must pass before anything is labelled with numpy;
#   4. the validation workload (8 x 75 + 8 x 225) labelled on the snapshot;
#   5. the MSCN training workload (8 x 312 + 8 x 938) labelled on the snapshot,
#      excluding test and validation.
# Every database job runs under the job name porto_mobilitydb (singleton), so it
# never overlaps another Porto database job; POL jobs are unaffected.
#   STAGE=pilot ./submit_porto_v3.sh   snapshot export + a 10-per-category pilot on
#                                      the snapshot (acceptance and label times), no SQL labels
#   STAGE=all   ./submit_porto_v3.sh   steps 1-5 (default)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
V3_SLURM="$REPO_ROOT/query_generation/benchmark_v3/slurm"
STAGE=${STAGE:-all}

value_of() {  # value_of KEY <<< "$output": the job id printed as KEY=<id>
  local line
  line=$(grep -E "^$1=" || true)
  if [ -z "$line" ]; then echo "missing $1 in the submit output" >&2; exit 3; fi
  echo "${line#*=}" | awk '{print $1}'
}

run() { echo "+ $*" >&2; env "$@"; }

if [ "$STAGE" = "pilot" ]; then
  out=$(run REPO_ROOT="$REPO_ROOT" TRAJ_DATASET=porto_v3 STAGE=export "$SCRIPT_DIR/submit_snapshot.sh")
  echo "$out"
  export_job=$(value_of export_job <<< "$out")
  out=$(run REPO_ROOT="$REPO_ROOT" V3_DATASET=porto PILOT=1 LABEL_BACKEND=numpy AFTER_OK_JOB="$export_job" \
    "$V3_SLURM/submit_v3.sh")
  echo "$out"
  exit 0
fi
if [ "$STAGE" != "all" ]; then echo "STAGE must be pilot or all" >&2; exit 2; fi

# 1. test workload, SQL labels
out=$(run REPO_ROOT="$REPO_ROOT" V3_DATASET=porto "$V3_SLURM/submit_v3.sh"); echo "$out"
test_merge=$(value_of merge_job <<< "$out")
# 2. snapshot export (skipped with SKIP_EXPORT=1 when it exists already, e.g. after a pilot)
export_dependency=""
if [ "${SKIP_EXPORT:-0}" != "1" ]; then
  out=$(run REPO_ROOT="$REPO_ROOT" TRAJ_DATASET=porto_v3 STAGE=export "$SCRIPT_DIR/submit_snapshot.sh"); echo "$out"
  export_dependency=":$(value_of export_job <<< "$out")"
fi
# 3. label gate on the test set, after its merge and the export
out=$(run REPO_ROOT="$REPO_ROOT" TRAJ_DATASET=porto_v3 STAGE=gate WORKLOADS=test \
  GATE_AFTER_OK="$test_merge$export_dependency" "$SCRIPT_DIR/submit_snapshot.sh"); echo "$out"
gate_merge=$(value_of test_gate_merge <<< "$out")
# 4. validation on the snapshot, only if the gate passed (its merge exits 1 on any mismatch)
out=$(run REPO_ROOT="$REPO_ROOT" V3_DATASET=porto VALIDATION=1 LABEL_BACKEND=numpy AFTER_OK_JOB="$gate_merge" \
  "$V3_SLURM/submit_v3.sh"); echo "$out"
validation_merge=$(value_of merge_job <<< "$out")
# 5. training on the snapshot, after the validation set exists (it is excluded)
out=$(run REPO_ROOT="$REPO_ROOT" V3_DATASET=porto TRAINING=1 AFTER_OK_JOB="$validation_merge" \
  "$V3_SLURM/submit_v3.sh"); echo "$out"
echo "chain: test merge $test_merge -> gate merge $gate_merge -> validation merge $validation_merge -> training"
