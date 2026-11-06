#!/bin/bash
# Segment-MBR overcount statistic on the snapshot (no database): a checksum
# verification of the snapshot, then the shard array (after the verification
# succeeded), then the merge. The source files must be committed (the shards
# refuse uncommitted changes).
#   ./submit_mbr_overcount.sh                       test workload, 16 shards
#   WORKLOAD=validation ./submit_mbr_overcount.sh   the validation workload
#   OVERCOUNT_SHARDS=8 ./submit_mbr_overcount.sh    other shard count (new results only)
#   TRAJ_DATASET=porto_v3 ./submit_mbr_overcount.sh the Porto snapshot and benchmark
#   OVERCOUNT_ROOT=...                              other results root
# Output (one directory per statistic version, so older results never block a run):
#   $SNAPSHOT_DIR/analysis/mbr_overcount/v<version>/<workload>_report.{json,md},
#   <workload>_report_per_query.csv, <workload>_report_overcounted.csv; raw shard lines in <workload>/.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
source "$SCRIPT_DIR/traj_env.sh"
traj_resolve_repo_root
WORKLOAD=${WORKLOAD:-test}
SHARDS=${OVERCOUNT_SHARDS:-16}
benchmark=$(traj_benchmark_for "$WORKLOAD")
if [ ! -f "$benchmark" ] || [ ! -f "$SNAPSHOT_DIR/manifest.json" ]; then
  echo "missing $benchmark or $SNAPSHOT_DIR/manifest.json" >&2
  exit 2
fi
version=$(sed -n 's/^STATISTIC_VERSION = \([0-9][0-9]*\).*/\1/p' "$REPO_ROOT/evaluation/trajectory_pol_porto/traj_eval/mbr_overcount.py")
if [ -z "$version" ]; then echo "cannot read STATISTIC_VERSION" >&2; exit 2; fi
OVERCOUNT_ROOT=${OVERCOUNT_ROOT:-$SNAPSHOT_DIR/analysis/mbr_overcount/v$version}
export_vars="ALL,OVERCOUNT_ROOT=$OVERCOUNT_ROOT,REPO_ROOT=$REPO_ROOT,TRAJ_DATASET=$TRAJ_DATASET,SNAPSHOT_DIR=$SNAPSHOT_DIR,V3_CONFIG=$V3_CONFIG,TEST_BENCHMARK=$TEST_BENCHMARK,VALIDATION_BENCHMARK=$VALIDATION_BENCHMARK,WORKLOAD=$WORKLOAD"
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
verify=$(submit --export="$export_vars" "$SCRIPT_DIR/mbr_overcount_verify.sbatch")
array=$(submit --dependency="afterok:$verify" --array=0-$((SHARDS - 1)) --export="$export_vars,OVERCOUNT_SHARDS=$SHARDS" "$SCRIPT_DIR/mbr_overcount.sbatch")
merge=$(submit --dependency="afterany:$array" --export="$export_vars" "$SCRIPT_DIR/mbr_overcount_merge.sbatch")
echo "overcount_verify=$verify"
echo "overcount_array=$array ($SHARDS shards)"
echo "overcount_merge=$merge -> $OVERCOUNT_ROOT/${WORKLOAD}_report.md"
