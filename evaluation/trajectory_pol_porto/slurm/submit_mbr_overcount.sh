#!/bin/bash
# Segment-MBR overcount statistic on the snapshot (no database).
#   ./submit_mbr_overcount.sh                       test workload, 16 shards
#   WORKLOAD=validation ./submit_mbr_overcount.sh   the validation workload
#   OVERCOUNT_SHARDS=8 ./submit_mbr_overcount.sh    other shard count (new results only)
#   TRAJ_DATASET=porto_v3 ./submit_mbr_overcount.sh the Porto snapshot and benchmark
# Output: $SNAPSHOT_DIR/analysis/mbr_overcount/<workload>_report.{json,md},
#         <workload>_report_per_query.csv; raw shard lines in <workload>/.
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
export_vars="ALL,REPO_ROOT=$REPO_ROOT,TRAJ_DATASET=$TRAJ_DATASET,SNAPSHOT_DIR=$SNAPSHOT_DIR,V3_CONFIG=$V3_CONFIG,TEST_BENCHMARK=$TEST_BENCHMARK,VALIDATION_BENCHMARK=$VALIDATION_BENCHMARK,WORKLOAD=$WORKLOAD"
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
array=$(submit --array=0-$((SHARDS - 1)) --export="$export_vars,OVERCOUNT_SHARDS=$SHARDS" "$SCRIPT_DIR/mbr_overcount.sbatch")
merge=$(submit --dependency="afterany:$array" --export="$export_vars" "$SCRIPT_DIR/mbr_overcount_merge.sbatch")
echo "overcount_array=$array ($SHARDS shards)"
echo "overcount_merge=$merge -> $SNAPSHOT_DIR/analysis/mbr_overcount/${WORKLOAD}_report.md"
