#!/bin/bash
# P8b: MSCN on POL v3 (in process, geo-mlp/torch, CPU training with 8 threads).
# Needs the P2 snapshot (sample bitmaps) and the POL training workload.
#   ./submit_p8b.sh                              the main variant (single + multi)
#   CONFIG_NAMES="mscn mscn_multi_only" ./submit_p8b.sh   also the fallback variant
#   COMPARE_WITH="..." ./submit_p8b.sh           configs in the comparison (default: P3 + P4)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
CONFIG_NAMES=${CONFIG_NAMES:-mscn}
COMPARE_WITH=${COMPARE_WITH-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000 mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/p8b_comparison}
base_export="ALL,REPO_ROOT=$REPO_ROOT"

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

jobs=()
configs=()
for name in $CONFIG_NAMES; do
  config="$CONFIG_DIR/$name.yaml"
  if [ ! -f "$config" ]; then echo "missing config $config" >&2; exit 2; fi
  seeds=$(grep -E '^  seeds:' "$config" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ')
  job=$(submit --cpus-per-task=8 --job-name=traj_mscn --array="$seeds" --export="$base_export,CONFIG=$config" \
    "$SCRIPT_DIR/run_inprocess.sbatch")
  echo "$name: job $job (seeds $seeds)"
  jobs+=("$job")
  configs+=("$config")
done
for name in $COMPARE_WITH; do configs+=("$CONFIG_DIR/$name.yaml"); done
aggregate=$(submit --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
