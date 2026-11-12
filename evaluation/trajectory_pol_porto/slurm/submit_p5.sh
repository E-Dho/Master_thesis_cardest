#!/bin/bash
# P5: DeepDB-MBR on POL v3. Shared preparation (trip-table export from the P2
# snapshot + DeepDB HDF), then one array task per seed, then aggregate and
# compare with the earlier methods. No database.
#   ./submit_p5.sh
#   SKIP_SHARED=1 ./submit_p5.sh        the shared export and HDF already exist
#   COMPARE_WITH="..." ./submit_p5.sh   configs in the comparison (default: P3 + P4)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
CONFIG="$CONFIG_DIR/deepdb_mbr.yaml"
COMPARE_WITH=${COMPARE_WITH-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000 mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/p5_comparison}
base_export="ALL,REPO_ROOT=$REPO_ROOT"

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

dependency=()
if [ "${SKIP_SHARED:-0}" != "1" ]; then
  shared=$(submit --export="$base_export" "$SCRIPT_DIR/deepdb_mbr_prepare_shared.sbatch")
  echo "shared preparation: job $shared"
  dependency=(--dependency="afterok:$shared")
fi
seeds=$(grep -E '^  seeds:' "$CONFIG" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ')
runs=$(submit ${dependency[@]+"${dependency[@]}"} --array="$seeds" --export="$base_export,CONFIG=$CONFIG" "$SCRIPT_DIR/run_external.sbatch")
echo "deepdb_mbr: job $runs (seeds $seeds)"
configs=("$CONFIG")
for name in $COMPARE_WITH; do configs+=("$CONFIG_DIR/$name.yaml"); done
aggregate=$(submit --dependency="afterok:$runs" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
