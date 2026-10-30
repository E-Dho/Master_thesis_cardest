#!/bin/bash
# P4: the MobilityDB planner on POL v3 (variants v3_sql, segment_stbox,
# trip_stbox), then one comparison of P3 and P4. Needs the loaded MobilityDB
# database, not the snapshot.
#   ./submit_p4.sh                                   all three variants
#   CONFIG_NAMES="mobilitydb_v3_sql" ./submit_p4.sh  one variant
#   AFTER_JOB=<id> ./submit_p4.sh                    also wait for job <id>
# The database job runs under the shared job name (DB_JOB_NAME, default
# pol_mobilitydb) with --dependency=singleton, like every v3 database job.
# COMPARE_WITH lists further configs for the comparison (default: the P3 ones;
# their runs must already exist).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
CONFIG_NAMES=${CONFIG_NAMES:-"mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox"}
COMPARE_WITH=${COMPARE_WITH-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/p4_comparison}
DB_JOB_NAME=${DB_JOB_NAME:-pol_mobilitydb}
base_export="ALL,REPO_ROOT=$REPO_ROOT"

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
config_path() {
  local path="$CONFIG_DIR/$1.yaml"
  if [ ! -f "$path" ]; then echo "missing config $path" >&2; exit 2; fi
  echo "$path"
}

p4=()
for name in $CONFIG_NAMES; do p4+=("$(config_path "$name")"); done
all=("${p4[@]}")
for name in $COMPARE_WITH; do all+=("$(config_path "$name")"); done

deps="singleton"
if [ -n "${AFTER_JOB:-}" ]; then deps="afterany:$AFTER_JOB,$deps"; fi
db_job=$(submit --job-name="$DB_JOB_NAME" --dependency="$deps" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${p4[*]}")" "$SCRIPT_DIR/run_mobilitydb.sbatch")
echo "mobilitydb: job $db_job (job name $DB_JOB_NAME, singleton): $CONFIG_NAMES"
aggregate=$(submit --dependency="afterok:$db_job" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${all[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
