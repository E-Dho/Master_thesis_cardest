#!/bin/bash
# Profile-controlled latency of the POL v3 trajectory methods, then a fresh
# aggregate and comparison that includes it.
#   ./submit_timing.sh                         both profiles (P3 on cpu_1core, P4 on cpu_postgres_2core)
#   PROFILES=cpu_1core ./submit_timing.sh      one profile
#   CPU_CONFIGS="exact_mbr traj_sampling_n1000" ./submit_timing.sh
#   AFTER_JOB=<id> ./submit_timing.sh          the planner timing starts after job <id> succeeded
#                                              (the P4 accuracy job); cpu_1core needs no wait
# The plans are written when each job starts (latest complete accuracy run per
# config and seed), so the jobs may be submitted before those runs exist.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
PROFILES=${PROFILES:-"cpu_1core cpu_postgres_2core"}
CPU_CONFIGS=${CPU_CONFIGS:-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000"}
PG_CONFIGS=${PG_CONFIGS:-"mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/timed_comparison}
DB_JOB_NAME=${DB_JOB_NAME:-pol_mobilitydb}
base_export="ALL,REPO_ROOT=$REPO_ROOT"

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
paths_of() {
  local list=() name
  for name in $1; do
    if [ ! -f "$CONFIG_DIR/$name.yaml" ]; then echo "missing config $CONFIG_DIR/$name.yaml" >&2; exit 2; fi
    list+=("$CONFIG_DIR/$name.yaml")
  done
  (IFS=:; echo "${list[*]}")
}

jobs=()
all_configs=()
for profile in $PROFILES; do
  case "$profile" in
    cpu_1core)
      configs=$(paths_of "$CPU_CONFIGS")
      job=$(submit --export="$base_export,CONFIGS=$configs" "$SCRIPT_DIR/timing_cpu_1core.sbatch")
      ;;
    cpu_postgres_2core)
      configs=$(paths_of "$PG_CONFIGS")
      deps="singleton"
      if [ -n "${AFTER_JOB:-}" ]; then deps="afterok:$AFTER_JOB,singleton"; fi
      job=$(submit --job-name="$DB_JOB_NAME" --dependency="$deps" --export="$base_export,CONFIGS=$configs" \
        "$SCRIPT_DIR/timing_cpu_postgres_2core.sbatch")
      ;;
    *) echo "unknown profile $profile (cpu_1core or cpu_postgres_2core)" >&2; exit 2 ;;
  esac
  echo "$profile: job $job"
  jobs+=("$job")
  all_configs+=("$configs")
done
aggregate=$(submit --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${all_configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
