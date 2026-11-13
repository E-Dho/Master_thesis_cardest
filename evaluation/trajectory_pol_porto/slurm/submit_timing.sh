#!/bin/bash
# Profile-controlled latency of the POL v3 trajectory methods (TRAJ_DATASET=porto_v3:
# Porto v3; the planner timing then starts the Porto server), then a fresh
# aggregate and comparison that includes it.
#   ./submit_timing.sh                         both profiles (P3 on cpu_1core, P4 on cpu_postgres_2core)
#   PROFILES=cpu_1core ./submit_timing.sh      one profile
#   CPU_CONFIGS="exact_mbr traj_sampling_n1000" ./submit_timing.sh
#   PROFILES="cpu_1core gpu_single_query" CPU_CONFIGS=own_model GPU_CONFIGS=own_model ./submit_timing.sh
#                                              neural methods: GPU latency in its own table (one L40S)
#   GPU_AFTER_JOB=<id>[:<id>...] ./submit_timing.sh   gpu_single_query waits for these jobs (afterok)
#   AFTER_JOB=<id> ./submit_timing.sh          the planner timing starts after job <id> succeeded
#                                              (the P4 accuracy job)
#   CPU_AFTER_JOB=<id>[:<id>...] ./submit_timing.sh   cpu_1core waits for these jobs (afterok)
#   TIMING_SBATCH_ARGS="--nodelist=n152" ./submit_timing.sh   extra sbatch arguments for both
#                                              timing jobs (the hardware class is AMD EPYC 7313)
#   COMPARE_WITH="exact_mbr mobilitydb_v3_sql" ./submit_timing.sh   further configs (already timed)
#                                              in the final comparison
# The plans are written when each job starts (latest complete accuracy run per
# config and seed), so the jobs may be submitted before those runs exist.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
source "$SCRIPT_DIR/_submit_dataset.sh"
PROFILES=${PROFILES:-"cpu_1core cpu_postgres_2core"}
CPU_CONFIGS=${CPU_CONFIGS:-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000"}
PG_CONFIGS=${PG_CONFIGS:-"mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox"}
GPU_CONFIGS=${GPU_CONFIGS:-own_model}
COMPARISON_DIR=${COMPARISON_DIR:-$RESULTS_DIR/timed_comparison}
DB_JOB_NAME=${DB_JOB_NAME:-$DATASET_DB_JOB_NAME}
COMPARE_WITH=${COMPARE_WITH:-}
read -r -a extra_sbatch <<< "${TIMING_SBATCH_ARGS:-}"

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
      cpu_after=()
      if [ -n "${CPU_AFTER_JOB:-}" ]; then cpu_after=(--dependency="afterok:$CPU_AFTER_JOB"); fi
      job=$(submit ${extra_sbatch[@]+"${extra_sbatch[@]}"} ${cpu_after[@]+"${cpu_after[@]}"} --export="$base_export,CONFIGS=$configs" "$SCRIPT_DIR/timing_cpu_1core.sbatch")
      ;;
    cpu_postgres_2core)
      configs=$(paths_of "$PG_CONFIGS")
      deps="singleton"
      if [ -n "${AFTER_JOB:-}" ]; then deps="afterok:$AFTER_JOB,singleton"; fi
      job=$(submit ${extra_sbatch[@]+"${extra_sbatch[@]}"} --job-name="$DB_JOB_NAME" --dependency="$deps" --export="$base_export,CONFIGS=$configs" \
        "$SCRIPT_DIR/timing_cpu_postgres_2core.sbatch")
      ;;
    gpu_single_query)
      configs=$(paths_of "$GPU_CONFIGS")
      gpu_after=()
      if [ -n "${GPU_AFTER_JOB:-}" ]; then gpu_after=(--dependency="afterok:$GPU_AFTER_JOB"); fi
      # the GPU job selects its node class itself (--constraint=L40); TIMING_SBATCH_ARGS is for the CPU jobs
      job=$(submit ${gpu_after[@]+"${gpu_after[@]}"} --export="$base_export,CONFIGS=$configs" "$SCRIPT_DIR/timing_gpu_single_query.sbatch")
      ;;
    *) echo "unknown profile $profile (cpu_1core, cpu_postgres_2core or gpu_single_query)" >&2; exit 2 ;;
  esac
  echo "$profile: job $job"
  jobs+=("$job")
  all_configs+=("$configs")
done
if [ -n "$COMPARE_WITH" ]; then all_configs+=("$(paths_of "$COMPARE_WITH")"); fi
# a config timed under several profiles (own model: cpu_1core and gpu_single_query) is aggregated once
compare_configs=$(IFS=:; echo "${all_configs[*]}" | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd: -)
aggregate=$(submit --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
  --export="$base_export,CONFIGS=$compare_configs,COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
