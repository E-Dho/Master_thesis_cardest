#!/bin/bash
# P6: TraHist, TraHist-Temporal, CASE+ and CASE+-Temporal on POL v3. No database.
#   STAGE=tune ./submit_p6.sh   sweeps on the validation workload (1,800 multi queries):
#                               base: unit_size_m; temporal: unit_size_m x time_buckets
#   STAGE=run  ./submit_p6.sh   test runs with the committed configs, then aggregate and
#                               compare with the earlier POL methods
#   CONFIG_NAMES="trahist_temporal" ...   a subset
#   UNIT_SIZES / TIME_BUCKETS             override the sweep grids (values separated by /)
#   AFTER_JOB=<id>                        start after that job succeeded
# The cpu_1core timing follows with submit_timing.sh
# (CPU_CONFIGS="trahist trahist_temporal case_plus case_plus_temporal").
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
STAGE=${STAGE:?set STAGE=tune or STAGE=run}
CONFIG_NAMES=${CONFIG_NAMES:-"trahist trahist_temporal case_plus case_plus_temporal"}
UNIT_SIZES=${UNIT_SIZES:-"25/50/100/200/400"}
TIME_BUCKETS=${TIME_BUCKETS:-"25/50/100/200/400"}
VALIDATION_JSONL=${VALIDATION_JSONL:-/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/query_runs/qgen_pol_v3_validation_seed20261026/benchmark.jsonl}
VALIDATION_MULTI=${VALIDATION_MULTI:-1800}
TUNING_ROOT=${TUNING_ROOT:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3_tuning}
COMPARE_WITH=${COMPARE_WITH-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000 mobilitydb_v3_sql mobilitydb_segment_stbox mobilitydb_trip_stbox deepdb_mbr mscn"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/p6_comparison}
base_export="ALL,REPO_ROOT=$REPO_ROOT"
if [ -n "${PYTHON:-}" ]; then base_export="$base_export,PYTHON=$PYTHON"; fi

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
dependency_args=()
if [ -n "${AFTER_JOB:-}" ]; then dependency_args=(--dependency="afterok:$AFTER_JOB"); fi

case "$STAGE" in
  tune)
    for name in $CONFIG_NAMES; do
      config="$CONFIG_DIR/$name.yaml"
      if [ ! -f "$config" ]; then echo "missing config $config" >&2; exit 2; fi
      grids="unit_size_m=$UNIT_SIZES"
      if grep -qE '^  temporal: true' "$config"; then grids="$grids+time_buckets=$TIME_BUCKETS"; fi
      job=$(submit "${dependency_args[@]}" \
        --export="$base_export,TUNE_CONFIG=$config,VALIDATION_JSONL=$VALIDATION_JSONL,EXPECTED_QUERIES=$VALIDATION_MULTI,GRIDS=$grids,OUTPUT=$TUNING_ROOT/$name" \
        "$SCRIPT_DIR/tune.sbatch")
      echo "$name: tuning job $job ($grids) -> $TUNING_ROOT/$name"
    done
    ;;
  run)
    jobs=()
    configs=()
    for name in $CONFIG_NAMES; do
      config="$CONFIG_DIR/$name.yaml"
      if [ ! -f "$config" ]; then echo "missing config $config" >&2; exit 2; fi
      job=$(submit "${dependency_args[@]}" --array=0 --export="$base_export,CONFIG=$config" \
        "$SCRIPT_DIR/run_inprocess.sbatch")
      echo "$name: job $job (seed 0; deterministic)"
      jobs+=("$job")
      configs+=("$config")
    done
    for name in $COMPARE_WITH; do configs+=("$CONFIG_DIR/$name.yaml"); done
    aggregate=$(submit --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
      --export="$base_export,CONFIGS=$(IFS=:; echo "${configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
      "$SCRIPT_DIR/aggregate_compare.sbatch")
    echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
    ;;
  *) echo "STAGE must be tune or run" >&2; exit 2 ;;
esac
