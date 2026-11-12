#!/bin/bash
# P7: the own model (ResMADE + traj_dedup_factor) on POL v3. Checkpoints and their
# training configs are read from P7_MODELS/seed_<s>/{checkpoint_best.pt,training_config.yaml}.
# The configs' bridge commands name that directory themselves (it is part of the
# recorded command and the accuracy config hash), so STAGE=run refuses a P7_MODELS
# that differs from the directory in the configs: edit the configs instead.
#   STAGE=check ./submit_p7.sh   code check: this repository's model code vs the pinned training
#                                state (MODEL_ROOT) on CHECK_QUERIES test queries per relation, seed 0
#   STAGE=truth ./submit_p7.sh   true matching-segment counts of the test multi queries (verify,
#                                SEGTRUTH_SHARDS shards, merge) for the error split
#   STAGE=run   ./submit_p7.sh   accuracy runs (seeds 0-2) of CONFIG_NAMES, then aggregate and compare
#                                with COMPARE_WITH; timing follows with
#                                PROFILES="cpu_1core gpu_single_query" CPU_CONFIGS=own_model ./submit_timing.sh
#   AFTER_JOB=<id>               the first job of the stage waits for it (afterok)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
source "$SCRIPT_DIR/traj_env.sh"
traj_resolve_repo_root
STAGE=${STAGE:?set STAGE=check, truth or run}
P7_MODELS=${P7_MODELS:-$THESIS_WORK_ROOT/p7_models/$TRAJ_DATASET}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/$TRAJ_DATASET"
CONFIG_NAMES=${CONFIG_NAMES:-"own_model own_model_no_dedup"}
COMPARE_WITH=${COMPARE_WITH-}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_$TRAJ_DATASET/p7_comparison}
WORKLOAD=${WORKLOAD:-test}
base_export="ALL,REPO_ROOT=$REPO_ROOT,TRAJ_DATASET=$TRAJ_DATASET"
submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
after=()
if [ -n "${AFTER_JOB:-}" ]; then after=(--dependency="afterok:$AFTER_JOB"); fi

case "$STAGE" in
  check)
    MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the worktree of model/pol-segments-first-training-state}
    CHECK_DIR=${CHECK_DIR:-$THESIS_WORK_ROOT/p7_checks/${TRAJ_DATASET}_code_check_$(date -u +%Y%m%dT%H%M%SZ)}
    job=$(submit ${after[@]+"${after[@]}"} --export="$base_export,MODEL_ROOT=$MODEL_ROOT,MODELS=$P7_MODELS/seed_0,CHECK_DIR=$CHECK_DIR,CHECK_QUERIES=${CHECK_QUERIES:-500}" \
      "$SCRIPT_DIR/own_model_check.sbatch")
    echo "code check: job $job -> $CHECK_DIR/check.json"
    ;;
  truth)
    version=$(sed -n 's/^STATISTIC_VERSION = \([0-9][0-9]*\).*/\1/p' "$REPO_ROOT/evaluation/trajectory_pol_porto/traj_eval/segment_truth.py")
    SEGTRUTH_ROOT=${SEGTRUTH_ROOT:-$SNAPSHOT_DIR/analysis/segment_truth/v$version}
    SHARDS=${SEGTRUTH_SHARDS:-16}
    export_vars="$base_export,SEGTRUTH_ROOT=$SEGTRUTH_ROOT,WORKLOAD=$WORKLOAD,SNAPSHOT_DIR=$SNAPSHOT_DIR,V3_CONFIG=$V3_CONFIG,TEST_BENCHMARK=$TEST_BENCHMARK,VALIDATION_BENCHMARK=$VALIDATION_BENCHMARK"
    verify=$(submit ${after[@]+"${after[@]}"} --mem=4G --time=01:00:00 --export="$export_vars,SEGTRUTH_STEP=verify" "$SCRIPT_DIR/segment_truth.sbatch")
    array=$(submit --dependency="afterok:$verify" --array=0-$((SHARDS - 1)) \
      --export="$export_vars,SEGTRUTH_STEP=run,SEGTRUTH_SHARDS=$SHARDS" "$SCRIPT_DIR/segment_truth.sbatch")
    merge=$(submit --dependency="afterany:$array" --mem=8G --time=00:30:00 --export="$export_vars,SEGTRUTH_STEP=merge" \
      "$SCRIPT_DIR/segment_truth.sbatch")
    echo "segment_truth_verify=$verify"
    echo "segment_truth_array=$array ($SHARDS shards)"
    echo "segment_truth_merge=$merge -> $SEGTRUTH_ROOT/${WORKLOAD}_segment_truth.json"
    ;;
  run)
    jobs=()
    configs=()
    for name in $CONFIG_NAMES; do
      config="$CONFIG_DIR/$name.yaml"
      if [ ! -f "$config" ]; then echo "missing config $config" >&2; exit 2; fi
      used=$(grep -oE -- '--(checkpoint|training-config) [^ ]+' "$config" | sed -E 's/^--[a-z-]+ //; s#/[^/]+$##' | sort -u)
      if [ "$used" != "$P7_MODELS/seed_{seed}" ]; then
        echo "$config reads its checkpoints from '$used', not P7_MODELS=$P7_MODELS/seed_{seed}; edit the config" >&2
        exit 2
      fi
      seeds=$(grep -E '^  seeds:' "$config" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ')
      job=$(submit ${after[@]+"${after[@]}"} --job-name=traj_own_model --array="$seeds" --export="$base_export,CONFIG=$config" \
        "$SCRIPT_DIR/run_external.sbatch")
      echo "$name: job $job (seeds $seeds)"
      jobs+=("$job")
      configs+=("$config")
    done
    for name in $COMPARE_WITH; do configs+=("$CONFIG_DIR/$name.yaml"); done
    aggregate=$(submit --dependency="afterok:$(IFS=:; echo "${jobs[*]}")" \
      --export="$base_export,CONFIGS=$(IFS=:; echo "${configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
      "$SCRIPT_DIR/aggregate_compare.sbatch")
    echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
    ;;
  *) echo "STAGE must be check, truth or run" >&2; exit 2 ;;
esac
