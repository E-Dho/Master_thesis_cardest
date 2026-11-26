#!/bin/bash
# Submit the evaluation chain of the rare-support-off own-model main rows (decided 2026-11-25),
# everything after the trainings:
#   per workload and seed: accuracy at step 11,000 (rule row) and the final-checkpoint sensitivity,
#   then the timing plans, CPU and GPU timing, the sensitivity aggregate, and the table refresh
#   (which also waits for the baseline reruns).
#
#   TRAIN_JOB_LIGHT="<seed0> <seed1> <seed2>" TRAIN_RANGES="<seed0> <seed1> <seed2>" \
#   BASELINE_RERUNS="<id>:<id>:..." [DRY_RUN=1] bash slurm/submit_rare_off_own_model_tables.sh
# Run from the repository root (Master_thesis_cardest_joblight_evaluation).

set -euo pipefail

EVAL=evaluation/job_light_imdb_non_trajectory
REPO=$(pwd)
PLAN_DIRECTORY=$REPO/$EVAL/results/joblight_phase1/timing_plans_rare_off
RULE_RUN_ID=rule_step11000_rare_off
FINAL_RUN_ID=final_sensitivity_rare_off
read -r -a TRAIN_JL <<< "${TRAIN_JOB_LIGHT:?three training job ids, seeds 0 1 2}"
read -r -a TRAIN_RG <<< "${TRAIN_RANGES:?three training job ids, seeds 0 1 2}"
BASELINE_RERUNS=${BASELINE_RERUNS:?colon-separated baseline rerun job ids}
[[ ${#TRAIN_JL[@]} -eq 3 && ${#TRAIN_RG[@]} -eq 3 ]] || { echo "need three ids per workload" >&2; exit 2; }
[[ -f $EVAL/configs/own_model_job_light_rare_off_rule_step_11000.yaml ]] || { echo "run from the repository root" >&2; exit 2; }

submit() {  # submit <dependency> <export> <script>; prints the job id
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "sbatch --parsable --dependency=$1 --export=ALL${2:+,$2} $3" >&2
    echo "DRY$RANDOM"
  else
    sbatch --parsable --dependency="$1" --export="ALL${2:+,$2}" "$3"
  fi
}

ACCURACY=() FINAL=()
for index in 0 1 2; do
  ACCURACY+=("$(submit "afterok:${TRAIN_JL[$index]}" \
    "CONFIG=$EVAL/configs/own_model_job_light_rare_off_rule_step_11000.yaml,SEED=$index,RUN_ID=$RULE_RUN_ID" \
    "$EVAL/slurm/own_model_accuracy.sbatch")")
  ACCURACY+=("$(submit "afterok:${TRAIN_RG[$index]}" \
    "CONFIG=$EVAL/configs/own_model_job_light_ranges_rare_off_rule_step_11000.yaml,SEED=$index,RUN_ID=$RULE_RUN_ID" \
    "$EVAL/slurm/own_model_accuracy.sbatch")")
  FINAL+=("$(submit "afterok:${TRAIN_JL[$index]}" "WORKLOAD=job_light,SEED=$index,RUN_ID=$FINAL_RUN_ID" \
    "$EVAL/slurm/own_model_rare_off_final_sensitivity.sbatch")")
  FINAL+=("$(submit "afterok:${TRAIN_RG[$index]}" "WORKLOAD=job_light_ranges,SEED=$index,RUN_ID=$FINAL_RUN_ID" \
    "$EVAL/slurm/own_model_rare_off_final_sensitivity.sbatch")")
done
ALL_ACCURACY=$(IFS=:; echo "${ACCURACY[*]}")
ALL_FINAL=$(IFS=:; echo "${FINAL[*]}")
PLANS=$(submit "afterok:$ALL_ACCURACY" "PLAN_DIRECTORY=$PLAN_DIRECTORY,RUN_ID=$RULE_RUN_ID" \
  "$EVAL/slurm/rare_off_timing_plans.sbatch")
TIMING_CPU=$(submit "afterok:$PLANS" "PLAN=$PLAN_DIRECTORY/timing_plan_own_model_rare_off_rule_step_11000_cpu.txt" \
  "$EVAL/slurm/timing_cpu_1core.sbatch")
TIMING_GPU=$(submit "afterok:$PLANS" "PLAN=$PLAN_DIRECTORY/timing_plan_own_model_rare_off_rule_step_11000_gpu.txt" \
  "$EVAL/slurm/timing_gpu_single_query.sbatch")
SENSITIVITY=$(submit "afterok:$ALL_ACCURACY:$ALL_FINAL" \
  "RULE_RUN_ID=$RULE_RUN_ID,FINAL_RUN_ID=$FINAL_RUN_ID" "$EVAL/slurm/aggregate_own_model_sensitivity_rare_off.sbatch")
REFRESH=$(submit "afterok:$TIMING_CPU:$TIMING_GPU:$BASELINE_RERUNS" "" \
  "$EVAL/slurm/refresh_rare_off_own_model_tables.sbatch")

cat <<REPORT
accuracy (JL s0, ranges s0, JL s1, ranges s1, JL s2, ranges s2): ${ACCURACY[*]}
final sensitivity (same order): ${FINAL[*]}
timing plans: $PLANS  timing cpu_1core: $TIMING_CPU  timing gpu_single_query: $TIMING_GPU
sensitivity aggregate: $SENSITIVITY  table refresh: $REFRESH
REPORT
