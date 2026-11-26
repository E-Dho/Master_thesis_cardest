#!/bin/bash
#SBATCH --job-name=own_model_sensitivity_rare_off
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Best validation checkpoint versus final checkpoint (step 20,000) of the rare-support-off main runs, as
# aggregate_own_model_sensitivity.sbatch did for the earlier rare-auxiliary runs.

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
EVAL=evaluation/job_light_imdb_non_trajectory
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

mapfile -t ARGS < <("$PYTHON" "$EVAL/scripts/rare_off_own_model_runs.py" sensitivity-args \
  --rule-run-id "${RULE_RUN_ID:-best_val_rare_off}" \
  --final-run-id "${FINAL_RUN_ID:-final_sensitivity_rare_off}")
[[ ${#ARGS[@]} -eq 24 ]] || { echo "expected 24 arguments, got ${#ARGS[@]}" >&2; exit 2; }
"$PYTHON" "$EVAL/scripts/aggregate_own_model_sensitivity.py" "${ARGS[@]}" \
  --output-directory "$REPO/$EVAL/results/joblight_phase1/own_model_sensitivity_rare_off"
