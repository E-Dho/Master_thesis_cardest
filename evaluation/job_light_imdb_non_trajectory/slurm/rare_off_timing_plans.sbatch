#!/bin/bash
#SBATCH --job-name=rare_off_timing_plans
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Writes the timing plans of the six rare-support-off rule runs once their accuracy runs
# are complete (the run directories carry the config hash, known only after the run).
#   PLAN_DIRECTORY=... RUN_ID=... sbatch slurm/rare_off_timing_plans.sbatch

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
cd "$REPO"
"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/rare_off_own_model_runs.py timing-plans \
  --run-id "${RUN_ID:?}" --output-directory "${PLAN_DIRECTORY:?}"
cat "$PLAN_DIRECTORY"/timing_plan_own_model_rare_off_rule_step_11000_cpu.txt
