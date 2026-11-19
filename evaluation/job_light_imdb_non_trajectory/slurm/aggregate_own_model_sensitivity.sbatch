#!/bin/bash
#SBATCH --job-name=own_model_sensitivity
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
EVAL=evaluation/job_light_imdb_non_trajectory
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
RESULTS=$REPO/$EVAL/results/joblight_phase1/own_model

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

RULE_JL=$RESULTS/duet_neurocard_distjoin_rare_aux_rule_step_11000_corrected_model_code
RULE_RANGES=$RESULTS/duet_neurocard_distjoin_rare_aux_content_rule_step_11000_corrected_model_code
FINAL_JL0=$RESULTS/own_model_job_light_final_seed0_sensitivity/seed_0/final_sensitivity-d6cbdb943df19af9bfee5761731cc590031aca06918a0b5894efc3a36160cbc7
FINAL_JL1=$RESULTS/own_model_job_light_final_seed1_sensitivity/seed_1/final_sensitivity-54135318cf1be4b9b600cd13e785be234192a448cd7b4a927ee25d39f8374c40
FINAL_JL2=$RESULTS/own_model_job_light_final_seed2_sensitivity/seed_2/final_sensitivity-da950654396d908c0bdda99cf840d518e2aef3667dd429ce6cc01a9ae129a5f4
FINAL_RANGES=$RESULTS/own_model_job_light_ranges_final_sensitivity

"$PYTHON" "$EVAL/scripts/aggregate_own_model_sensitivity.py" \
  --job-light-rule "$RULE_JL/seed_0/reused_rule_step11000-7cd422ccfed0ac87bbce7531dbb5d073573f2e7169e5fb96c71022dcb44fadc5" \
  --job-light-rule "$RULE_JL/seed_1/reused_rule_step11000-7cd422ccfed0ac87bbce7531dbb5d073573f2e7169e5fb96c71022dcb44fadc5" \
  --job-light-rule "$RULE_JL/seed_2/reused_rule_step11000-7cd422ccfed0ac87bbce7531dbb5d073573f2e7169e5fb96c71022dcb44fadc5" \
  --job-light-final "$FINAL_JL0" \
  --job-light-final "$FINAL_JL1" \
  --job-light-final "$FINAL_JL2" \
  --ranges-rule "$RULE_RANGES/seed_0/rule_step11000-bd920737b3854cbb67b5f72b2c1f02ac0cbe6b56931dc2f4239b764f04c2020d" \
  --ranges-rule "$RULE_RANGES/seed_1/rule_step11000-bd920737b3854cbb67b5f72b2c1f02ac0cbe6b56931dc2f4239b764f04c2020d" \
  --ranges-rule "$RULE_RANGES/seed_2/rule_step11000-bd920737b3854cbb67b5f72b2c1f02ac0cbe6b56931dc2f4239b764f04c2020d" \
  --ranges-final "$FINAL_RANGES/seed_0/final_sensitivity-0220c7d9202dee91ee398322dc9f3a05578340ca621a6610e4bb84a61e3c446d" \
  --ranges-final "$FINAL_RANGES/seed_1/final_sensitivity-0220c7d9202dee91ee398322dc9f3a05578340ca621a6610e4bb84a61e3c446d" \
  --ranges-final "$FINAL_RANGES/seed_2/final_sensitivity-0220c7d9202dee91ee398322dc9f3a05578340ca621a6610e4bb84a61e3c446d" \
  --output-directory "$REPO/$EVAL/results/joblight_phase1/own_model_sensitivity"
