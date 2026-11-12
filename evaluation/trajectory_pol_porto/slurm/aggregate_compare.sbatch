#!/bin/bash
#SBATCH --job-name=traj_eval_aggregate
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_eval_aggregate_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_eval_aggregate_%j.err
# Aggregate the latest complete seed runs of each config in CONFIGS (colon
# separated paths) and compare them into COMPARISON_DIR, then the same-query
# comparison into COMPARISON_DIR/same_query (SAME_QUERY_REFERENCES: methods compared
# pairwise with every other, default own_model:segments_first_dedup, ignored if absent).
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
PYTHON=${PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
CONFIGS=${CONFIGS:?}
COMPARISON_DIR=${COMPARISON_DIR:?}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
aggregates=()
IFS=: read -r -a config_list <<< "$CONFIGS"
for config in "${config_list[@]}"; do
  output=$("$PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.cli aggregate --config "$config" | tail -n 1)
  echo "aggregate $config -> $output"
  aggregates+=(--aggregate-json "$output/comparison.json")
done
"$PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.cli compare "${aggregates[@]}" --output "$COMPARISON_DIR"
# the same comparison on common queries only (traj_eval/same_query.py), overall and per query type
references=()
for reference in ${SAME_QUERY_REFERENCES-own_model:segments_first_dedup}; do references+=(--reference "$reference"); done
"$PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.same_query "${aggregates[@]}" \
  ${references[@]+"${references[@]}"} --output "$COMPARISON_DIR/same_query"
