#!/bin/bash
# P3: Exact-MBR and the trajectory sampling ladder on POL v3, then aggregate
# and compare. Needs the P2 snapshot (pass AFTER_JOB=<gate merge job id> to
# start right after P2, or submit once the gate reports passed).
#   ./submit_p3.sh                      all P3 configs
#   CONFIG_NAMES="exact_mbr" ./submit_p3.sh
# Accuracy runs time one pass; reported latency comes later from `time`.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
CONFIG_DIR="$REPO_ROOT/evaluation/trajectory_pol_porto/configs/pol_v3"
CONFIG_NAMES=${CONFIG_NAMES:-"exact_mbr traj_sampling_n1000 traj_sampling_n10000 traj_sampling_n100000 traj_sampling_n1000000"}
COMPARISON_DIR=${COMPARISON_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/traj_phase1_pol_v3/p3_comparison}
base_export="ALL,REPO_ROOT=$REPO_ROOT"
if [ -n "${PYTHON:-}" ]; then base_export="$base_export,PYTHON=$PYTHON"; fi

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

seeds_of() {  # the seeds list of a config (its own or the base's), as a Slurm array spec
  local list
  list=$(grep -E '^  seeds:' "$1" | tail -n 1 | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' ')
  if [ -z "$list" ]; then list=$(grep -E '^  seeds:' "$CONFIG_DIR/_base.yaml" | sed -E 's/.*\[(.*)\].*/\1/' | tr -d ' '); fi
  echo "$list"
}

dependency_args=()
if [ -n "${AFTER_JOB:-}" ]; then dependency_args=(--dependency="afterok:$AFTER_JOB"); fi
jobs=()
configs=()
for name in $CONFIG_NAMES; do
  config="$CONFIG_DIR/$name.yaml"
  if [ ! -f "$config" ]; then echo "missing config $config" >&2; exit 2; fi
  seeds=$(seeds_of "$config")
  job=$(submit "${dependency_args[@]}" --array="$seeds" --export="$base_export,CONFIG=$config" \
    "$SCRIPT_DIR/run_inprocess.sbatch")
  echo "$name: job $job (seeds $seeds)"
  jobs+=("$job")
  configs+=("$config")
done
dependency=$(IFS=:; echo "afterok:${jobs[*]}")
aggregate=$(submit --dependency="$dependency" \
  --export="$base_export,CONFIGS=$(IFS=:; echo "${configs[*]}"),COMPARISON_DIR=$COMPARISON_DIR" \
  "$SCRIPT_DIR/aggregate_compare.sbatch")
echo "aggregate_compare: job $aggregate -> $COMPARISON_DIR"
