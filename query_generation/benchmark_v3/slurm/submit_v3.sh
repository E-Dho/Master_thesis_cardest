#!/bin/bash
# Submit the v3 pipeline: pools -> 16 category tasks -> merge -> verify.
#   PILOT=1 ./submit_v3.sh        10 queries per category
#   SKIP_POOLS=1 ./submit_v3.sh   resume generation after a deadline stop
#   STAGE=check ./submit_v3.sh    only the Phase 1 cluster check
#   STAGE=verify ./submit_v3.sh   only (re)run the verification of a merged run
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
export_vars="ALL,REPO_ROOT=$REPO_ROOT,PILOT=${PILOT:-0},SEED=${SEED:-20261021}"
if [ -n "${RUN_LABEL:-}" ]; then export_vars="$export_vars,RUN_LABEL=$RUN_LABEL"; fi

submit() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }

if [ "${STAGE:-all}" = "check" ]; then
  echo "cluster_check_job=$(submit --export="$export_vars" "$SCRIPT_DIR/run_v3_cluster_check.sbatch")"
  exit 0
fi

if [ "${STAGE:-all}" = "verify" ]; then
  echo "verify_job=$(submit --export="$export_vars" "$SCRIPT_DIR/run_v3_verify.sbatch")"
  exit 0
fi

dependency=()
if [ "${SKIP_POOLS:-0}" != "1" ]; then
  pools=$(submit --export="$export_vars" "$SCRIPT_DIR/run_v3_pools.sbatch")
  echo "pools_job=$pools"
  dependency=(--dependency="afterok:$pools")
fi
generate=$(submit "${dependency[@]}" --array=0-15%1 --export="$export_vars" "$SCRIPT_DIR/run_v3_generate.sbatch")
merge=$(submit --dependency="afterok:$generate" --export="$export_vars" "$SCRIPT_DIR/run_v3_merge.sbatch")
verify=$(submit --dependency="afterok:$merge" --export="$export_vars" "$SCRIPT_DIR/run_v3_verify.sbatch")
echo "generate_array_job=$generate"
echo "merge_job=$merge"
echo "verify_job=$verify"
