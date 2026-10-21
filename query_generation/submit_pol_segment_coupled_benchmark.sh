#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
QUERIES_PER_CATEGORY=${QUERIES_PER_CATEGORY:-500}
RUN_LABEL=${RUN_LABEL:-qgen_pol_50m_${QUERIES_PER_CATEGORY}qpc_segment_coupled_v2}
NONZERO_RUN_LABEL=${NONZERO_RUN_LABEL:-qgen_pol_50m_${QUERIES_PER_CATEGORY}qpc_segment_coupled_nonzero_v2}

common_export="ALL,REPO_ROOT=$REPO_ROOT,QUERIES_PER_CATEGORY=$QUERIES_PER_CATEGORY,RUN_LABEL=$RUN_LABEL"
generation=$(sbatch --parsable --array=0-63%1 --export="$common_export" \
  "$REPO_ROOT/query_generation/run_pol_segment_coupled_benchmark.sbatch")
generation=${generation%%;*}

merge=$(sbatch --parsable --dependency="afterok:$generation" --export="$common_export" \
  "$REPO_ROOT/query_generation/run_pol_segment_coupled_generation_merge.sbatch")
merge=${merge%%;*}

input_jsonl="/work_beegfs/sunip956/master_thesis_trajectories/mobilitydb/query_runs/$RUN_LABEL/queries.jsonl"
replacement_export="ALL,REPO_ROOT=$REPO_ROOT,INPUT_JSONL=$input_jsonl,RUN_LABEL=$NONZERO_RUN_LABEL"
replacement=$(sbatch --parsable --dependency="afterok:$merge" --export="$replacement_export" \
  "$REPO_ROOT/query_generation/run_pol_nonzero_replacement.sbatch")
replacement=${replacement%%;*}

printf 'generation_array_job=%s\n' "$generation"
printf 'merge_job=%s\n' "$merge"
printf 'nonzero_replacement_job=%s\n' "$replacement"
