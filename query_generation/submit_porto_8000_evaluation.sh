#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
INPUT_JSONL=${INPUT_JSONL:-$PORTO_MOBILITYDB_ROOT/query_runs/porto_taxi_50m_8000_segment_coupled_v1_frozen/queries.jsonl}
SLICE_PREFIX=${SLICE_PREFIX:-porto_taxi_50m_8000_segment_coupled_v1_eval}
test -f "$INPUT_JSONL"

previous_job=
for chunk in $(seq 0 31); do
  start=$((chunk * 250))
  end=$((start + 250))
  dependency=()
  if [ -n "$previous_job" ]; then dependency=(--dependency="afterok:$previous_job"); fi
  job=$(INPUT_JSONL="$INPUT_JSONL" START_INDEX="$start" END_INDEX="$end" RUN_LABEL="${SLICE_PREFIX}_$(printf %02d "$chunk")" \
    sbatch --parsable --job-name="qporto_$(printf %02d "$chunk")" "${dependency[@]}" "$REPO_ROOT/query_generation/run_porto_query_evaluator.sbatch")
  previous_job=${job%%;*}
  printf 'chunk=%02d job_id=%s start=%d end=%d\n' "$chunk" "$previous_job" "$start" "$end"
done
merge=$(INPUT_JSONL="$INPUT_JSONL" SLICE_PREFIX="$SLICE_PREFIX" sbatch --parsable --dependency="afterok:$previous_job" --job-name=qmerge_porto "$REPO_ROOT/query_generation/run_porto_query_merge.sbatch")
printf 'merge_job=%s\n' "${merge%%;*}"
