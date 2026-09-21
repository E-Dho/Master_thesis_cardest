#!/bin/bash
#SBATCH --job-name=qmerge_pol
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_pol_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_pol_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/query_generation/merge_segment_coupled_benchmark.py" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi

ORIGINAL_JSONL=${ORIGINAL_JSONL:?ORIGINAL_JSONL is required}
CORRECTED_RUN_ROOT=${CORRECTED_RUN_ROOT:?CORRECTED_RUN_ROOT is required}
OUTPUT_JSONL=${OUTPUT_JSONL:?OUTPUT_JSONL is required}
SUMMARY_JSON=${SUMMARY_JSON:?SUMMARY_JSON is required}

corrected_args=()
for chunk in 00 01 02 03 04 05 06 07; do
  shard="$CORRECTED_RUN_ROOT/qeval_pol_50m_segment_coupled_v1_chunk_$chunk/queries.jsonl"
  if [ "$chunk" = "05" ]; then
    shard="$CORRECTED_RUN_ROOT/qeval_pol_50m_segment_coupled_v1_chunk_05_recovery/queries.jsonl"
  fi
  corrected_args+=(--corrected "$shard")
done

cd "$REPO_ROOT"
python3 query_generation/merge_segment_coupled_benchmark.py \
  --original "$ORIGINAL_JSONL" \
  "${corrected_args[@]}" \
  --output "$OUTPUT_JSONL" \
  --summary "$SUMMARY_JSON"
