#!/bin/bash
#SBATCH --job-name=qmerge_porto
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_porto_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/qmerge_porto_%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh" ]; then
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
source "$REPO_ROOT/dataset_generation/porto_taxi_loader/common.sh"
INPUT_JSONL=${INPUT_JSONL:?INPUT_JSONL is required}
BENCHMARK_LABEL=${BENCHMARK_LABEL:-porto_taxi_50m_8000_segment_coupled_v1}
SLICE_PREFIX=${SLICE_PREFIX:-porto_taxi_50m_8000_segment_coupled_v1_eval}
OUTPUT_DIR="$PORTO_MOBILITYDB_ROOT/query_runs/$BENCHMARK_LABEL"
mkdir -p "$OUTPUT_DIR"
slice_args=()
for chunk in $(seq -w 0 31); do
  slice_args+=(--slice "$PORTO_MOBILITYDB_ROOT/query_runs/${SLICE_PREFIX}_${chunk}/queries.jsonl")
done
cd "$REPO_ROOT"
python query_generation/merge_evaluated_slices.py --original "$INPUT_JSONL" "${slice_args[@]}" \
  --output "$OUTPUT_DIR/queries.jsonl" --summary "$OUTPUT_DIR/benchmark_summary.json" \
  --queries-per-category 500 --expected-distinct-rows 4000
