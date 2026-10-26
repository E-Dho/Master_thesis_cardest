#!/bin/bash
#SBATCH --job-name=deepdb_ranges_adapted_prepare
#SBATCH --partition=base
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err
#
# Adapted DeepDB JOB-light-ranges preprocessing, gated by validation:
#   1. PostgreSQL rank domains + ranked, checksummed headerless CSVs
#   2. validation before the expensive HDF stage (support 1000/1000 and 70/70,
#      rank rewrite vs PostgreSQL, collation-dependent label reproduction)
#   3. DeepDB HDF and sampled HDF on the adapted schema
#   4. validation again, now also checking the columns DeepDB kept in the HDF
# Never touches the native deepdb_shared root.

set -euo pipefail
source "$(dirname "$0")/_deepdb_ranges_adapted_common.sh" 2>/dev/null \
  || source /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest_joblight_evaluation/evaluation/job_light_imdb_non_trajectory/slurm/_deepdb_ranges_adapted_common.sh
LABEL_CHECK_LIMIT=${LABEL_CHECK_LIMIT:--1}   # -1: every collation-dependent query
RANK_COLLATION=${RANK_COLLATION:-C}
start_postgres

common_args=(--source-root "$DEEPDB_SOURCE" --revision "$DEEPDB_REVISION")
workloads=(--workload "$RANGES" --workload "$NORMAL")
validate_args=(--workload "$RANGES" --expected-query-count 1000 --workload "$NORMAL" --expected-query-count 70
               --label-check-limit "$LABEL_CHECK_LIMIT")
if [[ -n "${PG_TEMP_FILE_LIMIT:-}" ]]; then
  validate_args+=(--temp-file-limit "$PG_TEMP_FILE_LIMIT")
fi

"$DEEPDB_PYTHON" "$BRIDGE" prepare-shared "${common_args[@]}" --dataset-root "$DATASET" \
  --output-directory "$ADAPTED_SHARED" --postgres-dsn "$DSN" "${workloads[@]}" --stage dataset \
  --collation "$RANK_COLLATION"
"$DEEPDB_PYTHON" "$BRIDGE" validate "${common_args[@]}" --shared-root "$ADAPTED_SHARED" \
  --postgres-dsn "$DSN" "${validate_args[@]}" --output "$ADAPTED_SHARED/validation/pre_hdf_validation.json"
"$DEEPDB_PYTHON" "$BRIDGE" prepare-shared "${common_args[@]}" --dataset-root "$DATASET" \
  --output-directory "$ADAPTED_SHARED" --postgres-dsn "$DSN" "${workloads[@]}" --stage all \
  --collation "$RANK_COLLATION"
"$DEEPDB_PYTHON" "$BRIDGE" validate "${common_args[@]}" --shared-root "$ADAPTED_SHARED" \
  --postgres-dsn "$DSN" "${validate_args[@]}" --label-check-limit 0 \
  --output "$ADAPTED_SHARED/validation/post_hdf_validation.json"
cat "$ADAPTED_SHARED/shared_preprocessing_manifest.json"
echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
