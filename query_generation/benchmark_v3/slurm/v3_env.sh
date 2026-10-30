#!/bin/bash
# Shared settings of the v3 Slurm jobs. Source after common.sh.
#   PILOT=1        -> 10 queries per category, separate run label
#   VALIDATION=1   -> validation workload: 8 x 75 single + 8 x 225 multi, its own
#                     seed, every query of the test benchmark excluded
#   TRAINING=1     -> MSCN training workload: 8 x 312 single + 8 x 938 multi
#                     (10,000), its own seed, every query of the test and the
#                     validation benchmark excluded, labelled on the numpy
#                     snapshot by default (no database for the generation)
#   RUN_LABEL=...  -> override the run directory name
#   SEED=...       -> base seed (pools and every query slot)
#   EXCLUDE_HASHES=a.jsonl:b.jsonl -> benchmarks whose queries must not reappear
#   LABEL_BACKEND=sql|numpy        -> label with SQL in the database job (default)
#                     or with NumpyBackend on SNAPSHOT_DIR (parallel, no database)
#   SNAPSHOT_DIR=...               -> numpy snapshot (default traj_snapshots/pol_v3)

V3_DIR="$REPO_ROOT/query_generation/benchmark_v3"
V3_CONFIG=${V3_CONFIG:-$V3_DIR/pol_v3_config.json}
TEST_RUN_LABEL=${TEST_RUN_LABEL:-qgen_pol_v3_exact_st_seed20261021}
VALIDATION_RUN_LABEL=${VALIDATION_RUN_LABEL:-qgen_pol_v3_validation_seed20261026}
EXCLUDE_ARGS=()
if [ "${PILOT:-0}" = "1" ]; then
  SEED=${SEED:-20261021}
  SINGLE_COUNT=${SINGLE_COUNT:-10}
  MULTI_COUNT=${MULTI_COUNT:-10}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_pilot_seed${SEED}}
elif [ "${VALIDATION:-0}" = "1" ]; then
  SEED=${SEED:-20261026}
  SINGLE_COUNT=${SINGLE_COUNT:-75}
  MULTI_COUNT=${MULTI_COUNT:-225}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_validation_seed${SEED}}
  EXCLUDE_HASHES=${EXCLUDE_HASHES:-$MOBILITYDB_ROOT/query_runs/$TEST_RUN_LABEL/benchmark.jsonl}
  PER_CATEGORY=${PER_CATEGORY:-10}
elif [ "${TRAINING:-0}" = "1" ]; then
  SEED=${SEED:-20261030}
  SINGLE_COUNT=${SINGLE_COUNT:-312}
  MULTI_COUNT=${MULTI_COUNT:-938}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_training_seed${SEED}}
  EXCLUDE_HASHES=${EXCLUDE_HASHES:-$MOBILITYDB_ROOT/query_runs/$TEST_RUN_LABEL/benchmark.jsonl:$MOBILITYDB_ROOT/query_runs/$VALIDATION_RUN_LABEL/benchmark.jsonl}
  LABEL_BACKEND=${LABEL_BACKEND:-numpy}
  PER_CATEGORY=${PER_CATEGORY:-10}
else
  SEED=${SEED:-20261021}
  SINGLE_COUNT=${SINGLE_COUNT:-250}
  MULTI_COUNT=${MULTI_COUNT:-750}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_exact_st_seed${SEED}}
fi
if [ -n "${EXCLUDE_HASHES:-}" ]; then
  IFS=: read -r -a _exclude_list <<< "$EXCLUDE_HASHES"
  for _exclude in "${_exclude_list[@]}"; do
    if [ ! -f "$_exclude" ]; then
      echo "EXCLUDE_HASHES entry $_exclude does not exist" >&2
      exit 2
    fi
    EXCLUDE_ARGS+=(--exclude-hashes "$_exclude")
  done
fi
LABEL_BACKEND=${LABEL_BACKEND:-sql}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-$THESIS_WORK_ROOT/traj_snapshots/pol_v3}
case "$LABEL_BACKEND" in
  sql|numpy) ;;
  *) echo "LABEL_BACKEND must be sql or numpy" >&2; exit 2 ;;
esac
RUN_DIR="$MOBILITYDB_ROOT/query_runs/$RUN_LABEL"
POOLS_JSON="$RUN_DIR/pools.json"
CATEGORY_DIR="$RUN_DIR/categories"
GRID_ARGS=(--single-count "$SINGLE_COUNT" --multi-count "$MULTI_COUNT")
DB_ARGS=(--host "$MOBILITYDB_SOCKET_DIR" --port "$MOBILITYDB_PORT" --dbname "$MOBILITYDB_DB")
if [ -n "${MOBILITYDB_USER:-}" ]; then
  DB_ARGS+=(--user "$MOBILITYDB_USER")
fi

V3_CATEGORIES=(
  standard.range.single standard.range.multi
  standard.unbounded.single standard.unbounded.multi
  temporal.range.single temporal.range.multi
  temporal.unbounded.single temporal.unbounded.multi
  spatial.range.single spatial.range.multi
  spatial.unbounded.single spatial.unbounded.multi
  spatio_temporal.range.single spatio_temporal.range.multi
  spatio_temporal.unbounded.single spatio_temporal.unbounded.multi
)

v3_resolve_repo_root() {
  REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
  if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/dataset_generation/mobilitydb_loader/common.sh" ]; then
    echo "Set REPO_ROOT to the checkout of feature/exact-st-benchmark-v3" >&2
    exit 2
  fi
}
