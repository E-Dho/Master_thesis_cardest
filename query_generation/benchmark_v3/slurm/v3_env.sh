#!/bin/bash
# Shared settings of the v3 Slurm jobs. Source after common.sh.
#   PILOT=1        -> 10 queries per category, separate run label
#   VALIDATION=1   -> validation workload: 8 x 75 single + 8 x 225 multi, its own
#                     seed, every query of the test benchmark excluded
#   RUN_LABEL=...  -> override the run directory name
#   SEED=...       -> base seed (pools and every query slot)

V3_DIR="$REPO_ROOT/query_generation/benchmark_v3"
V3_CONFIG=${V3_CONFIG:-$V3_DIR/pol_v3_config.json}
TEST_RUN_LABEL=${TEST_RUN_LABEL:-qgen_pol_v3_exact_st_seed20261021}
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
else
  SEED=${SEED:-20261021}
  SINGLE_COUNT=${SINGLE_COUNT:-250}
  MULTI_COUNT=${MULTI_COUNT:-750}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_exact_st_seed${SEED}}
fi
if [ -n "${EXCLUDE_HASHES:-}" ]; then
  if [ ! -f "$EXCLUDE_HASHES" ]; then
    echo "EXCLUDE_HASHES=$EXCLUDE_HASHES does not exist" >&2
    exit 2
  fi
  EXCLUDE_ARGS=(--exclude-hashes "$EXCLUDE_HASHES")
fi
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
