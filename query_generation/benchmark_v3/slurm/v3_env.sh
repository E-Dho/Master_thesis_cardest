#!/bin/bash
# Shared settings of the v3 Slurm jobs. Source after common.sh.
#   PILOT=1        -> 10 queries per category, separate run label
#   RUN_LABEL=...  -> override the run directory name
#   SEED=...       -> base seed (pools and every query slot)

V3_DIR="$REPO_ROOT/query_generation/benchmark_v3"
V3_CONFIG=${V3_CONFIG:-$V3_DIR/pol_v3_config.json}
SEED=${SEED:-20261021}
if [ "${PILOT:-0}" = "1" ]; then
  SINGLE_COUNT=${SINGLE_COUNT:-10}
  MULTI_COUNT=${MULTI_COUNT:-10}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_pilot_seed${SEED}}
else
  SINGLE_COUNT=${SINGLE_COUNT:-250}
  MULTI_COUNT=${MULTI_COUNT:-750}
  RUN_LABEL=${RUN_LABEL:-qgen_pol_v3_exact_st_seed${SEED}}
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
