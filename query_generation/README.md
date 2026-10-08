# Query Generation

This directory contains the config-driven SQL workload generator used to create labeled cardinality-estimation queries.

The generator samples COUNT queries across:

- dimensions: `standard`, `temporal`, `spatial`, `spatio_temporal`
- intervals: `range`, `unbounded`
- relations: `single`, `multi`

## Generate SQL Only

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --no-execute
```

## Generate And Execute

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --execute \
  --host 127.0.0.1 \
  --port 55432 \
  --dbname pol_mobilitydb \
  --user sunip956
```

## Create A Zero-Free POL Variant

`resample_nonzero_benchmark.py` keeps every positive row in an evaluated POL
workload and replaces only its true-zero rows. Each replacement is a fresh
category-local draw, normalized to the same-segment spatial-temporal semantics
before evaluation, and accepted only when its exact join cardinality is
positive. The output retains the original query IDs and final category counts.

On the cluster, submit the serialized job after setting `REPO_ROOT` to the
checkout containing the desired branch:

```bash
REPO_ROOT=/zfshome/sunip956/master_thesis_trajectories/Master_thesis_cardest_git \
sbatch query_generation/run_pol_nonzero_replacement.sbatch
```

The default source is the finalized segment-coupled workload and the default
output label is `qgen_pol_50m_500qpc_20260824_segment_coupled_nonzero_v1`.
The generated `benchmark_summary.json` records the source hash, replacement
seed, replaced-row count, category counts, table-subset counts before and after
replacement, the live-center pool fingerprints, and rejection-sampling attempts.

Rejection sampling only ever accepts a positive draw, so the zero-free variant
has no more small-cardinality queries than the source workload had. That is
intentional: the workload is meant to mirror the data, and if hard queries are
rare in the data they stay rare here. Read aggregate q-error on this benchmark
with that in mind, and stratify by true cardinality when the question is about
behaviour near zero.

Each accepted replacement is appended to `queries.jsonl.partial.jsonl` and
fsynced before the next candidate is evaluated, and the live-center pools are
snapshotted to `queries.jsonl.centers.json`. Resubmitting the job with the same
`RUN_LABEL`, `INPUT_JSONL` and `REPLACEMENT_SEED` resumes from both rather than
re-running every `COUNT(*)` against the 50M-row database; a sidecar written from
a different source workload, row ordering or seed is rejected instead of being
reused. The centers snapshot matters for more than speed: the pools come from an
unseeded `ORDER BY random()`, so it is the snapshot, not `--seed`, that makes a
run reconstructible. Pass `--no-progress` to disable both sidecars.

The output JSONL and summary are written before the final validation pass, so a
validation failure reports the problem without discarding the evaluated rows.

## Tests

```bash
python3 -m unittest query_generation.test_query_generator
```
