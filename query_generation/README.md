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

## Columns A Query Cannot Combine Freely

Some column pairs are empty for every row no matter which bounds are drawn,
because the schema says so. Two config keys let the generator avoid emitting
those queries instead of paying a database evaluation to discover them.

`exclusive_attribute_groups` lists sets of attributes of which a query
constrains at most one. Use it only for columns that are *genuinely* mutually
exclusive. Groups must not overlap, since the reduction picks one surviving
member per group and overlapping membership would make that ambiguous;
`validate_config` rejects overlapping, single-member, repeated and unknown
members. If a group makes a category unsatisfiable -- putting both the spatial
and the temporal attribute of a `spatio_temporal` query in one group, say --
generation raises `ConfigError` rather than looping.

`attribute_value_dependencies` is for the commoner case where a nominal
discriminator decides whether another column is non-null. Each entry names the
dependent `attribute`, the nominal `determined_by` column, and the `values` that
make the dependent live. When both are selected the discriminator's value is
drawn from that list; when the discriminator appears alone it is unrestricted.
Two dependents pinning one discriminator to disjoint values make a selection
unsatisfiable, and the sampler drops that combination rather than emitting it.

Porto uses both. `CALL_TYPE` partitions trips into `A` (origin_call set,
origin_stand null), `B` (origin_stand set, origin_call null) and `C` (both
null); across the full Kaggle file not one of 1,674,152 eligible rows has both
origin columns set. So the two origin columns form an exclusive group, while
`call_type` is declared as their discriminator rather than a third exclusive
peer -- `call_type = 'A' AND origin_call BETWEEN ...` is a good query and must
survive; only the mismatched pairing is impossible. Measured over 2,000
generated `standard.*` queries: 39.8% constrained an impossible combination
before, none do now, and 565 still pair `call_type` with an origin column, every
one of them correctly pinned.

This removes queries that are *provably* empty. It cannot remove queries that
happen to be empty for the data, so it reduces the zero rate rather than
guaranteeing zero; the rejection-sampling stage below is what makes a workload
actually zero-free.

## Create A Zero-Free Variant

`resample_nonzero_benchmark.py` keeps every positive row in an evaluated
workload and replaces only its true-zero rows. Each replacement is a fresh
category-local draw and is accepted only when its exact join cardinality is
positive. The output retains the original query IDs and final category counts.
Legacy POL candidates are normalized to the same-segment spatial-temporal
semantics before evaluation. Porto candidates already use only `segments` for
spatial and temporal predicates, so they need no SQL rewrite.

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
re-running every `COUNT(*)` against the 50M-row database. Pass `--no-progress`
to disable both sidecars.

The centers snapshot matters for more than speed: the pools come from an
unseeded `ORDER BY random()`, so it is the snapshot, not `--seed`, that makes a
run reconstructible. Two ordering and binding rules follow from that.

A newly fetched pool is written and fsynced as soon as the candidate that
fetched it has been generated — before that candidate is tested or evaluated,
and so before the replacement that eventually uses the pool is appended to the
progress file. Persisting on acceptance instead is not enough: a candidate that
is rejected for zero cardinality or duplicate SQL has already drawn its pools,
and an interruption after such a rejection would leave those pools unsaved. The
restart would refetch them from an unseeded `ORDER BY random()` and change the
rest of the benchmark.

Both sidecars are bound to the run that produced them, and a mismatch is a hard
error rather than a silent reuse. The same binding — generator config hash,
source workload SHA-256, sampling cache size and seed — is stored in the centers
snapshot and in a header line written as the first record of the progress file,
so neither sidecar can be reused across a changed run even if the other one is
missing. Each progress row is additionally checked against the source row it
claims to replace (row index, replaced SQL hash, true-zero status), against the
generator config hash the generator stamps on every record, and against the
replacement seed derived from `--seed`. Replacements must also be distinct from
one another, because their SQL hashes seed the duplicate guard for every row
still to be sampled, from the first candidate onwards rather than only once the
loop reaches their own index.

The output JSONL and summary are written before the final validation pass, so a
validation failure reports the problem without discarding the evaluated rows.

For Porto, submit the normal frozen-workload evaluator chain:

```bash
REPO_ROOT=/zfshome/sunip956/master_thesis_trajectories/Master_thesis_cardest_git \
bash query_generation/submit_porto_8000_evaluation.sh
```

It evaluates the frozen 8,000-row source in 32 serialized slices, merges it as
`porto_taxi_50m_8000_segment_coupled_v1`, then submits the dependent
`qreplace_porto_nonzero` job. That job keeps all exact-positive source rows and
replaces only true zeros. Its final workload and summary are written under
`$PORTO_MOBILITYDB_ROOT/query_runs/porto_taxi_50m_8000_segment_coupled_nonzero_v1/`.
It validates 8,000 unique IDs, 500 rows in each category, 4,000 multi-relation
distinct-trip truths, segment-coupled spatio-temporal predicates, and strictly
positive exact join cardinalities.

The replacement job also checks that every source row's `config_hash` matches
the `--config` it was given, so a workload frozen under a different query config
is refused rather than silently resampled from a different distribution.

Nothing resubmits the replacement job if it exceeds its wall clock. Re-submit
`run_porto_nonzero_replacement.sbatch` with the same `RUN_LABEL`, `INPUT_JSONL`
and `REPLACEMENT_SEED`; it resumes from the progress and centers sidecars rather
than re-evaluating the replacements it already accepted.

## Tests

```bash
python3 -m unittest query_generation.test_query_generator
```
