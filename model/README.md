# INV_FANOUT Predicate-Conditioned Join Cardinality Prototype

## Project Overview

This directory contains the first implementation milestone for a learned
join-cardinality estimator that combines three ideas:

1. NeuroCard-style modeling over uniform full-outer-join tuples.
2. Duet-style predicate-conditioned, sampling-free selectivity estimation.
3. A new `INV_FANOUT` virtual token with weighted cross entropy, so inverse
   fanout marginalization is amortized into later autoregressive heads.

The current code prioritizes mathematical correctness and testability over
benchmark-scale optimization. The lightweight autoregressive conditional-table
prototype remains available as `model.type: prototype_table` for fast exact
tests. Real training is now wired through a PyTorch predicate-conditioned
ResMADE backend, `model.type: predicate_resmade`.

## Relationship To Upstream Work

- NeuroCard: https://github.com/neurocard/neurocard
  - Conceptual base for full-outer-join training tuples, table-presence
    indicators, fanout columns, and checkpointed column ordering.
  - The ResMADE masking structure and Exact Weight sampler adapter boundary are
    adapted from NeuroCard's `made.py` and `factorized_sampler.py` design.
- Duet: https://github.com/GIS-PuppetMaster/Duet
  - Conceptual base for virtual predicate tokens, predicate-conditioned output
    heads, per-column logit slicing, and one-pass inference without progressive
    sampling.
- DistJoin: https://github.com/GIS-PuppetMaster/DistJoin
  - Conceptual base for lossless high-bit-to-low-bit column factorization and
    ANPM-style previous-factor embedding modulation. The implementation adapts
    those ideas to this repository's original-column-preserving ResMADE layout.

No upstream source files are vendored or copied. See `ATTRIBUTION.md`.

## ResMADE Architecture

`model/src/model/resmade.py` implements a PyTorch ResMADE over grouped virtual
predicate-token inputs. `model/src/model/masked_layers.py` provides
NeuroCard-inspired masked linear and residual layers. Masks are built from fixed
column degrees:

```text
input token group k has degree k
output slice i may only see inputs with degree < i
```

Residual hidden layers preserve their hidden degree, so residual connections do
not bypass the autoregressive contract. Optional direct input-output
connections use the same strict `< i` mask.

## Input Tokens Versus Output Values

The model input and output domains are intentionally separate:

```text
predicate_input_bins[i] != data_output_bins[i] is allowed
```

A fanout column may have two input tokens:

```text
WILDCARD, INV_FANOUT
```

while its real output domain may contain:

```text
1, 2, 3, 5, 10, 20, ...
```

Ordinary columns similarly preserve both operator identity and predicate value.
`EQUAL(5)`, `LESS_EQUAL(5)`, and `GREATER_EQUAL(5)` are different input tokens.
Output heads always classify actual encoded data, indicator, or fanout values.

## Lossless Column Factorization

Original columns remain the query-facing schema. For a selected high-cardinality
ordinary data column `X_i`, the model may internally represent the original
dictionary ID as bit factors:

```text
x <-> (z_1, ..., z_K)
```

The mapping is deterministic, most-significant-factor first, and derived only
from the complete original domain size in metadata. `sample_rows.npy` stays in
original encoded-column form; factor targets are produced by a deterministic
training adapter. Invalid bit combinations beyond the original domain are never
decoded as valid IDs.

## ANPM Factor Decoding

For a factorized original column, the outer ResMADE supplies context from
previous original predicate tokens:

```text
q_i(x | T_<i) = product_k q_{i,k}(z_k | T_<i, z_<k)
```

Each factorized column owns a local ANPM decoder. The first factor uses the base
ResMADE logits unchanged. Later factors use a faithful DistJoin-style
generated-weight hypernetwork, confined to that one original column. The
previous additive prefix-offset ablation remains available on the previous Git
branch; this branch replaces it with generated low-rank transforms.

The old ablation had the form:

```text
ell_additive(c, p) = b(c) + h(p)
```

The active decoder instead applies a prefix-generated transformation:

```text
ell_hyper(c, p) = G_p(b(c))
```

For factor `k > 0`, embeddings of preceding same-column factors are
concatenated:

```text
e_k(p) = concat(E_0[z_0], ..., E_{k-1}[z_{k-1}])
```

Six small MLPs generate vectors for two low-rank matrices and biases:

```text
W_1(p) = a_1(p) b_1(p)^T
W_2(p) = a_2(p) b_2(p)^T
```

With current factor domain size `D_k` and ANPM hidden size `H`, `W_1` has shape
`D_k x H` and `W_2` has shape `H x D_k`. The factor logits are:

```text
h_k     = ReLU(b_k(c) W_1(p) + beta_1(p))
ell_k   = ReLU(h_k W_2(p) + beta_2(p))
```

This mirrors DistJoin's two-stage generated low-rank ANPM path while preserving
the repository's original-column-preserving ResMADE layout. Training uses
teacher-forced true prefixes. Inference enumerates prefixes deterministically in
chunks; no progressive factor sampling is introduced.

## Original Columns Versus Model Heads

`ModelMetadata.columns` remains authoritative and contains original columns
only. `factorization_plan.output_head_specs` describes model heads:

```text
unfactorized X_i -> one output head
factorized X_i   -> one output head per factor
```

Every factor head for `X_i` receives outer autoregressive degree `i`, so it may
depend on `T_<i` but not on `T_i` or future tokens. Same-column factor
dependence enters only through ANPM.

## Interaction With INV_FANOUT

Indicators and fanout columns remain atomic in this milestone. The
`INV_FANOUT` token, reciprocal fanout mask, and cumulative inverse-fanout
weighted cross entropy are unchanged. Factorized ordinary columns can coexist
with fanout heads without changing fanout effective sample size or weighting
semantics.

## Factorized Training Loss

Factorized labels are trained with teacher forcing. For each row, the CE terms
for all factors of one original column are summed first. The original-column row
weight is then applied once:

```text
L_i = sum_b w_bi * sum_k CE(q_i,k, z_i,k) / (sum_b w_bi + eps)
```

This preserves original-column loss semantics and avoids normalizing each factor
as if it were an independent modeled column.

## Factorized Inference

Inference remains one predicate-conditioned ResMADE backbone pass per query row,
followed by deterministic ANPM decoding. The output adapter hides factor digits
from callers and computes original-column factors:

```text
s_i = sum_x q_i(x) phi_i(x)
```

For factorized columns, valid original IDs are enumerated in chunks. Their
factor paths are evaluated with the same prefix-conditioned ANPM decoder used
during training. Invalid factor completions are masked before each factor
softmax when `anpm.mask_invalid_combinations: true`, and original-column
predicate masks are applied through the output adapter. Production JOB-light
two-sided ranges now use native interval mass from one conditioned
original-column state rather than subtracting two independent full-query
cardinality estimates.

## Distinct Trajectory Cardinality

The POL trajectory branch adds an optional output-only terminal head named
`traj_dedup_factor`. It estimates a query-local correction for converting the
existing matching-segment estimate into a distinct-trajectory estimate:

```text
D_Q = sum_{s satisfies Q} 1 / m_traj(s)(Q)
D_Q = M_Q * E[1 / m_traj(s)(Q) | s satisfies Q]
```

Here `M_Q` is the existing segment-level cardinality estimate after the normal
full-join and static fanout corrections, and `m_t(Q)` is the number of segments
inside trajectory `t` that satisfy the same generated query context. The model
therefore returns:

```text
D_hat_Q = M_hat_Q * traj_dedup_factor
```

`traj_dedup_factor` is not an ordinary physical column. It has no predicate
vocabulary, no Duet input token, no categorical/factorized output domain, no
ANPM, and no cross entropy. It is a scalar sigmoid head placed after all data,
indicator, and fanout columns in the autoregressive order:

```text
data -> indicator -> fanout -> traj_dedup_factor
```

The terminal degree allows it to condition on every existing query token,
including the final fanout token, while earlier AR heads keep their original
strict masks.

### Query-Local Trajectory Multiplicity

Training uses the current row-first Duet context exactly as generated for the
ordinary AR loss:

```text
sample one FOJ tuple
generate one row-satisfied query context Q
identify the anchor trajectory id, e.g. POL trip_id
compute local m_t(Q) for that trajectory only
train target y = 1 / m_t(Q)
```

The reusable `TrajectorySegmentIndex` groups encoded segment rows by trajectory
id and evaluates the same `GeneratedTrainingContext` semantics used by ordinary
training. `INV_FANOUT` tokens are correction potentials, not segment-selection
predicates. Unsupported non-wildcard predicates are skipped/fail closed instead
of being ignored.

The correction is applied only when the ordinary estimator is a matching-segment
measure. In POL terms, `segments` must participate in the query table subset.
For trip-only or agent/trip-level queries, the normal fanout correction may
already collapse segment multiplicity, so multiplying by `traj_dedup_factor`
would double-correct the estimate. Such contexts are skipped during training and
`estimate_distinct_trajectories(...)` raises a not-applicable error unless the
caller passes an eligible query/table context.

Trajectory predicates are classified by config. Agent/trip columns that are
constant within one trajectory are `trajectory_static_columns`; they are checked
as part of the row-satisfied context but do not reduce local segment
multiplicity. Columns that can vary among segments are
`segment_varying_columns`; only these ordinary predicates are evaluated inside
`m_t(Q)`. Indicator tokens and `INV_FANOUT` remain correction-only.

POL temporal predicates use workload semantics rather than generic dictionary
comparisons. Temporal overlap is:

```text
segments.t_s < query_upper AND segments.t_e >= query_lower
```

Physical POL spatial workload predicates use exact line-segment versus
axis-aligned rectangle intersection for the
`ST_Intersects(segment_geom, ST_MakeEnvelope(...))` case. The semantic payload is
carried on `GeneratedTrainingContext` as an optional `TrajectoryQuerySemantics`,
while the normal model still consumes ordinary Duet predicate tokens. The
current one-pass base segment estimator can condition on scalar endpoint
coordinates, but endpoint containment is not equivalent to physical
line/rectangle intersection.

The production predicate generator now creates one physical query decision and
derives both representations from it:

```text
sampled FOJ row
  -> GeneratedPhysicalQuery
     -> Duet PredicateToken[] for ResMADE
     -> TrajectoryQuerySemantics for local m_t(Q)
```

For POL temporal predicates, the same `lower` and `upper` bounds produce
`segments:t_e >= lower`, `segments:t_s < upper`, and the trajectory overlap
predicate. For POL spatial predicates, exact multiplicity/oracle utilities
retain the physical `ST_Intersects` rectangle for future extensions. Spatial
`D_hat = M_hat * traj_dedup_factor` is intentionally disabled in this branch:
training skips spatial trajectory targets with
`unsupported_base_segment_spatial_measure`, and inference fails closed for
spatial/spatio-temporal distinct trajectory queries until the base segment
estimator represents the same physical event.

The terminal trajectory loss uses the main-batch tuple importance correction
when present and multiplies every active `INV_FANOUT` reciprocal because the
head sits after all fanout columns:

```text
w_traj = rho * product_{r: T_r = INV_FANOUT} 1 / f_r
```

For numerical stability, the batch loss may subtract a per-batch maximum log
weight before normalizing. Run-level and validation diagnostics aggregate the
unscaled log weights in log space before reporting global weighted MSE and ESS.

The current ablation is intentionally single-anchor and query-only. It does not
create labels from global `COUNT(DISTINCT trajectory_id)`, does not enumerate
all matching segments as training examples, and does not correct the additional
row-first query-generator factor `G(Q | s)`. Exact distinct counts are used for
evaluation only. Future ablations can compare multiple anchors per query,
Q-first matching-segment sampling, or an explicit correction for `G(Q | s)`.

### POL Workload Evaluation Truth

For production POL workload evaluation, population-compatible truth comes from
the structured query records generated by `query_generation/` after execution
against PostgreSQL/MobilityDB:

```text
M_true = record["join_cardinality"]
D_true = record["entity_cardinality"]
a_true = D_true / M_true when M_true > 0, otherwise unavailable
```

These fields correspond to the database population modeled by the join
cardinality `|J|`. Local fixture oracles over `sample_rows.npy` remain useful
for deterministic semantic tests, but fixture counts are reported separately as
fixture counts and are not used for production q-error. When workload records do
not contain database cardinalities, production q-error is reported as
unavailable rather than silently falling back to fixture truth.

When `M_true = 0`, `D_true` must also be `0`; otherwise the record is rejected as
inconsistent. Empty population queries report `a_true: null` and
`a_abs_error: null` because the deduplication ratio is undefined for an empty
matching-segment set.

Evaluation CLIs reject stale trajectory checkpoints when the runtime
`trajectory_distinct` table/key/static/varying-column/SRID configuration or the
target semantics version differs from the checkpoint metadata. This prevents
mixing checkpoints trained under incompatible trajectory-target semantics.

### Validation Source

Training validation is explicit about its source:

```text
validation.prepared_directory unset -> same_fixture_resampled
validation.prepared_directory set   -> held_out_fixture
```

`same_fixture_resampled` reuses the training fixture with independent validation
RNG streams. `held_out_fixture` constructs a separate prepared sample source
from `validation.prepared_directory` and validates that its schema matches the
training metadata before any validation batch is used.

### POL Trajectory Preparation

The production POL path uses a compact memory-mapped segment index rather than
storing every encoded FOJ/model column for every segment. The directory layout is:

```text
trajectory_segment_index/
  manifest.json
  trajectory_ids.npy
  offsets.npy
  segment_idx.npy
  t_s.npy
  t_e.npy
  s_x.npy
  s_y.npy
  e_x.npy
  e_y.npy
```

`trajectory_ids.npy` and `offsets.npy` form a CSR index from `trip_id` to its
segments. The per-segment arrays store only POL segment-varying fields, with
timestamps converted to numeric epoch seconds. The loader memory-maps these
arrays and uses sorted numeric `trip_id` lookup instead of rebuilding a Python
dictionary per batch. The manifest hash includes schema hash, trajectory/segment
keys, static/varying column lists, SRID, temporal/spatial semantic versions, and
index format version; stale indexes fail closed.

Build the compact index from MobilityDB staging output:

```bash
python3 -m model.scripts.prepare_pol_trajectory_distinct \
  --config model/configs/pol_50m_traj_dedup_single_anchor.yaml \
  --segments-tsv dataset_generation/mobilitydb_loader/staging/segments.tsv
```

If `sample_rows.npy` is a fixture, aligned provenance must be produced by the
same materialization step that creates the sample rows and passed as:

```bash
python3 -m model.scripts.prepare_pol_trajectory_distinct \
  --config model/configs/pol_50m_traj_dedup_single_anchor.yaml \
  --segments-tsv /path/to/segments.tsv \
  --sample-provenance-tsv /path/to/sample_trip_segment_ids.tsv
```

The provenance TSV must align row-for-row with `sample_rows.npy` and provide
`trip_id` plus `segment_idx`. Fixture sources validate
`sample_trajectory_ids.npy` and `sample_segment_ids.npy` lengths at startup.
Live POL/NeuroCard sampling with trajectory distinct remains fail-closed until
the sampler emits provenance together with each sampled FOJ row.

The standard fixture producer is:

```bash
python3 -m model.scripts.prepare_pol_full_join_fixture \
  --config model/configs/pol_50m_traj_dedup_single_anchor.yaml \
  --full-join-tsv /path/to/pol_full_join_with_trip_segment_ids.tsv \
  --sample-rows 2048
```

The input TSV must contain `trip_id`, `segment_idx`, and all model metadata
column names. `sample_rows.npy`, `sample_trajectory_ids.npy`, and numeric
`sample_segment_ids.npy` with shape `[N, 2]` are written from the same sampled
rows. The compact index builder requires `segments.tsv` to be ordered by
`(trip_id, segment_idx)` and writes arrays sequentially with
`numpy.lib.format.open_memmap`, avoiding Python lists of all 50M rows and
avoiding a giant in-memory sort.

## Optimized ANPM Inference

The first faithful DistJoin-style ANPM implementation matched the equations but
was computationally inefficient for exact inference: it generated full
rank-one matrices and evaluated factorized predicates by enumerating original
IDs. The optimized class-space path keeps the same checkpoint and trained
weights, but evaluates predicate masses through factor prefixes whenever
possible.

NeuroCard-style systems often rely on progressive sampling for inference. This
project keeps deterministic sampling-free inference: exact predicate-specific
prefix algorithms replace full original-domain enumeration for supported
predicates, while arbitrary masks retain a correct fallback.

## Rank-One Contractions

The generated matrices are rank one:

```text
W_1(p) = a_1(p)b_1(p)^T
W_2(p) = a_2(p)b_2(p)^T
```

The implementation now applies them with:

```text
x(a b^T) = (x dot a)b^T
```

This removes the production allocation of `[batch,D,H]` and `[batch,H,D]`
matrices. It is checkpoint-compatible because the six generated vectors and all
trained parameters are unchanged.

## Unique Prefix Evaluation

During fallback enumeration, repeated factor prefixes are deduplicated within a
query. For a factorized domain such as `128 x 2048`, the second factor has at
most `128` first-factor prefixes, not one distinct hypernetwork call per
original ID. Prefix-conditioned distributions are cached only inside the
query-local evaluator and detached from autograd.

## Exact Equality and Range Inference

For equality predicates, the original dictionary ID is factorized and only that
single factor path is evaluated:

```text
P(X=v) = product_k P(Z_k=v_k | Z_<k=v_<k)
```

For one-sided ranges over factorized columns whose predicate semantics align
with encoded-ID order, the evaluator uses a most-significant-factor prefix CDF:
smaller current digits are accumulated with cumulative sums, and only the equal
digit continues to the next factor. Two-sided ranges use the same query-local
factor probability evaluator and compute interval mass as CDF differences
inside one normalized column distribution.

## Fallback Original-Domain Enumeration

If a factorized predicate cannot be represented by the optimized algorithms, or
if encoded-ID interval order would not preserve the original predicate
semantics, the adapter falls back to chunked original-ID enumeration. The
fallback still uses rank-one contractions and unique-prefix evaluation.

## Class-Space Versus Latent-Space ANPM

The current optimized mode is class-space ANPM: each factor head still has width
`D_k`, and existing class-space checkpoints load without conversion. A future
latent-space mode would move the ANPM transform into a smaller encoded output
space and project back to factor classes. That would be a new architecture and
would require fresh training and distinct checkpoint compatibility metadata.

## Factorized Inference Profiling

`TorchANPMFactorizedOutputAdapter.last_factorized_profile` records query-local
debug counters such as ANPM call count, largest ANPM batch, unique-prefix counts
by factor, and fallback usage. The JOB-light workload summaries still report
wall time, model latency, forward-call count, q-error, and status counts.

## Memory Reduction

For large domains, output width changes from the original domain size to the sum
of bit-factor domains. With the JOB-light template settings
`word_size_bits: 11` and `minimum_domain_size: 2048`, a column with millions of
IDs is represented by small factor heads instead of one flat million-way
softmax. The synthetic factorized smoke config intentionally uses tiny domains
for correctness and may not reduce width.

## Checkpoint Compatibility

Factorized checkpoints store original metadata, the full factorization plan,
output-head specs, ANPM configuration, model state, optimizer state, predicate
vocabularies, schema hash, and factorization hash. Legacy unfactorized
checkpoints remain loadable. Loading can reject a mismatched expected
factorization plan; factorized training requires a fresh checkpoint rather than
reusing unfactorized output-layer weights.

## Exact Autoregressive Masking Contract

For every column `i`:

```text
q_i(X_i | T_<i)
```

The output for `X_i` may depend on previous virtual tokens, but not on `T_i` or
future tokens. The torch tests include current-token and future-token leakage
checks, plus residual/direct-IO coverage when PyTorch is installed.

## Mathematical Formulation

The model represents a fixed full-outer-join distribution over columns:

```text
ordinary data columns, table-presence indicators, fanout columns
```

For each modeled column `X_i`, the predicate-conditioned autoregressive model
emits logits over the encoded domain `D(X_i)` and applies a separate softmax:

```text
q_i(v | T_<i) = softmax(z_i)_v
```

The current virtual token `T_i` is not part of the input to its own output head.
It is applied afterward as a known mask or potential.

For ordinary predicates and indicators:

```text
a_i = sum_{v in D(X_i)} q_i(v | T_<i) * 1[v satisfies T_i]
```

For wildcard tokens:

```text
a_i = 1
```

For an active fanout token:

```text
T_i = INV_FANOUT
a_i = sum_{f in D(F_i)} q_i(f | T_<i) * 1/f
```

This is the expected inverse fanout, not the inverse of the expected fanout.

Final cardinality is:

```text
|J| * product_i a_i
```

The implementation accumulates this in log space when configured:

```text
log |J| + sum_i log a_i
```

If any factor is zero, the estimate is explicitly zero.

## Weighted Cross Entropy

For row `b`, later fanout heads must be trained under the population reweighted
by earlier active inverse fanout potentials. For fanout head `F_j`:

```text
w_b^(j) = product_{r<j, T_r=INV_FANOUT} 1 / f_r^(b)
```

The current fanout does not weight its own loss. It starts affecting later
heads. Wildcard fanouts contribute one and do not enter the cumulative weight.

The normalized weighted objective is:

```text
sum_b w_b^(j) * [-log q_j(f_j^(b) | T_<j)] / (sum_b w_b^(j) + epsilon)
```

This makes the optimal softmax approximate the reweighted marginal:

```text
E[1(F_j=f) * product_{r<j} phi_r(F_r) | Q]
------------------------------------------------
E[product_{r<j} phi_r(F_r) | Q]
```

where `phi_r(f)=1/f` for `INV_FANOUT` and `phi_r(f)=1` for `WILDCARD`.
Thus weighted training amortizes the explicit reweighted-marginal calculation
that NeuroCard would otherwise handle through inverse-fanout marginalization
during progressive sampling.

For multiple active fanouts, the target quantity is:

```text
E[ product_j 1/F_j | Q ]
```

The local factors recover this when each later fanout head has learned the
correct reweighted marginal.

## Why The Model Does Not Use NeuroCard Progressive Sampling

NeuroCard estimates query cardinalities by autoregressive probabilistic
inference over full-outer-join samples and uses fanout corrections while
progressively sampling. This prototype instead feeds predicate tokens once,
computes all head distributions in one pass, and applies exact predicate masks
or reciprocal fanout masks to each output distribution.

Ordinary predicates and inverse-fanout potentials are amortized into
predicate-conditioned ResMADE heads. Inference for one query performs exactly one
model forward pass.

## Supported Assumptions

Initially supported:

- Acyclic join schemas.
- Equality joins.
- Connected query subgraphs.
- Table-presence indicators.
- Exact categorical fanout columns.
- `INV_FANOUT` and `WILDCARD` fanout tokens.
- One-pass predicate-conditioned inference.
- No progressive sampling.
- Lossless bitwise factorization for ordinary data columns.
- ANPM decoding for factorized ordinary data columns.

The included integration tests use a materialized synthetic full outer join for
oracle validation only.

## Installation

From the repository root:

```bash
python3 -m pip install -e ".[dev,model]"
```

For correctness-only tests without ResMADE, `.[dev]` is sufficient. ResMADE
training requires PyTorch. NeuroCard Exact Weight sampling additionally requires
the upstream NeuroCard environment, including its Rust/index preparation stack.

## Dataset Preparation

The synthetic dataset is built in code by:

```text
model/src/data/full_join_sampler.py
```

It is a tiny three-table chain with matched rows, unmatched full-outer-join
branches, duplicate join keys, and correlated fanouts.

For JOB-light, the real-data path is a NeuroCard-backed preparation step that
loads complete base-table dictionaries, complete join-key fanout metadata, and
an optional sampled full-outer-join fixture into
`data/neurocard_prepared/job_light/`. Smoke training can read the
manifest-backed fixture through `NeuroCardFullJoinSampleSource`; production
training should use live Exact Weight sampling once the external NeuroCard
sampler is connected.

## NeuroCard Sampler Preparation

The wrapper command is:

```bash
python3 -m model.scripts.prepare_neurocard_data \
  --config model/configs/job_light_resmade_factorized_anpm.yaml \
  --rebuild-domains \
  --sample-rows 512
```

For `dataset.type: neurocard_full_join`, the command validates CSV presence,
uses NeuroCard's JOB-light schema/join metadata, constructs complete domains,
samples validation rows, writes `manifest.json`, `sample_rows.npy`, and
`preparation_stats.json`, then reload-validates the artifacts. Without
`--rebuild-domains`, an existing complete manifest is validated and reused; an
old smoke-domain manifest fails with a rebuild hint.

The adapter boundary is `NeuroCardFullJoinSampleSource`. It expects the prepared
manifest to contain canonical `ModelMetadata` and join cardinality. It does not
estimate `|J|` from sampled row count.

## Predicate-Conditioned Training Contexts

Earlier ResMADE training used one fixed virtual token row for every sampled
tuple: ordinary columns were wildcard, every table indicator was forced to
`I_T = 1`, and every fanout column used `INV_FANOUT`. That meant query
predicate embeddings, wildcard indicator embeddings, and wildcard fanout
embeddings were not trained under the contexts used by JOB-light evaluation.

Training now uses `PredicateTrainingContextGenerator` to produce row-specific
query contexts. For every sampled full-join tuple, generated predicates must be
true for that tuple, included-table indicators must match rows where the table
is present, and fanout tokens are selected from the same included/excluded table
semantics used by evaluation.

## Row-Satisfied Predicate Generation

The generator supports a configurable mixture of wildcard, equality, lower-bound
range, and upper-bound range tokens under the current categorical predicate
vocabulary. Equality uses the sampled value. Lower-bound predicates choose a
domain threshold less than or equal to the sampled value and emit `>= threshold`.
Upper-bound predicates choose a threshold greater than or equal to the sampled
value and emit `<= threshold`. Sentinel values such as outer padding remain
wildcard unless explicit null predicates are added later.

## Connected Table-Subset Sampling

`predicate_generation.table_subset_sampling: connected` samples nonempty
connected subsets of the join graph inferred from fanout sources like `A->B`.
For a sampled row, candidate subsets are restricted to tables whose indicator is
`1`; the generator never trains an `I_T = 1` token against a row where `I_T = 0`.
Omitted table indicators are wildcard.

## Live NeuroCard Exact Weight Training

`sample_rows.npy` is a deterministic smoke or validation fixture, not a
production training population. The data layer now exposes explicit
`dataset.sampling_mode` values: `fixture`, `materialized_large_sample`, and
`live`. Fixture modes report `fixture_rows_reused`; live mode is reserved for a
NeuroCard `FactorizedSampler`/`FactorizedSamplerIterDataset` integration and
fails closed until those external artifacts are wired against the fixed
complete-domain manifest.

## Indicator and Fanout Token Semantics

Included table indicators use `EQUAL(1)` and omitted table indicators use
wildcard. For fanout columns, the child table determines the token: if the child
table participates in the query, the fanout is wildcard; if the child table is
omitted, the fanout uses `INV_FANOUT` to remove duplication. Cumulative
inverse-fanout weights still follow the upstream convention where previous
active fanouts weight later heads and the current fanout never weights its own
head.

## Predicate Token Coverage

Training writes `predicate_token_coverage`, `generated_predicate_contexts`,
`rejected_unsatisfied_contexts`, `fresh_sampler_rows`, and `fixture_rows_reused`
to `training_metrics.jsonl` and `training_summary.json`. These counters make it
visible when an evaluation token type was never observed during training.

## Native Multiple Predicates Per Column

JOB-light evaluation now groups SQL predicates by original database column and
normalizes each group before inference. `ColumnPredicateSet` supports wildcard,
single-predicate, equality, lower-bound, upper-bound, and two-bound interval
cases. It preserves inclusivity for closed, open, and half-open intervals and
detects contradictions such as incompatible equalities or empty ranges
explicitly. Contradictory query columns return a zero estimate with status
`zero_due_to_contradiction` rather than being silently overwritten.

The current trained checkpoints still use the categorical legacy input
vocabulary with one input token per original column. For compatibility, native
two-sided range evaluation conditions that column with `WILDCARD` and applies
the interval constraint to the output distribution from the same model pass.
The checkpoint-incompatible multi-slot predicate input encoder remains a later
training milestone.

## Native Two-Sided Range Estimation

Production JOB-light evaluation no longer estimates two-sided ranges by
subtracting two independently conditioned complete-query cardinalities. A range
such as `l < X <= u` is normalized into one interval token and evaluated as one
column factor from one ResMADE backbone state:

```text
P(l < X <= u | T_<i) = F(u | T_<i) - F(l | T_<i)
```

For factorized columns, both CDF terms share the same
`FactorizedColumnProbabilityEvaluator`, the same prefix cache, and the same
backbone output. For unfactorized columns, the adapter builds one decoded-domain
interval mask and sums the softmax probability mass under that mask.

## Why External Cardinality Subtraction Was Removed

The old inclusion-exclusion path computed two full cardinality estimates under
two different predicate-token rows and subtracted them. JOB-light query 20 made
the failure concrete:

```text
true_cardinality = 695701
C_upper_hat      = 144842.47
C_lower_hat      = 211711.12
C_lower_hat > C_upper_hat
```

Clamping `C_upper_hat - C_lower_hat` to zero produced an epsilon Q-error of
about `6.957e17`. The replacement only subtracts CDF values within one
normalized column distribution, so monotonicity is enforced at the probability
mass level instead of assumed between independent full-query estimates.

On the corrected fixed-fixture checkpoint, native evaluation scored all
`70/70` JOB-light queries with `70` model forward calls. Query 20 changed from
the clamped-zero failure to an estimate of `28724.26` with Q-error `24.22`.

## Range Bound Resolution

Range thresholds do not need to be dictionary literals. Bounds are resolved
against decoded comparable domain values, excluding SQL null sentinels,
outer-padding sentinels, and incomparable values from ordered comparisons.
Interval mass is checked with a `1e-7` tolerance: tiny numerical drift is
clamped, while larger negative or greater-than-one masses raise an explicit
diagnostic error.

## Compositional Predicate Encoding

The shipped training configs remain in
`predicate_encoding.mode: categorical_legacy`. The compatibility path stores
the requested `predicate_encoding` configuration surface, but the architecture
still allocates independent categorical embeddings for `EQUAL(v)`,
`LESS_EQUAL(v)`, and `GREATER_EQUAL(v)`. A future
`predicate_encoding.mode: compositional` checkpoint should share operator,
value, factor-digit, and range-rank representations rather than allocating a
separate learned vector for every operator/literal pair.

## Early Stopping Monitor

`training.validation_interval_steps` does not turn validation on; only
`validation.enabled` does. Until this was guarded, a config could ask for early
stopping every 250 steps, get no validation source, and silently monitor the
per-step training loss instead -- selecting a checkpoint off batch noise with
nothing in the summary to say so. One production run lost a checkpoint that
way: its best training loss landed 2,750 steps before the end, while every tail
q-error metric was still improving at the final step.

`training.early_stopping_monitor` makes the choice explicit. It is `validation`
(the default whenever `validation.enabled` is true) or `train_loss`. Early
stopping with validation disabled and no declaration is now rejected rather
than silently downgraded, and `early_stopping_monitor: validation` with
validation disabled is rejected too. `resolve_early_stopping_monitor` in
`model/src/config.py` is the single resolution rule the validator and the
trainer both read, so they cannot drift.

`validation.minimize` describes the validation selection metric, so it is not
applied when the monitor is `train_loss`: the training loss is always
minimized, and a config with `minimize: false` must not come to treat a rising
loss as progress.

Two things this does not fix. Watching `train_loss` is still a weak signal, so
a config that declares it is declaring a limitation, not a configuration. And
the available selection metrics -- `validation_nll`, `validation_weighted_nll`,
`validation_traj_weighted_mse` -- are mean-behaviour metrics; where the uniform
term dominates the loss they track the median far better than the tail, so
enabling validation does not by itself make checkpoint selection tail-aware.

## Validation and Best-Checkpoint Selection

The trainer records interval metrics and checkpoints, but full validation NLL,
held-out JOB-light query evaluation, and `checkpoint_best.pt` selection are not
yet implemented. The new configuration fields document the intended validation
contract for the next training milestone.

## Zero-Estimate Q-Error Reporting

JOB-light query-level CSVs now include both the continuity metric
`q_error_epsilon` and the supplementary `q_error_floor_one`, plus a
`zero_estimate` flag. Summary JSONs include `zero_estimate_count` and
floor-one percentiles so pathological zero predictions are visible without
changing the historical headline epsilon Q-error.

## JOB-light Tail Diagnostics

The JOB-light evaluator records per-query status, true and estimated
cardinality, epsilon and floor-one Q-error, zero-estimate status, native range
count, whether legacy external subtraction was used, predicate/table counts,
predicate columns/operators, inverse-fanout columns, per-column probability
factors, log factors, model forward calls, and latency. Summary JSONs include
status counts, normal-query and native-range groups, and the worst 20 queries
with their diagnostics. Finer bucketed summaries by all tail dimensions remain
planned.

## Complete Domains Versus Smoke Samples

For JOB-light, `manifest.json` is the model schema contract and
`sample_rows.npy` is only a small training or validation fixture. The manifest
domains must be built from the complete JOB-light base tables and join metadata,
not from the sampled join rows. Changing `--sample-rows` can change the fixture
shape, but it must not change ordinary domains, indicator domains, fanout
domains, schema hashes, or factorization mappings.

The preparation path now uses complete source-table columns for ordinary data
domains, adds the canonical outer-padding token separately, writes explicit
indicator domains `(0, 1)`, and derives fanout domains as dense positive ranges
`1..f_max` from complete join-key frequencies. The neutral outer-join fanout is
`1`; zero or negative fanouts fail validation.

Factorization only becomes useful once these complete high-cardinality domains
are present. Old smoke-domain manifests used only 512 sampled rows, so every
ordinary domain stayed below `factorization.minimum_domain_size: 2048` and the
factorization plan had no selected columns. Old checkpoints trained against a
smoke manifest are incompatible with a rebuilt complete manifest because output
dimensions, predicate vocabularies, schema hashes, and factorization mappings
can change.

## Regenerating the JOB-light Manifest

On the cluster, archive old smoke artifacts if they are present:

```bash
cd /work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest
stamp=$(date +%Y%m%d%H%M%S)
mkdir -p data/neurocard_prepared/job_light/archive_$stamp
mv data/neurocard_prepared/job_light/manifest.json data/neurocard_prepared/job_light/archive_$stamp/ 2>/dev/null || true
mv data/neurocard_prepared/job_light/sample_rows.npy data/neurocard_prepared/job_light/archive_$stamp/ 2>/dev/null || true
mv data/neurocard_prepared/job_light/preparation_stats.json data/neurocard_prepared/job_light/archive_$stamp/ 2>/dev/null || true
```

Rebuild complete domains and encode a 512-row validation fixture:

```bash
python3 -m model.scripts.prepare_neurocard_data \
  --config model/configs/job_light_resmade_factorized_anpm.yaml \
  --rebuild-domains \
  --sample-rows 512
```

Inspect the prepared metadata and factorization reduction:

```bash
python3 -m model.scripts.inspect_sampler \
  --config model/configs/job_light_resmade_factorized_anpm.yaml \
  --sample-rows 2
```

The rebuilt cluster manifest reported `original_output_width=374732`,
`factorized_output_width=9915`, and reduction ratio `0.026459`. The selected
factorized columns were `movie_companies:company_id` with factor domains
`(128, 2048)` and `movie_keyword:keyword_id` with factor domains
`(128, 2048)`.

Run the short JOB-light factorized training smoke:

```bash
python3 -m model.scripts.train_resmade \
  --config model/configs/job_light_resmade_factorized_smoke.yaml
```

## Sampler Inspection

```bash
python3 -m model.scripts.inspect_sampler \
  --config model/configs/resmade_smoke.yaml
```

This prints join cardinality, column order, column types, domain sizes,
indicator frequencies, summarized fanout domains/min/max, padded-row
percentages, decoded sample rows, factorization selection, output-width
reduction, and estimated parameter size when PyTorch is available.

## Training Command

Correctness prototype:

```bash
python3 -m model.scripts.train_synthetic \
  --config model/configs/inv_fanout_baseline.yaml \
  --checkpoint model/examples/synthetic_checkpoint.json
```

ResMADE smoke:

```bash
python3 -m model.scripts.train_resmade \
  --config model/configs/resmade_smoke.yaml
```

JOB-light template:

```bash
python3 -m model.scripts.train_resmade \
  --config model/configs/job_light_resmade_inv_fanout.yaml
```

Synthetic factorized ANPM smoke:

```bash
python3 -m model.scripts.train_resmade \
  --config model/configs/resmade_factorized_smoke.yaml
```

JOB-light factorized ANPM template:

```bash
python3 -m model.scripts.train_resmade \
  --config model/configs/job_light_resmade_factorized_anpm.yaml
```

The production JOB-light factorized template currently uses
`batch_size=2048`, `steps_per_epoch=3500`, and therefore sees
`7,168,000` nominal sampled tuples in one epoch. Training summaries report both
`total_sampled_tuples` and `nominal_rows_seen`, plus parameter counts, model
size, training time, per-column losses, factor losses, and fanout-head
effective sample size.

## Evaluation Command

Correctness prototype:

```bash
python3 -m model.scripts.evaluate_synthetic \
  --config model/configs/inv_fanout_baseline.yaml \
  --checkpoint model/examples/synthetic_checkpoint.json
```

ResMADE:

```bash
python3 -m model.scripts.evaluate_resmade \
  --config model/configs/resmade_smoke.yaml \
  --checkpoint model/runs/resmade_smoke/checkpoint_step_200.pt
```

Factorized ResMADE:

```bash
python3 -m model.scripts.evaluate_resmade \
  --config model/configs/resmade_factorized_smoke.yaml \
  --checkpoint model/runs/resmade_factorized_smoke/checkpoint_step_20.pt
```

## Exact Oracle Command

```bash
python3 -m model.scripts.exact_oracle
```

This command runs a tiny synthetic oracle only. It scans the materialized
synthetic full outer join and prints exact quantities used to validate
`INV_FANOUT`, table indicators, predicate masks, and weighted fanout
corrections. It does not train a model, use JOB-light, query MobilityDB, or
invoke NeuroCard.

## Minimal Synthetic Example

The two-fanout exact test uses:

```text
P(F1=1,F2=1)=0.5
P(F1=10,F2=10)=0.5
```

Directly:

```text
E[1/(F1*F2)] = 0.5*1 + 0.5*0.01 = 0.505
```

Separate ordinary marginals give the intentionally wrong value:

```text
E[1/F1] * E[1/F2] = 0.55 * 0.55 = 0.3025
```

The reweighted second marginal restores the correct product.

## Configuration Reference

Baseline config:

```text
model/configs/inv_fanout_baseline.yaml
```

ResMADE configs:

```text
model/configs/resmade_inv_fanout_example.yaml
model/configs/resmade_smoke.yaml
model/configs/job_light_resmade_inv_fanout.yaml
model/configs/resmade_factorized_smoke.yaml
model/configs/job_light_resmade_factorized_smoke.yaml
model/configs/job_light_resmade_factorized_anpm.yaml
```

Important settings:

- `model.type`: `prototype_table` for the lightweight scaffold or
  `predicate_resmade` for real training.
- `model.hidden_sizes`: ResMADE hidden width/depth.
- `model.residual_connections`: enables masked residual blocks.
- `model.direct_io_connections`: enables masked direct input-output path.
- `model.input_encoding`: `embed` or `one_hot`.
- `model.column_order`: `data_indicators_fanouts`.
- `factorization.enabled`: enables lossless ordinary-column bit factorization.
- `factorization.word_size_bits`: maximum bit width per factor head.
- `factorization.minimum_domain_size`: minimum original domain size considered
  for factorization.
- `anpm.enabled`: required when factorization is enabled.
- `anpm.decode_chunk_size`: valid original IDs accumulated per decoding chunk.
- `fanout.compute_weights_in_log_space`: builds cumulative inverse weights in
  log space.
- `fanout.weight_clipping`: `null` by default. Clipping would bias the target
  distribution and is not applied.
- `inference.progressive_sampling`: must be `false`.
- `inference.use_log_space_product`: controls log-space product accumulation.

Startup validation fails for incompatible settings, including factorization with
direct input-output connections.

## Conjunctive Rare-Support Strata

A rare-support stratum is a region of the full outer join that the predicate
generator almost never produces, so the auxiliary batch oversamples it. Until
now a stratum was a single column region, which made the deficit objective a
*marginal* one: a stratum could report a large expected support while the joint
prefix that actually starves the model had almost none. Reported per-head
auxiliary effective sample sizes in the tens of millions were therefore
measuring the wrong quantity.

A stratum is now a conjunction of per-column terms. `RootDataStratum.terms` is
the full conjunction, ordered by column index, and the legacy scalar fields
(`column_index`, `region_type`, `value`, `lower`, `upper`) describe its last
term, so depth-1 strata and every existing consumer behave exactly as before.
Scoring changed with it: `conditional_head_support` is the expected number of
contexts in which *every* term is emitted as a predicate, every table the
conjunction touches survives NeuroCard table dropout along with the scored
head's own table, and the head sits strictly after the last term. Surviving a
table means surviving every table between it and the join root, because the
sampler runs the subset through the root-connected component: on a chain
`root -> parent -> child`, keeping the child requires the parent too. Counting
only the explicitly required tables overestimates survival on any join deeper
than one hop, so the required set is closed over its root paths first, using
the same `root_paths_closure` the context generator uses. A stratum's
`support_score` is the minimum over the heads it can serve and its
`support_deficit` is the summed shortfall against the threshold, so a
conjunction that starves three suffix heads outranks a marginal that starves
one. The heads it serves are every original column after its last term,
indicator and fanout heads included, matching `auxiliary_eligibility_mask`
exactly: the two functions state the same contract and live side by side in
`model/src/data/strata.py` so they cannot drift apart. Scoring only the DATA
heads would leave out the fanout heads whose effective sample size motivated
rare-support training in the first place.

Each term also carries a realization route and its probability: which operator
the region is rarest through, and the chance that a row already inside the
region emits exactly that predicate through it. For an equality region realized
as an equality that is just `p_equality`, because the row determines the
literal. For a tail or a native range the generator must also pick that exact
threshold or interval out of the ones the row admits, and on a
high-cardinality ordered column that selection probability is small. Leaving it
out would overestimate support by the mass-weighted mean number of admissible
thresholds, which is how a rare tail predicate ends up ranked as sufficiently
covered. The factor is derived from the depth-1 `expected_<family>_count`
statistics, and a non-equality term that reaches scoring without one raises
rather than falling back to the bare operator probability.

The route matters beyond scoring, because it decides which predicate the
auxiliary context actually forces: an equality region whose native-range
support is lower than its equality support is rare as `[v, v]`, not as `= v`,
and forcing the wrong one trains a different event from the one discovery
selected. The route therefore lives on the term rather than in the stratum's
`support_bottleneck` string, which conditional rescoring overwrites and which
cannot describe more than one term of a conjunction anyway.

Support itself is the realized support mass: the expectation over full
outer-join rows of the indicator that the row is in the region times the
product of the per-row realization probabilities. For a single term that is the
region mass times the per-row factor, which is exactly the depth-1 statistic.
For a conjunction it is **not** the product of the terms' marginal averages:
a tail or range factor depends on the row's rank, and the other terms change
the rank distribution inside the conjunction, so correlated terms would be
over- or underestimated. The provider already holds the materialized rows and
their weights, so it averages the product of the per-row factors over the joint
rows instead, which is exact. A conjunction scored without that mass raises
rather than falling back to the independence product. `per_value_realization_factors`
is the per-row form of the same quantities the depth-1 vectors compute, and a
test asserts the two agree for every route at every rank.

Forcing a term pulls its table into the query subset, and with it every table
on the path from the join root. In a chain `root -> parent -> child`, forcing a
predicate on the child when table dropout removed the parent would otherwise
leave `{root, child}`, which is not a rooted query and produces meaningless
indicator and fanout tokens. If the row does not carry every table on that
path, the context degrades to all wildcards instead.

Candidates are found by beam search over the depth-1 seeds, with exact joint
counts taken from the provider's materialized rows. Regions whose joint row
count falls below `minimum_joint_rows` are pruned, because oversampling a
region with a handful of rows means resampling those same rows every step.

Seeds and extension terms come from different pools. A depth-1 stratum has to
be imposable on the live path, which filters the root join-count table, so
seeds stay root-only exactly as before. A conjunction is only ever sampled
from materialized rows, so its extension terms are drawn from every DATA
column. With a root-only extension pool, `require_distinct_tables` rejects
every extension and discovery finds nothing at all, which is inert rather than
conservative. The extension pool honours the same
`max_candidate_domain_size` cap as the root pool.

Configuration lives under `rare_support.discovery.conjunctive`:

- `enabled`: turns conjunctive discovery on. Default `false`, which keeps the
  previous depth-1 behaviour exactly.
- `max_depth`: maximum number of terms per stratum; must be at least `2`.
- `beam_width`: candidates carried forward per search level.
- `seed_width`: depth-1 strata used to seed the search.
- `minimum_joint_rows`: joint row floor below which a region is pruned.
- `extension_width`: how many extension candidates the search may draw terms
  from, across all DATA columns. Defaults to `seed_width`. The width is applied
  only after candidates that cannot extend any seed are dropped -- those on a
  column a seed already pins, or on a table a seed already covers when distinct
  tables are required. Ranking first would let high-scoring but unusable
  root-table regions consume the whole budget, which on a real root table with
  many candidate regions makes discovery inert in the same way a root-only pool
  does. `scored_extension_candidates` and `eligible_extension_candidates` in
  `conjunctive_status` report both sides of that filter.
- `require_distinct_tables`: when true, each term must come from a table not
  already used by the conjunction, which targets the cross-table prefixes that
  single-column strata cannot express.
- `max_stratum_share`: caps any one stratum's share of the auxiliary batch. The
  surplus is water-filled toward the cap across the remaining strata rather
  than renormalized away, so capping a dominant conjunction does not silently
  reorder the strata underneath it. A cap at or below `1/n` is unsatisfiable
  and is ignored.

Every auxiliary context pins the complete conjunction. Pinning only part of it
would be unsound here: the row is drawn from the whole conjunction, so the
omitted terms stay in the targets and the heads on those columns get trained
toward `p(row | complete conjunction)` while the model is shown a partial one.
Representing partial conjunctions honestly means drawing the rows from the
partial stratum, which is a decision that has to be made before sampling, not
during context generation. `rare_auxiliary.term_subset_probability` is rejected
by config validation rather than ignored.

### Capability gate

Imposing a conjunction requires a source that can condition on several columns
at once. `ExactRootStratumProvider` can do this only when it holds materialized
encoded rows, which is the fixture and prepared-directory path. The live
NeuroCard sampler imposes a stratum by filtering the root join-count table one
column at a time (`model/src/data/full_join_sampler.py`), so it cannot sample a
conjunction that spans non-root tables and reports
`supports_conjunctive_strata = False`.

When conjunctive discovery is requested on a source that cannot serve it,
discovery falls back to depth-1 and records why in `conjunctive_status`, which
is surfaced in the rare-support summary and printed once at startup. It is a
loud fallback, not a silent one.

`conjunctive_status.applied` means conjunctions are actually present in the
selection, not merely that the search ran: a run where none were found or none
ranked into `max_selected_strata` reports `applied: false` with a reason naming
the cause, alongside `extension_candidates` and
`cross_table_extension_candidates` so an empty cross-table pool is visible
rather than inferred. Reporting `applied: true` with no reason while every
selected stratum is depth-1 is how a run can look like it exercised the feature
when it did not. In practice this means the JOB-light configs
that run `sampling_mode: live` keep the previous depth-1 behaviour; a run that
needs conjunctions has to use a fixture or prepared-directory source.

`model/configs/job_light_duet_binary_native_anpm_20k_conjunctive_rare_auxiliary_fixture.yaml`
is that source for JOB-light, and the only JOB-light configuration that can
exercise conjunctive discovery at all. Because fixture mode trains on a fixed
materialized sample rather than fresh draws from the join, its q-errors answer
"does the feature work end to end", not "does the feature help" -- comparing it
against a live run confounds the sampling mode with the feature. A test asserts
that at least one JOB-light config can reach conjunctions, so this cannot
silently revert to a configuration that only looks like it exercises them.

It also does not run against the prepared fixture as shipped.
`prepare_neurocard_data` writes `sample_rows.npy` at `dataset.sample_batch_size`
rows, 512 by default, and returns early when a manifest already exists, so the
prepared JOB-light directory holds 512 rows against a 2.1e12-row join. Rarity
estimated from 512 rows is not rarity, and `minimum_joint_rows: 100` against it
would have required a "rare" conjunction to cover a fifth of the sample. The
preconditions below make that refuse to start rather than produce counts that
look exact; the config states the fixture it needs and the command that builds
it. Live cross-table conditional sampling remains the only way to get
conjunctions without a large materialized reservoir, and is not implemented.

### Discovery preconditions

Conjunctive discovery estimates joint support from the materialized rows, so
several things have to hold before the numbers mean anything. All of them are
checked at discovery time and refuse the run rather than degrade it:

- `minimum_materialized_rows`: the row count the fixture must hold. The error
  names the row count found and the `prepare_neurocard_data --rebuild-domains
  --sample-rows N` command that fixes it. Zero disables the check, which is
  what a deliberately tiny smoke fixture declares.
- `max_joint_row_fraction`: refuses a `minimum_joint_rows` that is a large
  fraction of the available rows, because a region that common is not rare and
  the search would only return regions the model already sees. Defaults to 5%.
- `required_extension_columns`: column names that must be in the extension
  pool. The error says which of `max_candidate_domain_size` and
  `candidate_column_names` excluded them, so "the column carrying the tail is
  not in the pool" is distinguishable from "no conjunction was found". On
  JOB-light the tail lives on `movie_keyword.keyword_id`, whose ~134k domain a
  4096 cap silently excludes.
- `require_conjunctions`: makes selecting no conjunctive stratum a failure
  rather than a logged fallback, for a run whose purpose is to exercise them.
  It also covers the oldest fallback of all: a provider with no materialized
  rows raises immediately instead of quietly returning depth-1 strata.
- `max_candidates_per_column` (under `discovery`, not `conjunctive`): bounds a
  categorical column to its *marginally* rarest regions. A categorical column
  yields one equality region per observed value, so at 2M rows a 134k-domain id
  column yields ~134k of them. Ordered columns are never trimmed, because their
  tail regions are cumulative and dropping values would move the thresholds.
  This is a prior, not the selection mechanism -- see below.
- `conditional_extension_width`: how many conditionally-rarest values each
  seed contributes per extension column. `0` disables conditional ranking and
  leaves the marginal pool as the only route, which is useful for isolating
  that path and is not a sensible default.

### Why extensions are ranked conditionally

Marginal rarity is the wrong objective for an extension term. A value can be
common across the corpus and almost absent alongside a particular seed -- a
keyword that appears everywhere except in the latest production years -- and
that dependence is the entire reason a conjunctive stratum exists. Ranking
extensions by marginal count, or capping a column to its marginally rarest
values, drops exactly those values before the beam search can see them.

So the extension pool is filled from conditional candidates first, topped up
from the marginal pool only if slots remain. For each seed and each extension
column, one `bincount` of that column over the rows the seed matches gives the
joint count for *every* value of that column at once; the rarest of those that
still clear `minimum_joint_rows` become candidates. Conditioning therefore
costs one pass per (seed, column) rather than one per candidate pair, and the
count it returns is the exact joint count the beam would have computed anyway.
Across seeds a value keeps its smallest joint count, since being rare with any
one seed is what makes it a tail.

Join padding -- `OUTER_MISSING`, SQL null, `None` -- is excluded before the
ranking rather than after it, in both the conditional and the marginal path. A
full-outer-join child column is largely padding, and on many seeds the padding
is the rarest encoded id, so ranking first and discarding afterwards spends
candidate slots on values no builder will emit and can leave a column with no
candidates at all while real values were available.

Literals are never seeded from the benchmark workload. Pinning the literals a
failing query happens to use would make discovery depend on the test set, which
is the leakage this whole mechanism is built to avoid; `required_extension_columns`
names columns, not values.

`required_extension_columns` is checked twice: once against the candidate pool
before the search, and once against the strata actually selected. Reaching the
pool is not the same as surviving capping, ranking, the beam and the final
deficit sort, and a run whose purpose is a `movie_keyword:keyword_id`
conjunction should not pass by selecting conjunctions that use something else.
Only conjunctive strata count as coverage; a depth-1 stratum on the same column
says nothing about the joint region. `conjunctive_status` reports
`required_columns_covered` and `required_columns_uncovered` either way, and
`require_conjunctions` turns an uncovered column into a failure.

That failure names which stage lost the column, because the stages need
opposite fixes. `required_column_diagnosis` distinguishes a column whose every
pairing was rejected by `minimum_joint_rows` -- reporting the largest joint
counts any pairing reached and the fixture size that would be needed to clear
the floor at each of them -- from one that was discovered and then out-ranked
in the final deficit sort.

`fixture_rows_for_n_conjunctions` matters more than the single figure next to
it. Growing a fixture until the best pairing squeaks past the floor buys one
stratum; a usable experiment needs a number of them, and the rejected counts
are usually flat near the top, so the fixture needed for ten or fifty can be
far larger than the fixture needed for one. Widening `extension_width` or `max_selected_strata`
cannot help in the first case, because the pairings were never candidates.

Note what the first case usually means. A fixture is a sample of the join, so a
joint cell's count in it scales with the sample. On a 2M-row fixture of a
2.1e12-row join, a pairing of a 150-value year column with a 134k-value keyword
column has well under one expected row per cell, so no pairing clears a floor
of 100 and the counts that do appear are sampling noise rather than evidence of
rarity. Lowering the floor in that situation does not find rare regions; it
finds regions the fixture failed to resolve. The resolvable range at a given
fixture size is roughly `rows / (floor * d1 * d2)` >= 1, which at 2M rows and a
floor of 100 means the two domains multiplied must stay under about 20,000.

### The deficit objective

`support_deficit` sums, over the heads a stratum serves, how far each head's
support falls short of `minimum_expected_context_support`. With the default
`clipped_sum` a head contributes `threshold - support`, which is capped at
`threshold`, so once support is far below the threshold the sum sits at
`heads * threshold` and stops distinguishing a stratum from one a thousand
times rarer. That regime is every stratum this feature targets, so ranking
collapses onto the served-head count -- which, since heads are the columns
after the stratum's last term, is the column's position in the schema. A
conjunction on a later column then cannot outrank one on an earlier column
however much rarer it is.

This is observable, not hypothetical. On JOB-light with equal rarity the
deficits run `cast_info:role_id` (7 heads) > `movie_info:info_type_id` (6) >
`movie_keyword:keyword_id` (5) > `movie_companies:company_id` (4), and a run
selected 42 `info_type_id` and 11 `role_id` conjunctions and none from the
other two.

`rare_support.discovery.deficit_objective: log_ratio` contributes
`log(threshold / support)` per deficient head instead. It keeps growing as
support shrinks, so a stratum several orders of magnitude rarer can outweigh a
head or two: at supports of 0.03 against 0.00003, `clipped_sum` scores the
six-head stratum 599.8 against 500.0 and `log_ratio` scores them 48.2 against
74.6, reversing the order. Zero support is bounded by `epsilon` rather than
diverging. The default stays `clipped_sum` so no existing run changes
behaviour; `conjunctive_status.deficit_objective` records which one ran.

Note that `log_ratio` also compresses the spread between deficits, so the
`alpha` allocation derived from them is more uniform across strata. That is a
behaviour change in its own right and a reason to compare the two rather than
switch blindly.

### Auditing discovery without training

`model/scripts/inspect_rare_support.py --config <config>` runs discovery alone
-- no GPU, no optimizer, no training loop -- and prints the pool sizes, the
selected strata by depth, the required-column coverage and the diagnosis, then
writes the whole report as JSON. `--allow-uncovered` turns off
`require_conjunctions` for the audit so the diagnosis can be read without the
run stopping. On a cluster this turns a configuration assumption from a queue
cycle into a couple of minutes on a login node.

Column names in `candidate_column_names` and `required_extension_columns` are
model columns, spelled `table:column`. The dotted `table.column` form is the
NeuroCard root join-count dataframe's spelling: it looks right, matches no
model column, and would leave the pool empty. Config validation rejects it.

### Discovery cost

Term membership depends only on the encoded id, so it is decided once per
domain value and then gathered, rather than decoding every row to a Python
object and testing it. `joint_count`, `realized_support_mass` and
`membership_matrix` all work this way, the per-term lookups and realization
factors are cached on the provider, and the table-survival factor is memoized
because scoring calls it once per candidate per served head. Measured on a
2M-row, 10-column fixture with a 134k-domain keyword column,
`beam_width = extension_width = 64` and `conditional_extension_width = 8`, full
discovery takes about 105 seconds, of which roughly a minute is the conditional
passes.
The per-row form it replaced needed roughly that long for 200 of the 4096 beam
pairs. Tests assert the vectorized counts equal a direct row scan and that a
batch of calls stays inside a time budget.

Two filters decide which columns can carry a stratum at all, and both are easy
to set wrongly in a way nothing else reveals. `include_categorical_root_data`
skips every root column whose semantic type is `categorical`, and
`_semantic_for_column` defaults to `categorical` for any column
`root_column_semantics` does not name -- so a config naming only
`production_year` leaves that as the *only* seed column in existence, whatever
else the schema holds. `candidate_column_names` is an allowlist and hides a
column just as completely. `conjunctive_status` therefore reports
`candidate_columns`, `columns_excluded_by_name` and
`columns_skipped_by_domain_cap`, so "the column that carries the tail is not in
the pool" is visible rather than inferred. That combination is what made a
JOB-light run select only `production_year` pairings while the three columns
enriched 7-9x in its own tail -- `title:season_nr`, `title:episode_nr`,
`title:series_years` -- were filtered out before discovery ran.

`candidate_column_names` now also applies on the materialized path, not only
the root-JCT one. Restricting the pool by name is what lets a fixture raise
`max_candidate_domain_size` high enough to include a 134k-domain keyword column
while still keeping key columns such as `title:id` out of it. A wide domain
still costs one candidate per *observed* value, which at 2M rows is very nearly
the whole domain; `max_candidates_per_column` and the per-seed conditional
extensions below are what bound that.

### Diagnostics

`prefix_head_coverage` in the rare-auxiliary diagnostics reports, per stratum,
its depth, how many auxiliary contexts it produced, how many of those forced
the complete conjunction, how many were dropped because a required table was
absent from the sampled table subset, the histogram of how many terms were
forced, and which heads the resulting contexts actually cover. A context that
degraded to wildcards is counted under `incomplete_forced_context_count` and
not as a `generated_exact_support_event_count`, so amplification and coverage
do not read better than they are. This is the
quantity to watch when a joint prefix is suspected of starving: a global
auxiliary effective sample size says nothing about it.

## Factorization Status

The baseline and unfactorized configs leave factorization disabled. The
factorized smoke and JOB-light configs enable it explicitly:

```yaml
factorization:
  enabled: true
  strategy: bitwise_lossless
  word_size_bits: 11
  minimum_domain_size: 2048
  blacklist_columns: []
  blacklist_kinds: [indicator, fanout]

anpm:
  enabled: true
  previous_factor_embedding_size: 64
  hidden_size: 64
  mask_invalid_combinations: true
  decode_chunk_size: 4096
  final_activation: relu
```

`IdentityOutputAdapter` handles unfactorized outputs.
`ANPMFactorizedOutputAdapter` decodes factorized torch outputs back to
original-column semantics through chunked valid-ID enumeration. On the rebuilt
complete-domain JOB-light manifest, inspection selected
`movie_companies:company_id` and `movie_keyword:keyword_id`, reducing output
width from `374732` to `9915`.

## Testing

Run all tests from the repository root:

```bash
python3 -m unittest discover
python3 -m pytest
```

The tests cover logit slicing, predicate masks, reciprocal fanout masks,
expected inverse fanout, cumulative weights, wildcard exclusion, weighted cross
entropy, checkpoint metadata, complete-domain preparation, manifest validation,
sample-size-independent domains, OOD literal classification, lossless
factorization round-trips, invalid factor tuples, output-width reduction for a
large synthetic domain, exact two-fanout arithmetic, synthetic oracle cases, and
a deterministic training smoke.

When PyTorch is installed, additional tests cover ResMADE output width, separate
input/output bins, per-column softmax, residual shape preservation through the
forward/backward path, checkpoint save/load, current/future token leakage,
factor-head masking, ANPM prefix dependence, grouped factorized loss gradients,
factorized adapter normalization, factorized checkpoint reload, and CUDA smoke
when CUDA is available.

## Checkpoint Contents

ResMADE checkpoints include:

- Model state dictionary.
- Optimizer state.
- Epoch and step.
- ResMADE configuration.
- Resolved project configuration.
- Schema and column metadata.
- Factorization plan and factorization hash.
- Output-head specifications.
- ANPM configuration for factorized checkpoints.
- Predicate vocabularies.
- Output slices.
- Join cardinality.
- Factorization configuration.
- Preparation manifest identifier.

Loading can reject incompatible schema hashes.

## Evaluation Metrics

The in-repo ResMADE evaluation command reports:

- Estimated cardinality.
- Inference latency.
- Backbone forward time.
- ANPM decode time.
- Model forward-call count.

For the synthetic dataset it also reports true cardinality and q-error from the
exact oracle. JOB-light query-workload summaries with median/p90/p95 q-error
are produced by `model/scripts/evaluate_job_light_queries.py` in the cluster
environment that has the JOB-light CSVs and checkpoint artifacts.

The training utilities expose fanout-head weight statistics, including
effective sample size:

```text
(sum_b w_b)^2 / sum_b w_b^2
```

## Known Limitations

- The NeuroCard adapter is a manifest/sample-source boundary; this repository
  does not vendor NeuroCard's full Rust/index preparation stack, and live
  Exact Weight sampling still needs the external NeuroCard sampler connection.
- Complete JOB-light preparation and smoke training have been run on the shared
  cluster; reproducing them still requires the cluster dataset files and
  upstream NeuroCard checkout/environment.
- Local validation in this workspace skipped torch-specific tests because
  PyTorch is not installed here.
- Direct input-output connections are intentionally disabled in factorized mode
  until dedicated factor-head masks are implemented.
- Indicators and fanouts are intentionally not factorized; only ordinary data
  columns are factorized.
- Native two-sided range evaluation is implemented for JOB-light inference, but
  existing categorical checkpoints condition the ranged column with wildcard.
  Training a true multi-predicate input encoder is still checkpoint
  incompatible future work.
- Compositional predicate encoding and real validation/checkpoint selection are
  planned follow-up milestones.
- Factorized checkpoints require fresh training.
- Evaluation against stored JOB-light query workloads is still a cluster-side
  workflow rather than a self-contained local command in this repository.

## Troubleshooting

- `PyTorch is required for predicate_resmade`: install `.[model]` or run the
  prototype-table scripts.
- Missing CSV directory: update `dataset.csv_directory` or download/export the
  dataset on the cluster.
- Missing or smoke-domain manifest: run
  `python3 -m model.scripts.prepare_neurocard_data --config <cfg> --rebuild-domains`
  in an environment with JOB-light CSVs and NeuroCard available.
- Non-positive fanout value: inspect sampler output; only known outer-padding
  neutral branches may canonicalize to fanout `1`.
- Direct I/O validation error: set `model.direct_io_connections: false` when
  `factorization.enabled: true`.

## Reproducibility

The shipped configs set `training.seed: 0`. The ResMADE trainer seeds Python,
NumPy, and PyTorch at startup; the synthetic trainer is deterministic because
it uses closed-form weighted counts.

## Upstream Attribution And Licenses

See `ATTRIBUTION.md`. NeuroCard is published under Apache-2.0 according to its
GitHub repository page. Duet and DistJoin are referenced as upstream research
implementations; this milestone adapts DistJoin's bit-factorization and ANPM
ideas without vendoring or copying full source files.
