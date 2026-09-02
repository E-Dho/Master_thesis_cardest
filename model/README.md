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
