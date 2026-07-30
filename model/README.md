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
ResMADE logits. Later factors add a learned offset from embeddings of preceding
factor values, adapted from DistJoin's ANPM modulation pattern and confined to
that one original column.

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
factor paths are evaluated, invalid combinations are excluded by normalizing
over valid IDs, and the existing original-domain predicate mask is applied.
Two-sided ranges continue to use the external inclusion-exclusion path.

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

For real datasets, the future integration point is a NeuroCard-style sampler
that emits uniform full-outer-join samples in the same fixed column order.

## NeuroCard Sampler Preparation

The wrapper command is:

```bash
python3 -m model.scripts.prepare_neurocard_data \
  --config model/configs/job_light_resmade_inv_fanout.yaml
```

For `dataset.type: neurocard_full_join`, the command validates CSV presence and
checks for prepared Exact Weight artifacts under `dataset.prepared_directory`.
If join-count tables, index files, or the preparation manifest are absent, it
prints the missing paths. The actual JOB-light download/preparation and first
real sampler run should be executed on the shared compute cluster, then synced
back or run in place there.

The adapter boundary is `NeuroCardFullJoinSampleSource`. It expects the prepared
manifest to contain canonical `ModelMetadata` and join cardinality. It does not
estimate `|J|` from sampled row count.

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
indicator frequencies, fanout domains/min/max, padded-row percentages, and
decoded sample rows.

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

Factorization is disabled by default and enabled explicitly:

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
  decode_chunk_size: 4096
```

`IdentityOutputAdapter` handles unfactorized outputs.
`ANPMFactorizedOutputAdapter` decodes factorized torch outputs back to
original-column semantics through chunked valid-ID enumeration.

## Testing

Run all tests from the repository root:

```bash
python3 -m unittest discover
python3 -m pytest
```

The tests cover logit slicing, predicate masks, reciprocal fanout masks,
expected inverse fanout, cumulative weights, wildcard exclusion, weighted cross
entropy, checkpoint metadata, lossless factorization round-trips, invalid factor
tuples, output-width reduction for a large synthetic domain, exact two-fanout
arithmetic, synthetic oracle cases, and a deterministic training smoke.

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

The evaluation utilities report:

- Estimated cardinality.
- True cardinality.
- Q-error.
- Median, p90, p95, p99, and max Q-error.
- Inference latency.

The training utilities expose fanout-head weight statistics, including
effective sample size:

```text
(sum_b w_b)^2 / sum_b w_b^2
```

## Known Limitations

- The NeuroCard adapter is a manifest/sample-source boundary; this repository
  does not vendor NeuroCard's full Rust/index preparation stack.
- Real JOB-light preparation and first smoke training must run on the shared
  cluster with dataset files and NeuroCard artifacts available.
- Local validation in this workspace skipped torch-specific tests because
  PyTorch is not installed here.
- Direct input-output connections are disabled in factorized mode.
- Indicators and fanouts are not factorized.
- Factorized inference uses chunk enumeration, not optimized prefix dynamic
  programming.
- Two-sided ranges still use external inclusion-exclusion.
- Factorized checkpoints require fresh training.
- Full benchmark execution still depends on prepared cluster data.

## Troubleshooting

- `PyTorch is required for predicate_resmade`: install `.[model]` or run the
  prototype-table scripts.
- Missing CSV directory: update `dataset.csv_directory` or download/export the
  dataset on the cluster.
- Missing join-count/index artifacts: run or sync NeuroCard Exact Weight
  preparation artifacts into `dataset.prepared_directory`.
- Non-positive fanout value: inspect sampler output; only known outer-padding
  neutral branches may canonicalize to fanout `1`.
- Direct I/O validation error: set `model.direct_io_connections: false` when
  `factorization.enabled: true`.

## Reproducibility

The baseline config sets `training.seed: 0`. The current synthetic trainer is
deterministic because it uses closed-form weighted counts. Future neural
training should seed Python, NumPy, and PyTorch explicitly.

## Upstream Attribution And Licenses

See `ATTRIBUTION.md`. NeuroCard is published under Apache-2.0 according to its
GitHub repository page. Duet and DistJoin are referenced as upstream research
implementations; this milestone adapts DistJoin's bit-factorization and ANPM
ideas without vendoring or copying full source files.
