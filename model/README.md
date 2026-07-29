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
  - Inspected only for the future ANPM/factorized-output boundary. ANPM and
    lossless column factorization are not implemented in this milestone.

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
- No lossless column factorization.

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
```

Important settings:

- `model.type`: `prototype_table` for the lightweight scaffold or
  `predicate_resmade` for real training.
- `model.hidden_sizes`: ResMADE hidden width/depth.
- `model.residual_connections`: enables masked residual blocks.
- `model.direct_io_connections`: enables masked direct input-output path.
- `model.input_encoding`: `embed` or `one_hot`.
- `model.column_order`: `data_indicators_fanouts`.
- `factorization.enabled`: must be `false`.
- `fanout.compute_weights_in_log_space`: builds cumulative inverse weights in
  log space.
- `fanout.weight_clipping`: `null` by default. Clipping would bias the target
  distribution and is not applied.
- `inference.progressive_sampling`: must be `false`.
- `inference.use_log_space_product`: controls log-space product accumulation.

Startup validation fails for incompatible milestone settings.

## Factorization Status

Factorization is disabled by default:

```yaml
factorization:
  enabled: false
  strategy: none
```

`OutputDistributionAdapter` defines the extension boundary. The current
`IdentityOutputAdapter` handles unfactorized columns. The
`ANPMFactorizedOutputAdapter` exists only as a documented placeholder and raises
`NotImplementedError`.

## Testing

Run all tests from the repository root:

```bash
python3 -m unittest discover
python3 -m pytest
```

The tests cover logit slicing, predicate masks, reciprocal fanout masks,
expected inverse fanout, cumulative weights, wildcard exclusion, weighted cross
entropy, checkpoint metadata, factorization failure, exact two-fanout arithmetic,
synthetic oracle cases, and a deterministic training smoke.

When PyTorch is installed, additional tests cover ResMADE output width, separate
input/output bins, per-column softmax, residual shape preservation through the
forward/backward path, checkpoint save/load, current/future token leakage, and
CUDA smoke when CUDA is available.

## Checkpoint Contents

ResMADE checkpoints include:

- Model state dictionary.
- Optimizer state.
- Epoch and step.
- ResMADE configuration.
- Resolved project configuration.
- Schema and column metadata.
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
- Only synthetic oracle validation is included.
- No ANPM.
- No lossless column factorization.
- No benchmark workload or JOB-light pipeline in this milestone.

## Troubleshooting

- `PyTorch is required for predicate_resmade`: install `.[model]` or run the
  prototype-table scripts.
- Missing CSV directory: update `dataset.csv_directory` or download/export the
  dataset on the cluster.
- Missing join-count/index artifacts: run or sync NeuroCard Exact Weight
  preparation artifacts into `dataset.prepared_directory`.
- Non-positive fanout value: inspect sampler output; only known outer-padding
  neutral branches may canonicalize to fanout `1`.

## Reproducibility

The baseline config sets `training.seed: 0`. The current synthetic trainer is
deterministic because it uses closed-form weighted counts. Future neural
training should seed Python, NumPy, and PyTorch explicitly.

## Upstream Attribution And Licenses

See `ATTRIBUTION.md`. NeuroCard is published under Apache-2.0 according to its
GitHub repository page. Duet and DistJoin are referenced as upstream research
implementations; this milestone does not vendor or copy their source code.
